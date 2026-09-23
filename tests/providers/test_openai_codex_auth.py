import asyncio
import base64
import json
import os
import stat
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
)
from free_claude_code.providers.openai_codex import auth as openai_auth
from free_claude_code.providers.openai_codex import login as openai_login
from free_claude_code.providers.openai_codex.auth import (
    OpenAIAuthManager,
    OpenAIReconnectRequired,
)
from free_claude_code.providers.openai_codex.login import BrowserAuthorization


def _jwt(payload: dict[str, object]) -> str:
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )
    return f"header.{encoded}.signature"


def _credential_document(*, expires_at: int) -> dict[str, object]:
    id_token = _jwt(
        {
            "email": "person@example.com",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "account_1",
                "chatgpt_account_is_fedramp": False,
            },
        }
    )
    return {
        "version": 1,
        "credentials": {
            "access_token": _jwt({"exp": expires_at}),
            "refresh_token": "refresh_secret",
            "id_token": id_token,
            "account_id": "account_1",
            "email": "person@example.com",
            "expires_at": expires_at,
            "fedramp": False,
        },
    }


@pytest.mark.asyncio
async def test_auth_manager_refreshes_atomically_without_exposing_tokens(
    tmp_path: Path,
) -> None:
    credential_path = tmp_path / "auth" / "openai.json"
    credential_path.parent.mkdir()
    credential_path.write_text(
        json.dumps(_credential_document(expires_at=1)),
        encoding="utf-8",
    )
    refreshed_access = _jwt({"exp": 4_000_000_000})
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"access_token": refreshed_access},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    manager = OpenAIAuthManager(
        credential_path=credential_path,
        lock_path=tmp_path / "auth" / "openai.lock",
        client=client,
    )

    access = await manager.access()

    assert access.access_token == refreshed_access
    assert access.account_id == "account_1"
    assert len(requests) == 1
    assert b"refresh_secret" in requests[0].content
    assert requests[0].headers["content-type"] == "application/json"
    assert json.loads(requests[0].content) == {
        "grant_type": "refresh_token",
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
        "refresh_token": "refresh_secret",
    }
    safe_status = json.dumps(manager.status().as_dict())
    assert "refresh_secret" not in safe_status
    assert refreshed_access not in safe_status
    saved = json.loads(credential_path.read_text(encoding="utf-8"))
    assert saved["credentials"]["access_token"] == refreshed_access
    assert "email" not in saved["credentials"]
    if os.name != "nt":
        assert stat.S_IMODE(credential_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(credential_path.stat().st_mode) == 0o600
    await manager.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_refresh_is_deduplicated_across_auth_managers(
    tmp_path: Path,
) -> None:
    credential_path = tmp_path / "openai.json"
    credential_path.write_text(
        json.dumps(_credential_document(expires_at=1)),
        encoding="utf-8",
    )
    refreshed_access = _jwt({"exp": 4_000_000_000})
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={"access_token": refreshed_access},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    managers = [
        OpenAIAuthManager(
            credential_path=credential_path,
            lock_path=tmp_path / "openai.lock",
            client=client,
        )
        for _ in range(2)
    ]

    accesses = await asyncio.gather(*(manager.access() for manager in managers))

    assert request_count == 1
    assert {access.access_token for access in accesses} == {refreshed_access}
    await asyncio.gather(*(manager.close() for manager in managers))
    await client.aclose()


@pytest.mark.asyncio
async def test_permanent_refresh_failure_requires_reconnect_and_deletes_file(
    tmp_path: Path,
) -> None:
    credential_path = tmp_path / "openai.json"
    credential_path.write_text(
        json.dumps(_credential_document(expires_at=1)),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": "invalid_grant"},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    manager = OpenAIAuthManager(
        credential_path=credential_path,
        lock_path=tmp_path / "openai.lock",
        client=client,
    )

    with pytest.raises(OpenAIReconnectRequired, match="expired"):
        await manager.access()

    assert not credential_path.exists()
    assert manager.status().connected is False
    assert manager.status().state == "error"
    await manager.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_proactive_transient_refresh_failure_preserves_unauthorized_recovery(
    tmp_path: Path,
) -> None:
    credential_path = tmp_path / "openai.json"
    document = _credential_document(expires_at=1)
    credential_path.write_text(json.dumps(document), encoding="utf-8")
    current_access = _jwt({"exp": 1})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    manager = OpenAIAuthManager(
        credential_path=credential_path,
        lock_path=tmp_path / "openai.lock",
        client=client,
    )

    access = await manager.access()

    assert access.access_token == current_access
    assert manager.is_connected() is True
    assert credential_path.exists()
    await manager.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_unauthorized_recovery_retries_one_transient_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    credential_path = tmp_path / "openai.json"
    credential_path.write_text(
        json.dumps(_credential_document(expires_at=4_000_000_000)),
        encoding="utf-8",
    )
    current_access = _jwt({"exp": 4_000_000_000})
    refreshed_access = _jwt({"exp": 4_100_000_000})
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(
            200,
            json={"access_token": refreshed_access},
            request=request,
        )

    monkeypatch.setattr(openai_auth, "_UNAUTHORIZED_REFRESH_RETRY_DELAY_SECONDS", 0)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    manager = OpenAIAuthManager(
        credential_path=credential_path,
        lock_path=tmp_path / "openai.lock",
        client=client,
    )

    access = await manager.recover_unauthorized(current_access)

    assert access.access_token == refreshed_access
    assert len(requests) == 2
    assert all(request.url.path == "/oauth/token" for request in requests)
    await manager.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_disconnect_revokes_refresh_token_and_always_deletes_credentials(
    tmp_path: Path,
) -> None:
    credential_path = tmp_path / "openai.json"
    credential_path.write_text(
        json.dumps(_credential_document(expires_at=4_000_000_000)),
        encoding="utf-8",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    manager = OpenAIAuthManager(
        credential_path=credential_path,
        lock_path=tmp_path / "openai.lock",
        client=client,
    )

    status = await manager.disconnect()

    assert status.connected is False
    assert not credential_path.exists()
    assert len(requests) == 1
    assert requests[0].headers["content-type"] == "application/json"
    assert json.loads(requests[0].content) == {
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
        "token": "refresh_secret",
        "token_type_hint": "refresh_token",
    }
    await manager.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_browser_authorization_validates_state_and_returns_pkce_grant(
    monkeypatch,
) -> None:
    callback_hosts: list[str] = []
    tcp_site = openai_login.web.TCPSite

    def recording_site(runner, host: str, port: int):
        callback_hosts.append(host)
        return tcp_site(runner, host, port)

    monkeypatch.setattr(openai_login.web, "TCPSite", recording_site)
    browser = await BrowserAuthorization.start()
    query = parse_qs(urlparse(browser.auth_url).query)
    redirect_uri = query["redirect_uri"][0]
    state = query["state"][0]

    async with httpx.AsyncClient(trust_env=False) as client:
        rejected = await client.get(
            redirect_uri,
            params={"code": "wrong_state_code", "state": "wrong_state"},
        )
        response = await client.get(
            redirect_uri,
            params={"code": "authorization_code", "state": state},
        )
    grant = await browser.wait()

    assert rejected.status_code == 400
    assert rejected.text == "State mismatch"
    assert response.status_code == 200
    assert grant.code == "authorization_code"
    assert grant.redirect_uri == redirect_uri
    assert grant.code_verifier
    assert urlparse(redirect_uri).hostname == "localhost"
    assert callback_hosts and set(callback_hosts) == {"localhost"}
    assert query["code_challenge_method"] == ["S256"]
    assert query["originator"] == ["codex_cli_rs"]
    assert "window.close()" in response.text
    await browser.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "document",
    [
        [],
        {"version": 2, "credentials": {}},
    ],
)
async def test_auth_manager_rejects_invalid_credential_schema_without_crashing(
    tmp_path: Path,
    document: object,
) -> None:
    credential_path = tmp_path / "openai.json"
    credential_path.write_text(json.dumps(document), encoding="utf-8")
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: pytest.fail(f"unexpected request: {request.url}")
        )
    )

    manager = OpenAIAuthManager(
        credential_path=credential_path,
        lock_path=tmp_path / "openai.lock",
        client=client,
    )

    assert manager.is_connected() is False
    assert manager.status().state == "error"
    assert manager.status().message == (
        "Saved OpenAI credentials are invalid. Disconnect and sign in again."
    )
    await manager.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_device_login_persists_account_and_reports_only_safe_state(
    tmp_path: Path,
) -> None:
    id_token = _jwt(
        {
            "email": "device@example.com",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "account_device",
            },
        }
    )
    access_token = "opaque_access_token"
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/deviceauth/usercode"):
            return httpx.Response(
                200,
                json={
                    "device_auth_id": "device_private",
                    "user_code": "ABCD-EFGH",
                    "interval": "0",
                },
                request=request,
            )
        if request.url.path.endswith("/deviceauth/token"):
            return httpx.Response(
                200,
                json={
                    "authorization_code": "authorization_private",
                    "code_challenge": "challenge",
                    "code_verifier": "verifier_private",
                },
                request=request,
            )
        if request.url.path.endswith("/oauth/token"):
            return httpx.Response(
                200,
                json={
                    "access_token": access_token,
                    "refresh_token": "refresh_private",
                    "id_token": id_token,
                },
                request=request,
            )
        raise AssertionError(request.url)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    manager = OpenAIAuthManager(
        credential_path=tmp_path / "openai.json",
        lock_path=tmp_path / "openai.lock",
        client=client,
    )

    pending = await manager.start_login(ConnectedAccountLoginMode.DEVICE)
    assert pending.state == "connecting"
    assert pending.user_code == "ABCD-EFGH"
    for _ in range(50):
        status = manager.status()
        if status.state != "connecting":
            break
        await asyncio.sleep(0.01)

    assert status.state == "connected"
    assert status.email == "device@example.com"
    serialized = json.dumps(status.as_dict())
    assert "device_private" not in serialized
    assert "refresh_private" not in serialized
    assert calls == [
        "/api/accounts/deviceauth/usercode",
        "/api/accounts/deviceauth/token",
        "/oauth/token",
    ]
    await manager.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_close_waits_for_pending_login_resource_cleanup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class PendingBrowser:
        auth_url = "https://auth.openai.com/pending"

        def __init__(self) -> None:
            self.closed = False

        async def wait(self):
            await asyncio.Event().wait()

        async def close(self) -> None:
            self.closed = True

    browser = PendingBrowser()
    monkeypatch.setattr(
        BrowserAuthorization,
        "start",
        AsyncMock(return_value=browser),
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: pytest.fail(f"unexpected request: {request.url}")
        )
    )
    manager = OpenAIAuthManager(
        credential_path=tmp_path / "openai.json",
        lock_path=tmp_path / "openai.lock",
        client=client,
    )

    await manager.start_login(ConnectedAccountLoginMode.BROWSER)
    task = manager._login_task
    assert task is not None

    await manager.close()

    assert task.done()
    assert browser.closed is True
    with pytest.raises(OpenAIReconnectRequired, match="shutting down"):
        await manager.access()
    await client.aclose()

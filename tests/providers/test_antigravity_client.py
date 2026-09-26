"""Cloud Code client behavior for the Antigravity provider."""

import hashlib
import json
from pathlib import Path
from typing import cast

import httpx
import pytest

from free_claude_code.providers.antigravity.auth import AntigravityAuthManager
from free_claude_code.providers.antigravity.chat_adapter import AntigravityChatAdapter
from free_claude_code.providers.antigravity.client import (
    AntigravityClient,
    AntigravityUpstreamError,
)
from free_claude_code.providers.antigravity.credentials import (
    AntigravityCredentialError,
    AntigravityCredentials,
)


def future_credentials() -> AntigravityCredentials:
    return AntigravityCredentials(
        access_token="access",
        refresh_token="refresh",
        expires_at=4_000_000_000.0,
        source="test",
    )


async def connected_manager(tmp_path: Path) -> AntigravityAuthManager:
    manager = AntigravityAuthManager(
        state_path=tmp_path / "state.json",
        credential_loader=future_credentials,
    )
    await manager.start_login(manager.status().default_login_mode)
    return manager


@pytest.mark.asyncio
async def test_fetch_available_models_uses_native_bearer_token(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"models": {"gemini-test": {"displayName": "Gemini Test"}}},
        )

    manager = await connected_manager(tmp_path)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AntigravityClient(
        auth=manager,
        base_url="https://daily-cloudcode-pa.googleapis.com",
        client=http,
    )

    result = await client.fetch_available_models()

    assert "gemini-test" in result["models"]
    assert seen[0].url.path == "/v1internal:fetchAvailableModels"
    assert seen[0].headers["authorization"] == "Bearer access"
    assert json.loads(seen[0].content) == {}
    await http.aclose()


@pytest.mark.asyncio
async def test_fetch_available_models_reuses_hour_cache(tmp_path: Path) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"models": {"gemini-test": {"displayName": "Gemini Test"}}},
        )

    manager = await connected_manager(tmp_path)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AntigravityClient(auth=manager, base_url="https://example.test", client=http)

    first = await client.fetch_available_models()
    second = await client.fetch_available_models()

    assert first is second
    assert calls == 1
    await http.aclose()


@pytest.mark.asyncio
async def test_model_cache_is_invalidated_after_account_revision(tmp_path: Path) -> None:
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"models": {f"model-{calls}": {}}})

    manager = await connected_manager(tmp_path)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AntigravityClient(auth=manager, base_url="https://example.test", client=http)

    first = await client.fetch_available_models()
    await manager.disconnect()
    await manager.start_login(manager.status().default_login_mode)
    second = await client.fetch_available_models()

    assert first != second
    assert calls == 2
    await http.aclose()


@pytest.mark.asyncio
async def test_load_code_assist_identifies_antigravity_ide(tmp_path: Path) -> None:
    payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"project": "example"})

    manager = await connected_manager(tmp_path)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AntigravityClient(auth=manager, base_url="https://example.test", client=http)

    assert (await client.load_code_assist())["project"] == "example"
    assert payloads == [{"metadata": {"ideType": "ANTIGRAVITY"}}]
    await http.aclose()


@pytest.mark.asyncio
async def test_project_cache_is_invalidated_after_account_revision() -> None:
    class RevisionClient:
        def __init__(self) -> None:
            self.auth_revision = 1
            self.calls = 0

        async def load_code_assist(self) -> dict[str, object]:
            self.calls += 1
            return {"cloudaicompanionProject": f"project-{self.calls}"}

    fake = RevisionClient()
    adapter = AntigravityChatAdapter(
        cast(AntigravityClient, fake),
        base_url="https://example.test",
    )

    assert await adapter.project_id() == "project-1"
    assert await adapter.project_id() == "project-1"
    fake.auth_revision = 2
    assert await adapter.project_id() == "project-2"
    assert fake.calls == 2


@pytest.mark.asyncio
async def test_generation_envelope_gets_agent_metadata_and_session(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"response": {}})

    manager = await connected_manager(tmp_path)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AntigravityClient(auth=manager, base_url="https://example.test", client=http)

    await client.generate_content(
        {
            "model": "gemini-test",
            "project": "p",
            "request": {
                "contents": [{"role": "user", "parts": [{"text": "hello"}]}]
            },
        }
    )

    assert requests[0]["userAgent"] == "antigravity"
    assert requests[0]["requestType"] == "agent"
    assert str(requests[0]["requestId"]).startswith("agent/")
    request = requests[0]["request"]
    assert isinstance(request, dict)
    assert request["sessionId"] == hashlib.sha256(b"hello").hexdigest()[:32]
    await http.aclose()


@pytest.mark.asyncio
async def test_owned_http_client_uses_fcc_timeout_values() -> None:
    client = AntigravityClient(
        auth=cast(AntigravityAuthManager, object()),
        base_url="https://example.test",
        read_timeout=71.0,
        write_timeout=72.0,
        connect_timeout=73.0,
    )
    try:
        assert client._client.timeout.read == 71.0
        assert client._client.timeout.write == 72.0
        assert client._client.timeout.connect == 73.0
        assert client._client.timeout.pool == 73.0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_expiring_native_token_requires_agy_refresh(tmp_path: Path) -> None:
    expired = False

    def loader() -> AntigravityCredentials:
        if not expired:
            return future_credentials()
        return AntigravityCredentials(
            access_token="old",
            refresh_token="refresh",
            expires_at=1.0,
            source="test",
        )

    manager = AntigravityAuthManager(
        state_path=tmp_path / "state.json",
        credential_loader=loader,
    )
    status = await manager.start_login(manager.status().default_login_mode)
    assert status.connected

    expired = True
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    client = AntigravityClient(auth=manager, base_url="https://example.test", client=http)

    with pytest.raises(AntigravityCredentialError, match="Refresh the account"):
        await client.fetch_available_models()
    await http.aclose()


@pytest.mark.asyncio
async def test_rejected_native_session_invalidates_connected_state(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "unauthorized"}})

    manager = AntigravityAuthManager(
        state_path=path,
        credential_loader=future_credentials,
    )
    await manager.start_login(manager.status().default_login_mode)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AntigravityClient(auth=manager, base_url="https://example.test", client=http)

    with pytest.raises(AntigravityCredentialError, match="rejected"):
        await client.fetch_available_models()

    assert calls == 2
    assert not manager.is_connected()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "enabled": False,
        "revision": 2,
    }
    await http.aclose()


@pytest.mark.asyncio
async def test_stream_error_preserves_body_without_logging_raw_text(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sensitive-upstream-value"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"content-type": "application/json"},
            content=(f'{{"error":{{"message":"quota exhausted {secret}"}}}}').encode(),
        )

    manager = await connected_manager(tmp_path)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AntigravityClient(auth=manager, base_url="https://example.test", client=http)

    stream = client.stream_generate_content(
        {"model": "gemini-test", "project": "p", "request": {"contents": []}}
    )
    with pytest.raises(AntigravityUpstreamError) as captured:
        await anext(stream)

    assert captured.value.status_code == 429
    assert secret in captured.value.body
    assert secret not in str(captured.value)
    assert secret not in caplog.text
    assert "body_chars=" in caplog.text
    await http.aclose()


@pytest.mark.asyncio
async def test_project_override_skips_load_code_assist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoLoadClient:
        async def load_code_assist(self) -> dict[str, object]:
            raise AssertionError("loadCodeAssist should not be called with project override")

    monkeypatch.setenv("CLOUDCODE_GCP_PROJECT_ID", "managed-project")
    adapter = AntigravityChatAdapter(
        cast(AntigravityClient, NoLoadClient()),
        base_url="https://example.test",
    )

    assert await adapter.project_id() == "managed-project"

"""Cloud Code client behavior for the Antigravity provider."""

import json
from pathlib import Path

import httpx
import pytest

from free_claude_code.providers.antigravity.auth import AntigravityAuthManager
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
async def test_generation_envelope_gets_agent_metadata(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"response": {}})

    manager = await connected_manager(tmp_path)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AntigravityClient(auth=manager, base_url="https://example.test", client=http)

    await client.generate_content(
        {"model": "gemini-test", "project": "p", "request": {"contents": []}}
    )

    assert requests[0]["userAgent"] == "antigravity"
    assert requests[0]["requestType"] == "agent"
    assert str(requests[0]["requestId"]).startswith("agent-")
    await http.aclose()


@pytest.mark.asyncio
async def test_expiring_native_token_requires_agy_refresh(tmp_path: Path) -> None:
    def expiring() -> AntigravityCredentials:
        return AntigravityCredentials(
            access_token="old",
            refresh_token="refresh",
            expires_at=1.0,
            source="test",
        )

    manager = AntigravityAuthManager(
        state_path=tmp_path / "state.json",
        credential_loader=expiring,
    )
    await manager.start_login(manager.status().default_login_mode)
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    client = AntigravityClient(auth=manager, base_url="https://example.test", client=http)

    with pytest.raises(AntigravityCredentialError, match="Refresh the account"):
        await client.fetch_available_models()
    await http.aclose()


@pytest.mark.asyncio
async def test_stream_error_reads_body_before_classification(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"content-type": "application/json"},
            content=b'{"error":{"message":"quota exhausted"}}',
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
    assert "quota exhausted" in captured.value.body
    assert "quota exhausted" in str(captured.value)
    await http.aclose()

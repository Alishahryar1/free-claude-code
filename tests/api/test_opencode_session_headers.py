"""Session identity across real ingress, routing, provider, and SDK boundaries."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.config.settings import Settings
from free_claude_code.providers.opencode import create_opencode_provider
from tests.api.support import create_test_app, provider_manager_for_app
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_opencode import (
    _catalog_payload,
    _chat_event_stream,
    _responses_event_stream,
)

pytestmark = pytest.mark.asyncio


def successful_response(request):
    body = (
        _responses_event_stream("hello")
        if request.url.path.endswith("/responses")
        else _chat_event_stream("hello")
    )
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, text=body
    )


@asynccontextmanager
async def wire_client(provider_id="opencode_zen", handler=None):
    requests = []
    catalog_requests = []

    async def generation(request):
        requests.append(request)
        if handler is not None:
            return await handler(request)
        if not request.headers.get("x-opencode-session"):
            return httpx2.Response(
                400,
                json={
                    "type": "MissingSessionID",
                    "message": "Missing x-opencode-session",
                },
            )
        return successful_response(request)

    def catalog(request):
        catalog_requests.append(request)
        return httpx.Response(
            200,
            json=_catalog_payload(
                provider_key="opencode-go"
                if provider_id == "opencode_go"
                else "opencode"
            ),
        )

    def sdk(**kwargs):
        kwargs["http_client"] = httpx2.AsyncClient(
            transport=httpx2.MockTransport(generation)
        )
        return AsyncOpenAI(**kwargs)

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI", side_effect=sdk
    ):
        provider = create_opencode_provider(
            provider_id,
            make_provider_config(
                api_key="test_opencode_key",
                base_url="https://opencode.ai/zen/v1",
            ),
            immediate_admission(provider_name=provider_id),
            catalog_client=httpx.AsyncClient(transport=httpx.MockTransport(catalog)),
        )
    app = create_test_app(
        Settings(
            model=f"{provider_id}/responses-selector",
            opencode_api_key="test_opencode_key",
            proxy_auth_enabled=True,
            proxy_auth_token="fcc-private-token",
        ),
        providers={provider_id: provider},
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://fcc.test",
            headers={"Authorization": "Bearer fcc-private-token"},
        ) as client:
            yield client, provider, requests, catalog_requests
    finally:
        await provider_manager_for_app(app).close()


def payload(ingress, provider_id, selector):
    request = {"model": f"{provider_id}/{selector}", "stream": True}
    if ingress == "responses":
        request["input"] = "hello"
    else:
        request.update(messages=[{"role": "user", "content": "hello"}], max_tokens=128)
    return request


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize("ingress", ["responses", "messages"])
async def test_existing_session_id_reaches_opencode_without_forwarding_other_headers(
    provider_id, selector, ingress
):
    async with wire_client(provider_id) as (client, provider, requests, catalogs):
        defaults = {
            name: value
            for name, value in provider._client.default_headers.items()
            if isinstance(value, str)
        }
        for name in (
            "Session-Id",
            "X-OpenCode-Session",
            "X-Session-Id",
            "X-Claude-Code-Session-Id",
            "session_id",
            "X-Grok-Session-Id",
            "X-Meta-Ai-Gateway-Session-Id",
            "X-Tbh-Session-Id",
        ):
            for conversation in ("conversation-a", "conversation-a", "conversation-b"):
                response = await client.post(
                    f"/v1/{ingress}",
                    json=payload(ingress, provider_id, selector),
                    headers=[
                        (name.encode(), conversation.encode()),
                        (b"X-Fcc-Launch-Id", b"launcher-fallback"),
                        (b"Cookie", b"private=cookie"),
                        (b"X-Private", b"caf\xe9"),
                    ],
                )
                assert response.status_code == 200, response.text
                upstream = requests[-1]
                assert upstream.headers["x-opencode-session"] == conversation
                assert upstream.headers["authorization"] == "Bearer test_opencode_key"
                assert "cookie" not in upstream.headers
                assert "x-private" not in upstream.headers
                assert "x-fcc-launch-id" not in upstream.headers
        assert {
            name: value
            for name, value in provider._client.default_headers.items()
            if isinstance(value, str)
        } == defaults
        assert all("x-opencode-session" not in request.headers for request in catalogs)


@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
async def test_session_header_precedence_and_absence_do_not_invent_or_reuse_identity(
    selector,
):
    async with wire_client() as (client, _provider, requests, _catalogs):
        request = payload("responses", "opencode_zen", selector)
        response = await client.post(
            "/v1/responses",
            json=request,
            headers={
                "x-opencode-session": "explicit-session",
                "session-id": "native-session",
                "X-Claude-Code-Session-Id": "claude-session",
                "session_id": "dsh-session",
                "x-grok-session-id": "grok-session",
                "x-meta-ai-gateway-session-id": "muse-session",
                "x-fcc-launch-id": "launcher-fallback",
            },
        )
        assert response.status_code == 200, response.text
        assert requests[-1].headers["x-opencode-session"] == "explicit-session"
        response = await client.post(
            "/v1/responses",
            json={**request, "prompt_cache_key": "not-a-conversation"},
        )
        assert response.status_code == 400
        assert "MissingSessionID" in response.text
        assert "x-opencode-session" not in requests[-1].headers


@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize("ingress", ["responses", "messages"])
async def test_launch_fallback_is_stable_until_a_native_conversation_id_is_available(
    selector, ingress
):
    async with wire_client() as (client, _provider, requests, _catalogs):
        for launch_id, native_id in (
            ("launch-a", ""),
            ("launch-a", ""),
            ("launch-a", "native-conversation"),
            ("launch-b", ""),
        ):
            response = await client.post(
                f"/v1/{ingress}",
                json=payload(ingress, "opencode_zen", selector),
                headers={"x-fcc-launch-id": launch_id, "session-id": native_id},
            )
            assert response.status_code == 200, response.text
            assert requests[-1].headers["x-opencode-session"] == (
                native_id or launch_id
            )
            assert "x-fcc-launch-id" not in requests[-1].headers


@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize(
    "session_header",
    [
        "session-id",
        "X-Claude-Code-Session-Id",
        "session_id",
        "X-Grok-Session-Id",
        "X-Meta-Ai-Gateway-Session-Id",
        "X-Tbh-Session-Id",
        "x-fcc-launch-id",
    ],
)
async def test_concurrent_conversations_keep_their_session_ids_during_retry(
    selector, session_header
):
    first_arrived = asyncio.Event()
    second_arrived = asyncio.Event()
    attempts = []

    async def upstream(request):
        session = request.headers.get("x-opencode-session")
        attempts.append(session)
        if session == "conversation-a" and attempts.count(session) == 1:
            first_arrived.set()
            await second_arrived.wait()
            return httpx2.Response(503, json={"error": {"message": "Try again"}})
        if session == "conversation-b":
            second_arrived.set()
        return successful_response(request)

    async with wire_client(handler=upstream) as (
        client,
        _provider,
        requests,
        _catalogs,
    ):
        request = payload("responses", "opencode_zen", selector)
        async with asyncio.timeout(5):
            first = asyncio.create_task(
                client.post(
                    "/v1/responses",
                    json=request,
                    headers={session_header: "conversation-a"},
                )
            )
            try:
                await first_arrived.wait()
                second = await client.post(
                    "/v1/responses",
                    json=request,
                    headers={session_header: "conversation-b"},
                )
                assert second.status_code == 200, second.text
                result = await first
                assert result.status_code == 200, result.text
            finally:
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
        assert attempts == ["conversation-a", "conversation-b", "conversation-a"]
        assert len(requests) == 3

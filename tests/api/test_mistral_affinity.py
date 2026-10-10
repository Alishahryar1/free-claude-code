"""Mistral affinity reaches upstream requests through both public APIs."""

import json
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.config.settings import Settings
from free_claude_code.providers.mistral import MistralProvider
from tests.api.support import create_test_app, provider_manager_for_app
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_opencode import _chat_event_stream


@asynccontextmanager
async def mistral_wire(handler=None):
    requests = []

    async def generation(request):
        requests.append(request)
        if handler is not None:
            return await handler(request, len(requests))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_chat_event_stream("hello"),
        )

    def sdk(**kwargs):
        kwargs["http_client"] = httpx2.AsyncClient(
            transport=httpx2.MockTransport(generation)
        )
        return AsyncOpenAI(**kwargs)

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI", side_effect=sdk
    ):
        provider = MistralProvider(
            make_provider_config(
                api_key="test-mistral-key", base_url="https://api.mistral.ai/v1"
            ),
            admission=immediate_admission(provider_name="mistral"),
        )
    app = create_test_app(
        Settings(
            model="mistral/mistral-large-latest",
            mistral_api_key="test-mistral-key",
            proxy_auth_enabled=False,
        ),
        providers={"mistral": provider},
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fcc.test"
        ) as client:
            yield client, requests
    finally:
        await provider_manager_for_app(app).close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ingress", ["messages", "responses"])
@pytest.mark.parametrize(
    "header",
    [
        "x-opencode-session",
        "x-claude-code-session-id",
        "session-id",
        "session_id",
        "x-grok-session-id",
        "x-meta-ai-gateway-session-id",
        "x-tbh-session-id",
        "x-fcc-launch-id",
    ],
)
async def test_affinity_for_existing_harness_identity(ingress, header):
    payload = {"model": "mistral/mistral-large-latest", "stream": True}
    if ingress == "messages":
        payload.update(messages=[{"role": "user", "content": "hello"}], max_tokens=128)
    else:
        payload["input"] = "hello"
    async with mistral_wire() as (client, requests):
        response = await client.post(
            f"/v1/{ingress}", json=payload, headers={header: "conversation-a"}
        )
        assert response.status_code == 200
        assert len(requests) == 1
        assert requests[0].headers.get("x-affinity") == "conversation-a"


class InterruptedStream(httpx2.AsyncByteStream):
    async def __aiter__(self):
        event = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": "mistral-large-latest",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "prefix " * 10000},
                    "finish_reason": None,
                }
            ],
        }
        yield ("data: " + json.dumps(event) + "\n\n").encode()
        raise httpx2.ReadError("upstream disconnected after partial text")


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", [None, "conversation-a"])
async def test_nonstream_recovery_keeps_affinity(session_id):
    async def handler(request, number):
        if number == 1:
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=InterruptedStream(),
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_chat_event_stream("recovered"),
        )

    async with mistral_wire(handler) as (client, requests):
        response = await client.post(
            "/v1/messages",
            json={
                "model": "mistral/mistral-large-latest",
                "stream": False,
                "max_tokens": 32768,
                "messages": [{"role": "user", "content": "long answer"}],
            },
            headers={"x-opencode-session": session_id} if session_id else {},
        )
        assert response.status_code == 200
        assert len(requests) == 2
        assert "recovered" in str(response.json()["content"])
        assert [request.headers.get("x-affinity") for request in requests] == [
            session_id,
            session_id,
        ]

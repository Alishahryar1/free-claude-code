from unittest.mock import patch

import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers import stream_recovery
from free_claude_code.providers.openai_chat import transport as chat_transport
from tests.api.model_fallback_support import (
    ControlledFallbackProvider,
    messages_payload,
)
from tests.api.support import create_test_app
from tests.providers.test_chat_tool_completion import _reply
from tests.providers.test_openai_chat_transport import _transport


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("use_fallback", [False, True])
def test_invalid_chat_tool_public_failure_and_fallback(
    monkeypatch, stream, committed, use_fallback
):
    def recovery():
        controller = stream_recovery.RecoveryController()
        controller._holdback = stream_recovery.RecoveryHoldbackBuffer(
            holdback_seconds=0 if committed else 100000, max_bytes=10**9
        )
        return controller

    monkeypatch.setattr(chat_transport, "RecoveryController", recovery)
    calls = []

    def reply(request):
        calls.append(request)
        return _reply(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "bad",
                        "type": "function",
                        "function": {"name": "Task", "arguments": '{"flag":'},
                    }
                ]
            }
        )

    class Primary(ControlledFallbackProvider):
        async def stream_messages(
            self, request, input_tokens=0, *, request_headers=None, **kwargs
        ):
            async with AsyncOpenAI(
                api_key="test",
                base_url="https://provider.invalid/v1",
                max_retries=0,
                http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(reply)),
            ) as client:
                async for event in _transport(client, max_attempts=3).stream_messages(
                    request, input_tokens, **kwargs
                ):
                    yield event

    fallback = ControlledFallbackProvider(text="fallback worked")
    primary = Primary()
    app = create_test_app(
        Settings(
            model="nvidia_nim/primary-model",
            model_fallbacks=("groq/fallback-model",) if use_fallback else (),
        )
    )
    payload = messages_payload(stream=stream)
    payload["tools"] = [{"name": "Task", "input_schema": {"type": "object"}}]

    def resolve(provider_id, **kwargs):
        return primary if provider_id == "nvidia_nim" else fallback

    with (
        patch("free_claude_code.api.routes.resolve_provider", side_effect=resolve),
        TestClient(app) as client,
    ):
        response = client.post("/v1/messages", json=payload)
    assert len(calls) == 1
    if use_fallback and not committed:
        assert response.status_code == 200
        assert "fallback worked" in response.text
        assert fallback.stream_models == ["fallback-model"]
    else:
        assert not fallback.stream_models
        if stream and committed:
            assert response.status_code == 200
            events = parse_sse_text(response.text)
            assert sum(e.event == "error" for e in events) == 1
            assert not any(e.event in ("message_delta", "message_stop") for e in events)
            starts = [
                e.data["index"] for e in events if e.event == "content_block_start"
            ]
            stops = [e.data["index"] for e in events if e.event == "content_block_stop"]
            assert starts == stops
        else:
            assert response.status_code == 502
            assert response.json()["error"]["type"] == "api_error"

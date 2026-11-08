"""Generic fallback into signed Gemini output remains replayable next turn."""

import json
from unittest.mock import patch

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from free_claude_code.config.model_refs import ModelFallback
from free_claude_code.config.settings import Settings
from free_claude_code.providers.gemini import GeminiProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.support import create_test_app
from tests.api.test_delivered_stream_recovery import text_events
from tests.api.test_four_protocol_matrix import response_value
from tests.api.test_gemini_continuation_replay import body, native
from tests.providers.support import immediate_admission, make_provider_config


@pytest.mark.parametrize("wire", ["messages", "chat", "responses", "gemini"])
def test_fallback_into_gemini_and_followup(wire, monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    calls = []

    def chat_reply(request):
        calls.append(("chat", json.loads(request.content)))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(
                "data: " + json.dumps(e) + "\n\n"
                for e in text_events("chat", "The answer ", complete=False)
            ),
        )

    def gemini_reply(request):
        calls.append(("gemini", json.loads(request.content)))
        event = native(
            "is 42." if sum(k == "gemini" for k, b in calls) == 1 else "FOLLOWUP_OK",
            True,
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(event) + "\n\n",
        )

    sdk = AsyncOpenAI(
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(chat_reply)),
        max_retries=0,
    )
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI", return_value=sdk
    ):
        chat = OpenRouterProvider(
            make_provider_config(api_key="test", base_url="https://chat.invalid/v1"),
            admission=immediate_admission(max_attempts=1),
        )
    gemini = GeminiProvider(
        make_provider_config(api_key="test", base_url="https://google.invalid/v1beta"),
        admission=immediate_admission(max_attempts=1),
        transport=httpx.MockTransport(gemini_reply),
    )
    app = create_test_app(
        Settings(
            MODEL="open_router/primary",
            MODEL_FALLBACKS=[ModelFallback(model="gemini/models/backup")],
        ),
        providers={"open_router": chat, "gemini": gemini},
    )
    first = (
        {"role": "user", "parts": [{"text": "hello"}]}
        if wire == "gemini"
        else {"role": "user", "content": "hello"}
    )
    path, request = body(wire, [first])
    with TestClient(app) as client:
        response = client.post(path, json=request)
        value = response_value(response, wire, True)
        assert [k for k, b in calls] == ["chat", "gemini"]
        if wire == "messages":
            history = [
                first,
                {"role": "assistant", "content": value["content"]},
                {"role": "user", "content": "next"},
            ]
        elif wire == "chat":
            history = [
                first,
                value["choices"][0]["message"],
                {"role": "user", "content": "next"},
            ]
        elif wire == "responses":
            history = [first, *value["output"], {"role": "user", "content": "next"}]
        else:
            history = [
                first,
                value["candidates"][0]["content"],
                {"role": "user", "parts": [{"text": "next"}]},
            ]
        _, request = body(wire, history)
        response = client.post(path, json=request)
        response_value(response, wire, True)
        assert "FOLLOWUP_OK" in response.text
        assert [k for k, b in calls] == ["chat", "gemini", "gemini"]

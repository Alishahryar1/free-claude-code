"""Regression: non-streaming Chat preserves decimal JSON through the real API.

The test uses a synthetic in-memory SSE source; no cloud or localhost model
requests are made.
"""

import json
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from free_claude_code.api.response_streams import ManagedStreamingResponse
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app


@pytest.mark.parametrize(
    ("extension", "expected"),
    [
        (
            "0.123456789012345678901234567890",
            Decimal("0.123456789012345678901234567890"),
        ),
        ("null", None),
    ],
)
def test_nonstreaming_chat_json_preserves_precise_sse_numbers(
    monkeypatch, extension, expected
):
    from free_claude_code.api.handlers import chat as chat_module

    async def fake_stream(_upstream, *, request_id, pre_start_error_response):
        async def body():
            yield (
                'data: {"id":"chatcmpl-synthetic","created":1,'
                '"model":"ollama/demo","choices":[{"index":0,'
                '"delta":{"role":"assistant","content":"سلام"},'
                '"finish_reason":null}],"vendor_extension":' + extension + "}\\n\\n"
            ).replace("\\n", "\n")
            yield (
                'data: {"id":"chatcmpl-synthetic","created":1,'
                '"model":"ollama/demo","choices":[{"index":0,'
                '"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            yield "data: [DONE]\n\n"

        return ManagedStreamingResponse(body(), media_type="text/event-stream")

    monkeypatch.setattr(chat_module, "chat_sse_streaming_response", fake_stream)
    app = create_test_app(Settings(MODEL="ollama/demo"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "ollama/demo",
                "messages": [{"role": "user", "content": "synthetic"}],
                "stream": False,
            },
        )

    assert response.status_code == 200, response.text
    payload = json.loads(response.content, parse_float=Decimal)
    assert payload["choices"][0]["message"]["content"] == "سلام"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["vendor_extension"] == expected
    assert payload["object"] == "chat.completion"
    if isinstance(expected, Decimal):
        assert extension.encode() in response.content

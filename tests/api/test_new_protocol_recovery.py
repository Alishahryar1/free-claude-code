"""New protocol paths extend a single delivered answer after truncation."""

import json

import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient

from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.support import provider_manager_for_app
from tests.api.test_delivered_stream_recovery import text_events
from tests.api.test_four_protocol_matrix import (
    ingress_request,
    matrix_app,
    response_value,
)

PAIRS = [(source, "chat") for source in ("chat", "messages", "responses", "gemini")] + [
    ("gemini", wire) for wire in ("messages", "responses", "gemini")
]


def native_text(text, complete):
    candidate = {"index": 0, "content": {"role": "model", "parts": [{"text": text}]}}
    if complete:
        candidate["finishReason"] = "STOP"
    return [
        {"responseId": "upstream", "modelVersion": "backend", "candidates": [candidate]}
    ]


@pytest.mark.parametrize(("source", "wire"), PAIRS)
def test_truncated_text_continues_without_duplicate_output(source, wire, monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        text = "The answer is " if len(bodies) == 1 else "The answer is forty-two."
        events = (
            native_text(text, len(bodies) > 1)
            if source == "gemini"
            else text_events(source, text, complete=len(bodies) > 1)
        )
        if source == "chat" and len(bodies) > 1:
            events[0]["choices"][0]["finish_reason"] = "stop"
            events = events[:1]
        raw = "".join(
            (f"event: {event['type']}\n" if source == "messages" else "")
            + f"data: {json.dumps(event)}\n\n"
            for event in events
        )
        response = (
            httpx.Response if source in {"messages", "gemini"} else httpx2.Response
        )
        return response(200, headers={"content-type": "text/event-stream"}, text=raw)

    app = matrix_app(source, reply)
    with TestClient(app) as client:
        path, body = ingress_request(wire, True)
        response = client.post(path, json=body)
        value = response_value(response, wire, True)
        assert not value.get("error"), response.text
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(bodies) == 2
    assert "The answer is " in json.dumps(bodies[1])
    if wire == "chat":
        actual = value["choices"][0]["message"]["content"]
        assert response.text.count("data: [DONE]") == 1
    elif wire == "messages":
        actual = "".join(block.get("text", "") for block in value["content"])
    elif wire == "responses":
        actual = "".join(
            part.get("text", "")
            for item in value["output"]
            for part in item.get("content", [])
        )
    else:
        actual = "".join(
            part.get("text", "") for part in value["candidates"][0]["content"]["parts"]
        )
    assert actual == "The answer is forty-two."


@pytest.mark.parametrize(
    "item",
    [
        {"type": "function_call", "name": "read", "arguments": "{}"},
        {"type": "function_call_output", "output": "result"},
        {"role": "user", "content": [{"type": "input_text"}]},
    ],
)
def test_malformed_responses_history_is_rejected_before_native_gemini(item):
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(500)

    app = matrix_app("gemini", reply)
    with TestClient(app) as client:
        response = client.post(
            "/v1/responses", json={"model": "public", "input": [item], "stream": True}
        )
    assert response.status_code == 400, response.text
    assert not calls


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/v1/messages",
            {
                "model": "public",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
                "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
            },
        ),
        (
            "/v1/chat/completions",
            {
                "model": "public",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "parameters": {"type": "object"},
                            "strict": True,
                        },
                    }
                ],
            },
        ),
        (
            "/v1/responses",
            {
                "model": "public",
                "stream": True,
                "input": "hello",
                "tools": [
                    {
                        "type": "function",
                        "name": "read",
                        "parameters": {"type": "object"},
                        "strict": True,
                    }
                ],
            },
        ),
    ],
)
def test_unsupported_native_gemini_controls_fail_before_inference(path, body):
    calls = []

    def reply(request):
        calls.append(request)
        return httpx.Response(500)

    with TestClient(matrix_app("gemini", reply)) as client:
        response = client.post(path, json=body)
    assert response.status_code == 400, response.text
    assert not calls


@pytest.mark.parametrize(
    "config",
    [
        {"responseMimeType": "application/json"},
        {"responseFormat": {"text": {"mimeType": "APPLICATION_JSON"}}},
    ],
)
def test_native_gemini_structured_output_never_continues_partial_json(
    config, monkeypatch
):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        events = native_text(
            '{"x":' if len(bodies) == 1 else '{"y":1}', len(bodies) > 1
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        )

    with TestClient(matrix_app("gemini", reply)) as client:
        response = client.post(
            "/v1beta/models/public:streamGenerateContent?alt=sse",
            json={
                "contents": [{"role": "user", "parts": [{"text": "json"}]}],
                "generationConfig": config,
            },
        )
    assert len(bodies) == 1
    assert '"error"' in response.text
    assert '"finishReason": "STOP"' not in response.text

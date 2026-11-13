"""Native protocol defaults survive public requests and protected replay."""

import json

import httpx2
import pytest
from fastapi.testclient import TestClient

from tests.api.test_four_protocol_matrix import (
    ingress_request,
    matrix_app,
    response_value,
)
from tests.providers.test_gemini import _sse


@pytest.mark.parametrize("wire", ["chat", "messages", "responses"])
@pytest.mark.parametrize(
    "native_part",
    [
        {"functionCall": {"name": "read"}, "thoughtSignature": "signature"},
        {
            "functionCall": {"name": "read", "id": None, "args": None},
            "thoughtSignature": "signature",
        },
        {
            "functionCall": {"name": "read", "args": {}},
            "thought": False,
            "thoughtSignature": "signature",
        },
    ],
)
def test_signed_no_argument_call_can_be_echoed(wire, native_part):
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        return _sse([native_part] if len(sent) == 1 else [{"text": "done"}])

    streaming = wire == "responses"
    with TestClient(matrix_app("gemini", reply)) as client:
        path, body = ingress_request(wire, streaming)
        first = response_value(client.post(path, json=body), wire, streaming)
        initial = {"role": "user", "content": "hello"}
        if wire == "chat":
            message = first["choices"][0]["message"]
            call = message["tool_calls"][0]
            history = [
                initial,
                message,
                {"role": "tool", "tool_call_id": call["id"], "content": "done"},
            ]
        elif wire == "messages":
            call = next(
                block for block in first["content"] if block["type"] == "tool_use"
            )
            history = [
                initial,
                {"role": "assistant", "content": first["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": "done",
                        }
                    ],
                },
            ]
        else:
            call = next(
                item for item in first["output"] if item["type"] == "function_call"
            )
            history = [
                initial,
                *first["output"],
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "done",
                },
            ]
        path, body = ingress_request(wire, streaming, history)
        second = client.post(path, json=body)
        assert second.status_code == 200, second.text
        assert len(sent) == 2
        assert sent[1]["contents"][1]["parts"] == [native_part]


def test_native_chat_allows_documented_allowed_tools_choice():
    sent = []

    def reply(request):
        sent.append(json.loads(request.content))
        event = {
            "id": "chatcmpl-test",
            "created": 1,
            "model": "backend",
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"content": "done"}, "finish_reason": "stop"}
            ],
        }
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )

    choice = {
        "type": "allowed_tools",
        "allowed_tools": {
            "mode": "auto",
            "tools": [{"type": "function", "function": {"name": "read"}}],
        },
    }
    with TestClient(matrix_app("chat", reply)) as client:
        path, body = ingress_request("chat", False)
        body["tool_choice"] = choice
        response = client.post(path, json=body)
        assert response.status_code == 200, response.text
        assert sent[0]["tool_choice"] == choice

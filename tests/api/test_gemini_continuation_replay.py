"""Recovered Gemini answers retain exact signed history on every public API."""

import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.support import provider_manager_for_app
from tests.api.test_four_protocol_matrix import (
    ingress_request,
    matrix_app,
    reply_history,
    response_value,
)


def native(text, finish=False):
    parts = [{"text": text}]
    candidate = {"index": 0, "content": {"role": "model", "parts": parts}}
    if finish:
        parts.append({"text": "", "thoughtSignature": "test-native-signature"})
        candidate["finishReason"] = "STOP"
    return {"candidates": [candidate]}


def body(wire, history):
    if wire == "gemini":
        return "/v1beta/models/public:streamGenerateContent?alt=sse", {
            "contents": history
        }
    return "/v1/chat/completions" if wire == "chat" else "/v1/" + wire, {
        "model": "public",
        "stream": True,
        "max_output_tokens" if wire == "responses" else "max_tokens": 200,
        "input" if wire == "responses" else "messages": history,
    }


@pytest.mark.parametrize("wire", ["messages", "chat", "responses", "gemini"])
@pytest.mark.parametrize("interruptions", [1, 2])
@pytest.mark.parametrize("ambiguous", [False, True])
@pytest.mark.parametrize("signature_shape", ["empty_text", "signature_only", "on_text"])
@pytest.mark.parametrize("thinking", [None, "visible", "hidden"])
def test_recovered_signed_answer_replays_exact_history(
    wire, interruptions, ambiguous, signature_shape, thinking, monkeypatch
):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    requests = []
    final_native = None

    def reply(request):
        nonlocal final_native
        value = json.loads(request.content)
        requests.append(deepcopy(value))
        n = len(requests)
        if n <= interruptions:
            event = native(
                ("abc" if n == 1 else "def")
                if ambiguous
                else ("The answer " if n == 1 else "is ")
            )
        elif n == interruptions + 1:
            event = native(
                ("ab" if interruptions == 1 else "de")
                if ambiguous
                else ("is 42." if interruptions == 1 else "42."),
                True,
            )
            content = event["candidates"][0]["content"]
            if signature_shape == "signature_only":
                content["parts"][-1].pop("text")
            elif signature_shape == "on_text":
                signature = content["parts"].pop()["thoughtSignature"]
                content["parts"][0]["thoughtSignature"] = signature
        else:
            assert value["contents"][:-2] == requests[interruptions]["contents"]
            assert value["contents"][-2] == final_native
            event = native("FOLLOWUP_OK", True)
        if thinking and (wire != "gemini" or thinking == "visible"):
            event["candidates"][0]["content"]["parts"].insert(
                0, {"text": f"Reasoning step {n}.", "thought": True}
            )
        if n == interruptions + 1:
            final_native = deepcopy(event["candidates"][0]["content"])
            if thinking == "hidden":
                assert "Reasoning step" not in json.dumps(value["contents"])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(event) + "\n\n",
        )

    initial = (
        {"role": "user", "parts": [{"text": "hello"}]}
        if wire == "gemini"
        else {"role": "user", "content": "hello"}
    )
    path, request = body(wire, [initial])
    if thinking:
        if wire == "gemini":
            request["generationConfig"] = {
                "thinkingConfig": {"includeThoughts": thinking == "visible"}
            }
        elif wire == "messages":
            request["thinking"] = (
                {"type": "enabled", "budget_tokens": 100}
                if thinking == "visible"
                else {"type": "disabled"}
            )
        elif wire == "chat":
            request["reasoning_effort"] = "high" if thinking == "visible" else "none"
        else:
            request["reasoning"] = {
                "effort": "high" if thinking == "visible" else "none"
            }
    app = matrix_app("gemini", reply, max_attempts=interruptions + 1)
    with TestClient(app) as client:
        first = client.post(path, json=request)
        value = response_value(first, wire, True)
        if wire == "messages":
            assistant = {"role": "assistant", "content": value["content"]}
            text = "".join(b.get("text", "") for b in value["content"])
            history = [initial, assistant, {"role": "user", "content": "next"}]
        elif wire == "chat":
            assistant = value["choices"][0]["message"]
            text = assistant["content"]
            history = [initial, assistant, {"role": "user", "content": "next"}]
        elif wire == "responses":
            text = "".join(
                p.get("text", "")
                for item in value["output"]
                for p in item.get("content", [])
            )
            history = [initial, *value["output"], {"role": "user", "content": "next"}]
        else:
            assistant = value["candidates"][0]["content"]
            text = "".join(
                p.get("text", "") for p in assistant["parts"] if not p.get("thought")
            )
            history = [
                initial,
                assistant,
                {"role": "user", "parts": [{"text": "next"}]},
            ]
        assert (
            text == ("abcab" if interruptions == 1 else "abcdefde")
            if ambiguous
            else text == "The answer is 42."
        )
        if thinking:
            assert ("Reasoning step" in json.dumps(value)) == (thinking == "visible")
        _, request = body(wire, history)
        second = client.post(path, json=request)
        response_value(second, wire, True)
        assert "FOLLOWUP_OK" in second.text
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(requests) == interruptions + 2


@pytest.mark.parametrize("wire", ["messages", "chat", "responses", "gemini"])
@pytest.mark.parametrize("native_id", [False, True])
def test_recovered_signed_tool_call_accepts_its_result(wire, native_id, monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    bodies = []
    call = {"name": "read", "args": {"path": "kept"}}
    if native_id:
        call["id"] = "native-call"
    signed = {
        "role": "model",
        "parts": [{"functionCall": call, "thoughtSignature": "signed-call"}],
    }

    def reply(request):
        value = json.loads(request.content)
        bodies.append(value)
        if len(bodies) == 1:
            event = native("I will read it. ")
        elif len(bodies) == 2:
            event = {
                "candidates": [{"index": 0, "content": signed, "finishReason": "STOP"}]
            }
        else:
            assert value["contents"][:-2] == bodies[1]["contents"]
            assert value["contents"][-2] == signed
            result = value["contents"][-1]["parts"][0]["functionResponse"]
            assert result.get("id") == ("native-call" if native_id else None)
            assert "tool-value" in json.dumps(result)
            event = native("TOOL_RESULT_OK", True)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(event) + "\n\n",
        )

    app = matrix_app("gemini", reply)
    with TestClient(app) as client:
        path, request = ingress_request(wire, True)
        value = response_value(client.post(path, json=request), wire, True)
        if wire == "gemini" and not native_id:
            next(
                p["functionCall"]
                for p in value["candidates"][0]["content"]["parts"]
                if "functionCall" in p
            )["id"] = "client-generated"
        path, request = ingress_request(wire, True, reply_history(value, wire))
        response = client.post(path, json=request)
        response_value(response, wire, True)
        assert "TOOL_RESULT_OK" in response.text
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(bodies) == 3

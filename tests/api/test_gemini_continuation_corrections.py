"""Continuation replay follows history rows that survive request correction."""

import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.support import provider_manager_for_app
from tests.api.test_four_protocol_matrix import matrix_app, response_value
from tests.api.test_gemini_continuation_replay import body, native


@pytest.mark.parametrize("wire", ["messages", "chat", "responses", "gemini"])
@pytest.mark.parametrize("thinking_only", [False, True])
@pytest.mark.parametrize("interrupt_again", [False, True])
def test_corrected_continuation_and_followup_preserve_actual_context(
    wire, thinking_only, interrupt_again, monkeypatch
):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    sent = []
    corrected = []
    completed_context = None
    completed_native = None

    def reply(request):
        nonlocal completed_context, completed_native
        value = json.loads(request.content)
        sent.append(deepcopy(value))
        if completed_context is not None:
            assert value["contents"][:-2] == completed_context
            assert value["contents"][-2] == completed_native
            event = native("FOLLOWUP_OK", True)
        elif len(sent) == 1:
            parts = [{"text": "Some thinking.", "thought": True}]
            if not thinking_only:
                parts.append({"text": "Prefix "})
            event = {"candidates": [{"content": {"role": "model", "parts": parts}}]}
        elif any(p.get("thought") for c in value["contents"] for p in c["parts"]):
            return httpx.Response(
                400,
                json={
                    "error": {"code": 400, "message": "Unsupported thinking history"}
                },
            )
        else:
            corrected.append(deepcopy(value))
            if interrupt_again and len(corrected) == 1:
                event = native("Answer ")
            else:
                event = native("done.", True)
                completed_context = deepcopy(value["contents"])
                completed_native = deepcopy(event["candidates"][0]["content"])
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
    app = matrix_app("gemini", reply, max_attempts=7)
    with TestClient(app) as client:
        response = client.post(path, json=request)
        value = response_value(response, wire, True)
        assert "Some thinking." in json.dumps(value)
        assert "done." in json.dumps(value)
        if wire == "gemini":
            history = [
                initial,
                value["candidates"][0]["content"],
                {"role": "user", "parts": [{"text": "next"}]},
            ]
        elif wire == "responses":
            history = [initial, *value["output"], {"role": "user", "content": "next"}]
        else:
            assistant = (
                value["choices"][0]["message"]
                if wire == "chat"
                else {"role": "assistant", "content": value["content"]}
            )
            history = [initial, assistant, {"role": "user", "content": "next"}]
        _, request = body(wire, history)
        followup = client.post(path, json=request)
        response_value(followup, wire, True)
        assert "FOLLOWUP_OK" in followup.text
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(corrected) == 1 + interrupt_again
    assert [row["role"] for row in corrected[0]["contents"]] == (
        ["user", "user"] if thinking_only else ["user", "model", "user"]
    )

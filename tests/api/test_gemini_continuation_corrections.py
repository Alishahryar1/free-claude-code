"""Continuation replay follows history rows that survive request correction."""

import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.api.support import provider_manager_for_app
from tests.api.test_four_protocol_matrix import matrix_app, response_value
from tests.api.test_gemini_continuation_replay import body, native
from tests.wire_payload import array_payload, decode_object, object_payload


@pytest.mark.parametrize("wire", ["messages", "chat", "responses", "gemini"])
@pytest.mark.parametrize("thinking_only", [False, True])
@pytest.mark.parametrize("interrupt_again", [False, True])
def test_corrected_continuation_and_followup_preserve_actual_context(
    wire: str,
    thinking_only: bool,
    interrupt_again: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    sent = []
    corrected = []
    completed_context = None
    completed_native = None

    def reply(request: httpx.Request):
        nonlocal completed_context, completed_native
        value = decode_object(request.content)
        sent.append(deepcopy(value))
        if completed_context is not None:
            assert array_payload(value["contents"])[:-2] == completed_context
            assert array_payload(value["contents"])[-2] == completed_native
            event = native("FOLLOWUP_OK", True)
        elif len(sent) == 1:
            parts = [{"text": "Some thinking.", "thought": True}]
            if not thinking_only:
                parts.append({"text": "Prefix "})
            event = {"candidates": [{"content": {"role": "model", "parts": parts}}]}
        elif any(
            object_payload(p).get("thought")
            for c in array_payload(value["contents"])
            for p in array_payload(object_payload(c)["parts"])
        ):
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
    app = matrix_app("gemini", native_reply=reply, max_attempts=7)
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
    assert [
        object_payload(row)["role"] for row in array_payload(corrected[0]["contents"])
    ] == (["user", "user"] if thinking_only else ["user", "model", "user"])

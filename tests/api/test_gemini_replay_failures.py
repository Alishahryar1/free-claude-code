"""Incomplete or unretainable signed continuations must never report success."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.support import provider_manager_for_app
from tests.api.test_four_protocol_matrix import ingress_request, matrix_app
from tests.api.test_gemini_continuation_replay import native


@pytest.mark.parametrize("wire", ["messages", "chat", "responses", "gemini"])
@pytest.mark.parametrize(
    "failure",
    ["signed_eof", "signed_tool_eof", "after_stop", "retention", "no_progress"],
)
def test_failed_signed_continuation_stops_and_releases_attempt(
    wire, failure, monkeypatch
):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )
    calls = []
    closed = []

    class Wire(httpx.AsyncByteStream):
        def __init__(self, events):
            self.events = events

        async def __aiter__(self):
            for event in self.events:
                yield ("data: " + json.dumps(event) + "\n\n").encode()

        async def aclose(self):
            closed.append(True)

    def reply(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            events = [native("prefix ")]
        else:
            event = native("suffix", True)
            candidate = event["candidates"][0]
            if failure == "no_progress":
                candidate["content"]["parts"][0]["text"] = "prefix "
            if failure in {"signed_eof", "signed_tool_eof"}:
                candidate.pop("finishReason")
                if failure == "signed_tool_eof":
                    candidate["content"]["parts"][0] = {
                        "functionCall": {"name": "read", "args": {"path": "file"}}
                    }
            if failure == "retention":
                # Native state fits, but the complete serialized replay does not.
                monkeypatch.setattr(
                    "free_claude_code.core.history_replay.RECOVERY_RETENTION_BYTES",
                    500,
                )
            events = [event]
            if failure == "after_stop":
                events.append(native("invalid tail"))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Wire(events)
        )

    app = matrix_app("gemini", reply, max_attempts=4)
    with TestClient(app) as client:
        path, request = ingress_request(wire, True)
        response = client.post(path, json=request)
        events = parse_sse_text(response.text)
        assert any(
            event.data.get("error") or event.event in {"error", "response.failed"}
            for event in events
        ), response.text
        assert not any(
            event.event in {"message_stop", "response.completed"} for event in events
        )
        assert not any(
            candidate.get("finishReason") == "STOP"
            for event in events
            for candidate in event.data.get("candidates", [])
        )
        assert "test-native-signature" not in response.text
        assert "fcc:history:" not in response.text
        assert provider_manager_for_app(app)._current.active_leases == 0
    assert len(calls) == 2
    assert len(closed) == 2

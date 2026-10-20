"""Recovery exhaustion preserves failure semantics and public stream ownership."""

from copy import deepcopy
from itertools import pairwise

import pytest

from free_claude_code.api.request_outcomes import _observe_event
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.continuation_stream import ContinuationStream
from free_claude_code.core.request_outcomes import (
    RequestOutcome,
    current_request_outcome,
)
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderExecutionState,
)
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import delivered
from tests.providers.test_history_transports import _harness


@pytest.fixture
def immediate_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


def assert_unavailable_snapshot(events):
    assert events[-1].event == "response.failed"
    assert events[-1].data["type"] == "response.failed"
    assert sum(event.event == "response.failed" for event in events) == 1
    assert not any(event.event == "response.completed" for event in events)
    response = events[-1].data["response"]
    assert response["status"] == "failed"
    assert response["output"] == []
    assert response["usage"] is None
    assert "snapshot" in response["error"]["message"].lower()
    assert response["id"] == events[0].data["response"]["id"]
    numbers = [event.data["sequence_number"] for event in events]
    assert all(left < right for left, right in pairwise(numbers))


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("continuing", [False, True])
async def test_retention_failure_does_not_claim_a_partial_final_snapshot(
    monkeypatch, immediate_events, protocol, wire, continuing
):
    monkeypatch.setattr(
        "free_claude_code.core.delivered_response.RECOVERY_RETENTION_BYTES", 1200
    )

    def reply(bodies):
        text = "First " if continuing and len(bodies) == 1 else "x" * 2000
        return 200, text_events(protocol, text, complete=False)

    async with _harness(protocol, reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "write"}]), wire)
    events = parse_sse_text(raw)
    assert len(bodies) == (2 if continuing else 1)
    assert public_text(events, wire) == ("First " if continuing else "") + "x" * 2000
    if wire == "responses":
        assert_unavailable_snapshot(events)
    else:
        assert events[-1].event == "error"
        assert sum(event.event == "error" for event in events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("code", [None, "context_length_exceeded"])
async def test_failed_frame_keeps_error_category_and_finishes_execution(
    monkeypatch, immediate_events, limited, code
):
    if limited:
        monkeypatch.setattr(
            "free_claude_code.core.delivered_response.RECOVERY_RETENTION_BYTES", 1200
        )
    executions = []
    original = ProviderAdmissionController.start_execution

    def capture(self, **kwargs):
        execution = original(self, **kwargs)
        executions.append(execution)
        return execution

    monkeypatch.setattr(ProviderAdmissionController, "start_execution", capture)
    error = {
        "type": "rate_limit_error" if code is None else "invalid_request_error",
        "code": code,
        "param": "input",
        "message": "Provider rejected the request.",
    }

    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "First ", complete=False)
        events = text_events("responses", "x" * 2000)
        events[-1]["type"] = "response.failed"
        events[-1]["response"].update(status="failed", error=error)
        return 200, events

    async with _harness("responses", reply, max_attempts=2) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "write"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert executions[-1].state is ProviderExecutionState.FAILED
    failure = events[-1].data["response"]
    assert failure["error"]["type"] == error["type"]
    assert failure["error"]["code"] == code
    assert failure["error"]["param"] == "input"
    assert failure["error"]["message"].startswith(error["message"])
    outcome = RequestOutcome()
    token = current_request_outcome.set(outcome)
    try:
        _observe_event(events[-1])
    finally:
        current_request_outcome.reset(token)
    assert outcome.failure_reason == (code or error["type"])
    if limited:
        assert_unavailable_snapshot(events)
    else:
        assert (
            "".join(
                part.get("text", "")
                for item in failure["output"]
                for part in item.get("content", [])
            )
            == "First " + "x" * 2000
        )
        assert failure["error"] == error


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate", ["", "New text."])
async def test_production_cap_during_closures_does_not_count_as_new_output(candidate):
    # The production holdback releases this prefix by size before the disconnect.
    prefix = "A" * 400_000

    def reply(bodies):
        return 200, text_events(
            "responses",
            prefix if len(bodies) == 1 else candidate,
            complete=len(bodies) > 1,
        )

    async with _harness("responses", reply, max_attempts=2) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "write"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert public_text(events, "responses") == prefix + candidate
    if candidate:
        assert_completed(events, "responses")
    else:
        assert_unavailable_snapshot(events)


@pytest.mark.asyncio
async def test_cap_during_closures_does_not_turn_pure_echo_into_success(
    monkeypatch, immediate_events
):
    monkeypatch.setattr(
        "free_claude_code.core.delivered_response.RECOVERY_RETENTION_BYTES", 1200
    )
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events("responses", "A" * 600, complete=len(bodies) > 1),
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "write"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert public_text(events, "responses") == "A" * 600
    assert_unavailable_snapshot(events)


@pytest.mark.asyncio
async def test_cap_failure_formatting_does_not_reenter_a_failed_mapper(
    monkeypatch, immediate_events
):
    monkeypatch.setattr(
        "free_claude_code.core.delivered_response.RECOVERY_RETENTION_BYTES", 1200
    )
    original = ContinuationStream.prepare

    def prepare(self, payload):
        if payload["type"] == "response.output_text.done":
            raise ValueError("public processing failed")
        return original(self, payload)

    monkeypatch.setattr(ContinuationStream, "prepare", prepare)
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events(
                "responses",
                "First " if len(bodies) == 1 else "x" * 2000,
                complete=len(bodies) > 1,
            ),
        ),
        max_attempts=3,
    ) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "write"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert public_text(events, "responses") == "First " + "x" * 2000
    assert_unavailable_snapshot(events)
    assert "public processing failed" in events[-1].data["response"]["error"]["message"]


ANNOTATION = {
    "type": "url_citation",
    "url": "https://example.com/source",
    "title": "Source",
    "start_index": 0,
    "end_index": 5,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("part_event", ["added", "done"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("continuing", [False, True])
async def test_annotated_part_blocks_continuation_before_its_item_snapshot(
    immediate_events, part_event, wire, continuing
):
    annotated_attempt = 2 if continuing else 1

    def reply(bodies):
        if continuing and len(bodies) == 1:
            return 200, text_events("responses", "First ", complete=False)
        if len(bodies) != annotated_attempt:
            return 200, text_events("responses", "Tail.")
        events = deepcopy(text_events("responses", "Cited "))[:6]
        events[2 if part_event == "added" else 5]["part"]["annotations"] = [ANNOTATION]
        return 200, events

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "cite"}]), wire)
    events = parse_sse_text(raw)
    assert len(bodies) == annotated_attempt
    assert events[-1].event == ("error" if wire == "messages" else "response.failed")
    assert "Tail." not in public_text(events, wire)
    if continuing:
        assert not any(
            event.data.get("part", {}).get("annotations") for event in events
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("annotated", [False, True])
async def test_healthy_original_response_preserves_annotations(
    immediate_events, annotated
):
    source = text_events("responses", "Cited ")
    if annotated:
        source[2]["part"]["annotations"] = [ANNOTATION]
        source[5]["part"]["annotations"] = [ANNOTATION]
    async with _harness("responses", lambda _: (200, source)) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "cite"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 1
    assert_completed(events, "responses")
    assert public_text(events, "responses") == "Cited "
    assert events[-1].data["response"]["output"][0]["content"][0]["annotations"] == (
        [ANNOTATION] if annotated else []
    )

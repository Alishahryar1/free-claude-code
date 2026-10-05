"""Recovery observes native reasoning without changing its wire or replay meaning."""

from copy import deepcopy

import pytest

from free_claude_code.core.anthropic import aggregate_anthropic_sse_to_message
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.history_replay import decode_replay
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import delivered
from tests.providers.test_history_transports import _events_for, _harness


@pytest.fixture(autouse=True)
def immediate_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


def reasoning_events(*, field="summary", seeded="", closed=True):
    events = deepcopy(_events_for("responses"))
    part_type = "summary_text" if field == "summary" else "reasoning_text"
    for event in events:
        item = event.get("item")
        items = (
            [item] if item is not None else event.get("response", {}).get("output", [])
        )
        for item in items:
            item.pop("encrypted_content", None)
            item["content"] = None
            item["summary"] = []
            text = seeded if event["type"].endswith(".added") else seeded + "Find 17."
            if text:
                item[field] = [{"type": part_type, "text": text}]
        if (
            event["type"] == "response.reasoning_summary_text.delta"
            and field == "content"
        ):
            event["type"] = "response.reasoning_text.delta"
            event["content_index"] = event.pop("summary_index")
    return events if closed else events[:3]


def followed_by_text(events, text):
    result = deepcopy(events)
    tail = text_events("responses", text, complete=False)[1:]
    for event in tail:
        if "output_index" in event:
            event["output_index"] = 1
    return [
        {**event, "sequence_number": index}
        for index, event in enumerate([*result, *tail])
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_nullable_reasoning_content_preserves_healthy_response(wire):
    events = _events_for("responses")
    for event in events:
        if "item" in event:
            event["item"]["content"] = None
        for item in event.get("response", {}).get("output", []):
            item["content"] = None
    async with _harness("responses", lambda _: (200, events), max_attempts=1) as (
        send,
        bodies,
        _,
    ):
        raw = await delivered(send(wire, [{"role": "user", "content": "solve"}]), wire)
    assert len(bodies) == 1
    assert_completed(parse_sse_text(raw), wire)
    if wire == "responses":
        item = parse_sse_text(raw)[-1].data["response"]["output"][0]
        assert item["content"] is None
        assert (
            decode_replay(item["encrypted_content"]).native
            == events[-1]["response"]["output"][0]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["summary", "content"])
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_readable_reasoning_continues_from_released_content(field, closed, wire):
    first = reasoning_events(field=field, closed=closed)
    if closed:
        first = first[:-1]
    first = followed_by_text(first, "Hello ")
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            first if len(bodies) == 1 else text_events("responses", "world."),
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "solve"}]), wire)
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert_completed(events, wire)
    assert public_text(events, wire) == "Hello world."
    assert "Find 17." in str(bodies[1]["input"])
    assert "Hello " in str(bodies[1]["input"])


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_readable_reasoning_is_allowed_in_the_continuation(wire):
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events("responses", "Hello ", complete=False)
            if len(bodies) == 1
            else reasoning_events(),
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "solve"}]), wire)
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert_completed(events, wire)
    assert public_text(events, wire) == "Hello "


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_completed_readable_carrier_without_deltas_counts_as_output(wire):
    continuation = reasoning_events()
    continuation.pop(2)
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events("responses", "Hello ", complete=False)
            if len(bodies) == 1
            else continuation,
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "solve"}]), wire)
    assert len(bodies) == 2
    assert_completed(parse_sse_text(raw), wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_completed_readable_carrier_without_deltas_is_continuation_context(wire):
    first = reasoning_events()[:-1]
    first.pop(2)
    first = followed_by_text(first, "Hello ")
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            first if len(bodies) == 1 else text_events("responses", "world."),
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "solve"}]), wire)
    assert len(bodies) == 2
    assert_completed(parse_sse_text(raw), wire)
    assert "Find 17." in str(bodies[1]["input"])


@pytest.mark.asyncio
async def test_nullable_reasoning_in_terminal_only_continuation_is_output():
    continuation = reasoning_events()
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events("responses", "Hello ", complete=False)
            if len(bodies) == 1
            else [continuation[0], continuation[-1]],
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "solve"}]), "responses"
        )
    assert len(bodies) == 2
    events = parse_sse_text(raw)
    assert_completed(events, "responses")
    assert events[-1].data["response"]["output"][-1]["summary"][0]["text"] == "Find 17."


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["summary", "content"])
@pytest.mark.parametrize("seeded", ["", "First, "])
async def test_synthetic_reasoning_closure_preserves_followup_history(field, seeded):
    first = followed_by_text(
        reasoning_events(field=field, seeded=seeded, closed=False), "Hello "
    )

    def reply(bodies):
        if len(bodies) == 1:
            return 200, first
        return 200, text_events("responses", "world." if len(bodies) == 2 else "Next.")

    async with _harness("responses", reply, max_attempts=2) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "solve"}]), "responses"
        )
        events = parse_sse_text(raw)
        assert_completed(events, "responses")
        output = events[-1].data["response"]["output"]
        history = [*deepcopy(output), {"role": "user", "content": "Next question."}]
        await delivered(send("responses", history), "responses")
    assert len(bodies) == 3
    native = bodies[2]["input"][0]
    assert native["type"] == "reasoning"
    assert native["id"] == first[1]["item"]["id"]
    assert native["status"] == "completed"
    assert native[field][0]["text"] == seeded + "Find 17."
    assert native["extension"] == first[1]["item"]["extension"]
    assert "previous provider stream" not in str(bodies[2])
    assert history[:-1] == output


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_readable_reasoning_survives_repeated_continuation_and_followup(wire):
    second = followed_by_text(reasoning_events()[:-1], "middle ")

    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "First ", complete=False)
        if len(bodies) == 2:
            return 200, second
        return 200, text_events("responses", "last." if len(bodies) == 3 else "Next.")

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "solve"}]), wire)
        events = parse_sse_text(raw)
        assert_completed(events, wire)
        assert public_text(events, wire) == "First middle last."
        assert "Find 17." in str(bodies[2]["input"])
        if wire == "responses":
            output = events[-1].data["response"]["output"]
        else:

            async def chunks():
                yield raw

            message, error, stopped = await aggregate_anthropic_sse_to_message(chunks())
            assert error is None and stopped
            output = [{"role": "assistant", "content": message["content"]}]
        await delivered(
            send(wire, [*output, {"role": "user", "content": "Next question."}]), wire
        )
    assert len(bodies) == 4
    native = next(
        item for item in bodies[3]["input"] if item.get("type") == "reasoning"
    )
    assert native == second[3]["item"]
    assert "previous provider stream" not in str(bodies[3])


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["annotations", "encrypted_reasoning"])
@pytest.mark.parametrize(
    "failure", ["failed_context", "incomplete_context", "failed_server"]
)
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_native_failure_snapshot_preserves_error_and_blocks_recovery(
    native, failure, wire
):
    error = {
        "code": "server_error"
        if failure == "failed_server"
        else "context_length_exceeded",
        "type": "api_error" if failure == "failed_server" else "invalid_request_error",
        "param": "input",
        "message": "Model failed to continue.",
    }

    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        events = text_events("responses", "Unpublished failure snapshot")
        final = deepcopy(events[-1])
        response = final["response"]
        if failure == "incomplete_context":
            final["type"] = "response.incomplete"
            response.update(
                status="incomplete",
                incomplete_details={"reason": "model_context_window_exceeded"},
            )
        else:
            final["type"] = "response.failed"
            response.update(status="failed", error=error)
        if native == "annotations":
            response["output"][0]["content"][0]["annotations"] = [
                {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "title": "Example",
                    "start_index": 0,
                    "end_index": 5,
                }
            ]
        else:
            response["output"].append(
                {
                    "type": "reasoning",
                    "id": "rs_final",
                    "summary": [],
                    "content": None,
                    "encrypted_content": "opaque",
                    "status": "completed",
                }
            )
        return 200, [events[0], final]

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(send(wire, [{"role": "user", "content": "solve"}]), wire)
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert events[-1].event == ("error" if wire == "messages" else "response.failed")
    assert public_text(events, wire) == "Hello "
    assert "Unpublished failure snapshot" not in raw
    result = (
        events[-1].data["error"]
        if wire == "messages"
        else events[-1].data["response"]["error"]
    )
    assert result["type"] == error["type"]
    if failure == "failed_server":
        assert error["message"] in result["message"]
    if wire == "responses":
        assert result["code"] == error["code"]
        if failure != "incomplete_context":
            assert result == error

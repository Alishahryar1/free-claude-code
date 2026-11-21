"""Recovery observes native reasoning without changing its wire or replay meaning."""

from copy import deepcopy

import pytest

from free_claude_code.application.response_collection import WireProtocol
from free_claude_code.core.anthropic import aggregate_anthropic_sse_to_message
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.json_types import JsonObject
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import delivered
from tests.providers.test_history_transports import _events_for, _harness
from tests.wire_payload import (
    array_payload,
    object_array,
    object_payload,
    string_payload,
)


def reasoning_events(*, field: str = "summary", seeded: str = "", closed: bool = True):
    events = deepcopy(_events_for("responses"))
    part_type = "summary_text" if field == "summary" else "reasoning_text"
    for event in events:
        item = event.get("item")
        items = (
            [object_payload(item)]
            if item is not None
            else object_array(
                object_payload(event.get("response", {})).get("output", [])
            )
        )
        for item in items:
            item.pop("encrypted_content", None)
            item["content"] = None
            item.update(summary=[])
            text = (
                seeded
                if string_payload(event["type"]).endswith(".added")
                else seeded + "Find 17."
            )
            if text:
                item[field] = [{"type": part_type, "text": text}]
        if (
            event["type"] == "response.reasoning_summary_text.delta"
            and field == "content"
        ):
            event["type"] = "response.reasoning_text.delta"
            event["content_index"] = event.pop("summary_index")
    return events if closed else events[:3]


def followed_by_text(events: list[JsonObject], text: str):
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
async def test_nullable_reasoning_content_preserves_healthy_response(
    wire: WireProtocol,
):
    events = _events_for("responses")
    for event in events:
        if "item" in event:
            object_payload(event["item"])["content"] = None
        for item in object_array(
            object_payload(event.get("response", {})).get("output", [])
        ):
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
        item = array_payload(
            object_payload(parse_sse_text(raw)[-1].data["response"])["output"]
        )[0]
        assert object_payload(item)["content"] is None
        assert item == object_array(object_payload(events[-1]["response"])["output"])[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["summary", "content"])
@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_readable_reasoning_continues_from_released_content(
    field: str, closed: bool, wire: WireProtocol
):
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
    assert "Find 17." not in str(array_payload(bodies[1]["input"])[-1])
    assert any(
        item.get("type") == "reasoning" for item in object_array(bodies[1]["input"])
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_readable_reasoning_is_allowed_in_the_continuation(wire: WireProtocol):
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
async def test_completed_readable_carrier_without_deltas_counts_as_output(
    wire: WireProtocol,
):
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
async def test_completed_readable_carrier_without_deltas_is_continuation_context(
    wire: WireProtocol,
):
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
    assert (
        object_array(
            object_array(object_payload(events[-1].data["response"])["output"])[-1][
                "summary"
            ]
        )[0]["text"]
        == "Find 17."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["summary", "content"])
@pytest.mark.parametrize("seeded", ["", "First, "])
async def test_synthetic_reasoning_closure_preserves_followup_history(
    field: str, seeded: str
):
    first = followed_by_text(
        reasoning_events(field=field, seeded=seeded, closed=False), "Hello "
    )

    def reply(bodies: list[JsonObject]):
        if len(bodies) == 1:
            return 200, first
        return 200, text_events("responses", "world." if len(bodies) == 2 else "Next.")

    async with _harness("responses", reply, max_attempts=2) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "solve"}]), "responses"
        )
        events = parse_sse_text(raw)
        assert_completed(events, "responses")
        output = object_payload(events[-1].data["response"])["output"]
        history = [
            *deepcopy(array_payload(output)),
            {"role": "user", "content": "Next question."},
        ]
        await delivered(send("responses", history), "responses")
    assert len(bodies) == 3
    native = array_payload(output)[0]
    assert object_payload(native)["type"] == "reasoning"
    assert object_payload(native)["id"] == object_payload(first[1]["item"])["id"]
    assert object_payload(native)["status"] == "completed"
    assert object_array(object_payload(native)[field])[0]["text"] == seeded + "Find 17."
    assert (
        object_payload(native)["extension"]
        == object_payload(first[1]["item"])["extension"]
    )
    assert str(bodies[2]).count(seeded + "Find 17.") == 1
    assert "previous provider stream" not in str(bodies[2])
    assert history[:-1] == output


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_readable_reasoning_survives_repeated_continuation_and_followup(
    wire: WireProtocol,
):
    second = followed_by_text(reasoning_events()[:-1], "middle ")

    def reply(bodies: list[JsonObject]):
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
            output = object_payload(events[-1].data["response"])["output"]
        else:

            async def chunks():
                yield raw

            message, error, stopped = await aggregate_anthropic_sse_to_message(chunks())
            assert error is None and stopped
            output = [{"role": "assistant", "content": message["content"]}]
        await delivered(
            send(
                wire,
                [*array_payload(output), {"role": "user", "content": "Next question."}],
            ),
            wire,
        )
    assert len(bodies) == 4
    assert str(bodies[3]).count("Find 17.") == 1
    assert "previous provider stream" not in str(bodies[3])


@pytest.mark.asyncio
@pytest.mark.parametrize("native", ["annotations", "encrypted_reasoning"])
@pytest.mark.parametrize(
    "failure", ["failed_context", "incomplete_context", "failed_server"]
)
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_native_failure_snapshot_preserves_error_and_blocks_recovery(
    native: str, failure: str, wire: WireProtocol
):
    error = {
        "code": "server_error"
        if failure == "failed_server"
        else "context_length_exceeded",
        "type": "api_error" if failure == "failed_server" else "invalid_request_error",
        "param": "input",
        "message": "Model failed to continue.",
    }

    def reply(bodies: list[JsonObject]):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        events = text_events("responses", "Unpublished failure snapshot")
        final = deepcopy(events[-1])
        response = object_payload(final["response"])
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
            object_array(object_array(response["output"])[0]["content"])[0][
                "annotations"
            ] = [
                {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "title": "Example",
                    "start_index": 0,
                    "end_index": 5,
                }
            ]
        else:
            array_payload(response["output"]).append(
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
        else object_payload(events[-1].data["response"])["error"]
    )
    assert object_payload(result)["type"] == error["type"]
    if failure == "failed_server":
        assert string_payload(error["message"]) in string_payload(
            object_payload(result)["message"]
        )
    if wire == "responses":
        assert object_payload(result)["code"] == error["code"]
        if failure != "incomplete_context":
            assert result == error

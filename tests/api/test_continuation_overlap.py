"""Continuation replay is independent of how source text is represented."""

import asyncio
from copy import deepcopy
from itertools import pairwise

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.stream_delivery import current_stream_delivery
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import public_text, text_events
from tests.api.test_hidden_stream_retries import delivered
from tests.api.test_tool_call_buffer import _frames, _response
from tests.providers.test_history_transports import _harness


@pytest.fixture(autouse=True)
def immediate_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


def responses_parts(parts, *, separate_items=False, representation="deltas"):
    """Describe one answer with consistent source snapshots at every level."""
    template = text_events("responses", "")
    items = []
    events = [template[0]]
    groups = [[part] for part in parts] if separate_items else [parts]
    for index, group in enumerate(groups):
        item = {
            "id": f"msg_piece_{index}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": text, "annotations": []}
                for text in group
            ],
        }
        items.append(item)
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {
                    **item,
                    "content": item["content"]
                    if representation == "item_added"
                    else [],
                    "status": "in_progress",
                },
            }
        )
        for position, part in enumerate(item["content"]):
            identity = {
                "output_index": index,
                "item_id": item["id"],
                "content_index": position,
            }
            events.append(
                {
                    "type": "response.content_part.added",
                    **identity,
                    "part": {
                        **part,
                        "text": part["text"]
                        if representation in {"part_added", "item_added"}
                        else part["text"][:1]
                        if representation == "mixed"
                        else "",
                    },
                }
            )
            if representation in {"deltas", "mixed"}:
                events.extend(
                    {
                        "type": "response.output_text.delta",
                        **identity,
                        "delta": character,
                    }
                    for character in (
                        part["text"][1:] if representation == "mixed" else part["text"]
                    )
                )
            if representation not in {"item_done", "part_done", "terminal"}:
                events.append(
                    {
                        "type": "response.output_text.done",
                        **identity,
                        "text": part["text"],
                    }
                )
            if representation not in {"item_done", "terminal"}:
                events.append(
                    {"type": "response.content_part.done", **identity, "part": part}
                )
        events.append(
            {"type": "response.output_item.done", "output_index": index, "item": item}
        )
    terminal = deepcopy(template[-1])
    terminal["response"]["output"] = items
    events.append(terminal)
    if representation == "terminal":
        events = [events[0], terminal]
    return [
        {**deepcopy(event), "sequence_number": number}
        for number, event in enumerate(events)
    ]


def response_text(event):
    return "".join(
        part.get("text", "")
        for item in event.data["response"]["output"]
        for part in item.get("content", [])
    )


def assert_completion_text_agrees(events):
    final = {
        item["id"]: item
        for item in events[-1].data["response"]["output"]
        if item["type"] == "message"
    }
    for event in events:
        data = event.data
        if event.event == "response.output_text.done":
            part = final[data["item_id"]]["content"][data["content_index"]]
            assert data["text"] == part["text"]
        elif event.event == "response.content_part.done":
            part = final[data["item_id"]]["content"][data["content_index"]]
            assert data["part"] == part
        elif event.event == "response.output_item.done":
            assert data["item"]["content"] == final[data["item"]["id"]]["content"]
    numbers = [event.data["sequence_number"] for event in events]
    assert all(before < after for before, after in pairwise(numbers)), numbers


@pytest.mark.asyncio
@pytest.mark.parametrize("separate_items", [False, True])
@pytest.mark.parametrize(
    "representation",
    [
        "terminal",
        "item_done",
        "part_done",
        "text_done",
        "deltas",
        "part_added",
        "item_added",
        "mixed",
    ],
)
async def test_split_replay_has_one_answer_across_representations(
    separate_items, representation
):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        return 200, responses_parts(
            ["He", "llo world."],
            separate_items=separate_items,
            representation=representation,
        )

    async with _harness("responses", reply, max_attempts=2) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert events[-1].event == "response.completed"
    assert response_text(events[-1]) == "Hello world."
    retained = [
        part["text"]
        for item in events[-1].data["response"]["output"][1:]
        for part in item["content"]
    ]
    assert retained == ["", "world."]
    assert_completion_text_agrees(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("representation", ["terminal", "text_done", "mixed"])
async def test_split_pure_replay_is_not_public_progress(representation):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        return 200, responses_parts(["He", "llo "], representation=representation)

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert events[-1].event == "response.failed"
    assert response_text(events[-1]) == "Hello "


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("seeded", [False, True])
async def test_messages_replay_across_blocks_uses_the_same_rule(wire, seeded):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("messages", "abcabc", complete=False)
        template = text_events("messages", "")
        events = [template[0]]
        for index, text in enumerate(["abc", "abcX"]):
            part = text_events("messages", text)[1:-2]
            for event in part:
                event["index"] = index
            if seeded:
                part[0]["content_block"]["text"] = text
                part.pop(1)
            events.extend(part)
        return 200, [*events, *template[-2:]]

    async with _harness("messages", reply, max_attempts=2) as (send, bodies, _):
        raw = await delivered(
            send(wire, [{"role": "user", "content": "continue"}]), wire
        )
    events = parse_sse_text(raw)
    assert len(bodies) == 2
    assert events[-1].event == (
        "message_stop" if wire == "messages" else "response.completed"
    )
    assert public_text(events, wire) == "abcabcX"
    if wire == "responses":
        assert response_text(events[-1]) == "abcabcX"
        assert_completion_text_agrees(events)


@pytest.mark.asyncio
async def test_overflow_preserves_public_prefix_and_closes_the_source(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.core.continuation_overlap.RECOVERY_RETENTION_BYTES", 1000
    )
    closed = []
    attempts = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        try:
            state.begin_attempt()
            attempts.append(1)
            for frame in _frames(
                "responses",
                [
                    (event["type"], event)
                    for event in text_events("responses", "Hello ", complete=False)
                ],
            ):
                yield frame
            state.begin_continuation()
            state.begin_attempt()
            attempts.append(2)
            for event in responses_parts(["He"], representation="text_done")[:4]:
                yield _frames("responses", [(event["type"], event)])[0]
            for _ in range(30):
                event = {"type": "ping", "padding": "x" * 60}
                yield _frames("responses", [(event["type"], event)])[0]
            pytest.fail("the owner must stop reading after its overlap limit")
        finally:
            closed.append(True)

    raw = await delivered(source(), "responses")
    events = parse_sse_text(raw)
    assert attempts == [1, 2]
    assert closed == [True]
    assert events[-1].event == "response.failed"
    assert response_text(events[-1]) == "Hello "
    assert sum(event.event == "response.failed" for event in events) == 1
    assert not any(event.event == "response.completed" for event in events)


@pytest.mark.asyncio
async def test_overlap_release_records_each_frame_only_when_returned():
    states = []
    resumed = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        states.append(state)
        for frame in _frames(
            "responses",
            [
                (event["type"], event)
                for event in text_events("responses", "Hello ", complete=False)
            ],
        ):
            yield frame
        state.begin_continuation()
        for event in responses_parts(["He", "llo world."], representation="text_done"):
            yield _frames("responses", [(event["type"], event)])[0]
            prefix = state.prefix
            assert prefix is not None
            if (
                event["type"] == "response.output_text.done"
                and event["content_index"] == 0
            ):
                assert prefix.text == "Hello "
            if (
                event["type"] == "response.output_text.done"
                and event["content_index"] == 1
            ):
                resumed.append(prefix.text)

    response = await _response("responses", source())
    events = []
    try:
        async for chunk in response.body_iterator:
            frames = parse_sse_text(str(chunk))
            assert len(frames) == 1
            event = frames[0]
            events.append(event)
            if (
                event.event == "response.output_text.done"
                and event.data["text"] == "world."
            ):
                assert states[0].prefix.text == "Hello world."
                assert resumed == []
        assert resumed == ["Hello world."]
        assert_completion_text_agrees(events)
    finally:
        await response.aclose()


@pytest.mark.asyncio
async def test_hidden_attempt_replacement_discards_queued_replay():
    unpublished = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        state.begin_attempt()
        prefix = text_events("responses", "Hello ", complete=False)
        for frame in _frames("responses", [(event["type"], event) for event in prefix]):
            yield frame
        state.begin_continuation()
        for _ in range(2):
            state.begin_attempt()
            event = {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "item_id": "msg_abandoned",
                "delta": "He",
            }
            yield _frames("responses", [(event["type"], event)])[0]
            prefix = state.prefix
            assert prefix is not None
            unpublished.append((state.attempt_content_released, prefix.text))
        state.begin_attempt()
        for event in responses_parts(["world."], representation="text_done"):
            yield _frames("responses", [(event["type"], event)])[0]

    raw = await delivered(source(), "responses")
    assert unpublished == [(False, "Hello "), (False, "Hello ")]
    events = parse_sse_text(raw)
    assert events[-1].event == "response.completed"
    assert response_text(events[-1]) == "Hello world."
    assert "msg_abandoned" not in raw
    assert_completion_text_agrees(events)


@pytest.mark.asyncio
async def test_cancellation_drops_ambiguous_replay_and_closes_the_source():
    waiting = asyncio.Event()
    closed = []
    states = []

    async def source():
        state = current_stream_delivery()
        assert state is not None
        states.append(state)
        try:
            prefix = text_events("responses", "Hello ", complete=False)
            for frame in _frames(
                "responses", [(event["type"], event) for event in prefix]
            ):
                yield frame
            state.begin_continuation()
            for event in responses_parts(["He"], representation="text_done")[:4]:
                yield _frames("responses", [(event["type"], event)])[0]
            waiting.set()
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    response = await _response("responses", source())
    returned = []

    async def consume():
        async for chunk in response.body_iterator:
            returned.extend(parse_sse_text(str(chunk)))

    reader = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        assert states[0].prefix.text == "Hello "
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await response.aclose()
    assert closed == [True]
    assert not any(
        event.event in {"response.failed", "response.completed"} for event in returned
    )
    assert all(
        event.data["text"] == "Hello "
        for event in returned
        if event.event == "response.output_text.done"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("representation", ["text_done", "part_done", "item_done"])
async def test_snapshot_text_is_published_once_before_another_interruption(
    representation,
):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("responses", "Hello ", complete=False)
        if len(bodies) == 2:
            return 200, responses_parts(
                ["Hello world."], representation=representation
            )[:-1]
        return 200, text_events("responses", " Finished.")

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        raw = await delivered(
            send("responses", [{"role": "user", "content": "greet"}]), "responses"
        )
    assert len(bodies) == 3
    assert bodies[2]["input"][-2]["content"] == [
        {"type": "output_text", "text": "Hello world."}
    ]
    events = parse_sse_text(raw)
    assert events[-1].event == "response.completed"
    assert response_text(events[-1]) == "Hello world. Finished."
    assert_completion_text_agrees(events)

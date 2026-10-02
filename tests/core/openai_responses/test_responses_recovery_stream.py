"""Responses recovery retains IDs, complete calls, native data and stop reasons."""

from copy import deepcopy
from weakref import WeakKeyDictionary

import pytest

from free_claude_code.core.history_replay import ReplayOrigin
from free_claude_code.core.openai_responses import (
    ResponsesRecoveryWriter,
)
from free_claude_code.core.openai_responses.source_state import ResponsesSourceState
from free_claude_code.core.stream_events import StreamEvent
from free_claude_code.core.stream_observations import DecodedStreamEvent

ORIGIN = ReplayOrigin("first", "responses", "https://first.test", "key", "model")
SOURCES: WeakKeyDictionary[ResponsesRecoveryWriter, ResponsesSourceState] = (
    WeakKeyDictionary()
)


def _feed(
    writer: ResponsesRecoveryWriter, kind: str, **body: object
) -> list[StreamEvent]:
    event = StreamEvent(kind, {"type": kind, **body})
    if kind == "response.created":
        SOURCES[writer] = ResponsesSourceState()
    source = SOURCES[writer]
    return list(
        writer.feed(
            DecodedStreamEvent(
                ORIGIN,
                event,
                observation=source.observe(event),
                native_reasoning_pending=source.native_reasoning_pending,
            )
        )
    )


def _start(
    writer: ResponsesRecoveryWriter, identity: str = "original"
) -> list[StreamEvent]:
    writer.begin_attempt()
    return _feed(
        writer,
        "response.created",
        response={
            "id": identity,
            "model": "provider",
            "created_at": 1,
            "output": [],
            "status": "in_progress",
        },
    )


def _text(
    writer: ResponsesRecoveryWriter, value: str, index: int = 0
) -> list[StreamEvent]:
    return [
        *_feed(
            writer,
            "response.output_item.added",
            output_index=index,
            item={
                "id": "msg_text",
                "type": "message",
                "role": "assistant",
                "content": [],
                "status": "in_progress",
            },
        ),
        *_feed(
            writer,
            "response.content_part.added",
            output_index=index,
            item_id="msg_text",
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": []},
        ),
        *_feed(
            writer,
            "response.output_text.delta",
            output_index=index,
            item_id="msg_text",
            content_index=0,
            delta=value,
        ),
    ]


def test_unknown_response_state_is_preserved_and_blocks_continuation():
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    writer.begin_attempt()
    frames = _feed(
        writer,
        "response.created",
        response={
            "id": "original",
            "output": [],
            "future_state": {"handle": "opaque"},
        },
    )
    assert frames[0].payload["response"]["future_state"] == {"handle": "opaque"}
    assert writer.checkpoint.blocked_reason is not None


def test_done_only_call_is_retained_in_a_failed_response():
    from free_claude_code.core.failures import ExecutionFailure, FailureKind

    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    call = {
        "id": "fc_done",
        "type": "function_call",
        "name": "read",
        "call_id": "call_done",
        "arguments": "{}",
        "status": "completed",
    }
    frames = _feed(writer, "response.output_item.done", output_index=0, item=call)
    assert sum(frame.kind == "response.output_item.done" for frame in frames) == 1
    assert frames[-1].payload["item"] == call
    assert writer.checkpoint.published_tools
    failed = writer.failure(
        ExecutionFailure(FailureKind.UPSTREAM, 502, "failed", False)
    )
    assert failed[0].payload["response"]["output"] == [call]


@pytest.mark.parametrize("recovered", [False, True])
def test_terminal_snapshot_keeps_text_that_has_no_delta_events(recovered):
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    if recovered:
        _text(writer, "First. ")
        writer.interrupt()
        _start(writer, "replacement")
    before = writer.revision
    item = {
        "id": "msg_snapshot",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Second.", "annotations": []}],
    }
    emitted = _feed(
        writer,
        "response.completed",
        response={"id": "replacement", "status": "completed", "output": [item]},
    )
    assert writer.revision > before
    emitted += writer.finish()
    final = emitted[-1].payload["response"]
    assert final["output"][-1] == item
    if recovered:
        assert writer.checkpoint.text == "First. Second."
    else:
        assert sum(event.kind == "response.completed" for event in emitted) == 1
        assert writer.checkpoint.text == "Second."


def test_repeated_failures_share_identity_and_allocate_distinct_items() -> None:
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    emitted = _start(writer)
    for value in ("First ", "second ", "third"):
        emitted += _text(writer, value)
        emitted += writer.interrupt()
        if value != "third":
            emitted += _start(writer, "replacement")
    _feed(
        writer,
        "response.completed",
        response={
            "id": "replacement",
            "model": "provider",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 4000},
        },
    )
    emitted += writer.finish()

    assert sum(event.kind == "response.created" for event in emitted) == 1
    assert sum(event.kind == "response.completed" for event in emitted) == 1
    assert writer.checkpoint.text == "First second third"
    items = [
        event.payload["item"]
        for event in emitted
        if event.kind == "response.output_item.added"
    ]
    assert len({item["id"] for item in items}) == 3
    assert [
        event.payload["output_index"]
        for event in emitted
        if event.kind == "response.output_item.added"
    ] == [0, 1, 2]
    assert [event.payload["sequence_number"] for event in emitted] == list(
        range(len(emitted))
    )
    response = emitted[-1].payload["response"]
    assert response["id"] == "original"
    assert response["model"] == "public"
    assert response["usage"]["input_tokens"] == 12
    assert len(response["output"]) == 3


def test_tool_identity_and_raw_arguments_are_held_until_item_done() -> None:
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    item = {
        "id": "fc_a",
        "call_id": "call_a",
        "type": "function_call",
        "namespace": "files",
        "name": "read",
        "arguments": "",
    }
    assert not _feed(writer, "response.output_item.added", output_index=0, item=item)
    arguments = '{"value":"' + "x" * 70000 + '"}'
    assert not _feed(
        writer,
        "response.function_call_arguments.delta",
        output_index=0,
        item_id="fc_a",
        delta=arguments,
    )
    assert not writer.checkpoint.published_tools
    complete = _feed(
        writer,
        "response.output_item.done",
        output_index=0,
        item={**item, "arguments": arguments},
    )
    assert len(complete) == 3
    assert complete[-1].payload["item"]["arguments"] == arguments
    assert complete[-1].payload["item"]["namespace"] == "files"
    assert writer.checkpoint.published_tools
    assert writer.finish(salvage=True)[-1].kind == "response.completed"


def test_unfinished_custom_call_and_later_text_are_discarded() -> None:
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    assert not _feed(
        writer,
        "response.output_item.added",
        output_index=0,
        item={
            "id": "ct_a",
            "call_id": "call_a",
            "type": "custom_tool_call",
            "name": "patch",
            "input": "",
        },
    )
    assert not _text(writer, "Dependent tail", index=1)
    assert not writer.interrupt()
    assert not writer.checkpoint.content


def test_native_completed_payload_is_preserved_without_recovery() -> None:
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    item = {"id": "future", "type": "future_native_item", "future": {"opaque": [1, 2]}}
    _feed(writer, "response.output_item.added", output_index=9, item=item)
    _feed(writer, "response.output_item.done", output_index=9, item=item)
    response = {
        "id": "original",
        "model": "public",
        "output": [deepcopy(item)],
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "usage": {"input_tokens": 16, "output_tokens": 32000},
        "future": {"nested": True},
    }
    _feed(writer, "response.incomplete", response=response)
    terminal = writer.finish()[-1]
    assert terminal.kind == "response.incomplete"
    assert terminal.payload["response"] == response


def test_partial_native_reasoning_has_no_safe_continuation() -> None:
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    _feed(
        writer,
        "response.output_item.added",
        output_index=0,
        item={"id": "rs_a", "type": "reasoning", "summary": []},
    )
    _feed(
        writer,
        "response.reasoning_summary_text.delta",
        output_index=0,
        item_id="rs_a",
        delta="Partial",
    )
    assert writer.checkpoint.blocked_reason is not None
    assert not writer.interrupt()


def test_replayed_prefix_is_removed_from_deltas_and_final_output() -> None:
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    _text(writer, "First. ")
    writer.interrupt()
    _start(writer, "replacement")
    emitted = _text(writer, "First. Second.")
    _feed(
        writer,
        "response.output_item.done",
        output_index=0,
        item={
            "id": "msg_text",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "First. Second."}],
        },
    )
    _feed(
        writer,
        "response.completed",
        response={"id": "replacement", "output": [], "status": "completed"},
    )
    terminal = writer.finish()[-1]
    assert [
        event.payload["delta"]
        for event in emitted
        if event.kind == "response.output_text.delta"
    ] == ["Second."]
    assert writer.checkpoint.text == "First. Second."
    assert terminal.payload["response"]["output"][1]["content"][0]["text"] == "Second."


@pytest.mark.parametrize(
    "boundary", ["response.content_part.done", "response.output_item.done"]
)
def test_recovery_keeps_text_first_received_in_a_complete_snapshot(boundary):
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    _text(writer, "First. ")
    writer.interrupt()
    _start(writer, "replacement")
    _text(writer, "")
    part = {"type": "output_text", "text": "First. Second.", "annotations": []}
    if boundary == "response.content_part.done":
        _feed(
            writer,
            boundary,
            output_index=0,
            content_index=0,
            item_id="msg_text",
            part=part,
        )
    else:
        _feed(
            writer,
            boundary,
            output_index=0,
            item={
                "id": "msg_text",
                "type": "message",
                "role": "assistant",
                "content": [part],
            },
        )
    writer.interrupt()
    assert writer.checkpoint.text == "First. Second."


def test_refusal_interruption_preserves_its_content_type():
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    _feed(
        writer,
        "response.output_item.added",
        output_index=0,
        item={
            "id": "msg_refusal",
            "type": "message",
            "role": "assistant",
            "content": [],
        },
    )
    _feed(
        writer,
        "response.refusal.delta",
        output_index=0,
        item_id="msg_refusal",
        content_index=0,
        delta="I cannot do that.",
    )
    closed = writer.interrupt()
    assert closed[0].kind == "response.refusal.done"
    assert closed[0].payload["refusal"] == "I cannot do that."
    parts = writer.checkpoint.content[0]["content"]
    assert isinstance(parts, list)
    assert isinstance(parts[0], dict)
    assert parts[0]["type"] == "refusal"


def test_unknown_extension_on_a_known_item_prevents_recovery():
    writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
    _start(writer)
    events = _feed(
        writer,
        "response.output_item.added",
        output_index=0,
        item={
            "id": "msg_text",
            "type": "message",
            "role": "assistant",
            "content": [],
            "future_reference": {"item_id": "remote"},
        },
    )
    assert events[0].payload["item"]["future_reference"] == {"item_id": "remote"}
    assert writer.checkpoint.blocked_reason is not None


def test_recovered_usage_excludes_opaque_reasoning_and_foreign_cache_counts():
    usages = []
    for encrypted in ("short", "opaque" * 10000):
        writer = ResponsesRecoveryWriter(model="public", input_tokens=12)
        _start(writer)
        item = {
            "id": "rs_first",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Plan."}],
            "encrypted_content": encrypted,
        }
        _feed(writer, "response.output_item.added", output_index=0, item=item)
        _feed(writer, "response.output_item.done", output_index=0, item=item)
        writer.interrupt()
        _start(writer, "replacement")
        _text(writer, "Answer.")
        _feed(
            writer,
            "response.completed",
            response={
                "id": "replacement",
                "output": [],
                "usage": {
                    "input_tokens": 900,
                    "output_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 800},
                },
            },
        )
        usages.append(writer.finish()[-1].payload["response"]["usage"])
    assert usages[0] == usages[1]
    assert usages[0]["input_tokens"] == 12
    assert "input_tokens_details" not in usages[0]

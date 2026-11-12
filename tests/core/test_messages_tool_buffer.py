import pytest

from free_claude_code.core.anthropic.tool_calls import MessagesToolCallBuffer
from free_claude_code.core.failures import ExecutionFailure


def event(kind, **data):
    return kind, {"type": kind, **data}


def test_stopped_tool_waits_for_successful_message_reason():
    buffer = MessagesToolCallBuffer()
    start = event(
        "content_block_start",
        index=0,
        content_block={"type": "tool_use", "id": "a", "name": "read", "input": {}},
    )
    delta = event(
        "content_block_delta",
        index=0,
        delta={"type": "input_json_delta", "partial_json": "opaque"},
    )
    stop = event("content_block_stop", index=0)
    for item in (start, delta, stop):
        assert buffer.feed(*item) == []
    terminal = event("message_delta", delta={"stop_reason": "tool_use"})
    assert buffer.feed(*terminal) == [(start, delta, stop), (terminal,)]


@pytest.mark.parametrize("reason", ["max_tokens", "refusal", "pause_turn"])
def test_truncation_discards_stopped_tool_and_releases_later_text(reason):
    buffer = MessagesToolCallBuffer()
    buffer.feed(
        *event(
            "content_block_start",
            index=0,
            content_block={"type": "tool_use", "id": "a", "name": "read", "input": {}},
        )
    )
    buffer.feed(*event("content_block_stop", index=0))
    text = event(
        "content_block_start", index=1, content_block={"type": "text", "text": "later"}
    )
    assert buffer.feed(*text) == []
    terminal = event("message_delta", delta={"stop_reason": reason})
    assert buffer.feed(*terminal) == [(text,), (terminal,)]


def test_failure_does_not_promote_stopped_tool():
    buffer = MessagesToolCallBuffer()
    buffer.feed(
        *event(
            "content_block_start",
            index=0,
            content_block={"type": "tool_use", "id": "a", "name": "read", "input": {}},
        )
    )
    buffer.feed(*event("content_block_stop", index=0))
    assert buffer.abandon() == []


def test_confirmed_tool_cannot_emit_late_argument_fragments():
    buffer = MessagesToolCallBuffer()
    buffer.feed(
        *event(
            "content_block_start",
            index=0,
            content_block={"type": "tool_use", "id": "a", "name": "read", "input": {}},
        )
    )
    buffer.feed(*event("content_block_stop", index=0))
    buffer.feed(*event("message_delta", delta={"stop_reason": "tool_use"}))
    with pytest.raises(ExecutionFailure):
        buffer.feed(
            *event(
                "content_block_delta",
                index=0,
                delta={"type": "input_json_delta", "partial_json": "late"},
            )
        )

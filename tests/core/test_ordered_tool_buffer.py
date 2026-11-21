"""Pending calls preserve order without forcing other calls to complete."""

import pytest

from free_claude_code.core.tool_call_buffer import (
    OrderedToolCallBuffer,
    RetentionBudget,
    ToolBufferLimitError,
)


def test_parallel_ready_prefix_preserves_source_order():
    buffer = OrderedToolCallBuffer[str]()
    buffer.reserve("a", "A")
    buffer.append("text")
    buffer.reserve("b", "B")
    buffer.complete("b")
    assert list(buffer.drain()) == []
    buffer.complete("a")
    assert list(buffer.drain()) == ["A", "text", "B"]
    assert buffer.budget.bytes == 0
    assert buffer.budget.records == 0


def test_failure_discards_only_unfinished_slots():
    buffer = OrderedToolCallBuffer[str]()
    buffer.reserve("a", "unfinished")
    buffer.append("text")
    buffer.reserve("b", "B")
    buffer.complete("b")
    buffer.discard_pending()
    assert list(buffer.drain()) == ["text", "B"]


def test_pending_payload_and_record_limits_never_release_fragments():
    budget = RetentionBudget(max_bytes=4, max_records=2)
    buffer = OrderedToolCallBuffer[str](budget)
    buffer.reserve("a", "A", size=2)
    buffer.grow("a", 2)
    with pytest.raises(ToolBufferLimitError):
        buffer.grow("a", 1)
    assert list(buffer.drain()) == []
    buffer.clear()
    assert (budget.bytes, budget.records) == (0, 0)
    buffer.reserve("a", "A")
    buffer.grow("a", 0)
    with pytest.raises(ToolBufferLimitError):
        buffer.grow("a", 0)


def test_drain_does_not_retain_completed_payloads():
    buffer = OrderedToolCallBuffer[str]()
    for index in range(20):
        buffer.reserve(index, "call", size=10)
        buffer.complete(index)
        assert list(buffer.drain()) == ["call"]
    assert not buffer.pending
    assert (buffer.budget.bytes, buffer.budget.records) == (0, 0)

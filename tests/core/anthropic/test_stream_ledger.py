"""Tests for the provider-neutral Anthropic stream ledger."""

import json
from unittest.mock import patch

import pytest

from free_claude_code.core.anthropic.streaming import (
    AnthropicStreamLedger,
    StreamBlockLedger,
    ToolSchema,
    map_stop_reason,
)


def _payload(event: str) -> dict:
    return json.loads(
        next(line[6:] for line in event.splitlines() if line.startswith("data:"))
    )


@pytest.mark.parametrize(
    ("upstream", "anthropic"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("content_filter", "end_turn"),
        (None, "end_turn"),
    ],
)
def test_map_stop_reason(upstream: str | None, anthropic: str) -> None:
    assert map_stop_reason(upstream) == anthropic


def test_stream_block_ledger_allocates_monotonic_indexes() -> None:
    blocks = StreamBlockLedger()

    assert (blocks.allocate_index(), blocks.allocate_index()) == (0, 1)


def test_message_lifecycle() -> None:
    ledger = AnthropicStreamLedger("msg_1", "model", input_tokens=7)

    start = _payload(ledger.message_start())
    delta = _payload(ledger.message_delta("end_turn", 3))
    stop = _payload(ledger.message_stop())

    assert start["message"]["id"] == "msg_1"
    assert start["message"]["usage"]["input_tokens"] == 7
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"]["output_tokens"] == 3
    assert stop == {"type": "message_stop"}
    assert ledger.has_terminal_message()


def test_text_and_thinking_blocks_accumulate_content() -> None:
    ledger = AnthropicStreamLedger("msg_1", "model")

    ledger.start_thinking_block()
    ledger.emit_thinking_delta("step")
    ledger.stop_thinking_block()
    ledger.start_text_block()
    ledger.emit_text_delta("answer")
    ledger.stop_text_block()

    assert ledger.accumulated_reasoning == "step"
    assert ledger.accumulated_text == "answer"


def test_ensure_block_switches_close_the_previous_kind() -> None:
    ledger = AnthropicStreamLedger("msg_1", "model")
    ledger.start_thinking_block()

    events = list(ledger.ensure_text_block())

    assert [_payload(event)["type"] for event in events] == [
        "content_block_stop",
        "content_block_start",
    ]
    assert not ledger.blocks.thinking_started
    assert ledger.blocks.text_started


def test_tool_blocks_drive_stop_reason_and_salvage() -> None:
    ledger = AnthropicStreamLedger("msg_1", "model")
    ledger.start_tool_block(0, "toolu_1", "Read")
    ledger.emit_tool_delta(0, '{"path":"test.py"}')

    assert ledger.final_stop_reason("end_turn") == "tool_use"
    assert ledger.can_salvage_tool_use(
        {
            "Read": ToolSchema(
                name="Read",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            )
        }
    )


def test_close_unclosed_blocks_closes_each_block_once() -> None:
    ledger = AnthropicStreamLedger("msg_1", "model")
    ledger.start_text_block()
    ledger.start_tool_block(0, "toolu_1", "Read")

    events = list(ledger.close_unclosed_blocks())

    assert len(events) == 2
    assert all(_payload(event)["type"] == "content_block_stop" for event in events)
    assert list(ledger.close_unclosed_blocks()) == []


def test_output_token_estimate_combines_shared_estimates_and_block_overhead() -> None:
    ledger = AnthropicStreamLedger("msg_1", "model")
    ledger.start_thinking_block()
    ledger.emit_thinking_delta("why")
    ledger.stop_thinking_block()
    ledger.start_text_block()
    ledger.emit_text_delta("abcd")
    ledger.stop_text_block()
    ledger.start_tool_block(0, "toolu_1", "Read")
    ledger.emit_tool_delta(0, "{}")

    with patch(
        "free_claude_code.core.anthropic.streaming.ledger.estimate_text_tokens",
        side_effect=len,
    ):
        assert ledger.estimate_output_tokens() == 40

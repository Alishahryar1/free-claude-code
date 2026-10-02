"""Source assembly and public Messages lifecycle contracts."""

from unittest.mock import patch

import pytest

from free_claude_code.core.anthropic.recovery_stream import MessagesRecoveryWriter
from free_claude_code.core.anthropic.streaming import map_stop_reason
from free_claude_code.core.chat_observations import ChatStreamUsage
from tests.protocol_stream_support import ChatSourceHarness


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


def test_writer_allocates_monotonic_indexes() -> None:
    writer = MessagesRecoveryWriter(model="model", input_tokens=0)
    assert (writer.allocate_block_index(), writer.allocate_block_index()) == (0, 1)


def test_message_lifecycle() -> None:
    source = ChatSourceHarness(model="model", input_tokens=7)
    events = source.project(
        [
            *source.start_events(),
            *source.finish_success(stop_reason="end_turn", usage=ChatStreamUsage(7, 3)),
        ]
    )
    start, delta, stop = [event.payload for event in events]
    assert start["message"]["id"].startswith("msg_")
    assert start["message"]["usage"]["input_tokens"] == 7
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"]["output_tokens"] == 3
    assert stop == {"type": "message_stop"}


def test_text_and_thinking_blocks_accumulate_content() -> None:
    source = ChatSourceHarness(model="model", input_tokens=0)
    source.ensure_reasoning_block()
    source.emit_reasoning_delta("step")
    source.ensure_text_block()
    source.emit_text_delta("answer")
    source.close_content_blocks()
    assert source.accumulated_reasoning == "step"
    assert source.accumulated_text == "answer"


def test_ensure_block_switches_close_the_previous_kind() -> None:
    source = ChatSourceHarness(model="model", input_tokens=0)
    source.project([*source.start_events(), *source.ensure_reasoning_block()])
    events = source.project(source.ensure_text_block())
    assert [event.kind for event in events] == [
        "content_block_stop",
        "content_block_start",
    ]
    assert events[-1].payload["content_block"]["type"] == "text"


def test_complete_tool_controls_handoff_and_stop_reason() -> None:
    source = ChatSourceHarness(model="model", input_tokens=0)
    events = source.project(
        [
            *source.start_events(),
            source.start_tool_block(0, "toolu_1", "Read"),
            *source.emit_tool_delta(0, '{"path":"test.py"}'),
        ]
    )
    assert [event.kind for event in events] == ["message_start"]
    assert not source.writer.checkpoint.published_tools
    source.project(source.close_all_blocks())
    assert source.writer.checkpoint.published_tools
    assert source.final_stop_reason("end_turn") == "tool_use"


def test_close_unclosed_blocks_closes_each_block_once() -> None:
    source = ChatSourceHarness(model="model", input_tokens=0)
    source.project(
        [
            *source.start_events(),
            *source.ensure_text_block(),
            source.start_tool_block(0, "toolu_1", "Read"),
            *source.emit_tool_delta(0, "{}"),
        ]
    )
    events = source.project(source.close_all_blocks())
    assert sum(event.kind == "content_block_stop" for event in events) == 2
    assert source.project(source.close_all_blocks()) == []


def test_output_token_estimate_combines_shared_estimates_and_block_overhead() -> None:
    source = ChatSourceHarness(model="model", input_tokens=0)
    source.ensure_reasoning_block()
    source.emit_reasoning_delta("why")
    source.ensure_text_block()
    source.emit_text_delta("abcd")
    source.close_content_blocks()
    source.start_tool_block(0, "toolu_1", "Read")
    source.emit_tool_delta(0, "{}")
    with patch(
        "free_claude_code.providers.openai_chat.source_state.estimate_text_tokens",
        side_effect=len,
    ):
        assert source.estimate_output_tokens() == 40

"""Tests for providers/nvidia_nim/utils/sse_builder.py."""

import json
import pytest
from unittest.mock import patch

from providers.nvidia_nim.utils.sse_builder import (
    SSEBuilder,
    ContentBlockManager,
    map_stop_reason,
    STOP_REASON_MAP,
)


def _parse_sse(sse_str: str) -> dict:
    """Parse an SSE event string into its data payload."""
    for line in sse_str.strip().split("\n"):
        if line.startswith("data: "):
            return json.loads(line[len("data: ") :])
    raise ValueError(f"No data line found in SSE: {sse_str}")


class TestMapStopReason:
    """Tests for map_stop_reason function."""

    @pytest.mark.parametrize(
        "openai_reason,expected",
        [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "end_turn"),
            (None, "end_turn"),
            ("unknown_value", "end_turn"),
            ("", "end_turn"),
        ],
        ids=[
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "none",
            "unknown",
            "empty_string",
        ],
    )
    def test_map_stop_reason(self, openai_reason, expected):
        assert map_stop_reason(openai_reason) == expected


class TestContentBlockManager:
    """Tests for ContentBlockManager."""

    def test_allocate_index_increments(self):
        mgr = ContentBlockManager()
        assert mgr.allocate_index() == 0
        assert mgr.allocate_index() == 1
        assert mgr.allocate_index() == 2

    def test_initial_state(self):
        mgr = ContentBlockManager()
        assert mgr.thinking_index == -1
        assert mgr.text_index == -1
        assert mgr.thinking_started is False
        assert mgr.text_started is False
        assert mgr.tool_indices == {}


class TestSSEBuilderMessageLifecycle:
    """Tests for message_start, message_delta, message_stop, done."""

    def test_message_start(self):
        builder = SSEBuilder("msg_123", "test-model", input_tokens=50)
        sse = builder.message_start()

        assert "event: message_start" in sse
        data = _parse_sse(sse)
        assert data["type"] == "message_start"
        msg = data["message"]
        assert msg["id"] == "msg_123"
        assert msg["model"] == "test-model"
        assert msg["role"] == "assistant"
        assert msg["content"] == []
        assert msg["usage"]["input_tokens"] == 50
        assert msg["usage"]["output_tokens"] == 1

    def test_message_delta(self):
        builder = SSEBuilder("msg_1", "model")
        sse = builder.message_delta("end_turn", 42)

        assert "event: message_delta" in sse
        data = _parse_sse(sse)
        assert data["type"] == "message_delta"
        assert data["delta"]["stop_reason"] == "end_turn"
        assert data["usage"]["output_tokens"] == 42

    def test_message_stop(self):
        builder = SSEBuilder("msg_1", "model")
        sse = builder.message_stop()

        assert "event: message_stop" in sse
        data = _parse_sse(sse)
        assert data["type"] == "message_stop"

    def test_done(self):
        builder = SSEBuilder("msg_1", "model")
        assert builder.done() == "[DONE]\n\n"


class TestSSEBuilderContentBlocks:
    """Tests for content block start/delta/stop events."""

    def test_content_block_start_text(self):
        builder = SSEBuilder("msg_1", "model")
        sse = builder.content_block_start(0, "text", text="hello")

        data = _parse_sse(sse)
        assert data["type"] == "content_block_start"
        assert data["index"] == 0
        assert data["content_block"]["type"] == "text"
        assert data["content_block"]["text"] == "hello"

    def test_content_block_start_thinking(self):
        builder = SSEBuilder("msg_1", "model")
        sse = builder.content_block_start(1, "thinking")

        data = _parse_sse(sse)
        assert data["content_block"]["type"] == "thinking"
        assert data["content_block"]["thinking"] == ""

    def test_content_block_start_tool_use(self):
        builder = SSEBuilder("msg_1", "model")
        sse = builder.content_block_start(
            2, "tool_use", id="tool_123", name="Read", input={}
        )

        data = _parse_sse(sse)
        assert data["content_block"]["type"] == "tool_use"
        assert data["content_block"]["id"] == "tool_123"
        assert data["content_block"]["name"] == "Read"
        assert data["content_block"]["input"] == {}

    def test_content_block_delta_text(self):
        builder = SSEBuilder("msg_1", "model")
        sse = builder.content_block_delta(0, "text_delta", "hello world")

        data = _parse_sse(sse)
        assert data["type"] == "content_block_delta"
        assert data["index"] == 0
        assert data["delta"]["type"] == "text_delta"
        assert data["delta"]["text"] == "hello world"

    def test_content_block_delta_thinking(self):
        builder = SSEBuilder("msg_1", "model")
        sse = builder.content_block_delta(1, "thinking_delta", "reasoning...")

        data = _parse_sse(sse)
        assert data["delta"]["type"] == "thinking_delta"
        assert data["delta"]["thinking"] == "reasoning..."

    def test_content_block_delta_input_json(self):
        builder = SSEBuilder("msg_1", "model")
        sse = builder.content_block_delta(2, "input_json_delta", '{"key": "val"}')

        data = _parse_sse(sse)
        assert data["delta"]["type"] == "input_json_delta"
        assert data["delta"]["partial_json"] == '{"key": "val"}'

    def test_content_block_stop(self):
        builder = SSEBuilder("msg_1", "model")
        sse = builder.content_block_stop(0)

        data = _parse_sse(sse)
        assert data["type"] == "content_block_stop"
        assert data["index"] == 0


class TestSSEBuilderHighLevelHelpers:
    """Tests for high-level thinking/text/tool block helpers."""

    def test_start_and_stop_thinking_block(self):
        builder = SSEBuilder("msg_1", "model")

        start_sse = builder.start_thinking_block()
        data = _parse_sse(start_sse)
        assert data["content_block"]["type"] == "thinking"
        assert builder.blocks.thinking_started is True
        assert builder.blocks.thinking_index == 0

        stop_sse = builder.stop_thinking_block()
        data = _parse_sse(stop_sse)
        assert data["type"] == "content_block_stop"
        assert builder.blocks.thinking_started is False

    def test_emit_thinking_delta_accumulates(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_thinking_block()

        builder.emit_thinking_delta("part1 ")
        builder.emit_thinking_delta("part2")

        assert builder.accumulated_reasoning == "part1 part2"

    def test_start_and_stop_text_block(self):
        builder = SSEBuilder("msg_1", "model")

        start_sse = builder.start_text_block()
        data = _parse_sse(start_sse)
        assert data["content_block"]["type"] == "text"
        assert builder.blocks.text_started is True
        assert builder.blocks.text_index == 0

        stop_sse = builder.stop_text_block()
        assert builder.blocks.text_started is False

    def test_emit_text_delta_accumulates(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_text_block()

        builder.emit_text_delta("hello ")
        builder.emit_text_delta("world")

        assert builder.accumulated_text == "hello world"

    def test_start_tool_block(self):
        builder = SSEBuilder("msg_1", "model")
        sse = builder.start_tool_block(0, "tool_abc", "Grep")

        data = _parse_sse(sse)
        assert data["content_block"]["type"] == "tool_use"
        assert data["content_block"]["id"] == "tool_abc"
        assert data["content_block"]["name"] == "Grep"
        assert 0 in builder.blocks.tool_indices

    def test_emit_tool_delta(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_tool_block(0, "tool_abc", "Grep")

        sse = builder.emit_tool_delta(0, '{"pattern":')
        data = _parse_sse(sse)
        assert data["delta"]["partial_json"] == '{"pattern":'
        assert builder.blocks.tool_contents[0] == '{"pattern":'

    def test_stop_tool_block(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_tool_block(0, "tool_abc", "Grep")

        sse = builder.stop_tool_block(0)
        data = _parse_sse(sse)
        assert data["type"] == "content_block_stop"


class TestSSEBuilderStateManagement:
    """Tests for ensure_thinking_block, ensure_text_block, close_all_blocks."""

    def test_ensure_thinking_block_closes_text_first(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_text_block()
        assert builder.blocks.text_started is True

        events = list(builder.ensure_thinking_block())
        # Should close text then start thinking
        assert len(events) == 2
        assert builder.blocks.text_started is False
        assert builder.blocks.thinking_started is True

    def test_ensure_thinking_block_noop_if_already_started(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_thinking_block()

        events = list(builder.ensure_thinking_block())
        assert events == []

    def test_ensure_text_block_closes_thinking_first(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_thinking_block()
        assert builder.blocks.thinking_started is True

        events = list(builder.ensure_text_block())
        # Should close thinking then start text
        assert len(events) == 2
        assert builder.blocks.thinking_started is False
        assert builder.blocks.text_started is True

    def test_ensure_text_block_noop_if_already_started(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_text_block()

        events = list(builder.ensure_text_block())
        assert events == []

    def test_close_content_blocks(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_thinking_block()
        builder.stop_thinking_block()
        builder.start_text_block()

        events = list(builder.close_content_blocks())
        # Should close text (thinking already stopped)
        assert len(events) == 1
        assert builder.blocks.text_started is False

    def test_close_all_blocks(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_thinking_block()
        builder.stop_thinking_block()
        builder.start_text_block()
        builder.start_tool_block(0, "t1", "Read")
        builder.start_tool_block(1, "t2", "Write")

        events = list(builder.close_all_blocks())
        # Close text + 2 tool blocks (thinking already stopped)
        assert len(events) == 3
        assert builder.blocks.text_started is False

    def test_close_all_blocks_empty(self):
        builder = SSEBuilder("msg_1", "model")
        events = list(builder.close_all_blocks())
        assert events == []


class TestSSEBuilderError:
    """Tests for emit_error."""

    def test_emit_error(self):
        builder = SSEBuilder("msg_1", "model")
        events = list(builder.emit_error("Something went wrong"))

        assert len(events) == 3  # start, delta, stop
        start_data = _parse_sse(events[0])
        assert start_data["content_block"]["type"] == "text"

        delta_data = _parse_sse(events[1])
        assert delta_data["delta"]["text"] == "Something went wrong"

        stop_data = _parse_sse(events[2])
        assert stop_data["type"] == "content_block_stop"


class TestSSEBuilderTokenEstimation:
    """Tests for estimate_output_tokens."""

    def test_estimate_with_text_only(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_text_block()
        builder.emit_text_delta("hello world")

        tokens = builder.estimate_output_tokens()
        assert tokens > 0

    def test_estimate_with_reasoning(self):
        builder = SSEBuilder("msg_1", "model")
        builder.start_thinking_block()
        builder.emit_thinking_delta("deep thought")
        builder.stop_thinking_block()
        builder.start_text_block()
        builder.emit_text_delta("answer")

        tokens = builder.estimate_output_tokens()
        assert tokens > 0

    def test_estimate_empty(self):
        builder = SSEBuilder("msg_1", "model")
        tokens = builder.estimate_output_tokens()
        assert tokens == 0

    def test_estimate_without_tiktoken(self):
        """Fallback estimation when tiktoken is not available."""
        builder = SSEBuilder("msg_1", "model")
        builder.start_text_block()
        builder.emit_text_delta("a" * 100)  # 100 chars -> ~25 tokens

        with patch("providers.nvidia_nim.utils.sse_builder.ENCODER", None):
            tokens = builder.estimate_output_tokens()
            assert tokens == 25  # 100 // 4

    def test_estimate_with_tools_no_tiktoken(self):
        """Fallback tool token estimation."""
        builder = SSEBuilder("msg_1", "model")
        builder.start_tool_block(0, "t1", "Read")
        builder.emit_tool_delta(0, '{"path":"test.py"}')

        with patch("providers.nvidia_nim.utils.sse_builder.ENCODER", None):
            tokens = builder.estimate_output_tokens()
            # 1 tool * 50 = 50
            assert tokens == 50

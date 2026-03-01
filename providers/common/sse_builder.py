"""SSE event builder for Anthropic-format streaming responses."""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

try:
    import tiktoken

    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None


# Map OpenAI finish_reason to Anthropic stop_reason
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def map_stop_reason(openai_reason: str | None) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    return (
        STOP_REASON_MAP.get(openai_reason, "end_turn") if openai_reason else "end_turn"
    )


@dataclass
class ToolCallState:
    """State for a single streaming tool call."""

    block_index: int  # -1 if not yet allocated
    tool_id: str
    name: str
    contents: list[str] = field(default_factory=list)
    started: bool = False
    task_arg_buffer: str = ""
    task_args_emitted: bool = False


@dataclass
class ContentBlockManager:
    """Manages content block indices and state."""

    next_index: int = 0
    thinking_index: int = -1
    text_index: int = -1
    thinking_started: bool = False
    text_started: bool = False
    tool_states: dict[int, ToolCallState] = field(default_factory=dict)

    def allocate_index(self) -> int:
        """Allocate and return the next block index."""
        idx = self.next_index
        self.next_index += 1
        return idx

    def register_tool_name(self, index: int, name: str) -> None:
        """Register or merge a streaming tool name fragment.

        Handles providers that stream names as fragments and those that
        resend the full name on every chunk.
        """
        if index not in self.tool_states:
            self.tool_states[index] = ToolCallState(
                block_index=-1, tool_id="", name=name
            )
            return
        state = self.tool_states[index]
        prev = state.name
        if not prev or name.startswith(prev):
            state.name = name
        elif not prev.startswith(name):
            state.name = prev + name

    def buffer_task_args(self, index: int, args: str) -> dict | None:
        """Buffer Task tool args and return parsed JSON when complete.

        Returns the parsed (and patched) args dict once the buffer forms
        valid JSON, or None if still accumulating.
        """
        state = self.tool_states.get(index)
        if state is None or state.task_args_emitted:
            return None

        state.task_arg_buffer += args
        try:
            args_json = json.loads(state.task_arg_buffer)
        except Exception:
            return None

        if args_json.get("run_in_background") is not False:
            args_json["run_in_background"] = False

        state.task_args_emitted = True
        state.task_arg_buffer = ""
        return args_json

    def flush_task_arg_buffers(self) -> list[tuple[int, str]]:
        """Flush any remaining Task arg buffers. Returns (tool_index, json_str) pairs."""
        results: list[tuple[int, str]] = []
        for tool_index, state in list(self.tool_states.items()):
            if not state.task_arg_buffer or state.task_args_emitted:
                continue

            out = "{}"
            try:
                args_json = json.loads(state.task_arg_buffer)
                if args_json.get("run_in_background") is not False:
                    args_json["run_in_background"] = False
                out = json.dumps(args_json)
            except Exception as e:
                prefix = state.task_arg_buffer[:120]
                logger.warning(
                    "Task args invalid JSON (id=%s len=%d prefix=%r): %s",
                    state.tool_id or "unknown",
                    len(state.task_arg_buffer),
                    prefix,
                    e,
                )

            state.task_args_emitted = True
            state.task_arg_buffer = ""
            results.append((tool_index, out))
        return results


class SSEBuilder:
    """Builder for Anthropic SSE streaming events."""

    def __init__(self, message_id: str, model: str, input_tokens: int = 0):
        self.message_id = message_id
        self.model = model
        self.input_tokens = input_tokens
        self.blocks = ContentBlockManager()
        self._accumulated_text_parts: list[str] = []
        self._accumulated_reasoning_parts: list[str] = []

    def _format_event(self, event_type: str, data: dict[str, Any]) -> str:
        """Format as SSE string."""
        event_str = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        logger.debug("SSE_EVENT: {} - {}", event_type, event_str.strip())
        return event_str

    # Message lifecycle events
    def message_start(self) -> str:
        """Generate message_start event."""
        usage = {"input_tokens": self.input_tokens, "output_tokens": 1}
        return self._format_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage,
                },
            },
        )

    def message_delta(self, stop_reason: str, output_tokens: int) -> str:
        """Generate message_delta event with stop reason."""
        return self._format_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": output_tokens,
                },
            },
        )

    def message_stop(self) -> str:
        """Generate message_stop event."""
        return self._format_event("message_stop", {"type": "message_stop"})

    # Content block events
    def content_block_start(self, index: int, block_type: str, **kwargs) -> str:
        """Generate content_block_start event."""
        content_block: dict[str, Any] = {"type": block_type}
        if block_type == "thinking":
            content_block["thinking"] = kwargs.get("thinking", "")
        elif block_type == "text":
            content_block["text"] = kwargs.get("text", "")
        elif block_type == "tool_use":
            content_block["id"] = kwargs.get("id", "")
            content_block["name"] = kwargs.get("name", "")
            content_block["input"] = kwargs.get("input", {})

        return self._format_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": content_block,
            },
        )

    def content_block_delta(self, index: int, delta_type: str, content: str) -> str:
        """Generate content_block_delta event."""
        delta: dict[str, Any] = {"type": delta_type}
        if delta_type == "thinking_delta":
            delta["thinking"] = content
        elif delta_type == "text_delta":
            delta["text"] = content
        elif delta_type == "input_json_delta":
            delta["partial_json"] = content

        return self._format_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": delta,
            },
        )

    def content_block_stop(self, index: int) -> str:
        """Generate content_block_stop event."""
        return self._format_event(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": index,
            },
        )

    # High-level helpers for thinking blocks
    def start_thinking_block(self) -> str:
        """Start a thinking block, allocating index."""
        self.blocks.thinking_index = self.blocks.allocate_index()
        self.blocks.thinking_started = True
        return self.content_block_start(self.blocks.thinking_index, "thinking")

    def emit_thinking_delta(self, content: str) -> str:
        """Emit thinking content delta."""
        self._accumulated_reasoning_parts.append(content)
        return self.content_block_delta(
            self.blocks.thinking_index, "thinking_delta", content
        )

    def stop_thinking_block(self) -> str:
        """Stop the current thinking block."""
        self.blocks.thinking_started = False
        return self.content_block_stop(self.blocks.thinking_index)

    # High-level helpers for text blocks
    def start_text_block(self) -> str:
        """Start a text block, allocating index."""
        self.blocks.text_index = self.blocks.allocate_index()
        self.blocks.text_started = True
        return self.content_block_start(self.blocks.text_index, "text")

    def emit_text_delta(self, content: str) -> str:
        """Emit text content delta."""
        self._accumulated_text_parts.append(content)
        return self.content_block_delta(self.blocks.text_index, "text_delta", content)

    def stop_text_block(self) -> str:
        """Stop the current text block."""
        self.blocks.text_started = False
        return self.content_block_stop(self.blocks.text_index)

    # High-level helpers for tool blocks
    def start_tool_block(self, tool_index: int, tool_id: str, name: str) -> str:
        """Start a tool_use block."""
        block_idx = self.blocks.allocate_index()
        if tool_index in self.blocks.tool_states:
            state = self.blocks.tool_states[tool_index]
            state.block_index = block_idx
            state.tool_id = tool_id
            state.started = True
        else:
            self.blocks.tool_states[tool_index] = ToolCallState(
                block_index=block_idx,
                tool_id=tool_id,
                name=name,
                started=True,
            )
        return self.content_block_start(block_idx, "tool_use", id=tool_id, name=name)

    def emit_tool_delta(self, tool_index: int, partial_json: str) -> str:
        """Emit tool input delta."""
        state = self.blocks.tool_states[tool_index]
        state.contents.append(partial_json)
        return self.content_block_delta(
            state.block_index, "input_json_delta", partial_json
        )

    def stop_tool_block(self, tool_index: int) -> str:
        """Stop a tool block."""
        block_idx = self.blocks.tool_states[tool_index].block_index
        return self.content_block_stop(block_idx)

    # State management helpers
    def ensure_thinking_block(self) -> Iterator[str]:
        """Ensure a thinking block is started, closing text block if needed."""
        if self.blocks.text_started:
            yield self.stop_text_block()
        if not self.blocks.thinking_started:
            yield self.start_thinking_block()

    def ensure_text_block(self) -> Iterator[str]:
        """Ensure a text block is started, closing thinking block if needed."""
        if self.blocks.thinking_started:
            yield self.stop_thinking_block()
        if not self.blocks.text_started:
            yield self.start_text_block()

    def close_content_blocks(self) -> Iterator[str]:
        """Close thinking and text blocks (before tool calls)."""
        if self.blocks.thinking_started:
            yield self.stop_thinking_block()
        if self.blocks.text_started:
            yield self.stop_text_block()

    def close_all_blocks(self) -> Iterator[str]:
        """Close all open blocks (thinking, text, tools)."""
        if self.blocks.thinking_started:
            yield self.stop_thinking_block()
        if self.blocks.text_started:
            yield self.stop_text_block()
        for tool_index, state in list(self.blocks.tool_states.items()):
            if state.started:
                yield self.stop_tool_block(tool_index)

    # Error handling
    def emit_error(self, error_message: str) -> Iterator[str]:
        """Emit an error as a text block."""
        error_index = self.blocks.allocate_index()
        yield self.content_block_start(error_index, "text")
        yield self.content_block_delta(error_index, "text_delta", error_message)
        yield self.content_block_stop(error_index)

    # Accumulated content access
    @property
    def accumulated_text(self) -> str:
        """Get accumulated text content."""
        return "".join(self._accumulated_text_parts)

    @property
    def accumulated_reasoning(self) -> str:
        """Get accumulated reasoning content."""
        return "".join(self._accumulated_reasoning_parts)

    def estimate_output_tokens(self) -> int:
        """Estimate output tokens from accumulated content."""
        accumulated_text = self.accumulated_text
        accumulated_reasoning = self.accumulated_reasoning
        if ENCODER:
            text_tokens = len(ENCODER.encode(accumulated_text))
            reasoning_tokens = len(ENCODER.encode(accumulated_reasoning))
            # Tool calls are harder to tokenize exactly without reconstruction, but we can approximate
            # by tokenizing the json dumps of tool contents
            tool_tokens = 0
            started_tool_count = 0
            for state in self.blocks.tool_states.values():
                tool_tokens += len(ENCODER.encode(state.name))
                tool_tokens += len(ENCODER.encode("".join(state.contents)))
                tool_tokens += 15  # Control tokens overhead per tool
                if state.started:
                    started_tool_count += 1

            # Per-block overhead (~4 tokens per content block)
            block_count = (
                (1 if accumulated_reasoning else 0)
                + (1 if accumulated_text else 0)
                + started_tool_count
            )
            block_overhead = block_count * 4

            return text_tokens + reasoning_tokens + tool_tokens + block_overhead

        text_tokens = len(accumulated_text) // 4
        reasoning_tokens = len(accumulated_reasoning) // 4
        tool_tokens = sum(1 for s in self.blocks.tool_states.values() if s.started) * 50
        return text_tokens + reasoning_tokens + tool_tokens

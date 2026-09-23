"""Anthropic stream state ledger."""

import hashlib
import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from free_claude_code.core.token_estimation import estimate_text_tokens

from .emitter import AnthropicSseEmitter
from .recovery import (
    ToolSchema,
    parse_complete_tool_input,
)


def _safe_usage_int(value: object) -> int:
    return value if isinstance(value, int) else 0


@dataclass
class ToolBlockState:
    """State for one streamed tool call."""

    block_index: int
    tool_id: str
    name: str
    extra_content: dict[str, Any] | None = None
    started: bool = False
    task_arg_buffer: str = ""
    task_args_emitted: bool = False
    pre_start_args: str = ""


@dataclass
class StreamBlockState:
    """Tracked downstream Anthropic content block."""

    index: int
    block_type: str
    open: bool = True
    tool_id: str = ""
    name: str = ""
    parts: list[str] = field(default_factory=list)
    extra_content: dict[str, Any] | None = None

    @property
    def content(self) -> str:
        return "".join(self.parts)


@dataclass
class StreamBlockLedger:
    """Allocate and track Anthropic content block indexes."""

    next_index: int = 0
    thinking_index: int = -1
    text_index: int = -1
    thinking_started: bool = False
    text_started: bool = False
    tool_states: dict[int, ToolBlockState] = field(default_factory=dict)

    def allocate_index(self) -> int:
        idx = self.next_index
        self.next_index += 1
        return idx

    def reserve_index(self, index: int) -> None:
        self.next_index = max(self.next_index, index + 1)

    def ensure_tool_state(self, index: int) -> ToolBlockState:
        if index not in self.tool_states:
            self.tool_states[index] = ToolBlockState(
                block_index=-1, tool_id="", name=""
            )
        return self.tool_states[index]

    def set_tool_extra_content(
        self, index: int, extra_content: dict[str, Any] | None
    ) -> None:
        if extra_content:
            self.ensure_tool_state(index).extra_content = extra_content

    def register_tool_name(self, index: int, name: str) -> None:
        if index not in self.tool_states:
            self.tool_states[index] = ToolBlockState(
                block_index=-1, tool_id="", name=name
            )
            return
        state = self.tool_states[index]
        prev = state.name
        if not prev or name.startswith(prev):
            state.name = name
        elif not prev.startswith(name):
            state.name = prev + name

    def buffer_task_args(self, index: int, args: str) -> dict[str, Any] | None:
        state = self.tool_states.get(index)
        if state is None or state.task_args_emitted:
            return None

        state.task_arg_buffer += args
        try:
            args_json = json.loads(state.task_arg_buffer)
        except Exception:
            return None
        if not isinstance(args_json, dict):
            return None

        _normalize_task_run_in_background(args_json)
        state.task_args_emitted = True
        state.task_arg_buffer = ""
        return args_json

    def flush_task_arg_buffers(self) -> list[tuple[int, str]]:
        results: list[tuple[int, str]] = []
        for tool_index, state in list(self.tool_states.items()):
            if not state.task_arg_buffer or state.task_args_emitted:
                continue

            out = "{}"
            try:
                args_json = json.loads(state.task_arg_buffer)
                if isinstance(args_json, dict):
                    _normalize_task_run_in_background(args_json)
                    out = json.dumps(args_json)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                digest = hashlib.sha256(
                    state.task_arg_buffer.encode("utf-8", errors="replace")
                ).hexdigest()[:16]
                logger.warning(
                    "Task args invalid JSON (id={} len={} buffer_sha256_prefix={}): {}",
                    state.tool_id or "unknown",
                    len(state.task_arg_buffer),
                    digest,
                    exc,
                )

            state.task_args_emitted = True
            state.task_arg_buffer = ""
            results.append((tool_index, out))
        return results


class AnthropicStreamLedger:
    """Own mutable Anthropic stream state and produce serialized SSE events."""

    def __init__(
        self,
        message_id: str | None,
        model: str,
        input_tokens: int = 0,
        *,
        log_raw_events: bool = False,
    ) -> None:
        self.message_id = message_id or f"msg_{uuid.uuid4()}"
        self.model = model
        self.input_tokens = input_tokens
        self.blocks = StreamBlockLedger()
        self._emitter = AnthropicSseEmitter(log_raw_events=log_raw_events)
        self._text_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._open_stack: list[int] = []
        self._content_blocks: dict[int, StreamBlockState] = {}
        self.message_started = False
        self.message_stopped = False
        self.stop_reason: str | None = None

    def message_start(self) -> str:
        self.message_started = True
        safe_input = _safe_usage_int(self.input_tokens)
        return self._emitter.event(
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
                    "usage": {"input_tokens": safe_input, "output_tokens": 1},
                },
            },
        )

    def message_delta(
        self,
        stop_reason: str,
        output_tokens: int | None,
        *,
        input_tokens: int | None = None,
        usage_fields: Mapping[str, int] | None = None,
    ) -> str:
        self.stop_reason = stop_reason
        safe_in = _safe_usage_int(
            self.input_tokens if input_tokens is None else input_tokens
        )
        safe_out = output_tokens if isinstance(output_tokens, int) else 0
        usage = {"input_tokens": safe_in, "output_tokens": safe_out}
        if usage_fields:
            usage.update(
                {
                    key: value
                    for key, value in usage_fields.items()
                    if isinstance(value, int)
                }
            )
        return self._emitter.event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": usage,
            },
        )

    def message_stop(self) -> str:
        self.message_stopped = True
        return self._emitter.event("message_stop", {"type": "message_stop"})

    def content_block_start(self, index: int, block_type: str, **kwargs: Any) -> str:
        content_block: dict[str, Any] = {"type": block_type}
        if block_type == "thinking":
            content_block["thinking"] = kwargs.get("thinking", "")
        elif block_type == "text":
            content_block["text"] = kwargs.get("text", "")
        elif block_type == "tool_use":
            content_block["id"] = kwargs.get("id", "")
            content_block["name"] = kwargs.get("name", "")
            content_block["input"] = kwargs.get("input", {})
            extra_content = kwargs.get("extra_content")
            if isinstance(extra_content, dict) and extra_content:
                content_block["extra_content"] = extra_content
        elif block_type == "redacted_thinking":
            data = kwargs.get("data")
            if isinstance(data, str):
                content_block["data"] = data

        self._record_block_start(index, content_block)
        return self._emitter.event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": content_block,
            },
        )

    def content_block_delta(self, index: int, delta_type: str, content: str) -> str:
        delta: dict[str, Any] = {"type": delta_type}
        if delta_type == "thinking_delta":
            delta["thinking"] = content
        elif delta_type == "signature_delta":
            delta["signature"] = content
        elif delta_type == "text_delta":
            delta["text"] = content
        elif delta_type == "input_json_delta":
            delta["partial_json"] = content

        self._record_block_delta(index, delta)
        return self._emitter.event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": delta,
            },
        )

    def content_block_stop(self, index: int) -> str:
        self._record_block_stop(index)
        return self._emitter.event(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        )

    def start_thinking_block(self) -> str:
        self.blocks.thinking_index = self.blocks.allocate_index()
        self.blocks.thinking_started = True
        return self.content_block_start(self.blocks.thinking_index, "thinking")

    def emit_thinking_delta(self, content: str) -> str:
        return self.content_block_delta(
            self.blocks.thinking_index, "thinking_delta", content
        )

    def stop_thinking_block(self) -> str:
        self.blocks.thinking_started = False
        return self.content_block_stop(self.blocks.thinking_index)

    def start_text_block(self) -> str:
        self.blocks.text_index = self.blocks.allocate_index()
        self.blocks.text_started = True
        return self.content_block_start(self.blocks.text_index, "text")

    def emit_text_delta(self, content: str) -> str:
        return self.content_block_delta(self.blocks.text_index, "text_delta", content)

    def stop_text_block(self) -> str:
        self.blocks.text_started = False
        return self.content_block_stop(self.blocks.text_index)

    def start_tool_block(
        self,
        tool_index: int,
        tool_id: str,
        name: str,
        *,
        extra_content: dict[str, Any] | None = None,
    ) -> str:
        block_idx = self.blocks.allocate_index()
        if tool_index in self.blocks.tool_states:
            state = self.blocks.tool_states[tool_index]
            state.block_index = block_idx
            state.tool_id = tool_id
            state.name = name
            if extra_content:
                state.extra_content = extra_content
            state.started = True
        else:
            self.blocks.tool_states[tool_index] = ToolBlockState(
                block_index=block_idx,
                tool_id=tool_id,
                name=name,
                extra_content=extra_content,
                started=True,
            )
        return self.content_block_start(
            block_idx,
            "tool_use",
            id=tool_id,
            name=name,
            extra_content=extra_content,
        )

    def emit_tool_delta(self, tool_index: int, partial_json: str) -> str:
        state = self.blocks.tool_states[tool_index]
        return self.content_block_delta(
            state.block_index, "input_json_delta", partial_json
        )

    def stop_tool_block(self, tool_index: int) -> str:
        return self.content_block_stop(self.blocks.tool_states[tool_index].block_index)

    def ensure_thinking_block(self) -> Iterator[str]:
        if self.blocks.text_started:
            yield self.stop_text_block()
        if not self.blocks.thinking_started:
            yield self.start_thinking_block()

    def ensure_text_block(self) -> Iterator[str]:
        if self.blocks.thinking_started:
            yield self.stop_thinking_block()
        if not self.blocks.text_started:
            yield self.start_text_block()

    def close_content_blocks(self) -> Iterator[str]:
        if self.blocks.thinking_started:
            yield self.stop_thinking_block()
        if self.blocks.text_started:
            yield self.stop_text_block()

    def close_all_blocks(self) -> Iterator[str]:
        yield from self.close_content_blocks()
        for tool_index, state in list(self.blocks.tool_states.items()):
            if state.started:
                yield self.stop_tool_block(tool_index)

    def close_unclosed_blocks(self) -> Iterator[str]:
        while self._open_stack:
            idx = self._open_stack.pop()
            state = self._content_blocks.get(idx)
            if state is not None:
                state.open = False
                self._clear_active_content_block(state)
            yield self._emitter.event(
                "content_block_stop",
                {"type": "content_block_stop", "index": idx},
            )

    def can_salvage_tool_use(self, schemas: dict[str, ToolSchema]) -> bool:
        tool_blocks = self.tool_blocks()
        if not tool_blocks:
            return False
        for block in tool_blocks:
            if not block.tool_id or not block.name:
                return False
            if parse_complete_tool_input(block.content, block.name, schemas) is None:
                return False
        return True

    def tool_blocks(self) -> list[StreamBlockState]:
        return [
            block
            for block in self._content_blocks.values()
            if block.block_type == "tool_use"
        ]

    def tool_block_for_tool_index(self, tool_index: int) -> StreamBlockState | None:
        state = self.blocks.tool_states.get(tool_index)
        if state is None or state.block_index < 0:
            return None
        block = self._content_blocks.get(state.block_index)
        if block is None or block.block_type != "tool_use":
            return None
        return block

    def has_emitted_tool_block(self) -> bool:
        return bool(self.tool_blocks())

    def has_content_block(self) -> bool:
        return bool(self._content_blocks)

    def final_stop_reason(self, fallback: str) -> str:
        if self.has_emitted_tool_block():
            return "tool_use"
        return fallback

    def has_terminal_message(self) -> bool:
        return self.message_stopped

    @property
    def accumulated_text(self) -> str:
        return "".join(self._text_parts)

    @property
    def accumulated_reasoning(self) -> str:
        return "".join(self._thinking_parts)

    def estimate_output_tokens(self) -> int:
        text_tokens = estimate_text_tokens(self.accumulated_text)
        reasoning_tokens = estimate_text_tokens(self.accumulated_reasoning)
        tool_tokens = 0
        tool_count = 0
        for name, content in self._iter_tool_token_payloads():
            tool_tokens += estimate_text_tokens(name)
            tool_tokens += estimate_text_tokens(content)
            tool_tokens += 15
            tool_count += 1

        block_count = (
            (1 if self.accumulated_reasoning else 0)
            + (1 if self.accumulated_text else 0)
            + tool_count
        )
        return text_tokens + reasoning_tokens + tool_tokens + (block_count * 4)

    def _iter_tool_token_payloads(self) -> Iterator[tuple[str, str]]:
        for block in self.tool_blocks():
            yield block.name, block.content

    def _record_block_start(self, index: int, block: dict[str, Any]) -> None:
        self.blocks.reserve_index(index)
        block_type = str(block.get("type", ""))
        state = StreamBlockState(index=index, block_type=block_type)
        if block_type == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            state.tool_id = tool_id if isinstance(tool_id, str) else ""
            state.name = name if isinstance(name, str) else ""
            extra_content = block.get("extra_content")
            state.extra_content = (
                extra_content if isinstance(extra_content, dict) else None
            )
        elif block_type == "text":
            self.blocks.text_index = index
            self.blocks.text_started = True
            text = block.get("text")
            if isinstance(text, str) and text:
                state.parts.append(text)
                self._text_parts.append(text)
        elif block_type == "thinking":
            self.blocks.thinking_index = index
            self.blocks.thinking_started = True
            thinking = block.get("thinking")
            if isinstance(thinking, str) and thinking:
                state.parts.append(thinking)
                self._thinking_parts.append(thinking)
        self._content_blocks[index] = state
        self._open_stack.append(index)

    def _record_block_delta(self, index: int, delta: dict[str, Any]) -> None:
        state = self._content_blocks.get(index)
        if state is None:
            return
        if state.block_type == "text":
            text = delta.get("text")
            if isinstance(text, str):
                state.parts.append(text)
                self._text_parts.append(text)
        elif state.block_type == "thinking":
            thinking = delta.get("thinking")
            if isinstance(thinking, str):
                state.parts.append(thinking)
                self._thinking_parts.append(thinking)
        elif state.block_type == "tool_use":
            partial = delta.get("partial_json")
            if isinstance(partial, str):
                state.parts.append(partial)

    def _record_block_stop(self, index: int) -> None:
        if self._open_stack and self._open_stack[-1] == index:
            self._open_stack.pop()
        else:
            with suppress(ValueError):
                self._open_stack.remove(index)
        state = self._content_blocks.get(index)
        if state is not None:
            state.open = False
            self._clear_active_content_block(state)

    def _last_open_block(self, block_type: str) -> StreamBlockState | None:
        for index in reversed(self._open_stack):
            block = self._content_blocks.get(index)
            if block is not None and block.block_type == block_type and block.open:
                return block
        return None

    def _clear_active_content_block(self, state: StreamBlockState) -> None:
        if state.block_type == "text" and self.blocks.text_index == state.index:
            self.blocks.text_started = False
        elif (
            state.block_type == "thinking" and self.blocks.thinking_index == state.index
        ):
            self.blocks.thinking_started = False


def _normalize_task_run_in_background(args_json: dict[str, Any]) -> None:
    if args_json.get("run_in_background") is not False:
        args_json["run_in_background"] = False

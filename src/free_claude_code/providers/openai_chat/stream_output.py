"""Chat-source output writers for Anthropic Messages and OpenAI Responses."""

import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

from loguru import logger

from free_claude_code.core.anthropic.streaming import (
    AnthropicStreamLedger,
    ToolSchema,
    parse_complete_tool_input,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    ReasoningBlockState,
    ResponseBlockCompleter,
    ResponseEventBuilder,
    ResponsesConversionError,
    ResponsesOutputLedger,
    ResponsesToolAdapter,
    TextBlockState,
    ToolBlockState,
    new_call_id,
    new_message_item_id,
    new_reasoning_item_id,
    new_response_id,
    openai_error_from_failure,
    reasoning_output_item,
    replay_unsafe_function_call_error,
    tool_item,
)
from free_claude_code.core.token_estimation import estimate_text_tokens
from free_claude_code.core.trace import trace_event


@dataclass(frozen=True, slots=True)
class ChatStreamUsage:
    """Final Chat usage normalized for either client wire protocol."""

    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cache_write_tokens: int | None = None
    reasoning_tokens: int = 0
    anthropic_fields: Mapping[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ChatToolState:
    """Chat parser state for one streamed tool call."""

    tool_id: str = ""
    name: str = ""
    extra_content: JsonObject | None = None
    started: bool = False
    open: bool = False
    task_arg_buffer: str = ""
    task_args_emitted: bool = False
    pre_start_args: str = ""
    argument_parts: list[str] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "".join(self.argument_parts)


class ChatStreamOutput(ABC):
    """Source-specific semantic output boundary for one Chat stream epoch."""

    consumes_terminal_failure = False

    def __init__(self, *, input_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.replay_origin: ReplayOrigin | None = None
        self.reasoning_replay_events: (
            Callable[[ChatStreamOutput], Iterator[str]] | None
        ) = None
        self.tool_states: dict[int, ChatToolState] = {}
        self._text_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._text_started = False
        self._reasoning_started = False
        self._content_started = False
        self._terminal = False

    @property
    def committed_output(self) -> bool:
        return self._content_started

    @property
    def accumulated_text(self) -> str:
        return "".join(self._text_parts)

    @property
    def accumulated_reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    def start_events(self) -> list[str]:
        return self._start_events()

    def ensure_reasoning_block(self) -> list[str]:
        events: list[str] = []
        if self._text_started:
            events.extend(self._stop_text_block())
            self._text_started = False
        if not self._reasoning_started:
            events.extend(self._start_reasoning_block())
            self._reasoning_started = True
            self._content_started = True
        return events

    def emit_reasoning_delta(self, content: str) -> str:
        self._reasoning_parts.append(content)
        return self._emit_reasoning_delta(content)

    def emit_reasoning_replay(self, native: JsonObject) -> list[str]:
        data = json.dumps(native["reasoning_details"], separators=(",", ":"))
        if self.replay_origin is not None:
            data = encode_replay(ReplayRecord(self.replay_origin, native))
            if self._reasoning_started:
                return self._attach_reasoning_replay(data)
        events = self._close_content_blocks()
        events.extend(self._emit_opaque_reasoning(data))
        if data:
            self._content_started = True
        return events

    def flush_reasoning_replay(self) -> list[str]:
        """Save the complete record when parsed output ends a reasoning block."""
        if self.reasoning_replay_events is None:
            return []
        return list(self.reasoning_replay_events(self))

    def ensure_text_block(self) -> list[str]:
        events = self.flush_reasoning_replay()
        if self._reasoning_started:
            events.extend(self._stop_reasoning_block())
            self._reasoning_started = False
        if not self._text_started:
            events.extend(self._start_text_block())
            self._text_started = True
            self._content_started = True
        return events

    def emit_text_delta(self, content: str) -> str:
        self._text_parts.append(content)
        return self._emit_text_delta(content)

    def close_content_blocks(self) -> list[str]:
        events = self.flush_reasoning_replay()
        events.extend(self._close_content_blocks())
        return events

    def _close_content_blocks(self) -> list[str]:
        events: list[str] = []
        if self._reasoning_started:
            events.extend(self._stop_reasoning_block())
            self._reasoning_started = False
        if self._text_started:
            events.extend(self._stop_text_block())
            self._text_started = False
        return events

    def ensure_tool_state(self, tool_index: int) -> ChatToolState:
        return self.tool_states.setdefault(tool_index, ChatToolState())

    def set_tool_extra_content(
        self, tool_index: int, extra_content: JsonObject | None
    ) -> None:
        if extra_content:
            self.ensure_tool_state(tool_index).extra_content = extra_content

    def register_tool_name(self, tool_index: int, name: str) -> None:
        state = self.ensure_tool_state(tool_index)
        previous = state.name
        if not previous or name.startswith(previous):
            state.name = name
        elif not previous.startswith(name):
            state.name = previous + name

    def start_tool_block(
        self,
        tool_index: int,
        tool_id: str,
        name: str,
        *,
        extra_content: JsonObject | None = None,
    ) -> str:
        state = self.ensure_tool_state(tool_index)
        state.tool_id = tool_id
        state.name = name
        if extra_content:
            state.extra_content = extra_content
        state.started = True
        state.open = True
        self._content_started = True
        return self._start_tool_block(tool_index, state)

    def emit_tool_delta(self, tool_index: int, partial_json: str) -> str:
        state = self.tool_states[tool_index]
        state.argument_parts.append(partial_json)
        return self._emit_tool_delta(tool_index, state, partial_json)

    def stop_tool_block(self, tool_index: int) -> list[str]:
        state = self.tool_states[tool_index]
        if not state.open:
            return []
        state.open = False
        return self._stop_tool_block(tool_index, state)

    def close_all_blocks(self) -> list[str]:
        events = self.close_content_blocks()
        for tool_index, state in self.tool_states.items():
            if state.open:
                events.extend(self.stop_tool_block(tool_index))
        return events

    def close_unclosed_blocks(self) -> list[str]:
        return self.close_all_blocks()

    def has_emitted_tool_block(self) -> bool:
        return any(state.started for state in self.tool_states.values())

    def has_content_block(self) -> bool:
        return self._content_started

    def final_stop_reason(self, fallback: str) -> str:
        return "tool_use" if self.has_emitted_tool_block() else fallback

    def tool_block_for_tool_index(self, tool_index: int) -> ChatToolState | None:
        state = self.tool_states.get(tool_index)
        return state if state is not None and state.started else None

    def started_tool_states(self) -> list[tuple[int, ChatToolState]]:
        return [
            (tool_index, state)
            for tool_index, state in self.tool_states.items()
            if state.started
        ]

    def can_salvage_tool_use(self, schemas: dict[str, ToolSchema]) -> bool:
        states = [state for state in self.tool_states.values() if state.started]
        return bool(states) and all(
            state.tool_id
            and state.name
            and parse_complete_tool_input(state.content, state.name, schemas)
            is not None
            for state in states
        )

    def buffer_task_args(
        self, tool_index: int, arguments: str
    ) -> dict[str, object] | None:
        state = self.tool_states.get(tool_index)
        if state is None or state.task_args_emitted:
            return None
        state.task_arg_buffer += arguments
        try:
            parsed = json.loads(state.task_arg_buffer)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        _normalize_task_args(parsed)
        state.task_args_emitted = True
        state.task_arg_buffer = ""
        return parsed

    def flush_task_arg_buffers(self) -> list[tuple[int, str]]:
        results: list[tuple[int, str]] = []
        for tool_index, state in self.tool_states.items():
            if not state.task_arg_buffer or state.task_args_emitted:
                continue
            output = "{}"
            try:
                parsed = json.loads(state.task_arg_buffer)
                if isinstance(parsed, dict):
                    _normalize_task_args(parsed)
                    output = json.dumps(parsed)
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
            results.append((tool_index, output))
        return results

    def estimate_output_tokens(self) -> int:
        tool_tokens = sum(
            estimate_text_tokens(state.name) + estimate_text_tokens(state.content) + 15
            for state in self.tool_states.values()
            if state.started
        )
        block_count = (
            (1 if self.accumulated_reasoning else 0)
            + (1 if self.accumulated_text else 0)
            + sum(1 for state in self.tool_states.values() if state.started)
        )
        return (
            estimate_text_tokens(self.accumulated_text)
            + estimate_text_tokens(self.accumulated_reasoning)
            + tool_tokens
            + (block_count * 4)
        )

    def finish_success(self, *, stop_reason: str, usage: ChatStreamUsage) -> list[str]:
        if self._terminal:
            return []
        events = self.close_all_blocks()
        events.extend(self._finish_success(stop_reason=stop_reason, usage=usage))
        self._terminal = True
        return events

    def finish_failure(self, failure: ExecutionFailure) -> list[str]:
        if self._terminal:
            return []
        events = self.close_unclosed_blocks()
        events.extend(self._finish_failure(failure))
        self._terminal = True
        return events

    @abstractmethod
    def _start_events(self) -> list[str]: ...

    @abstractmethod
    def _start_reasoning_block(self) -> list[str]: ...

    @abstractmethod
    def _emit_reasoning_delta(self, content: str) -> str: ...

    @abstractmethod
    def _emit_opaque_reasoning(self, data: str) -> list[str]: ...

    @abstractmethod
    def _attach_reasoning_replay(self, data: str) -> list[str]: ...

    @abstractmethod
    def _stop_reasoning_block(self) -> list[str]: ...

    @abstractmethod
    def _start_text_block(self) -> list[str]: ...

    @abstractmethod
    def _emit_text_delta(self, content: str) -> str: ...

    @abstractmethod
    def _stop_text_block(self) -> list[str]: ...

    @abstractmethod
    def _start_tool_block(self, tool_index: int, state: ChatToolState) -> str: ...

    @abstractmethod
    def _emit_tool_delta(
        self, tool_index: int, state: ChatToolState, partial_json: str
    ) -> str: ...

    @abstractmethod
    def _stop_tool_block(self, tool_index: int, state: ChatToolState) -> list[str]: ...

    @abstractmethod
    def _finish_success(
        self, *, stop_reason: str, usage: ChatStreamUsage
    ) -> list[str]: ...

    @abstractmethod
    def _finish_failure(self, failure: ExecutionFailure) -> list[str]: ...


class AnthropicChatStreamOutput(ChatStreamOutput):
    """Anthropic SSE writer preserving the established Chat provider wire."""

    def __init__(
        self,
        *,
        message_id: str,
        model: str,
        input_tokens: int,
        log_raw_events: bool = False,
    ) -> None:
        super().__init__(input_tokens=input_tokens)
        self._ledger = AnthropicStreamLedger(
            message_id,
            model,
            input_tokens,
            log_raw_events=log_raw_events,
        )

    def close_unclosed_blocks(self) -> list[str]:
        events = self.flush_reasoning_replay()
        events.extend(self._ledger.close_unclosed_blocks())
        self._text_started = False
        self._reasoning_started = False
        for state in self.tool_states.values():
            state.open = False
        return events

    def _start_events(self) -> list[str]:
        return [self._ledger.message_start()]

    def _start_reasoning_block(self) -> list[str]:
        return [self._ledger.start_thinking_block()]

    def _emit_reasoning_delta(self, content: str) -> str:
        return self._ledger.emit_thinking_delta(content)

    def _emit_opaque_reasoning(self, data: str) -> list[str]:
        index = self._ledger.blocks.allocate_index()
        return [
            self._ledger.content_block_start(index, "redacted_thinking", data=data),
            self._ledger.content_block_stop(index),
        ]

    def _attach_reasoning_replay(self, data: str) -> list[str]:
        return [
            self._ledger.content_block_delta(
                self._ledger.blocks.thinking_index, "signature_delta", data
            )
        ]

    def _stop_reasoning_block(self) -> list[str]:
        return [self._ledger.stop_thinking_block()]

    def _start_text_block(self) -> list[str]:
        return [self._ledger.start_text_block()]

    def _emit_text_delta(self, content: str) -> str:
        return self._ledger.emit_text_delta(content)

    def _stop_text_block(self) -> list[str]:
        return [self._ledger.stop_text_block()]

    def _start_tool_block(self, tool_index: int, state: ChatToolState) -> str:
        return self._ledger.start_tool_block(
            tool_index,
            state.tool_id,
            state.name,
            extra_content=state.extra_content,
        )

    def _emit_tool_delta(
        self, tool_index: int, state: ChatToolState, partial_json: str
    ) -> str:
        return self._ledger.emit_tool_delta(tool_index, partial_json)

    def _stop_tool_block(self, tool_index: int, state: ChatToolState) -> list[str]:
        return [self._ledger.stop_tool_block(tool_index)]

    def _finish_success(self, *, stop_reason: str, usage: ChatStreamUsage) -> list[str]:
        return [
            self._ledger.message_delta(
                self.final_stop_reason(stop_reason),
                usage.output_tokens,
                input_tokens=usage.input_tokens,
                usage_fields=usage.anthropic_fields,
            ),
            self._ledger.message_stop(),
        ]

    def _finish_failure(self, failure: ExecutionFailure) -> list[str]:
        return []


class ResponsesChatStreamOutput(ChatStreamOutput):
    """Direct Chat-to-Responses writer with one coherent Responses lifecycle."""

    consumes_terminal_failure = True

    def __init__(
        self,
        tool_adapter: ResponsesToolAdapter,
        *,
        input_tokens: int,
        response_model: str | None = None,
    ) -> None:
        super().__init__(input_tokens=input_tokens)
        self._request = tool_adapter.original
        self._response_model = response_model or self._request.model
        self._response_id = new_response_id()
        self._created_at = int(time.time())
        self._ledger = ResponsesOutputLedger()
        tool_events = tool_adapter.event_adapter()
        self._events = ResponseEventBuilder(tool_events.feed if tool_events else None)
        self._completer = ResponseBlockCompleter(
            self._ledger,
            events=self._events,
            on_invalid_function_call=self._fail_invalid_function_call,
            prepare_tool_arguments=tool_adapter.prepare_arguments,
        )
        self._text_state: TextBlockState | None = None
        self._reasoning_state: ReasoningBlockState | None = None
        self._tool_output_states: dict[int, ToolBlockState] = {}
        self._usage: dict[str, object] | None = None
        self._provisional_error: dict[str, object] | None = None
        self._started = False

    def _response_payload(
        self,
        *,
        status: str,
        error: Mapping[str, object] | None = None,
        incomplete_details: Mapping[str, str] | None = None,
    ) -> dict[str, object]:
        return {
            "id": self._response_id,
            "object": "response",
            "created_at": self._created_at,
            "status": status,
            "model": self._response_model,
            "output": self._ledger.output(),
            "parallel_tool_calls": (
                True
                if self._request.parallel_tool_calls is None
                else self._request.parallel_tool_calls
            ),
            "tool_choice": (
                "auto"
                if self._request.tool_choice is None
                else self._request.tool_choice
            ),
            "temperature": self._request.temperature,
            "top_p": self._request.top_p,
            "max_output_tokens": self._request.max_output_tokens,
            "usage": self._usage,
            "error": dict(error) if error is not None else None,
            "incomplete_details": (
                dict(incomplete_details) if incomplete_details is not None else None
            ),
        }

    def _start_events(self) -> list[str]:
        if self._started:
            return []
        self._started = True
        return [
            self._events.response_created(self._response_payload(status="in_progress"))
        ]

    def _start_reasoning_block(self) -> list[str]:
        output_index = self._ledger.reserve_output_slot()
        state = ReasoningBlockState(
            index=output_index,
            output_index=output_index,
            item_id=new_reasoning_item_id(),
        )
        self._reasoning_state = state
        self._ledger.set_active_block(state)
        return [
            self._events.output_item_added(
                output_index,
                reasoning_output_item(state, status="in_progress"),
            )
        ]

    def _emit_reasoning_delta(self, content: str) -> str:
        state = self._reasoning_state
        if state is None or not content:
            return ""
        state.text_parts.append(content)
        return self._events.reasoning_text_delta(
            state.item_id, state.output_index, content
        )

    def _emit_opaque_reasoning(self, data: str) -> list[str]:
        output_index = self._ledger.reserve_output_slot()
        state = ReasoningBlockState(
            index=output_index,
            output_index=output_index,
            item_id=new_reasoning_item_id(),
            encrypted_content=data,
        )
        self._ledger.set_active_block(state)
        events = [
            self._events.output_item_added(
                output_index,
                reasoning_output_item(state, status="in_progress"),
            )
        ]
        self._ledger.pop_active_block(state.index)
        events.extend(self._completer.complete_block(state))
        return events

    def _attach_reasoning_replay(self, data: str) -> list[str]:
        if self._reasoning_state is not None:
            self._reasoning_state.encrypted_content = data
        return []

    def _stop_reasoning_block(self) -> list[str]:
        state = self._reasoning_state
        self._reasoning_state = None
        if state is None:
            return []
        self._ledger.pop_active_block(state.index)
        return self._completer.complete_block(state)

    def _start_text_block(self) -> list[str]:
        output_index = self._ledger.reserve_output_slot()
        state = TextBlockState(
            index=output_index,
            output_index=output_index,
            item_id=new_message_item_id(),
        )
        self._text_state = state
        self._ledger.set_active_block(state)
        item = {
            "id": state.item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        return [
            self._events.output_item_added(output_index, item),
            self._events.content_part_added(state.item_id, output_index),
        ]

    def _emit_text_delta(self, content: str) -> str:
        state = self._text_state
        if state is None or not content:
            return ""
        state.text_parts.append(content)
        return self._events.output_text_delta(
            state.item_id, state.output_index, content
        )

    def _stop_text_block(self) -> list[str]:
        state = self._text_state
        self._text_state = None
        if state is None:
            return []
        self._ledger.pop_active_block(state.index)
        return self._completer.complete_block(state)

    def _start_tool_block(self, tool_index: int, state: ChatToolState) -> str:
        output_index = self._ledger.reserve_output_slot()
        output_state = ToolBlockState(
            index=output_index,
            output_index=output_index,
            item_id=f"fc_{uuid.uuid4().hex[:24]}",
            call_id=state.tool_id or new_call_id(),
            kind="function",
            name=state.name,
        )
        self._tool_output_states[tool_index] = output_state
        self._ledger.set_active_block(output_state)
        return self._events.output_item_added(
            output_index,
            tool_item(output_state, status="in_progress"),
        )

    def _emit_tool_delta(
        self, tool_index: int, state: ChatToolState, partial_json: str
    ) -> str:
        output_state = self._tool_output_states.get(tool_index)
        if output_state is not None:
            output_state.argument_parts.append(partial_json)
        return ""

    def _stop_tool_block(self, tool_index: int, state: ChatToolState) -> list[str]:
        output_state = self._tool_output_states.get(tool_index)
        if output_state is None:
            return []
        self._ledger.pop_active_block(output_state.index)
        return self._completer.complete_block(output_state)

    def _finish_success(self, *, stop_reason: str, usage: ChatStreamUsage) -> list[str]:
        self._usage = _responses_usage(usage)
        if self._provisional_error is not None:
            response = self._response_payload(
                status="failed",
                error=self._provisional_error,
            )
            return [self._events.response_failed(response)]
        if stop_reason in {"length", "max_tokens"}:
            response = self._response_payload(
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            )
            return [self._events.response_incomplete(response)]
        response = self._response_payload(status="completed")
        return [self._events.response_completed(response)]

    def _finish_failure(self, failure: ExecutionFailure) -> list[str]:
        response = self._response_payload(
            status="failed",
            error=openai_error_from_failure(failure),
        )
        return [self._events.response_failed(response)]

    def _fail_invalid_function_call(
        self, state: ToolBlockState, exc: ResponsesConversionError
    ) -> list[str]:
        trace_event(
            stage="responses",
            event="responses.output.function_call_invalid_arguments",
            source="openai_responses",
            call_id=state.call_id,
            tool_name=state.name,
            error_type=type(exc).__name__,
        )
        if self._provisional_error is None:
            self._provisional_error = replay_unsafe_function_call_error()
        return []


def _responses_usage(usage: ChatStreamUsage) -> dict[str, object]:
    cached_tokens = usage.cached_tokens
    if (
        not isinstance(cached_tokens, int)
        or isinstance(cached_tokens, bool)
        or not 0 <= cached_tokens <= usage.input_tokens
    ):
        cached_tokens = 0
    reasoning_tokens = max(0, min(usage.reasoning_tokens, usage.output_tokens))
    input_token_details = {"cached_tokens": cached_tokens}
    cache_write_tokens = usage.cache_write_tokens
    if (
        isinstance(cache_write_tokens, int)
        and not isinstance(cache_write_tokens, bool)
        and 0 <= cache_write_tokens <= usage.input_tokens - cached_tokens
    ):
        input_token_details["cache_write_tokens"] = cache_write_tokens
    return {
        "input_tokens": usage.input_tokens,
        "input_tokens_details": input_token_details,
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }


def _normalize_task_args(arguments: dict[str, object]) -> None:
    if arguments.get("run_in_background") is not False:
        arguments["run_in_background"] = False

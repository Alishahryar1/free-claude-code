"""Chat-source output writers for Anthropic Messages and OpenAI Responses."""

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from free_claude_code.core.anthropic.streaming import (
    AnthropicStreamLedger,
    ToolSchema,
    parse_complete_tool_input,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import (
    AssociatedReplayRecord,
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
    tool_item,
)
from free_claude_code.core.token_estimation import estimate_text_tokens


class ReasoningReplayLifecycle(Protocol):
    @property
    def active(self) -> bool: ...

    def before_reasoning(self, output: ChatStreamOutput) -> Iterator[str]: ...

    def before_content(self, output: ChatStreamOutput) -> Iterator[str]: ...

    def finish(self, output: ChatStreamOutput) -> Iterator[str]: ...


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
        self.reasoning_replay: ReasoningReplayLifecycle | None = None
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
        if self.reasoning_replay is not None:
            events.extend(self.reasoning_replay.before_reasoning(self))
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

    def begin_reasoning_record(self) -> list[str]:
        return []

    def pause_reasoning_record(self, group_id: str, record: ReplayRecord) -> list[str]:
        return []

    def complete_reasoning_record(
        self, group_id: str, record: ReplayRecord
    ) -> list[str]:
        data = encode_replay(record)
        if self._reasoning_started:
            events = self._attach_reasoning_replay(data)
            events.extend(self._stop_reasoning_block())
            self._reasoning_started = False
            return events
        return self._emit_opaque_reasoning(data)

    def flush_reasoning_replay(self) -> list[str]:
        if self.reasoning_replay is None:
            return []
        return list(self.reasoning_replay.finish(self))

    def _pause_reasoning(self) -> list[str]:
        events: list[str] = []
        if self.reasoning_replay is not None:
            events.extend(self.reasoning_replay.before_content(self))
        if self._reasoning_started and not (
            self.reasoning_replay and self.reasoning_replay.active
        ):
            events.extend(self._stop_reasoning_block())
            self._reasoning_started = False
        return events

    def ensure_text_block(self) -> list[str]:
        events = self._pause_reasoning()
        if not self._text_started:
            events.extend(self._start_text_block())
            self._text_started = True
            self._content_started = True
        return events

    def emit_text_delta(self, content: str) -> str:
        self._text_parts.append(content)
        return self._emit_text_delta(content)

    def close_content_blocks(self) -> list[str]:
        events = self._pause_reasoning()
        if self._text_started:
            events.extend(self._stop_text_block())
            self._text_started = False
        return events

    def finish_reasoning_group(self) -> list[str]:
        events = self.flush_reasoning_replay()
        events.extend(self._close_content_blocks())
        return events

    def finish_replay_carriers(self) -> list[str]:
        return []

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
        events = self.finish_reasoning_group()
        for tool_index, state in self.tool_states.items():
            if state.open:
                events.extend(self.stop_tool_block(tool_index))
        events.extend(self.finish_replay_carriers())
        return events

    def close_unclosed_blocks(self) -> list[str]:
        events = self.finish_reasoning_group()
        for state in self.tool_states.values():
            state.open = False
        events.extend(self.finish_replay_carriers())
        return events

    def has_emitted_tool_block(self) -> bool:
        return any(state.started for state in self.tool_states.values())

    def has_content_block(self) -> bool:
        return self._content_started

    def final_stop_reason(self, fallback: str) -> str:
        if self.has_emitted_tool_block():
            return "tool_use"
        return "end_turn" if fallback == "tool_use" else fallback

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

    def failure_payload(self, failure: ExecutionFailure) -> JsonObject | None:
        return None

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
        self._anchored_groups: set[str] = set()
        self._pending_replay: list[str] = []
        self._ledger = AnthropicStreamLedger(
            message_id,
            model,
            input_tokens,
            log_raw_events=log_raw_events,
        )

    def pause_reasoning_record(self, group_id: str, record: ReplayRecord) -> list[str]:
        if group_id in self._anchored_groups:
            return []
        if not self._reasoning_started and any(
            state.open for state in self.tool_states.values()
        ):
            return []
        self._anchored_groups.add(group_id)
        data = encode_replay(
            AssociatedReplayRecord(record.origin, record.native, group_id, "anchor")
        )
        self._content_started = True
        if self._reasoning_started:
            events = self._attach_reasoning_replay(data)
            events.extend(self._stop_reasoning_block())
            self._reasoning_started = False
            return events
        events = self._close_content_blocks()
        events.extend(self._emit_opaque_reasoning(data))
        return events

    def complete_reasoning_record(
        self, group_id: str, record: ReplayRecord
    ) -> list[str]:
        if group_id in self._anchored_groups:
            self._anchored_groups.remove(group_id)
            self._pending_replay.append(
                encode_replay(
                    AssociatedReplayRecord(
                        record.origin, record.native, group_id, "final"
                    )
                )
            )
            return []
        if self._reasoning_started:
            return super().complete_reasoning_record(group_id, record)
        self._pending_replay.append(encode_replay(record))
        self._content_started = True
        return []

    def finish_replay_carriers(self) -> list[str]:
        pending, self._pending_replay = self._pending_replay, []
        return [
            event for data in pending for event in self._emit_opaque_reasoning(data)
        ]

    def close_unclosed_blocks(self) -> list[str]:
        events = self.flush_reasoning_replay()
        events.extend(self._ledger.close_unclosed_blocks())
        events.extend(self.finish_replay_carriers())
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
        )
        self._text_state: TextBlockState | None = None
        self._reasoning_state: ReasoningBlockState | None = None
        self._tool_output_states: dict[int, ToolBlockState] = {}
        self._usage: dict[str, object] | None = None
        self._conversion_failure: ExecutionFailure | None = None
        self._started = False

    def begin_reasoning_record(self) -> list[str]:
        if self._reasoning_started:
            return []
        self._reasoning_started = True
        self._content_started = True
        return self._start_reasoning_block()

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
        try:
            return self._completer.complete_block(output_state)
        except ResponsesConversionError:
            # Object-valued protocol conversions (such as tool search) cannot
            # carry opaque malformed JSON. Ordinary function arguments can.
            self._conversion_failure = ExecutionFailure(
                FailureKind.UPSTREAM,
                502,
                "Provider tool output cannot be represented in the requested protocol.",
                False,
            )
            return []

    def _finish_success(self, *, stop_reason: str, usage: ChatStreamUsage) -> list[str]:
        self._usage = _responses_usage(usage)
        if self._conversion_failure is not None:
            return self._finish_failure(self._conversion_failure)
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

    def failure_payload(self, failure: ExecutionFailure) -> JsonObject:
        self._completer.retain_incomplete_blocks()
        self._terminal = True
        return self._events.response_failed_payload(
            self._response_payload(
                status="failed", error=openai_error_from_failure(failure)
            )
        )


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

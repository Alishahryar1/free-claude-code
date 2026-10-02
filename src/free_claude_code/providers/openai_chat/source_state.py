"""Assemble Chat content and source boundaries without a public response ledger."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

from free_claude_code.core.chat_observations import (
    ChatChange,
    ChatStreamUsage,
    ChatToolObservation,
)
from free_claude_code.core.history_replay import ReplayOrigin, ReplayRecord
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.token_estimation import estimate_text_tokens


class ReasoningReplayLifecycle(Protocol):
    @property
    def active(self) -> bool: ...

    def before_reasoning(self, output: ChatSourceState) -> Iterator[ChatChange]: ...

    def before_content(self, output: ChatSourceState) -> Iterator[ChatChange]: ...

    def finish(self, output: ChatSourceState) -> Iterator[ChatChange]: ...


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


class ChatSourceState:
    """Source-specific semantic output boundary for one Chat stream epoch."""

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
        self._text_offset = 0
        self._reasoning_offset = 0

    @property
    def accumulated_text(self) -> str:
        return "".join(self._text_parts)

    @property
    def accumulated_reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    def start_events(self) -> list[ChatChange]:
        return self._start_events()

    def ensure_reasoning_block(self) -> list[ChatChange]:
        events: list[ChatChange] = []
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

    def emit_reasoning_delta(self, content: str) -> ChatChange:
        self._reasoning_parts.append(content)
        return self._emit_reasoning_delta(content)

    def begin_reasoning_record(self) -> list[ChatChange]:
        self._content_started = True
        if not self._reasoning_started:
            self._reasoning_offset = len(self._reasoning_parts)
        return [ChatChange("reasoning_record.start")]

    def pause_reasoning_record(
        self, group_id: str, record: ReplayRecord
    ) -> list[ChatChange]:
        return [ChatChange("reasoning_record.pause", group_id=group_id, record=record)]

    def complete_reasoning_record(
        self, group_id: str, record: ReplayRecord
    ) -> list[ChatChange]:
        self._reasoning_started = False
        self._content_started = True
        return [
            ChatChange(
                "reasoning_record.complete",
                text="".join(self._reasoning_parts[self._reasoning_offset :]),
                group_id=group_id,
                record=record,
            )
        ]

    def flush_reasoning_replay(self) -> list[ChatChange]:
        if self.reasoning_replay is None:
            return []
        return list(self.reasoning_replay.finish(self))

    def _pause_reasoning(self) -> list[ChatChange]:
        events: list[ChatChange] = []
        if self.reasoning_replay is not None:
            events.extend(self.reasoning_replay.before_content(self))
        if self._reasoning_started and not (
            self.reasoning_replay and self.reasoning_replay.active
        ):
            events.extend(self._stop_reasoning_block())
            self._reasoning_started = False
        return events

    def ensure_text_block(self) -> list[ChatChange]:
        events = self._pause_reasoning()
        if not self._text_started:
            events.extend(self._start_text_block())
            self._text_started = True
            self._content_started = True
        return events

    def emit_text_delta(self, content: str) -> ChatChange:
        self._text_parts.append(content)
        return self._emit_text_delta(content)

    def close_content_blocks(self) -> list[ChatChange]:
        events = self._pause_reasoning()
        if self._text_started:
            events.extend(self._stop_text_block())
            self._text_started = False
        return events

    def finish_reasoning_group(self) -> list[ChatChange]:
        events = self.flush_reasoning_replay()
        events.extend(self._close_content_blocks())
        return events

    def finish_replay_carriers(self) -> list[ChatChange]:
        return [ChatChange("replay.flush")]

    def _close_content_blocks(self) -> list[ChatChange]:
        events: list[ChatChange] = []
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
    ) -> ChatChange:
        state = self.ensure_tool_state(tool_index)
        state.tool_id = tool_id
        state.name = name
        if extra_content:
            state.extra_content = extra_content
        state.started = True
        state.open = True
        self._content_started = True
        return self._start_tool_block(tool_index, state)

    def emit_tool_delta(self, tool_index: int, partial_json: str) -> list[ChatChange]:
        state = self.tool_states[tool_index]
        state.argument_parts.append(partial_json)
        return self._emit_tool_delta(tool_index, state, partial_json)

    def stop_tool_block(self, tool_index: int) -> list[ChatChange]:
        state = self.tool_states[tool_index]
        if not state.open:
            return []
        state.open = False
        return self._stop_tool_block(tool_index, state)

    def close_all_blocks(self) -> list[ChatChange]:
        events = self.finish_reasoning_group()
        for tool_index, state in self.tool_states.items():
            if state.open:
                events.extend(self.stop_tool_block(tool_index))
        events.extend(self.finish_replay_carriers())
        return events

    def has_emitted_tool_block(self) -> bool:
        return any(state.started for state in self.tool_states.values())

    def has_content_block(self) -> bool:
        return self._content_started

    def final_stop_reason(self, fallback: str) -> str:
        if fallback not in {"end_turn", "tool_use"}:
            return fallback
        if self.has_emitted_tool_block():
            return "tool_use"
        return "end_turn" if fallback == "tool_use" else fallback

    def tool_block_for_tool_index(self, tool_index: int) -> ChatToolState | None:
        state = self.tool_states.get(tool_index)
        return state if state is not None and state.started else None

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

    def finish_success(
        self, *, stop_reason: str, usage: ChatStreamUsage
    ) -> list[ChatChange]:
        if self._terminal:
            return []
        events = self.close_all_blocks()
        events.extend(self._finish_success(stop_reason=stop_reason, usage=usage))
        self._terminal = True
        return events

    def _start_events(self) -> list[ChatChange]:
        return [ChatChange("start")]

    def _start_reasoning_block(self) -> list[ChatChange]:
        self._reasoning_offset = len(self._reasoning_parts)
        return [ChatChange("reasoning.start")]

    def _emit_reasoning_delta(self, content: str) -> ChatChange:
        return ChatChange("reasoning.delta", text=content)

    def _stop_reasoning_block(self) -> list[ChatChange]:
        return [
            ChatChange(
                "reasoning.stop",
                text="".join(self._reasoning_parts[self._reasoning_offset :]),
            )
        ]

    def _start_text_block(self) -> list[ChatChange]:
        self._text_offset = len(self._text_parts)
        return [ChatChange("text.start")]

    def _emit_text_delta(self, content: str) -> ChatChange:
        return ChatChange("text.delta", text=content)

    def _stop_text_block(self) -> list[ChatChange]:
        return [
            ChatChange("text.stop", text="".join(self._text_parts[self._text_offset :]))
        ]

    def _start_tool_block(self, tool_index: int, state: ChatToolState) -> ChatChange:
        return ChatChange(
            "tool.start",
            tool_index=tool_index,
            tool=ChatToolObservation(
                state.tool_id, state.name, extra_content=state.extra_content
            ),
        )

    def _emit_tool_delta(
        self, tool_index: int, state: ChatToolState, partial_json: str
    ) -> list[ChatChange]:
        return [ChatChange("tool.delta", tool_index=tool_index, text=partial_json)]

    def _stop_tool_block(
        self, tool_index: int, state: ChatToolState
    ) -> list[ChatChange]:
        return [
            ChatChange(
                "tool.stop",
                tool_index=tool_index,
                tool=ChatToolObservation(
                    state.tool_id, state.name, state.content, state.extra_content
                ),
            )
        ]

    def _finish_success(
        self, *, stop_reason: str, usage: ChatStreamUsage
    ) -> list[ChatChange]:
        return [
            ChatChange(
                "complete", text=self.final_stop_reason(stop_reason), usage=usage
            )
        ]

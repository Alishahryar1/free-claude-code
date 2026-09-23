"""Transcript event application and open-block tracking."""

from typing import Any

from .context import RenderCtx
from .renderer import render_segments
from .segments import (
    ErrorSegment,
    Segment,
    SubagentSegment,
    TextSegment,
    ThinkingSegment,
    ToolCallSegment,
    ToolResultSegment,
)
from .subagents import SubagentState, task_heading_from_input

_SUBAGENT_SUPPRESSED_EVENTS = frozenset(
    {
        "thinking_start",
        "thinking_delta",
        "thinking_chunk",
        "text_start",
        "text_delta",
        "text_chunk",
    }
)


class TranscriptBuffer:
    """Maintains an ordered, truncatable transcript of parsed CLI events."""

    def __init__(
        self,
        *,
        show_tool_results: bool = True,
        debug_subagent_stack: bool = False,
    ) -> None:
        self._segments: list[Segment] = []
        self._open_thinking_by_index: dict[int, ThinkingSegment] = {}
        self._open_text_by_index: dict[int, TextSegment] = {}
        self._open_tools_by_index: dict[int, ToolCallSegment] = {}
        self._tool_name_by_id: dict[str, str] = {}
        self._show_tool_results = bool(show_tool_results)
        self._subagents = SubagentState(debug=debug_subagent_stack)

    def apply(self, event: dict[str, Any]) -> None:
        """Apply a parsed CLI transcript event."""
        event_type = event.get("type")
        if self._subagents.in_subagent() and event_type in _SUBAGENT_SUPPRESSED_EVENTS:
            return

        if event_type == "thinking_start":
            self._start_thinking(_event_index(event))
            return
        if event_type in ("thinking_delta", "thinking_chunk"):
            self._append_thinking(_event_index(event), str(event.get("text", "")))
            return
        if event_type == "thinking_stop":
            self._open_thinking_by_index.pop(_event_index(event), None)
            return

        if event_type == "text_start":
            self._start_text(_event_index(event))
            return
        if event_type in ("text_delta", "text_chunk"):
            self._append_text(_event_index(event), str(event.get("text", "")))
            return
        if event_type == "text_stop":
            self._open_text_by_index.pop(_event_index(event), None)
            return

        if event_type == "tool_use_start":
            self._start_tool_use(event)
            return
        if event_type == "tool_use_delta":
            return
        if event_type == "tool_use_stop":
            segment = self._open_tools_by_index.pop(_event_index(event), None)
            if segment is not None:
                segment.closed = True
            return

        if event_type == "block_stop":
            self._close_block(_event_index(event))
            return
        if event_type == "tool_use":
            self._append_complete_tool_use(event)
            return
        if event_type == "tool_result":
            self._append_tool_result(event)
            return
        if event_type == "error":
            self._segments.append(ErrorSegment(str(event.get("message", ""))))

    def render(self, ctx: RenderCtx, *, limit_chars: int, status: str | None) -> str:
        return render_segments(
            self._segments,
            ctx,
            limit_chars=limit_chars,
            status=status,
        )

    def _start_thinking(self, index: int) -> None:
        if index >= 0:
            self._close_block(index)
        segment = ThinkingSegment()
        self._segments.append(segment)
        if index >= 0:
            self._open_thinking_by_index[index] = segment

    def _append_thinking(self, index: int, text: str) -> None:
        segment = self._open_thinking_by_index.get(index)
        if segment is None:
            segment = ThinkingSegment()
            self._segments.append(segment)
            if index >= 0:
                self._open_thinking_by_index[index] = segment
        segment.append(text)

    def _start_text(self, index: int) -> None:
        if index >= 0:
            self._close_block(index)
        segment = TextSegment()
        self._segments.append(segment)
        if index >= 0:
            self._open_text_by_index[index] = segment

    def _append_text(self, index: int, text: str) -> None:
        segment = self._open_text_by_index.get(index)
        if segment is None:
            segment = TextSegment()
            self._segments.append(segment)
            if index >= 0:
                self._open_text_by_index[index] = segment
        segment.append(text)

    def _start_tool_use(self, event: dict[str, Any]) -> None:
        index = _event_index(event)
        if index >= 0:
            self._close_block(index)

        tool_id = _event_tool_id(event, "id")
        name = str(event.get("name", "") or "tool")
        if tool_id:
            self._tool_name_by_id[tool_id] = name

        if name == "Task":
            segment = SubagentSegment(task_heading_from_input(event.get("input")))
            self._segments.append(segment)
            self._subagents.push(tool_id, segment)
            return

        segment = self._append_tool_call(tool_id, name)
        if index >= 0:
            self._open_tools_by_index[index] = segment

    def _append_complete_tool_use(self, event: dict[str, Any]) -> None:
        tool_id = _event_tool_id(event, "id")
        name = str(event.get("name", "") or "tool")
        if tool_id:
            self._tool_name_by_id[tool_id] = name

        if name == "Task":
            segment = SubagentSegment(task_heading_from_input(event.get("input")))
            self._segments.append(segment)
            self._subagents.push(tool_id, segment)
            return

        segment = self._append_tool_call(tool_id, name)
        segment.closed = True

    def _append_tool_call(self, tool_id: str, name: str) -> ToolCallSegment:
        if self._subagents.in_subagent():
            parent = self._subagents.current_segment()
            if parent is not None:
                return parent.set_current_tool_call(tool_id, name)

        segment = ToolCallSegment(tool_id, name)
        self._segments.append(segment)
        return segment

    def _append_tool_result(self, event: dict[str, Any]) -> None:
        tool_id = _event_tool_id(event, "tool_use_id")
        name = self._tool_name_by_id.get(tool_id)

        if self._subagents.in_subagent():
            self._subagents.close_for_tool_result(tool_id, tool_name=name)

        if not self._show_tool_results:
            return

        self._segments.append(
            ToolResultSegment(
                tool_id,
                event.get("content"),
                name=name,
                is_error=bool(event.get("is_error", False)),
            )
        )

    def _close_block(self, index: int) -> None:
        if index in self._open_tools_by_index:
            segment = self._open_tools_by_index.pop(index, None)
            if segment is not None:
                segment.closed = True
            return
        if index in self._open_thinking_by_index:
            self._open_thinking_by_index.pop(index, None)
            return
        if index in self._open_text_by_index:
            self._open_text_by_index.pop(index, None)


def _event_index(event: dict[str, Any]) -> int:
    return int(event.get("index", -1))


def _event_tool_id(event: dict[str, Any], key: str) -> str:
    return str(event.get(key, "") or "").strip()

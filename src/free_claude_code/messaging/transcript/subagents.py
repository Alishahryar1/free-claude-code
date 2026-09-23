"""Task/subagent display state for messaging transcripts."""

from typing import Any

from loguru import logger

from .segments import SubagentSegment


class SubagentState:
    """Track active Task tool calls that suppress nested text/thinking output."""

    def __init__(self, *, debug: bool = False) -> None:
        self._stack: list[str] = []
        self._segments: list[SubagentSegment] = []
        self._debug = debug

    @property
    def open_ids(self) -> tuple[str, ...]:
        return tuple(self._stack)

    def in_subagent(self) -> bool:
        return bool(self._stack)

    def current_segment(self) -> SubagentSegment | None:
        return self._segments[-1] if self._segments else None

    def push(self, tool_id: str, segment: SubagentSegment) -> None:
        marker = str(tool_id or "").strip() or f"__task_{len(self._stack) + 1}"
        self._stack.append(marker)
        self._segments.append(segment)
        if self._debug:
            logger.debug(
                "SUBAGENT_STACK: push id=%r depth=%d heading=%r",
                marker,
                len(self._stack),
                segment.description,
            )

    def close_for_tool_result(self, tool_id: str, *, tool_name: str | None) -> bool:
        tool_id = str(tool_id or "").strip()
        popped = self._pop(tool_id)
        top = self._stack[-1] if self._stack else ""
        looks_like_task_id = "task" in tool_id.lower()

        if (
            not popped
            and tool_id
            and top.startswith("__task_")
            and tool_name in (None, "Task")
            and looks_like_task_id
        ):
            return self._pop("")
        return popped

    def _pop(self, tool_id: str) -> bool:
        tool_id = str(tool_id or "").strip()
        if not self._stack:
            return False

        if tool_id:
            if _ids_roughly_match(self._stack[-1], tool_id):
                self._pop_to_depth(len(self._stack) - 1, tool_id, "LIFO")
                return True

            for idx in range(len(self._stack) - 1, -1, -1):
                if _ids_roughly_match(self._stack[idx], tool_id):
                    self._pop_to_depth(idx, tool_id, "matched")
                    return True
            return False

        if self._stack[-1].startswith("__task_"):
            self._pop_to_depth(len(self._stack) - 1, self._stack[-1], "synthetic")
            return True
        return False

    def _pop_to_depth(self, idx: int, requested_id: str, reason: str) -> None:
        while len(self._stack) > idx:
            popped = self._stack.pop()
            if self._segments:
                self._segments.pop()
            if self._debug:
                logger.debug(
                    "SUBAGENT_STACK: pop id=%r depth=%d (%s=%r)",
                    popped,
                    len(self._stack),
                    reason,
                    requested_id,
                )


def task_heading_from_input(input_value: Any) -> str:
    if isinstance(input_value, dict):
        for key in ("description", "subagent_type", "type"):
            value = str(input_value.get(key, "") or "").strip()
            if value:
                return value
    return "Subagent"


def _ids_roughly_match(stack_id: str, result_id: str) -> bool:
    if not stack_id or not result_id:
        return False
    return (
        stack_id == result_id
        or stack_id.startswith(result_id)
        or result_id.startswith(stack_id)
    )

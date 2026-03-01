"""Ordered transcript builder for messaging UIs (Telegram, etc.).

This module maintains an ordered list of "segments" that represent what the user
should see in the chat transcript: thinking, tool calls, tool results, subagent
headers, and assistant text. It is designed for in-place message editing where
the transcript grows over time and older content must be truncated.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


@dataclass
class Segment(ABC):
    kind: str

    @abstractmethod
    def render(self, ctx: RenderCtx) -> str: ...


@dataclass
class ThinkingSegment(Segment):
    def __init__(self) -> None:
        super().__init__(kind="thinking")
        self._parts: list[str] = []

    def append(self, t: str) -> None:
        if t:
            self._parts.append(t)

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def render(self, ctx: RenderCtx) -> str:
        raw = self.text or ""
        if ctx.thinking_tail_max is not None and len(raw) > ctx.thinking_tail_max:
            raw = "..." + raw[-(ctx.thinking_tail_max - 3) :]
        inner = ctx.escape_code(raw)
        return f"💭 {ctx.bold('Thinking')}\n```\n{inner}\n```"


@dataclass
class TextSegment(Segment):
    def __init__(self) -> None:
        super().__init__(kind="text")
        self._parts: list[str] = []

    def append(self, t: str) -> None:
        if t:
            self._parts.append(t)

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def render(self, ctx: RenderCtx) -> str:
        raw = self.text or ""
        if ctx.text_tail_max is not None and len(raw) > ctx.text_tail_max:
            raw = "..." + raw[-(ctx.text_tail_max - 3) :]
        return ctx.render_markdown(raw)


@dataclass
class ToolCallSegment(Segment):
    tool_use_id: str
    name: str
    closed: bool = False
    indent_level: int = 0

    def __init__(self, tool_use_id: str, name: str, *, indent_level: int = 0) -> None:
        super().__init__(kind="tool_call")
        self.tool_use_id = str(tool_use_id or "")
        self.name = str(name or "tool")
        self.indent_level = max(0, int(indent_level))

    def render(self, ctx: RenderCtx) -> str:
        name = ctx.code_inline(self.name)
        # Per UX requirement: do not display tool args/results, only the tool call.
        prefix = "  " * self.indent_level
        return f"{prefix}🛠 {ctx.bold('Tool call:')} {name}"


@dataclass
class ToolResultSegment(Segment):
    tool_use_id: str
    name: str | None
    content_text: str
    is_error: bool = False

    def __init__(
        self,
        tool_use_id: str,
        content: Any,
        *,
        name: str | None = None,
        is_error: bool = False,
    ) -> None:
        super().__init__(kind="tool_result")
        self.tool_use_id = str(tool_use_id or "")
        self.name = str(name) if name is not None else None
        self.is_error = bool(is_error)
        if isinstance(content, str):
            self.content_text = content
        else:
            self.content_text = _safe_json_dumps(content)

    def render(self, ctx: RenderCtx) -> str:
        raw = self.content_text or ""
        if ctx.tool_output_tail_max is not None and len(raw) > ctx.tool_output_tail_max:
            raw = "..." + raw[-(ctx.tool_output_tail_max - 3) :]
        inner = ctx.escape_code(raw)
        label = "Tool error:" if self.is_error else "Tool result:"
        maybe_name = f" {ctx.code_inline(self.name)}" if self.name else ""
        return f"📤 {ctx.bold(label)}{maybe_name}\n```\n{inner}\n```"


@dataclass
class SubagentSegment(Segment):
    description: str
    tool_calls: int = 0
    tools_used: set[str] = field(default_factory=set)
    current_tool: ToolCallSegment | None = None

    def __init__(self, description: str) -> None:
        super().__init__(kind="subagent")
        self.description = str(description or "Subagent")
        self.tool_calls = 0
        self.tools_used = set()
        self.current_tool = None

    def set_current_tool_call(self, tool_use_id: str, name: str) -> ToolCallSegment:
        tool_use_id = str(tool_use_id or "")
        name = str(name or "tool")
        self.tools_used.add(name)
        self.tool_calls += 1
        self.current_tool = ToolCallSegment(tool_use_id, name, indent_level=1)
        return self.current_tool

    def render(self, ctx: RenderCtx) -> str:
        inner_prefix = "  "

        lines: list[str] = [
            f"🤖 {ctx.bold('Subagent:')} {ctx.code_inline(self.description)}"
        ]

        if self.current_tool is not None:
            try:
                rendered = self.current_tool.render(ctx)
            except Exception:
                rendered = ""
            if rendered:
                lines.append(rendered)

        tools_used = sorted(self.tools_used)
        tools_set_raw = "{{{}}}".format(", ".join(tools_used)) if tools_used else "{}"

        # Keep braces inside a code entity so MarkdownV2 doesn't require escaping them.
        lines.append(
            f"{inner_prefix}{ctx.bold('Tools used:')} {ctx.code_inline(tools_set_raw)}"
        )
        lines.append(
            f"{inner_prefix}{ctx.bold('Tool calls:')} {ctx.code_inline(str(self.tool_calls))}"
        )
        return "\n".join(lines)


@dataclass
class ErrorSegment(Segment):
    message: str

    def __init__(self, message: str) -> None:
        super().__init__(kind="error")
        self.message = str(message or "Unknown error")

    def render(self, ctx: RenderCtx) -> str:
        return f"⚠️ {ctx.bold('Error:')} {ctx.code_inline(self.message)}"


@dataclass
class RenderCtx:
    bold: Callable[[str], str]
    code_inline: Callable[[str], str]
    escape_code: Callable[[str], str]
    escape_text: Callable[[str], str]
    render_markdown: Callable[[str], str]

    thinking_tail_max: int | None = 1000
    tool_input_tail_max: int | None = 1200
    tool_output_tail_max: int | None = 1600
    text_tail_max: int | None = 2000


class TranscriptBuffer:
    """Maintains an ordered, truncatable transcript of events."""

    def __init__(self, *, show_tool_results: bool = True) -> None:
        self._segments: list[Segment] = []
        self._open_thinking_by_index: dict[int, ThinkingSegment] = {}
        self._open_text_by_index: dict[int, TextSegment] = {}

        # content_block index -> tool call segment (for streaming tool args)
        self._open_tools_by_index: dict[int, ToolCallSegment] = {}

        # tool_use_id -> tool name (for tool_result labeling)
        self._tool_name_by_id: dict[str, str] = {}

        self._show_tool_results = bool(show_tool_results)

        # subagent context stack. Each entry is the Task tool_use_id we are waiting to close.
        self._subagent_stack: list[str] = []
        # Parallel stack of segments for rendering nested subagents.
        self._subagent_segments: list[SubagentSegment] = []
        self._debug_subagent_stack = os.getenv("DEBUG_SUBAGENT_STACK") == "1"

    def _in_subagent(self) -> bool:
        return bool(self._subagent_stack)

    def _subagent_current(self) -> SubagentSegment | None:
        return self._subagent_segments[-1] if self._subagent_segments else None

    def _task_heading_from_input(self, inp: Any) -> str:
        # We never display full JSON args; only extract a short heading.
        if isinstance(inp, dict):
            desc = str(inp.get("description", "") or "").strip()
            if desc:
                return desc
            subagent_type = str(inp.get("subagent_type", "") or "").strip()
            if subagent_type:
                return subagent_type
            typ = str(inp.get("type", "") or "").strip()
            if typ:
                return typ
        return "Subagent"

    def _subagent_push(self, tool_id: str, seg: SubagentSegment) -> None:
        # Some providers can omit ids; still track depth for UI suppression.
        tool_id = (
            str(tool_id or "").strip() or f"__task_{len(self._subagent_stack) + 1}"
        )
        self._subagent_stack.append(tool_id)
        self._subagent_segments.append(seg)
        if self._debug_subagent_stack:
            logger.debug(
                "SUBAGENT_STACK: push id=%r depth=%d heading=%r",
                tool_id,
                len(self._subagent_stack),
                getattr(seg, "description", None),
            )

    def _subagent_pop(self, tool_id: str) -> bool:
        tool_id = str(tool_id or "").strip()
        if not self._subagent_stack:
            return False

        def _ids_roughly_match(stack_id: str, result_id: str) -> bool:
            if not stack_id or not result_id:
                return False
            if stack_id == result_id:
                return True
            # Some providers emit Task result ids with a suffix/prefix variant.
            # Treat those as the same logical Task invocation.
            return stack_id.startswith(result_id) or result_id.startswith(stack_id)

        if tool_id:
            # O(1) common case: LIFO - top of stack matches.
            if _ids_roughly_match(self._subagent_stack[-1], tool_id):
                self._subagent_stack.pop()
                if self._subagent_segments:
                    self._subagent_segments.pop()
                if self._debug_subagent_stack:
                    logger.debug(
                        "SUBAGENT_STACK: pop id=%r depth=%d (LIFO)",
                        tool_id,
                        len(self._subagent_stack),
                    )
                return True
            # Pop to the matching id (defensive against non-LIFO emissions).
            idx = -1
            for i in range(len(self._subagent_stack) - 1, -1, -1):
                if _ids_roughly_match(self._subagent_stack[i], tool_id):
                    idx = i
                    break
            if idx < 0:
                return False
            while len(self._subagent_stack) > idx:
                popped = self._subagent_stack.pop()
                if self._subagent_segments:
                    self._subagent_segments.pop()
                if self._debug_subagent_stack:
                    logger.debug(
                        "SUBAGENT_STACK: pop id=%r depth=%d (matched=%r)",
                        popped,
                        len(self._subagent_stack),
                        tool_id,
                    )
            return True

        # No id in result; only close if we have a synthetic top marker.
        if self._subagent_stack and self._subagent_stack[-1].startswith("__task_"):
            popped = self._subagent_stack.pop()
            if self._subagent_segments:
                self._subagent_segments.pop()
            if self._debug_subagent_stack:
                logger.debug(
                    "SUBAGENT_STACK: pop id=%r depth=%d (synthetic)",
                    popped,
                    len(self._subagent_stack),
                )
            return True
        return False

    def _ensure_thinking(self) -> ThinkingSegment:
        seg = ThinkingSegment()
        self._segments.append(seg)
        return seg

    def _ensure_text(self) -> TextSegment:
        seg = TextSegment()
        self._segments.append(seg)
        return seg

    def apply(self, ev: dict[str, Any]) -> None:
        """Apply a parsed event to the transcript."""
        et = ev.get("type")

        # Subagent rules: inside a Task/subagent, we only show tool calls/results.
        if self._in_subagent() and et in (
            "thinking_start",
            "thinking_delta",
            "thinking_chunk",
            "text_start",
            "text_delta",
            "text_chunk",
        ):
            return

        if et == "thinking_start":
            idx = int(ev.get("index", -1))
            if idx >= 0:
                # Defensive: if a provider reuses indices without emitting a stop,
                # close the previous open segment first.
                self.apply({"type": "block_stop", "index": idx})
            seg = self._ensure_thinking()
            if idx >= 0:
                self._open_thinking_by_index[idx] = seg
            return
        if et in ("thinking_delta", "thinking_chunk"):
            idx = int(ev.get("index", -1))
            seg = self._open_thinking_by_index.get(idx)
            if seg is None:
                seg = self._ensure_thinking()
                if idx >= 0:
                    self._open_thinking_by_index[idx] = seg
            seg.append(str(ev.get("text", "")))
            return
        if et == "thinking_stop":
            idx = int(ev.get("index", -1))
            if idx >= 0:
                self._open_thinking_by_index.pop(idx, None)
            return

        if et == "text_start":
            idx = int(ev.get("index", -1))
            if idx >= 0:
                self.apply({"type": "block_stop", "index": idx})
            seg = self._ensure_text()
            if idx >= 0:
                self._open_text_by_index[idx] = seg
            return
        if et in ("text_delta", "text_chunk"):
            idx = int(ev.get("index", -1))
            seg = self._open_text_by_index.get(idx)
            if seg is None:
                seg = self._ensure_text()
                if idx >= 0:
                    self._open_text_by_index[idx] = seg
            seg.append(str(ev.get("text", "")))
            return
        if et == "text_stop":
            idx = int(ev.get("index", -1))
            if idx >= 0:
                self._open_text_by_index.pop(idx, None)
            return

        if et == "tool_use_start":
            idx = int(ev.get("index", -1))
            if idx >= 0:
                self.apply({"type": "block_stop", "index": idx})
            tool_id = str(ev.get("id", "") or "").strip()
            name = str(ev.get("name", "") or "tool")
            if tool_id:
                self._tool_name_by_id[tool_id] = name

            # Task tool indicates subagent.
            if name == "Task":
                heading = self._task_heading_from_input(ev.get("input"))
                seg = SubagentSegment(heading)
                self._segments.append(seg)
                self._subagent_push(tool_id, seg)
                return

            # Normal tool call.
            if self._in_subagent():
                parent = self._subagent_current()
                if parent is not None:
                    seg = parent.set_current_tool_call(tool_id, name)
                else:
                    seg = ToolCallSegment(tool_id, name)
                    self._segments.append(seg)
            else:
                seg = ToolCallSegment(tool_id, name)
                self._segments.append(seg)

            if idx >= 0:
                self._open_tools_by_index[idx] = seg
            return

        if et == "tool_use_delta":
            # Track open tool by index for tool_use_stop (closing state).
            return

        if et == "tool_use_stop":
            idx = int(ev.get("index", -1))
            seg = self._open_tools_by_index.pop(idx, None)
            if seg is not None:
                seg.closed = True
            return

        if et == "block_stop":
            idx = int(ev.get("index", -1))
            if idx in self._open_tools_by_index:
                self.apply({"type": "tool_use_stop", "index": idx})
                return
            if idx in self._open_thinking_by_index:
                self.apply({"type": "thinking_stop", "index": idx})
                return
            if idx in self._open_text_by_index:
                self.apply({"type": "text_stop", "index": idx})
                return
            return

        if et == "tool_use":
            tool_id = str(ev.get("id", "") or "").strip()
            name = str(ev.get("name", "") or "tool")
            if tool_id:
                self._tool_name_by_id[tool_id] = name

            if name == "Task":
                heading = self._task_heading_from_input(ev.get("input"))
                seg = SubagentSegment(heading)
                self._segments.append(seg)
                self._subagent_push(tool_id, seg)
                return

            if self._in_subagent():
                parent = self._subagent_current()
                if parent is not None:
                    seg = parent.set_current_tool_call(tool_id, name)
                else:
                    seg = ToolCallSegment(tool_id, name)
                    self._segments.append(seg)
            else:
                seg = ToolCallSegment(tool_id, name)
                self._segments.append(seg)

            seg.closed = True
            return

        if et == "tool_result":
            tool_id = str(ev.get("tool_use_id", "") or "").strip()
            name = self._tool_name_by_id.get(tool_id)

            # If this was the Task tool result, close subagent context.
            if self._subagent_stack:
                popped = self._subagent_pop(tool_id)
                top = self._subagent_stack[-1] if self._subagent_stack else ""
                looks_like_task_id = "task" in tool_id.lower()
                # Some streams omit Task tool_use ids (synthetic stack ids), but include
                # a real Task id on tool_result (e.g. "functions.Task:0"). Reconcile that.
                if (
                    not popped
                    and tool_id
                    and top.startswith("__task_")
                    and (name in (None, "Task"))
                    and looks_like_task_id
                ):
                    self._subagent_pop("")

            if not self._show_tool_results:
                return

            seg = ToolResultSegment(
                tool_id,
                ev.get("content"),
                name=name,
                is_error=bool(ev.get("is_error", False)),
            )
            self._segments.append(seg)
            return

        if et == "error":
            self._segments.append(ErrorSegment(str(ev.get("message", ""))))
            return

    def render(self, ctx: RenderCtx, *, limit_chars: int, status: str | None) -> str:
        """Render transcript with truncation (drop oldest segments)."""
        # Filter out empty rendered segments.
        rendered: list[str] = []
        for seg in self._segments:
            try:
                out = seg.render(ctx)
            except Exception:
                continue
            if out:
                rendered.append(out)

        status_text = f"\n\n{status}" if status else ""
        prefix_marker = ctx.escape_text("... (truncated)\n")

        def _join(parts: Iterable[str], add_marker: bool) -> str:
            body = "\n".join(parts)
            if add_marker and body:
                body = prefix_marker + body
            return body + status_text if (body or status_text) else status_text

        # Fast path.
        candidate = _join(rendered, add_marker=False)
        if len(candidate) <= limit_chars:
            return candidate

        # Drop oldest segments until under limit (keep the tail).
        # Use deque for O(1) popleft; list.pop(0) would be O(n) per iteration.
        parts: deque[str] = deque(rendered)
        dropped = False
        last_part: str | None = None
        while parts:
            candidate = _join(parts, add_marker=True)
            if len(candidate) <= limit_chars:
                return candidate
            last_part = parts.popleft()
            dropped = True

        # Nothing fits - preserve tail of last segment instead of only marker+status.
        if dropped and last_part:
            budget = limit_chars - len(prefix_marker) - len(status_text)
            if budget > 20:
                if len(last_part) > budget:
                    tail = "..." + last_part[-(budget - 3) :]
                else:
                    tail = last_part
                candidate = prefix_marker + tail + status_text
                if len(candidate) <= limit_chars:
                    return candidate

        # Fallback: marker + status only.
        if dropped:
            minimal = prefix_marker + status_text.lstrip("\n")
            if len(minimal) <= limit_chars:
                return minimal
        return status or ""

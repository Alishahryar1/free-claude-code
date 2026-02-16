"""Ordered transcript builder for messaging UIs (Telegram, etc.).

This module maintains an ordered list of "segments" that represent what the user
should see in the chat transcript: thinking, tool calls, tool results, subagent
headers, and assistant text. It is designed for in-place message editing where
the transcript grows over time and older content must be truncated.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional
from loguru import logger


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


@dataclass
class Segment:
    kind: str

    def render(self, ctx: "RenderCtx") -> str:
        raise NotImplementedError


@dataclass
class ThinkingSegment(Segment):
    text: str = ""

    def __init__(self) -> None:
        super().__init__(kind="thinking")

    def append(self, t: str) -> None:
        if t:
            self.text += t

    def render(self, ctx: "RenderCtx") -> str:
        raw = self.text or ""
        if ctx.thinking_tail_max is not None and len(raw) > ctx.thinking_tail_max:
            raw = "..." + raw[-(ctx.thinking_tail_max - 3) :]
        inner = ctx.escape_code(raw)
        return f"💭 {ctx.bold('Thinking')}\n```\n{inner}\n```"


@dataclass
class TextSegment(Segment):
    text: str = ""

    def __init__(self) -> None:
        super().__init__(kind="text")

    def append(self, t: str) -> None:
        if t:
            self.text += t

    def render(self, ctx: "RenderCtx") -> str:
        raw = self.text or ""
        if ctx.text_tail_max is not None and len(raw) > ctx.text_tail_max:
            raw = "..." + raw[-(ctx.text_tail_max - 3) :]
        return ctx.render_markdown(raw)


@dataclass
class ToolCallSegment(Segment):
    tool_use_id: str
    name: str
    input_text: str = ""
    closed: bool = False
    indent_level: int = 0

    def __init__(self, tool_use_id: str, name: str, *, indent_level: int = 0) -> None:
        super().__init__(kind="tool_call")
        self.tool_use_id = str(tool_use_id or "")
        self.name = str(name or "tool")
        self.indent_level = max(0, int(indent_level))

    def set_initial_input(self, inp: Any) -> None:
        if inp is None:
            return
        if isinstance(inp, str):
            self.input_text = inp
        else:
            self.input_text = _safe_json_dumps(inp)

    def append_input_delta(self, partial: str) -> None:
        if partial:
            self.input_text += partial

    def render(self, ctx: "RenderCtx") -> str:
        name = ctx.code_inline(self.name)
        # Per UX requirement: do not display tool args/results, only the tool call.
        prefix = "  " * self.indent_level
        return f"{prefix}🛠 {ctx.bold('Tool call:')} {name}"


@dataclass
class ToolResultSegment(Segment):
    tool_use_id: str
    name: Optional[str]
    content_text: str
    is_error: bool = False

    def __init__(
        self,
        tool_use_id: str,
        content: Any,
        *,
        name: Optional[str] = None,
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

    def render(self, ctx: "RenderCtx") -> str:
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
    indent_level: int = 0
    tool_calls: int = 0
    tools_used: set[str] = field(default_factory=set)
    current_tool: Optional[ToolCallSegment] = None
    subagents: List["SubagentSegment"] = field(default_factory=list)

    def __init__(self, description: str, *, indent_level: int = 0) -> None:
        super().__init__(kind="subagent")
        self.description = str(description or "Subagent")
        self.indent_level = max(0, int(indent_level))
        self.tool_calls = 0
        self.tools_used = set()
        self.current_tool = None
        self.subagents = []

    def set_current_tool_call(self, tool_use_id: str, name: str) -> ToolCallSegment:
        tool_use_id = str(tool_use_id or "")
        name = str(name or "tool")
        self.tools_used.add(name)
        self.tool_calls += 1
        seg = ToolCallSegment(tool_use_id, name, indent_level=self.indent_level + 1)
        self.current_tool = seg
        return seg

    def add_subagent(self, seg: "SubagentSegment") -> None:
        # Nesting just adds more indent: the nested SubagentSegment carries its own indent.
        self.subagents.append(seg)

    def render(self, ctx: "RenderCtx") -> str:
        prefix = "  " * self.indent_level
        inner_prefix = "  " * (self.indent_level + 1)

        lines: List[str] = [
            f"{prefix}🤖 {ctx.bold('Subagent:')} {ctx.code_inline(self.description)}"
        ]

        if self.current_tool is not None:
            try:
                rendered = self.current_tool.render(ctx)
            except Exception:
                rendered = ""
            if rendered:
                lines.append(rendered)

        for sub in self.subagents:
            try:
                rendered = sub.render(ctx)
            except Exception:
                continue
            if rendered:
                lines.append(rendered)

        tools_used = sorted(self.tools_used)
        if tools_used:
            tools_set_raw = "{%s}" % (", ".join(tools_used))
        else:
            tools_set_raw = "{}"

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

    def render(self, ctx: "RenderCtx") -> str:
        return f"⚠️ {ctx.bold('Error:')} {ctx.code_inline(self.message)}"


@dataclass
class RenderCtx:
    bold: Callable[[str], str]
    code_inline: Callable[[str], str]
    escape_code: Callable[[str], str]
    escape_text: Callable[[str], str]
    render_markdown: Callable[[str], str]

    thinking_tail_max: Optional[int] = 1000
    tool_input_tail_max: Optional[int] = 1200
    tool_output_tail_max: Optional[int] = 1600
    text_tail_max: Optional[int] = 2000


class TranscriptBuffer:
    """Maintains an ordered, truncatable transcript of events."""

    def __init__(self, *, show_tool_results: bool = True) -> None:
        self._segments: List[Segment] = []
        self._open_thinking_by_index: Dict[int, ThinkingSegment] = {}
        self._open_text_by_index: Dict[int, TextSegment] = {}

        # content_block index -> tool call segment (for streaming tool args)
        self._open_tools_by_index: Dict[int, ToolCallSegment] = {}

        # tool_use_id -> tool name (for tool_result labeling)
        self._tool_name_by_id: Dict[str, str] = {}

        self._show_tool_results = bool(show_tool_results)

        # subagent context stack. Each entry is the Task tool_use_id we are waiting to close.
        self._subagent_stack: List[str] = []
        # Parallel stack of segments for rendering nested subagents.
        self._subagent_segments: List[SubagentSegment] = []
        self._debug_subagent_stack = os.getenv("DEBUG_SUBAGENT_STACK") == "1"

    def _in_subagent(self) -> bool:
        return bool(self._subagent_stack)

    def _subagent_depth(self) -> int:
        return len(self._subagent_stack)

    def _subagent_current(self) -> Optional[SubagentSegment]:
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

    def _subagent_pop(self, tool_id: str) -> None:
        tool_id = str(tool_id or "").strip()
        if not self._subagent_stack:
            return
        if tool_id:
            # O(1) common case: LIFO - top of stack matches.
            if self._subagent_stack[-1] == tool_id:
                self._subagent_stack.pop()
                if self._subagent_segments:
                    self._subagent_segments.pop()
                if self._debug_subagent_stack:
                    logger.debug(
                        "SUBAGENT_STACK: pop id=%r depth=%d (LIFO)",
                        tool_id,
                        len(self._subagent_stack),
                    )
                return
            # Pop to the matching id (defensive against non-LIFO emissions).
            try:
                idx = (
                    len(self._subagent_stack)
                    - 1
                    - self._subagent_stack[::-1].index(tool_id)
                )
            except ValueError:
                return
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
            return

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

    def _ensure_thinking(self) -> ThinkingSegment:
        seg = ThinkingSegment()
        self._segments.append(seg)
        return seg

    def _ensure_text(self) -> TextSegment:
        seg = TextSegment()
        self._segments.append(seg)
        return seg

    def apply(self, ev: Dict[str, Any]) -> None:
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
                seg = SubagentSegment(heading, indent_level=self._subagent_depth())
                parent = self._subagent_current()
                if parent is not None:
                    parent.add_subagent(seg)
                else:
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

            seg.set_initial_input(ev.get("input"))
            if idx >= 0:
                self._open_tools_by_index[idx] = seg
            return

        if et == "tool_use_delta":
            idx = int(ev.get("index", -1))
            partial = str(ev.get("partial_json", "") or "")
            seg = self._open_tools_by_index.get(idx)
            if seg is not None:
                seg.append_input_delta(partial)
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
                seg = SubagentSegment(heading, indent_level=self._subagent_depth())
                parent = self._subagent_current()
                if parent is not None:
                    parent.add_subagent(seg)
                else:
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

            seg.set_initial_input(ev.get("input"))
            seg.closed = True
            return

        if et == "tool_result":
            tool_id = str(ev.get("tool_use_id", "") or "").strip()
            name = self._tool_name_by_id.get(tool_id)

            # If this was the Task tool result, close subagent context.
            if self._subagent_stack and (
                not tool_id or self._subagent_stack[-1] == tool_id
            ):
                self._subagent_pop(tool_id)

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

    def render(self, ctx: RenderCtx, *, limit_chars: int, status: Optional[str]) -> str:
        """Render transcript with truncation (drop oldest segments)."""
        # Filter out empty rendered segments.
        rendered: List[str] = []
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
        while parts:
            candidate = _join(parts, add_marker=True)
            if len(candidate) <= limit_chars:
                return candidate
            parts.popleft()
            dropped = True

        # Nothing fits; return status only with marker if possible.
        if dropped:
            minimal = prefix_marker + status_text.lstrip("\n")
            if len(minimal) <= limit_chars:
                return minimal
        return status or ""

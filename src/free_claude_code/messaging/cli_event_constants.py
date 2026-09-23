"""CLI event types and status-line mapping for transcript / UI updates."""

from collections.abc import Callable
from typing import Any

# Status message prefixes used to filter our own messages (ignore echo)
STATUS_MESSAGE_PREFIXES = (
    "⏳",
    "💭",
    "🔧",
    "✅",
    "❌",
    "🚀",
    "🤖",
    "📋",
    "📊",
    "🔄",
)

# Event types that update the transcript (frozenset for O(1) membership)
TRANSCRIPT_EVENT_TYPES = frozenset(
    {
        "thinking_start",
        "thinking_delta",
        "thinking_chunk",
        "thinking_stop",
        "text_start",
        "text_delta",
        "text_chunk",
        "text_stop",
        "tool_use_start",
        "tool_use_delta",
        "tool_use_stop",
        "tool_use",
        "tool_result",
        "block_stop",
        "error",
    }
)

# Event type -> (emoji, label) for status updates (O(1) lookup)
_EVENT_STATUS_MAP: dict[str, tuple[str, str]] = {
    "thinking_start": ("🧠", "Claude is thinking..."),
    "thinking_delta": ("🧠", "Claude is thinking..."),
    "thinking_chunk": ("🧠", "Claude is thinking..."),
    "text_start": ("🧠", "Claude is working..."),
    "text_delta": ("🧠", "Claude is working..."),
    "text_chunk": ("🧠", "Claude is working..."),
    "tool_result": ("⏳", "Executing tools..."),
}


def get_status_for_event(
    ptype: str,
    parsed: dict[str, Any],
    format_status_fn: Callable[..., str],
) -> str | None:
    """Return status string for event type, or None if no status update needed."""
    entry = _EVENT_STATUS_MAP.get(ptype)
    if entry is not None:
        emoji, label = entry
        return format_status_fn(emoji, label)
    if ptype in ("tool_use_start", "tool_use_delta", "tool_use"):
        if parsed.get("name") == "Task":
            return format_status_fn("🤖", "Subagent working...")
        return format_status_fn("⏳", "Executing tools...")
    return None

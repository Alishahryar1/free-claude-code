"""Trace-safe snapshots of Anthropic protocol requests."""

from typing import Any

from free_claude_code.core.trace import sanitize_trace_value

from .models import MessagesRequest, TokenCountRequest


def anthropic_request_snapshot(
    request: MessagesRequest | TokenCountRequest,
) -> dict[str, Any]:
    """Return the traceable public fields of an Anthropic request."""
    data = request.model_dump(mode="python")
    snapshot = {
        key: data[key]
        for key in (
            "model",
            "messages",
            "system",
            "tools",
            "tool_choice",
            "max_tokens",
            "thinking",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "metadata",
            "stream",
        )
        if key in data and data[key] is not None
    }
    sanitized = sanitize_trace_value(snapshot)
    return sanitized if isinstance(sanitized, dict) else {}

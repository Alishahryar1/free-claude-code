"""Anthropic SSE serialization helpers."""

import json
from typing import Any

from loguru import logger

from free_claude_code.core.failures import ExecutionFailure

from ..errors import anthropic_error_payload, anthropic_failure_payload

ANTHROPIC_SSE_RESPONSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def map_stop_reason(openai_reason: str | None) -> str:
    """Map OpenAI ``finish_reason`` values to Anthropic ``stop_reason`` values."""
    return (
        STOP_REASON_MAP.get(openai_reason, "end_turn") if openai_reason else "end_turn"
    )


def format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format one Anthropic-style SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def anthropic_terminal_error_frame(
    message: str, *, error_type: str = "api_error"
) -> str:
    """Serialize a terminal Anthropic SSE error event for egress failures."""
    return format_sse_event(
        "error", anthropic_error_payload(error_type=error_type, message=message)
    )


def anthropic_terminal_failure_frame(failure: ExecutionFailure) -> str:
    """Serialize a canonical execution failure as a terminal SSE event."""
    return format_sse_event("error", anthropic_failure_payload(failure))


class AnthropicSseEmitter:
    """Serialize Anthropic SSE events and optionally log raw event bodies."""

    def __init__(self, *, log_raw_events: bool = False) -> None:
        self._log_raw_events = log_raw_events

    def event(self, event_type: str, data: dict[str, Any]) -> str:
        event = format_sse_event(event_type, data)
        if self._log_raw_events:
            logger.debug("SSE_EVENT: {} - {}", event_type, event.strip())
        return event

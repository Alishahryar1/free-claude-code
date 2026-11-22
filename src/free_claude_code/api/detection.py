"""Detect known Claude auto-mode classifier requests."""

from free_claude_code.core.anthropic import (
    MessagesRequest,
    extract_text_from_content,
)


def _request_system_text(request_data: MessagesRequest) -> str:
    """Return the top-level system text for request-shape detection."""
    if not request_data.system:
        return ""
    return extract_text_from_content(request_data.system)


def detect_safety_classifier_stop_sequence(
    request_data: MessagesRequest,
) -> str | None:
    """Return the stop hint owned by a known Claude auto-mode classifier."""
    if request_data.tools:
        return None

    messages_text = "\n".join(
        extract_text_from_content(message.content) for message in request_data.messages
    )
    if "<transcript>" not in messages_text:
        return None

    system_text = _request_system_text(request_data)
    if "Output <severity>N</severity>" in system_text:
        return "</severity>"
    if "yes</block>" in system_text or "no</block>" in system_text:
        return "</block>"
    return None

"""Google thought-signature replay for OpenAI-compatible tool calls."""

from copy import deepcopy
from typing import Any

GOOGLE_SKIP_THOUGHT_SIGNATURE_VALIDATOR = "skip_thought_signature_validator"


def apply_google_thought_signatures(
    body: dict[str, Any],
    *,
    tool_call_extra_content_by_id: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Restore Google tool-call signatures required for conversation replay."""

    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    _apply_cached_tool_call_signatures(messages, tool_call_extra_content_by_id or {})
    _apply_missing_current_turn_signatures(messages)


def _thought_signature_from_extra_content(extra_content: Any) -> str | None:
    if not isinstance(extra_content, dict):
        return None
    google = extra_content.get("google")
    if not isinstance(google, dict):
        return None
    signature = google.get("thought_signature")
    return signature if isinstance(signature, str) and signature else None


def _tool_call_thought_signature(tool_call: dict[str, Any]) -> str | None:
    return _thought_signature_from_extra_content(tool_call.get("extra_content"))


def _set_tool_call_thought_signature(tool_call: dict[str, Any], signature: str) -> None:
    extra_content = tool_call.get("extra_content")
    if not isinstance(extra_content, dict):
        extra_content = {}
        tool_call["extra_content"] = extra_content
    google = extra_content.get("google")
    if not isinstance(google, dict):
        google = {}
        extra_content["google"] = google
    google["thought_signature"] = signature


def _message_has_standard_user_content(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and bool(part["text"].strip())
            for part in content
        )
    return False


def _current_turn_start_index(messages: list[Any]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and _message_has_standard_user_content(message):
            return index
    return -1


def _apply_cached_tool_call_signatures(
    messages: list[Any], tool_call_extra_content_by_id: dict[str, dict[str, Any]]
) -> None:
    if not tool_call_extra_content_by_id:
        return
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or _tool_call_thought_signature(
                tool_call
            ):
                continue
            tool_call_id = tool_call.get("id")
            if tool_call_id is None:
                continue
            cached_extra_content = tool_call_extra_content_by_id.get(str(tool_call_id))
            if not cached_extra_content:
                continue
            cached_signature = _thought_signature_from_extra_content(
                cached_extra_content
            )
            if cached_signature:
                tool_call["extra_content"] = deepcopy(cached_extra_content)


def _apply_missing_current_turn_signatures(messages: list[Any]) -> None:
    start_index = _current_turn_start_index(messages)
    for message in messages[start_index + 1 :]:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        first_tool_call = tool_calls[0]
        if not isinstance(first_tool_call, dict):
            continue
        if _tool_call_thought_signature(first_tool_call):
            continue
        _set_tool_call_thought_signature(
            first_tool_call, GOOGLE_SKIP_THOUGHT_SIGNATURE_VALIDATOR
        )

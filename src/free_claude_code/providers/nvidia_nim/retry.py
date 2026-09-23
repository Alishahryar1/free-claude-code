"""NVIDIA NIM retry-body downgrade helpers."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from free_claude_code.core.history_replay import reasoning_context


def clone_body_without_reasoning_budget(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip only reasoning_budget fields."""
    return _clone_strip_extra_body(body, _strip_reasoning_budget_fields)


def clone_body_without_chat_template(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip NIM chat-template control fields."""
    return _clone_strip_extra_body(body, _strip_chat_template_fields)


def clone_body_without_reasoning_content(
    body: dict[str, Any],
) -> dict[str, Any] | None:
    """Move rejected native reasoning fields into readable assistant context."""
    cloned_body = deepcopy(body)
    if not _strip_message_reasoning_content(cloned_body):
        return None
    return cloned_body


def _clone_strip_extra_body(
    body: dict[str, Any],
    strip: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    cloned_body = deepcopy(body)
    extra_body = cloned_body.get("extra_body")
    if not isinstance(extra_body, dict):
        return None
    if not strip(extra_body):
        return None
    if not extra_body:
        cloned_body.pop("extra_body", None)
    return cloned_body


def _strip_reasoning_budget_fields(extra_body: dict[str, Any]) -> bool:
    removed = extra_body.pop("reasoning_budget", None) is not None
    chat_template_kwargs = extra_body.get("chat_template_kwargs")
    if (
        isinstance(chat_template_kwargs, dict)
        and chat_template_kwargs.pop("reasoning_budget", None) is not None
    ):
        removed = True
    return removed


def _strip_chat_template_fields(extra_body: dict[str, Any]) -> bool:
    removed = extra_body.pop("chat_template", None) is not None
    if extra_body.pop("chat_template_kwargs", None) is not None:
        removed = True
    return removed


def _strip_message_reasoning_content(body: dict[str, Any]) -> bool:
    removed = False
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        reasoning = message.pop("reasoning_content", None)
        if reasoning is not None:
            removed = True
        if (
            isinstance(reasoning, str)
            and reasoning
            and message.get("role") == "assistant"
        ):
            context = reasoning_context(reasoning)
            content = message.get("content")
            if isinstance(content, list):
                content.append({"type": "text", "text": context})
            else:
                message["content"] = f"{content}\n\n{context}" if content else context
    return removed

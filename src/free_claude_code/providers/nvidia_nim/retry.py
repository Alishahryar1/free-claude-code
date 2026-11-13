"""NVIDIA NIM retry-body downgrade helpers."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any


def clone_body_without_chat_template(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip NIM chat-template control fields."""
    return _clone_strip_extra_body(body, _strip_chat_template_fields)


def clone_body_without_reasoning_effort(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip NIM reasoning_effort."""
    removed = False
    cloned_body = deepcopy(body)
    if cloned_body.pop("reasoning_effort", None) is not None:
        removed = True
    extra_body = cloned_body.get("extra_body")
    if isinstance(extra_body, dict):
        if extra_body.pop("reasoning_effort", None) is not None:
            removed = True
        if not extra_body:
            cloned_body.pop("extra_body", None)
    return cloned_body if removed else None


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


def _strip_chat_template_fields(extra_body: dict[str, Any]) -> bool:
    removed = extra_body.pop("chat_template", None) is not None
    if extra_body.pop("chat_template_kwargs", None) is not None:
        removed = True
    return removed

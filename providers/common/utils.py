"""Shared utility helpers for provider request builders."""

from typing import Any


def set_if_not_none(body: dict[str, Any], key: str, value: Any) -> None:
    """Set body[key] = value only when value is not None."""
    if value is not None:
        body[key] = value

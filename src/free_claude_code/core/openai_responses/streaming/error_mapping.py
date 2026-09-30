"""Responses stream error mapping."""

from collections.abc import Mapping
from typing import Any


def openai_error_from_anthropic_error(data: Mapping[str, Any]) -> dict[str, Any]:
    error = data.get("error")
    if not isinstance(error, dict):
        error = {"type": "api_error", "message": str(data)}
    return {
        "message": str(error.get("message", "")),
        "type": str(error.get("type", "api_error")),
        "param": None,
        "code": None,
    }

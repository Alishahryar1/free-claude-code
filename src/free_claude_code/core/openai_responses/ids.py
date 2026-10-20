"""Identifier helpers for OpenAI Responses payloads."""

import uuid
from typing import Literal


def tool_item_id_prefix(kind: Literal["function", "custom"]) -> str:
    return "ctc_" if kind == "custom" else "fc_"


def tool_item_id_for_kind(item_id: str, *, kind: Literal["function", "custom"]) -> str:
    """Retag only a known opposite prefix, preserving the opaque suffix."""
    opposite = tool_item_id_prefix("function" if kind == "custom" else "custom")
    if item_id.startswith(opposite):
        return tool_item_id_prefix(kind) + item_id[len(opposite) :]
    return item_id


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def new_message_item_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def new_reasoning_item_id() -> str:
    return f"rs_{uuid.uuid4().hex}"


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"

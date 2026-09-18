"""Identifier helpers for OpenAI Responses payloads."""

import uuid


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def new_message_item_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def new_reasoning_item_id() -> str:
    return f"rs_{uuid.uuid4().hex}"


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def new_tool_item_id(kind: str = "function") -> str:
    """Return an item ID with the correct prefix for the tool kind.

    OpenAI requires ``fc_`` for function_call items and ``ctc_`` for
    custom_tool_call items.
    """
    prefix = "ctc" if kind == "custom" else "fc"
    return f"{prefix}_{uuid.uuid4().hex[:24]}"

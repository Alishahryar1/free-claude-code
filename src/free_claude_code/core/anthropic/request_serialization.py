"""Shared Anthropic request serialization helpers."""

import json
from typing import Any

from .models import MessagesRequest

_MESSAGES_REQUEST_FIELDS = (
    "model",
    "messages",
    "system",
    "max_tokens",
    "stop_sequences",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "metadata",
    "tools",
    "tool_choice",
    "thinking",
    "context_management",
    "output_config",
    "mcp_servers",
    "extra_body",
)


def dump_messages_request(request: MessagesRequest) -> dict[str, Any]:
    """Return JSON-ready public Messages fields without FCC routing state."""
    raw = request.model_dump(exclude_none=True)
    return {
        field: raw[field]
        for field in _MESSAGES_REQUEST_FIELDS
        if field in raw and raw[field] is not None
    }


def serialize_tool_result_content(content: Any) -> str:
    """Serialize Anthropic ``tool_result.content`` into provider-safe text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def is_noop_context_management(
    context_management: dict[str, Any] | None,
) -> bool:
    if not context_management:
        return True
    if set(context_management) != {"edits"}:
        return False
    edits = context_management["edits"]
    return isinstance(edits, list) and all(
        isinstance(edit, dict)
        and edit
        == {
            "type": "clear_thinking_20251015",
            "keep": "all",
        }
        for edit in edits
    )

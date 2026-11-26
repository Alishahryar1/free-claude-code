"""Shared Anthropic request serialization helpers."""

import json
from collections.abc import Mapping

from ..json_types import JsonObject
from .models import (
    ContentBlockDocument,
    ContentBlockToolResult,
    ExtensionBlock,
    Message,
    MessagesRequest,
)

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


def dump_messages_request(request: MessagesRequest) -> JsonObject:
    """Return JSON-ready public Messages fields without FCC routing state."""
    raw: JsonObject = request.model_dump(exclude_none=True)
    return {
        field: raw[field]
        for field in _MESSAGES_REQUEST_FIELDS
        if field in raw and raw[field] is not None
    }


def translated_messages(request: MessagesRequest) -> list[Message]:
    """Keep usable source turns before a foreign builder coalesces their roles."""
    latest = next(
        (turn for turn in reversed(request.messages) if turn.role == "user"), None
    )
    messages = []
    for turn in request.messages:
        if not isinstance(turn.content, list):
            messages.append(turn)
            continue
        blocks = []
        for block in turn.content:
            if isinstance(block, (ExtensionBlock, ContentBlockDocument)):
                continue
            if isinstance(block, ContentBlockToolResult) and isinstance(
                block.content, list
            ):
                items = [
                    item
                    for item in block.content
                    if not isinstance(item, dict)
                    or item.get("type") in (None, "text", "image")
                ]
                block = block.model_copy(update={"content": items})
            blocks.append(block)
        if not blocks and not turn.reasoning_content:
            if turn is latest:
                raise ValueError("The latest user message has no translatable content")
            continue
        messages.append(turn.model_copy(update={"content": blocks}))
    if not messages:
        raise ValueError("Messages request has no translatable conversation")
    return messages


def openai_output_schema(request: MessagesRequest) -> JsonObject | None:
    """Map Messages' required JSON output schema to an OpenAI schema declaration."""
    output: Mapping[str, object] = request.output_config or {}
    value = output.get("format")
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("type") != "json_schema"
        or not isinstance(schema := value.get("schema"), dict)
    ):
        raise ValueError("output_config.format requires a JSON schema object")
    return {"name": "response", "schema": schema, "strict": True}


def serialize_tool_result_content(content: object) -> str:
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
    context_management: Mapping[str, object] | None,
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

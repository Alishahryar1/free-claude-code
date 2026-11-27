"""Shared Anthropic request serialization helpers."""

import json
from collections.abc import Mapping

from ..history_replay import HistoryReplayError, tool_history_context
from ..json_types import JsonObject
from .models import (
    ContentBlockDocument,
    ContentBlockServerToolUse,
    ContentBlockText,
    ContentBlockToolResult,
    ExtensionBlock,
    Message,
    MessageContentBlock,
    MessagesRequest,
)

_CODE_TOOLS = {"code_execution", "bash_code_execution", "text_editor_code_execution"}
_CODE_RESULTS = {f"{name}_tool_result": name for name in _CODE_TOOLS}

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


def quote_code_execution_history(messages: list[Message]) -> list[Message]:
    """Carry completed server code exchanges across turns in their original order."""
    calls: dict[str, str | None] = {}
    projected = messages.copy()
    for turn_index, message in enumerate(messages):
        if not isinstance(message.content, list):
            continue
        blocks: list[MessageContentBlock] | None = None
        for block_index, block in enumerate(message.content):
            if (
                isinstance(block, ContentBlockServerToolUse)
                and block.name in _CODE_TOOLS
            ):
                if message.role != "assistant" or not block.id or block.id in calls:
                    raise HistoryReplayError(
                        "Invalid code execution call in Messages history"
                    )
                calls[block.id] = block.name
            elif isinstance(block, ExtensionBlock) and block.type in _CODE_RESULTS:
                data = block.model_extra or {}
                call_id = data.get("tool_use_id")
                if (
                    message.role != "assistant"
                    or not isinstance(call_id, str)
                    or calls.get(call_id) != _CODE_RESULTS[block.type]
                    or not isinstance(data.get("content"), dict)
                ):
                    raise HistoryReplayError(
                        "Unmatched code execution result in Messages history"
                    )
                calls[call_id] = None
            else:
                continue
            if blocks is None:
                blocks = message.content.copy()
            blocks[block_index] = ContentBlockText(
                type="text",
                text=tool_history_context(block.model_dump(exclude_unset=True)),
            )
        if blocks is not None:
            projected[turn_index] = message.model_copy(update={"content": blocks})
    if any(calls.values()):
        raise HistoryReplayError(
            "An active server tool cannot continue through a different protocol."
        )
    return projected


def translated_messages(request: MessagesRequest) -> list[Message]:
    """Keep usable source turns before a foreign builder coalesces their roles."""
    if request.uses_container:
        raise ValueError("A Messages container requires native Messages egress")
    latest = next(
        (turn for turn in reversed(request.messages) if turn.role == "user"), None
    )
    messages = []
    for turn in quote_code_execution_history(request.messages):
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

"""Shared Anthropic request serialization helpers."""

import json
from collections.abc import Mapping

from ..history_replay import HistoryReplayError, tool_history_context
from ..json_types import JsonObject, JsonValue
from .models import (
    CODE_EXECUTION_INPUT_FIELDS,
    ContentBlockDocument,
    ContentBlockImage,
    ContentBlockRedactedThinking,
    ContentBlockServerToolUse,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ExtensionBlock,
    Message,
    MessageContentBlock,
    MessagesRequest,
)

_CODE_RESULTS = {f"{name}_tool_result": name for name in CODE_EXECUTION_INPUT_FIELDS}

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
                and block.name in CODE_EXECUTION_INPUT_FIELDS
            ):
                if message.role != "assistant" or not block.id or block.id in calls:
                    raise HistoryReplayError(
                        "Invalid code execution call in Messages history"
                    )
                calls[block.id] = block.name
                field = CODE_EXECUTION_INPUT_FIELDS[block.name]
                public: JsonObject = {
                    "name": block.name,
                    "id": block.id,
                    field: block.input.get(field),
                }
            elif isinstance(block, ExtensionBlock) and block.type in _CODE_RESULTS:
                data = block.model_extra or {}
                call_id = data.get("tool_use_id")
                if (
                    message.role != "assistant"
                    or not isinstance(call_id, str)
                    or calls.get(call_id) != _CODE_RESULTS[block.type]
                    or not isinstance(result := data.get("content"), dict)
                ):
                    raise HistoryReplayError(
                        "Unmatched code execution result in Messages history"
                    )
                calls[call_id] = None
                name = _CODE_RESULTS[block.type]
                if result.get("type") not in {
                    f"{name}_result",
                    f"{name}_tool_result_error",
                } or result.get("content"):
                    raise HistoryReplayError(
                        "This code execution result requires native Messages egress"
                    )
                public = {
                    "type": block.type,
                    "tool_use_id": call_id,
                    **{
                        field: result[field]
                        for field in ("stdout", "stderr", "return_code", "error_code")
                        if field in result
                    },
                }
            else:
                continue
            if blocks is None:
                blocks = message.content.copy()
            blocks[block_index] = ContentBlockText(
                type="text",
                text=tool_history_context(public),
            )
        if blocks is not None:
            projected[turn_index] = message.model_copy(update={"content": blocks})
    if any(calls.values()):
        raise HistoryReplayError(
            "An active server tool cannot continue through a different protocol."
        )
    return projected


def _document_parts(
    document: Mapping[str, object],
) -> list[ContentBlockText | ContentBlockImage]:
    source = document.get("source")
    if not isinstance(source, dict) or source.get("type") not in ("text", "content"):
        return []
    headings = [
        text
        for text in (document.get("title"), document.get("context"))
        if isinstance(text, str) and text
    ]
    content = source.get("data" if source["type"] == "text" else "content")
    if isinstance(content, str):
        text = "\n\n".join([*headings, content] if content else headings)
        return [ContentBlockText(type="text", text=text)] if text else []
    parts: list[ContentBlockText | ContentBlockImage] = []
    if source["type"] == "content" and isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if (
                part.get("type") == "text"
                and isinstance(text := part.get("text"), str)
                and text
            ):
                parts.append(ContentBlockText(type="text", text=text))
            elif part.get("type") == "image":
                parts.append(ContentBlockImage.model_validate(part))
    if headings and parts:
        parts.insert(0, ContentBlockText(type="text", text="\n\n".join(headings)))
    return parts


def _search_result_text(result: Mapping[str, object]) -> ContentBlockText:
    title, source, content = (
        result.get("title"),
        result.get("source"),
        result.get("content"),
    )
    if (
        not isinstance(title, str)
        or not isinstance(source, str)
        or not isinstance(content, list)
    ):
        raise ValueError(
            "Messages search_result requires a title, source and text content"
        )
    excerpts = [
        text
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(text := part.get("text"), str)
        and text
    ]
    if not excerpts:
        raise ValueError("Messages search_result contains no readable text")
    return ContentBlockText(type="text", text="\n\n".join([title, source, *excerpts]))


def _translated_result(block: ContentBlockToolResult) -> ContentBlockToolResult:
    if not isinstance(block.content, list):
        return block
    items: list[JsonValue] = []
    for item in block.content:
        if isinstance(item, dict) and item.get("type") == "document":
            items.extend(
                part.model_dump(exclude_unset=True) for part in _document_parts(item)
            )
        elif isinstance(item, dict) and item.get("type") == "search_result":
            items.append(_search_result_text(item).model_dump(exclude_unset=True))
        elif not isinstance(item, dict) or item.get("type") in (None, "text", "image"):
            items.append(item)
    if block.content and not items:
        raise ValueError("The tool result has no translatable content")
    return block.model_copy(update={"content": items})


def translated_messages(request: MessagesRequest) -> list[Message]:
    """Keep usable source turns before a foreign builder coalesces their roles."""
    if context := request.native_context:
        raise ValueError(f"Messages {context} requires native Messages egress")
    turns = quote_code_execution_history(request.messages)
    latest = next(
        (i for i in reversed(range(len(turns))) if turns[i].role == "user"), None
    )
    messages = []
    for index, turn in enumerate(turns):
        if turn.role == "user" and turn.reasoning_content is not None:
            turn = turn.model_copy(update={"reasoning_content": None})
        if not isinstance(turn.content, list):
            if index == latest and not turn.content:
                raise ValueError("The latest user message has no translatable content")
            messages.append(turn)
            continue
        blocks: list[MessageContentBlock] = []
        for block in turn.content:
            if isinstance(block, ExtensionBlock):
                if block.type == "search_result":
                    blocks.append(_search_result_text(block.model_extra or {}))
                continue
            if isinstance(block, ContentBlockDocument):
                blocks.extend(_document_parts(block.model_dump(exclude_unset=True)))
                continue
            if turn.role == "user" and (
                isinstance(block, (ContentBlockThinking, ContentBlockRedactedThinking))
                or (isinstance(block, ContentBlockText) and not block.text)
            ):
                continue
            if isinstance(block, ContentBlockToolResult):
                block = _translated_result(block)
            blocks.append(block)
        if not blocks and not turn.reasoning_content:
            if index == latest:
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

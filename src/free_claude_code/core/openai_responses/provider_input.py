"""Convert Anthropic Messages into an upstream OpenAI Responses request."""

import json
from typing import Any, cast

from free_claude_code.core.anthropic.content import get_block_attr, get_block_type
from free_claude_code.core.anthropic.conversion import resolve_anthropic_tool_choice
from free_claude_code.core.anthropic.image_sources import (
    AnthropicImageSourceError,
    portable_anthropic_image_url,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.request_serialization import (
    serialize_tool_result_content,
)
from free_claude_code.core.anthropic.tool_results import (
    ToolResultImage,
    ToolResultText,
    decompose_tool_result_content,
)
from free_claude_code.core.anthropic.tool_schema import sanitize_tool_schema_patterns
from free_claude_code.core.history_replay import (
    HistoryReplayError,
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
    has_readable_replay,
    is_replay,
    tool_history_context,
    validate_hosted_tool_history,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.reasoning import ReasoningPolicy

from .errors import ResponsesConversionError
from .reasoning import responses_reasoning_config


def build_responses_provider_request(
    request: MessagesRequest,
    *,
    reasoning: ReasoningPolicy,
) -> dict[str, Any]:
    """Build a stateless Responses request without silently dropping fields."""

    _validate_supported_request(request)
    tool_names = OpenAIToolNameCodec.from_request(request)
    instructions = _system_text(request)
    input_items: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "system":
            text = _message_text(message.content, context="system message")
            if text:
                instructions.append(text)
        elif message.role == "assistant":
            input_items.extend(
                _assistant_items(
                    message.content,
                    reasoning_content=message.reasoning_content,
                    tool_names=tool_names,
                )
            )
        else:
            input_items.extend(_user_items(message.content))

    if not input_items:
        raise ResponsesConversionError(
            "OpenAI Responses conversion requires at least one user or assistant item."
        )

    body: dict[str, Any] = {
        "model": request.model,
        "input": input_items,
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    if request.max_tokens is not None:
        body["max_output_tokens"] = request.max_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.metadata is not None:
        body["metadata"] = request.metadata
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "name": tool_names.encode(tool.name),
                "description": tool.description,
                "parameters": (
                    sanitize_tool_schema_patterns(tool.input_schema)
                    if tool.input_schema
                    else {"type": "object", "properties": {}}
                ),
                "strict": False,
            }
            for tool in request.tools
        ]
    tool_choice = resolve_anthropic_tool_choice(request.tools, request.tool_choice)
    if tool_choice is not None:
        body["tool_choice"] = _tool_choice(tool_choice, tool_names=tool_names)
    if reasoning_config := responses_reasoning_config(reasoning):
        body["reasoning"] = reasoning_config
    return body


def _validate_supported_request(request: MessagesRequest) -> None:
    if request.model_extra:
        raise ResponsesConversionError(
            "OpenAI Responses does not support these request fields: "
            f"{sorted(str(key) for key in request.model_extra)}."
        )
    provider_tool_types = sorted(
        {tool.type for tool in request.tools or () if tool.type is not None}
    )
    if provider_tool_types:
        raise ResponsesConversionError(
            "OpenAI Responses cannot represent provider-managed tool types: "
            f"{provider_tool_types}."
        )
    unsupported: list[str] = []
    if request.stop_sequences:
        unsupported.append("stop_sequences")
    if request.top_k is not None:
        unsupported.append("top_k")
    if not _is_noop_context_management(request.context_management):
        unsupported.append("context_management")
    unsupported.extend(_unsupported_output_config_paths(request.output_config))
    if request.mcp_servers:
        unsupported.append("mcp_servers")
    if request.extra_body:
        unsupported.append("extra_body")
    if unsupported:
        raise ResponsesConversionError(
            "OpenAI Responses cannot represent these fields without data loss: "
            f"{unsupported}."
        )


def _unsupported_output_config_paths(
    output_config: dict[str, Any] | None,
) -> list[str]:
    if not output_config:
        return []
    return [f"output_config.{key}" for key in sorted(output_config) if key != "effort"]


def _is_noop_context_management(
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


def _system_text(request: MessagesRequest) -> list[str]:
    if request.system is None:
        return []
    if isinstance(request.system, str):
        return [request.system] if request.system else []
    return [part.text for part in request.system if part.text]


def _assistant_items(
    content: Any,
    *,
    reasoning_content: str | None,
    tool_names: OpenAIToolNameCodec,
) -> list[dict[str, Any]]:
    blocks = (
        [{"type": "text", "text": content}] if isinstance(content, str) else content
    )
    if not isinstance(blocks, list):
        raise ResponsesConversionError(
            "Assistant content must be text or content blocks."
        )
    try:
        validate_hosted_tool_history(
            [
                {
                    "type": get_block_type(block),
                    "id": get_block_attr(block, "id", None),
                    "tool_use_id": get_block_attr(block, "tool_use_id", None),
                }
                for block in blocks
            ]
        )
    except HistoryReplayError as error:
        raise ResponsesConversionError(str(error)) from error
    items: list[dict[str, Any]] = []
    text_parts: list[dict[str, Any]] = []

    def flush_text() -> None:
        if text_parts:
            items.append(_assistant_message(list(text_parts)))
            text_parts.clear()

    if reasoning_content is not None and not any(
        get_block_type(block) == "thinking" for block in blocks
    ):
        items.append(
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": reasoning_content}],
                "summary": [],
            }
        )
    for block in blocks:
        kind = get_block_type(block)
        if kind == "text":
            text_parts.append(
                {"type": "output_text", "text": str(get_block_attr(block, "text", ""))}
            )
            continue
        flush_text()
        if kind == "thinking":
            signature = get_block_attr(block, "signature", None)
            item: dict[str, Any] = {"type": "reasoning", "summary": []}
            if isinstance(signature, str) and signature:
                native = (
                    block
                    if isinstance(block, dict)
                    else block.model_dump(mode="json", exclude_none=True)
                )
                item["encrypted_content"] = (
                    signature
                    if is_replay(signature)
                    else encode_replay(
                        ReplayRecord(
                            ReplayOrigin(
                                "unattributed/messages", "messages", "", "", "unknown"
                            ),
                            cast(JsonObject, native),
                        )
                    )
                )
            if not signature or (
                is_replay(signature) and not has_readable_replay(signature)
            ):
                item["content"] = [
                    {
                        "type": "reasoning_text",
                        "text": str(get_block_attr(block, "thinking", "")),
                    }
                ]
            items.append(item)
        elif kind == "redacted_thinking":
            items.append(
                {
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": str(get_block_attr(block, "data", "")),
                }
            )
        elif kind == "tool_use":
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(get_block_attr(block, "id", "")),
                    "name": tool_names.encode(str(get_block_attr(block, "name", ""))),
                    "arguments": json.dumps(
                        get_block_attr(block, "input", {}),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        elif kind in {
            "server_tool_use",
            "web_search_tool_result",
            "web_fetch_tool_result",
        }:
            native = (
                block
                if isinstance(block, dict)
                else block.model_dump(mode="json", exclude_none=True)
            )
            items.append(
                _assistant_message(
                    [{"type": "output_text", "text": tool_history_context(native)}]
                )
            )
        else:
            raise ResponsesConversionError(
                f"OpenAI Responses cannot represent assistant content block {kind!r}."
            )
    flush_text()
    return items


def _user_items(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [_user_message([{"type": "input_text", "text": content}])]
    if not isinstance(content, list):
        raise ResponsesConversionError("User content must be text or content blocks.")

    items: list[dict[str, Any]] = []
    message_parts: list[dict[str, Any]] = []

    def flush_message() -> None:
        if message_parts:
            items.append(_user_message(list(message_parts)))
            message_parts.clear()

    for block in content:
        block_type = get_block_type(block)
        if block_type == "text":
            message_parts.append(
                {
                    "type": "input_text",
                    "text": str(get_block_attr(block, "text", "")),
                }
            )
        elif block_type == "image":
            message_parts.append(_image_part(block))
        elif block_type == "tool_result":
            flush_message()
            tool_content = get_block_attr(block, "content")
            try:
                decomposed = decompose_tool_result_content(tool_content)
            except AnthropicImageSourceError as exc:
                raise ResponsesConversionError(str(exc)) from exc
            if decomposed.has_images:
                output: str | list[JsonObject] = []
                for part in decomposed.parts:
                    if isinstance(part, ToolResultText):
                        output.append({"type": "input_text", "text": part.text})
                    elif isinstance(part, ToolResultImage):
                        output.append({"type": "input_image", "image_url": part.url})
            else:
                output = serialize_tool_result_content(tool_content)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(get_block_attr(block, "tool_use_id", "")),
                    "output": output,
                }
            )
        elif block_type == "document":
            raise ResponsesConversionError(
                "OpenAI Responses provider does not support Anthropic document blocks."
            )
        else:
            raise ResponsesConversionError(
                f"OpenAI Responses cannot represent user content block {block_type!r}."
            )
    flush_message()
    return items


def _image_part(block: Any) -> dict[str, Any]:
    try:
        url = portable_anthropic_image_url(get_block_attr(block, "source", {}))
    except AnthropicImageSourceError as exc:
        raise ResponsesConversionError(str(exc)) from exc
    return {"type": "input_image", "image_url": url}


def _message_text(content: Any, *, context: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesConversionError(f"{context} must contain only text.")
    parts: list[str] = []
    for block in content:
        if get_block_type(block) != "text":
            raise ResponsesConversionError(f"{context} must contain only text.")
        parts.append(str(get_block_attr(block, "text", "")))
    return "\n\n".join(parts)


def _assistant_message(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": content}


def _user_message(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": content}


def _tool_choice(
    choice: dict[str, Any],
    *,
    tool_names: OpenAIToolNameCodec,
) -> str | dict[str, str]:
    choice_type = choice.get("type")
    if choice_type in {"auto", "none"}:
        return str(choice_type)
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise ResponsesConversionError("Forced tool choice requires a tool name.")
        return {"type": "function", "name": tool_names.encode(name)}
    raise ResponsesConversionError(f"Unsupported tool_choice type {choice_type!r}.")

"""Message and tool format converters."""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from free_claude_code.core.history_replay import (
    HistoryReplayError,
    has_readable_replay,
    is_replay,
    readable_reasoning,
    reasoning_detail,
    tool_history_context,
    unprotected_replay,
    validate_hosted_tool_history,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_chat import (
    IMAGE_TOOL_RESULT_MARKER,
    ChatToolResultImages,
    close_chat_tool_result_turns,
    image_tool_result_label,
)
from free_claude_code.core.tool_schema_patterns import translate_tool_schema_patterns

from .content import get_block_attr, get_block_type
from .image_sources import AnthropicImageSourceError, portable_anthropic_image_url
from .models import ExtensionBlock, MessagesRequest, Tool
from .request_serialization import (
    openai_output_schema,
    serialize_tool_result_content,
    translated_messages,
)
from .tool_results import (
    ToolResultImage,
    ToolResultText,
    decompose_tool_result_content,
)
from .utils import set_if_not_none


class OpenAIConversionError(Exception):
    """Raised when Anthropic content cannot be converted to OpenAI chat without data loss."""


class _MessageInput(Protocol):
    @property
    def role(self) -> str: ...

    @property
    def content(self) -> object: ...


class _ToolInput(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str | None: ...


class ReasoningReplayMode(StrEnum):
    """How assistant reasoning history is replayed to OpenAI-compatible providers."""

    REASONING_CONTENT = "reasoning_content"
    REASONING = "reasoning"


def resolve_anthropic_tool_choice(
    tools: Sequence[Tool] | None,
    tool_choice: JsonObject | None,
) -> JsonObject | None:
    """Validate the caller's choice against the functions that survived translation."""
    kind = tool_choice.get("type") if tool_choice else "auto"
    if kind == "tool" and not any(
        tool.name == (tool_choice or {}).get("name") for tool in tools or ()
    ):
        raise ValueError("Named tool choice requires an available function")
    if kind in ("any", "required") and not tools:
        raise ValueError("Required tool choice has no available functions")
    if kind not in ("auto", "any", "required", "none", "tool"):
        raise ValueError("Unsupported tool choice")
    return (tool_choice or {"type": "auto"}) if tools else None


def unrepresentable_client_tool_constraints(tool: Tool) -> list[str]:
    """Identify Anthropic constraints a plain upstream function cannot enforce."""
    options: Mapping[str, object] = tool.model_extra or {}
    unsupported = [
        key
        for key in ("strict", "defer_loading")
        if options.get(key) not in (None, False)
    ]
    if options.get("allowed_callers") not in (None, ["direct"]):
        unsupported.append("allowed_callers")
    return unsupported


def _set_replayed_reasoning(
    message: dict[str, Any], reasoning: str, mode: ReasoningReplayMode
) -> None:
    message[mode.value] = reasoning


def _tool_input_schema(tool: object) -> dict[str, Any]:
    schema: object = getattr(tool, "input_schema", None)
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}


def _clean_reasoning_content(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value


def _tool_call_from_tool_use(block: object) -> dict[str, Any]:
    tool_input = get_block_attr(block, "input", {})
    tool_call: dict[str, Any] = {
        "id": get_block_attr(block, "id"),
        "type": "function",
        "function": {
            "name": get_block_attr(block, "name"),
            "arguments": json.dumps(tool_input)
            if isinstance(tool_input, dict)
            else str(tool_input),
        },
    }
    extra_content = get_block_attr(block, "extra_content", None)
    if isinstance(extra_content, dict) and extra_content:
        tool_call["extra_content"] = deepcopy(extra_content)
    return tool_call


@dataclass
class _PlainSegment:
    messages: list[dict[str, Any]]


@dataclass
class _ToolTurnSegment:
    assistant_message: dict[str, Any]
    required_tool_ids: list[str]
    deferred_blocks: Sequence[object] = field(default_factory=list)
    top_level_reasoning: str | None = None
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.REASONING_CONTENT
    assistant_emitted: bool = False


_TranscriptSegment = _PlainSegment | _ToolTurnSegment


def _tool_call_ids(tool_calls: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for tool_call in tool_calls:
        tool_id: object = tool_call.get("id")
        if tool_id is not None and str(tool_id).strip() != "":
            ids.append(str(tool_id))
    return ids


def _index_first_tool_use(blocks: Sequence[object]) -> int | None:
    for i, block in enumerate(blocks):
        if get_block_type(block) == "tool_use":
            return i
    return None


def _iter_tool_uses_in_order(blocks: Sequence[object]) -> list[dict[str, Any]]:
    return [
        _tool_call_from_tool_use(block)
        for block in blocks
        if get_block_type(block) == "tool_use"
    ]


def _deferred_post_tool_blocks(
    content: Sequence[object], *, first_tool_index: int
) -> list[object]:
    return [
        b
        for i, b in enumerate(content)
        if i > first_tool_index and get_block_type(b) != "tool_use"
    ]


def _assert_no_forbidden_assistant_block(block: object) -> None:
    block_type = get_block_type(block)
    if block_type == "image":
        raise OpenAIConversionError(
            "Assistant image blocks are not supported for OpenAI chat conversion."
        )


def _openai_system_text(
    content: object,
    *,
    context: str,
) -> str | None:
    """Return OpenAI-compatible system text without silently dropping blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise OpenAIConversionError(
            f"OpenAI chat conversion requires {context} content to be text."
        )
    if not content:
        return None

    text_parts: list[str] = []
    for block in content:
        block_type = get_block_type(block)
        if block_type != "text":
            raise OpenAIConversionError(
                f"OpenAI chat conversion cannot represent {context} content block "
                f"{block_type!r} without data loss."
            )
        text_parts.append(str(get_block_attr(block, "text", "")))

    return "\n\n".join(text_parts)


def _openai_user_image_part(block: object) -> JsonObject:
    """Convert one Anthropic user image block without performing I/O."""
    try:
        url = portable_anthropic_image_url(get_block_attr(block, "source", {}))
    except AnthropicImageSourceError as exc:
        raise OpenAIConversionError(str(exc)) from exc

    return {"type": "image_url", "image_url": {"url": url}}


@dataclass(frozen=True)
class _OpenAIChatToolResult:
    tool_message: JsonObject
    rich_user_message: JsonObject | None = None


def _openai_chat_tool_result(block: object) -> _OpenAIChatToolResult:
    tool_id = get_block_attr(block, "tool_use_id")
    if not isinstance(tool_id, str):
        raise OpenAIConversionError("OpenAI chat replay requires a tool result ID.")
    tool_content = get_block_attr(block, "content", "")
    try:
        decomposed = decompose_tool_result_content(tool_content)
    except AnthropicImageSourceError as exc:
        raise OpenAIConversionError(str(exc)) from exc

    if not decomposed.has_images:
        serialized = serialize_tool_result_content(tool_content)
        return _OpenAIChatToolResult(
            tool_message={
                "role": "tool",
                "tool_call_id": tool_id,
                "content": serialized if serialized else "",
            }
        )

    rich_parts: list[JsonObject] = [
        {"type": "text", "text": image_tool_result_label(tool_id)}
    ]
    for part in decomposed.parts:
        if isinstance(part, ToolResultText):
            rich_parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ToolResultImage):
            rich_parts.append({"type": "image_url", "image_url": {"url": part.url}})

    return _OpenAIChatToolResult(
        tool_message={
            "role": "tool",
            "tool_call_id": tool_id,
            "content": IMAGE_TOOL_RESULT_MARKER,
        },
        rich_user_message=ChatToolResultImages(role="user", content=rich_parts),
    )


def _openai_user_content_parts(content: object) -> list[dict[str, Any]]:
    """Return an owned content-part list for one converted user message."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list) and all(isinstance(part, dict) for part in content):
        return deepcopy(content)
    raise OpenAIConversionError(
        "OpenAI chat conversion produced unsupported user message content."
    )


def _merge_openai_user_content(
    first: object, second: object
) -> str | list[dict[str, Any]]:
    """Combine adjacent user content while preserving multimodal part order."""
    if isinstance(first, str) and isinstance(second, str):
        return f"{first}\n\n{second}"

    first_parts = _openai_user_content_parts(first)
    second_parts = _openai_user_content_parts(second)
    first_ends_in_text = bool(first_parts) and first_parts[-1].get("type") == "text"
    second_starts_with_text = (
        bool(second_parts) and second_parts[0].get("type") == "text"
    )
    if first_ends_in_text and second_starts_with_text:
        first_text: object = first_parts[-1].get("text", "")
        second_text: object = second_parts.pop(0).get("text", "")
        first_parts[-1]["text"] = f"{first_text}\n\n{second_text}"
    return [*first_parts, *second_parts]


def _coalesce_openai_user_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine adjacent user messages after transcript ordering is complete."""
    result: list[dict[str, Any]] = []
    for message in messages:
        if (
            message.get("role") == "user"
            and result
            and result[-1].get("role") == "user"
        ):
            previous = result[-1]
            previous["content"] = _merge_openai_user_content(
                previous.get("content"), message.get("content")
            )
            continue
        result.append(message)
    return result


class _OpenAIChatHistoryLedger:
    """Assemble OpenAI chat history while respecting tool-result dependencies."""

    def __init__(self) -> None:
        self._output: list[dict[str, Any]] = []
        self._segments: list[_TranscriptSegment] = []
        self._tool_results: dict[str, _OpenAIChatToolResult] = {}

    def add_plain(self, messages: list[dict[str, Any]]) -> None:
        if messages:
            self._segments.append(_PlainSegment(messages))
            self._drain_ready_segments()

    def add_tool_turn(self, segment: _ToolTurnSegment) -> None:
        self._segments.append(segment)
        self._drain_ready_segments()

    def add_user_blocks(self, blocks: Sequence[object]) -> None:
        text_blocks: list[object] = []
        for block in blocks:
            block_type = get_block_type(block)
            if block_type == "tool_result":
                self._add_text_blocks(text_blocks)
                self._record_tool_result(block)
            else:
                text_blocks.append(block)
        self._add_text_blocks(text_blocks)
        self._drain_ready_segments()

    def finish(self) -> list[dict[str, Any]]:
        self._drain_ready_segments()
        missing = self._missing_required_tool_ids()
        if missing:
            raise OpenAIConversionError(
                "OpenAI chat conversion cannot replay incomplete tool history; "
                f"missing tool_result blocks for tool_use ids: {missing}"
            )
        while self._segments:
            segment = self._segments.pop(0)
            if isinstance(segment, _PlainSegment):
                self._output.extend(segment.messages)
                continue
            self._emit_tool_turn(segment)
        return self._output

    def _add_text_blocks(self, blocks: list[object]) -> None:
        if not blocks:
            return
        self.add_plain(AnthropicToOpenAIConverter._convert_user_message(blocks))
        blocks.clear()

    def _record_tool_result(self, block: object) -> None:
        tuid = get_block_attr(block, "tool_use_id")
        tuid_s = str(tuid) if tuid is not None else ""
        if not tuid_s:
            self.add_plain(AnthropicToOpenAIConverter._convert_user_message([block]))
            return
        converted = _openai_chat_tool_result(block)
        if self._has_pending_tool_id(tuid_s):
            self._tool_results[tuid_s] = converted
        else:
            messages = [converted.tool_message]
            if converted.rich_user_message is not None:
                messages.append(converted.rich_user_message)
            self.add_plain(messages)

    def _drain_ready_segments(self) -> None:
        while self._segments:
            segment = self._segments[0]
            if isinstance(segment, _PlainSegment):
                self._output.extend(segment.messages)
                self._segments.pop(0)
                continue

            if not segment.assistant_emitted:
                self._output.append(segment.assistant_message)
                segment.assistant_emitted = True

            missing = [
                tool_id
                for tool_id in segment.required_tool_ids
                if tool_id not in self._tool_results
            ]
            if missing:
                break

            self._segments.pop(0)
            results = [
                self._tool_results.pop(tool_id) for tool_id in segment.required_tool_ids
            ]
            self._output.extend(result.tool_message for result in results)
            self._output.extend(
                result.rich_user_message
                for result in results
                if result.rich_user_message is not None
            )
            deferred_messages = (
                AnthropicToOpenAIConverter._deferred_post_tool_to_messages(segment)
            )
            self._output.extend(deferred_messages)

    def _emit_tool_turn(self, segment: _ToolTurnSegment) -> None:
        if not segment.assistant_emitted:
            self._output.append(segment.assistant_message)
            segment.assistant_emitted = True
        results = [
            tool_result
            for tool_id in segment.required_tool_ids
            if (tool_result := self._tool_results.pop(tool_id, None)) is not None
        ]
        self._output.extend(result.tool_message for result in results)
        self._output.extend(
            result.rich_user_message
            for result in results
            if result.rich_user_message is not None
        )
        self._output.extend(
            AnthropicToOpenAIConverter._deferred_post_tool_to_messages(segment)
        )

    def _missing_required_tool_ids(self) -> list[str]:
        missing: list[str] = []
        for segment in self._segments:
            if not isinstance(segment, _ToolTurnSegment):
                continue
            missing.extend(
                tool_id
                for tool_id in segment.required_tool_ids
                if tool_id not in self._tool_results
            )
        return missing

    def _has_pending_tool_id(self, tool_id: str) -> bool:
        return any(
            isinstance(segment, _ToolTurnSegment)
            and tool_id in segment.required_tool_ids
            for segment in self._segments
        )


class AnthropicToOpenAIConverter:
    """Convert Anthropic message format to OpenAI-compatible format."""

    @staticmethod
    def convert_messages(
        messages: Sequence[_MessageInput],
        *,
        reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.REASONING_CONTENT,
        structured_reasoning_details: bool = False,
    ) -> list[dict[str, Any]]:
        ledger = _OpenAIChatHistoryLedger()

        for msg in messages:
            role = msg.role
            content = msg.content
            reasoning_content = _clean_reasoning_content(
                getattr(msg, "reasoning_content", None)
            )

            if role == "user" and isinstance(content, list):
                ledger.add_user_blocks(content)
                continue

            segments = AnthropicToOpenAIConverter._convert_message_to_segments(
                role,
                content,
                reasoning_content=reasoning_content,
                reasoning_replay=reasoning_replay,
                structured_reasoning_details=structured_reasoning_details,
            )
            for segment in segments:
                if isinstance(segment, _PlainSegment):
                    ledger.add_plain(segment.messages)
                else:
                    ledger.add_tool_turn(segment)

        ordered_messages = ledger.finish()
        closed_messages = close_chat_tool_result_turns(ordered_messages)
        return _coalesce_openai_user_messages(closed_messages)

    @staticmethod
    def _convert_message_to_segments(
        role: str,
        content: object,
        *,
        reasoning_content: str | None,
        reasoning_replay: ReasoningReplayMode,
        structured_reasoning_details: bool = False,
    ) -> list[_TranscriptSegment]:
        if role == "assistant" and isinstance(content, list):
            try:
                validate_hosted_tool_history(
                    [
                        {
                            "type": get_block_type(block),
                            "id": get_block_attr(block, "id", None),
                            "tool_use_id": get_block_attr(block, "tool_use_id", None),
                        }
                        for block in content
                    ]
                )
            except HistoryReplayError as error:
                raise OpenAIConversionError(str(error)) from error
            if (first_i := _index_first_tool_use(content)) is not None:
                for block in content:
                    if get_block_type(block) == "tool_use":
                        continue
                    _assert_no_forbidden_assistant_block(block)
                return [
                    AnthropicToOpenAIConverter._convert_assistant_message_with_split(
                        content,
                        first_tool_index=first_i,
                        reasoning_content=reasoning_content,
                        reasoning_replay=reasoning_replay,
                        structured_reasoning_details=structured_reasoning_details,
                    )
                ]
            for block in content:
                _assert_no_forbidden_assistant_block(block)
            return [
                _PlainSegment(
                    AnthropicToOpenAIConverter._convert_assistant_message(
                        content,
                        reasoning_content=reasoning_content,
                        reasoning_replay=reasoning_replay,
                        structured_reasoning_details=structured_reasoning_details,
                    )
                )
            ]
        if role == "user" and isinstance(content, list):
            return [
                _PlainSegment(AnthropicToOpenAIConverter._convert_user_message(content))
            ]
        if isinstance(content, str):
            converted = {"role": role, "content": content}
            if role == "assistant" and reasoning_content is not None:
                _set_replayed_reasoning(converted, reasoning_content, reasoning_replay)
            return [_PlainSegment([converted])]
        if isinstance(content, list):
            return []
        return [_PlainSegment([{"role": role, "content": str(content)}])]

    @staticmethod
    def _convert_assistant_message_with_split(
        content: Sequence[object],
        *,
        first_tool_index: int,
        reasoning_content: str | None,
        reasoning_replay: ReasoningReplayMode,
        structured_reasoning_details: bool = False,
    ) -> _ToolTurnSegment:
        pre = [
            block
            for index, block in enumerate(content)
            if index < first_tool_index
            or get_block_type(block) in {"thinking", "redacted_thinking"}
        ]
        tool_calls = _iter_tool_uses_in_order(content)
        if not tool_calls:
            return _ToolTurnSegment(
                assistant_message=AnthropicToOpenAIConverter._convert_assistant_message(
                    content,
                    reasoning_content=reasoning_content,
                    reasoning_replay=reasoning_replay,
                    structured_reasoning_details=structured_reasoning_details,
                )[0],
                required_tool_ids=[],
            )
        deferred_blocks = _deferred_post_tool_blocks(
            content, first_tool_index=first_tool_index
        )
        deferred_blocks = [
            block
            for block in deferred_blocks
            if get_block_type(block) not in {"thinking", "redacted_thinking"}
        ]

        pre_msg: dict[str, Any]
        if not pre:
            pre_msg = {
                "role": "assistant",
                "content": "",
            }
            if reasoning_content is not None:
                _set_replayed_reasoning(pre_msg, reasoning_content, reasoning_replay)
        else:
            pre_msg = AnthropicToOpenAIConverter._convert_assistant_message(
                pre,
                reasoning_content=reasoning_content,
                reasoning_replay=reasoning_replay,
                structured_reasoning_details=structured_reasoning_details,
            )[0]
        pre_msg["tool_calls"] = tool_calls
        if tool_calls and pre_msg.get("content") == " ":
            pre_msg["content"] = ""
        return _ToolTurnSegment(
            assistant_message=pre_msg,
            required_tool_ids=_tool_call_ids(tool_calls),
            deferred_blocks=deferred_blocks,
            reasoning_replay=reasoning_replay,
        )

    @staticmethod
    def _convert_assistant_message(
        content: Sequence[object],
        *,
        reasoning_content: str | None = None,
        reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.REASONING_CONTENT,
        structured_reasoning_details: bool = False,
    ) -> list[dict[str, Any]]:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        thinking_seen = False
        tool_calls: list[dict[str, Any]] = []
        details: list[Any] = []
        for block in content:
            block_type = get_block_type(block)
            carrier = (
                get_block_attr(block, "signature", None)
                if block_type == "thinking"
                else get_block_attr(block, "data", None)
                if block_type == "redacted_thinking"
                else None
            )
            if record := unprotected_replay(carrier):
                thinking_seen = True
                thinking_parts.extend(
                    text for text, _ in readable_reasoning(record.native)
                )
                continue
            if block_type == "text":
                text = get_block_attr(block, "text", "")
                if not isinstance(text, str):
                    raise OpenAIConversionError(
                        "Anthropic text blocks must contain text."
                    )
                content_parts.append(text)
            elif block_type == "thinking":
                signature = get_block_attr(block, "signature", None)
                thinking = get_block_attr(block, "thinking", "")
                if not isinstance(thinking, str):
                    raise OpenAIConversionError(
                        "Anthropic thinking blocks must contain text."
                    )
                if is_replay(signature):
                    details.extend(reasoning_detail(signature))
                    if has_readable_replay(signature):
                        continue
                elif isinstance(signature, str) and signature:
                    if structured_reasoning_details:
                        details.append(
                            {
                                "type": "reasoning.text",
                                "text": thinking,
                                "signature": signature,
                            }
                        )
                        continue
                    details.append({"type": "reasoning.encrypted", "data": signature})
                if reasoning_content is None:
                    thinking_seen = True
                    thinking_parts.append(thinking)
            elif block_type == "redacted_thinking":
                data = get_block_attr(block, "data", None)
                if isinstance(data, str) and data:
                    details.extend(reasoning_detail(data))
            elif block_type in {
                "server_tool_use",
                "web_search_tool_result",
                "web_fetch_tool_result",
            }:
                value = block
                if not isinstance(value, dict):
                    dump: object = getattr(block, "model_dump", None)
                    if not callable(dump):
                        raise OpenAIConversionError(
                            "Anthropic hosted tool blocks require an object."
                        )
                    value = dump(mode="json", exclude_none=True)
                if not isinstance(value, dict):
                    raise OpenAIConversionError(
                        "Anthropic hosted tool blocks require an object."
                    )
                content_parts.append(tool_history_context(value))
            elif block_type == "tool_use":
                tool_calls.append(_tool_call_from_tool_use(block))
            else:
                _assert_no_forbidden_assistant_block(block)

        content_str = "\n\n".join(content_parts)
        if not content_str and not tool_calls:
            content_str = " "

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content_str,
        }
        if details:
            msg["reasoning_details"] = details
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if reasoning_content is not None:
            _set_replayed_reasoning(msg, reasoning_content, reasoning_replay)
        elif thinking_seen:
            _set_replayed_reasoning(msg, "\n".join(thinking_parts), reasoning_replay)

        return [msg]

    @staticmethod
    def _deferred_post_tool_to_messages(
        pending: _ToolTurnSegment,
    ) -> list[dict[str, Any]]:
        if not pending.deferred_blocks:
            return []
        return AnthropicToOpenAIConverter._convert_assistant_message(
            pending.deferred_blocks,
            reasoning_content=pending.top_level_reasoning,
            reasoning_replay=pending.reasoning_replay,
        )

    @staticmethod
    def _convert_user_message(content: Sequence[object]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        content_parts: list[dict[str, Any]] = []
        tool_results: list[_OpenAIChatToolResult] = []

        def flush_content() -> None:
            if not content_parts:
                return
            if all(part["type"] == "text" for part in content_parts):
                message_content: str | list[dict[str, Any]] = "\n".join(
                    part["text"] for part in content_parts
                )
            else:
                message_content = list(content_parts)
            result.append({"role": "user", "content": message_content})
            content_parts.clear()

        def flush_tool_results() -> None:
            if not tool_results:
                return
            result.extend(item.tool_message for item in tool_results)
            result.extend(
                item.rich_user_message
                for item in tool_results
                if item.rich_user_message is not None
            )
            tool_results.clear()

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                flush_tool_results()
                content_parts.append(
                    {"type": "text", "text": get_block_attr(block, "text", "")}
                )
            elif block_type == "image":
                flush_tool_results()
                content_parts.append(_openai_user_image_part(block))
            elif block_type == "tool_result":
                flush_content()
                tool_results.append(_openai_chat_tool_result(block))

        flush_tool_results()
        flush_content()
        return result

    @staticmethod
    def convert_tools(tools: Sequence[_ToolInput]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": translate_tool_schema_patterns(
                        _tool_input_schema(tool)
                    ),
                },
            }
            for tool in tools
        ]

    @staticmethod
    def convert_tool_choice(tool_choice: JsonValue) -> JsonValue:
        if not isinstance(tool_choice, dict):
            return tool_choice

        choice_type = tool_choice.get("type")
        if choice_type == "tool":
            name = tool_choice.get("name")
            if name:
                return {"type": "function", "function": {"name": name}}
        if choice_type == "any":
            return "required"
        if choice_type in {"auto", "none", "required"}:
            return choice_type
        if choice_type == "function" and isinstance(tool_choice.get("function"), dict):
            return tool_choice

        return tool_choice

    @staticmethod
    def convert_system_prompt(system: object) -> dict[str, str] | None:
        if system is None:
            return None
        system_text = _openai_system_text(
            system,
            context="the top-level Anthropic system prompt",
        )
        if system_text is None:
            return None
        if isinstance(system, list):
            system_text = system_text.strip()
        return {"role": "system", "content": system_text}


def build_base_request_body(
    request_data: MessagesRequest,
    *,
    default_max_tokens: int | None = None,
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.REASONING_CONTENT,
    structured_reasoning_details: bool = False,
) -> dict[str, Any]:
    """Build the common parts of an OpenAI-format request body."""
    tools = request_data.client_tools
    try:
        source_messages = translated_messages(request_data)
        output_schema = openai_output_schema(request_data)
        tool_choice = resolve_anthropic_tool_choice(tools, request_data.tool_choice)
    except ValueError as error:
        raise OpenAIConversionError(str(error)) from error
    messages = AnthropicToOpenAIConverter.convert_messages(
        source_messages,
        reasoning_replay=reasoning_replay,
        structured_reasoning_details=structured_reasoning_details,
    )

    system = request_data.system
    if isinstance(system, list):
        system = [block for block in system if not isinstance(block, ExtensionBlock)]
    if system:
        system_msg = AnthropicToOpenAIConverter.convert_system_prompt(system)
        if system_msg:
            messages.insert(0, system_msg)

    body: dict[str, Any] = {"model": request_data.model, "messages": messages}

    if output_schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": output_schema}
    max_tokens = request_data.max_tokens
    set_if_not_none(body, "max_tokens", max_tokens or default_max_tokens)
    set_if_not_none(body, "temperature", request_data.temperature)
    set_if_not_none(body, "top_p", request_data.top_p)

    stop_sequences = request_data.stop_sequences
    if stop_sequences:
        body["stop"] = stop_sequences

    if tools:
        body["tools"] = AnthropicToOpenAIConverter.convert_tools(tools)
        for declaration, tool in zip(body["tools"], tools, strict=True):
            options = tool.model_extra or {}
            if "strict" in options:
                declaration["function"]["strict"] = options["strict"]
            if any(
                key != "strict" for key in unrepresentable_client_tool_constraints(tool)
            ):
                raise OpenAIConversionError(
                    f"Unsupported caller constraints on tool {tool.name!r}"
                )
    if (
        tool_choice
        and tool_choice.get("disable_parallel_tool_use")
        and tool_choice.get("type") != "none"
    ):
        body["parallel_tool_calls"] = False
    if tool_choice:
        body["tool_choice"] = AnthropicToOpenAIConverter.convert_tool_choice(
            tool_choice
        )

    return body

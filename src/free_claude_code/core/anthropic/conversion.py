"""Message and tool format converters."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from free_claude_code.core.history_replay import (
    HistoryReplayError,
    has_readable_replay,
    is_replay,
    reasoning_context,
    reasoning_detail,
    tool_history_context,
    validate_hosted_tool_history,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_chat import (
    IMAGE_TOOL_RESULT_MARKER,
    ChatToolResultImages,
    close_chat_tool_result_turns,
    image_tool_result_label,
)
from free_claude_code.core.tool_schema_patterns import translate_tool_schema_patterns

from .content import get_block_attr, get_block_type
from .image_sources import AnthropicImageSourceError, portable_anthropic_image_url
from .models import MessagesRequest
from .request_serialization import serialize_tool_result_content
from .tool_results import (
    ToolResultImage,
    ToolResultText,
    decompose_tool_result_content,
)
from .utils import set_if_not_none


class OpenAIConversionError(Exception):
    """Raised when Anthropic content cannot be converted to OpenAI chat without data loss."""


class ReasoningReplayMode(StrEnum):
    """How assistant reasoning history is replayed to OpenAI-compatible providers."""

    DISABLED = "disabled"
    THINK_TAGS = "think_tags"
    REASONING_CONTENT = "reasoning_content"
    REASONING = "reasoning"


def resolve_anthropic_tool_choice(
    tools: list[Any] | None,
    tool_choice: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Materialize Anthropic's automatic choice when tools are available."""
    if tool_choice is not None:
        return tool_choice
    if tools:
        return {"type": "auto"}
    return None


def _reasoning_replay_field(mode: ReasoningReplayMode) -> str | None:
    if mode in (
        ReasoningReplayMode.REASONING_CONTENT,
        ReasoningReplayMode.REASONING,
    ):
        return mode.value
    return None


def _set_replayed_reasoning(
    message: dict[str, Any], reasoning: str, mode: ReasoningReplayMode
) -> None:
    field_name = _reasoning_replay_field(mode)
    if field_name is not None:
        message[field_name] = reasoning


def _openai_reject_native_only_top_level_fields(
    request_data: MessagesRequest,
) -> None:
    """OpenAI chat providers may only convert known top-level request fields.

    First-class model fields (e.g. ``context_management``) are not forwarded to
    the OpenAI API but are allowed so clients do not hit spurious 400s.
    Unknown extra keys (``__pydantic_extra__``) are still rejected.
    """
    extra = request_data.model_extra
    if not extra:
        return
    raise OpenAIConversionError(
        "OpenAI chat conversion does not support these top-level request fields: "
        f"{sorted(str(k) for k in extra)}. Remove the unsupported fields."
    )


def _tool_input_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", None)
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}


def _clean_reasoning_content(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value


def _think_tag_content(reasoning: str) -> str:
    return f"<think>\n{reasoning}\n</think>"


def _tool_call_from_tool_use(block: Any) -> dict[str, Any]:
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
    deferred_blocks: list[Any] = field(default_factory=list)
    top_level_reasoning: str | None = None
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS
    assistant_emitted: bool = False


_TranscriptSegment = _PlainSegment | _ToolTurnSegment


def _tool_call_ids(tool_calls: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for tool_call in tool_calls:
        tool_id = tool_call.get("id")
        if tool_id is not None and str(tool_id).strip() != "":
            ids.append(str(tool_id))
    return ids


def _index_first_tool_use(blocks: list[Any]) -> int | None:
    for i, block in enumerate(blocks):
        if get_block_type(block) == "tool_use":
            return i
    return None


def _iter_tool_uses_in_order(blocks: list[Any]) -> list[dict[str, Any]]:
    return [
        _tool_call_from_tool_use(block)
        for block in blocks
        if get_block_type(block) == "tool_use"
    ]


def _deferred_post_tool_blocks(
    content: list[Any], *, first_tool_index: int
) -> list[Any]:
    return [
        b
        for i, b in enumerate(content)
        if i > first_tool_index and get_block_type(b) != "tool_use"
    ]


def _assert_no_forbidden_assistant_block(block: Any) -> None:
    block_type = get_block_type(block)
    if block_type == "image":
        raise OpenAIConversionError(
            "Assistant image blocks are not supported for OpenAI chat conversion."
        )


def _openai_system_text(
    content: Any,
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


def _openai_user_image_part(block: Any) -> JsonObject:
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


def _openai_chat_tool_result(block: Any) -> _OpenAIChatToolResult:
    tool_id = get_block_attr(block, "tool_use_id")
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
        {"type": "text", "text": image_tool_result_label(str(tool_id))}
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


def _openai_user_content_parts(content: Any) -> list[dict[str, Any]]:
    """Return an owned content-part list for one converted user message."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list) and all(isinstance(part, dict) for part in content):
        return deepcopy(content)
    raise OpenAIConversionError(
        "OpenAI chat conversion produced unsupported user message content."
    )


def _merge_openai_user_content(first: Any, second: Any) -> str | list[dict[str, Any]]:
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
        first_text = first_parts[-1].get("text", "")
        second_text = second_parts.pop(0).get("text", "")
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

    def add_user_blocks(self, blocks: list[Any]) -> None:
        text_blocks: list[Any] = []
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

    def _add_text_blocks(self, blocks: list[Any]) -> None:
        if not blocks:
            return
        self.add_plain(AnthropicToOpenAIConverter._convert_user_message(blocks))
        blocks.clear()

    def _record_tool_result(self, block: Any) -> None:
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
        messages: list[Any],
        *,
        reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS,
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
        content: Any,
        *,
        reasoning_content: str | None,
        reasoning_replay: ReasoningReplayMode,
    ) -> list[_TranscriptSegment]:
        if role == "system":
            system_text = _openai_system_text(
                content,
                context="an inline Anthropic system message",
            )
            if system_text is None:
                raise OpenAIConversionError(
                    "OpenAI chat conversion requires an inline Anthropic system "
                    "message to contain text."
                )
            # Reserve the downstream system role for request.system at index zero.
            return [_PlainSegment([{"role": "user", "content": system_text}])]
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
                if _reasoning_replay_field(reasoning_replay) is not None:
                    _set_replayed_reasoning(
                        converted, reasoning_content, reasoning_replay
                    )
                elif (
                    reasoning_replay
                    in {ReasoningReplayMode.THINK_TAGS, ReasoningReplayMode.DISABLED}
                    and reasoning_content
                ):
                    content_parts = [
                        _think_tag_content(reasoning_content)
                        if reasoning_replay == ReasoningReplayMode.THINK_TAGS
                        else reasoning_context(reasoning_content)
                    ]
                    if content:
                        content_parts.append(content)
                    converted["content"] = "\n\n".join(content_parts)
            return [_PlainSegment([converted])]
        if isinstance(content, list):
            return []
        return [_PlainSegment([{"role": role, "content": str(content)}])]

    @staticmethod
    def _convert_assistant_message_with_split(
        content: list[Any],
        *,
        first_tool_index: int,
        reasoning_content: str | None,
        reasoning_replay: ReasoningReplayMode,
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
            )[0]
        pre_msg["tool_calls"] = tool_calls
        if reasoning_replay is ReasoningReplayMode.REASONING_CONTENT:
            pre_msg.setdefault("reasoning_content", "")
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
        content: list[Any],
        *,
        reasoning_content: str | None = None,
        reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS,
    ) -> list[dict[str, Any]]:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        thinking_seen = False
        tool_calls: list[dict[str, Any]] = []
        details: list[Any] = []
        for block in content:
            block_type = get_block_type(block)
            if block_type == "text":
                content_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "thinking":
                signature = get_block_attr(block, "signature", None)
                if is_replay(signature):
                    details.extend(reasoning_detail(signature))
                    if has_readable_replay(signature):
                        continue
                thinking = get_block_attr(block, "thinking", "")
                if reasoning_replay == ReasoningReplayMode.DISABLED:
                    if thinking:
                        content_parts.append(reasoning_context(thinking))
                    continue
                if reasoning_replay == ReasoningReplayMode.THINK_TAGS:
                    content_parts.append(_think_tag_content(thinking))
                elif reasoning_content is None:
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
                value = (
                    block
                    if isinstance(block, dict)
                    else block.model_dump(mode="json", exclude_none=True)
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
        if _reasoning_replay_field(reasoning_replay) is not None:
            if reasoning_content is not None:
                _set_replayed_reasoning(msg, reasoning_content, reasoning_replay)
            elif thinking_seen:
                _set_replayed_reasoning(
                    msg, "\n".join(thinking_parts), reasoning_replay
                )

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
    def _convert_user_message(content: list[Any]) -> list[dict[str, Any]]:
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
    def convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
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
    def convert_tool_choice(tool_choice: Any) -> Any:
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
    def convert_system_prompt(system: Any) -> dict[str, str] | None:
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
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS,
) -> dict[str, Any]:
    """Build the common parts of an OpenAI-format request body."""
    _openai_reject_native_only_top_level_fields(request_data)
    messages = AnthropicToOpenAIConverter.convert_messages(
        request_data.messages,
        reasoning_replay=reasoning_replay,
    )

    system = request_data.system
    if system:
        system_msg = AnthropicToOpenAIConverter.convert_system_prompt(system)
        if system_msg:
            messages.insert(0, system_msg)

    body: dict[str, Any] = {"model": request_data.model, "messages": messages}

    max_tokens = request_data.max_tokens
    set_if_not_none(body, "max_tokens", max_tokens or default_max_tokens)
    set_if_not_none(body, "temperature", request_data.temperature)
    set_if_not_none(body, "top_p", request_data.top_p)

    stop_sequences = request_data.stop_sequences
    if stop_sequences:
        body["stop"] = stop_sequences

    tools = request_data.tools
    if tools:
        body["tools"] = AnthropicToOpenAIConverter.convert_tools(tools)
    tool_choice = resolve_anthropic_tool_choice(tools, request_data.tool_choice)
    if tool_choice:
        body["tool_choice"] = AnthropicToOpenAIConverter.convert_tool_choice(
            tool_choice
        )

    return body

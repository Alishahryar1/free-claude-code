"""Direct OpenAI Responses-to-Chat Completions request translation."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import assert_never, cast

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.history_replay import (
    has_readable_replay,
    is_replay,
    readable_reasoning,
    reasoning_detail,
    unprotected_replay,
)
from free_claude_code.core.json_types import JsonObject, JsonValue, load_json_value
from free_claude_code.core.openai_chat import (
    TOOL_RESULT_ATTACHMENT_MARKER,
    ChatToolResultAttachments,
    close_chat_tool_result_turns,
    inline_file_fields,
    tool_result_attachment_label,
)
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec

from .errors import ResponsesConversionError
from .input_history import (
    HistoryItem,
    InputCall,
    InputMessage,
    InputResult,
    ReasoningItem,
    Screenshot,
    input_items,
    read_history,
)
from .models import OpenAIResponsesRequest
from .reasoning import (
    combine_reasoning,
    encrypted_reasoning_from_item,
)
from .tool_adaptation import ResponsesToolAdapter, ResponsesToolPolicy
from .tools import (
    optional_str,
    required_str,
)

_CHAT_OPTION_FIELDS = (
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "n",
    "presence_penalty",
    "prompt_cache_key",
    "seed",
    "service_tier",
    "stop",
    "top_logprobs",
)


@dataclass(frozen=True, slots=True)
class ResponsesChatRequest:
    """One translated Chat body and its reversible tool-name scope."""

    body: dict[str, object]
    tool_names: OpenAIToolNameCodec
    reserved_tool_ids: frozenset[str]
    tool_adapter: ResponsesToolAdapter


@dataclass(slots=True)
class _PendingReasoning:
    items: list[Mapping[str, JsonValue]] = field(default_factory=list)

    def add(self, item: Mapping[str, JsonValue]) -> None:
        self.items.append(item)

    @property
    def empty(self) -> bool:
        return not self.items

    def take(self) -> list[Mapping[str, JsonValue]]:
        items = self.items
        self.items = []
        return items


class _ResponsesChatInputBuilder:
    def __init__(
        self,
        *,
        reasoning_replay: ReasoningReplayMode,
        structured_reasoning_details: bool,
    ) -> None:
        self.system_parts: list[str] = []
        self.messages: list[dict[str, object]] = []
        self._reasoning_replay = reasoning_replay
        self._structured_reasoning_details = structured_reasoning_details
        self._pending_reasoning = _PendingReasoning()
        self._pending_rich_output_parts: list[dict[str, object]] = []
        self._discovery_outputs: list[dict[str, object]] = []

    def add(self, item: HistoryItem, *, source_type: str | None) -> None:
        if not isinstance(item, (InputResult, Screenshot)):
            self._flush_rich_outputs()
        if isinstance(item, InputMessage):
            self._add_message(item)
        elif isinstance(item, ReasoningItem):
            self._pending_reasoning.add(item.data)
        elif isinstance(item, InputCall):
            self._add_tool_call(item)
        elif isinstance(item, InputResult):
            self._add_tool_output(item, source_type=source_type)
        elif isinstance(item, Screenshot):
            self._flush_reasoning()
            content = _message_content(item.content, allow_attachments=True)
            assert isinstance(content, list)
            self._pending_rich_output_parts.extend(content)
        else:
            assert_never(item)

    def finish(self) -> tuple[list[str], list[dict[str, object]]]:
        self._flush_rich_outputs()
        self._flush_reasoning()
        return self.system_parts, self.messages

    def encode_discovery_outputs(self, tool_names: OpenAIToolNameCodec) -> None:
        """Encode FCC's discovery definitions with the final Chat name mapping."""
        if not tool_names.has_aliases:
            return
        for message in self._discovery_outputs:
            tools = load_json_value(
                required_str(message["content"], "discovery output")
            )
            assert isinstance(tools, list)
            for tool in tools:
                assert isinstance(tool, dict)
                name = required_str(tool.get("name"), "discovery tool name")
                tool["name"] = tool_names.encode(name)
            message["content"] = json.dumps(tools)

    def _add_message(self, item: InputMessage) -> None:
        role = item.role
        if role in {"developer", "system"}:
            text = _content_text(item.content)
            if text:
                if self.messages or not self._pending_reasoning.empty:
                    self._flush_reasoning()
                    self.messages.append({"role": role, "content": text})
                else:
                    self.system_parts.append(text)
            return

        if role == "user":
            self._flush_reasoning()
        content = _message_content(item.content, allow_attachments=role == "user")
        message: dict[str, object] = {"role": role, "content": content}
        if role == "assistant":
            self._apply_pending_reasoning(message)
        if not content and len(message) == 2:
            return
        self.messages.append(message)

    def _add_tool_call(self, item: InputCall) -> None:
        call_id = item.call_id
        name = item.identity.name
        arguments = _arguments_text(item.arguments)

        message: dict[str, object] | None = self._last_tool_call_message()
        if message is None:
            message = {"role": "assistant", "content": "", "tool_calls": []}
            self._apply_pending_reasoning(message)
            self.messages.append(message)
        elif not self._pending_reasoning.empty:
            self._apply_pending_reasoning(message)

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            raise AssertionError("assistant tool-call message must contain a list")
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    def _add_tool_output(self, item: InputResult, *, source_type: str | None) -> None:
        call_id = item.call_id
        if not self._pending_reasoning.empty:
            previous = self._last_tool_call_message()
            if previous is not None:
                self._apply_pending_reasoning(previous)
            else:
                self._flush_reasoning()
        output = item.output
        rich_parts = _rich_tool_output_parts(output, call_id=call_id)
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": (
                    TOOL_RESULT_ATTACHMENT_MARKER
                    if rich_parts is not None
                    else _tool_output_text(output)
                ),
            }
        )
        if source_type == "tool_search_output":
            self._discovery_outputs.append(self.messages[-1])
        if rich_parts is not None:
            self._pending_rich_output_parts.extend(rich_parts)

    def _last_tool_call_message(self) -> dict[str, object] | None:
        if not self.messages:
            return None
        message = self.messages[-1]
        if message.get("role") != "assistant" or not isinstance(
            message.get("tool_calls"), list
        ):
            return None
        return message

    def _flush_reasoning(self) -> None:
        if self._pending_reasoning.empty:
            return
        if self.messages and self.messages[-1].get("role") == "assistant":
            self._apply_pending_reasoning(self.messages[-1])
            return
        message: dict[str, object] = {"role": "assistant", "content": ""}
        self._apply_pending_reasoning(message)
        if len(message) > 2 or message.get("content"):
            self.messages.append(message)

    def _flush_rich_outputs(self) -> None:
        if not self._pending_rich_output_parts:
            return
        self.messages.append(
            cast(
                dict[str, object],
                ChatToolResultAttachments(
                    role="user",
                    content=cast(
                        list[JsonValue], list(self._pending_rich_output_parts)
                    ),
                ),
            )
        )
        self._pending_rich_output_parts.clear()

    def _apply_pending_reasoning(self, message: dict[str, object]) -> None:
        contexts: list[str] = []
        details: list[JsonValue] = []
        for item in self._pending_reasoning.take():
            encrypted = encrypted_reasoning_from_item(item)
            unsigned = unprotected_replay(encrypted)
            if unsigned is not None or not encrypted:
                for text, _ in readable_reasoning(
                    unsigned.native if unsigned else item
                ):
                    _apply_reasoning_text(message, text, self._reasoning_replay)
                continue
            if encrypted and has_readable_replay(encrypted):
                details.extend(reasoning_detail(encrypted))
                continue
            readable = readable_reasoning(item)
            full_text = [text for text, summary in readable if not summary]
            content = item.get("content")
            signed = (
                self._structured_reasoning_details
                and encrypted
                and not is_replay(encrypted)
                and (
                    full_text
                    or (
                        isinstance(content, list)
                        and any(
                            isinstance(part, dict)
                            and part.get("type") == "reasoning_text"
                            and part.get("text") == ""
                            for part in content
                        )
                    )
                )
            )
            if signed:
                details.append(
                    {
                        "type": "reasoning.text",
                        "text": "".join(full_text),
                        "signature": encrypted,
                    }
                )
            else:
                for text in full_text:
                    _apply_reasoning_text(message, text, self._reasoning_replay)
            for text, summary in readable:
                if not summary:
                    continue
                if self._structured_reasoning_details:
                    details.append({"type": "reasoning.summary", "summary": text})
                elif encrypted:
                    contexts.append(text)
            if encrypted and not signed:
                details.extend(reasoning_detail(encrypted))
        if contexts:
            message["content"] = "\n\n".join(
                [*contexts, str(message.get("content") or "")]
            ).rstrip()
        if details:
            existing = message.setdefault("reasoning_details", [])
            if isinstance(existing, list):
                existing.extend(details)


def build_responses_chat_request(
    request: OpenAIResponsesRequest,
    *,
    reasoning_replay: ReasoningReplayMode,
    structured_reasoning_details: bool = False,
) -> ResponsesChatRequest:
    """Translate a Responses request directly into one Chat Completions body."""
    request.require_inline_context()
    adapter = ResponsesToolAdapter(
        request,
        ResponsesToolPolicy(
            custom_tools_as_functions=True,
            flatten_namespaces=True,
            client_tool_search=True,
            direct_tools_only=True,
        ),
    )
    request = adapter.request
    builder = _ResponsesChatInputBuilder(
        reasoning_replay=reasoning_replay,
        structured_reasoning_details=structured_reasoning_details,
    )
    if request.instructions:
        builder.system_parts.append(request.instructions)
    originals = input_items(adapter.original.input)
    for index, item in read_history(request.input):
        original = originals[adapter.input_source_indices[index]]
        source_type = (
            optional_str(original.get("type")) if isinstance(original, dict) else None
        )
        builder.add(item, source_type=source_type)
    system_parts, raw_messages = builder.finish()
    messages = cast(
        list[dict[str, object]],
        close_chat_tool_result_turns(cast(list[JsonObject], raw_messages)),
    )
    if not messages:
        raise ResponsesConversionError(
            "Responses request must contain usable input for Chat Completions"
        )

    body: dict[str, object] = {"model": request.model, "messages": messages}
    if system_parts:
        messages.insert(
            0,
            {"role": "system", "content": "\n\n".join(system_parts)},
        )

    tools, available_tool_names = _chat_tools(request.tools)
    choice = _chat_tool_choice(request.tool_choice, available_tool_names)
    if request.tool_choice != "none" and tools:
        body["tools"] = tools
        if choice is not None:
            body["tool_choice"] = choice

    if (
        request.parallel_tool_calls is not None
        and tools
        and request.tool_choice != "none"
    ):
        body["parallel_tool_calls"] = request.parallel_tool_calls
    if request.max_output_tokens is not None:
        body["max_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.metadata is not None:
        body["metadata"] = request.metadata

    extra: JsonObject = request.model_extra or {}
    for field_name in _CHAT_OPTION_FIELDS:
        value = extra.get(field_name)
        if value is not None:
            body[field_name] = value
    if response_format := _chat_response_format(extra.get("text")):
        body["response_format"] = response_format

    reserved_tool_ids = frozenset(_body_tool_call_ids(body))
    tool_names = OpenAIToolNameCodec.from_names(_body_tool_names(body))
    builder.encode_discovery_outputs(tool_names)
    return ResponsesChatRequest(
        body=body,
        tool_names=tool_names,
        reserved_tool_ids=reserved_tool_ids,
        tool_adapter=adapter,
    )


def _message_content(
    value: JsonValue, *, allow_attachments: bool
) -> str | list[dict[str, object]]:
    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray):
        if isinstance(value, Mapping):
            return _message_content([value], allow_attachments=allow_attachments)
        return ""

    parts: list[dict[str, object]] = []
    for part in value:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if part_type in ("input_text", "output_text", "text", "refusal"):
            parts.append({"type": "text", "text": _text_from_part(part)})
            continue
        if allow_attachments and part_type == "input_image":
            parts.append(_image_part(part, context="input_image"))
        elif part_type == "input_file":
            fields = inline_file_fields(part, pdf_only=True)
            if (
                not allow_attachments
                or fields is None
                or part.get("detail") not in (None, "auto")
            ):
                raise ResponsesConversionError(
                    "A Chat file must be an inline PDF with automatic detail in user content"
                )
            file: dict[str, object] = {"type": "file", "file": fields}
            if "prompt_cache_breakpoint" in part:
                file["prompt_cache_breakpoint"] = part["prompt_cache_breakpoint"]
            parts.append(file)

    if not parts:
        return ""
    if all(part.get("type") == "text" for part in parts):
        return "\n\n".join(str(part.get("text", "")) for part in parts)
    return parts


def _content_text(value: JsonValue) -> str:
    content = _message_content(value, allow_attachments=False)
    if isinstance(content, str):
        return content
    return "\n\n".join(
        str(part.get("text", "")) for part in content if isinstance(part, Mapping)
    )


def _text_from_part(part: Mapping[str, JsonValue]) -> str:
    for key in ("text", "input_text", "output_text", "refusal"):
        value = part.get(key)
        if isinstance(value, str):
            return value
    return ""


def _image_part(part: Mapping[str, JsonValue], *, context: str) -> dict[str, object]:
    source = part.get("image_url")
    if isinstance(source, str) and source:
        image_url: dict[str, object] = {"url": source}
    elif isinstance(source, Mapping):
        url = source.get("url")
        if not isinstance(url, str) or not url:
            raise ResponsesConversionError(
                f"{context}.image_url requires a non-empty URL"
            )
        image_url = {"url": url}
        source_detail = source.get("detail")
        if isinstance(source_detail, str):
            image_url["detail"] = source_detail
    else:
        if part.get("file_id") is not None:
            raise ResponsesConversionError(
                f"{context}.file_id cannot be represented in Chat Completions"
            )
        raise ResponsesConversionError(f"{context}.image_url requires a non-empty URL")
    detail = part.get("detail")
    if isinstance(detail, str):
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _arguments_text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, separators=(",", ":"))


def _tool_output_text(value: JsonValue) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _rich_tool_output_parts(
    value: JsonValue, *, call_id: str
) -> list[dict[str, object]] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return None
    if not any(
        isinstance(part, Mapping) and part.get("type") in ("input_image", "input_file")
        for part in value
    ):
        return None

    parts = _message_content(value, allow_attachments=True)
    assert isinstance(parts, list)
    return [{"type": "text", "text": tool_result_attachment_label(call_id)}, *parts]


def _apply_reasoning_text(
    message: dict[str, object], text: str, mode: ReasoningReplayMode
) -> None:
    field = (
        "reasoning" if mode is ReasoningReplayMode.REASONING else "reasoning_content"
    )
    previous = message.get(field)
    message[field] = combine_reasoning(
        previous if isinstance(previous, str) else None, text
    )


def _chat_tools(
    tools: list[JsonObject] | None,
) -> tuple[list[dict[str, object]], frozenset[str]]:
    converted: list[dict[str, object]] = []
    names: set[str] = set()
    for tool in tools or ():
        tool_type = tool.get("type")
        if tool_type == "function":
            converted_tool, name = _chat_function_tool(tool)
            _append_unique_chat_tool(converted, names, converted_tool, name)
    return converted, frozenset(names)


def _append_unique_chat_tool(
    converted: list[dict[str, object]],
    names: set[str],
    tool: dict[str, object],
    name: str,
) -> None:
    if name in names:
        raise ResponsesConversionError(
            f"Responses tools map to the same Chat-compatible name {name!r}"
        )
    converted.append(tool)
    names.add(name)


def _chat_function_tool(tool: Mapping[str, JsonValue]) -> tuple[dict[str, object], str]:
    name = required_str(tool.get("name"), "tool.name")
    parameters = tool.get("parameters")
    if parameters is None:
        parameters = {"type": "object", "properties": dict[str, JsonValue]()}
    if not isinstance(parameters, Mapping):
        raise ResponsesConversionError(
            f"Responses tool {name!r} parameters must be an object"
        )
    function: dict[str, object] = {
        "name": name,
        "parameters": dict(parameters),
    }
    if description := optional_str(tool.get("description")):
        function["description"] = description
    strict = tool.get("strict")
    if isinstance(strict, bool):
        function["strict"] = strict
    return {"type": "function", "function": function}, name


def _chat_tool_choice(
    value: JsonValue, available_names: frozenset[str]
) -> object | None:
    if value is None or value == "auto":
        return "auto" if available_names else None
    if value == "required":
        if not available_names:
            raise ResponsesConversionError(
                "Required tool choice has no available functions"
            )
        return "required"
    if value == "none":
        return None
    if not isinstance(value, Mapping):
        raise ResponsesConversionError("Unsupported tool choice")
    choice_type = value.get("type")
    if choice_type in {"auto", "any", "required"}:
        return _chat_tool_choice(
            "auto" if choice_type == "auto" else "required", available_names
        )
    if choice_type != "function":
        raise ResponsesConversionError(
            "Named tool choice requires an available function"
        )
    name = optional_str(value.get("name"))
    if not name or name not in available_names:
        raise ResponsesConversionError(
            "Named tool choice requires an available function"
        )
    return {"type": "function", "function": {"name": name}}


def _chat_response_format(value: JsonValue) -> JsonObject | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ResponsesConversionError("Responses text controls require an object")
    format_value = value.get("format")
    if format_value is None:
        return None
    if not isinstance(format_value, Mapping):
        raise ResponsesConversionError("Output format requires an object")
    format_type = format_value.get("type")
    if format_type in {"text", "json_object"}:
        return {"type": format_type}
    if format_type != "json_schema":
        raise ResponsesConversionError(
            "Chat cannot enforce the requested output format"
        )
    if not isinstance(format_value.get("schema"), dict):
        raise ResponsesConversionError("Output format requires a JSON schema object")
    json_schema = {
        key: format_value[key]
        for key in ("name", "description", "schema", "strict")
        if key in format_value
    }
    return {"type": "json_schema", "json_schema": json_schema}


def _body_tool_names(body: Mapping[str, object]) -> list[str]:
    names: list[str] = []
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            function = tool.get("function")
            if isinstance(function, Mapping) and isinstance(function.get("name"), str):
                names.append(function["name"])
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                if isinstance(function, Mapping) and isinstance(
                    function.get("name"), str
                ):
                    names.append(function["name"])
    choice = body.get("tool_choice")
    if isinstance(choice, Mapping):
        function = choice.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return names


def _body_tool_call_ids(body: Mapping[str, object]) -> list[str]:
    call_ids: list[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return call_ids
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        call_ids.extend(
            call["id"]
            for call in calls
            if isinstance(call, Mapping) and isinstance(call.get("id"), str)
        )
    return call_ids

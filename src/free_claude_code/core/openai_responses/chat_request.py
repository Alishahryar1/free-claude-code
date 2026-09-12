"""Direct OpenAI Responses-to-Chat Completions request translation."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.history_replay import (
    has_readable_replay,
    is_replay,
    readable_reasoning,
    reasoning_context,
    reasoning_detail,
    tool_history_context,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_chat import (
    IMAGE_TOOL_RESULT_MARKER,
    ChatToolResultImages,
    close_chat_tool_result_turns,
    computer_screenshot_label,
    image_tool_result_label,
)
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.trace import trace_event

from .errors import ResponsesConversionError
from .models import OpenAIResponsesRequest
from .reasoning import (
    combine_reasoning,
    encrypted_reasoning_from_item,
)
from .tool_adaptation import ResponsesToolAdapter, ResponsesToolPolicy
from .tools import (
    call_id_from_item,
    optional_str,
    parse_arguments,
    required_str,
)

_CHAT_OPTION_FIELDS = (
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "n",
    "presence_penalty",
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
    tool_schemas: dict[str, JsonObject]
    reserved_tool_ids: frozenset[str]
    tool_adapter: ResponsesToolAdapter


@dataclass(slots=True)
class _PendingReasoning:
    text: str | None = None
    encrypted: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)

    def add(self, item: Mapping[str, JsonValue]) -> None:
        if encrypted := encrypted_reasoning_from_item(item):
            self.encrypted.append(encrypted)
            if has_readable_replay(encrypted):
                return
        for text, summary in readable_reasoning(item):
            if summary:
                self.contexts.append(reasoning_context(text, summary=True))
            else:
                self.text = combine_reasoning(self.text, text)

    @property
    def empty(self) -> bool:
        return self.text is None and not self.encrypted and not self.contexts

    def take(self) -> tuple[str | None, list[str], list[str]]:
        text = self.text
        encrypted = list(self.encrypted)
        self.text = None
        self.encrypted.clear()
        contexts = list(self.contexts)
        self.contexts.clear()
        return text, encrypted, contexts


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
        self._quarantined_call_ids: set[str] = set()
        self._discovery_outputs: list[dict[str, object]] = []

    def add(self, item: JsonValue, *, source_type: str | None = None) -> None:
        if isinstance(item, str):
            self._flush_rich_outputs()
            self._flush_reasoning()
            self.messages.append({"role": "user", "content": item})
            return
        if not isinstance(item, Mapping):
            self._flush_rich_outputs()
            return

        item_type = item.get("type")
        if item_type not in {
            "function_call_output",
            "custom_tool_call_output",
            "computer_call_output",
        }:
            self._flush_rich_outputs()
        if item_type in (None, "message") or "role" in item:
            self._add_message(item)
            return
        if item_type == "reasoning":
            self._pending_reasoning.add(item)
            return
        if item_type == "function_call":
            self._add_tool_call(item)
            return
        if item_type == "function_call_output":
            self._add_tool_output(item, source_type=source_type)
            return
        if item_type == "computer_call_output":
            self._add_computer_output(item)
            return
        if item_type in {"input_text", "output_text", "text"}:
            self._flush_reasoning()
            self.messages.append({"role": "user", "content": _text_from_part(item)})
            return
        if item_type == "input_image":
            image = _image_part(item, context="input_image")
            self._flush_reasoning()
            self.messages.append({"role": "user", "content": [image]})
            return
        if isinstance(item_type, str) and item_type.endswith(("_call", "_result")):
            if item.get("status") not in {
                None,
                "completed",
                "failed",
                "incomplete",
                "interrupted",
            }:
                raise ResponsesConversionError(
                    "An active hosted tool cannot continue through Chat Completions."
                )
            self._flush_reasoning()
            self.messages.append(
                {"role": "assistant", "content": tool_history_context(item)}
            )

    def finish(self) -> tuple[list[str], list[dict[str, object]]]:
        self._flush_rich_outputs()
        self._flush_reasoning()
        return self.system_parts, self.messages

    def encode_discovery_outputs(self, tool_names: OpenAIToolNameCodec) -> None:
        """Encode FCC's discovery definitions with the final Chat name mapping."""
        if not tool_names.has_aliases:
            return
        for message in self._discovery_outputs:
            tools = json.loads(cast(str, message["content"]))
            for tool in tools:
                tool["name"] = tool_names.encode(tool["name"])
            message["content"] = json.dumps(tools)

    def _add_message(self, item: Mapping[str, JsonValue]) -> None:
        role = required_str(item.get("role", "user"), "input.role")
        if role in {"developer", "system"}:
            text = _content_text(item.get("content"))
            if text:
                self.system_parts.append(text)
            return
        if role not in {"user", "assistant"}:
            raise ResponsesConversionError(
                f"Unsupported Responses message role: {role!r}"
            )

        if role == "user":
            self._flush_reasoning()
        content = _message_content(item.get("content"), allow_images=role == "user")
        message: dict[str, object] = {"role": role, "content": content}
        if role == "assistant":
            self._apply_pending_reasoning(message)
        if not content and len(message) == 2:
            return
        self.messages.append(message)

    def _add_tool_call(self, item: Mapping[str, JsonValue]) -> None:
        call_id = call_id_from_item(item)
        name = required_str(item.get("name"), "function_call.name")
        raw_arguments = item.get("arguments")
        try:
            parse_arguments(raw_arguments)
        except ResponsesConversionError as exc:
            self._quarantined_call_ids.add(call_id)
            self._pending_reasoning.take()
            trace_event(
                stage="responses",
                event="responses.input.function_call_quarantined",
                source="openai_responses",
                call_id=call_id,
                error_type=type(exc).__name__,
            )
            return
        arguments = _arguments_text(raw_arguments)

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
        if self._reasoning_replay is ReasoningReplayMode.REASONING_CONTENT:
            message.setdefault("reasoning_content", "")

    def _add_tool_output(
        self, item: Mapping[str, JsonValue], *, source_type: str | None
    ) -> None:
        function = source_type != "custom_tool_call_output"
        call_id = call_id_from_item(item)
        if function and call_id in self._quarantined_call_ids:
            return
        if not self._pending_reasoning.empty:
            previous = self._last_tool_call_message()
            if previous is not None:
                self._apply_pending_reasoning(previous)
            else:
                self._flush_reasoning()
        output = item.get("output")
        rich_parts = (
            _rich_function_output_parts(output, call_id=call_id) if function else None
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": (
                    IMAGE_TOOL_RESULT_MARKER
                    if rich_parts is not None
                    else _tool_output_text(output)
                ),
            }
        )
        if source_type == "tool_search_output":
            self._discovery_outputs.append(self.messages[-1])
        if rich_parts is not None:
            self._pending_rich_output_parts.extend(rich_parts)

    def _add_computer_output(self, item: Mapping[str, JsonValue]) -> None:
        call_id = call_id_from_item(item)
        output = item.get("output")
        if not isinstance(output, Mapping) or output.get("type") != (
            "computer_screenshot"
        ):
            raise ResponsesConversionError(
                "computer_call_output.output must be a computer_screenshot object"
            )
        image = _image_part(output, context="computer_call_output.output")
        self._flush_reasoning()
        self._pending_rich_output_parts.extend(
            [
                {"type": "text", "text": computer_screenshot_label(call_id)},
                image,
            ]
        )

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
                ChatToolResultImages(
                    role="user",
                    content=cast(
                        list[JsonValue], list(self._pending_rich_output_parts)
                    ),
                ),
            )
        )
        self._pending_rich_output_parts.clear()

    def _apply_pending_reasoning(self, message: dict[str, object]) -> None:
        text, encrypted, contexts = self._pending_reasoning.take()
        if contexts:
            message["content"] = "\n\n".join(
                [*contexts, str(message.get("content") or "")]
            ).rstrip()
        if text is not None:
            _apply_reasoning_text(message, text, self._reasoning_replay)
        if encrypted:
            details = message.setdefault("reasoning_details", [])
            if isinstance(details, list):
                for value in encrypted:
                    if self._structured_reasoning_details or is_replay(value):
                        details.extend(reasoning_detail(value))
            if not details:
                message.pop("reasoning_details", None)


def build_responses_chat_request(
    request: OpenAIResponsesRequest,
    *,
    reasoning_replay: ReasoningReplayMode,
    structured_reasoning_details: bool = False,
) -> ResponsesChatRequest:
    """Translate a Responses request directly into one Chat Completions body."""
    adapter = ResponsesToolAdapter(
        request,
        ResponsesToolPolicy(
            custom_tools_as_functions=True,
            flatten_namespaces=True,
            client_tool_search=True,
        ),
    )
    request = adapter.request
    builder = _ResponsesChatInputBuilder(
        reasoning_replay=reasoning_replay,
        structured_reasoning_details=structured_reasoning_details,
    )
    if request.instructions:
        builder.system_parts.append(request.instructions)
    original_items = _input_items(adapter.original.input)
    for index, item in enumerate(_input_items(request.input)):
        source = original_items[index]
        builder.add(
            item,
            source_type=optional_str(source.get("type"))
            if isinstance(source, dict)
            else None,
        )
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
    if request.tool_choice != "none" and tools:
        body["tools"] = tools
        choice = _chat_tool_choice(request.tool_choice, available_tool_names)
        if choice is not None:
            body["tool_choice"] = choice

    if request.parallel_tool_calls is not None and tools:
        body["parallel_tool_calls"] = request.parallel_tool_calls
    if request.max_output_tokens is not None:
        body["max_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.metadata is not None:
        body["metadata"] = request.metadata

    extra = request.model_extra or {}
    for field_name in _CHAT_OPTION_FIELDS:
        value = extra.get(field_name)
        if value is not None:
            body[field_name] = value
    if response_format := _chat_response_format(extra.get("text")):
        body["response_format"] = response_format

    tool_schemas = _body_tool_schemas(body)
    reserved_tool_ids = frozenset(_body_tool_call_ids(body))
    tool_names = OpenAIToolNameCodec.from_names(_body_tool_names(body))
    builder.encode_discovery_outputs(tool_names)
    return ResponsesChatRequest(
        body=body,
        tool_names=tool_names,
        tool_schemas=tool_schemas,
        reserved_tool_ids=reserved_tool_ids,
        tool_adapter=adapter,
    )


def _input_items(value: JsonValue) -> Sequence[JsonValue]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return (value,)


def _message_content(
    value: JsonValue, *, allow_images: bool
) -> str | list[dict[str, object]]:
    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray):
        if isinstance(value, Mapping):
            return _text_from_part(value)
        return ""

    parts: list[dict[str, object]] = []
    for part in value:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, Mapping):
            continue
        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text", "refusal"} or (
            "text" in part
        ):
            parts.append({"type": "text", "text": _text_from_part(part)})
            continue
        if allow_images and part_type == "input_image":
            parts.append(_image_part(part, context="input_image"))

    if not parts:
        return ""
    if all(part.get("type") == "text" for part in parts):
        return "\n\n".join(str(part.get("text", "")) for part in parts)
    return parts


def _content_text(value: JsonValue) -> str:
    content = _message_content(value, allow_images=False)
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


def _rich_function_output_parts(
    value: JsonValue, *, call_id: str
) -> list[dict[str, object]] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return None
    if not any(
        isinstance(part, Mapping) and part.get("type") == "input_image"
        for part in value
    ):
        return None

    result: list[dict[str, object]] = [
        {"type": "text", "text": image_tool_result_label(call_id)}
    ]
    for part in value:
        if isinstance(part, Mapping) and part.get("type") == "input_text":
            result.append({"type": "text", "text": _text_from_part(part)})
        elif isinstance(part, Mapping) and part.get("type") == "input_image":
            result.append(
                _image_part(part, context="function_call_output.output.input_image")
            )
        else:
            result.append({"type": "text", "text": _tool_output_text(part)})
    return result


def _apply_reasoning_text(
    message: dict[str, object], text: str, mode: ReasoningReplayMode
) -> None:
    if mode in {ReasoningReplayMode.REASONING_CONTENT, ReasoningReplayMode.REASONING}:
        previous = message.get(mode.value)
        message[mode.value] = combine_reasoning(
            previous if isinstance(previous, str) else None, text
        )
        return
    replay = (
        f"<think>\n{text}\n</think>"
        if mode is ReasoningReplayMode.THINK_TAGS
        else reasoning_context(text)
    )
    if not replay:
        return
    content = message.get("content")
    if isinstance(content, str) and content:
        message["content"] = f"{replay}\n\n{content}"
    else:
        message["content"] = replay


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
        parameters = {"type": "object", "properties": {}}
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
    if not available_names:
        return None
    if value is None or value == "auto":
        return "auto"
    if value == "required":
        return "required"
    if value == "none":
        return None
    if not isinstance(value, Mapping):
        return None
    choice_type = value.get("type")
    if choice_type in {"auto", "any", "required"}:
        return "required" if choice_type in {"any", "required"} else "auto"
    if choice_type != "function":
        return None
    name = optional_str(value.get("name"))
    if not name:
        return None
    if name not in available_names:
        return None
    return {"type": "function", "function": {"name": name}}


def _chat_response_format(value: JsonValue) -> object | None:
    if not isinstance(value, Mapping):
        return None
    format_value = value.get("format")
    if not isinstance(format_value, Mapping):
        return None
    format_type = format_value.get("type")
    if format_type in {"text", "json_object"}:
        return {"type": format_type}
    if format_type != "json_schema":
        return None
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


def _body_tool_schemas(body: Mapping[str, object]) -> dict[str, JsonObject]:
    schemas: dict[str, JsonObject] = {}
    tools = body.get("tools")
    if not isinstance(tools, list):
        return schemas
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if isinstance(name, str) and isinstance(parameters, Mapping):
            schemas[name] = dict(parameters)
    return schemas


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

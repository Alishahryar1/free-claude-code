"""Direct Responses-to-Messages conversion with explicit replay and tool identity."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast
from urllib.parse import urlsplit

from free_claude_code.core.anthropic.image_sources import (
    AnthropicImageSourceError,
    portable_anthropic_image_url,
)
from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    NativeMessagesOptions,
    apply_messages_options,
    validate_messages_json,
)
from free_claude_code.core.history_replay import (
    responses_reasoning_blocks,
    tool_history_context,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec

from .errors import ResponsesConversionError
from .models import OpenAIResponsesRequest
from .tools import (
    ResponsesToolIdentity,
    custom_tool_input_schema,
    flatten_responses_tool_name,
)

_REQUEST_FIELDS = {
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "stream",
    "temperature",
    "top_p",
    "max_output_tokens",
    "metadata",
    "reasoning",
    "previous_response_id",
    "store",
    "text",
    "include",
    "truncation",
}


@dataclass(frozen=True, slots=True)
class ResponsesMessagesRequest:
    body: JsonObject
    tool_identities: Mapping[str, ResponsesToolIdentity]


def _fields(value: Mapping[str, JsonValue], allowed: set[str], context: str) -> None:
    unsupported = sorted(
        key for key, item in value.items() if key not in allowed and item is not None
    )
    if unsupported:
        raise ResponsesConversionError(
            f"{context} cannot represent field(s): {', '.join(unsupported)}."
        )


def _string(value: JsonValue, context: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ResponsesConversionError(
            f"{context} must be a {'string' if empty else 'nonempty string'}."
        )
    return value


def _items(value: JsonValue) -> list[JsonValue]:
    if isinstance(value, str | Mapping):
        return [value]
    if isinstance(value, list):
        return value
    raise ResponsesConversionError(
        "Messages upstream requires explicit Responses input history."
    )


def _identity(
    value: Mapping[str, JsonValue],
    kind: Literal["function", "custom"],
    namespace: str | None = None,
) -> ResponsesToolIdentity:
    own_namespace = value.get("namespace")
    if own_namespace is not None:
        own_namespace = _string(own_namespace, "tool.namespace")
        if namespace is not None and namespace != own_namespace:
            raise ResponsesConversionError(
                "Tool namespace conflicts with its declaration."
            )
        namespace = own_namespace
    return ResponsesToolIdentity(
        kind=kind, name=_string(value.get("name"), "tool.name"), namespace=namespace
    )


def _arguments(value: JsonValue) -> JsonObject:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise ResponsesConversionError("Function arguments must be a JSON object.")
    try:
        parsed = cast(JsonValue, json.loads(value))
    except (ValueError, RecursionError) as exc:
        raise ResponsesConversionError(
            "Function arguments must be valid JSON."
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ResponsesConversionError("Function arguments must decode to an object.")
    return dict(parsed)


class _ToolScope:
    def __init__(self, tools: list[JsonObject] | None, items: list[JsonValue]) -> None:
        self._flat: dict[str, ResponsesToolIdentity] = {}
        self._declared: set[ResponsesToolIdentity] = set()
        definitions: list[tuple[ResponsesToolIdentity, JsonObject]] = []
        for tool in tools or ():
            if tool.get("type") == "namespace":
                _fields(tool, {"type", "name", "description", "tools"}, "Namespace")
                namespace = _string(tool.get("name"), "namespace.name")
                nested = tool.get("tools")
                if not isinstance(nested, list):
                    raise ResponsesConversionError("Namespace tools must be a list.")
                description = tool.get("description")
                if description is not None:
                    description = _string(
                        description, "namespace.description", empty=True
                    )
                for child in nested:
                    if not isinstance(child, Mapping):
                        raise ResponsesConversionError(
                            "Namespace tool must be an object."
                        )
                    identity, body = self._definition(child, namespace)
                    if description:
                        own = body.get("description", "")
                        body["description"] = (
                            f"Namespace {namespace}: {description}\n\n{own}".rstrip()
                        )
                    definitions.append((identity, body))
            else:
                definitions.append(self._definition(tool, None))
        for item in items:
            if isinstance(item, Mapping) and item.get("type") in (
                "function_call",
                "custom_tool_call",
            ):
                kind = (
                    "custom" if item.get("type") == "custom_tool_call" else "function"
                )
                self._register(_identity(item, kind))
        self._codec = OpenAIToolNameCodec.from_names(self._flat)
        self.identities = MappingProxyType(
            {
                self._codec.encode(name): identity
                for name, identity in self._flat.items()
            }
        )
        self.tools: list[JsonValue] = []
        for identity, body in definitions:
            body["name"] = self.alias(identity)
            self.tools.append(body)

    def _register(self, identity: ResponsesToolIdentity) -> None:
        name = flatten_responses_tool_name(identity.name, namespace=identity.namespace)
        previous = self._flat.get(name)
        if previous is not None and previous != identity:
            raise ResponsesConversionError(
                "Tool identities collide after namespace flattening."
            )
        self._flat[name] = identity

    def _definition(
        self, value: Mapping[str, JsonValue], namespace: str | None
    ) -> tuple[ResponsesToolIdentity, JsonObject]:
        kind_value = value.get("type")
        if kind_value not in ("function", "custom"):
            raise ResponsesConversionError(
                f"Messages upstream cannot represent tool type {kind_value!r}."
            )
        kind: Literal["function", "custom"] = (
            "custom" if kind_value == "custom" else "function"
        )
        allowed = {"type", "name", "namespace", "description"}
        allowed |= {"format"} if kind == "custom" else {"parameters", "strict"}
        _fields(value, allowed, "Tool declaration")
        identity = _identity(value, kind, namespace)
        if identity in self._declared:
            raise ResponsesConversionError("Duplicate tool declaration.")
        self._declared.add(identity)
        self._register(identity)
        body: JsonObject = {}
        if value.get("description") is not None:
            body["description"] = _string(
                value["description"], "tool.description", empty=True
            )
        if kind == "custom":
            format_value = value.get("format")
            if format_value is not None:
                if (
                    not isinstance(format_value, Mapping)
                    or format_value.get("type") != "text"
                ):
                    raise ResponsesConversionError(
                        "Messages upstream cannot enforce custom tool grammars."
                    )
                _fields(format_value, {"type"}, "Custom tool format")
            body["input_schema"] = {
                **custom_tool_input_schema(description=None),
                "additionalProperties": False,
            }
        else:
            schema = value.get("parameters", {"type": "object", "properties": {}})
            if not isinstance(schema, Mapping):
                raise ResponsesConversionError(
                    "Function parameters must be a schema object."
                )
            body["input_schema"] = dict(schema)
            if value.get("strict") is not None:
                strict = value["strict"]
                if not isinstance(strict, bool):
                    raise ResponsesConversionError("Tool strict must be a boolean.")
                body["strict"] = strict
        return identity, body

    def alias(self, identity: ResponsesToolIdentity) -> str:
        name = flatten_responses_tool_name(identity.name, namespace=identity.namespace)
        if self._flat.get(name) != identity:
            raise ResponsesConversionError("Unknown tool identity.")
        return self._codec.encode(name)

    def choice(self, value: JsonValue, parallel: bool | None) -> JsonObject | None:
        choice: JsonObject | None = None
        if value is not None:
            if isinstance(value, str) and value in {"auto", "none", "required"}:
                choice = {"type": "any" if value == "required" else value}
            elif isinstance(value, Mapping):
                kind = value.get("type")
                if kind not in ("function", "custom"):
                    raise ResponsesConversionError("Unsupported named tool choice.")
                _fields(value, {"type", "name", "namespace"}, "Tool choice")
                identity = _identity(
                    value, "custom" if kind == "custom" else "function"
                )
                if identity not in self._declared:
                    raise ResponsesConversionError(
                        "Named tool choice must refer to a declared tool."
                    )
                choice = {"type": "tool", "name": self.alias(identity)}
            else:
                raise ResponsesConversionError("Unsupported Responses tool choice.")
        if choice is not None and choice["type"] in {"any", "tool"} and not self.tools:
            raise ResponsesConversionError(
                "Forced tool choice requires declared tools."
            )
        if (
            parallel is False
            and self.tools
            and (choice is None or choice["type"] != "none")
        ):
            choice = choice or {"type": "auto"}
            choice["disable_parallel_tool_use"] = True
        return choice


def _image(value: Mapping[str, JsonValue]) -> JsonObject:
    _fields(value, {"type", "image_url", "detail", "file_id"}, "Image")
    if value.get("file_id") is not None:
        raise ResponsesConversionError(
            "A Messages upstream requires image content or a portable URL, not file_id."
        )
    if value.get("detail") not in (None, "auto"):
        raise ResponsesConversionError(
            "Messages upstream cannot represent the requested image detail control."
        )
    url = _string(value.get("image_url"), "input_image.image_url")
    if url.lower().startswith("data:"):
        prefix, separator, data = url.partition(";base64,")
        if not separator:
            raise ResponsesConversionError(
                "Image data URL must contain base64 content."
            )
        source: JsonObject = {"type": "base64", "media_type": prefix[5:], "data": data}
        try:
            normalized = portable_anthropic_image_url(source)
        except AnthropicImageSourceError as exc:
            raise ResponsesConversionError(str(exc)) from exc
        source["data"] = normalized.partition(";base64,")[2]
        source["media_type"] = prefix[5:].lower()
    else:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ResponsesConversionError("Image URL must use HTTP or HTTPS.")
        source = {"type": "url", "url": url}
    return {"type": "image", "source": source}


def _content(value: JsonValue, *, images: bool, empty: bool = False) -> list[JsonValue]:
    if isinstance(value, str):
        text = _string(value, "content", empty=empty)
        return [{"type": "text", "text": text}] if text else []
    if not isinstance(value, list):
        raise ResponsesConversionError(
            "Message content must be text or a list of supported blocks."
        )
    blocks: list[JsonValue] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ResponsesConversionError("Message content blocks must be objects.")
        kind = item.get("type")
        if kind in ("input_text", "output_text", "text"):
            _fields(item, {"type", "text", "annotations", "logprobs"}, "Text block")
            if item.get("annotations") or item.get("logprobs"):
                raise ResponsesConversionError(
                    "Messages history cannot represent text annotations or logprobs."
                )
            text = _string(item.get("text"), "content.text", empty=empty)
            if text:
                blocks.append({"type": "text", "text": text})
        elif kind == "input_image" and images:
            blocks.append(_image(item))
        else:
            raise ResponsesConversionError(
                f"Messages upstream cannot represent content type {kind!r}."
            )
    return blocks


class _MessagesInput:
    def __init__(self, tools: _ToolScope) -> None:
        self.messages: list[JsonObject] = []
        self.system: list[JsonValue] = []
        self._tools = tools
        self._calls: dict[str, Literal["function", "custom"]] = {}
        self._pending: set[str] = set()

    def _append(
        self,
        role: Literal["user", "assistant"],
        parts: list[JsonValue],
        *,
        result: bool = False,
    ) -> None:
        if (
            self._pending
            and not result
            and (
                role == "user"
                or not self.messages
                or self.messages[-1]["role"] != "assistant"
            )
        ):
            raise ResponsesConversionError(
                "Tool results must immediately follow their assistant call group."
            )
        if self.messages and self.messages[-1]["role"] == role:
            content = self.messages[-1]["content"]
            if not isinstance(content, list):
                raise AssertionError("Native content must be a list.")
            content.extend(parts)
        else:
            self.messages.append({"role": role, "content": parts})

    def add(self, value: JsonValue) -> None:
        if isinstance(value, str):
            self._append("user", [{"type": "text", "text": value}])
            return
        if not isinstance(value, Mapping):
            raise ResponsesConversionError(
                "Responses input items must be text or objects."
            )
        kind = value.get("type")
        if kind is None or kind == "message":
            _fields(value, {"type", "role", "content", "id", "status"}, "Message")
            role = value.get("role", "user")
            if role in ("system", "developer"):
                if self.messages:
                    raise ResponsesConversionError(
                        "Messages upstream supports only leading system/developer input."
                    )
                self.system.extend(_content(value.get("content"), images=False))
            elif role in ("user", "assistant"):
                self._append(
                    "user" if role == "user" else "assistant",
                    _content(value.get("content"), images=role == "user"),
                )
            else:
                raise ResponsesConversionError("Unsupported Responses message role.")
            return
        if kind == "reasoning":
            if blocks := responses_reasoning_blocks(value):
                self._append("assistant", blocks)
            return
        if kind in ("function_call", "custom_tool_call"):
            self._call(value, custom=kind == "custom_tool_call")
            return
        if kind in ("function_call_output", "custom_tool_call_output"):
            self._result(value, custom=kind == "custom_tool_call_output")
            return
        if kind in ("input_text", "output_text", "text", "input_image"):
            self._append("user", _content([value], images=True))
            return
        if (
            isinstance(kind, str)
            and kind.endswith(("_call", "_result"))
            and value.get("status")
            in {None, "completed", "failed", "incomplete", "interrupted"}
        ):
            self._append(
                "assistant", [{"type": "text", "text": tool_history_context(value)}]
            )
            return
        raise ResponsesConversionError(
            f"Messages upstream cannot represent input item {kind!r}."
        )

    def _call(self, value: Mapping[str, JsonValue], *, custom: bool) -> None:
        _fields(
            value,
            {
                "type",
                "id",
                "status",
                "call_id",
                "name",
                "namespace",
                "input" if custom else "arguments",
            },
            "Tool call",
        )
        call_id = _string(value.get("call_id"), "tool.call_id")
        if call_id in self._calls:
            raise ResponsesConversionError(
                "Duplicate tool call_id in Responses history."
            )
        kind = "custom" if custom else "function"
        identity = _identity(value, kind)
        arguments: JsonObject = (
            {"input": _string(value.get("input"), "custom_tool_call.input", empty=True)}
            if custom
            else _arguments(value.get("arguments"))
        )
        self._append(
            "assistant",
            [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": self._tools.alias(identity),
                    "input": arguments,
                }
            ],
        )
        self._calls[call_id] = kind
        self._pending.add(call_id)

    def _result(self, value: Mapping[str, JsonValue], *, custom: bool) -> None:
        _fields(
            value,
            {"type", "id", "status", "call_id", "output", "is_error"},
            "Tool output",
        )
        call_id = _string(value.get("call_id"), "tool output.call_id")
        if call_id not in self._pending:
            raise ResponsesConversionError("Tool output is orphaned or duplicated.")
        if self._calls[call_id] != ("custom" if custom else "function"):
            raise ResponsesConversionError("Tool output type does not match its call.")
        block: JsonObject = {
            "type": "tool_result",
            "tool_use_id": call_id,
        }
        content = _content(value.get("output"), images=True, empty=True)
        if content:
            block["content"] = content
        if value.get("is_error") is not None:
            if not isinstance(value["is_error"], bool):
                raise ResponsesConversionError(
                    "Tool result is_error must be a boolean."
                )
            block["is_error"] = value["is_error"]
        self._append("user", [block], result=True)
        self._pending.remove(call_id)

    def finish(self) -> None:
        if self._pending:
            raise ResponsesConversionError(
                "Responses history has unresolved tool calls."
            )
        if not self.messages:
            raise ResponsesConversionError(
                "Messages upstream requires conversational input."
            )


def _output_format(value: JsonValue) -> JsonObject | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ResponsesConversionError("Responses text options must be an object.")
    _fields(value, {"format"}, "Text options")
    format_value = value.get("format")
    if format_value is None:
        return None
    if not isinstance(format_value, Mapping):
        raise ResponsesConversionError("Responses text.format must be an object.")
    kind = format_value.get("type")
    if kind == "text":
        _fields(format_value, {"type"}, "Text format")
        return None
    if kind != "json_schema":
        raise ResponsesConversionError(
            "Messages upstream supports text or json_schema output."
        )
    _fields(
        format_value,
        {"type", "name", "description", "schema", "strict"},
        "Output format",
    )
    schema = format_value.get("schema")
    if not isinstance(schema, Mapping):
        raise ResponsesConversionError("Output format schema must be an object.")
    strict = format_value.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise ResponsesConversionError("Output format strict must be a boolean.")
    if strict is False:
        raise ResponsesConversionError(
            "Messages structured output always enforces its schema."
        )
    converted = dict(schema)
    # Carry descriptive schema metadata into its native JSON Schema locations.
    if format_value.get("name") is not None:
        name = _string(format_value["name"], "output format.name")
        if "title" in converted and converted["title"] != name:
            raise ResponsesConversionError(
                "Output format name conflicts with schema title."
            )
        converted["title"] = name
    if format_value.get("description") is not None:
        description = _string(
            format_value["description"], "output format.description", empty=True
        )
        if "description" in converted and converted["description"] != description:
            raise ResponsesConversionError(
                "Output description conflicts with its schema."
            )
        converted["description"] = description
    return {"type": "json_schema", "schema": converted}


def build_responses_messages_request(
    request: OpenAIResponsesRequest,
    *,
    options: NativeMessagesOptions,
) -> ResponsesMessagesRequest:
    """Convert caller-owned history directly to the native Messages wire shape."""

    raw = cast(JsonObject, request.model_dump(mode="json", exclude_none=True))
    _fields(raw, _REQUEST_FIELDS, "Responses request")
    if raw.get("truncation") not in (None, "disabled"):
        raise ResponsesConversionError(
            "Messages upstream does not support automatic input truncation."
        )
    include = raw.get("include")
    if include is not None and (
        not isinstance(include, list)
        or any(item != "reasoning.encrypted_content" for item in include)
    ):
        raise ResponsesConversionError(
            "Messages upstream cannot represent the requested include fields."
        )
    if request.reasoning is not None:
        _fields(request.reasoning, {"effort", "summary"}, "Reasoning controls")
        if request.reasoning.get("summary") not in (None, "auto"):
            raise ResponsesConversionError(
                "Messages upstream supports only automatic reasoning summaries."
            )
    items = _items(request.input)
    scope = _ToolScope(request.tools, items)
    builder = _MessagesInput(scope)
    if request.instructions is not None:
        builder.system.append({"type": "text", "text": request.instructions})
    for item in items:
        builder.add(item)
    builder.finish()
    body: JsonObject = {"messages": builder.messages}
    if builder.system:
        body["system"] = builder.system
    if scope.tools:
        body["tools"] = scope.tools
    choice = scope.choice(request.tool_choice, request.parallel_tool_calls)
    if choice is not None:
        body["tool_choice"] = choice
    for key in ("temperature", "top_p"):
        if key in raw:
            body[key] = raw[key]
    if request.metadata is not None:
        _fields(request.metadata, {"user_id"}, "Metadata")
        if "user_id" in request.metadata:
            body["metadata"] = {
                "user_id": _string(
                    request.metadata["user_id"], "metadata.user_id", empty=True
                )
            }
    output_format = _output_format(raw.get("text"))
    if output_format is not None:
        body["output_config"] = {"format": output_format}
    try:
        apply_messages_options(body, options)
        validate_messages_json(body)
    except NativeMessagesError as exc:
        raise ResponsesConversionError(str(exc)) from exc
    return ResponsesMessagesRequest(body=body, tool_identities=scope.identities)

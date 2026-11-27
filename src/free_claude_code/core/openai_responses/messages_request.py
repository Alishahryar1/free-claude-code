"""Direct Responses-to-Messages conversion with explicit replay and tool identity."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, assert_never, cast
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
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec

from .errors import ResponsesConversionError
from .input_history import (
    HistoryItem,
    InputCall,
    InputMessage,
    InputResult,
    ReasoningItem,
    Screenshot,
    read_history,
)
from .models import OpenAIResponsesRequest
from .tools import (
    ResponsesToolIdentity,
    custom_tool_input_schema,
    flatten_responses_tool_name,
)


@dataclass(frozen=True, slots=True)
class ResponsesMessagesRequest:
    body: JsonObject
    tool_identities: Mapping[str, ResponsesToolIdentity]


def _string(value: JsonValue, context: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ResponsesConversionError(
            f"{context} must be a {'string' if empty else 'nonempty string'}."
        )
    return value


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


class _ToolScope:
    def __init__(
        self, tools: list[JsonObject] | None, history: list[HistoryItem]
    ) -> None:
        self._flat: dict[str, ResponsesToolIdentity] = {}
        self._declared: set[ResponsesToolIdentity] = set()
        definitions: list[tuple[ResponsesToolIdentity, JsonObject]] = []
        for tool in tools or ():
            if tool.get("type") == "namespace":
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
                    if not isinstance(child.get("type"), str):
                        raise ResponsesConversionError("Namespace tool requires a type")
                    if child.get("type") not in ("function", "custom"):
                        continue
                    identity, body = self._definition(child, namespace)
                    if description:
                        own = body.get("description", "")
                        body["description"] = (
                            f"Namespace {namespace}: {description}\n\n{own}".rstrip()
                        )
                    definitions.append((identity, body))
            elif tool.get("type") in ("function", "custom"):
                definitions.append(self._definition(tool, None))
            elif not isinstance(tool.get("type"), str):
                raise ResponsesConversionError("Tool declaration requires a type")
        for item in history:
            if isinstance(item, InputCall):
                self._register(item.identity)
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
            if format_value is not None and (
                not isinstance(format_value, Mapping)
                or format_value.get("type") != "text"
            ):
                raise ResponsesConversionError(
                    "Messages upstream cannot enforce custom tool grammars."
                )
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
        return choice if self.tools else None


def _image(value: Mapping[str, JsonValue]) -> JsonObject:
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
            text = _string(item.get("text"), "content.text", empty=empty)
            if text:
                blocks.append({"type": "text", "text": text})
        elif kind == "input_image" and images:
            blocks.append(_image(item))
        elif kind == "input_file":
            raise ResponsesConversionError(
                "FCC cannot translate this Responses file to Messages"
            )
        elif not isinstance(kind, str):
            raise ResponsesConversionError("Content block requires a type.")
    return blocks


class _MessagesInput:
    def __init__(self, tools: _ToolScope, *, assistant_fragment: bool = False) -> None:
        self.messages: list[JsonObject] = []
        self.system: list[JsonValue] = []
        self._tools = tools
        self._calls: dict[str, Literal["function", "custom"]] = {}
        self._pending: set[str] = set()
        self._assistant_fragment = assistant_fragment

    def _append(
        self,
        role: Literal["user", "assistant"],
        parts: list[JsonValue],
        *,
        result: bool = False,
    ) -> None:
        if self._assistant_fragment and role == "assistant" and not parts:
            return
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

    def add(self, item: HistoryItem) -> None:
        if isinstance(item, InputMessage):
            role = item.role
            if role in ("system", "developer"):
                if self.messages:
                    raise ResponsesConversionError(
                        "Messages upstream supports only leading system/developer input."
                    )
                self.system.extend(_content(item.content, images=False))
            else:
                parts = _content(
                    item.content,
                    images=role == "user",
                    empty=self._assistant_fragment and role == "assistant",
                )
                if parts or self._assistant_fragment:
                    self._append("user" if role == "user" else "assistant", parts)
        elif isinstance(item, ReasoningItem):
            if blocks := responses_reasoning_blocks(item.data):
                self._append("assistant", blocks)
        elif isinstance(item, InputCall):
            self._call(item)
        elif isinstance(item, InputResult):
            self._result(item)
        elif isinstance(item, Screenshot):
            self._append("user", _content(item.content, images=True))
        else:
            assert_never(item)

    def _call(self, item: InputCall) -> None:
        call_id = item.call_id
        if call_id in self._calls:
            raise ResponsesConversionError(
                "Duplicate tool call_id in Responses history."
            )
        self._append(
            "assistant",
            [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": self._tools.alias(item.identity),
                    "input": item.object_arguments(),
                }
            ],
        )
        self._calls[call_id] = item.identity.kind
        self._pending.add(call_id)

    def _result(self, item: InputResult) -> None:
        call_id = item.call_id
        if call_id not in self._pending:
            raise ResponsesConversionError("Tool output is orphaned or duplicated.")
        if self._calls[call_id] != item.kind:
            raise ResponsesConversionError("Tool output type does not match its call.")
        block: JsonObject = {
            "type": "tool_result",
            "tool_use_id": call_id,
        }
        content = _content(item.output, images=True, empty=True)
        if content:
            block["content"] = content
        if item.is_error is not None:
            block["is_error"] = item.is_error
        self._append("user", [block], result=True)
        self._pending.remove(call_id)

    def finish(self) -> None:
        if self._pending:
            raise ResponsesConversionError(
                "Responses history has unresolved tool calls."
            )


def build_responses_messages_suffix(items: list[JsonValue]) -> list[JsonObject]:
    """Convert delivered assistant fragments, which may contain only a summary."""
    history = [item for _, item in read_history(items)]
    builder = _MessagesInput(_ToolScope(None, history), assistant_fragment=True)
    for item in history:
        builder.add(item)
    builder.finish()
    try:
        validate_messages_json({"messages": builder.messages})
    except NativeMessagesError as exc:
        raise ResponsesConversionError(str(exc)) from exc
    return builder.messages


def _output_format(value: JsonValue) -> JsonObject | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ResponsesConversionError("Responses text options must be an object.")
    format_value = value.get("format")
    if format_value is None:
        return None
    if not isinstance(format_value, Mapping):
        raise ResponsesConversionError("Responses text.format must be an object.")
    kind = format_value.get("type")
    if kind == "text":
        return None
    if kind != "json_schema":
        raise ResponsesConversionError(
            "Messages upstream supports text or json_schema output."
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

    request.require_inline_context()
    request = request.select_direct_tools()
    raw = cast(JsonObject, request.model_dump(mode="json", exclude_none=True))
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
    if request.reasoning is not None and request.reasoning.get("summary") not in (
        None,
        "auto",
    ):
        raise ResponsesConversionError(
            "Messages upstream supports only automatic reasoning summaries."
        )
    history = [item for _, item in read_history(request.input)]
    scope = _ToolScope(request.tools, history)
    builder = _MessagesInput(scope)
    if request.instructions is not None:
        builder.system.append({"type": "text", "text": request.instructions})
    for item in history:
        builder.add(item)
    builder.finish()
    if not builder.messages:
        raise ResponsesConversionError(
            "Messages upstream requires conversational input."
        )
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
    if request.metadata is not None and "user_id" in request.metadata:
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

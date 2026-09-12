"""Provider-selected adaptations of native Responses tool representations."""

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import ResponsesConversionError
from .models import OpenAIResponsesRequest
from .tool_search import (
    active_client_tools,
    is_client_search,
    normalize_tool_search,
    search_function_name,
)
from .tools import (
    ResponsesToolIdentity,
    custom_tool_description,
    custom_tool_input_schema,
    custom_tool_input_text,
    custom_tool_input_text_from_arguments,
    flatten_responses_tool_name,
    optional_str,
    required_str,
)


@dataclass(frozen=True, slots=True)
class ResponsesToolPolicy:
    custom_tools_as_functions: bool = False
    explicit_search_parameters: bool = False
    text_only_web_search: bool = False
    client_tool_search: bool = False
    flatten_namespaces: bool = False


@dataclass(frozen=True, slots=True)
class _DefinitionEdit:
    original: JsonObject
    adapted: JsonObject
    namespace: str | None
    scope: str | None


class ResponsesToolAdapter:
    """Prepare one request and retain only the information needed to undo edits."""

    def __init__(
        self, request: OpenAIResponsesRequest, policy: ResponsesToolPolicy
    ) -> None:
        self.original = request
        self.request = request.model_copy(deep=True)
        self._policy = policy
        self._identities: dict[tuple[str | None, str], ResponsesToolIdentity] = {}
        self._edits: list[_DefinitionEdit] = []
        self._search_name: str | None = None
        if policy == ResponsesToolPolicy():
            return
        source_items = [
            *(request.tools or []),
            *(request.input if isinstance(request.input, list) else []),
        ]
        if policy.client_tool_search and any(
            isinstance(item, dict) and is_client_search(item) for item in source_items
        ):
            self.request.tools = active_client_tools(self.request)
            self._search_name = search_function_name(self.request.tools)
        if self.request.tools:
            self.request.tools = cast(list[JsonObject], self._tools(self.request.tools))
        if isinstance(self.request.input, list):
            self.request.input = [self._input(item) for item in self.request.input]
        if (
            policy.custom_tools_as_functions
            or policy.client_tool_search
            or policy.flatten_namespaces
        ):
            self.request.tool_choice = self._choice(self.request.tool_choice)

    def _register(self, identity: ResponsesToolIdentity, wire_name: str) -> None:
        key = (
            None if self._policy.flatten_namespaces else identity.namespace,
            wire_name,
        )
        existing = self._identities.get(key)
        if existing is not None and existing != identity:
            raise ResponsesConversionError("Tool names collide after conversion.")
        self._identities[key] = identity

    def _tools(
        self,
        tools: JsonValue,
        namespace: str | None = None,
        scope: str | None = None,
        *,
        preserve_namespaces: bool = False,
    ) -> JsonValue:
        if not isinstance(tools, list):
            return tools
        result: list[JsonValue] = []
        for tool in tools:
            if (
                self._policy.flatten_namespaces
                and not preserve_namespaces
                and isinstance(tool, dict)
                and tool.get("type") == "namespace"
            ):
                children = self._tools(tool.get("tools"), _name(tool), scope)
                if isinstance(children, list):
                    result.extend(children)
            else:
                result.append(self._tool(tool, namespace, scope))
        return result

    def _tool(
        self, value: JsonValue, namespace: str | None, scope: str | None
    ) -> JsonValue:
        if not isinstance(value, dict):
            return value
        tool = value
        kind = tool.get("type")
        if (
            self._policy.client_tool_search
            and kind == "tool_search"
            and is_client_search(tool)
        ):
            if self._policy.explicit_search_parameters:
                tool = normalize_tool_search(tool)
            if not isinstance(tool.get("parameters"), dict):
                raise ResponsesConversionError(
                    "Client tool search requires an argument schema."
                )
            return {
                "type": "function",
                "name": self._search_name,
                "description": tool.get("description", "Find callable tools."),
                "parameters": deepcopy(tool["parameters"]),
                "strict": False,
            }
        if kind == "namespace":
            return {
                **tool,
                "tools": self._tools(tool.get("tools"), _name(tool), scope),
            }
        if (
            self._policy.custom_tools_as_functions or self._policy.flatten_namespaces
        ) and kind in {"custom", "function"}:
            nested = tool.get(str(kind))
            source = nested if isinstance(nested, dict) else tool
            name = _name(source)
            identity = ResponsesToolIdentity(
                kind="custom" if kind == "custom" else "function",
                name=name,
                namespace=namespace or _namespace(source),
            )
            wire_name = (
                flatten_responses_tool_name(name, namespace=identity.namespace)
                if kind == "custom" or self._policy.flatten_namespaces
                else name
            )
            self._register(identity, wire_name)
            if self._policy.flatten_namespaces:
                tool = {**tool, "name": wire_name}
                tool.pop("namespace", None)
            if kind == "custom":
                tool = {
                    **{key: child for key, child in tool.items() if key != "custom"},
                    **source,
                    "type": "function",
                    "name": wire_name,
                    "parameters": custom_tool_input_schema(),
                    "strict": False,
                }
                tool.pop("format", None)
                if description := custom_tool_description(source):
                    tool["description"] = description
        if self._policy.flatten_namespaces:
            tool.pop("namespace", None)
        if self._policy.explicit_search_parameters:
            tool = normalize_tool_search(tool)
        if self._policy.text_only_web_search and kind in {
            "web_search",
            "web_search_preview",
        }:
            content_types = tool.get("search_content_types")
            if content_types is not None:
                if not isinstance(content_types, list) or "text" not in content_types:
                    raise ResponsesConversionError(
                        "The selected provider supports text web search only."
                    )
                tool = {
                    key: child
                    for key, child in tool.items()
                    if key != "search_content_types"
                }
        if tool != value:
            self._edits.append(_DefinitionEdit(value, tool, namespace, scope))
        return tool

    def _input(self, item: JsonValue) -> JsonValue:
        if not isinstance(item, dict):
            return item
        kind = item.get("type")
        if self._policy.client_tool_search and is_client_search(item):
            common = {
                key: value
                for key, value in item.items()
                if key not in {"execution", "tools", "arguments"}
            }
            if kind == "tool_search_call":
                arguments = item.get("arguments")
                if not isinstance(arguments, dict):
                    raise ResponsesConversionError(
                        "Client search arguments must be an object."
                    )
                return {
                    **common,
                    "type": "function_call",
                    "name": self._search_name,
                    "arguments": json.dumps(arguments),
                }
            if kind == "tool_search_output":
                active = active_client_tools(
                    OpenAIResponsesRequest(model=self.request.model, input=[item])
                )
                return {
                    **common,
                    "type": "function_call_output",
                    "output": json.dumps(self._tools(active, scope=_scope(item))),
                }
        if kind == "tool_search_output":
            return {
                **item,
                "tools": self._tools(
                    item.get("tools"), scope=_scope(item), preserve_namespaces=True
                ),
            }
        if (
            not self._policy.custom_tools_as_functions
            and not self._policy.flatten_namespaces
        ):
            return item
        if kind in {"custom_tool_call", "function_call"}:
            name = _name(item)
            identity = ResponsesToolIdentity(
                kind="custom" if kind == "custom_tool_call" else "function",
                name=name,
                namespace=_namespace(item),
            )
            wire_name = (
                flatten_responses_tool_name(name, namespace=identity.namespace)
                if kind == "custom_tool_call" or self._policy.flatten_namespaces
                else name
            )
            self._register(identity, wire_name)
            if kind == "custom_tool_call":
                return {
                    **{key: value for key, value in item.items() if key != "input"},
                    "type": "function_call",
                    "name": wire_name,
                    "arguments": json.dumps(
                        {"input": custom_tool_input_text(item.get("input"))},
                        ensure_ascii=False,
                    ),
                }
            if self._policy.flatten_namespaces:
                return {
                    **{key: value for key, value in item.items() if key != "namespace"},
                    "name": wire_name,
                }
        if kind == "custom_tool_call_output":
            return {**item, "type": "function_call_output"}
        return item

    def _choice(self, choice: JsonValue) -> JsonValue:
        if not isinstance(choice, dict):
            return choice
        if (
            self._search_name is not None
            and choice.get("type") == "tool_search"
            and choice.get("execution") != "server"
        ):
            return {"type": "function", "name": self._search_name}
        if self._policy.flatten_namespaces and choice.get("type") in {
            "function",
            "custom",
        }:
            return {
                **{key: value for key, value in choice.items() if key != "namespace"},
                "type": "function",
                "name": flatten_responses_tool_name(
                    _name(choice), namespace=_namespace(choice)
                ),
            }
        if choice.get("type") == "custom":
            return {
                **choice,
                "type": "function",
                "name": flatten_responses_tool_name(
                    _name(choice), namespace=_namespace(choice)
                ),
            }
        children = choice.get("tools")
        if isinstance(children, list):
            return {**choice, "tools": [self._choice(tool) for tool in children]}
        return choice

    def _custom(self, item: Mapping[str, JsonValue]) -> ResponsesToolIdentity | None:
        name = item.get("name")
        if not isinstance(name, str):
            return None
        namespace = _namespace(item)
        if namespace is not None:
            identity = self._identities.get((namespace, name))
        else:
            candidates = {
                identity
                for (_, wire_name), identity in self._identities.items()
                if wire_name == name
            }
            identity = next(iter(candidates)) if len(candidates) == 1 else None
        return identity if identity is not None and identity.kind == "custom" else None

    def restore_item(self, value: JsonValue) -> JsonValue:
        if not isinstance(value, dict):
            return value
        if (
            value.get("type") == "function_call"
            and value.get("status") == "completed"
            and (self._policy.client_tool_search or self._policy.flatten_namespaces)
        ):
            raw = value.get("arguments")
            if isinstance(raw, str) and raw:
                try:
                    canonical = json.loads(raw, parse_float=_canonical_number)
                except ValueError as exc:
                    raise ResponsesConversionError(
                        "Invalid tool call arguments."
                    ) from exc
                value = {
                    **value,
                    "arguments": json.dumps(canonical, ensure_ascii=False),
                }
        if (
            value.get("type") == "function_call"
            and self._search_name is not None
            and value.get("name") == self._search_name
        ):
            raw = value.get("arguments")
            try:
                arguments = (
                    {}
                    if value.get("status") == "in_progress"
                    else json.loads(raw)
                    if isinstance(raw, str) and raw
                    else None
                )
            except ValueError as exc:
                raise ResponsesConversionError(
                    "Invalid client search arguments."
                ) from exc
            if not isinstance(arguments, dict) or (
                not raw and value.get("status") != "in_progress"
            ):
                raise ResponsesConversionError(
                    "Client search arguments must be a JSON object."
                )
            return {
                **{
                    key: child
                    for key, child in value.items()
                    if key not in {"name", "namespace", "arguments"}
                },
                "type": "tool_search_call",
                "execution": "client",
                "arguments": arguments,
            }
        if self._policy.flatten_namespaces and value.get("type") == "function_call":
            identity = self._identity(value)
            if identity is not None:
                if value.get("status") == "completed":
                    try:
                        raw = value.get("arguments")
                        arguments = json.loads(raw) if isinstance(raw, str) else None
                    except ValueError as exc:
                        raise ResponsesConversionError(
                            "Invalid tool call arguments."
                        ) from exc
                    if not isinstance(arguments, dict):
                        raise ResponsesConversionError(
                            "Tool call arguments must be a JSON object."
                        )
                value = {**value, "name": identity.name}
                if identity.namespace is not None:
                    value["namespace"] = identity.namespace
                if identity.kind == "custom":
                    return {
                        **{
                            key: child
                            for key, child in value.items()
                            if key not in {"arguments", "type"}
                        },
                        "type": "custom_tool_call",
                        "input": custom_tool_input_text_from_arguments(
                            str(value.get("arguments", ""))
                        ),
                    }
        if value.get("type") == "tool_search_output":
            return {
                **value,
                "tools": self.restore_tools(value.get("tools"), scope=_scope(value)),
            }
        if (
            value.get("type") != "function_call"
            or (identity := self._custom(value)) is None
        ):
            return value
        arguments = value.get("arguments")
        item: JsonObject = {
            **{key: child for key, child in value.items() if key != "arguments"},
            "type": "custom_tool_call",
            "name": identity.name,
            "input": custom_tool_input_text_from_arguments(arguments)
            if isinstance(arguments, str)
            else "",
        }
        if identity.namespace is not None:
            item["namespace"] = identity.namespace
        return item

    def _identity(self, item: Mapping[str, JsonValue]) -> ResponsesToolIdentity | None:
        name = item.get("name")
        exact = self._identities.get((None, str(name)))
        if exact is not None:
            return exact
        candidates = {
            identity
            for identity in self._identities.values()
            if name in {identity.name, f"{identity.namespace}.{identity.name}"}
        }
        if len(candidates) > 1:
            raise ResponsesConversionError("Ambiguous tool name returned by provider.")
        return next(iter(candidates)) if len(candidates) == 1 else None

    def restore_tools(
        self, tools: JsonValue, namespace: str | None = None, scope: str | None = None
    ) -> JsonValue:
        if not isinstance(tools, list):
            return tools
        result: list[JsonValue] = []
        for tool in tools:
            if isinstance(tool, dict):
                if tool.get("type") == "namespace":
                    tool = {
                        **tool,
                        "tools": self.restore_tools(
                            tool.get("tools"), _name(tool), scope
                        ),
                    }
                else:
                    edits = [
                        edit
                        for edit in self._edits
                        if edit.namespace == namespace
                        and edit.adapted.get("type") == tool.get("type")
                        and edit.adapted.get("name") == tool.get("name")
                    ]
                    scoped = [edit for edit in edits if edit.scope == scope]
                    if scoped:
                        edits = scoped
                    if edits:
                        incoming = tool
                        edit = max(
                            edits,
                            key=lambda edit: sum(
                                key in incoming and incoming[key] == value
                                for key, value in edit.adapted.items()
                            ),
                        )
                        tool = dict(tool)
                        for key in edit.original.keys() | edit.adapted.keys():
                            if (key in edit.original) != (
                                key in edit.adapted
                            ) or edit.original.get(key) != edit.adapted.get(key):
                                if key in edit.original:
                                    tool[key] = deepcopy(edit.original[key])
                                else:
                                    tool.pop(key, None)
            result.append(tool)
        return result

    def restore_choice(self, value: JsonValue) -> JsonValue:
        if not isinstance(value, dict):
            return value
        if (
            value.get("type") == "function"
            and (identity := self._custom(value)) is not None
        ):
            value = {**value, "type": "custom", "name": identity.name}
            if identity.namespace is not None:
                value["namespace"] = identity.namespace
        children = value.get("tools")
        if isinstance(children, list):
            value = {
                **value,
                "tools": [self.restore_choice(tool) for tool in children],
            }
        return value

    def event_adapter(self) -> ResponsesToolEventAdapter | None:
        if (
            self._policy.client_tool_search
            or self._policy.flatten_namespaces
            or self._edits
            or any(identity.kind == "custom" for identity in self._identities.values())
        ):
            return ResponsesToolEventAdapter(self)
        return None


class ResponsesToolEventAdapter:
    """Undo one request's tool edits with fresh event state for each attempt."""

    def __init__(self, tools: ResponsesToolAdapter) -> None:
        self._tools = tools
        self._custom_items: set[str] = set()
        self._search_items: set[str] = set()
        self._function_items: set[str] = set()
        self._sequence = 0

    def feed(
        self, event_type: str, payload: JsonObject
    ) -> Iterable[tuple[str, JsonObject]]:
        data = deepcopy(payload)
        sequence = data.get("sequence_number")
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            self._sequence = max(self._sequence, sequence)
        original_item = data.get("item")
        item = self._tools.restore_item(original_item)
        if isinstance(item, dict):
            data["item"] = item
            if item.get("type") == "function_call" and (
                self._tools._policy.client_tool_search
                or self._tools._policy.flatten_namespaces
            ):
                if isinstance(item_id := item.get("id"), str):
                    self._function_items.add(item_id)
                if (
                    event_type == "response.output_item.done"
                    and item.get("status") == "completed"
                ):
                    coordinates = {
                        "item_id": item.get("id"),
                        "output_index": data.get("output_index"),
                    }
                    arguments = item.get("arguments", "")
                    yield self._emit(
                        "response.function_call_arguments.delta",
                        {**coordinates, "delta": arguments},
                    )
                    identity = {
                        key: item[key] for key in ("name", "namespace") if key in item
                    }
                    yield self._emit(
                        "response.function_call_arguments.done",
                        {**coordinates, **identity, "arguments": arguments},
                    )
            if item.get("type") == "tool_search_call" and isinstance(
                item_id := item.get("id"), str
            ):
                self._search_items.add(item_id)
            if (
                isinstance(original_item, dict)
                and original_item.get("type") == "function_call"
                and item.get("type") == "custom_tool_call"
            ):
                if isinstance(item_id := item.get("id"), str):
                    self._custom_items.add(item_id)
                if event_type == "response.output_item.done":
                    coordinates = {
                        "item_id": item.get("id"),
                        "output_index": data.get("output_index"),
                    }
                    if item["input"]:
                        yield self._emit(
                            "response.custom_tool_call_input.delta",
                            {**coordinates, "delta": item["input"]},
                        )
                    yield self._emit(
                        "response.custom_tool_call_input.done",
                        {**coordinates, "input": item["input"]},
                    )
        if (
            event_type
            in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }
            and data.get("item_id")
            in self._custom_items | self._search_items | self._function_items
        ):
            return
        if (
            self._tools._policy.flatten_namespaces
            and event_type == "response.function_call_arguments.done"
        ):
            identity = self._tools._identity(data)
            if identity is not None:
                data["name"] = identity.name
                if identity.namespace is not None:
                    data["namespace"] = identity.namespace
        response = data.get("response")
        if isinstance(response, dict):
            if isinstance(output := response.get("output"), list):
                response["output"] = [
                    self._tools.restore_item(value) for value in output
                ]
            if "tools" in response:
                response["tools"] = (
                    deepcopy(self._tools.original.tools or [])
                    if self._tools._policy.client_tool_search
                    or self._tools._policy.flatten_namespaces
                    else self._tools.restore_tools(response["tools"])
                )
            if "tool_choice" in response:
                response["tool_choice"] = (
                    self._tools.original.tool_choice or "auto"
                    if self._tools._policy.client_tool_search
                    or self._tools._policy.flatten_namespaces
                    else self._tools.restore_choice(response["tool_choice"])
                )
        yield self._emit(event_type, data)

    def _emit(self, event_type: str, payload: JsonObject) -> tuple[str, JsonObject]:
        payload = {**payload, "type": event_type, "sequence_number": self._sequence}
        self._sequence += 1
        return event_type, payload


def _name(value: Mapping[str, JsonValue]) -> str:
    return required_str(value.get("name"), "tool.name")


def _namespace(value: Mapping[str, JsonValue]) -> str | None:
    return optional_str(value.get("namespace"))


def _scope(value: Mapping[str, JsonValue]) -> str | None:
    return optional_str(value.get("call_id")) or optional_str(value.get("id"))


def _canonical_number(value: str) -> int | float:
    """Codex integer parameters reject equivalent JSON floats such as 8.0."""
    number = Decimal(value)
    return int(number) if number == number.to_integral_value() else float(number)

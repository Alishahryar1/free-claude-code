"""Provider-selected adaptations of native Responses tool representations."""

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Never, cast

import simplejson

from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import ResponsesConversionError
from .models import OpenAIResponsesRequest
from .tool_search import (
    ClientSearchHistory,
    active_client_tools,
    is_client_search,
    normalize_tool_search,
    resolve_client_search_history,
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
        self._wire_names: dict[ResponsesToolIdentity, str] = {}
        self._declared: set[ResponsesToolIdentity] = set()
        self._namespace_headers: dict[tuple[str | None, str], JsonObject] = {}
        self._edits: list[_DefinitionEdit] = []
        self._search_name: str | None = None
        self._search_history = (
            resolve_client_search_history(self.request.input)
            if policy.client_tool_search
            else ClientSearchHistory(frozenset(), {})
        )
        if policy == ResponsesToolPolicy():
            return
        client_search = policy.client_tool_search and (
            self._search_history.client_items
            or any(is_client_search(tool) for tool in request.tools or [])
        )
        if client_search:
            self.request.tools = active_client_tools(
                self.request.tools, self._search_history
            )
        if (
            policy.custom_tools_as_functions
            or policy.flatten_namespaces
            or policy.client_tool_search
        ):
            self._register_tools(self.request.tools)
            if isinstance(self.request.input, list):
                for index, item in enumerate(self.request.input):
                    if isinstance(item, dict) and item.get("type") in (
                        "function_call",
                        "custom_tool_call",
                    ):
                        self._wire_name(_call_identity(item))
                    elif (
                        isinstance(item, dict)
                        and item.get("type") == "tool_search_output"
                    ):
                        self._register_tools(
                            self._search_history.output_tools.get(
                                index, item.get("tools")
                            )
                        )
        if client_search:
            self._search_name = search_function_name(self._wire_names.values())
        if self.request.tools:
            self.request.tools = cast(list[JsonObject], self._tools(self.request.tools))
        if isinstance(self.request.input, list):
            self.request.input = [
                self._input(item, index)
                for index, item in enumerate(self.request.input)
            ]
        if (
            policy.custom_tools_as_functions
            or policy.client_tool_search
            or policy.flatten_namespaces
        ):
            self.request.tool_choice = self._choice(self.request.tool_choice)

    def _wire_name(self, identity: ResponsesToolIdentity) -> str:
        if identity in self._wire_names:
            return self._wire_names[identity]
        wire_name = (
            flatten_responses_tool_name(identity.name, namespace=identity.namespace)
            if self._policy.flatten_namespaces
            or (identity.kind == "custom" and self._policy.custom_tools_as_functions)
            else identity.name
        )
        key = (
            None if self._policy.flatten_namespaces else identity.namespace,
            wire_name,
        )
        existing = self._identities.get(key)
        if existing is not None and existing != identity:
            raise ResponsesConversionError("Tool names collide after conversion.")
        self._identities[key] = identity
        self._wire_names[identity] = wire_name
        return wire_name

    def _register_tools(self, tools: JsonValue, namespace: str | None = None) -> None:
        if not isinstance(tools, list):
            return
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "namespace":
                if not isinstance(tool.get("tools"), list):
                    raise ResponsesConversionError("Namespace tools must be a list.")
                self._register_tools(tool.get("tools"), _name(tool))
            elif tool.get("type") in ("function", "custom"):
                identity = _definition_identity(tool, namespace)
                self._wire_name(identity)
                self._declared.add(identity)

    def _tools(
        self,
        tools: JsonValue,
        namespace: str | None = None,
        scope: str | None = None,
    ) -> JsonValue:
        if not isinstance(tools, list):
            return tools
        result: list[JsonValue] = []
        for tool in tools:
            if (
                self._policy.flatten_namespaces
                and isinstance(tool, dict)
                and tool.get("type") == "namespace"
            ):
                name = _name(tool)
                self._namespace_headers[(scope, name)] = {
                    key: deepcopy(value)
                    for key, value in tool.items()
                    if key != "tools"
                }
                children = self._tools(tool.get("tools"), name, scope)
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
        tool = dict(value)
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
            identity = _definition_identity(tool, namespace)
            wire_name = self._wire_name(identity)
            tool = {
                **{key: child for key, child in tool.items() if key != kind},
                **source,
                "type": kind,
            }
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

    def _input(self, item: JsonValue, index: int) -> JsonValue:
        if not isinstance(item, dict):
            return item
        kind = item.get("type")
        if index in self._search_history.client_items:
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
                    "arguments": simplejson.dumps(arguments, use_decimal=True),
                }
            if kind == "tool_search_output":
                active = self._search_history.output_tools[index]
                return {
                    **common,
                    "type": "function_call_output",
                    "output": json.dumps(self._tools(active, scope=_scope(item))),
                }
        if kind == "tool_search_output":
            return {
                **item,
                "tools": self._tools(item.get("tools"), scope=_scope(item)),
            }
        if (
            not self._policy.custom_tools_as_functions
            and not self._policy.flatten_namespaces
        ):
            return item
        if kind in {"custom_tool_call", "function_call"}:
            wire_name = self._wire_name(_call_identity(item))
            if kind == "custom_tool_call":
                call: JsonObject = {
                    **{key: value for key, value in item.items() if key != "input"},
                    "type": "function_call",
                    "name": wire_name,
                    "arguments": json.dumps(
                        {"input": custom_tool_input_text(item.get("input"))},
                        ensure_ascii=False,
                    ),
                }
                if self._policy.flatten_namespaces:
                    call.pop("namespace", None)
                return call
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
        kind = choice.get("type")
        if (
            self._policy.flatten_namespaces and kind in ("function", "custom", "tool")
        ) or (self._policy.custom_tools_as_functions and kind == "custom"):
            identity = _definition_identity(choice)
            if kind == "tool":
                identity = next(
                    (
                        declared
                        for declared in self._declared
                        if declared.name == identity.name
                        and declared.namespace == identity.namespace
                    ),
                    identity,
                )
            result: JsonObject = {
                **{
                    key: value
                    for key, value in choice.items()
                    if key not in ("custom", "function")
                },
                "type": "function",
                "name": self._wire_name(identity),
            }
            if self._policy.flatten_namespaces:
                result.pop("namespace", None)
            return result
        children = choice.get("tools")
        if isinstance(children, list):
            return {**choice, "tools": [self._choice(tool) for tool in children]}
        return choice

    def _is_search(self, item: Mapping[str, JsonValue]) -> bool:
        return (
            self._search_name is not None
            and item.get("name") == self._search_name
            and _namespace(item) is None
        )

    def _identity(self, item: Mapping[str, JsonValue]) -> ResponsesToolIdentity | None:
        name = item.get("name")
        if not isinstance(name, str) or self._is_search(item):
            return None
        namespace = _namespace(item)
        key = (None if self._policy.flatten_namespaces else namespace, name)
        exact = self._identities.get(key)
        if exact is not None and (namespace is None or exact.namespace == namespace):
            return exact
        candidates = {
            identity
            for (_, wire_name), identity in self._identities.items()
            if wire_name == name
            and (namespace is None or identity.namespace == namespace)
        }
        if not candidates and self._policy.flatten_namespaces:
            candidates = {
                identity
                for identity in self._declared
                if (namespace is None or identity.namespace == namespace)
                and name in (identity.name, f"{identity.namespace}.{identity.name}")
            }
        if len(candidates) > 1 and self._policy.flatten_namespaces:
            raise ResponsesConversionError("Ambiguous tool name returned by provider.")
        return next(iter(candidates)) if len(candidates) == 1 else None

    def prepare_arguments(
        self, name: str, arguments: str, *, namespace: str | None = None
    ) -> str:
        """Validate provider arguments before publishing a completed tool call."""
        identity = self._identity({"name": name, "namespace": namespace})
        if identity is not None and identity.kind == "custom":
            return json.dumps(
                {"input": custom_tool_input_text_from_arguments(arguments)},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        try:
            parsed = json.loads(
                arguments,
                parse_float=_canonical_number,
                parse_constant=_reject_json_constant,
            )
        except ValueError as exc:
            raise ResponsesConversionError("Invalid tool call arguments.") from exc
        if not isinstance(parsed, dict):
            raise ResponsesConversionError("Tool call arguments must be a JSON object.")
        return simplejson.dumps(
            parsed,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            use_decimal=True,
        )

    def restore_item(self, value: JsonValue) -> JsonValue:
        if not isinstance(value, dict):
            return value
        if value.get("type") == "tool_search_output":
            return {
                **value,
                "tools": self.restore_tools(value.get("tools"), scope=_scope(value)),
            }
        if value.get("type") != "function_call":
            return value
        identity = self._identity(value)
        if value.get("status") == "completed" and (
            self._policy.client_tool_search or self._policy.flatten_namespaces
        ):
            raw = value.get("arguments")
            value = {
                **value,
                "arguments": self.prepare_arguments(
                    _name(value),
                    raw if isinstance(raw, str) else "",
                    namespace=_namespace(value),
                ),
            }
        if self._is_search(value):
            raw = value.get("arguments")
            try:
                arguments = (
                    {}
                    if value.get("status") == "in_progress"
                    else json.loads(
                        raw,
                        parse_float=_canonical_number,
                        parse_constant=_reject_json_constant,
                    )
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
        if identity is None:
            return value
        if self._policy.flatten_namespaces:
            value = {**value, "name": identity.name}
            if identity.namespace is not None:
                value["namespace"] = identity.namespace
        if identity.kind != "custom" or not (
            self._policy.custom_tools_as_functions or self._policy.flatten_namespaces
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

    def restore_tools(
        self, tools: JsonValue, namespace: str | None = None, scope: str | None = None
    ) -> JsonValue:
        if not isinstance(tools, list):
            return tools
        result: list[JsonValue] = []
        namespaces: dict[str, JsonObject] = {}
        for tool in tools:
            if isinstance(tool, dict):
                if tool.get("type") == "namespace":
                    tool = {
                        **tool,
                        "tools": self.restore_tools(
                            tool.get("tools"), _name(tool), scope
                        ),
                    }
                    if self._policy.flatten_namespaces and namespace is None:
                        name = _name(tool)
                        if name in namespaces:
                            cast(list[JsonValue], namespaces[name]["tools"]).extend(
                                cast(list[JsonValue], tool["tools"])
                            )
                            continue
                        namespaces[name] = tool
                else:
                    identity = (
                        self._identity(
                            {**tool, "namespace": namespace or _namespace(tool)}
                        )
                        if self._policy.flatten_namespaces
                        and tool.get("type") in ("function", "custom")
                        else None
                    )
                    wire_name = (
                        self._wire_names[identity]
                        if identity is not None
                        else tool.get("name")
                    )
                    edits = [
                        edit
                        for edit in self._edits
                        if edit.adapted.get("type") == tool.get("type")
                        and edit.adapted.get("name") == wire_name
                    ]
                    if self._policy.flatten_namespaces and tool.get("type") in (
                        "function",
                        "custom",
                    ):
                        if identity is None:
                            edits = []
                    else:
                        edits = [edit for edit in edits if edit.namespace == namespace]
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
                    elif identity is not None:
                        tool = {**tool, "name": identity.name}
                    if (
                        identity is not None
                        and identity.namespace is not None
                        and namespace is None
                    ):
                        name = identity.namespace
                        if name not in namespaces:
                            header = self._namespace_headers.get(
                                (scope, name),
                                self._namespace_headers.get(
                                    (None, name), {"type": "namespace", "name": name}
                                ),
                            )
                            namespaces[name] = {**deepcopy(header), "tools": []}
                            result.append(namespaces[name])
                        cast(list[JsonValue], namespaces[name]["tools"]).append(
                            {
                                key: value
                                for key, value in tool.items()
                                if key != "namespace"
                            }
                        )
                        continue
            result.append(tool)
        return result

    def restore_choice(self, value: JsonValue) -> JsonValue:
        if not isinstance(value, dict):
            return value
        if (
            value.get("type") == "function"
            and (identity := self._identity(value)) is not None
            and identity.kind == "custom"
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
                if event_type == "response.output_item.added":
                    item["arguments"] = ""
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
                if event_type == "response.output_item.added":
                    item["input"] = ""
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


def _definition_identity(
    value: Mapping[str, JsonValue], namespace: str | None = None
) -> ResponsesToolIdentity:
    kind = "custom" if value.get("type") == "custom" else "function"
    nested = value.get(kind)
    if value.get("type") == "tool" and isinstance(value.get("custom"), dict):
        nested = value["custom"]
    source = nested if isinstance(nested, dict) else value
    return ResponsesToolIdentity(
        kind=kind,
        name=_name(source),
        namespace=namespace or _namespace(source) or _namespace(value),
    )


def _call_identity(item: Mapping[str, JsonValue]) -> ResponsesToolIdentity:
    return ResponsesToolIdentity(
        kind="custom" if item.get("type") == "custom_tool_call" else "function",
        name=_name(item),
        namespace=_namespace(item),
    )


def _namespace(value: Mapping[str, JsonValue]) -> str | None:
    return optional_str(value.get("namespace"))


def _scope(value: Mapping[str, JsonValue]) -> str | None:
    return optional_str(value.get("call_id")) or optional_str(value.get("id"))


def _canonical_number(value: str) -> int | Decimal:
    """Codex integer parameters reject equivalent JSON floats such as 8.0."""
    number = Decimal(value)
    return int(number) if number == number.to_integral_value() else number


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"Non-finite JSON constant: {value}")

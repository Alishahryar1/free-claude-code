"""Request-scoped client discovery and provider search argument schemas."""

from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from free_claude_code.core.json_types import DecimalJsonValue, JsonObject, JsonValue

from .errors import ResponsesConversionError
from .tools import (
    ResponsesToolIdentity,
    definition_identity,
    optional_str,
    required_str,
)


def is_client_search(value: Mapping[str, DecimalJsonValue]) -> bool:
    return value.get("execution") == "client" and value.get("type") in {
        "tool_search",
        "tool_search_call",
        "tool_search_output",
    }


def is_unfinished_search_call(value: Mapping[str, JsonValue]) -> bool:
    return value.get("status") in ("in_progress", "incomplete", "failed")


@dataclass(frozen=True, slots=True)
class ClientSearchHistory:
    client_items: frozenset[int]
    output_tools: dict[int, list[JsonObject]]
    omitted_items: frozenset[int] = frozenset()
    completed_outputs: tuple[list[JsonValue], ...] = ()

    def loaded_tools(self) -> set[ResponsesToolIdentity]:
        return {
            definition_identity(tool, namespace)
            for output in self.completed_outputs
            for namespace, tool in _declarations(output)
            if tool.get("type") in ("function", "custom")
        }


def _declarations(
    tools: Iterable[JsonValue],
) -> Iterator[tuple[str | None, JsonObject]]:
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "namespace":
            yield None, tool
            continue
        namespace = required_str(tool.get("name"), "tool.namespace.name")
        children = tool.get("tools")
        if not isinstance(children, list):
            raise ResponsesConversionError("Namespace tools must be a list.")
        for child in children:
            if isinstance(child, dict):
                yield namespace, child


def public_search_tools(tools: JsonValue) -> list[JsonObject]:
    """Return only callable descriptions fit for a model-visible search result."""
    if not isinstance(tools, list):
        return []
    public: list[JsonObject] = []
    for namespace, tool in _declarations(tools):
        kind = tool.get("type")
        if kind not in ("function", "custom"):
            continue
        nested = tool.get(str(kind))
        definition = nested if isinstance(nested, dict) else tool
        fields = (
            ("name", "description", "parameters", "strict")
            if kind == "function"
            else ("name", "description", "format")
        )
        entry = {key: definition[key] for key in fields if key in definition}
        entry["type"] = kind
        if namespace is not None:
            entry["namespace"] = namespace
        public.append(entry)
    return public


def resolve_client_search_history(items: JsonValue) -> ClientSearchHistory:
    """Infer omitted execution only from search records with the same call ID."""
    if not isinstance(items, list):
        return ClientSearchHistory(frozenset(), {})
    searches = {
        index: item
        for index, item in enumerate(items)
        if isinstance(item, dict)
        and item.get("type") in ("tool_search_call", "tool_search_output")
    }
    executions: dict[str, str] = {}
    for item in searches.values():
        call_id, execution = item.get("call_id"), item.get("execution")
        if (
            not isinstance(call_id, str)
            or not call_id
            or execution not in ("client", "server")
        ):
            continue
        if call_id in executions and executions[call_id] != execution:
            raise ResponsesConversionError(
                "Conflicting tool search execution for one call ID."
            )
        executions[call_id] = cast(str, execution)
    client_items: set[int] = set()
    outputs: dict[int, list[JsonObject]] = {}
    omitted_items: set[int] = set()
    completed_outputs: list[list[JsonValue]] = []
    unfinished: dict[tuple[str, str | None], bool] = {}
    for index, item in searches.items():
        execution = item.get("execution")
        call_id = item.get("call_id")
        if execution is None and isinstance(call_id, str) and call_id:
            execution = executions.get(call_id)
        if execution not in ("client", "server"):
            continue
        key = (str(execution), optional_str(call_id))
        if item.get("type") == "tool_search_call":
            unfinished[key] = is_unfinished_search_call(item)
        if unfinished.get(key):
            if execution == "client":
                omitted_items.add(index)
            continue
        if execution == "client":
            client_items.add(index)
        if item.get("type") == "tool_search_output":
            tools = item.get("tools")
            accepted = (
                tools
                if item.get("status") in (None, "completed") and isinstance(tools, list)
                else []
            )
            if accepted:
                completed_outputs.append(accepted)
            if execution == "client":
                outputs[index] = _merge_tool_groups(
                    [[tool for tool in accepted if isinstance(tool, dict)], []]
                )
    return ClientSearchHistory(
        frozenset(client_items),
        outputs,
        frozenset(omitted_items),
        tuple(completed_outputs),
    )


def active_client_tools(
    tools: list[JsonObject] | None,
    history: ClientSearchHistory,
    *,
    keep_current_defer_loading: bool = False,
) -> list[JsonObject]:
    """Resolve discoveries in history order, with explicit current tools last."""
    return _merge_tool_groups(
        [*history.output_tools.values(), tools or []],
        keep_current_defer_loading=keep_current_defer_loading,
    )


def _merge_tool_groups(
    groups: list[list[JsonObject]], *, keep_current_defer_loading: bool = False
) -> list[JsonObject]:
    active: dict[ResponsesToolIdentity, JsonObject] = {}
    hosted: list[JsonObject] = []
    for group in groups:
        declarations: dict[ResponsesToolIdentity, JsonObject] = {}
        for namespace, child in _declarations(group):
            kind = child.get("type")
            if kind not in {"function", "custom"}:
                if group is groups[-1]:
                    hosted.append(deepcopy(child))
                continue
            source = child.get(str(kind))
            definition = deepcopy(source if isinstance(source, dict) else child)
            if "allowed_callers" in child:
                definition["allowed_callers"] = deepcopy(child["allowed_callers"])
            definition["type"] = kind
            if (
                keep_current_defer_loading
                and group is groups[-1]
                and child.get("defer_loading", definition.get("defer_loading"))
            ):
                definition["defer_loading"] = True
            else:
                definition.pop("defer_loading", None)
            identity = definition_identity(child, namespace)
            if identity.namespace is not None:
                definition["namespace"] = identity.namespace
            if identity in declarations and declarations[identity] != definition:
                raise ResponsesConversionError("Conflicting tool definitions.")
            declarations[identity] = definition
        active.update(declarations)
    result = list(hosted)
    descriptions = {
        required_str(tool.get("name"), "tool.namespace.name"): tool.get("description")
        for group in groups
        for tool in group
        if tool.get("type") == "namespace"
    }
    namespaces: dict[str, JsonObject] = {}
    for identity, definition in active.items():
        ns = identity.namespace
        if ns is None:
            result.append(definition)
        else:
            if ns not in namespaces:
                namespaces[ns] = {"type": "namespace", "name": ns, "tools": []}
                if isinstance(description := descriptions.get(ns), str):
                    namespaces[ns]["description"] = description
                result.append(namespaces[ns])
            definition.pop("namespace", None)
            cast(list[JsonValue], namespaces[ns]["tools"]).append(definition)
    return result


def search_function_name(reserved: Iterable[str]) -> str:
    names = set(reserved)
    name = "fcc_tool_search"
    suffix = 0
    while name in names:
        suffix += 1
        name = f"fcc_tool_search_{suffix}"
    return name


def normalize_tool_search(tool: JsonObject) -> JsonObject:
    """Represent omitted search arguments as null without changing function tools."""
    if tool.get("type") != "tool_search" or tool.get("execution") != "client":
        return tool
    parameters = tool.get("parameters")
    if not isinstance(parameters, Mapping):
        return tool
    return {**tool, "parameters": _explicit_arguments(parameters)}


def _explicit_arguments(schema: JsonValue) -> JsonValue:
    if not isinstance(schema, Mapping):
        return schema
    result = dict(schema)
    for keyword in ("properties", "$defs", "definitions"):
        children = schema.get(keyword)
        if isinstance(children, Mapping):
            result[keyword] = {
                name: _explicit_arguments(child) for name, child in children.items()
            }
    for keyword in ("items", "additionalProperties"):
        if keyword in schema:
            result[keyword] = _explicit_arguments(schema[keyword])
    for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
        children = schema.get(keyword)
        if isinstance(children, list):
            result[keyword] = [_explicit_arguments(child) for child in children]
    properties = result.get("properties")
    if isinstance(properties, Mapping):
        required = schema.get("required")
        names = required if isinstance(required, list) else []
        result["properties"] = {
            name: child if name in names else {"anyOf": [child, {"type": "null"}]}
            for name, child in properties.items()
        }
        result["required"] = [
            *names,
            *(name for name in properties if name not in names),
        ]
    return result

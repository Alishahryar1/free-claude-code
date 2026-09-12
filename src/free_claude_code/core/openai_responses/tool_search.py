"""Request-scoped client discovery and provider search argument schemas."""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import ResponsesConversionError
from .tools import optional_str, required_str


def is_client_search(value: Mapping[str, JsonValue]) -> bool:
    return value.get("execution") == "client" and value.get("type") in {
        "tool_search",
        "tool_search_call",
        "tool_search_output",
    }


@dataclass(frozen=True, slots=True)
class ClientSearchHistory:
    client_items: frozenset[int]
    output_tools: dict[int, list[JsonObject]]


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
    for index, item in searches.items():
        execution = item.get("execution")
        call_id = item.get("call_id")
        if execution is None and isinstance(call_id, str) and call_id:
            execution = executions.get(call_id)
        if execution != "client":
            continue
        client_items.add(index)
        if item.get("type") == "tool_search_output":
            tools = item.get("tools")
            accepted = item.get("status") in (None, "completed") and isinstance(
                tools, list
            )
            outputs[index] = _merge_tool_groups(
                [
                    [tool for tool in tools if isinstance(tool, dict)]
                    if accepted and isinstance(tools, list)
                    else [],
                    [],
                ]
            )
    return ClientSearchHistory(frozenset(client_items), outputs)


def active_client_tools(
    tools: list[JsonObject] | None, history: ClientSearchHistory
) -> list[JsonObject]:
    """Resolve discoveries in history order, with explicit current tools last."""
    return _merge_tool_groups([*history.output_tools.values(), tools or []])


def _merge_tool_groups(groups: list[list[JsonObject]]) -> list[JsonObject]:
    active: dict[tuple[str, str | None, str], JsonObject] = {}
    hosted: list[JsonObject] = []
    for group in groups:
        declarations: dict[tuple[str, str | None, str], JsonObject] = {}
        for tool in group:
            namespace = None
            children = [tool]
            if tool.get("type") == "namespace":
                namespace = required_str(tool.get("name"), "tool.namespace.name")
                value = tool.get("tools")
                if not isinstance(value, list):
                    raise ResponsesConversionError("Namespace tools must be a list.")
                children = [child for child in value if isinstance(child, dict)]
            for child in children:
                kind = child.get("type")
                if kind not in {"function", "custom"}:
                    if group is groups[-1]:
                        hosted.append(deepcopy(child))
                    continue
                source = child.get(str(kind))
                definition = deepcopy(source if isinstance(source, dict) else child)
                definition["type"] = kind
                definition.pop("defer_loading", None)
                name = required_str(definition.get("name"), "tool.name")
                ns = (
                    namespace
                    or optional_str(definition.get("namespace"))
                    or optional_str(child.get("namespace"))
                )
                if ns is not None:
                    definition["namespace"] = ns
                identity = (str(kind), ns, name)
                if identity in declarations and declarations[identity] != definition:
                    raise ResponsesConversionError("Conflicting tool definitions.")
                declarations[identity] = definition
        active.update(declarations)
    result = list(hosted)
    namespaces: dict[str, JsonObject] = {}
    for (_, ns, _), definition in active.items():
        if ns is None:
            result.append(definition)
        else:
            if ns not in namespaces:
                namespaces[ns] = {"type": "namespace", "name": ns, "tools": []}
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

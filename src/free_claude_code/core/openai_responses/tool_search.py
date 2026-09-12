"""Explicit argument schemas for providers requiring every search property."""

from collections.abc import Mapping
from copy import deepcopy
from typing import cast

from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import ResponsesConversionError
from .models import OpenAIResponsesRequest
from .tools import flatten_responses_tool_name, required_str


def is_client_search(value: Mapping[str, JsonValue]) -> bool:
    return value.get("execution") == "client" and value.get("type") in {
        "tool_search",
        "tool_search_call",
        "tool_search_output",
    }


def active_client_tools(request: OpenAIResponsesRequest) -> list[JsonObject]:
    """Resolve discoveries in history order, with explicit current tools last."""
    groups: list[list[JsonObject]] = []
    if isinstance(request.input, list):
        groups.extend(
            [tool for tool in tools if isinstance(tool, dict)]
            for item in request.input
            if (
                isinstance(item, dict)
                and item.get("type") == "tool_search_output"
                and is_client_search(item)
                and item.get("status") == "completed"
                and isinstance(tools := item.get("tools"), list)
            )
        )
    groups.append(request.tools or [])
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
                ns = namespace or definition.get("namespace")
                ns = ns if isinstance(ns, str) else None
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


def search_function_name(tools: list[JsonObject]) -> str:
    names: set[str] = set()
    for tool in tools:
        ns = tool.get("name") if tool.get("type") == "namespace" else None
        children = tool.get("tools") if ns else [tool]
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict) and isinstance(
                    name := child.get("name"), str
                ):
                    names.add(
                        flatten_responses_tool_name(
                            name, namespace=ns if isinstance(ns, str) else None
                        )
                    )
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

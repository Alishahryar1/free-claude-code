"""Tool conversion helpers for the OpenAI Responses adapter."""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from free_claude_code.core.json_types import JsonObject

from .errors import ResponsesConversionError
from .ids import new_call_id

_MAX_TOOL_NAME_LEN = 64
_NAMESPACE_TOOL_SEPARATOR = "__"
_INVALID_TOOL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class ResponsesToolIdentity:
    kind: Literal["function", "custom"]
    name: str
    namespace: str | None = None


def flatten_responses_tool_name(name: str, *, namespace: str | None = None) -> str:
    """Return a deterministic flat tool name for a Responses tool identity."""

    if not namespace:
        return name
    combined = (
        f"{_tool_name_part(namespace)}"
        f"{_NAMESPACE_TOOL_SEPARATOR}"
        f"{_tool_name_part(name)}"
    )
    if len(combined) <= _MAX_TOOL_NAME_LEN:
        return combined
    digest = hashlib.sha1(combined.encode("utf-8")).hexdigest()[:8]
    prefix_len = _MAX_TOOL_NAME_LEN - len(digest) - 1
    return f"{combined[:prefix_len]}_{digest}"


def responses_tool_identity_from_wire_name(
    tools: list[dict[str, Any]] | None, wire_name: str
) -> ResponsesToolIdentity:
    """Return the Responses namespace/name represented by a flat tool name."""

    if tools is None:
        return ResponsesToolIdentity(kind="function", name=wire_name)
    aliases: set[ResponsesToolIdentity] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type == "function":
            source = tool.get("function")
            function = source if isinstance(source, dict) else tool
            if (name := optional_str(function.get("name"))) and (
                flatten_responses_tool_name(name) == wire_name
            ):
                return ResponsesToolIdentity(kind="function", name=name)
            continue
        if tool_type == "custom":
            source = custom_tool_source(tool)
            if (name := optional_str(source.get("name"))) and (
                flatten_responses_tool_name(name) == wire_name
            ):
                return ResponsesToolIdentity(kind="custom", name=name)
            continue
        if tool_type != "namespace":
            continue
        namespace = optional_str(tool.get("name"))
        nested_tools = tool.get("tools")
        if not namespace or not isinstance(nested_tools, list):
            continue
        for nested_tool in nested_tools:
            if not isinstance(nested_tool, dict):
                continue
            nested_tool_type = nested_tool.get("type")
            if nested_tool_type in {"function", "custom"}:
                candidate_name = optional_str(nested_tool.get("name"))
                if candidate_name and wire_name in {
                    candidate_name,
                    f"{namespace}.{candidate_name}",
                }:
                    aliases.add(
                        ResponsesToolIdentity(
                            kind="custom"
                            if nested_tool_type == "custom"
                            else "function",
                            name=candidate_name,
                            namespace=namespace,
                        )
                    )
            if nested_tool_type == "function":
                source = nested_tool.get("function")
                function = source if isinstance(source, dict) else nested_tool
                if (name := optional_str(function.get("name"))) and (
                    flatten_responses_tool_name(name, namespace=namespace) == wire_name
                ):
                    return ResponsesToolIdentity(
                        kind="function", name=name, namespace=namespace
                    )
                continue
            if nested_tool_type == "custom":
                source = custom_tool_source(nested_tool)
                if (name := optional_str(source.get("name"))) and (
                    flatten_responses_tool_name(name, namespace=namespace) == wire_name
                ):
                    return ResponsesToolIdentity(
                        kind="custom", name=name, namespace=namespace
                    )
    if len(aliases) == 1:
        return next(iter(aliases))
    if len(aliases) > 1:
        raise ResponsesConversionError("Ambiguous tool name returned by provider.")
    return ResponsesToolIdentity(kind="function", name=wire_name)


def parse_arguments(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ResponsesConversionError("Responses function_call arguments must be JSON")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ResponsesConversionError(
            f"Responses function_call arguments are invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ResponsesConversionError(
            "Responses function_call arguments must decode to an object"
        )
    return parsed


def normalized_function_call_arguments(value: Any) -> str:
    return json.dumps(parse_arguments(value), separators=(",", ":"))


def custom_tool_input_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _json_dumps(value)


def custom_tool_input_text_from_wrapper(value: Any) -> str:
    if isinstance(value, Mapping):
        raw_input = value.get("input")
        if isinstance(raw_input, str):
            return raw_input
        if raw_input is not None:
            return custom_tool_input_text(raw_input)
        if not value:
            return ""
        return _json_dumps(value)
    return custom_tool_input_text(value)


def custom_tool_input_text_from_arguments(arguments: str) -> str:
    if not arguments:
        return ""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    return custom_tool_input_text_from_wrapper(parsed)


def call_id_from_item(item: Mapping[str, Any]) -> str:
    for key in ("call_id", "id"):
        if value := optional_str(item.get(key)):
            return value
    return new_call_id()


def required_str(value: Any, field_name: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise ResponsesConversionError(
        f"Responses field {field_name} must be a non-empty string"
    )


def optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def custom_tool_source(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    custom = tool.get("custom")
    return custom if isinstance(custom, Mapping) else tool


def custom_tool_description(source: Mapping[str, Any]) -> str | None:
    parts: list[str] = []
    if description := optional_str(source.get("description")):
        parts.append(description)
    format_value = source.get("format")
    if isinstance(format_value, Mapping):
        format_type = optional_str(format_value.get("type"))
        if format_type == "text":
            parts.append("Custom tool input format: unconstrained text.")
        elif format_type == "grammar":
            syntax = optional_str(format_value.get("syntax"))
            definition = optional_str(format_value.get("definition"))
            guidance = "Custom tool input format: grammar"
            if syntax:
                guidance = f"{guidance} ({syntax})"
            guidance = f"{guidance}: {definition}" if definition else f"{guidance}."
            parts.append(guidance)
        elif format_type:
            parts.append(f"Custom tool input format: {format_type}.")
        else:
            parts.append(f"Custom tool input format: {_json_dumps(format_value)}")
    return "\n\n".join(parts) if parts else None


def _tool_name_part(value: str) -> str:
    normalized = _INVALID_TOOL_NAME_CHARS.sub("_", value).strip("_")
    return normalized or "tool"


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def custom_tool_input_schema(
    *, description: str | None = "Free-form input for the custom tool."
) -> JsonObject:
    input_property: JsonObject = {"type": "string"}
    if description is not None:
        input_property["description"] = description
    return {
        "type": "object",
        "properties": {"input": input_property},
        "required": ["input"],
    }

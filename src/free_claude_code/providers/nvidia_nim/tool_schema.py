"""NVIDIA NIM tool schema sanitization and private argument aliases."""

from typing import Any

_SCHEMA_VALUE_KEYS = frozenset(
    {
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "items",
        "contains",
        "propertyNames",
        "if",
        "then",
        "else",
        "not",
    }
)
_SCHEMA_LIST_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_MAP_KEYS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)
NIM_TOOL_ARGUMENT_ALIASES_KEY = "_fcc_nim_tool_argument_aliases"
_NIM_TOOL_PARAMETER_ALIAS_PREFIX = "_fcc_arg_"
_NIM_UNSAFE_TOOL_PARAMETER_NAMES = frozenset({"type"})


def sanitize_nim_tool_schemas(body: dict[str, Any]) -> None:
    """Sanitize only tool parameter schemas, preserving tool calls/history."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return

    tool_argument_aliases: dict[str, dict[str, str]] = {}
    sanitized_tools: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            sanitized_tools.append(tool)
            continue
        sanitized_tool = dict(tool)
        function = tool.get("function")
        if isinstance(function, dict):
            sanitized_function = dict(function)
            parameters = function.get("parameters")
            if isinstance(parameters, dict):
                _, sanitized_parameters = _sanitize_nim_schema_node(parameters)
                sanitized_parameters, argument_aliases = _alias_nim_tool_parameters(
                    sanitized_parameters
                )
                sanitized_function["parameters"] = sanitized_parameters
                tool_name = function.get("name")
                if argument_aliases and isinstance(tool_name, str) and tool_name:
                    tool_argument_aliases[tool_name] = argument_aliases
            sanitized_tool["function"] = sanitized_function
        sanitized_tools.append(sanitized_tool)

    body["tools"] = sanitized_tools
    if tool_argument_aliases:
        body[NIM_TOOL_ARGUMENT_ALIASES_KEY] = tool_argument_aliases
    else:
        body.pop(NIM_TOOL_ARGUMENT_ALIASES_KEY, None)


def nim_tool_argument_aliases_from_body(
    body: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Return validated private NIM tool argument aliases from a built body."""
    raw_aliases = body.get(NIM_TOOL_ARGUMENT_ALIASES_KEY)
    if not isinstance(raw_aliases, dict):
        return {}

    aliases: dict[str, dict[str, str]] = {}
    for tool_name, tool_aliases in raw_aliases.items():
        if not isinstance(tool_name, str) or not isinstance(tool_aliases, dict):
            continue
        sanitized_aliases = {
            alias: original
            for alias, original in tool_aliases.items()
            if isinstance(alias, str) and isinstance(original, str)
        }
        if sanitized_aliases:
            aliases[tool_name] = sanitized_aliases
    return aliases


def body_without_nim_tool_argument_aliases(body: dict[str, Any]) -> dict[str, Any]:
    """Return a request body with private alias metadata stripped before upstream I/O."""
    if NIM_TOOL_ARGUMENT_ALIASES_KEY not in body:
        return body
    upstream_body = dict(body)
    upstream_body.pop(NIM_TOOL_ARGUMENT_ALIASES_KEY, None)
    return upstream_body


def _sanitize_nim_schema_node(value: Any) -> tuple[bool, Any]:
    """Remove boolean JSON Schema subschemas that hosted NIM rejects."""
    if isinstance(value, bool):
        return False, None
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SCHEMA_VALUE_KEYS:
                keep, sanitized_item = _sanitize_nim_schema_node(item)
                if keep:
                    sanitized[key] = sanitized_item
            elif key in _SCHEMA_LIST_KEYS and isinstance(item, list):
                sanitized_items: list[Any] = []
                for schema_item in item:
                    keep, sanitized_item = _sanitize_nim_schema_node(schema_item)
                    if keep:
                        sanitized_items.append(sanitized_item)
                if sanitized_items:
                    sanitized[key] = sanitized_items
            elif key in _SCHEMA_MAP_KEYS and isinstance(item, dict):
                sanitized_map: dict[str, Any] = {}
                for map_key, schema_item in item.items():
                    keep, sanitized_item = _sanitize_nim_schema_node(schema_item)
                    if keep:
                        sanitized_map[map_key] = sanitized_item
                sanitized[key] = sanitized_map
            else:
                sanitized[key] = item
        return True, sanitized
    if isinstance(value, list):
        sanitized_items = []
        for item in value:
            keep, sanitized_item = _sanitize_nim_schema_node(item)
            if keep:
                sanitized_items.append(sanitized_item)
        return True, sanitized_items
    return True, value


def _needs_nim_tool_parameter_alias(name: str) -> bool:
    return name in _NIM_UNSAFE_TOOL_PARAMETER_NAMES


def _make_nim_tool_parameter_alias(name: str, reserved: set[str]) -> str:
    safe_tail = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in name
    ).strip("_")
    if not safe_tail:
        safe_tail = "arg"
    candidate = f"{_NIM_TOOL_PARAMETER_ALIAS_PREFIX}{safe_tail}"
    alias = candidate
    suffix = 2
    while alias in reserved:
        alias = f"{candidate}_{suffix}"
        suffix += 1
    reserved.add(alias)
    return alias


def _collect_nim_tool_property_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            for property_name, property_schema in properties.items():
                if isinstance(property_name, str):
                    names.add(property_name)
                names.update(_collect_nim_tool_property_names(property_schema))
        for key, item in value.items():
            if key != "properties":
                names.update(_collect_nim_tool_property_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(_collect_nim_tool_property_names(item))
    return names


def _alias_nim_schema_property_names(
    value: Any,
    *,
    reserved: set[str],
    alias_to_original: dict[str, str],
    original_to_alias: dict[str, str],
) -> Any:
    if isinstance(value, list):
        return [
            _alias_nim_schema_property_names(
                item,
                reserved=reserved,
                alias_to_original=alias_to_original,
                original_to_alias=original_to_alias,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    local_aliases: dict[str, str] = {}
    aliased_value: dict[str, Any] = {}
    properties = value.get("properties")
    if isinstance(properties, dict):
        aliased_properties: dict[str, Any] = {}
        for property_name, property_schema in properties.items():
            aliased_schema = _alias_nim_schema_property_names(
                property_schema,
                reserved=reserved,
                alias_to_original=alias_to_original,
                original_to_alias=original_to_alias,
            )
            if isinstance(property_name, str) and _needs_nim_tool_parameter_alias(
                property_name
            ):
                alias = original_to_alias.get(property_name)
                if alias is None:
                    alias = _make_nim_tool_parameter_alias(property_name, reserved)
                    alias_to_original[alias] = property_name
                    original_to_alias[property_name] = alias
                local_aliases[property_name] = alias
                aliased_properties[alias] = aliased_schema
            else:
                aliased_properties[property_name] = aliased_schema
        aliased_value["properties"] = aliased_properties

    for key, item in value.items():
        if key == "properties":
            continue
        if key == "required" and isinstance(item, list):
            aliased_value[key] = [
                local_aliases.get(required_item, required_item)
                if isinstance(required_item, str)
                else required_item
                for required_item in item
            ]
            continue
        aliased_value[key] = _alias_nim_schema_property_names(
            item,
            reserved=reserved,
            alias_to_original=alias_to_original,
            original_to_alias=original_to_alias,
        )

    return aliased_value


def _alias_nim_tool_parameters(
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    alias_to_original: dict[str, str] = {}
    original_to_alias: dict[str, str] = {}
    reserved = _collect_nim_tool_property_names(parameters)
    aliased_parameters = _alias_nim_schema_property_names(
        parameters,
        reserved=reserved,
        alias_to_original=alias_to_original,
        original_to_alias=original_to_alias,
    )
    if not alias_to_original:
        return parameters, {}
    return aliased_parameters, alias_to_original

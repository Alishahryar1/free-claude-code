"""JSON Schema helpers for text-emitted Anthropic tool input."""

import json
from collections.abc import Mapping
from typing import Any

import jsonschema

# JSON Schema keywords whose values are name->subschema maps. Keywords holding
# a single subschema or a list of them are covered by the union below; literal-
# value keywords such as ``enum`` or ``default`` are deliberately absent: their
# contents are data, not schema. ``dependencies`` values are either a
# property-name array (data, left as-is by the non-dict passthrough) or a
# subschema (sanitized), so map traversal handles both forms safely.
_SCHEMA_MAP_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "dependencies",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)
_SUBSCHEMA_KEYS = _SCHEMA_MAP_KEYS | frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "allOf",
        "anyOf",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "oneOf",
        "prefixItems",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)


def schema_type(schema: Mapping[str, Any]) -> str | None:
    """Resolve one useful non-null JSON type from a simple or union schema."""
    declared = schema.get("type")
    if isinstance(declared, str):
        return declared
    if isinstance(declared, list):
        non_null = [
            item for item in declared if isinstance(item, str) and item != "null"
        ]
        if len(non_null) == 1:
            return non_null[0]

    for keyword in ("oneOf", "anyOf"):
        alternatives = schema.get(keyword)
        if not isinstance(alternatives, list):
            continue
        types = {
            nested_type
            for alternative in alternatives
            if isinstance(alternative, Mapping)
            if (nested_type := schema_type(alternative)) not in (None, "null")
        }
        if len(types) == 1:
            return types.pop()
    return None


def coerce_text_argument(value: str, schema: Mapping[str, Any]) -> Any:
    """Decode a textual tool argument according to its declared JSON type."""
    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        for enum_value in enum_values:
            if str(enum_value) == value:
                return enum_value
    if "const" in schema and str(schema["const"]) == value:
        return schema["const"]

    value_type = schema_type(schema)
    stripped = value.strip()
    try:
        if value_type == "integer":
            return int(stripped)
        if value_type == "number":
            number = json.loads(stripped)
            if isinstance(number, int | float) and not isinstance(number, bool):
                return number
            raise ValueError
        if value_type == "boolean":
            if stripped.lower() == "true":
                return True
            if stripped.lower() == "false":
                return False
            raise ValueError
        if value_type == "null":
            if stripped.lower() == "null":
                return None
            raise ValueError
        if value_type == "array":
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
            raise ValueError
        if value_type == "object":
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError
    except json.JSONDecodeError as error:
        raise ValueError from error
    return value


def arguments_match_schema(
    arguments: dict[str, Any], schema: Mapping[str, Any]
) -> bool:
    """Return whether arguments satisfy a valid JSON Schema."""
    try:
        validator_type = jsonschema.validators.validator_for(schema)
        validator_type.check_schema(schema)
        validator_type(schema).validate(arguments)
    except jsonschema.exceptions.SchemaError:
        return True
    except jsonschema.exceptions.ValidationError:
        return False
    return True


def sanitize_tool_schema_patterns(schema: Any) -> Any:
    """Return a tool schema with regex ``pattern`` values usable by OpenAI-compatible APIs.

    Some providers validate ``pattern`` values against regex dialects that reject
    Unicode property escapes (``\\p{...}``), failing the whole request with a 400
    before it reaches the model. Any ``pattern`` containing such an escape is
    dropped entirely — removing the escape could narrow what the pattern accepts,
    and a stricter remainder would reject arguments the original schema allowed.
    Schemas without offending patterns are returned unchanged — no copy and no
    mutation — and literal-value keywords such as ``enum`` are never traversed.
    """
    if not isinstance(schema, dict):
        return schema
    items: list[tuple[str, Any]] = []
    changed = False
    for key, value in schema.items():
        if (
            key == "pattern"
            and isinstance(value, str)
            and ("\\p{" in value or "\\P{" in value)
        ):
            changed = True
            continue
        if key in _SUBSCHEMA_KEYS:
            sanitized = _sanitize_subschema(key, value)
            if sanitized is not value:
                changed = True
            items.append((key, sanitized))
            continue
        items.append((key, value))
    return dict(items) if changed else schema


def _sanitize_subschema(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        if key in _SCHEMA_MAP_KEYS:
            return _sanitize_schema_map(value)
        return sanitize_tool_schema_patterns(value)
    if isinstance(value, list):
        items: list[Any] = []
        changed = False
        for item in value:
            sanitized = sanitize_tool_schema_patterns(item)
            if sanitized is not item:
                changed = True
            items.append(sanitized)
        return items if changed else value
    return value


def _sanitize_schema_map(mapping: dict[str, Any]) -> Any:
    items: list[tuple[str, Any]] = []
    changed = False
    for key, value in mapping.items():
        sanitized = sanitize_tool_schema_patterns(value)
        if sanitized is not value:
            changed = True
        items.append((key, sanitized))
    return dict(items) if changed else mapping

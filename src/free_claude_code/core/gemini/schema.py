"""Lossless conversion of the supported Gemini Schema into JSON Schema."""

from copy import deepcopy

from ..json_types import JsonObject, JsonValue
from .errors import GeminiConversionError

_TYPES: dict[str, str] = {
    name.upper(): name
    for name in ("string", "number", "integer", "boolean", "array", "object", "null")
}
_FIELDS = frozenset(
    {
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "enum",
        "default",
        "items",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "properties",
        "additionalProperties",
        "required",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "anyOf",
    }
)


def json_schema(schema: JsonObject) -> JsonObject:
    aliases = {
        "min_items": "minItems",
        "max_items": "maxItems",
        "min_properties": "minProperties",
        "max_properties": "maxProperties",
        "min_length": "minLength",
        "max_length": "maxLength",
        "any_of": "anyOf",
        "additional_properties": "additionalProperties",
    }
    result: JsonObject = {}
    for key, value in schema.items():
        name = aliases.get(key, key)
        if name in result:
            raise GeminiConversionError(f"Duplicate Gemini Schema field: {name}")
        if value is not None or name == "default":
            result[name] = deepcopy(value)
    unknown = set(result) - _FIELDS
    if unknown:
        raise GeminiConversionError(
            f"Unsupported Gemini Schema field: {sorted(unknown)[0]}"
        )
    for name in (
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "minLength",
        "maxLength",
    ):
        if name not in result:
            continue
        count = result[name]
        if isinstance(count, str):
            try:
                count = int(count)
            except ValueError as error:
                raise GeminiConversionError(
                    f"Schema {name} must be an integer"
                ) from error
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= 2**63 - 1
        ):
            raise GeminiConversionError(f"Schema {name} must be a nonnegative int64")
        result[name] = count
    kind = result.get("type")
    if kind is not None:
        if not isinstance(kind, str) or kind not in _TYPES:
            raise GeminiConversionError(
                "Gemini Schema type must be a supported uppercase enum"
            )
        result["type"] = _TYPES[kind]
    properties = result.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise GeminiConversionError("Schema properties must be an object")
        result["properties"] = {
            name: _subschema(value) for name, value in properties.items()
        }
    for name in ("items", "additionalProperties"):
        if isinstance(result.get(name), dict):
            result[name] = _subschema(result[name])
    if "anyOf" in result:
        choices = result["anyOf"]
        if not isinstance(choices, list):
            raise GeminiConversionError("Schema anyOf must be an array")
        result["anyOf"] = [_subschema(value) for value in choices]
    nullable = result.pop("nullable", False)
    if not isinstance(nullable, bool):
        raise GeminiConversionError("Schema nullable must be a boolean")
    return {"anyOf": [result, {"type": "null"}]} if nullable else result


def _subschema(value: JsonValue) -> JsonObject:
    if not isinstance(value, dict):
        raise GeminiConversionError("A schema must be an object")
    return json_schema(value)

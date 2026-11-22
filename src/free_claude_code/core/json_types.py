"""Shared JSON value vocabulary for wire and persistence boundaries."""

import json
from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal

type JsonScalar = bool | int | float | str | None
type JsonTree[ExtraScalar] = (
    JsonScalar
    | ExtraScalar
    | Sequence[JsonTree[ExtraScalar]]
    | Mapping[str, JsonTree[ExtraScalar]]
)
type JsonValue = JsonTree[None]
type JsonObject = dict[str, JsonValue]
type DecimalJsonValue = JsonTree[Decimal]
type DecimalJsonObject = dict[str, DecimalJsonValue]


def load_json_value(data: str | bytes | bytearray) -> JsonValue:
    """Decode JSON using the standard decoder's default value types."""
    return json.loads(data)


def json_object[T](value: JsonTree[T]) -> dict[str, JsonTree[T]]:
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return value


def json_objects[T](value: JsonTree[T]) -> Iterator[dict[str, JsonTree[T]]]:
    for item in json_array(value):
        yield json_object(item)


def json_array[T](value: JsonTree[T]) -> list[JsonTree[T]]:
    if not isinstance(value, list):
        raise TypeError("Expected a JSON array")
    return value


def normalize_empty_array_field(payload: JsonObject, key: str) -> None:
    """Canonicalize empty provider placeholders in a known array field."""
    value = payload.get(key)
    if isinstance(value, (dict, str)) and not value:
        empty: list[JsonValue] = []
        payload[key] = empty


def json_index(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError("Expected an integer index")
    return value


def json_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("Expected a JSON string")
    return value


def optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def wire_field(item: object, name: str) -> object:
    """Read one field from an SDK object or an already serialized mapping."""
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)

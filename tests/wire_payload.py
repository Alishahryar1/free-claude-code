"""Shape assertions for JSON payloads used in wire contract tests."""

from typing import Protocol, TypeGuard

from free_claude_code.core.json_types import (
    JsonObject,
    JsonTree,
    JsonValue,
    load_json_value,
)


class _JsonResponse(Protocol):
    def json(self) -> JsonValue: ...


def response_payload(response: _JsonResponse) -> JsonValue:
    return response.json()


def response_object(response: _JsonResponse) -> JsonObject:
    return object_payload(response_payload(response))


def object_payload[T](value: JsonTree[T]) -> dict[str, JsonTree[T]]:
    assert isinstance(value, dict), (
        f"Expected a JSON object, got {type(value).__name__}"
    )
    return value


def decode_object(data: str | bytes | bytearray) -> JsonObject:
    return object_payload(load_json_value(data))


def array_payload[T](value: JsonTree[T]) -> list[JsonTree[T]]:
    assert isinstance(value, list), f"Expected a JSON array, got {type(value).__name__}"
    return value


def _is_object_array[T](value: JsonTree[T]) -> TypeGuard[list[dict[str, JsonTree[T]]]]:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def object_array[T](value: JsonTree[T]) -> list[dict[str, JsonTree[T]]]:
    assert _is_object_array(value), "Expected a JSON array of objects"
    return value


def string_payload(value: object) -> str:
    assert isinstance(value, str), f"Expected a JSON string, got {type(value).__name__}"
    return value


def int_payload(value: object) -> int:
    assert isinstance(value, int), (
        f"Expected a JSON integer, got {type(value).__name__}"
    )
    return value

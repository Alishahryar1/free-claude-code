"""Shape assertions for JSON payloads used in wire contract tests."""

from typing import TypeGuard

from free_claude_code.core.json_types import JsonObject, JsonValue


def object_payload(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict), (
        f"Expected a JSON object, got {type(value).__name__}"
    )
    return value


def array_payload(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list), f"Expected a JSON array, got {type(value).__name__}"
    return value


def _is_object_array(value: JsonValue) -> TypeGuard[list[JsonObject]]:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def object_array(value: JsonValue) -> list[JsonObject]:
    assert _is_object_array(value), "Expected a JSON array of objects"
    return value


def string_payload(value: JsonValue) -> str:
    assert isinstance(value, str), f"Expected a JSON string, got {type(value).__name__}"
    return value


def int_payload(value: JsonValue) -> int:
    assert isinstance(value, int), (
        f"Expected a JSON integer, got {type(value).__name__}"
    )
    return value

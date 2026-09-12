from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_responses.tool_search import (
    ClientSearchHistory,
    active_client_tools,
    normalize_tool_search,
)


def test_nested_search_arguments_keep_constraints_and_literal_data() -> None:
    tool: JsonObject = {
        "type": "tool_search",
        "execution": "client",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/filter"},
                },
            },
            "required": ["query"],
            "additionalProperties": False,
            "$defs": {
                "filter": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["file", "directory"]},
                        "count": {
                            "anyOf": [
                                {"type": "integer", "minimum": 1},
                                {"type": "string", "enum": ["all"]},
                            ]
                        },
                    },
                    "required": ["kind"],
                    "additionalProperties": False,
                }
            },
            "examples": [{"properties": {"literal": "untouched"}}],
        },
    }
    original = deepcopy(tool)
    normalized = normalize_tool_search(tool)
    schema = normalized["parameters"]
    assert isinstance(schema, dict)
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"query": "files", "filters": None})
    assert validator.is_valid(
        {"query": "files", "filters": [{"kind": "file", "count": None}]}
    )
    assert validator.is_valid(
        {"query": "files", "filters": [{"kind": "file", "count": 2}]}
    )
    assert not validator.is_valid(
        {"query": "files", "filters": [{"kind": "file", "count": 0}]}
    )
    assert not validator.is_valid(
        {"query": "files", "filters": [{"kind": "unknown", "count": None}]}
    )
    assert not validator.is_valid({"query": "files", "filters": [{"kind": "file"}]})
    assert schema["examples"] == [{"properties": {"literal": "untouched"}}]
    assert normalize_tool_search(normalized) == normalized
    assert tool == original


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "tool_search"},
        {"type": "tool_search", "execution": "server"},
        {"type": "tool_search", "execution": "client"},
        {
            "type": "function",
            "name": "search",
            "strict": False,
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    ],
)
def test_other_tool_definitions_are_preserved(tool: JsonObject) -> None:
    assert normalize_tool_search(tool) == tool


@pytest.mark.parametrize("kind", ["function", "custom"])
@pytest.mark.parametrize(
    ("container", "inner", "expected"),
    [
        (None, None, "outer"),
        (None, "", "outer"),
        (None, 7, "outer"),
        (None, "inner", "inner"),
        ("container", "inner", "container"),
    ],
)
def test_nested_tool_namespace_precedence_during_discovery(
    kind: str, container: str | None, inner: JsonValue, expected: str
) -> None:
    tool: JsonObject = {
        "type": kind,
        "namespace": "outer",
        kind: {"name": "run", "namespace": inner},
    }
    if container:
        tool = {"type": "namespace", "name": container, "tools": [tool]}
    original = deepcopy(tool)
    resolved = active_client_tools([tool], ClientSearchHistory(frozenset(), {}))
    assert resolved == [
        {
            "type": "namespace",
            "name": expected,
            "tools": [{"type": kind, "name": "run"}],
        }
    ]
    assert tool == original

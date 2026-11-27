from collections.abc import Iterator, Mapping
from typing import get_args, get_type_hints

import pytest
from openai.types.responses.response_input_item_param import ResponseInputItemParam

from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    ResponsesConversionError,
    read_history,
)
from free_claude_code.core.openai_responses.hosted_history import public_tool_record
from free_claude_code.core.openai_responses.input_history import InputMessage


@pytest.mark.parametrize(
    "record",
    [
        {
            "type": "web_search_call",
            "action": {
                "type": "search",
                "query": "public-query",
                "sources": [{"url": "public-link", "internal": "private-setting"}],
            },
        },
        {
            "type": "file_search_call",
            "queries": ["public-query"],
            "results": [
                {"text": "public-excerpt", "attributes": {"flag": "private-setting"}}
            ],
        },
        {
            "type": "mcp_call",
            "name": "lookup",
            "arguments": "{}",
            "output": "public-output",
        },
        {
            "type": "mcp_call",
            "error": {
                "type": "mcp_tool_execution_error",
                "content": "public-explanation",
                "internal": "private-setting",
            },
        },
        {"type": "apply_patch_call_output", "output": "public-result"},
        {
            "type": "shell_call_output",
            "output": [
                {
                    "stdout": "public-output",
                    "stderr": "",
                    "outcome": {
                        "type": "exit",
                        "exit_code": 0,
                        "internal": "private-setting",
                    },
                }
            ],
        },
    ],
)
def test_completed_tools_expose_only_public_fields(record: JsonObject) -> None:
    [(_, message)] = read_history(
        [{**record, "status": "completed", "internal": "private-setting"}]
    )
    assert isinstance(message, InputMessage) and isinstance(message.content, str)
    assert "public-" in message.content
    assert "private-setting" not in message.content


def test_unknown_item_cannot_consume_a_pending_action() -> None:
    history = list(
        read_history(
            [
                {
                    "type": "local_shell_call",
                    "call_id": "s1",
                    "status": "completed",
                    "action": {"type": "exec", "command": ["pwd"]},
                },
                {"type": "future_annotation_output", "call_id": "s1"},
                {
                    "type": "local_shell_call_output",
                    "id": "s1",
                    "output": "working directory",
                },
            ]
        )
    )
    assert [index for index, _ in history] == [0, 2]


@pytest.mark.parametrize(
    ("kind", "field"),
    [
        ("file_search_call", "results"),
        ("web_search_call", "sources"),
        ("code_interpreter_call", "outputs"),
    ],
)
def test_optional_tool_evidence_distinguishes_absence_and_empty(
    kind: str, field: str
) -> None:
    cases: tuple[JsonObject, ...] = ({}, {field: None}, {field: []})
    for fields in cases:
        item: JsonObject = {"type": kind}
        if kind == "web_search_call":
            item["action"] = {"type": "search", "query": "catalog", **fields}
        else:
            item.update(fields)
        record = public_tool_record(kind, item)
        assert record is not None
        public = record["action"] if kind == "web_search_call" else record
        assert isinstance(public, dict)
        if field not in fields:
            assert field not in public
        else:
            assert public[field] == fields[field]


def test_stored_image_in_a_custom_result_requires_native_responses() -> None:
    with pytest.raises(ResponsesConversionError, match="file_id"):
        list(
            read_history(
                [
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "call1",
                        "output": [{"type": "input_image", "file_id": "file-private"}],
                    }
                ]
            )
        )


def literal_strings(annotation: object) -> Iterator[str]:
    arguments: tuple[object, ...] = get_args(annotation)
    for value in arguments:
        if isinstance(value, str):
            yield value
        else:
            yield from literal_strings(value)


def test_every_standard_input_item_is_translated_or_rejected() -> None:
    kinds: set[str] = set()
    variants: tuple[object, ...] = get_args(ResponseInputItemParam)
    for model in variants:
        assert isinstance(model, type)
        fields: Mapping[str, object] = get_type_hints(model)
        kinds.update(literal_strings(fields["type"]))
    assert "function_call" in kinds
    skipped = []
    for kind in sorted(kinds):
        try:
            items = list(read_history([{"type": kind}]))
        except ResponsesConversionError:
            continue
        if not items:
            skipped.append(kind)
    assert not skipped, (
        f"Standard Responses items fell through as optional extensions: {skipped}"
    )

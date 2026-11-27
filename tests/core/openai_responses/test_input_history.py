from collections.abc import Iterator, Mapping
from typing import get_args, get_type_hints

import pytest
from openai.types.responses.computer_action_list_param import ComputerActionParam
from openai.types.responses.mcp_tool_call_error_param import McpToolCallErrorParam
from openai.types.responses.response_computer_tool_call_param import Action
from openai.types.responses.response_input_item_param import (
    LocalShellCallAction,
    ResponseInputItemParam,
    ShellCallAction,
)

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


def test_file_search_preserves_public_evidence_and_keeps_attributes_private() -> None:
    result: JsonObject = {
        "filename": "entry.md",
        "file_id": "file-catalog",
        "text": "Excerpt",
        "score": 0.92345,
    }
    record = public_tool_record(
        "file_search_call",
        {
            "type": "file_search_call",
            "results": [{**result, "attributes": {"department": "internal"}}],
        },
    )
    assert record is not None and record["results"] == [result]


@pytest.mark.parametrize(
    ("kind", "schema"),
    [("shell_call", ShellCallAction), ("local_shell_call", LocalShellCallAction)],
)
def test_recorded_shell_actions_keep_their_public_parameters(
    kind: str, schema: type
) -> None:
    examples: JsonObject = {
        "type": "exec",
        "command": ["old-command"],
        "commands": ["old-command"],
        "timeout_ms": 1500,
        "max_output_length": 50,
        "working_directory": "/work",
        "user": "worker",
        "env": {"SETTING": "private-runtime-value"},
    }
    action = {name: examples[name] for name in get_type_hints(schema)}
    record = public_tool_record(kind, {"type": kind, "action": action})
    assert record is not None
    assert record["action"] == {
        name: value for name, value in action.items() if name != "env"
    }
    if kind == "shell_call":
        output = public_tool_record(
            "shell_call_output",
            {
                "type": "shell_call_output",
                "max_output_length": 50,
                "output": [
                    {"stdout": "", "stderr": "", "outcome": {"type": "timeout"}}
                ],
            },
        )
        assert output is not None and output["max_output_length"] == 50


def literal_strings(annotation: object) -> Iterator[str]:
    arguments: tuple[object, ...] = get_args(annotation)
    for value in arguments:
        if isinstance(value, str):
            yield value
        else:
            yield from literal_strings(value)


@pytest.mark.parametrize(
    ("source", "kind", "field"),
    [
        (Action, "computer_call", "action"),
        (ComputerActionParam, "computer_call", "action"),
        (McpToolCallErrorParam, "mcp_call", "error"),
    ],
)
def test_public_action_and_error_fields_follow_the_sdk(
    source: object, kind: str, field: str
) -> None:
    examples: JsonObject = {
        "button": "left",
        "x": 1,
        "y": 2,
        "keys": ["Shift"],
        "path": [{"x": 1, "y": 2}],
        "scroll_x": 0,
        "scroll_y": 30,
        "text": "sample",
        "code": 400,
        "message": "Unavailable",
        "content": "Lookup failed",
    }
    variants: tuple[object, ...] = get_args(source)
    for variant in variants:
        assert isinstance(variant, type)
        fields: Mapping[str, object] = get_type_hints(variant)
        [tag] = literal_strings(fields["type"])
        value: JsonObject = {name: examples[name] for name in fields if name != "type"}
        value["type"] = tag
        record = public_tool_record(
            kind, {"type": kind, field: {**value, "vendor_option": "private"}}
        )
        assert record is not None and record[field] == value


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

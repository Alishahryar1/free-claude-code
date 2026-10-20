"""Direct Responses-to-Messages contracts for caller-owned transcripts."""

from collections.abc import Mapping

import pytest

from free_claude_code.core.anthropic.native import NativeMessagesOptions
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
    prepare_history,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesMessagesRequest,
    build_responses_messages_request,
)

_SCOPE = "github_copilot/anthropic_messages"
_FUNCTION: JsonObject = {
    "type": "function",
    "name": "lookup",
    "description": "Find a record.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    "strict": True,
}


@pytest.mark.parametrize("arguments", ['{"x":NaN}', '{"x":"\\ud800"}'])
def test_historical_arguments_must_be_serializable_native_json(arguments: str) -> None:
    with pytest.raises(ResponsesConversionError, match="finite JSON"):
        _build(
            {
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "c",
                        "name": "lookup",
                        "arguments": arguments,
                    },
                    {"type": "function_call_output", "call_id": "c", "output": "ok"},
                ]
            }
        )


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("output", ["", [], [{"type": "input_text", "text": ""}]])
def test_empty_tool_results_omit_native_content(
    custom: bool, output: JsonValue
) -> None:
    call: JsonObject = {
        "type": "custom_tool_call" if custom else "function_call",
        "call_id": "c",
        "name": "lookup",
    }
    call["input" if custom else "arguments"] = "command" if custom else "{}"
    built = _build(
        {
            "input": [
                call,
                {
                    "type": "custom_tool_call_output"
                    if custom
                    else "function_call_output",
                    "call_id": "c",
                    "output": output,
                },
            ]
        }
    )
    messages = built.body["messages"]
    assert isinstance(messages, list)
    assert messages[-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "c"}],
    }


def _build(
    payload: JsonObject, *, thinking: JsonObject | None = None
) -> ResponsesMessagesRequest:
    return build_responses_messages_request(
        OpenAIResponsesRequest.model_validate(
            {"model": "public", "input": "hi", **payload}
        ),
        options=NativeMessagesOptions("concrete", 4096, thinking),
    )


def test_text_controls_and_stateless_history_policy() -> None:
    built = _build(
        {
            "input": "hello",
            "instructions": "Be concise.",
            "stream": False,
            "store": True,
            "previous_response_id": "old",
            "max_output_tokens": 99999,
            "temperature": 0.2,
            "top_p": 0.8,
            "metadata": {"user_id": "person"},
        }
    )
    assert built.body == {
        "model": "concrete",
        "max_tokens": 4096,
        "stream": True,
        "system": [{"type": "text", "text": "Be concise."}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "temperature": 0.2,
        "top_p": 0.8,
        "metadata": {"user_id": "person"},
    }


def test_parallel_function_and_custom_roundtrip_preserves_replay_order_and_images() -> (
    None
):
    native_thinking: JsonObject = {
        "type": "thinking",
        "thinking": "authoritative",
        "signature": "opaque",
        "extension": {"record": 1},
    }
    carrier = encode_replay(
        ReplayRecord(
            ReplayOrigin(_SCOPE, "messages", "", "", "source"), native_thinking
        )
    )
    tools: list[JsonValue] = [
        {
            "type": "namespace",
            "name": "records",
            "description": "Record operations.",
            "tools": [_FUNCTION],
        },
        {"type": "custom", "name": "shell.exec", "format": {"type": "text"}},
    ]
    built = _build(
        {
            "tools": tools,
            "input": [
                {"role": "system", "content": "first"},
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "second"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Find it"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,aGk=",
                            "detail": "auto",
                        },
                    ],
                },
                {
                    "type": "reasoning",
                    "encrypted_content": carrier,
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": "display copy must not duplicate",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "lookup-id",
                    "namespace": "records",
                    "name": "lookup",
                    "arguments": '{"query":"x"}',
                },
                {
                    "type": "custom_tool_call",
                    "call_id": "custom-id",
                    "name": "shell.exec",
                    "input": "pwd",
                },
                {
                    "type": "function_call_output",
                    "call_id": "lookup-id",
                    "output": [
                        {"type": "input_text", "text": "found"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.org/result.png",
                        },
                    ],
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "custom-id",
                    "output": "done",
                },
                {"role": "user", "content": "Summarize."},
            ],
        },
        thinking={"type": "disabled"},
    )
    assert built.body["system"] == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    messages = prepare_history(
        built.body, ReplayOrigin(_SCOPE, "messages", "", "", "target")
    )["messages"]
    assert isinstance(messages, list) and len(messages) == 3
    assistant = messages[1]
    assert isinstance(assistant, Mapping)
    blocks = assistant["content"]
    assert isinstance(blocks, list)
    assert blocks[0] == native_thinking
    assert blocks[1] == {
        "type": "tool_use",
        "id": "lookup-id",
        "name": "records__lookup",
        "input": {"query": "x"},
    }
    custom = blocks[2]
    assert isinstance(custom, Mapping) and isinstance(custom["name"], str)
    assert custom["input"] == {"input": "pwd"}
    assert built.tool_identities[custom["name"]].name == "shell.exec"
    assert built.tool_identities[custom["name"]].kind == "custom"
    last = messages[2]
    assert isinstance(last, Mapping)
    assert last["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "lookup-id",
            "content": [
                {"type": "text", "text": "found"},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.org/result.png"},
                },
            ],
        },
        {
            "type": "tool_result",
            "tool_use_id": "custom-id",
            "content": [{"type": "text", "text": "done"}],
        },
        {"type": "text", "text": "Summarize."},
    ]
    declarations = built.body["tools"]
    assert isinstance(declarations, list)
    assert isinstance(declarations[0], Mapping) and declarations[0]["strict"] is True
    assert "Record operations." in str(declarations[0]["description"])
    assert "Find a record." in str(declarations[0]["description"])


@pytest.mark.parametrize(
    ("choice", "parallel", "expected"),
    [
        ("none", False, {"type": "none"}),
        ("auto", False, {"type": "auto", "disable_parallel_tool_use": True}),
        ("required", True, {"type": "any"}),
        (
            {"type": "function", "name": "lookup"},
            False,
            {"type": "tool", "name": "lookup", "disable_parallel_tool_use": True},
        ),
    ],
)
def test_tool_choices_preserve_declarations(
    choice: JsonValue, parallel: bool, expected: JsonObject
) -> None:
    built = _build(
        {"tools": [_FUNCTION], "tool_choice": choice, "parallel_tool_calls": parallel}
    )
    assert built.body["tool_choice"] == expected
    assert built.body["tools"]


def test_named_choice_requires_declaration_and_manual_thinking_rejects_forcing() -> (
    None
):
    with pytest.raises(ResponsesConversionError, match="declared"):
        _build(
            {"tools": [_FUNCTION], "tool_choice": {"type": "function", "name": "other"}}
        )
    with pytest.raises(ResponsesConversionError, match="force a tool"):
        _build(
            {"tools": [_FUNCTION], "tool_choice": "required"},
            thinking={"type": "enabled", "budget_tokens": 1024},
        )


@pytest.mark.parametrize(
    "tools",
    [
        [_FUNCTION, _FUNCTION],
        [_FUNCTION, {"type": "custom", "name": "lookup"}],
        [
            {"type": "function", "name": "records__lookup"},
            {"type": "namespace", "name": "records", "tools": [_FUNCTION]},
        ],
        [
            {"type": "namespace", "name": "record.ops", "tools": [_FUNCTION]},
            {"type": "namespace", "name": "record_ops", "tools": [_FUNCTION]},
        ],
    ],
)
def test_duplicate_or_flattened_tool_identity_collisions_rejected(
    tools: list[JsonValue],
) -> None:
    with pytest.raises(ResponsesConversionError, match=r"Duplicate|collide"):
        _build({"tools": tools})


_CALL: JsonObject = {
    "type": "function_call",
    "call_id": "c",
    "name": "lookup",
    "arguments": '{"query":"x"}',
}
_RESULT: JsonObject = {"type": "function_call_output", "call_id": "c", "output": "ok"}


@pytest.mark.parametrize(
    "items",
    [
        [_CALL],
        [_RESULT],
        [_CALL, _CALL, _RESULT],
        [_CALL, _RESULT, _RESULT],
        [_CALL, {"role": "user", "content": "intervening"}, _RESULT],
        [_CALL, {**_RESULT, "type": "custom_tool_call_output"}],
        [{**_CALL, "call_id": ""}, _RESULT],
        [{**_CALL, "arguments": "[]"}, _RESULT],
        [{**_CALL, "arguments": ""}, _RESULT],
        [{**_CALL, "arguments": None}, _RESULT],
        [
            _CALL,
            {
                "type": "function_call",
                "call_id": "d",
                "name": "lookup",
                "arguments": "{}",
            },
            _RESULT,
            {"role": "assistant", "content": "too early"},
        ],
    ],
)
def test_malformed_tool_history_is_rejected_before_inference(
    items: list[JsonValue],
) -> None:
    with pytest.raises(ResponsesConversionError):
        _build({"tools": [_FUNCTION], "input": items})


@pytest.mark.parametrize(
    "payload",
    [
        {
            "input": [
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "late"},
            ]
        },
        {"input": [{"type": "item_reference", "id": "remote"}]},
        {
            "input": [
                {"role": "user", "content": [{"type": "input_file", "file_id": "file"}]}
            ]
        },
        {"input": [{"type": "input_image", "file_id": "file"}]},
        {"input": [{"type": "input_image", "image_url": "file:///private.png"}]},
        {"input": [{"type": "input_image", "image_url": "data:image/png;base64,!!!"}]},
        {
            "input": [
                {
                    "type": "input_image",
                    "image_url": "https://example.org/a.png",
                    "detail": "low",
                }
            ]
        },
        {
            "tools": [
                {
                    "type": "custom",
                    "name": "grammar",
                    "format": {"type": "grammar", "definition": "x"},
                }
            ]
        },
        {"tools": [{"type": "web_search"}]},
        {"text": {"verbosity": "high"}},
        {"text": {"format": {"type": "json_object"}}},
        {"metadata": {"unrepresentable": "value"}},
        {"reasoning": {"summary": "detailed"}},
        {"include": ["web_search_call.action.sources"]},
        {"truncation": "auto"},
        {"future_control": {"required": True}},
    ],
)
def test_unrepresentable_input_is_explicitly_rejected(payload: JsonObject) -> None:
    with pytest.raises(ResponsesConversionError):
        _build(payload)


def test_json_schema_output_preserves_structure_and_descriptive_metadata() -> None:
    built = _build(
        {
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "description": "Final answer",
                    "schema": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            }
        }
    )
    assert built.body["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
                "title": "answer",
                "description": "Final answer",
            },
        }
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"input": [{"type": []}]},
        {"tools": [{"type": {}}]},
        {"tool_choice": {"type": []}},
        {"input": [{"role": [], "content": "hi"}]},
        {"truncation": {}},
        {"reasoning": {"summary": []}},
    ],
)
def test_malformed_discriminators_produce_conversion_errors(
    payload: JsonObject,
) -> None:
    with pytest.raises(ResponsesConversionError):
        _build(payload)

"""Gemini input remains native until the selected transport converts it."""

import base64
from copy import deepcopy

import pytest
from pydantic import ValidationError

from free_claude_code.core.anthropic.native import NativeMessagesOptions
from free_claude_code.core.gemini import GeminiConversionError, GeminiRequest
from free_claude_code.core.gemini.replay import (
    CallReplay,
    ReplayBundle,
    apply_call_replay,
    decode_bundle,
    encode_bundle,
)
from free_claude_code.core.gemini.request import (
    build_gemini_chat_request,
    build_gemini_messages_request,
    build_gemini_responses_request,
)
from free_claude_code.core.gemini.schema import json_schema
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.reasoning import ReasoningPolicy


def request(**extra):
    return GeminiRequest.model_validate(
        {"model": "backend", "contents": [{"parts": [{"text": "hello"}]}], **extra}
    )


def build(value, protocol):
    if protocol == "chat":
        return build_gemini_chat_request(value)
    if protocol == "messages":
        return build_gemini_messages_request(
            value, NativeMessagesOptions(model=value.model, max_tokens=100)
        )
    return build_gemini_responses_request(value, ReasoningPolicy.provider_default())


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
def test_direct_builders_preserve_parallel_tool_history_and_schema(protocol):
    source = request(
        tools=[
            {
                "functionDeclarations": [
                    {
                        "name": "read.file",
                        "parametersJsonSchema": {
                            "type": "object",
                            "properties": {"path": {"type": "string", "default": None}},
                            "additionalProperties": False,
                        },
                    }
                ]
            }
        ],
        contents=[
            {"role": "user", "parts": [{"text": "Read both"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "id": "a",
                            "name": "read.file",
                            "args": {"path": "one"},
                        }
                    },
                    {
                        "functionCall": {
                            "id": "b",
                            "name": "read.file",
                            "args": {"path": "two"},
                        }
                    },
                ],
            },
            {
                "role": "model",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "b",
                            "name": "read.file",
                            "response": {"output": "two", "value": None},
                        }
                    },
                    {
                        "functionResponse": {
                            "id": "a",
                            "name": "read.file",
                            "response": {"output": "one"},
                        }
                    },
                ],
            },
        ],
    )
    before = source.model_dump()
    prepared = build(source, protocol)
    assert prepared.reserved_tool_ids == {"a", "b"}
    assert prepared.tool_schemas["read.file"]["properties"]["path"]["default"] is None
    assert prepared.tool_schemas["read.file"]["additionalProperties"] is False
    assert source.model_dump() == before
    assert prepared.body["model"] == "backend"
    values = prepared.body["input" if protocol == "responses" else "messages"]
    assert "two" in str(values) and "one" in str(values)
    assert (
        prepared.tool_names.decode(prepared.tool_names.encode("read.file"))
        == "read.file"
    )


def test_protocol_aliases_and_nulls_do_not_rewrite_arbitrary_json():
    value = request(
        generation_config={
            "temperature": 0,
            "thinking_config": {"thinking_budget": 0, "include_thoughts": False},
            "top_p": None,
        }
    )
    assert value.generation_config.temperature == 0
    assert value.reasoning_policy() == ReasoningPolicy.off()
    assert not value.include_thoughts
    body = build_gemini_chat_request(value).body
    assert body["temperature"] == 0 and "top_p" not in body
    with pytest.raises(ValidationError, match="Duplicate"):
        request(generationConfig={}, generation_config={})


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidateCount", 2),
        ("responseModalities", ["AUDIO"]),
        ("responseLogprobs", True),
    ],
)
def test_unsupported_generation_fields_are_explicit(field, value):
    with pytest.raises(ValidationError):
        request(generationConfig={field: value})


@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_destination_does_not_silently_drop_sampling_controls(protocol):
    with pytest.raises(GeminiConversionError, match="seed"):
        build(request(generationConfig={"seed": 42}), protocol)


def test_duplicate_named_calls_without_result_ids_are_not_guessed():
    value = request(
        contents=[
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "read", "args": {}}},
                    {"functionCall": {"name": "read", "args": {}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "read",
                            "response": {"output": "unknown"},
                        }
                    }
                ],
            },
        ]
    )
    with pytest.raises(GeminiConversionError, match="Ambiguous"):
        build_gemini_chat_request(value)


def test_gemini_schema_is_normalized_without_weakening_nullable_constraints():
    value = request(
        generationConfig={
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "answer": {"type": "STRING", "nullable": True, "enum": ["yes"]}
                },
                "required": ["answer"],
            },
        }
    )
    output = build_gemini_chat_request(value).body["response_format"]
    assert isinstance(output, dict)
    wrapper = output["json_schema"]
    assert isinstance(wrapper, dict)
    schema = wrapper["schema"]
    assert isinstance(schema, dict)
    assert schema["required"] == ["answer"]
    assert schema["properties"] == {
        "answer": {"anyOf": [{"type": "string", "enum": ["yes"]}, {"type": "null"}]}
    }


def test_call_replay_preserves_metadata_only_for_the_selected_connection():
    origin = ReplayOrigin(
        "google", "chat", "https://example.test", "connection-one", "model"
    )
    metadata: JsonObject = {"google": {"thought_signature": "source-opaque"}}
    record = CallReplay(origin, "a", metadata)
    bundle = ReplayBundle(
        "response",
        (
            ReplayRecord(
                origin,
                {"reasoning_details": [{"type": "reasoning.text", "text": "thought"}]},
            ),
        ),
        (record,),
    )
    encoded = encode_bundle(bundle)
    assert decode_bundle(encoded) == bundle
    body: JsonObject = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "function": {"name": "read", "arguments": "edited"}}
                ],
            }
        ]
    }
    before = deepcopy(body)
    restored = apply_call_replay(body, origin, bundle.calls)
    assert restored == {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "a",
                        "function": {"name": "read", "arguments": "edited"},
                        "extra_content": metadata,
                    }
                ],
            }
        ]
    }
    foreign = ReplayOrigin(
        "other", "chat", "https://else.test", "connection-two", "model"
    )
    assert apply_call_replay(body, foreign, bundle.calls) == before
    assert body == before
    assert (
        decode_bundle(base64.b64encode(b"unrecognized-google-bytes").decode()) is None
    )
    with pytest.raises(GeminiConversionError):
        decode_bundle(base64.b64encode(b"fcc:gemini-replay:v1:{}").decode())


def test_gemini_schema_proto_names_and_string_integers_preserve_constraints():
    assert json_schema(
        {
            "type": "OBJECT",
            "min_properties": "1",
            "properties": {
                "max_length": {"type": "STRING", "maxLength": "10", "default": None}
            },
        }
    ) == {
        "type": "object",
        "minProperties": 1,
        "properties": {
            "max_length": {"type": "string", "maxLength": 10, "default": None}
        },
    }


def test_conflicting_echoed_replay_bundles_are_not_silently_deduplicated():
    origin = ReplayOrigin("test", "responses", "endpoint", "account", "model")
    first = ReplayBundle(
        "same",
        (ReplayRecord(origin, {"type": "reasoning", "encrypted_content": "one"}),),
    )
    second = ReplayBundle(
        "same",
        (ReplayRecord(origin, {"type": "reasoning", "encrypted_content": "two"}),),
    )
    value = request(
        contents=[
            {
                "role": "model",
                "parts": [
                    {"text": "answer", "thoughtSignature": encode_bundle(first)},
                    {"thoughtSignature": encode_bundle(second)},
                ],
            },
            {"parts": [{"text": "continue"}]},
        ]
    )
    with pytest.raises(GeminiConversionError, match="Conflicting"):
        build_gemini_chat_request(value)

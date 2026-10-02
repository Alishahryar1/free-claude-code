"""Automatic recovery checks the request that the real provider client sends."""

from contextlib import asynccontextmanager
from copy import deepcopy

import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.openai_responses.tool_adaptation import ResponsesToolPolicy
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.core.recovery import CandidateIncompatible, RecoveryCheckpoint
from free_claude_code.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
)
from tests.providers.test_anthropic_messages_transport import Endpoint
from tests.providers.test_history_transports import _harness


@asynccontextmanager
async def _candidate(
    protocol, wire, raw, *, configure=None, reasoning=DEFAULT_REASONING_POLICY
):
    model = MessagesRequest if wire == "messages" else OpenAIResponsesRequest
    request = model.model_validate({"model": "model", **deepcopy(raw)})
    original = request.model_dump()
    async with _harness(protocol) as (_, bodies, provider):
        if configure is not None:
            configure(provider)
        options = {
            "request_id": "preservation",
            "reasoning": reasoning,
            "input_tokens": 0,
            "response_model": "public",
        }
        if protocol == "messages":
            options["endpoint_context"] = Endpoint()
        opener = (
            provider.open_messages if wire == "messages" else provider.open_responses
        )
        async with opener(request, **options) as candidate:
            yield candidate, bodies
    assert request.model_dump() == original


async def _send(candidate, wire, *, recovering=True):
    checkpoint = RecoveryCheckpoint(wire, recovering=recovering)
    return [
        event
        async for event in candidate.stream_attempt(
            checkpoint, wait_for_recovery=True, can_correct=lambda: False
        )
    ]


_TEXT = {"input": "hello"}
_MESSAGES = {"messages": [{"role": "user", "content": "hello"}]}
_FUNCTION = {
    "type": "function",
    "name": "inspect",
    "parameters": {"type": "object"},
    "strict": False,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_optional_reasoning_cannot_remove_explicit_recovery_output_limit(
    protocol,
):
    async with _candidate(
        protocol,
        "messages",
        {**_MESSAGES, "max_tokens": 4096},
        reasoning=ReasoningPolicy.prefer_off(),
    ) as (candidate, bodies):
        await _send(candidate, "messages")
        key = "max_output_tokens" if protocol == "responses" else "max_tokens"
        assert 0 < bodies[0][key] <= 4096


@pytest.mark.asyncio
async def test_native_messages_recovery_keeps_explicit_thinking_budget():
    thinking = {"type": "enabled", "budget_tokens": 1536, "display": "omitted"}
    async with _candidate(
        "messages", "messages", {**_MESSAGES, "max_tokens": 4096, "thinking": thinking}
    ) as (candidate, bodies):
        await _send(candidate, "messages")
        assert bodies[0]["thinking"] == thinking


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_target_without_effort_support_is_skipped_as_incompatible(wire):
    def configure(provider):
        provider._capabilities = MessagesModelCapabilities(supports_output_effort=False)

    raw = (
        {**_MESSAGES, "output_config": {"effort": "low"}}
        if wire == "messages"
        else {**_TEXT, "reasoning": {"effort": "low"}}
    )
    async with _candidate("messages", wire, raw, configure=configure) as (
        candidate,
        bodies,
    ):
        with pytest.raises(CandidateIncompatible):
            await _send(candidate, wire)
        assert bodies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
async def test_portable_image_reaches_the_final_recovery_body(protocol):
    url = "https://example.invalid/preserved-image.png"
    raw = {
        "input": [
            {"role": "user", "content": [{"type": "input_image", "image_url": url}]}
        ]
    }
    async with _candidate(protocol, "responses", raw) as (candidate, bodies):
        await _send(candidate, "responses")
        assert len(bodies) == 1 and url in str(bodies[0])


@pytest.mark.asyncio
async def test_normalized_native_recovery_does_not_drop_unknown_null_control():
    async with _candidate(
        "messages", "messages", {**_MESSAGES, "future_control": None}
    ) as (candidate, bodies):
        with pytest.raises(CandidateIncompatible):
            await _send(candidate, "messages")
        assert bodies == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Read the file"},
                        {
                            "type": "input_file",
                            "filename": "note.txt",
                            "file_data": "data:text/plain;base64,SEVMTE8=",
                        },
                    ],
                }
            ]
        },
        {**_TEXT, "include": ["message.output_text.logprobs"]},
        {**_TEXT, "truncation": "auto"},
        {**_TEXT, "text": {"verbosity": "low"}},
        {**_TEXT, "reasoning": {"summary": "detailed"}},
        {**_TEXT, "future_control": None},
        {"input": [{"type": "input_text", "text": "hello", "future_content": None}]},
        {
            "input": [
                {
                    "type": "output_text",
                    "text": "hello",
                    "annotations": [{"type": "citation", "text": "required"}],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "input_image",
                    "image_url": "https://example.invalid/image.png",
                    "future_image": None,
                }
            ]
        },
        {
            "input": [
                {
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "earlier",
                            "future_reasoning": None,
                        }
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello", "future_control": None}
                    ],
                }
            ]
        },
        {
            **_TEXT,
            "tools": [
                {
                    "type": "namespace",
                    "name": "ops",
                    "description": "Never remove files.",
                    "tools": [_FUNCTION],
                }
            ],
        },
        {
            **_TEXT,
            "tools": [
                {
                    "type": "custom",
                    "name": "answer",
                    "format": {
                        "type": "grammar",
                        "syntax": "regex",
                        "definition": "[0-9]+",
                    },
                }
            ],
        },
        {
            **_TEXT,
            "tools": [
                {key: value for key, value in _FUNCTION.items() if key != "strict"}
            ],
        },
        {**_TEXT, "tools": [{**_FUNCTION, "strict": None}]},
        {**_TEXT, "tools": [{**_FUNCTION, "defer_loading": True}]},
    ],
)
async def test_chat_recovery_skips_features_its_builder_would_discard(raw):
    async with _candidate("chat", "responses", raw) as (candidate, bodies):
        with pytest.raises(CandidateIncompatible):
            await _send(candidate, "responses")
        assert bodies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize(
    "raw",
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the document"},
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0=",
                            },
                        },
                    ],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hello",
                            "citations": [{"cited_text": "hello"}],
                        }
                    ],
                }
            ]
        },
        {
            **_MESSAGES,
            "tools": [
                {"name": "inspect", "input_schema": {"type": "object"}, "strict": True}
            ],
        },
        {
            **_MESSAGES,
            "tools": [{"name": "inspect", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        },
        {**_MESSAGES, "top_k": 5},
        {**_MESSAGES, "thinking": {"type": "enabled", "budget_tokens": 4096}},
    ],
)
async def test_messages_recovery_skips_unsupported_content_and_controls(protocol, raw):
    async with _candidate(protocol, "messages", raw) as (candidate, bodies):
        with pytest.raises(CandidateIncompatible):
            await _send(candidate, "messages")
        assert bodies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages"])
async def test_responses_omitted_strictness_is_not_chat_or_messages_default(protocol):
    raw = {
        **_TEXT,
        "tools": [{key: value for key, value in _FUNCTION.items() if key != "strict"}],
    }
    async with _candidate(protocol, "responses", raw) as (candidate, bodies):
        with pytest.raises(CandidateIncompatible):
            await _send(candidate, "responses")
        assert bodies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_portable_text_tools_and_caps_reach_real_http_client(protocol, wire):
    raw = (
        {
            **_MESSAGES,
            "tools": [{"name": "inspect", "input_schema": {"type": "object"}}],
            "max_tokens": 4096,
        }
        if wire == "messages"
        else {**_TEXT, "tools": [_FUNCTION], "max_output_tokens": 4096}
    )
    async with _candidate(protocol, wire, raw) as (candidate, bodies):
        await _send(candidate, wire)
        assert len(bodies) == 1
        assert (
            bodies[0]["max_output_tokens" if protocol == "responses" else "max_tokens"]
            == 4096
        )
        assert "hello" in str(bodies[0]) and "inspect" in str(bodies[0])


@pytest.mark.asyncio
async def test_same_revision_rebuilds_when_initial_route_enters_recovery():
    raw = {**_TEXT, "text": {"verbosity": "low"}}
    async with _candidate("chat", "responses", raw) as (candidate, bodies):
        await candidate.prepare(RecoveryCheckpoint("responses"))
        with pytest.raises(CandidateIncompatible):
            await _send(candidate, "responses")
        assert bodies == []


@pytest.mark.asyncio
async def test_initial_route_keeps_its_existing_adaptation():
    async with _candidate(
        "chat", "responses", {**_TEXT, "text": {"verbosity": "low"}}
    ) as (candidate, bodies):
        await _send(candidate, "responses", recovering=False)
        assert len(bodies) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("omitted", ["max_output_tokens", "metadata"])
async def test_native_provider_field_omission_cannot_weaken_recovery(omitted):
    def configure(provider):
        provider._omitted_request_fields = frozenset({omitted})

    raw = {**_TEXT, "max_output_tokens": 4096, "metadata": {"key": "value"}}
    async with _candidate("responses", "responses", raw, configure=configure) as (
        candidate,
        bodies,
    ):
        with pytest.raises(CandidateIncompatible):
            await _send(candidate, "responses")
        assert bodies == []


@pytest.mark.asyncio
async def test_same_protocol_custom_adapter_cannot_remove_grammar():
    def configure(provider):
        provider._tool_policy = ResponsesToolPolicy(custom_tools_as_functions=True)

    raw = {
        **_TEXT,
        "tools": [
            {
                "type": "custom",
                "name": "answer",
                "format": {
                    "type": "grammar",
                    "syntax": "regex",
                    "definition": "[0-9]+",
                },
            }
        ],
    }
    async with _candidate("responses", "responses", raw, configure=configure) as (
        candidate,
        bodies,
    ):
        with pytest.raises(CandidateIncompatible):
            await _send(candidate, "responses")
        assert bodies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "responses", "messages"])
async def test_original_native_origin_is_checked_before_readable_projection(protocol):
    carrier = encode_replay(
        ReplayRecord(
            ReplayOrigin(
                "foreign", "responses", "https://foreign.invalid", "connection", "old"
            ),
            {
                "type": "reasoning",
                "encrypted_content": "opaque",
                "summary": [{"type": "summary_text", "text": "Read me"}],
            },
        )
    )
    raw = {
        "input": [
            {"type": "reasoning", "encrypted_content": carrier, "summary": []},
            {"role": "user", "content": "hello"},
        ]
    }
    async with _candidate(protocol, "responses", raw) as (candidate, bodies):
        with pytest.raises(CandidateIncompatible):
            await _send(candidate, "responses")
        assert bodies == []


@pytest.mark.asyncio
async def test_native_recovery_preserves_unknown_null_control():
    async with _candidate(
        "responses", "responses", {**_TEXT, "future_control": None}
    ) as (candidate, bodies):
        await _send(candidate, "responses")
        assert "future_control" in bodies[0] and bodies[0]["future_control"] is None

"""Native Messages preserves wire content while applying FCC-owned controls."""

import json
from collections.abc import AsyncIterator

import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    NativeMessagesOptions,
    build_native_messages_request,
)
from free_claude_code.core.anthropic.sse_aggregation import (
    aggregate_anthropic_sse_to_message,
)
from free_claude_code.core.json_types import JsonObject


def test_native_body_rejects_nonfinite_json_before_http_serialization() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "public",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"extension": float("nan")},
        }
    )
    with pytest.raises(NativeMessagesError, match="finite JSON"):
        build_native_messages_request(
            request, options=NativeMessagesOptions("native", 4096)
        )


def test_native_body_preserves_blocks_and_merges_leading_system_in_order() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "public",
            "original_model": "private-routing",
            "max_tokens": 100,
            "stream": False,
            "system": "top",
            "messages": [
                {"role": "system", "content": "first"},
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "second",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "", "signature": "signed"},
                        {
                            "type": "tool_use",
                            "id": "call-a",
                            "name": "lookup",
                            "input": {"x": 1},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-a",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "url",
                                        "url": "https://example.org/image.png",
                                    },
                                }
                            ],
                            "is_error": False,
                        }
                    ],
                },
            ],
            "thinking": {"enabled": True, "type": "adaptive", "display": "omitted"},
            "output_config": {
                "effort": "high",
                "format": {"type": "json_schema", "schema": {"type": "object"}},
            },
            "extra_body": {"service_tier": "auto"},
            "betas": ["feature-2026-09-04", "feature-2026-09-04"],
        }
    )
    before = request.model_dump()
    prepared = build_native_messages_request(
        request,
        options=NativeMessagesOptions("upstream", 8192, {"type": "disabled"}),
    )
    assert prepared.betas == ("feature-2026-09-04",)
    assert prepared.body["model"] == "upstream"
    assert prepared.body["stream"] is True
    assert prepared.body["max_tokens"] == 8192
    assert prepared.body["thinking"] == {"type": "disabled"}
    assert prepared.body["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}}
    }
    assert prepared.body["service_tier"] == "auto"
    assert prepared.body["system"] == [
        {"type": "text", "text": "top"},
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second", "cache_control": {"type": "ephemeral"}},
    ]
    assert prepared.body["messages"] == [
        message.model_dump(exclude_none=True) for message in request.messages[2:]
    ]
    assert (
        not {"betas", "extra_body", "original_model", "resolved_provider_model"}
        & prepared.body.keys()
    )
    assert request.model_dump() == before


@pytest.mark.parametrize(
    "extra",
    [
        {"model": "other"},
        {"thinking": {"type": "enabled"}},
        {"Authorization": "secret"},
        {"headers": {"x-test": "x"}},
        {"max_tokens": 42},
        {"temperature": 0.9},
    ],
)
def test_extra_body_cannot_override_owned_or_explicit_fields(extra: JsonObject) -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.3,
            "extra_body": extra,
        }
    )
    with pytest.raises(NativeMessagesError, match="cannot override"):
        build_native_messages_request(request, options=NativeMessagesOptions("m", 1024))


def test_late_system_and_unsigned_compatibility_reasoning_rejected() -> None:
    for messages in (
        [{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}],
        [{"role": "assistant", "content": "hi", "reasoning_content": "unsigned"}],
    ):
        request = MessagesRequest.model_validate({"model": "m", "messages": messages})
        with pytest.raises(NativeMessagesError):
            build_native_messages_request(
                request, options=NativeMessagesOptions("m", 1024)
            )


def test_forced_tool_with_manual_thinking_rejected() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "any"},
        }
    )
    with pytest.raises(NativeMessagesError, match="force a tool"):
        build_native_messages_request(
            request,
            options=NativeMessagesOptions(
                "m", 4096, {"type": "enabled", "budget_tokens": 1024}
            ),
        )


def test_beta_header_injection_rejected() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "betas": ["bad\r\nAuthorization: secret"],
        }
    )
    with pytest.raises(NativeMessagesError, match="header tokens"):
        build_native_messages_request(request, options=NativeMessagesOptions("m", 1024))


@pytest.mark.asyncio
async def test_nonstreaming_aggregation_appends_split_signatures() -> None:
    events: list[JsonObject] = [
        {"type": "message_start", "message": {"id": "m", "model": "x", "usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": "start"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "-middle"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "-end"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 10},
        },
        {"type": "message_stop"},
    ]

    async def stream() -> AsyncIterator[str]:
        for event in events:
            frame = f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
            for pos in range(0, len(frame), 7):
                yield frame[pos : pos + 7]

    body, error, complete = await aggregate_anthropic_sse_to_message(stream())
    assert complete and error is None
    assert body["content"] == [
        {"type": "thinking", "thinking": "", "signature": "start-middle-end"}
    ]


def test_native_top_level_cache_and_tier_survive_and_cannot_be_shadowed() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "cache_control": {"type": "ephemeral"},
            "service_tier": "standard_only",
        }
    )
    body = build_native_messages_request(
        request, options=NativeMessagesOptions("m", 1024)
    ).body
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["service_tier"] == "standard_only"
    request.extra_body = {"service_tier": "auto"}
    with pytest.raises(NativeMessagesError, match="cannot override"):
        build_native_messages_request(request, options=NativeMessagesOptions("m", 1024))


def test_unknown_native_field_is_rejected_instead_of_lost() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "future_control": {"mode": "required"},
        }
    )
    with pytest.raises(NativeMessagesError, match="future_control"):
        build_native_messages_request(request, options=NativeMessagesOptions("m", 1024))


@pytest.mark.parametrize("value", [True, False])
def test_messages_wire_limit_rejects_boolean_json(value: object) -> None:
    with pytest.raises(ValueError, match="boolean"):
        MessagesRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": value,
            }
        )

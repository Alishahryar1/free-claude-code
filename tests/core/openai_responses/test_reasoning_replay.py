"""Signed native replay cannot leak into a different egress protocol."""

import base64
import json
from typing import cast

import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.history_replay import (
    HistoryReplayError,
    ReplayOrigin,
    ReplayRecord,
    decode_replay,
    encode_replay,
    prepare_history,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ReasoningBlockState,
    build_native_responses_request,
    build_responses_chat_request,
    reasoning_output_item,
)
from free_claude_code.core.reasoning import ReasoningPolicy

_ORIGIN = ReplayOrigin(
    "github_copilot/anthropic_messages", "messages", "", "", "upstream"
)


@pytest.mark.parametrize(
    "block",
    [
        {
            "type": "thinking",
            "thinking": "hello 🐍",
            "signature": "opaque",
            "extension": {"a": 1},
        },
        {"type": "thinking", "thinking": "", "signature": "signature-only"},
        {"type": "redacted_thinking", "data": "opaque-data"},
    ],
)
def test_replay_round_trip_preserves_exact_native_block(block: JsonObject) -> None:
    carrier = encode_replay(ReplayRecord(_ORIGIN, block))
    assert decode_replay(carrier).native == block
    assert decode_replay(carrier).origin == _ORIGIN


@pytest.mark.parametrize(
    "carrier",
    [
        "openai-opaque",
        "fcc:anthropic-reasoning:v2:e30",
        "fcc:anthropic-reasoning:v1:!bad",
        "fcc:anthropic-reasoning:v1:e30",  # empty JSON object
        "fcc:anthropic-reasoning:v1:W10",  # JSON array
    ],
)
def test_malformed_or_foreign_replay_rejected(carrier: str) -> None:
    with pytest.raises(HistoryReplayError):
        decode_replay(carrier)


def test_missing_signature_is_not_accepted_as_signed_history() -> None:
    data = {
        "protocol": "anthropic_messages",
        "replay_scope": _ORIGIN.provider,
        "source_model": "m",
        "block": {"type": "thinking", "thinking": "visible-only"},
    }
    carrier = (
        "fcc:anthropic-reasoning:v1:"
        + base64.urlsafe_b64encode(json.dumps(data).encode()).decode()
    )
    with pytest.raises(HistoryReplayError, match="signed"):
        decode_replay(carrier)


def test_completed_reasoning_keeps_both_display_and_opaque_replay() -> None:
    carrier = encode_replay(
        ReplayRecord(
            _ORIGIN, {"type": "thinking", "thinking": "visible", "signature": "opaque"}
        )
    )
    state = ReasoningBlockState(
        0, 0, "rs_1", text_parts=["vis", "ible"], encrypted_content=carrier
    )
    item = reasoning_output_item(state, status="completed")
    assert item["content"] == [{"type": "reasoning_text", "text": "visible"}]
    assert item["encrypted_content"] == carrier
    for egress in ("chat", "responses"):
        request = OpenAIResponsesRequest.model_validate({"model": "m", "input": [item]})
        if egress == "chat":
            body = build_responses_chat_request(
                request, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
            ).body
        else:
            body = build_native_responses_request(
                request, model="m", reasoning=ReasoningPolicy()
            )
        projected = prepare_history(
            cast(JsonObject, body), ReplayOrigin("other", egress, "", "", "m")
        )
        serialized = json.dumps(projected)
        assert "[Earlier reasoning]" in serialized
        assert "visible" in serialized
        assert "opaque" not in serialized
        assert carrier not in serialized


def test_singleton_reasoning_item_cannot_bypass_egress_guard() -> None:
    carrier = encode_replay(
        ReplayRecord(_ORIGIN, {"type": "redacted_thinking", "data": "opaque"})
    )
    request = OpenAIResponsesRequest(
        model="m", input={"type": "reasoning", "encrypted_content": carrier}
    )
    body = build_native_responses_request(
        request, model="m", reasoning=ReasoningPolicy()
    )
    assert (
        prepare_history(body, ReplayOrigin("other", "responses", "", "", "m"))["input"]
        == []
    )
    chat = build_responses_chat_request(
        request, reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
    )
    assert carrier not in json.dumps(
        prepare_history(
            cast(JsonObject, chat.body), ReplayOrigin("other", "chat", "", "", "m")
        )
    )


@pytest.mark.parametrize("value", [True, False])
def test_responses_wire_limit_rejects_boolean_json(value: object) -> None:
    with pytest.raises(ValueError, match="boolean"):
        OpenAIResponsesRequest.model_validate(
            {
                "model": "m",
                "input": "hi",
                "max_output_tokens": value,
            }
        )


@pytest.mark.parametrize(
    "extension",
    ["NaN", "Infinity", "-Infinity", "1e999", "[" * 1100 + "0" + "]" * 1100],
    ids=["nan", "infinity", "negative-infinity", "overflow", "deep"],
)
def test_carrier_cannot_hide_non_json_numbers_or_excessive_nesting(
    extension: str,
) -> None:
    raw = (
        '{"protocol":"anthropic_messages","replay_scope":"github_copilot/anthropic_messages",'
        '"source_model":"m","block":{"type":"thinking","thinking":"ok","signature":"sig","extension":'
        + extension
        + "}}"
    )
    carrier = (
        "fcc:anthropic-reasoning:v1:" + base64.urlsafe_b64encode(raw.encode()).decode()
    )
    with pytest.raises(HistoryReplayError):
        decode_replay(carrier)


def test_encoder_rejects_non_finite_extension_values() -> None:
    with pytest.raises(HistoryReplayError, match="finite"):
        encode_replay(
            ReplayRecord(
                _ORIGIN,
                {
                    "type": "thinking",
                    "thinking": "ok",
                    "signature": "sig",
                    "extension": float("nan"),
                },
            )
        )


def test_decoder_rejects_unpaired_surrogate_before_http_serialization() -> None:
    raw = (
        '{"protocol":"anthropic_messages","replay_scope":"github_copilot/anthropic_messages",'
        '"source_model":"m","block":{"type":"thinking","thinking":"'
        + chr(92)
        + 'ud800","signature":"sig"}}'
    )
    carrier = (
        "fcc:anthropic-reasoning:v1:" + base64.urlsafe_b64encode(raw.encode()).decode()
    )
    with pytest.raises(HistoryReplayError):
        decode_replay(carrier)

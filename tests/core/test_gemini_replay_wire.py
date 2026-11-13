"""Native replay uses the byte-field contract without changing record ownership."""

import base64
from dataclasses import replace
from decimal import Decimal
from typing import cast

import pytest
from pydantic import ValidationError

from free_claude_code.core.continuation_request import ContinuationRequest
from free_claude_code.core.history_replay import (
    HistoryReplayError,
    decode_gemini_replay,
    encode_gemini_replay,
    encode_replay,
    prepare_history,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_recovery import ContinuationSeed
from tests.core.test_gemini_continuation_records import recovered, request
from tests.core.test_native_gemini_history import ORIGIN


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("urlsafe", [False, True])
def test_native_bytes_can_be_reencoded_by_clients(padded, urlsafe):
    record = recovered()
    raw = encode_replay(record).encode()
    encoded = (base64.urlsafe_b64encode if urlsafe else base64.b64encode)(raw).decode()
    if not padded:
        encoded = encoded.rstrip("=")
    assert decode_gemini_replay(encoded) == record
    body = request(record)
    body["contents"][1]["parts"][-1]["thoughtSignature"] = encoded
    restored = prepare_history(body, ORIGIN)
    assert restored["contents"] == [
        body["contents"][0],
        *record.continuation.context,
        record.native,
        body["contents"][-1],
    ]


@pytest.mark.parametrize(
    "raw", [b"fcc:history:v1:invalid", b"fcc:history:v9:anything", b"fcc:history:\xff"]
)
def test_recognized_malformed_native_replay_is_rejected(raw):
    with pytest.raises(HistoryReplayError):
        decode_gemini_replay(base64.b64encode(raw).decode())


@pytest.mark.parametrize(
    "value",
    [
        "ordinary-signature",
        "not base64!",
        base64.b64encode(b"provider-owned bytes").decode(),
    ],
)
def test_non_fcc_signature_is_not_reinterpreted(value):
    assert decode_gemini_replay(value) is None


def test_final_wrapped_size_is_bounded_on_encode_and_decode(monkeypatch):
    record = recovered()
    inner = encode_replay(record)
    wrapped = encode_gemini_replay(record)
    monkeypatch.setattr(
        "free_claude_code.core.history_replay.RECOVERY_RETENTION_BYTES",
        len(wrapped) - 1,
    )
    assert encode_replay(record) == inner
    with pytest.raises(HistoryReplayError, match="retention"):
        encode_gemini_replay(record)
    with pytest.raises(HistoryReplayError, match="retention"):
        decode_gemini_replay(wrapped)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_public_echo_rejects_nonrepresentable_numbers(number):
    seed = ContinuationSeed(
        "gemini",
        (
            cast(
                JsonObject,
                {"functionCall": {"name": "read", "args": {"amount": Decimal(number)}}},
            ),
        ),
    )
    with pytest.raises(ValidationError):
        ContinuationRequest("gemini").history(
            seed, {"model": "model"}, allow_pending_calls=True
        )


@pytest.mark.parametrize("edit", ["text", "origin", "draft"])
def test_native_wrapper_preserves_provenance_and_echo_checks(edit):
    record = recovered(echo=edit != "draft")
    body = request(record)
    body["contents"][1]["parts"][-1]["thoughtSignature"] = encode_gemini_replay(record)
    origin = ORIGIN
    if edit == "text":
        body["contents"][1]["parts"][0]["text"] = "edited"
    elif edit == "origin":
        origin = replace(origin, connection="other")
    with pytest.raises(HistoryReplayError):
        prepare_history(body, origin)

"""Protected Gemini metadata excludes exactly its own displayed echo."""

from dataclasses import replace
from hashlib import sha256
from typing import Any, cast

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.gemini.errors import GeminiConversionError
from free_claude_code.core.gemini.replay import (
    CallReplay,
    ReplayBundle,
    decode_bundle,
    encode_bundle,
)
from free_claude_code.core.gemini.response import (
    GeminiOutput,
    GeminiResponseAccumulator,
)
from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    prepare_history,
    readable_reasoning,
)
from free_claude_code.core.json_types import JsonObject
from tests.core.test_gemini_requests import build, request

ORIGIN = ReplayOrigin("provider", "chat", "https://source.invalid", "account", "model")


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("visible", [True, False])
@pytest.mark.parametrize("edited", [True, False])
@pytest.mark.parametrize("thought", ["Same", "🧠é"])
def test_mixed_equal_thoughts_keep_only_protected_echo_exemption(
    protocol, visible, edited, thought
):
    output = GeminiOutput("response", "model", include_thoughts=visible)
    native: JsonObject = {
        "reasoning_details": [
            {"type": "reasoning.text", "text": thought, "signature": "signed"}
        ]
    }
    frames = [
        *output.part({"thought": True, "text": thought[:1]}, source=1),
        *output.part({"thought": True, "text": thought}, source=2),
        *output.part({"thought": True, "text": thought[1:]}, source=1),
    ]
    output.capture(ReplayRecord(ORIGIN, native), source=1)
    output.capture(ReplayRecord(ORIGIN, {"reasoning_content": thought}), source=2)
    frames.extend(output.finish("STOP"))
    aggregate = GeminiResponseAccumulator()
    for event in parse_sse_text("".join(frames)):
        aggregate.feed(event.data)
    parts = aggregate.result()["candidates"][0]["content"]["parts"]
    bundle = decode_bundle(parts[-1]["thoughtSignature"])
    assert bundle is not None and [record.native for record in bundle.records] == [
        native
    ]
    if visible and edited:
        parts[0]["text"] = "Edited"
    incoming = request(
        contents=[{"role": "model", "parts": parts}, {"parts": [{"text": "next"}]}]
    )
    prepared = cast(
        dict[str, Any],
        prepare_history(
            build(incoming, protocol).body, replace(ORIGIN, protocol=protocol)
        ),
    )
    if protocol == "chat":
        assistant = prepared["messages"][0]
        assert assistant["reasoning_details"] == native["reasoning_details"]
        assert (
            assistant.get("reasoning_content", "") == ("Edited" if edited else thought)
            if visible
            else not assistant.get("reasoning_content")
        )
    # Foreign protected state keeps its existing fallback; ordinary thinking is separate.
    ordinary = []
    if protocol == "responses":
        ordinary = [
            text
            for item in prepared["input"]
            if item.get("type") == "reasoning"
            for text, _ in readable_reasoning(item)
        ]
    elif protocol == "messages":
        ordinary = [
            block["thinking"]
            for message in prepared["messages"]
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "thinking"
        ]
    if protocol != "chat":
        assert "".join(ordinary) == (
            ("Edited" if edited else thought) if visible else ""
        )


@pytest.mark.parametrize(
    "ranges", [((1, 0, 1),), ((0, -1, 1),), ((0, 1, 1),), ((0, 0, 2), (0, 1, 3))]
)
def test_malformed_protected_ranges_are_rejected(ranges):
    record = ReplayRecord(
        ORIGIN,
        {
            "reasoning_details": [
                {"type": "reasoning.text", "text": "abc", "signature": "signed"}
            ]
        },
    )
    signature = encode_bundle(
        ReplayBundle(
            "response",
            (record,),
            thought_echo_sha256=sha256(b"abc").hexdigest(),
            protected_echo=ranges,
        )
    )
    with pytest.raises(GeminiConversionError, match="Malformed"):
        decode_bundle(signature)


def test_protected_ranges_cannot_exceed_verified_visible_text():
    record = ReplayRecord(
        ORIGIN,
        {
            "reasoning_details": [
                {"type": "reasoning.text", "text": "a", "signature": "signed"}
            ]
        },
    )
    signature = encode_bundle(
        ReplayBundle(
            "response",
            (record,),
            thought_echo_sha256=sha256(b"a").hexdigest(),
            protected_echo=((0, 0, 2),),
        )
    )
    incoming = request(
        contents=[
            {
                "role": "model",
                "parts": [
                    {"thought": True, "text": "a"},
                    {"thoughtSignature": signature},
                ],
            }
        ]
    )
    with pytest.raises(GeminiConversionError, match="exceeds thought text"):
        build(incoming, "chat")


def test_protected_pending_metadata_is_bounded_at_capture(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.core.gemini.response.RECOVERY_RETENTION_BYTES", 100
    )
    output = GeminiOutput("response", "model", include_thoughts=False)
    with pytest.raises(ExecutionFailure, match="retention limit"):
        output.capture(
            ReplayRecord(
                ORIGIN,
                {
                    "reasoning_details": [
                        {"type": "reasoning.encrypted", "data": "x" * 200}
                    ]
                },
            )
        )


def test_required_call_metadata_is_bounded_before_finish(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.core.gemini.response.RECOVERY_RETENTION_BYTES", 100
    )
    output = GeminiOutput("response", "model", include_thoughts=False)
    with pytest.raises(ExecutionFailure, match="retention limit"):
        output.capture_call(
            CallReplay(ORIGIN, "call", {"google": {"thought_signature": "x" * 200}})
        )


def test_signature_only_record_does_not_claim_independent_readable_text():
    output = GeminiOutput("response", "model", include_thoughts=True)
    output.part({"thought": True, "text": "Ordinary"}, source=0)
    output.capture(
        ReplayRecord(
            ORIGIN,
            {
                "reasoning_details": [],
                "extra_content": {"google": {"thought_signature": "signed"}},
            },
        ),
        source=0,
    )
    event = parse_sse_text("".join(output.finish("STOP")))[0]
    bundle = decode_bundle(
        event.data["candidates"][0]["content"]["parts"][0]["thoughtSignature"]
    )
    assert bundle is not None
    assert len(bundle.records) == 1
    assert not bundle.protected_echo and bundle.thought_echo_sha256 is None

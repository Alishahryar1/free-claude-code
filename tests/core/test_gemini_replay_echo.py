"""A thought signature preserves source state without swallowing caller text."""

import json
from dataclasses import replace

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.gemini.replay import (
    ReplayBundle,
    decode_bundle,
    encode_bundle,
)
from free_claude_code.core.gemini.response import (
    GeminiOutput,
    GeminiResponseAccumulator,
)
from free_claude_code.core.history_replay import (
    HistoryReplayError,
    HistoryScope,
    ReplayOrigin,
    ReplayRecord,
    decode_replay,
    encode_replay,
    prepare_history,
    readable_reasoning,
)
from tests.core.test_gemini_requests import build, request

ORIGIN = ReplayOrigin("provider", "chat", "https://source.invalid", "account", "model")


def generated():
    output = GeminiOutput("response", "model", include_thoughts=True)
    output.records.append(ReplayRecord(ORIGIN, {"reasoning_content": "native thought"}))
    frames = [
        *output.part({"text": "native ", "thought": True}),
        *output.part({"text": "thought", "thought": True}),
        *output.part({"text": "answer"}),
        *output.finish("STOP"),
    ]
    aggregate = GeminiResponseAccumulator()
    for event in parse_sse_text("".join(frames)):
        aggregate.feed(event.data)
    return aggregate.result()["candidates"][0]["content"]["parts"]


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize(
    "echo",
    [
        "coalesced",
        "fragmented",
        "relocated",
        "duplicate_bundle",
        "changed",
        "missing_digest",
        "multiple_bundles",
    ],
)
def test_only_exact_single_bundle_echo_is_suppressed(protocol, echo):
    parts = generated()
    signature = parts[-1]["thoughtSignature"]
    bundle = decode_bundle(signature)
    assert bundle is not None
    assert bundle.thought_echo_sha256
    if echo == "fragmented":
        parts[:1] = [
            {"text": "nat", "thought": True},
            {"text": "ive thought", "thought": True},
        ]
    elif echo == "relocated":
        parts[0]["thoughtSignature"] = signature
        parts.pop()
    elif echo == "duplicate_bundle":
        parts[0]["thoughtSignature"] = signature
    elif echo == "changed":
        parts[0]["text"] = "different thought"
    elif echo == "missing_digest":
        parts[-1]["thoughtSignature"] = encode_bundle(
            replace(bundle, thought_echo_sha256=None)
        )
    elif echo == "multiple_bundles":
        parts.append(
            {
                "thoughtSignature": encode_bundle(
                    ReplayBundle(
                        "second",
                        (
                            ReplayRecord(
                                ORIGIN, {"reasoning_content": "independent thought"}
                            ),
                        ),
                    )
                )
            }
        )
    source = request(
        contents=[{"role": "model", "parts": parts}, {"parts": [{"text": "next"}]}]
    )
    original = source.model_dump(mode="json")
    prepared = build(source, protocol).body
    final = prepare_history(
        prepared, replace(ORIGIN, protocol=protocol), scope=HistoryScope.ALL
    )
    wire = json.dumps(final)
    assert wire.count("answer") == 1
    assert wire.count("native thought") == (
        2 if echo in {"missing_digest", "multiple_bundles"} else 1
    )
    if echo == "changed":
        assert wire.count("different thought") == 1
    if echo == "multiple_bundles":
        assert wire.count("independent thought") == 1
    assert source.model_dump(mode="json") == original


@pytest.mark.parametrize(
    "native",
    [
        {"reasoning": "text"},
        {"reasoning_content": "text"},
        {"reasoning": "a", "reasoning_content": "b", "reasoning_details": []},
    ],
)
def test_plain_native_chat_replay_round_trips(native):
    record = ReplayRecord(ORIGIN, native)
    assert decode_replay(encode_replay(record)) == record


@pytest.mark.parametrize(
    "native",
    [
        {},
        {"reasoning": []},
        {"reasoning_content": 7},
        {"reasoning_details": [], "reasoning": None},
        {"reasoning_details": "bad", "reasoning_content": "good"},
    ],
)
def test_native_chat_replay_rejects_invalid_recognized_fields(native):
    with pytest.raises(HistoryReplayError):
        encode_replay(ReplayRecord(ORIGIN, native))


@pytest.mark.parametrize("digest", [None, "bad", 17, "g" * 64])
def test_bundle_rejects_invalid_echo_digest(digest):
    import base64

    from free_claude_code.core.gemini import GeminiConversionError

    value = base64.b64decode(encode_bundle(ReplayBundle("response")))
    prefix, payload = value.split(b":v1:", 1)
    data = json.loads(payload)
    data["thought_echo_sha256"] = digest
    invalid = base64.b64encode(prefix + b":v1:" + json.dumps(data).encode()).decode()
    with pytest.raises(GeminiConversionError):
        decode_bundle(invalid)


def test_native_aggregate_alias_does_not_duplicate_structured_full_text():
    native = {
        "reasoning_content": "firstsecond",
        "reasoning_details": [
            {"type": "reasoning.text", "text": "first"},
            {"type": "reasoning.text", "text": "second"},
            {"type": "reasoning.summary", "summary": "brief"},
        ],
    }
    assert readable_reasoning(native) == [
        ("first", False),
        ("second", False),
        ("brief", True),
    ]

"""Portable history preserves originals while adapting each destination copy."""

import json
from copy import deepcopy
from typing import Any, cast

import pytest

from free_claude_code.core.history_replay import (
    HistoryReplayError,
    HistoryScope,
    ReplayOrigin,
    ReplayRecord,
    decode_replay,
    encode_replay,
    prepare_history,
    responses_replay_item,
)


def _origin(
    provider="a", protocol="responses", connection="connection-a", model="arbitrary"
):
    return ReplayOrigin(provider, protocol, "https://example.org/v1", connection, model)


def _native():
    return {
        "type": "reasoning",
        "id": "rs_native:original",
        "summary": [],
        "encrypted_content": "opaque-native",
        "native_extension": {"v": 17},
    }


def test_response_history_roundtrip_across_providers_and_restart():
    native = _native()
    item = responses_replay_item(native, _origin())
    saved = {
        "input": [
            item,
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "continue"},
        ]
    }
    original = deepcopy(saved)
    foreign = prepare_history(saved, _origin("b"))
    assert foreign["input"] == saved["input"][1:]
    restored = cast(
        dict[str, Any],
        prepare_history(
            json.loads(json.dumps(saved)), _origin(model="completely-different")
        ),
    )
    assert restored["input"][0] == native
    assert saved == original
    assert prepare_history(saved, _origin("b")) == foreign


@pytest.mark.parametrize("protocol", ["responses", "messages", "chat"])
def test_foreign_readable_reasoning_keeps_context_without_ciphertext(protocol):
    native = {
        **_native(),
        "summary": [{"type": "summary_text", "text": "Search for the value."}],
    }
    carrier = encode_replay(ReplayRecord(_origin(), native))
    if protocol == "responses":
        saved = {
            "input": [
                {
                    "type": "reasoning",
                    "encrypted_content": carrier,
                    "summary": native["summary"],
                },
                {"role": "user", "content": "continue"},
            ]
        }
    elif protocol == "messages":
        saved = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": carrier}],
                },
                {"role": "user", "content": "continue"},
            ]
        }
    else:
        saved = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_details": [
                        {"type": "reasoning.encrypted", "data": carrier}
                    ],
                },
                {"role": "user", "content": "continue"},
            ]
        }
    result = prepare_history(saved, _origin("b", protocol))
    rendered = json.dumps(result)
    assert "Search for the value." in rendered
    assert "summary" in rendered.lower()
    assert "opaque-native" not in rendered
    assert "fcc:" not in rendered
    assert all(
        item.get("role") not in {"system", "developer"}
        for item in result.get("input", result.get("messages", []))
    )


@pytest.mark.parametrize(
    "block",
    [
        {
            "type": "thinking",
            "thinking": "",
            "signature": "empty-signed",
            "extension": 17,
        },
        {"type": "thinking", "thinking": "keep exactly", "signature": "signature"},
        {"type": "redacted_thinking", "data": "hidden"},
    ],
)
def test_messages_native_blocks_are_exact_after_carrier_replay(block):
    origin = _origin(protocol="messages")
    carrier = encode_replay(ReplayRecord(origin, block))
    public = (
        {**block, "signature": carrier}
        if block["type"] == "thinking"
        else {"type": "redacted_thinking", "data": carrier}
    )
    saved = {"messages": [{"role": "assistant", "content": [public]}]}
    result = cast(dict[str, Any], prepare_history(saved, origin))
    assert result["messages"][0]["content"] == [block]


def test_native_replay_connection_is_scoped_but_model_name_is_not_a_rule():
    item = responses_replay_item(_native(), _origin())
    assert (
        prepare_history({"input": [item]}, _origin(connection="other"))["input"] == []
    )
    assert prepare_history({"input": [item]}, _origin(model="any-model"))["input"] == [
        _native()
    ]


def test_codec_rejects_malformed_and_nonfinite_payloads():
    with pytest.raises(HistoryReplayError):
        decode_replay("fcc:history:v1:broken")
    with pytest.raises(HistoryReplayError):
        encode_replay(
            ReplayRecord(_origin(), {"type": "reasoning", "extension": float("nan")})
        )


def test_legacy_naked_opaque_history_is_tried_unchanged():
    saved = {"input": [_native()]}
    assert prepare_history(saved, _origin("other")) == saved


@pytest.mark.parametrize("native_text", ["Full reasoning.", "Short summary."])
def test_chat_replay_keeps_full_text_and_summary_meaning(native_text):
    native = {
        "reasoning_content": native_text,
        "reasoning_details": [
            {"type": "reasoning.summary", "summary": "Short summary.", "index": 0}
        ],
    }
    carrier = encode_replay(ReplayRecord(_origin(protocol="chat"), native))
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": carrier}],
            }
        ]
    }
    original = deepcopy(body)
    result = cast(
        dict[str, Any],
        prepare_history(body, _origin("b", "chat"), scope=HistoryScope.ALL),
    )["messages"][0]
    assert result["content"] == "answer\n\n[Earlier reasoning summary]\nShort summary."
    if native_text == "Short summary.":
        assert "reasoning_content" not in result
    else:
        assert result["reasoning_content"] == "Full reasoning."
    assert body == original

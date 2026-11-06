"""Native signatures retain their Part association across client protocols."""

from copy import deepcopy
from typing import Any, cast

import pytest

from free_claude_code.core.history_replay import (
    HistoryMode,
    HistoryReplayError,
    ReplayOrigin,
    ReplayRecord,
    decode_replay,
    encode_replay,
    prepare_history,
)

ORIGIN = ReplayOrigin(
    "gemini",
    "gemini",
    "https://generativelanguage.googleapis.com/v1beta",
    "key-hash",
    "models/model",
)


def test_native_signature_record_round_trip_keeps_every_part():
    native: dict[str, Any] = {
        "role": "model",
        "parts": [
            {"text": "editable thought", "thought": True},
            {"text": "signed answer", "thoughtSignature": "actual-google-signature"},
            {
                "functionCall": {"id": "call_1", "name": "read", "args": {}},
                "thoughtSignature": "call-signature",
            },
            {"thoughtSignature": "late-signature"},
        ],
    }
    record = ReplayRecord(ORIGIN, native)
    assert decode_replay(encode_replay(record)) == record


@pytest.mark.parametrize("mode", list(HistoryMode))
def test_native_signature_parts_survive_history_projection(mode):
    body = {
        "model": "models/model",
        "contents": [
            {
                "role": "model",
                "parts": [
                    {"text": "old thought", "thought": True},
                    {"text": "protected", "thought": True, "thoughtSignature": "sig"},
                    {"text": "answer"},
                ],
            },
            {"role": "user", "parts": [{"text": "next"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "current", "thought": True},
                    {"functionCall": {"id": "call_1", "name": "read", "args": {}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "call_1",
                            "name": "read",
                            "response": {"output": "ok"},
                        }
                    }
                ],
            },
        ],
    }
    original = deepcopy(body)
    result = cast(dict[str, Any], prepare_history(body, ORIGIN, scope=mode))
    assert body == original
    assert {"text": "protected", "thought": True, "thoughtSignature": "sig"} in result[
        "contents"
    ][0]["parts"]
    assert (
        {"text": "old thought", "thought": True} in result["contents"][0]["parts"]
    ) == (mode is HistoryMode.ALL)
    assert ({"text": "current", "thought": True} in result["contents"][2]["parts"]) == (
        mode is not HistoryMode.NONE
    )


def test_restore_native_content_from_messages_echo_and_reject_edits():
    native: dict[str, Any] = {
        "role": "model",
        "parts": [
            {"text": "thinking", "thought": True},
            {"text": "answer", "thoughtSignature": "sig"},
        ],
    }
    carrier = encode_replay(ReplayRecord(ORIGIN, native))
    body = {
        "contents": [
            {
                "role": "model",
                "parts": [{"text": "answer"}, {"thoughtSignature": carrier}],
            }
        ]
    }
    result = cast(dict[str, Any], prepare_history(body, ORIGIN, scope=HistoryMode.NONE))
    assert result["contents"] == [
        {"role": "model", "parts": [{"text": "answer", "thoughtSignature": "sig"}]}
    ]
    body["contents"][0]["parts"][0]["text"] = "edited"
    with pytest.raises(HistoryReplayError, match="edited"):
        prepare_history(body, ORIGIN)


def test_native_record_does_not_cross_credentials():
    carrier = encode_replay(
        ReplayRecord(
            ORIGIN,
            {"role": "model", "parts": [{"text": "answer", "thoughtSignature": "sig"}]},
        )
    )
    other = ReplayOrigin(
        "gemini", "gemini", ORIGIN.endpoint, "different-key", ORIGIN.model
    )
    with pytest.raises(HistoryReplayError, match="connection"):
        prepare_history(
            {
                "contents": [
                    {
                        "role": "model",
                        "parts": [{"text": "answer"}, {"thoughtSignature": carrier}],
                    }
                ]
            },
            other,
        )


@pytest.mark.parametrize("protocol", ["messages", "responses", "chat"])
def test_native_gemini_protected_state_is_not_degraded_on_foreign_egress(protocol):
    carrier = encode_replay(
        ReplayRecord(
            ORIGIN,
            {
                "role": "model",
                "parts": [{"text": "protected answer", "thoughtSignature": "sig"}],
            },
        )
    )
    if protocol == "messages":
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": carrier}],
                }
            ]
        }
    elif protocol == "responses":
        body = {"input": [{"type": "reasoning", "encrypted_content": carrier}]}
    else:
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "protected answer",
                    "reasoning_details": [
                        {"type": "reasoning.encrypted", "data": carrier}
                    ],
                }
            ]
        }
    destination = ReplayOrigin("different", protocol, "different", "different", "model")
    with pytest.raises(HistoryReplayError, match="native Gemini"):
        prepare_history(body, destination)


@pytest.mark.parametrize("added", [False, True])
def test_native_signed_call_accepts_only_declared_default_additions(added):
    native: dict[str, Any] = {
        "role": "model",
        "parts": [
            {
                "functionCall": {
                    "id": "call",
                    "name": "edit",
                    "args": {"text": "original"},
                },
                "thoughtSignature": "sig",
            }
        ],
    }
    body = {
        "model": ORIGIN.model,
        "contents": [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "id": "call",
                            "name": "edit",
                            "args": {"text": "original", "replace_all": added},
                        }
                    },
                    {
                        "thoughtSignature": encode_replay(
                            ReplayRecord(
                                ORIGIN,
                                native,
                                tool_defaults={"edit": {"replace_all": False}},
                            )
                        )
                    },
                ],
            }
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "edit",
                        "parametersJsonSchema": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "replace_all": {"type": "boolean", "default": False},
                            },
                        },
                    }
                ]
            }
        ],
    }
    body.pop("tools")
    if added:
        with pytest.raises(HistoryReplayError, match="edited"):
            prepare_history(body, ORIGIN)
    else:
        assert (
            cast(dict[str, Any], prepare_history(body, ORIGIN))["contents"][0] == native
        )

"""Signed continuation history survives serialization without server session state."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from free_claude_code.core.continuation_request import ContinuationRequest
from free_claude_code.core.history_replay import (
    ContinuationReplay,
    HistoryMode,
    HistoryReplayError,
    ReplayRecord,
    decode_replay,
    encode_gemini_replay,
    encode_replay,
    prepare_history,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_recovery import ContinuationSeed
from tests.core.test_native_gemini_history import ORIGIN


def recovered(*, echo=True):
    return ReplayRecord(
        ORIGIN,
        {
            "role": "model",
            "parts": [
                {"text": "hidden native thought", "thought": True},
                {"text": "suffix", "thoughtSignature": "native-signature"},
                {"text": "", "thoughtSignature": "late-signature"},
            ],
        },
        continuation=ContinuationReplay(
            (
                {"role": "model", "parts": [{"text": "prefix "}]},
                {"role": "user", "parts": [{"text": "Continue without repeating."}]},
            ),
            ({"text": "prefix suffix"},) if echo else None,
        ),
    )


def request(record):
    return {
        "contents": [
            {"role": "user", "parts": [{"text": "original"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "prefix suffix"},
                    {"thoughtSignature": encode_replay(record)},
                ],
            },
            {"role": "user", "parts": [{"text": "next"}]},
        ]
    }


@pytest.mark.parametrize("mode", list(HistoryMode))
@pytest.mark.parametrize("removed_prefix", [False, True])
def test_roundtrip_restores_exact_context_and_native_parts(mode, removed_prefix):
    record = recovered()
    if removed_prefix:
        record = replace(
            record,
            continuation=replace(
                record.continuation, context=record.continuation.context[1:]
            ),
        )
    encoded = encode_replay(record)
    assert encode_replay(decode_replay(encoded)) == encoded
    body = request(record)
    original = deepcopy(body)
    value = prepare_history(body, ORIGIN, scope=mode)
    assert value["contents"] == [
        body["contents"][0],
        *record.continuation.context,
        record.native,
        body["contents"][-1],
    ]
    assert body == original


@pytest.mark.parametrize("damage", ["text", "origin", "unfinished"])
def test_invalid_echo_or_origin_never_replays(damage):
    record = recovered(echo=damage != "unfinished")
    body = request(record)
    origin = ORIGIN
    if damage == "text":
        body["contents"][1]["parts"][0]["text"] = "edited"
    if damage == "origin":
        origin = replace(ORIGIN, connection="other-account")
    with pytest.raises(HistoryReplayError):
        prepare_history(body, origin)


@pytest.mark.parametrize(
    "damage",
    [
        "role",
        "signed_context",
        "signed_echo",
        "foreign_protocol",
        "empty_context",
        "model_only",
        "extra_context",
    ],
)
def test_continuation_metadata_is_validated(damage):
    record = recovered()
    context: list[JsonObject] = deepcopy(list(record.continuation.context))
    echo = deepcopy(list(record.continuation.echo))
    if damage == "role":
        context[0]["role"] = "user"
    if damage == "signed_context":
        context[0]["parts"] = [{"text": "prefix", "thoughtSignature": "unsafe"}]
    if damage == "signed_echo":
        echo[0]["thoughtSignature"] = "unsafe"
    if damage == "empty_context":
        context = []
    if damage == "model_only":
        context = context[:1]
    if damage == "extra_context":
        context.insert(0, {"role": "user", "parts": [{"text": "unrelated"}]})
    record = replace(
        record, continuation=ContinuationReplay(tuple(context), tuple(echo))
    )
    if damage == "foreign_protocol":
        record = replace(record, origin=replace(ORIGIN, protocol="chat"))
    with pytest.raises(HistoryReplayError):
        encode_replay(record)


def test_complete_encoded_record_is_bounded(monkeypatch):
    record = recovered()
    size = len(encode_replay(record))
    monkeypatch.setattr(
        "free_claude_code.core.history_replay.RECOVERY_RETENTION_BYTES", size - 1
    )
    with pytest.raises(HistoryReplayError, match="retention"):
        encode_replay(record)


def test_captures_corrected_sent_context():
    owner = ContinuationRequest("gemini")
    assert owner.sent_context({"contents": []}, source={}, source_paths={}) is None
    body = owner.build(
        {"contents": [{"role": "user", "parts": [{"text": "old"}]}]},
        ContinuationSeed("messages", ({"type": "text", "text": "prefix"},)),
    )
    body["contents"][-2]["parts"].append({"text": "", "thought": True})
    paths = {}
    sent = prepare_history(body, ORIGIN, source_paths=paths, continuation=True)
    captured = owner.sent_context(sent, source=body, source_paths=paths)
    assert captured == tuple(body["contents"][-2:])
    body["contents"][-2]["parts"][0]["text"] = "later mutation"
    assert captured[0] == {
        "role": "model",
        "parts": [{"text": "prefix"}, {"text": "", "thought": True}],
    }


@pytest.mark.parametrize("remove_prefix", [False, True])
def test_context_mapping_excludes_expanded_previous_replay(remove_prefix):
    owner = ContinuationRequest("gemini")
    original = request(recovered())
    original["model"] = ORIGIN.model
    body = owner.build(
        original,
        ContinuationSeed(
            "messages", ({"type": "thinking", "thinking": "Current thought"},)
        ),
    )
    paths = {}
    sent = prepare_history(
        body,
        ORIGIN,
        scope=HistoryMode.NONE if remove_prefix else HistoryMode.ALL,
        source_paths=paths,
        continuation=True,
    )
    captured = owner.sent_context(sent, source=body, source_paths=paths)
    assert captured == tuple(body["contents"][-(1 if remove_prefix else 2) :])
    # Prior replay expands one source row into three wire rows. It is not part
    # of the new continuation even when the current model row is removed.
    contents = sent["contents"]
    assert isinstance(contents, list)
    assert len(contents) > len(body["contents"])


@pytest.mark.parametrize("mode", list(HistoryMode))
def test_fragmented_model_turn_restores_once_and_maps_original_carrier(mode):
    record = recovered()
    body = request(record)
    body["contents"][1:2] = [
        {"role": "model", "parts": [{"text": "prefix "}]},
        {"role": "model", "parts": [{"text": "suffix"}]},
        {
            "role": "model",
            "parts": [{"thoughtSignature": encode_gemini_replay(record)}],
        },
    ]
    original = deepcopy(body)
    paths = {}
    prepared = prepare_history(body, ORIGIN, scope=mode, source_paths=paths)
    assert prepared["contents"] == [
        body["contents"][0],
        *record.continuation.context,
        record.native,
        body["contents"][-1],
    ]
    assert paths[("contents", 3, "parts", 0)] == ("contents", 3, "parts", 0)
    assert paths[("contents", 1)] == ("contents", 3)
    assert body == original


@pytest.mark.parametrize("damage", ["edited_prefix", "cross_user", "second_carrier"])
def test_fragmented_replay_cannot_hide_edits_or_cross_turns(damage):
    record = recovered()
    body = request(record)
    carrier = {"thoughtSignature": encode_gemini_replay(record)}
    body["contents"][1:2] = [
        {"role": "model", "parts": [{"text": "prefix suffix"}]},
        {"role": "model", "parts": [carrier]},
    ]
    if damage == "edited_prefix":
        body["contents"][1]["parts"][0]["text"] = "edited"
    elif damage == "cross_user":
        body["contents"].insert(
            2, {"role": "user", "parts": [{"text": "different turn"}]}
        )
    else:
        body["contents"].insert(2, {"role": "model", "parts": [carrier]})
    with pytest.raises(HistoryReplayError):
        prepare_history(body, ORIGIN)


def test_fragmented_previous_replay_does_not_absorb_injected_model_prefix():
    record = recovered()
    source = request(record)
    source["contents"][1:] = [
        {"role": "model", "parts": [{"text": "prefix suffix"}]},
        {
            "role": "model",
            "parts": [{"thoughtSignature": encode_gemini_replay(record)}],
        },
    ]
    owner = ContinuationRequest("gemini")
    body = owner.build(
        source, ContinuationSeed("messages", ({"type": "text", "text": "new prefix"},))
    )
    paths = {}
    prepared = prepare_history(
        body, ORIGIN, scope=HistoryMode.NONE, source_paths=paths, continuation=True
    )
    assert owner.sent_context(prepared, source=body, source_paths=paths) == tuple(
        body["contents"][-2:]
    )


def test_replay_works_in_a_fresh_process():
    record = recovered()
    script = """
import json, sys
from free_claude_code.core.history_replay import decode_replay, prepare_history
body=json.load(sys.stdin)
record=decode_replay(body['contents'][1]['parts'][1]['thoughtSignature'])
print(json.dumps(prepare_history(body,record.origin)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(request(record)),
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert json.loads(result.stdout)["contents"][1:-1] == [
        *record.continuation.context,
        record.native,
    ]


def test_recovered_tool_defaults_and_generated_id_preserve_original_call():
    record = recovered()
    call = {"name": "read", "args": {}}
    record = replace(
        record,
        native={
            "role": "model",
            "parts": [{"functionCall": call, "thoughtSignature": "call-signature"}],
        },
        tool_defaults={"read": {"limit": 10}},
        continuation=replace(
            record.continuation,
            echo=(
                {"text": "prefix "},
                {"functionCall": {"id": "generated", **call}},
            ),
        ),
    )
    body = request(record)
    body["contents"][1]["parts"] = [
        {"text": "prefix "},
        {"functionCall": {"id": "generated", "name": "read", "args": {"limit": 10}}},
        {"thoughtSignature": encode_replay(record)},
    ]
    body["contents"][-1] = {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "id": "generated",
                    "name": "read",
                    "response": {"result": "ok"},
                }
            }
        ],
    }
    prepared = prepare_history(body, ORIGIN)
    contents = prepared["contents"]
    assert isinstance(contents, list)
    assert contents[-2] == record.native
    assert contents[-1] == {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": "read",
                    "response": {"result": "ok"},
                }
            }
        ],
    }

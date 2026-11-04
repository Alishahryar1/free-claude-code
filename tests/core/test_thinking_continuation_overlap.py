"""Readable reasoning uses the same bounded replay comparison as answer text."""

from copy import deepcopy
from typing import Any

import pytest

from free_claude_code.core.continuation_overlap import ContinuationOverlap
from free_claude_code.core.continuation_stream import ContinuationStream
from free_claude_code.core.delivered_response import DeliveredResponse
from free_claude_code.core.history_replay import (
    AssociatedReplayRecord,
    ReplayOrigin,
    encode_replay,
)


@pytest.mark.parametrize("wire", ["messages", "gemini", "responses"])
@pytest.mark.parametrize("parts", [["Plan", " then act"], ["Pl", "an then act"]])
def test_thinking_overlap_retains_only_new_characters(wire, parts):
    source: list[dict[str, Any]]
    overlap = ContinuationOverlap("Plan", channel="thinking")
    if wire == "messages":
        source = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            }
        ]
        source += [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": part},
            }
            for part in parts
        ]
        source += [{"type": "content_block_stop", "index": 0}, {"type": "message_stop"}]
    elif wire == "gemini":
        source = [
            {"candidates": [{"content": {"parts": [{"thought": True, "text": part}]}}]}
            for part in parts
        ]
        source += [{"candidates": [{"finishReason": "STOP"}]}]
    else:
        source = [
            {
                "type": "response.reasoning_text.delta",
                "item_id": "rs",
                "output_index": 0,
                "content_index": 0,
                "delta": part,
            }
            for part in parts
        ]
        source += [
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "reasoning",
                            "id": "rs",
                            "content": [
                                {"type": "reasoning_text", "text": "".join(parts)}
                            ],
                            "summary": [],
                        }
                    ]
                },
            }
        ]
    before = deepcopy(source)
    events = [out for event in source for out in overlap.feed(event)]
    if wire == "messages":
        text = "".join(event.get("delta", {}).get("thinking", "") for event in events)
    elif wire == "gemini":
        text = "".join(
            part.get("text", "")
            for event in events
            for candidate in event["candidates"]
            for part in candidate.get("content", {}).get("parts", [])
        )
    else:
        text = "".join(event.get("delta", "") for event in events)
        assert events[-1]["response"]["output"][0]["content"][0]["text"] == " then act"
    assert text == " then act"
    assert source == before


def test_thinking_overlap_preserves_independent_equal_parts():
    overlap = ContinuationOverlap("Plan", channel="thinking")
    full = {
        "type": "response.reasoning_text.delta",
        "item_id": "rs",
        "output_index": 0,
        "content_index": 0,
        "delta": "Plan",
    }
    assert overlap.feed(full) == []
    summary = {
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs",
        "output_index": 0,
        "summary_index": 0,
        "delta": "Plan more",
    }
    result = overlap.feed(summary)
    assert result[0]["delta"] == "Plan more"


def test_summary_replay_can_be_followed_by_independent_full_text_in_same_item():
    overlap = ContinuationOverlap("Plan", channel="thinking")
    assert (
        overlap.feed(
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs",
                "output_index": 0,
                "summary_index": 0,
                "delta": "Plan",
            }
        )
        == []
    )
    full = {
        "type": "response.reasoning_text.delta",
        "item_id": "rs",
        "output_index": 0,
        "content_index": 0,
        "delta": "Details",
    }
    assert overlap.feed(full) == [full]
    terminal = {
        "type": "response.completed",
        "response": {
            "output": [
                {
                    "id": "rs",
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "Details"}],
                    "summary": [{"type": "summary_text", "text": "Plan"}],
                }
            ]
        },
    }
    output = overlap.feed(terminal)[0]["response"]["output"][0]
    assert output["content"] == [{"type": "reasoning_text", "text": "Details"}]
    assert output["summary"] == [{"type": "summary_text", "text": ""}]


@pytest.mark.parametrize("summary", [False, True])
def test_completed_scalar_alias_can_be_repeated_as_its_native_summary(summary):
    delivered = DeliveredResponse("responses")
    carrier = encode_replay(
        AssociatedReplayRecord(
            ReplayOrigin("provider", "chat", "https://provider.test", "key", "model"),
            {
                "reasoning_content": "Plan",
                "reasoning_details": [{"type": "reasoning.summary", "summary": "Plan"}],
            },
            "alias-record",
            "anchor",
        )
    )
    delivered.observe(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_first",
                "content": [{"type": "reasoning_text", "text": "Plan"}],
                "summary": [],
                "encrypted_content": carrier,
            },
        }
    )
    continuation = ContinuationStream(delivered)
    continuation.begin(handoff=False)
    kind = (
        "response.reasoning_summary_text.delta"
        if summary
        else "response.reasoning_text.delta"
    )
    events = continuation.prepare(
        {"type": kind, "item_id": "rs_next", "output_index": 0, "delta": "Plan more"}
    )
    assert [event["delta"] for event in events if event["type"] == kind] == [" more"]

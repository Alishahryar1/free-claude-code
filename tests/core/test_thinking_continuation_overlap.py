"""Readable reasoning uses the same bounded replay comparison as answer text."""

from copy import deepcopy
from typing import Any

import pytest

from free_claude_code.core.continuation_overlap import ContinuationOverlap


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


def test_summary_overlap_does_not_consume_equal_full_reasoning():
    overlap = ContinuationOverlap("Plan", channel="summary")
    full = {
        "type": "response.reasoning_text.delta",
        "item_id": "rs",
        "output_index": 0,
        "content_index": 0,
        "delta": "Plan",
    }
    assert overlap.feed(full) == [full]
    summary = {
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs",
        "output_index": 0,
        "summary_index": 0,
        "delta": "Plan more",
    }
    result = overlap.feed(summary)
    assert result[0]["delta"] == " more"

"""Readable reasoning uses the same bounded replay comparison as answer text."""

from copy import deepcopy
from typing import Any

import pytest

from free_claude_code.core.continuation_overlap import ContinuationOverlap
from free_claude_code.core.continuation_stream import ContinuationStream
from free_claude_code.core.delivered_response import DeliveredResponse
from free_claude_code.core.failures import ExecutionFailure
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


def reasoning_part(field, index, text, *, done=False):
    return {
        "type": f"response.reasoning{'_summary' if field == 'summary' else ''}_text.{'done' if done else 'delta'}",
        "item_id": "rs",
        "output_index": 0,
        f"{field}_index": index,
        "text" if done else "delta": text,
    }


@pytest.mark.parametrize("summary_first", [False, True])
@pytest.mark.parametrize("new_summary_first", [False, True])
@pytest.mark.parametrize("terminal", ["response.completed", "response.incomplete"])
def test_replayed_thinking_accepts_later_parts_in_either_array(
    summary_first, new_summary_first, terminal
):
    overlap = ContinuationOverlap("PlanAct", channel="thinking")
    prefix = [("content", "Plan"), ("summary", "Act")]
    if summary_first:
        prefix.reverse()
    for field, text in prefix:
        overlap.feed(reasoning_part(field, 0, text))
        overlap.feed(reasoning_part(field, 0, text, done=True))
    suffix = [("content", "Details"), ("summary", "Notes")]
    if new_summary_first:
        suffix.reverse()
    for field, text in suffix:
        delta = reasoning_part(field, 1, text)
        assert overlap.feed(delta) == [delta]
        done = reasoning_part(field, 1, text, done=True)
        assert overlap.feed(done) == [done]
    item: dict[str, Any] = {
        "type": "reasoning",
        "id": "rs",
        "content": [
            {"type": "reasoning_text", "text": "Plan"},
            {"type": "reasoning_text", "text": "Details"},
        ],
        "summary": [
            {"type": "summary_text", "text": "Act"},
            {"type": "summary_text", "text": "Notes"},
        ],
    }
    expected = deepcopy(item)
    expected["content"][0]["text"] = ""
    expected["summary"][0]["text"] = ""
    assert (
        overlap.feed(
            {"type": "response.output_item.done", "output_index": 0, "item": item}
        )[0]["item"]
        == expected
    )
    assert overlap.feed({"type": terminal, "response": {"output": [item]}})[0][
        "response"
    ]["output"] == [expected]


@pytest.mark.parametrize("field", ["content", "summary"])
@pytest.mark.parametrize("change", ["changed", "missing", "inserted", "earlier-item"])
def test_interleaved_thinking_still_rejects_rewritten_prefix(field, change):
    overlap = ContinuationOverlap("PlanAct", channel="thinking")
    for source, text in [("content", "Plan"), ("summary", "Act")]:
        overlap.feed(reasoning_part(source, 0, text))
        overlap.feed(reasoning_part(source, 0, text, done=True))
    item: dict[str, Any] = {
        "type": "reasoning",
        "id": "rs",
        "content": [{"type": "reasoning_text", "text": "Plan"}],
        "summary": [{"type": "summary_text", "text": "Act"}],
    }
    output = [item]
    if change == "changed":
        item[field][0]["text"] = "Rewritten"
    elif change == "missing":
        item[field] = []
    elif change == "inserted":
        item[field].insert(0, {"type": item[field][0]["type"], "text": "Earlier"})
    else:
        output.insert(0, {**deepcopy(item), "id": "earlier"})
    with pytest.raises(ExecutionFailure, match="already observed output prefix"):
        overlap.feed({"type": "response.completed", "response": {"output": output}})


@pytest.mark.parametrize("field", ["content", "summary"])
@pytest.mark.parametrize("snapshot", [False, True])
def test_new_thinking_part_cannot_precede_a_later_matched_item(field, snapshot):
    overlap = ContinuationOverlap("PlanAct", channel="thinking")
    for index, text in enumerate(("Plan", "Act")):
        identity = {"item_id": f"rs_{index}", "output_index": index}
        overlap.feed({**reasoning_part("content", 0, text), **identity})
        overlap.feed({**reasoning_part("content", 0, text, done=True), **identity})
    extra = {
        "type": "summary_text" if field == "summary" else "reasoning_text",
        "text": "Earlier",
    }
    first: dict[str, Any] = {
        "type": "reasoning",
        "id": "rs_0",
        "content": [{"type": "reasoning_text", "text": "Plan"}],
        "summary": [],
    }
    first[field].append(extra)
    terminal = {
        "type": "response.completed",
        "response": {
            "output": [
                first,
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "content": [{"type": "reasoning_text", "text": "Act"}],
                },
            ]
        },
    }
    delta = {
        **reasoning_part(field, 1 if field == "content" else 0, "Earlier"),
        "item_id": "rs_0",
    }
    with pytest.raises(ExecutionFailure, match="already observed output prefix"):
        overlap.feed(terminal if snapshot else delta)


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

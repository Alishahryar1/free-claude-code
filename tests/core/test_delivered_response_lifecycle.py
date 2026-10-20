"""Published completion stages determine the remaining recovery boundary."""

import pytest

from free_claude_code.core.continuation_stream import ContinuationStream
from free_claude_code.core.delivered_response import DeliveredResponse


@pytest.mark.parametrize("summary", [False, True])
@pytest.mark.parametrize("closed", ["none", "text", "part", "item"])
def test_readable_reasoning_closes_only_missing_stages(summary, closed):
    response = DeliveredResponse("responses")
    field = "summary" if summary else "content"
    part = {"type": "summary_text" if summary else "reasoning_text", "text": "Reason."}
    item = {
        "type": "reasoning",
        "id": "rs_first",
        "status": "in_progress",
        field: [part],
    }
    response.observe(
        {"type": "response.output_item.added", "output_index": 0, "item": item}
    )
    identity = {
        "output_index": 0,
        "item_id": "rs_first",
        "summary_index" if summary else "content_index": 0,
    }
    text_kind = (
        "response.reasoning_summary_text.done"
        if summary
        else "response.reasoning_text.done"
    )
    part_kind = (
        "response.reasoning_summary_part.done"
        if summary
        else "response.content_part.done"
    )
    if closed in {"text", "part", "item"}:
        response.observe({"type": text_kind, **identity, "text": "Reason."})
    if closed in {"part", "item"}:
        response.observe({"type": part_kind, **identity, "part": part})
    if closed == "item":
        response.observe(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {**item, "status": "completed"},
            }
        )
    expected = (
        ([] if closed != "none" else [text_kind])
        + ([part_kind] if summary and closed in {"none", "text"} else [])
        + ([] if closed == "item" else ["response.output_item.done"])
    )
    events = response.closing_events()
    assert [event["type"] for event in events] == expected
    assert response.closing_events() == events
    for event in events:
        response.observe(event)
    assert response.closing_events() == []


def test_finished_part_does_not_require_another_text_done():
    response = DeliveredResponse("responses")
    part = {"type": "output_text", "text": "Done."}
    response.observe(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_first", "content": [part]},
        }
    )
    response.observe(
        {
            "type": "response.content_part.done",
            "output_index": 0,
            "content_index": 0,
            "part": part,
        }
    )
    assert [event["type"] for event in response.closing_events()] == [
        "response.output_item.done"
    ]


def test_empty_blocks_and_usage_updates_do_not_finish_overlap():
    response = DeliveredResponse("messages")
    response.observe(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "Hello "},
        }
    )
    continuation = ContinuationStream(response)
    continuation.begin(handoff=False)
    for event in continuation.boundary():
        response.observe(event)
    continuation.prepare(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
    )
    assert (
        continuation.prepare(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": ""},
            }
        )
        == []
    )
    continuation.prepare({"type": "content_block_stop", "index": 0})
    continuation.prepare(
        {
            "type": "message_delta",
            "delta": {"stop_reason": None},
            "usage": {"output_tokens": 1},
        }
    )
    assert (
        continuation.prepare(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "l"},
            }
        )
        == []
    )
    events = continuation.prepare(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "o world."},
        }
    )
    assert events[0]["delta"]["text"] == "world."


@pytest.mark.parametrize("seed", ["added", "done"])
def test_refusal_content_without_deltas_counts_as_progress(seed):
    response = DeliveredResponse("responses")
    response.observe(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_refusal", "content": []},
        }
    )
    initial = response.progress
    response.observe(
        {
            "type": f"response.content_part.{seed}",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "refusal", "refusal": "Cannot continue."},
        }
    )
    assert response.progress > initial
    assert not response.snapshot().eligible
    assert not response.can_handoff


def test_handoff_fallback_preserves_already_closed_item_status(monkeypatch):
    response = DeliveredResponse("responses")
    response.observe(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_closed",
                "status": "incomplete",
                "content": [],
            },
        }
    )
    response.observe(
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "message",
                "id": "msg_open",
                "status": "in_progress",
                "content": [{"type": "output_text", "text": "Next."}],
            },
        }
    )
    continuation = ContinuationStream(response)
    continuation.begin(handoff=True)
    monkeypatch.setattr(
        "free_claude_code.core.delivered_response.RECOVERY_RETENTION_BYTES", 1
    )
    for event in continuation.boundary():
        response.observe(event)
    result = continuation.finalize(
        continuation.prepare(response.handoff_events()[0])[0]
    )["response"]
    assert [item["status"] for item in result["output"]] == ["incomplete", "completed"]

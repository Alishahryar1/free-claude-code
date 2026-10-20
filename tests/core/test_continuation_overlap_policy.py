"""Independent answer partitions exercise the overlap policy and its bounds."""

from copy import deepcopy

import pytest

from free_claude_code.core.continuation_overlap import ContinuationOverlap
from free_claude_code.core.delivered_response import DeliveredResponse
from free_claude_code.core.failures import ExecutionFailure
from tests.api.test_continuation_overlap import responses_parts


@pytest.mark.parametrize("wire", ["messages", "responses"])
def test_empty_delta_stays_invisible_with_no_prior_answer_text(wire):
    event = (
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": ""},
        }
        if wire == "messages"
        else {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg_text",
            "delta": "",
        }
    )
    assert ContinuationOverlap("").feed(event) == []


def retained_parts(prefix, parts):
    text = "".join(parts)
    overlap = max(
        size
        for size in range(min(len(prefix), len(text)) + 1)
        if prefix.endswith(text[:size])
    )
    result = []
    for part in parts:
        result.append(part[overlap:])
        overlap = max(0, overlap - len(part))
    return result


@pytest.mark.parametrize(
    "prefix,parts",
    [
        ("Hello ", ["He", "llo world."]),
        ("abcabc", ["", "abc", "", "abcX"]),
        ("abcabd", ["ab", "cX"]),
        ("Hello 世界", ["世", "界!"]),
        ("The answer is", ["answer", " is"]),
        ("Hello ", ["He", "l"]),
    ],
)
@pytest.mark.parametrize(
    "representation",
    [
        "deltas",
        "text_done",
        "part_done",
        "item_done",
        "terminal",
        "part_added",
        "item_added",
        "mixed",
    ],
)
@pytest.mark.parametrize("separate_items", [False, True])
def test_text_representation_preserves_each_source_part(
    prefix, parts, representation, separate_items
):
    source = responses_parts(
        parts, representation=representation, separate_items=separate_items
    )
    original = deepcopy(source)
    overlap = ContinuationOverlap(prefix)
    events = [ready for event in source for ready in overlap.feed(event)]
    expected = retained_parts(prefix, parts)
    output = events[-1]["response"]["output"]
    assert [part["text"] for item in output for part in item["content"]] == expected
    assert source == original
    finals = {item["id"]: item for item in output}
    deltas = {}
    for event in events:
        kind = event["type"]
        if kind == "response.output_text.delta":
            key = (event["item_id"], event["content_index"])
            deltas[key] = deltas.get(key, "") + event["delta"]
        elif kind == "response.output_text.done":
            assert (
                event["text"]
                == finals[event["item_id"]]["content"][event["content_index"]]["text"]
            )
        elif kind == "response.content_part.done":
            assert (
                event["part"]
                == finals[event["item_id"]]["content"][event["content_index"]]
            )
        elif kind == "response.output_item.done":
            assert event["item"] == finals[event["item"]["id"]]
    if representation == "deltas":
        for item in output:
            for position, part in enumerate(item["content"]):
                assert deltas.get((item["id"], position), "") == part["text"]


def test_pending_snapshots_are_idempotent_and_lifecycle_frames_keep_order():
    source = responses_parts(["He", "llo world."], representation="text_done")
    source.insert(4, deepcopy(source[3]))
    source.insert(5, {"type": "ping", "marker": "between repeated snapshots"})
    overlap = ContinuationOverlap("Hello ")
    released = []
    for index, event in enumerate(source):
        ready = overlap.feed(event)
        if 3 <= index <= 6:
            assert ready == []
        released.extend(ready)
    assert [event["type"] for event in released] == [event["type"] for event in source]
    assert [
        part["text"] for part in released[-1]["response"]["output"][0]["content"]
    ] == ["", "world."]


def test_interleaved_parts_wait_for_the_earlier_open_part():
    source = responses_parts(["He", "llo world."], representation="text_done")
    earlier_done = source[3:5]
    later = source[5:8]
    overlap = ContinuationOverlap("Hello ")
    released = []
    for event in [*source[:3], *later]:
        ready = overlap.feed(event)
        if event in later and event["type"].endswith(".done"):
            assert ready == []
        released.extend(ready)
    for event in [*earlier_done, *source[8:]]:
        released.extend(overlap.feed(event))
    assert released[-1]["response"]["output"][0]["content"] == [
        {"type": "output_text", "text": "", "annotations": []},
        {"type": "output_text", "text": "world.", "annotations": []},
    ]
    assert [
        event["content_index"]
        for event in released
        if event["type"] == "response.output_text.done"
    ] == [1, 0]


@pytest.mark.parametrize("change", ["prefix", "order", "insert"])
def test_late_snapshot_cannot_rewrite_a_decided_prefix(change):
    source = responses_parts(
        ["He", "llo world."], representation="text_done", separate_items=True
    )
    overlap = ContinuationOverlap("Hello ")
    for event in source[:-1]:
        overlap.feed(event)
    terminal = source[-1]
    items = terminal["response"]["output"]
    if change == "prefix":
        items[0]["content"][0]["text"] = "Ha"
    elif change == "order":
        items.reverse()
    else:
        extra = deepcopy(items[0])
        extra["id"] = "msg_late"
        items.insert(0, extra)
    with pytest.raises(ExecutionFailure, match="observed answer prefix"):
        overlap.feed(terminal)


def test_terminal_cannot_omit_pending_source_text():
    source = responses_parts(["He"], representation="text_done")
    overlap = ContinuationOverlap("Hello ")
    for event in source[:-1]:
        overlap.feed(event)
    source[-1]["response"]["output"] = []
    with pytest.raises(ExecutionFailure, match="observed answer prefix"):
        overlap.feed(source[-1])


def test_unresolved_frames_are_bounded_and_discarded_on_failure(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.core.continuation_overlap.RECOVERY_RETENTION_BYTES", 1000
    )
    overlap = ContinuationOverlap("Hello ")
    source = responses_parts(["He"], representation="text_done")
    for event in source[:4]:
        overlap.feed(event)
    with pytest.raises(ExecutionFailure, match="retained-data limit"):
        for _ in range(30):
            overlap.feed({"type": "ping", "padding": "x" * 60})
    failure = {"type": "response.failed", "response": {"output": []}}
    assert overlap.feed(failure) == [failure]


def test_large_immediately_resolvable_snapshot_does_not_fill_the_pending_buffer(
    monkeypatch,
):
    monkeypatch.setattr(
        "free_claude_code.core.continuation_overlap.RECOVERY_RETENTION_BYTES", 300
    )
    overlap = ContinuationOverlap("Hello ")
    source = responses_parts(["Hello " + "x" * 10000], representation="terminal")
    events = [ready for event in source for ready in overlap.feed(event)]
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "x" * 10000


def test_resolved_overlap_does_not_retain_unaffected_later_items(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.core.continuation_overlap.RECOVERY_RETENTION_BYTES", 400
    )
    overlap = ContinuationOverlap("Hello ")
    source = responses_parts(["Hello world."], representation="text_done")
    for event in source[:-1]:
        overlap.feed(event)
    for index in range(1, 1000):
        event = {
            "type": "response.output_item.done",
            "output_index": index,
            "item": {
                "id": f"msg_{index}",
                "type": "message",
                "content": [{"type": "output_text", "text": "More."}],
            },
        }
        assert overlap.feed(event) == [event]


@pytest.mark.parametrize(
    "surface", ["item_added", "part_added", "text_done", "part_done", "item_done"]
)
def test_snapshot_only_text_counts_as_new_public_progress_once(surface):
    response = DeliveredResponse("responses")
    source = responses_parts(["Novel"], representation=surface)
    kind = {
        "item_added": "response.output_item.added",
        "part_added": "response.content_part.added",
        "text_done": "response.output_text.done",
        "part_done": "response.content_part.done",
        "item_done": "response.output_item.done",
    }[surface]
    target = next(event for event in source if event["type"] == kind)
    if surface != "item_added":
        response.observe(source[1])
    initial = response.progress
    response.observe(target)
    assert response.progress > initial
    released = response.progress
    response.observe(target)
    assert response.progress == released

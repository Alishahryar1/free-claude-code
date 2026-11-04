"""Replay edits follow emitted source coordinates, not field order or equal text."""

from copy import deepcopy

import pytest

from free_claude_code.core.history_replay import ReplayOrigin, ReplayRecord
from free_claude_code.core.readable_replay import ReadableReplay


def record(native, protocol="responses"):
    return ReplayRecord(
        ReplayOrigin("provider", protocol, "https://test", "key", "model"), native
    )


def test_summary_before_full_content_trims_only_the_emitted_summary_prefix():
    native = {
        "type": "reasoning",
        "id": "rs_one",
        "content": [{"type": "reasoning_text", "text": "Detail"}],
        "summary": [{"type": "summary_text", "text": "Plan more"}],
    }
    original = deepcopy(native)
    mapping = ReadableReplay()
    mapping.add(("summary", 0, "text"), 0, -1, 0, 4)
    mapping.add(("content", 0, "text"), 0, -1, 4, 6)
    mapping.add(("summary", 0, "text"), 4, -1, 10, 5)
    projected = mapping.project(record(native), lambda _: (4, 11)).native
    assert projected["content"] == native["content"]
    assert projected["summary"] == [{"type": "summary_text", "text": " more"}]
    assert native == original


def test_equal_independent_parts_keep_separate_public_positions():
    native = {
        "type": "reasoning",
        "content": [
            {"type": "reasoning_text", "text": "Plan"},
            {"type": "reasoning_text", "text": "Plan"},
        ],
        "summary": [],
    }
    mapping = ReadableReplay()
    mapping.add(("content", 0, "text"), 0, -1, 0, 4)
    mapping.add(("content", 1, "text"), 0, -1, 4, 4)
    assert mapping.project(record(native), lambda _: (4, 4)).native["content"] == [
        {"type": "reasoning_text", "text": ""},
        {"type": "reasoning_text", "text": "Plan"},
    ]


def test_native_first_aliases_share_exact_retained_coordinates():
    native = {
        "reasoning_content": "PlanAct",
        "reasoning_details": [
            {"type": "reasoning.text", "text": "Plan", "index": 0},
            {"type": "reasoning.text", "text": "Act", "index": 1},
        ],
    }
    mapping = ReadableReplay()
    mapping.add(("reasoning_content",), 0, -1, 0, 7)
    mapping.chat_aliases(native)
    projected = mapping.project(record(native, "chat"), lambda _: (4, 3)).native
    assert projected["reasoning_content"] == "Act"
    assert projected["reasoning_details"] == [
        {"type": "reasoning.text", "text": "", "index": 0},
        {"type": "reasoning.text", "text": "Act", "index": 1},
    ]


@pytest.mark.parametrize(
    "removed,expected", [(0, "Plan\n\nAct"), (2, "an\n\nAct"), (4, "Act")]
)
def test_typed_first_alias_retains_native_separators(removed, expected):
    native = {
        "reasoning_content": "Plan\n\nAct",
        "reasoning_details": [
            {"type": "reasoning.text", "text": "Plan"},
            {"type": "reasoning.text", "text": "Act"},
        ],
    }
    mapping = ReadableReplay()
    mapping.add(("reasoning_details", 0, "text"), 0, -1, 0, 4)
    mapping.add(("reasoning_details", 1, "text"), 0, -1, 4, 3)
    mapping.chat_aliases(native)
    projected = mapping.project(
        record(native, "chat"), lambda _: (removed, 7 - removed)
    ).native
    assert projected["reasoning_content"] == expected
    assert native["reasoning_content"] == "Plan\n\nAct"


def test_unpublished_mapped_text_is_removed_but_hidden_and_protected_records_stay_intact():
    native = {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": "Hidden"}],
    }
    source = record(native)
    assert ReadableReplay().project(source, lambda _: (0, 0)) == source
    mapping = ReadableReplay()
    mapping.add(("content", 0, "text"), 0, -1, 0, 6)
    assert mapping.project(source, lambda _: (0, 0)).native["content"] == [
        {"type": "reasoning_text", "text": ""}
    ]
    protected = record({**native, "encrypted_content": "cipher"})
    assert mapping.project(protected, lambda _: (0, 0)) == protected

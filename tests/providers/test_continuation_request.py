"""Continuation tails are replaceable without rolling back accepted corrections."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from free_claude_code.core.history_replay import (
    HistoryMode,
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
    prepare_history,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_recovery import ContinuationSeed
from free_claude_code.providers.continuation import ContinuationRequest
from tests.providers.support import assistant_history_text, continuation_seed


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("initial_text", ["", "First "])
def test_next_continuation_preserves_corrected_history_and_options(
    protocol, initial_text
):
    field = "input" if protocol == "responses" else "messages"
    user_turn = {"role": "user", "content": "Continue"}
    body = {
        field: [
            {"role": "assistant", "content": "An old answer"},
            user_turn,
        ],
        "max_tokens": 100,
        "reasoning": {"effort": "none"},
        "extra_body": {"retained": True, "rejected": True},
        "temperature": 0.5,
    }
    original = deepcopy(body)
    request = ContinuationRequest(protocol)
    first = request.build(body, continuation_seed(initial_text, "First thought."))
    assert body == original
    corrected = deepcopy(first)
    # An accepted correction can remove a historical row and change request options.
    del corrected[field][0]
    del corrected["reasoning"]
    del corrected["extra_body"]["rejected"]
    del corrected["temperature"]
    corrected["max_tokens"] = 200
    before = deepcopy(corrected)
    second = request.build(
        corrected, continuation_seed("First answer.", "First thought. Second thought.")
    )
    assert corrected == before
    assert second[field][0] == user_turn
    assert assistant_history_text(second[field]) == "First answer."
    assert sum(row.get("role") == "user" for row in second[field]) == 2
    assert second["max_tokens"] == 200
    assert second["extra_body"] == {"retained": True}
    assert "reasoning" not in second and "temperature" not in second
    assert "First answer." in str(second[field][-2])
    assert "First thought. Second thought." in str(second[field][:-1])
    assert "First thought." not in str(second[field][-1])
    assert request.build(second, continuation_seed("First answer. More."))[field][
        :-2
    ] == [user_turn]


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
def test_original_turns_matching_private_instruction_are_preserved(protocol):
    field = "input" if protocol == "responses" else "messages"
    instruction_turn = ContinuationRequest(protocol).build(
        {field: []}, continuation_seed()
    )[field][-1]
    body = {field: [instruction_turn, {"role": "user", "content": "Continue"}]}
    original = deepcopy(body)
    request = ContinuationRequest(protocol)
    first = request.build(body, continuation_seed("One"))
    second = request.build(first, continuation_seed("One two"))
    assert second[field][:-2] == original[field]
    assert body == original


def test_responses_string_input_is_preserved_across_continuations():
    request = ContinuationRequest("responses")
    first = request.build(
        {"input": "Original prompt"}, continuation_seed(thinking="A thought.")
    )
    second = request.build(first, continuation_seed("Answer", "A thought."))
    assert second["input"][0] == {"role": "user", "content": "Original prompt"}
    assert assistant_history_text(second["input"]) == "Answer"
    assert str(second["input"]).count("A thought.") == 1


@pytest.mark.parametrize("parts", [None, [], [""], ["Half", ""], ["   ", ""]])
def test_messages_suffix_accepts_summary_only_and_incomplete_text(parts):
    items: list[JsonObject] = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Summary"}]}
    ]
    if parts is not None:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text} for text in parts],
            }
        )
    seed = ContinuationSeed("responses", tuple(items))
    source = {"model": "model", "messages": [{"role": "user", "content": "Continue"}]}
    original = deepcopy(source)
    original_items = deepcopy(seed.items)
    builder = ContinuationRequest("messages")
    first = builder.build(source, seed)
    assert source == original and seed.items == original_items
    assert "Summary" in str(first)
    assistant = [
        message for message in first["messages"] if message["role"] == "assistant"
    ]
    expected = [text for text in parts or [] if text]
    assert assistant == [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Summary"},
                *[{"type": "text", "text": text} for text in expected],
            ],
        }
    ]
    second = builder.build(first, continuation_seed("Recovered"))
    assert second["messages"][:-2] == original["messages"]
    assert assistant_history_text(second["messages"]) == "Recovered"


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
def test_native_continuation_carriers_project_and_replace_as_one_suffix(protocol):
    field = "input" if protocol == "responses" else "messages"
    source = ReplayOrigin("source", "chat", "endpoint", "account", "model")
    record = ReplayRecord(source, {"reasoning_content": "retained native thought"})
    original = {field: [{"role": "user", "content": "hello"}]}
    request = ContinuationRequest(protocol)
    first = request.build(
        original,
        ContinuationSeed(
            "messages",
            (
                {"type": "redacted_thinking", "data": encode_replay(record)},
                {"type": "text", "text": "First"},
            ),
        ),
    )
    second = request.build(
        first,
        ContinuationSeed(
            "messages",
            (
                {"type": "redacted_thinking", "data": encode_replay(record)},
                {"type": "text", "text": "First second"},
            ),
        ),
    )
    assert json.dumps(second).count("retained native thought") == (
        0 if protocol == "messages" else 1
    )
    projected = prepare_history(
        second, replace(source, protocol=protocol), scope=HistoryMode.ALL
    )
    assert json.dumps(projected).count("retained native thought") == 1
    assert "First second" in json.dumps(projected)
    assert original == {field: [{"role": "user", "content": "hello"}]}

"""Continuation tails are replaceable without rolling back accepted corrections."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from free_claude_code.application.response_collection import WireProtocol
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
from tests.wire_payload import array_payload, object_array, object_payload


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("initial_text", ["", "First "])
def test_next_continuation_preserves_corrected_history_and_options(
    protocol: WireProtocol, initial_text: str
) -> None:
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
    first: JsonObject = request.build(
        body, continuation_seed(initial_text, "First thought.")
    )
    assert body == original
    corrected: JsonObject = deepcopy(first)
    # An accepted correction can remove a historical row and change request options.
    del array_payload(corrected[field])[0]
    del corrected["reasoning"]
    del object_payload(corrected["extra_body"])["rejected"]
    del corrected["temperature"]
    corrected["max_tokens"] = 200
    before = deepcopy(corrected)
    second: JsonObject = request.build(
        corrected, continuation_seed("First answer.", "First thought. Second thought.")
    )
    assert corrected == before
    assert array_payload(second[field])[0] == user_turn
    assert assistant_history_text(second[field]) == "First answer."
    assert sum(row.get("role") == "user" for row in object_array(second[field])) == 2
    assert second["max_tokens"] == 200
    assert second["extra_body"] == {"retained": True}
    assert "reasoning" not in second and "temperature" not in second
    assert "First answer." in str(array_payload(second[field])[-2])
    assert "First thought. Second thought." in str(array_payload(second[field])[:-1])
    assert "First thought." not in str(array_payload(second[field])[-1])
    assert request.build(second, continuation_seed("First answer. More."))[field][
        :-2
    ] == [user_turn]


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
def test_original_turns_matching_private_instruction_are_preserved(
    protocol: WireProtocol,
) -> None:
    field = "input" if protocol == "responses" else "messages"
    instruction_body: JsonObject = ContinuationRequest(protocol).build(
        {field: []}, continuation_seed()
    )
    instruction_turn = object_array(instruction_body[field])[-1]
    body = {field: [instruction_turn, {"role": "user", "content": "Continue"}]}
    original = deepcopy(body)
    request = ContinuationRequest(protocol)
    first: JsonObject = request.build(body, continuation_seed("One"))
    second: JsonObject = request.build(first, continuation_seed("One two"))
    assert array_payload(second[field])[:-2] == original[field]
    assert body == original


def test_responses_string_input_is_preserved_across_continuations() -> None:
    request = ContinuationRequest("responses")
    first: JsonObject = request.build(
        {"input": "Original prompt"}, continuation_seed(thinking="A thought.")
    )
    second: JsonObject = request.build(first, continuation_seed("Answer", "A thought."))
    assert array_payload(second["input"])[0] == {
        "role": "user",
        "content": "Original prompt",
    }
    assert assistant_history_text(second["input"]) == "Answer"
    assert str(second["input"]).count("A thought.") == 1


@pytest.mark.parametrize("parts", [None, [], [""], ["Half", ""], ["   ", ""]])
def test_messages_suffix_accepts_summary_only_and_incomplete_text(
    parts: list[str] | None,
) -> None:
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
    first: JsonObject = builder.build(source, seed)
    assert source == original and seed.items == original_items
    assert "Summary" in str(first)
    assistant = [
        message
        for message in object_array(first["messages"])
        if message["role"] == "assistant"
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
    second: JsonObject = builder.build(first, continuation_seed("Recovered"))
    assert array_payload(second["messages"])[:-2] == original["messages"]
    assert assistant_history_text(second["messages"]) == "Recovered"


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
def test_native_continuation_carriers_project_and_replace_as_one_suffix(
    protocol: WireProtocol,
) -> None:
    field = "input" if protocol == "responses" else "messages"
    source = ReplayOrigin("source", "chat", "endpoint", "account", "model")
    record = ReplayRecord(source, {"reasoning_content": "retained native thought"})
    original = {field: [{"role": "user", "content": "hello"}]}
    request = ContinuationRequest(protocol)
    first: JsonObject = request.build(
        original,
        ContinuationSeed(
            "messages",
            (
                {"type": "redacted_thinking", "data": encode_replay(record)},
                {"type": "text", "text": "First"},
            ),
        ),
    )
    second: JsonObject = request.build(
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


def test_gemini_history_resolves_empty_streamed_tool_arguments_to_start_input() -> None:
    seed = ContinuationSeed(
        "messages",
        (
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_time",
                "input": {},
                "partial_json": "",
            },
        ),
    )
    history: list[JsonObject] = ContinuationRequest("gemini").history(
        seed, {"model": "model"}, allow_pending_calls=True
    )
    calls = [
        object_payload(part["functionCall"])
        for content in history
        for part in object_array(content["parts"])
        if "functionCall" in part
    ]
    assert [(call["name"], call["args"]) for call in calls] == [("get_time", {})]

"""Continuation tails are replaceable without rolling back accepted corrections."""

from copy import deepcopy

import pytest

from free_claude_code.providers.continuation import ContinuationRequest


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
    first = request.build(body, initial_text, "First thought.")
    assert body == original
    corrected = deepcopy(first)
    # An accepted correction can remove a historical row and change request options.
    del corrected[field][0]
    del corrected["reasoning"]
    del corrected["extra_body"]["rejected"]
    del corrected["temperature"]
    corrected["max_tokens"] = 200
    before = deepcopy(corrected)
    second = request.build(corrected, "First answer.", "First thought. Second thought.")
    assert corrected == before
    assert second[field][:-2] == [user_turn]
    assert second["max_tokens"] == 200
    assert second["extra_body"] == {"retained": True}
    assert "reasoning" not in second and "temperature" not in second
    assert "First answer." in str(second[field][-2])
    assert "First thought. Second thought." in str(second[field][-1])
    assert request.build(second, "First answer. More.", "")[field][:-2] == [user_turn]


@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
def test_original_turns_matching_private_instruction_are_preserved(protocol):
    field = "input" if protocol == "responses" else "messages"
    instruction_turn = ContinuationRequest(protocol).build({field: []}, "", "")[field][
        -1
    ]
    body = {field: [instruction_turn, {"role": "user", "content": "Continue"}]}
    original = deepcopy(body)
    request = ContinuationRequest(protocol)
    first = request.build(body, "One", "")
    second = request.build(first, "One two", "")
    assert second[field][:-2] == original[field]
    assert body == original


def test_responses_string_input_is_preserved_across_continuations():
    request = ContinuationRequest("responses")
    first = request.build({"input": "Original prompt"}, "", "A thought.")
    second = request.build(first, "Answer", "A thought.")
    assert second["input"][:-2] == [{"role": "user", "content": "Original prompt"}]

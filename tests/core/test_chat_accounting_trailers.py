"""Accounting after a choice finishes must not reopen or mutate that choice."""

from copy import deepcopy

import pytest

from free_claude_code.core.chat_completions.tool_calls import (
    ChatToolCallBuffer,
    CompletedChatBatch,
    CompletedChatCall,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.tool_call_buffer import RetentionBudget, ToolBufferLimitError
from tests.core.test_chat_tool_buffer import call, event


def terminal(reason="stop", choice=0):
    result = event(content="", finish=reason, choice=choice)
    result["choices"][0].update(native_finish_reason=reason)
    result["choices"][0]["delta"]["role"] = "assistant"
    return result


def accounting(reason="stop", choice=0):
    return {
        **terminal(reason, choice),
        "usage": {
            "prompt_tokens": 32,
            "completion_tokens": 13,
            "total_tokens": 45,
            "cost": 0,
            "extension": {"keep": True},
        },
    }


def frames(outputs):
    return [
        frame
        for item in outputs
        for frame in (item.frames if isinstance(item, CompletedChatBatch) else [item])
        if isinstance(frame, dict)
    ]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("reason", ["stop", "tool_calls", "length", "content_filter"])
def test_repeated_finish_is_an_accounting_frame_without_changing_input(native, reason):
    buffer = ChatToolCallBuffer(native=native)
    buffer.feed(terminal(reason))
    trailer = accounting(reason)
    trailer["provider_extension"] = {"keep": 7}
    original = deepcopy(trailer)
    assert frames(buffer.feed(trailer)) == [{**trailer, "choices": []}]
    assert trailer == original


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "delta",
    [
        {},
        {"content": None},
        {"role": None},
        {"reasoning": "", "reasoning_content": None},
        {"reasoning_details": [], "tool_calls": []},
        {"refusal": "", "audio": None, "function_call": None},
    ],
)
def test_empty_optional_fields_do_not_create_late_content(native, delta):
    buffer = ChatToolCallBuffer(native=native)
    buffer.feed(terminal())
    trailer = accounting()
    trailer["choices"][0]["delta"] = delta
    assert frames(buffer.feed(trailer)) == [{**trailer, "choices": []}]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        {"finish_reason": "length"},
        {"finish_reason": None},
        {"native_finish_reason": "other"},
        {"extra": None},
        {"logprobs": {}},
        {"delta": None},
        *[
            {"delta": {field: value}}
            for field, value in [
                ("content", "late"),
                ("content", 0),
                ("role", "user"),
                ("reasoning", "late"),
                ("reasoning_content", "late"),
                ("reasoning_details", [{"signature": "late"}]),
                ("refusal", "late"),
                ("audio", {}),
                ("tool_calls", [call(0, "late", "{}", "x")]),
                ("function_call", {}),
                ("unknown", ""),
            ]
        ],
    ],
)
def test_accounting_does_not_hide_late_content_or_contradictions(native, change):
    buffer = ChatToolCallBuffer(native=native)
    buffer.feed(terminal())
    invalid = accounting()
    invalid["choices"][0].update(change)
    # A valid active choice comes first: rejection must precede its state mutation.
    invalid["choices"].insert(0, terminal(choice=1)["choices"][0])
    with pytest.raises(ExecutionFailure):
        buffer.feed(invalid)
    assert frames(buffer.feed(terminal(choice=1)))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("usage", [None, [], "", 0])
def test_empty_repeated_choice_requires_accounting(native, usage):
    buffer = ChatToolCallBuffer(native=native)
    buffer.feed(terminal())
    with pytest.raises(ExecutionFailure):
        buffer.feed({**terminal(), "usage": usage})


@pytest.mark.parametrize("native", [False, True])
def test_first_finish_with_usage_still_completes_a_call(native):
    buffer = ChatToolCallBuffer(native=native)
    first = accounting("tool_calls")
    first["choices"][0]["delta"] = {"tool_calls": [call(0, "read", "{}", "a")]}
    result = buffer.feed(first)
    if native:
        assert isinstance(result[0], CompletedChatBatch)
        assert frames(result)[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "a"
    else:
        assert len([x for x in result if isinstance(x, CompletedChatCall)]) == 1
    assert any(frame.get("usage") == first["usage"] for frame in frames(result))
    assert frames(buffer.feed(accounting("tool_calls"))) == [
        {**accounting("tool_calls"), "choices": []}
    ]


@pytest.mark.parametrize("native", [False, True])
def test_mixed_accounting_and_active_tool_keeps_usage_and_source_order(native):
    buffer = ChatToolCallBuffer(native=native)
    buffer.feed(terminal())
    assert buffer.feed(event([call(0, "read", "{", "a")], choice=1)) == []
    mixed = accounting()
    mixed["choices"].append(event([call(0, arguments="}")], choice=1)["choices"][0])
    assert buffer.feed(mixed) == []
    result = buffer.feed(terminal("tool_calls", 1))
    output = frames(result)
    assert any(frame.get("usage") == mixed["usage"] for frame in output)
    assert all(c["index"] == 1 for frame in output for c in frame["choices"])
    if native:
        fragments = [
            t
            for frame in output
            for c in frame["choices"]
            for t in c["delta"].get("tool_calls", [])
        ]
        assert "".join(t["function"]["arguments"] for t in fragments) == "{}"
    else:
        assert isinstance(result[0], CompletedChatCall)
        assert result[0].body["function"]["arguments"] == "{}"


@pytest.mark.parametrize("native", [False, True])
def test_accounting_is_idempotent_but_finished_identity_retention_remains_bounded(
    native,
):
    budget = RetentionBudget(max_records=4)
    buffer = ChatToolCallBuffer(native=native, budget=budget)
    buffer.feed(terminal())
    retained = (budget.bytes, budget.records)
    for _ in range(20):
        buffer.feed(accounting())
        assert (budget.bytes, budget.records) == retained
    with pytest.raises(ToolBufferLimitError):
        for index in range(1, 6):
            buffer.feed(terminal(choice=index))


@pytest.mark.parametrize("native", [False, True])
def test_retained_finish_markers_count_toward_byte_limit(native):
    buffer = ChatToolCallBuffer(native=native, budget=RetentionBudget(max_bytes=512))
    final = terminal()
    final["choices"][0]["native_finish_reason"] = "x" * 600
    with pytest.raises(ToolBufferLimitError):
        buffer.feed(final)


@pytest.mark.parametrize("native", [False, True])
def test_duplicate_choice_index_is_rejected_before_first_completion(native):
    buffer = ChatToolCallBuffer(native=native)
    invalid = terminal()
    invalid["choices"].append(event(content="late")["choices"][0])
    with pytest.raises(ExecutionFailure):
        buffer.feed(invalid)
    assert frames(buffer.feed(terminal()))

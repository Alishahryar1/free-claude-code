"""Chat has choice-level completion; JSON appearance and indexes are not ends."""

import pytest
from loguru import logger

from free_claude_code.core.chat_completions.tool_calls import (
    ChatToolCallBuffer,
    CompletedChatBatch,
    CompletedChatCall,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.tool_call_buffer import RetentionBudget, ToolBufferLimitError


def event(calls=(), *, content=None, finish=None, choice=0):
    delta = {"tool_calls": list(calls)} if calls else {}
    if content is not None:
        delta["content"] = content
    return {
        "id": "chat",
        "choices": [{"index": choice, "delta": delta, "finish_reason": finish}],
    }


def call(index, name=None, arguments="", tool_id=None):
    return {
        "index": index,
        "id": tool_id,
        "function": {"name": name, "arguments": arguments},
    }


def test_interleaved_calls_and_text_wait_for_choice_completion():
    buffer = ChatToolCallBuffer()
    assert buffer.feed(event([call(8, "foo", '{"x":', "a")])) == []
    assert buffer.feed(event(content="between")) == []
    assert buffer.feed(event([call(2, "other", "{}", "b")])) == []
    assert buffer.feed(event([call(8, "foo", "1}")])) == []
    result = buffer.feed(event(finish="tool_calls"))
    calls = [item for item in result if isinstance(item, CompletedChatCall)]
    assert [(item.index, item.body["id"], item.body["function"]) for item in calls] == [
        (0, "a", {"name": "foofoo", "arguments": '{"x":1}'}),
        (1, "b", {"name": "other", "arguments": "{}"}),
    ]
    assert isinstance(result[1], dict)
    assert result[1]["choices"][0]["delta"]["content"] == "between"


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_unsuccessful_choice_does_not_publish_calls_even_with_valid_json(finish):
    buffer = ChatToolCallBuffer()
    assert buffer.feed(event([call(0, "read", "{}", "a")])) == []
    result = buffer.feed(event(finish=finish))
    assert not any(isinstance(item, CompletedChatCall) for item in result)
    assert isinstance(result[-1], dict)
    assert result[-1]["choices"][0]["finish_reason"] == finish


def test_reused_indexes_keep_distinct_ids_and_unique_public_indexes():
    buffer = ChatToolCallBuffer()
    buffer.feed(event([call(0, "a", "{}", "a"), call(0, "b", "{}", "b")]))
    calls = [
        item
        for item in buffer.feed(event(finish="tool_calls"))
        if isinstance(item, CompletedChatCall)
    ]
    assert [(item.index, item.body["id"]) for item in calls] == [(0, "a"), (1, "b")]


def test_failed_call_does_not_discard_later_text():
    buffer = ChatToolCallBuffer()
    buffer.feed(event([call(0, "read", "{", "a")]))
    buffer.feed(event(content="later"))
    result = buffer.abandon()
    assert [
        item["choices"][0]["delta"]["content"]
        for item in result
        if isinstance(item, dict)
    ] == ["later"]


def test_native_mixed_envelope_and_fragment_metadata_remain_associated():
    buffer = ChatToolCallBuffer(native=True)
    first = event(
        [call(4, "Re", '{"x":', "a"), call(2, "other", "{}", "b")], content="before"
    )
    first["extension"] = {"request": 1}
    first["choices"][0]["logprobs"] = {"content": [{"token": "before"}]}
    first["choices"][0]["delta"]["tool_calls"][0]["extension"] = "first"
    second = event([call(4, "ad", "1}")])
    second["choices"][0]["delta"]["tool_calls"][0]["extension"] = "second"
    assert buffer.feed(first) == []
    assert buffer.feed(second) == []
    result = buffer.feed(event(finish="tool_calls"))
    assert len(result) == 1 and isinstance(result[0], CompletedChatBatch)
    frames = result[0].frames
    assert frames[0]["extension"] == {"request": 1}
    assert "extension" not in frames[1]
    assert frames[0]["choices"][0]["logprobs"] == first["choices"][0]["logprobs"]
    start = frames[0]["choices"][0]["delta"]["tool_calls"][0]
    tail = frames[1]["choices"][0]["delta"]["tool_calls"][0]
    assert start["function"]["name"] == "Read"
    assert "name" not in tail["function"]
    assert (start["extension"], tail["extension"]) == ("first", "second")
    assert start["function"]["arguments"] + tail["function"]["arguments"] == '{"x":1}'


@pytest.mark.parametrize("native", [False, True])
def test_choice_completion_identities_remain_bounded_after_output_drains(native):
    buffer = ChatToolCallBuffer(native=native, budget=RetentionBudget(max_records=4))
    with pytest.raises(ToolBufferLimitError):
        for choice in range(10):
            buffer.feed(event(finish="stop", choice=choice))


def test_native_choice_cannot_change_after_termination():
    buffer = ChatToolCallBuffer(native=True)
    buffer.feed(event(finish="stop"))
    with pytest.raises(ExecutionFailure):
        buffer.feed(event(content="late"))


def test_native_standard_mixed_chunk_streams_text_before_its_pending_call():
    buffer = ChatToolCallBuffer(native=True)
    first = event([call(0, "read", "{}", "a")], content="before")
    first["usage"] = {"prompt_tokens": 3}
    first["choices"][0]["logprobs"] = {"content": [{"token": "before"}]}
    early = buffer.feed(first)
    assert len(early) == 1 and isinstance(early[0], dict)
    assert early[0]["choices"][0]["delta"] == {"content": "before"}
    assert early[0]["choices"][0]["logprobs"] == first["choices"][0]["logprobs"]
    assert early[0]["usage"] == first["usage"]
    completed = buffer.feed(event(finish="tool_calls"))
    assert len(completed) == 1 and isinstance(completed[0], CompletedChatBatch)
    frames = completed[0].frames
    assert all("usage" not in frame for frame in frames)
    assert all(
        not choice.get("logprobs") and not choice["delta"].get("content")
        for frame in frames
        for choice in frame["choices"]
    )
    assert (
        frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read"
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("late_id", [False, True])
@pytest.mark.parametrize("placeholder", [None, "", " \t", 7])
def test_parallel_calls_keep_resolved_ids_across_unusable_fragments(
    native, late_id, placeholder
):
    buffer = ChatToolCallBuffer(native=native)
    assert (
        buffer.feed(event([call(4, "read", "{", placeholder if late_id else "a")]))
        == []
    )
    assert buffer.feed(event([call(2, "other", "{", "b")])) == []
    assert buffer.feed(event([call(2, arguments="}", tool_id=placeholder)])) == []
    if late_id:
        assert buffer.feed(event([call(4, tool_id="a")])) == []
    assert buffer.feed(event([call(4, arguments="}", tool_id=placeholder)])) == []
    result = buffer.feed(event(finish="tool_calls"))
    if native:
        assert len(result) == 1 and isinstance(result[0], CompletedChatBatch)
        calls = [
            fragment
            for frame in result[0].frames
            for choice in frame["choices"]
            for fragment in choice["delta"].get("tool_calls", [])
        ]
        for index, identity in enumerate(("a", "b")):
            fragments = [fragment for fragment in calls if fragment["index"] == index]
            assert fragments[0]["id"] == identity
            assert all(
                fragment["id"] == identity for fragment in fragments if "id" in fragment
            )
            assert (
                "".join(fragment["function"]["arguments"] for fragment in fragments)
                == "{}"
            )
    else:
        calls = [item for item in result if isinstance(item, CompletedChatCall)]
        assert [
            (item.body["id"], item.body["function"]["arguments"]) for item in calls
        ] == [("a", "{}"), ("b", "{}")]


def _frames(outputs):
    return [
        frame
        for item in outputs
        for frame in (item.frames if isinstance(item, CompletedChatBatch) else [item])
        if isinstance(frame, dict)
    ]


@pytest.fixture
def warning_sink():
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink)


@pytest.mark.parametrize("native", [False, True])
def test_empty_tool_calls_finish_is_reported_as_stop(native, warning_sink):
    buffer = ChatToolCallBuffer(native=native)
    text = event(content="thinking done")
    text["model"] = "mistral-large-4"
    buffer.feed(text)
    result = buffer.feed(event(finish="tool_calls"))
    frames = _frames(result)
    assert not any(isinstance(item, CompletedChatCall) for item in result)
    assert [frame["choices"][0]["finish_reason"] for frame in frames] == ["stop"]
    assert any(
        "tool_calls" in record["message"] and "stop" in record["message"]
        for record in warning_sink
    )


@pytest.mark.parametrize("native", [False, True])
def test_tool_calls_finish_with_a_real_call_is_preserved(native, warning_sink):
    buffer = ChatToolCallBuffer(native=native)
    buffer.feed(event([call(0, "read", "{}", "a")]))
    result = buffer.feed(event(finish="tool_calls"))
    frames = _frames(result)
    finishes = [
        frame["choices"][0]["finish_reason"]
        for frame in frames
        if frame["choices"][0]["finish_reason"] is not None
    ]
    assert finishes == ["tool_calls"]
    if native:
        assert isinstance(result[0], CompletedChatBatch)
    else:
        assert any(isinstance(item, CompletedChatCall) for item in result)
    assert not warning_sink


@pytest.mark.parametrize("native", [False, True])
def test_empty_tool_calls_finish_repeated_with_usage_stays_accounting(native):
    buffer = ChatToolCallBuffer(native=native)
    buffer.feed(event(finish="tool_calls"))
    trailer = event(finish="tool_calls")
    trailer["usage"] = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    assert _frames(buffer.feed(trailer)) == [{**trailer, "choices": []}]
    with pytest.raises(ExecutionFailure):
        buffer.feed(event(finish="tool_calls"))


def test_native_empty_tool_calls_finish_is_per_choice():
    buffer = ChatToolCallBuffer(native=True)
    buffer.feed(event([call(0, "read", "{}", "a")], choice=0))
    buffer.feed(event(finish="tool_calls", choice=1))
    result = buffer.feed(event(finish="tool_calls", choice=0))
    frames = _frames(result)
    reasons = {
        (frame["choices"][0]["index"], frame["choices"][0]["finish_reason"])
        for frame in frames
    }
    assert (0, "tool_calls") in reasons
    assert (1, "stop") in reasons


def test_nameless_call_still_fails_instead_of_becoming_stop():
    buffer = ChatToolCallBuffer()
    buffer.feed(event([call(0, None, "{}", "a")]))
    with pytest.raises(ExecutionFailure):
        buffer.feed(event(finish="tool_calls"))


def _finish_reasons(outputs):
    reasons = {}
    for frame in _frames(outputs):
        for choice in frame["choices"]:
            if choice.get("finish_reason") is not None:
                reasons[choice.get("index", 0)] = choice["finish_reason"]
    return reasons


@pytest.mark.parametrize("call_first", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_same_chunk_multi_choice_guards_each_choice_independently(native, call_first):
    # Choice 0 seals a real call before choice 1's guard runs in the same
    # chunk; a global "any sealed call" check would wrongly keep tool_calls
    # for choice 1. The native path splits a chunk per choice and clears its
    # calls once a group completes, so it only exposes that mistake when the
    # call arrives in an earlier chunk (call_first).
    buffer = ChatToolCallBuffer(native=native)
    reasons = {}
    first = {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
    if call_first:
        reasons.update(
            _finish_reasons(buffer.feed(event([call(0, "read", "{}", "a")])))
        )
    else:
        first["delta"] = {"tool_calls": [call(0, "read", "{}", "a")]}
    chunk = {
        "id": "chat",
        "model": "m",
        "choices": [first, {"index": 1, "delta": {}, "finish_reason": "tool_calls"}],
    }
    reasons.update(_finish_reasons(buffer.feed(chunk)))
    assert reasons == {0: "tool_calls", 1: "stop"}


@pytest.mark.parametrize("native", [False, True])
def test_reverse_order_empty_finish_is_still_per_choice(native):
    buffer = ChatToolCallBuffer(native=native)
    first = {
        "id": "chat",
        "model": "m",
        "choices": [
            {
                "index": 1,
                "delta": {"tool_calls": [call(0, "read", "{}", "a")]},
                "finish_reason": "tool_calls",
            }
        ],
    }
    assert _finish_reasons(buffer.feed(first)) == {1: "tool_calls"}
    assert _finish_reasons(buffer.feed(event(finish="tool_calls", choice=0))) == {
        0: "stop"
    }

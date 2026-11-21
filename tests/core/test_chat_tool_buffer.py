"""Chat has choice-level completion; JSON appearance and indexes are not ends."""

from collections.abc import Mapping, Sequence

import pytest

from free_claude_code.core.chat_completions.tool_calls import (
    ChatToolCallBuffer,
    CompletedChatBatch,
    CompletedChatCall,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.tool_call_buffer import RetentionBudget, ToolBufferLimitError
from tests.wire_payload import (
    array_payload,
    object_array,
    object_payload,
    string_payload,
)


def event(
    calls: Sequence[Mapping[str, JsonValue]] = (),
    *,
    content: str | None = None,
    finish: str | None = None,
    choice: int = 0,
) -> JsonObject:
    delta: JsonObject = {"tool_calls": list(calls)} if calls else {}
    if content is not None:
        delta["content"] = content
    return {
        "id": "chat",
        "choices": [{"index": choice, "delta": delta, "finish_reason": finish}],
    }


def call(
    index: int,
    name: str | None = None,
    arguments: str = "",
    tool_id: int | str | None = None,
):
    return {
        "index": index,
        "id": tool_id,
        "function": {"name": name, "arguments": arguments},
    }


def test_interleaved_calls_and_text_wait_for_choice_completion() -> None:
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
    assert (
        object_payload(object_array(result[1]["choices"])[0]["delta"])["content"]
        == "between"
    )


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_unsuccessful_choice_does_not_publish_calls_even_with_valid_json(
    finish: str,
) -> None:
    buffer = ChatToolCallBuffer()
    assert buffer.feed(event([call(0, "read", "{}", "a")])) == []
    result = buffer.feed(event(finish=finish))
    assert not any(isinstance(item, CompletedChatCall) for item in result)
    assert isinstance(result[-1], dict)
    assert object_array(result[-1]["choices"])[0]["finish_reason"] == finish


def test_reused_indexes_keep_distinct_ids_and_unique_public_indexes() -> None:
    buffer = ChatToolCallBuffer()
    buffer.feed(event([call(0, "a", "{}", "a"), call(0, "b", "{}", "b")]))
    calls = [
        item
        for item in buffer.feed(event(finish="tool_calls"))
        if isinstance(item, CompletedChatCall)
    ]
    assert [(item.index, item.body["id"]) for item in calls] == [(0, "a"), (1, "b")]


def test_failed_call_does_not_discard_later_text() -> None:
    buffer = ChatToolCallBuffer()
    buffer.feed(event([call(0, "read", "{", "a")]))
    buffer.feed(event(content="later"))
    result = buffer.abandon()
    assert [
        object_payload(object_array(item["choices"])[0]["delta"])["content"]
        for item in result
        if isinstance(item, dict)
    ] == ["later"]


def test_native_mixed_envelope_and_fragment_metadata_remain_associated() -> None:
    buffer = ChatToolCallBuffer(native=True)
    first = event(
        [call(4, "Re", '{"x":', "a"), call(2, "other", "{}", "b")], content="before"
    )
    first["extension"] = {"request": 1}
    object_array(first["choices"])[0]["logprobs"] = {"content": [{"token": "before"}]}
    object_array(
        object_payload(object_array(first["choices"])[0]["delta"])["tool_calls"]
    )[0]["extension"] = "first"
    second = event([call(4, "ad", "1}")])
    object_array(
        object_payload(object_array(second["choices"])[0]["delta"])["tool_calls"]
    )[0]["extension"] = "second"
    assert buffer.feed(first) == []
    assert buffer.feed(second) == []
    result = buffer.feed(event(finish="tool_calls"))
    assert len(result) == 1 and isinstance(result[0], CompletedChatBatch)
    frames = result[0].frames
    assert frames[0]["extension"] == {"request": 1}
    assert "extension" not in frames[1]
    assert (
        object_array(frames[0]["choices"])[0]["logprobs"]
        == object_array(first["choices"])[0]["logprobs"]
    )
    start = array_payload(
        object_payload(object_array(frames[0]["choices"])[0]["delta"])["tool_calls"]
    )[0]
    tail = array_payload(
        object_payload(object_array(frames[1]["choices"])[0]["delta"])["tool_calls"]
    )[0]
    assert object_payload(object_payload(start)["function"])["name"] == "Read"
    assert "name" not in object_payload(object_payload(tail)["function"])
    assert (object_payload(start)["extension"], object_payload(tail)["extension"]) == (
        "first",
        "second",
    )
    assert (
        string_payload(object_payload(object_payload(start)["function"])["arguments"])
        + string_payload(object_payload(object_payload(tail)["function"])["arguments"])
        == '{"x":1}'
    )


@pytest.mark.parametrize("native", [False, True])
def test_choice_completion_identities_remain_bounded_after_output_drains(
    native: bool,
) -> None:
    buffer = ChatToolCallBuffer(native=native, budget=RetentionBudget(max_records=4))
    with pytest.raises(ToolBufferLimitError):
        for choice in range(10):
            buffer.feed(event(finish="stop", choice=choice))


def test_native_choice_cannot_change_after_termination() -> None:
    buffer = ChatToolCallBuffer(native=True)
    buffer.feed(event(finish="stop"))
    with pytest.raises(ExecutionFailure):
        buffer.feed(event(content="late"))


def test_native_standard_mixed_chunk_streams_text_before_its_pending_call() -> None:
    buffer = ChatToolCallBuffer(native=True)
    first = event([call(0, "read", "{}", "a")], content="before")
    first["usage"] = {"prompt_tokens": 3}
    object_array(first["choices"])[0]["logprobs"] = {"content": [{"token": "before"}]}
    early = buffer.feed(first)
    assert len(early) == 1 and isinstance(early[0], dict)
    assert object_array(early[0]["choices"])[0]["delta"] == {"content": "before"}
    assert (
        object_array(early[0]["choices"])[0]["logprobs"]
        == object_array(first["choices"])[0]["logprobs"]
    )
    assert early[0]["usage"] == first["usage"]
    completed = buffer.feed(event(finish="tool_calls"))
    assert len(completed) == 1 and isinstance(completed[0], CompletedChatBatch)
    frames = completed[0].frames
    assert all("usage" not in frame for frame in frames)
    assert all(
        not object_payload(choice).get("logprobs")
        and not object_payload(object_payload(choice)["delta"]).get("content")
        for frame in frames
        for choice in array_payload(frame["choices"])
    )
    assert (
        object_payload(
            object_array(
                object_payload(object_array(frames[0]["choices"])[0]["delta"])[
                    "tool_calls"
                ]
            )[0]["function"]
        )["name"]
        == "read"
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("late_id", [False, True])
@pytest.mark.parametrize("placeholder", [None, "", " \t", 7])
def test_parallel_calls_keep_resolved_ids_across_unusable_fragments(
    native: bool, late_id: bool, placeholder: int | str | None
) -> None:
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
            for choice in array_payload(frame["choices"])
            for fragment in array_payload(
                object_payload(object_payload(choice)["delta"]).get("tool_calls", [])
            )
        ]
        for index, identity in enumerate(("a", "b")):
            fragments = [
                fragment
                for fragment in calls
                if object_payload(fragment)["index"] == index
            ]
            assert object_payload(fragments[0])["id"] == identity
            assert all(
                object_payload(fragment)["id"] == identity
                for fragment in fragments
                if "id" in object_payload(fragment)
            )
            assert (
                "".join(
                    string_payload(
                        object_payload(object_payload(fragment)["function"])[
                            "arguments"
                        ]
                    )
                    for fragment in fragments
                )
                == "{}"
            )
    else:
        calls = [item for item in result if isinstance(item, CompletedChatCall)]
        assert [
            (item.body["id"], object_payload(item.body["function"])["arguments"])
            for item in calls
        ] == [("a", "{}"), ("b", "{}")]

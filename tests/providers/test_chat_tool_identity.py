"""Logical identity is independent of provider indexes and presentation."""

from collections.abc import Callable
from copy import deepcopy

import pytest

from free_claude_code.core.chat_completions.tool_calls import (
    AmbiguousToolCallIdentityError,
    ChatToolCallBuffer,
    CompletedChatCall,
)
from free_claude_code.core.json_types import JsonObject
from tests.core.test_chat_tool_buffer import event
from tests.wire_payload import object_payload


def call(
    index: bool | int | str | None = 0,
    tool_id: str | None = "call_a",
    arguments: str = "{}",
    *,
    name: str | None = "read",
    metadata: JsonObject | None = None,
) -> JsonObject:
    delta: JsonObject = {
        "id": tool_id,
        "function": {"name": name, "arguments": arguments},
    }
    if index != "missing":
        delta["index"] = index
    if metadata is not None:
        delta["extra_content"] = metadata
    return delta


@pytest.fixture
def call_consumer():
    buffer = ChatToolCallBuffer()

    def add(delta: JsonObject):
        assert buffer.feed(event([delta])) == []

    def completed():
        output = deepcopy(buffer).feed(event(finish="tool_calls"))
        return [
            {
                "id": item.body.get("id"),
                "name": object_payload(item.body["function"])["name"],
                "arguments": object_payload(item.body["function"])["arguments"],
                "metadata": item.body.get("extra_content"),
            }
            for item in output
            if isinstance(item, CompletedChatCall)
        ]

    return add, completed


@pytest.mark.parametrize(
    "indexes", [(0, 1), (0, 0), (None, None), ("missing", "missing")]
)
@pytest.mark.parametrize("fragmented", [False, True])
def test_parallel_calls_keep_arguments_and_metadata(
    call_consumer: tuple[Callable[[JsonObject], None], Callable[[], list[JsonObject]]],
    indexes: tuple[int | str | None, ...],
    fragmented: bool,
):
    add, completed = call_consumer
    arguments = ['{"path":"a"}', '{"path":"b"}']
    for index, tool_id, argument in zip(
        indexes, ("call_a", "call_b"), arguments, strict=True
    ):
        add(
            call(
                index,
                tool_id,
                argument[:8] if fragmented else argument,
                metadata={"google": {"thought_signature": tool_id}},
            )
        )
    if fragmented:
        for position in (1, 0):
            add(
                call(
                    indexes[position],
                    ("call_a", "call_b")[position],
                    arguments[position][8:],
                    name=None,
                )
            )

    assert completed() == [
        {
            "id": "call_a",
            "name": "read",
            "arguments": '{"path":"a"}',
            "metadata": {"google": {"thought_signature": "call_a"}},
        },
        {
            "id": "call_b",
            "name": "read",
            "arguments": '{"path":"b"}',
            "metadata": {"google": {"thought_signature": "call_b"}},
        },
    ]


def test_allocated_slots_do_not_alias_later_raw_indexes(
    call_consumer: tuple[Callable[[JsonObject], None], Callable[[], list[JsonObject]]],
):
    add, completed = call_consumer
    for index, tool_id, arguments in (
        (0, "call_a", '{"path":"a"}'),
        (0, "call_b", '{"path":"b"}'),
        (1, "call_c", '{"path":"c"}'),
    ):
        add(call(index, tool_id, arguments))
    assert [item["arguments"] for item in completed()] == [
        '{"path":"a"}',
        '{"path":"b"}',
        '{"path":"c"}',
    ]


@pytest.mark.parametrize("index", [None, "missing", -1, "invalid"])
def test_unusable_index_keeps_fragments_of_one_id_together(
    call_consumer: tuple[Callable[[JsonObject], None], Callable[[], list[JsonObject]]],
    index: int | str | None,
):
    add, completed = call_consumer
    add(call(index, "call_a", '{"path":', name=None))
    add(call(index, "call_a", '"a"}'))
    assert [item["arguments"] for item in completed()] == ['{"path":"a"}']


@pytest.mark.parametrize(
    "first,second",
    [
        (call(0, None, '{"path":', name=None), call(0, "call_a", '"a"}')),
        (call(None, "call_a", '{"path":', name=None), call(4, "call_a", '"a"}')),
        (call(0, "call_a", '{"path":'), call(0, None, '"a"}', name=None)),
    ],
)
def test_partial_identity_can_learn_missing_fields(
    call_consumer: tuple[Callable[[JsonObject], None], Callable[[], list[JsonObject]]],
    first: JsonObject,
    second: JsonObject,
):
    add, completed = call_consumer
    add(first)
    add(second)
    assert [item["arguments"] for item in completed()] == ['{"path":"a"}']


@pytest.mark.parametrize(
    "identities",
    [
        [(None, "call_a"), (1, None), (1, "call_b")],
        [(0, None), (None, "call_b"), (0, "call_a")],
    ],
)
def test_disjoint_partial_identities_keep_content_separate(
    call_consumer: tuple[Callable[[JsonObject], None], Callable[[], list[JsonObject]]],
    identities: list[tuple[int | str | None, str | None]],
):
    add, completed = call_consumer
    for (index, tool_id), path in zip(identities[:2], ("a", "b"), strict=True):
        add(
            call(
                index,
                tool_id,
                '{"path":"' + path + '"}',
                metadata={"google": {"thought_signature": path}},
            )
        )
    index, tool_id = identities[2]
    add(call(index, tool_id, "", name=None))
    assert [(item["arguments"], item["metadata"]) for item in completed()] == [
        ('{"path":"a"}', {"google": {"thought_signature": "a"}}),
        ('{"path":"b"}', {"google": {"thought_signature": "b"}}),
    ]


def test_new_index_does_not_attach_to_unindexed_calls(
    call_consumer: tuple[Callable[[JsonObject], None], Callable[[], list[JsonObject]]],
):
    add, completed = call_consumer
    add(call(None, "call_a", '{"path":"a"}'))
    add(call(None, "call_b", '{"path":"b"}'))
    add(call(3, None, '{"path":"c"}'))
    assert [item["arguments"] for item in completed()] == [
        '{"path":"a"}',
        '{"path":"b"}',
        '{"path":"c"}',
    ]


@pytest.mark.parametrize(
    "index,tool_id",
    [(None, None), ("missing", None), (-1, ""), ("invalid", " "), (False, None)],
)
@pytest.mark.parametrize("existing", [False, True])
def test_unidentified_fragment_is_rejected_before_content_changes(
    call_consumer: tuple[Callable[[JsonObject], None], Callable[[], list[JsonObject]]],
    index: bool | int | str | None,
    tool_id: str | None,
    existing: bool,
):
    add, completed = call_consumer
    if existing:
        add(call(0, "call_a", '{"path":"a"}', metadata={"original": True}))
    before = deepcopy(completed())
    with pytest.raises(AmbiguousToolCallIdentityError):
        add(call(index, tool_id, '{"path":"b"}', metadata={"wrong": True}))
    assert completed() == before


@pytest.mark.parametrize(
    "initial,fragment",
    [
        ([call(0, "call_a"), call(0, "call_b")], call(0, None, "bad")),
        ([call(0, "same"), call(1, "same")], call(None, "same", "bad")),
        ([call(0, None), call(1, None)], call(None, None, "bad")),
        ([call(0, "call_a"), call(None, "call_b")], call(0, None, "bad")),
        ([call(0, "call_a"), call(1, None)], call(None, "call_a", "bad")),
        ([call(0, None), call(None, "call_a")], call(0, "call_a", "bad")),
    ],
)
def test_ambiguous_fragment_is_rejected_before_content_changes(
    call_consumer: tuple[Callable[[JsonObject], None], Callable[[], list[JsonObject]]],
    initial: list[JsonObject],
    fragment: JsonObject,
):
    add, completed = call_consumer
    for delta in initial:
        add(delta)
    before = deepcopy(completed())
    with pytest.raises(AmbiguousToolCallIdentityError):
        add(fragment)
    assert completed() == before

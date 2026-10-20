"""Chat tool identity survives missing or reused upstream indexes."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import ToolSchema
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.providers.admission import ProviderOperationKind
from free_claude_code.providers.failure_policy import RetryableToolProtocolError
from free_claude_code.providers.openai_chat.tool_calls import (
    OpenAIToolCallAssembler,
    OpenAIToolCallCollector,
)
from free_claude_code.providers.request_recovery import RequestRecovery
from tests.providers.test_streaming_errors import (
    ClosableAsyncStreamMock,
    _make_anthropic_output,
    _make_chunk,
    _make_provider,
    _make_request,
    _make_stream_runner,
    _recovery_output,
)

SCHEMAS = {"read": ToolSchema(name="read", input_schema={"type": "object"})}


def call(index=0, tool_id="call_a", arguments="{}", *, name="read", metadata=None):
    delta = {
        "id": tool_id,
        "function": {"name": name, "arguments": arguments},
    }
    if index != "missing":
        delta["index"] = index
    if metadata is not None:
        delta["extra_content"] = metadata
    return delta


def sdk_call(delta):
    return SimpleNamespace(
        **{**delta, "function": SimpleNamespace(**delta["function"])}
    )


@pytest.fixture(params=["stream", "collector"])
def call_consumer(request):
    output = _make_anthropic_output()
    assembler = OpenAIToolCallAssembler(reserved_tool_ids=())
    collector = OpenAIToolCallCollector()

    def add(delta):
        if request.param == "stream":
            list(assembler.process_tool_call(delta, output))
        else:
            collector.add(sdk_call(delta))

    def completed():
        if request.param == "stream":
            return [
                {
                    "id": state.tool_id,
                    "name": state.name,
                    "arguments": state.content,
                    "metadata": state.extra_content,
                }
                for state in output.tool_states.values()
            ]
        calls = collector.completed_calls(SCHEMAS)
        assert calls is not None
        return [
            {
                "id": item["id"],
                "name": item["function"]["name"],
                "arguments": item["function"]["arguments"],
                "metadata": item.get("extra_content"),
            }
            for item in calls
        ]

    return add, completed


@pytest.mark.parametrize(
    "indexes", [(0, 1), (0, 0), (None, None), ("missing", "missing")]
)
@pytest.mark.parametrize("fragmented", [False, True])
def test_parallel_calls_keep_arguments_and_metadata(call_consumer, indexes, fragmented):
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


def test_allocated_slots_do_not_alias_later_raw_indexes(call_consumer):
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
def test_unusable_index_keeps_fragments_of_one_id_together(call_consumer, index):
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
def test_partial_identity_can_learn_missing_fields(call_consumer, first, second):
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
def test_disjoint_partial_identities_keep_content_separate(call_consumer, identities):
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


def test_new_index_does_not_attach_to_unindexed_calls(call_consumer):
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
    call_consumer, index, tool_id, existing
):
    add, completed = call_consumer
    if existing:
        add(call(0, "call_a", '{"path":"a"}', metadata={"original": True}))
    before = deepcopy(completed())
    with pytest.raises(RetryableToolProtocolError):
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
    call_consumer, initial, fragment
):
    add, completed = call_consumer
    for delta in initial:
        add(delta)
    before = deepcopy(completed())
    with pytest.raises(RetryableToolProtocolError):
        add(fragment)
    assert completed() == before


def test_collector_preserves_sparse_numeric_order():
    collector = OpenAIToolCallCollector()
    collector.add(sdk_call(call(8, "call_a", '{"path":"a"}')))
    collector.add(sdk_call(call(2, "call_b", '{"path":"b"}')))
    result = collector.completed_calls(SCHEMAS)
    assert result is not None
    assert [item["id"] for item in result] == ["call_b", "call_a"]


def test_name_flush_uses_logical_slots_for_repeated_index():
    request = _make_request(
        tools=[
            {"name": "tool", "input_schema": {"type": "object"}},
            {"name": "tool." + "x" * 70, "input_schema": {"type": "object"}},
        ]
    )
    codec = OpenAIToolNameCodec.from_request(request)
    output = _make_anthropic_output()
    assembler = OpenAIToolCallAssembler(reserved_tool_ids=())
    names = {}
    for tool_id, argument in (("call_a", '{"path":"a"}'), ("call_b", '{"path":"b"}')):
        assert (
            list(
                assembler.process_tool_call(
                    call(0, tool_id, argument, name="tool"),
                    output,
                    tool_names=codec,
                    tool_name_buffers=names,
                )
            )
            == []
        )
    list(
        assembler.flush_tool_name_buffers(
            output,
            tool_names=codec,
            tool_name_buffers=names,
            tool_argument_aliases={},
            tool_argument_alias_buffers={},
        )
    )
    assert [
        (state.tool_id, state.name, state.content)
        for state in output.tool_states.values()
    ] == [
        ("call_a", "tool", '{"path":"a"}'),
        ("call_b", "tool", '{"path":"b"}'),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("recovered_ids", [(None, None), ("same", "same")])
async def test_recovery_import_discards_unstarted_call_state(recovered_ids):
    provider = _make_provider()
    runner = _make_stream_runner(
        provider,
        request=_make_request(
            tools=[
                {"name": name, "input_schema": {"type": "object"}}
                for name in ("read", "tool", "tool." + "x" * 70)
            ]
        ),
    )
    assembler = runner._new_stream_assembler(output_reasoning=True)
    list(assembler.start_events())
    list(
        assembler.feed(
            _make_chunk(
                content="visible",
                tool_calls=[
                    sdk_call(
                        call(
                            0,
                            "abandoned",
                            '{"old":',
                            name="tool",
                            metadata={"old": True},
                        )
                    )
                ],
            )
        )
    )
    with patch.object(
        runner,
        "_collect_recovery_output",
        new_callable=AsyncMock,
        return_value=_recovery_output(
            text="visible",
            tool_calls=(
                {
                    "id": recovered_ids[0],
                    "function": {"name": "read", "arguments": "{}"},
                },
                {
                    "id": recovered_ids[1],
                    "function": {"name": "read", "arguments": '{"path":"b"}'},
                    "extra_content": {"new": True},
                },
            ),
        ),
    ):
        result = await runner._recovery_events(
            body=runner._body,
            assembler=assembler,
            error=TimeoutError("cutoff"),
            tool_argument_alias_buffers=assembler.tool_argument_alias_buffers,
            output_reasoning=True,
            request_recovery=RequestRecovery(provider._admission.start_execution()),
        )
    assert result is not None
    events = parse_sse_text("".join(result))
    starts = [
        event.data["content_block"]
        for event in events
        if event.event == "content_block_start"
    ]
    assert len(starts) == 2
    assert len({start["id"] for start in starts}) == 2
    assert "abandoned" not in {start["id"] for start in starts}
    assert "extra_content" not in starts[0]
    assert starts[1]["extra_content"] == {"new": True}
    assert [state.content for state in assembler.output.tool_states.values()] == [
        "{}",
        '{"path":"b"}',
    ]
    assert assembler.output.accumulated_text == "visible"
    assert assembler._tool_name_buffers == {}
    await provider.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [None, "same", "before"])
@pytest.mark.parametrize("fragment_index", [0, None])
async def test_collector_identity_failure_respects_stop_and_closes_attempts(
    stop, fragment_index
):
    provider = _make_provider()
    runner = _make_stream_runner(
        provider,
        request=_make_request(
            tools=[{"name": "read", "input_schema": {"type": "object"}}]
        ),
    )
    chunks = [
        _make_chunk(
            tool_calls=[sdk_call(call(0, "call_a")), sdk_call(call(0, "call_b"))]
        )
    ]
    if stop == "before":
        chunks.append(_make_chunk(finish_reason="tool_calls"))
    chunks.append(
        _make_chunk(
            tool_calls=[sdk_call(call(fragment_index, None, "uncertain", name=None))],
            finish_reason="tool_calls" if stop == "same" else None,
        )
    )
    first = ClosableAsyncStreamMock(chunks)
    winner = ClosableAsyncStreamMock(
        [
            _make_chunk(tool_calls=[sdk_call(call(0, "winner", '{"path":"winning"}'))]),
            _make_chunk(finish_reason="tool_calls"),
        ]
    )
    streams = iter((first, winner))

    async def create_stream(**kwargs):
        stream = next(streams)
        if stream is winner:
            assert first.close_calls == 1
        return stream

    execution = provider._admission.start_execution()
    try:
        with patch.object(
            provider._client.chat.completions, "create", side_effect=create_stream
        ):
            if stop:
                with pytest.raises(RetryableToolProtocolError):
                    await runner._collect_recovery_output(
                        runner._body,
                        include_reasoning=True,
                        request_recovery=RequestRecovery(execution),
                        operation_kind=ProviderOperationKind.CONTINUATION,
                    )
            else:
                result = await runner._collect_recovery_output(
                    runner._body,
                    include_reasoning=True,
                    request_recovery=RequestRecovery(execution),
                    operation_kind=ProviderOperationKind.CONTINUATION,
                )
                assert [item["id"] for item in result.tool_calls] == ["winner"]
                assert (
                    result.tool_calls[0]["function"]["arguments"]
                    == '{"path":"winning"}'
                )
        assert execution.attempts_started == (1 if stop else 2)
        assert first.close_calls == 1
        assert winner.close_calls == (0 if stop else 1)
    finally:
        await runner._request_client.aclose()
        await provider.cleanup()


@pytest.mark.parametrize("index", [0, None])
def test_colliding_calls_keep_name_and_argument_alias_buffers_separate(index):
    original = "tool." + "x" * 70
    request = _make_request(
        tools=[
            {"name": original, "input_schema": {"type": "object"}},
            {"name": "tool", "input_schema": {"type": "object"}},
        ]
    )
    codec = OpenAIToolNameCodec.from_request(request)
    assembler = OpenAIToolCallAssembler(reserved_tool_ids=())
    output = _make_anthropic_output()
    names = {}
    arguments = {}
    aliases = {name: {"_fcc_arg_type": "type"} for name in (original, "tool")}
    for delta in (
        call(index, "call_a", '{"_fcc_arg_type":"a', name=codec.encode(original)),
        call(index, "call_b", '{"_fcc_arg_type":"b', name="tool"),
        call(index, "call_b", '"}', name=None),
        call(index, "call_a", '"}', name=None),
    ):
        list(
            assembler.process_tool_call(
                delta,
                output,
                tool_names=codec,
                tool_name_buffers=names,
                tool_argument_aliases=aliases,
                tool_argument_alias_buffers=arguments,
            )
        )
    list(
        assembler.flush_tool_name_buffers(
            output,
            tool_names=codec,
            tool_name_buffers=names,
            tool_argument_aliases=aliases,
            tool_argument_alias_buffers=arguments,
        )
    )
    assert [
        (state.tool_id, state.name, json.loads(state.content))
        for state in output.tool_states.values()
    ] == [("call_a", original, {"type": "a"}), ("call_b", "tool", {"type": "b"})]


def test_name_prefix_without_output_state_reserves_its_own_slot():
    request = _make_request(
        tools=[
            {"name": "tool", "input_schema": {"type": "object"}},
            {"name": "tool." + "x" * 70, "input_schema": {"type": "object"}},
        ]
    )
    codec = OpenAIToolNameCodec.from_request(request)
    assembler = OpenAIToolCallAssembler(reserved_tool_ids=())
    output = _make_anthropic_output()
    names = {}
    for delta in (
        call(8, None, "", name="tool"),
        call(2, "call_b", '{"path":"b"}', name="tool"),
        call(8, "call_a", '{"path":"a"}', name=None),
    ):
        assert (
            list(
                assembler.process_tool_call(
                    delta,
                    output,
                    tool_names=codec,
                    tool_name_buffers=names,
                )
            )
            == []
        )
    events = list(
        assembler.flush_tool_name_buffers(
            output,
            tool_names=codec,
            tool_name_buffers=names,
            tool_argument_aliases={},
            tool_argument_alias_buffers={},
        )
    )
    starts = [
        event.data["content_block"]["id"]
        for event in parse_sse_text("".join(events))
        if event.event == "content_block_start"
    ]
    assert starts == ["call_a", "call_b"]
    assert {state.tool_id: state.content for state in output.tool_states.values()} == {
        "call_a": '{"path":"a"}',
        "call_b": '{"path":"b"}',
    }

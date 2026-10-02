"""Recovery decisions use source completion before either public projection."""

from copy import deepcopy
from typing import Any

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.core.recovery import RecoveryCheckpoint
from tests.providers.support import attempt_events
from tests.providers.test_history_transports import _harness
from tests.providers.test_native_tool_arguments import tool_events
from tests.providers.test_recovery_completion_limits import _limited_call


def _responses_call(arguments="{}", *, status="completed", terminal="completed"):
    events = _limited_call("responses")
    item = {
        **events[1]["item"],
        "call_id": "call_observed",
        "arguments": arguments,
    }
    if status is None:
        item.pop("status", None)
    else:
        item["status"] = status
    events[1]["item"]["call_id"] = item["call_id"]
    events[2]["delta"] = arguments
    done = {
        "type": "response.output_item.done",
        "sequence_number": 3,
        "output_index": 0,
        "item": item,
    }
    final = events.pop()
    final["type"] = "response." + terminal
    final["sequence_number"] = 4
    final["response"]["status"] = terminal
    final["response"]["output"] = [deepcopy(item)]
    if terminal == "completed":
        final["response"]["incomplete_details"] = None
    return [*events, done, final]


async def _collect(protocol, wire, events):
    async with _harness(protocol, lambda _: (200, events)) as (send, bodies, _):
        output = "".join(
            [
                frame
                async for frame in send(
                    wire,
                    [{"role": "user", "content": "Write the file"}],
                    tools=[
                        {
                            "type": "function",
                            "name": name,
                            "parameters": {"type": "object"},
                            "strict": False,
                        }
                        if wire == "responses"
                        else {"name": name, "input_schema": {"type": "object"}}
                        for name in ("Write", "read")
                    ],
                )
            ]
        )
    return output, parse_sse_text(output), bodies


@pytest.mark.asyncio
async def test_only_new_snapshot_input_counts_as_progress_in_real_transport():
    events = _responses_call('{"x":1}')
    events[2]["delta"] = '{"x":'
    complete_input = {
        "type": "response.function_call_arguments.done",
        "output_index": 0,
        "item_id": events[1]["item"]["id"],
        "arguments": '{"x":1}',
        "sequence_number": 3,
    }
    events[3:3] = [deepcopy(complete_input) for _ in range(4)]
    async with (
        _harness("responses", lambda _: (200, events)) as (_, bodies, provider),
        provider.open_responses(
            OpenAIResponsesRequest(model="model", input="hello"),
            input_tokens=0,
            request_id="snapshot-progress",
            response_model="public",
            reasoning=DEFAULT_REASONING_POLICY,
        ) as candidate,
    ):
        decoded = [
            event
            async for event in attempt_events(
                candidate,
                RecoveryCheckpoint("responses"),
                wait_for_recovery=True,
            )
        ]
    progress = [event for event in decoded if event.progress]
    assert [event.source.kind for event in progress] == [
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
    ]
    assert progress[0].source.payload["delta"] == '{"x":'
    assert progress[1].source.payload["arguments"] == '{"x":1}'
    assert decoded[-1].completed and len(bodies) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "status", ["incomplete", "in_progress", "failed", "future_status"]
)
@pytest.mark.parametrize("terminal", ["incomplete", "disconnect"])
async def test_unfinished_item_never_publishes_even_with_valid_json(
    wire, status, terminal
):
    events = _responses_call(status=status, terminal="incomplete")
    if terminal == "disconnect":
        events.pop()
    output, parsed, bodies = await _collect("responses", wire, events)
    assert "call_observed" not in output
    if terminal == "incomplete":
        assert len(bodies) == 1
        if wire == "responses":
            assert parsed[-1].event == "response.incomplete"
        else:
            assert parsed[-2].data["delta"]["stop_reason"] == "max_tokens"
    else:
        assert parsed[-1].event == (
            "error" if wire == "messages" else "response.failed"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_nonempty_added_text_and_matching_snapshot_publish_once(wire):
    events = _responses_call()
    text = {
        "id": "msg_text",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "Already supplied", "annotations": []}
        ],
    }
    events[1]["item"] = {**text, "status": "in_progress"}
    del events[2]
    events[2]["item"] = text
    events[-1]["response"]["output"] = [text]
    _, parsed, _ = await _collect("responses", wire, events)
    deltas = [event.data.get("delta") for event in parsed]
    text_out = "".join(
        delta
        if isinstance(delta, str)
        else delta.get("text", "")
        if isinstance(delta, dict)
        else ""
        for delta in deltas
    )
    assert text_out == "Already supplied"
    assert parsed[-1].event == (
        "response.completed" if wire == "responses" else "message_stop"
    )
    if wire == "responses":
        assert (
            parsed[-1].data["response"]["output"][0]["content"][0]["text"]
            == "Already supplied"
        )


@pytest.mark.asyncio
async def test_terminal_snapshot_refreshes_metadata_without_republishing_item():
    events = _responses_call()
    item: dict[str, Any] = {
        "id": "msg_citation",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Cited answer", "annotations": []}],
    }
    events[1]["item"] = {**item, "status": "in_progress", "content": []}
    events[2] = {
        "type": "response.output_text.delta",
        "output_index": 0,
        "item_id": item["id"],
        "content_index": 0,
        "delta": "Cited answer",
    }
    events[3]["item"] = deepcopy(item)
    citation = {
        "type": "url_citation",
        "start_index": 0,
        "end_index": 5,
        "url": "https://example.invalid/source",
        "title": "Source",
    }
    final_item = deepcopy(item)
    final_item["content"][0]["annotations"] = [citation]
    events[-1]["response"]["output"] = [final_item]

    _, parsed, bodies = await _collect("responses", "responses", events)

    assert len(bodies) == 1
    assert parsed[-1].data["response"]["output"] == [final_item]
    assert sum(event.event == "response.output_item.done" for event in parsed) == 1
    assert (
        "".join(
            event.data["delta"]
            for event in parsed
            if event.event == "response.output_text.delta"
        )
        == "Cited answer"
    )


@pytest.mark.asyncio
async def test_unknown_native_item_with_null_content_remains_opaque():
    events = _responses_call()
    native = {
        "id": "native",
        "type": "future_native",
        "content": None,
        "extension": {"keep": True},
    }
    events[1]["item"] = native
    del events[2]
    events[2]["item"] = native
    events[-1]["response"]["output"] = [native]
    _, parsed, bodies = await _collect("responses", "responses", events)
    assert len(bodies) == 1
    assert parsed[-1].data["response"]["output"] == [native]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("arguments", ['{"path":', "[]", "null", ""])
@pytest.mark.parametrize("terminal", ["completed", "incomplete"])
async def test_invalid_closed_input_waits_for_real_terminal(wire, arguments, terminal):
    output, parsed, bodies = await _collect(
        "responses", wire, _responses_call(arguments, terminal=terminal)
    )
    assert "call_observed" not in output
    if terminal == "incomplete":
        assert len(bodies) == 1
        if wire == "responses":
            assert parsed[-1].event == "response.incomplete"
        else:
            assert parsed[-2].data["delta"]["stop_reason"] == "max_tokens"
    else:
        assert 1 < len(bodies) <= 5
        assert parsed[-1].event == (
            "error" if wire == "messages" else "response.failed"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_complete_snapshot_reconciles_missing_argument_suffix(wire):
    arguments = '{"path": "file.txt"}'
    events = _responses_call(arguments)
    events[2]["delta"] = '{"path":'
    events.pop(-2)
    output, parsed, bodies = await _collect("responses", wire, events)
    assert len(bodies) == 1
    if wire == "messages":
        raw = "".join(
            event.data["delta"]["partial_json"]
            for event in parsed
            if event.event == "content_block_delta"
            and event.data["delta"]["type"] == "input_json_delta"
        )
    else:
        raw = next(
            event.data["item"]["arguments"]
            for event in parsed
            if event.event == "response.output_item.done"
        )
    assert raw == arguments
    assert "call_observed" in output


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_incomplete_response_can_contain_a_complete_snapshot_only_call(wire):
    events = _responses_call(terminal="incomplete")
    output, parsed, bodies = await _collect("responses", wire, [events[0], events[-1]])
    assert len(bodies) == 1
    assert "call_observed" in output
    if wire == "messages":
        assert parsed[-2].data["delta"]["stop_reason"] == "max_tokens"
    else:
        assert parsed[-1].event == "response.incomplete"


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_valid_final_call_with_optional_status_omitted_releases_once(wire):
    output, parsed, bodies = await _collect(
        "responses", wire, _responses_call('{"n":1e400}', status=None)
    )
    assert len(bodies) == 1
    if wire == "messages":
        calls = [
            event
            for event in parsed
            if event.event == "content_block_start"
            and event.data["content_block"]["type"] == "tool_use"
        ]
    else:
        calls = [
            event
            for event in parsed
            if event.event == "response.output_item.done"
            and event.data["item"]["type"] == "function_call"
        ]
    assert len(calls) == 1
    assert "1e400" in output


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("protocol", ["messages", "chat"])
@pytest.mark.parametrize("limited", [False, True])
async def test_native_invalid_input_does_not_become_a_successful_call(
    protocol, wire, limited
):
    events = tool_events(protocol, '{"path":')
    if limited:
        if protocol == "chat":
            events[-1]["choices"][0]["finish_reason"] = "length"
        else:
            events[-2]["delta"]["stop_reason"] = "max_tokens"
    output, parsed, bodies = await _collect(protocol, wire, events)
    assert "call_probe" not in output
    if limited:
        assert len(bodies) == 1
        if wire == "messages":
            assert parsed[-2].data["delta"]["stop_reason"] == "max_tokens"
        else:
            assert parsed[-1].event == "response.incomplete"
    else:
        assert 1 < len(bodies) <= 5
        assert parsed[-1].event == (
            "error" if wire == "messages" else "response.failed"
        )


@pytest.mark.asyncio
async def test_terminal_snapshot_cannot_remove_an_exposed_call():
    events = _responses_call()
    events[-1]["response"]["output"] = []
    _, parsed, bodies = await _collect("responses", "responses", events)
    assert len(bodies) == 1
    assert parsed[-1].data["response"]["output"][0]["call_id"] == "call_observed"


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("closed", [False, True])
async def test_hidden_or_incomplete_native_reasoning_prevents_tool_salvage(
    wire, closed
):
    events = _responses_call()
    events.pop()
    for event in events[1:]:
        event["output_index"] = 1
    reasoning = {
        "type": "reasoning",
        "id": "rs_pending",
        "summary": [],
        "status": "in_progress",
    }
    prefix = [
        {"type": "response.output_item.added", "output_index": 0, "item": reasoning}
    ]
    if closed:
        prefix.append(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    **reasoning,
                    "status": "incomplete",
                    "encrypted_content": "partial-native-state",
                },
            }
        )
    events[1:1] = prefix
    _, parsed, bodies = await _collect("responses", wire, events)
    assert len(bodies) == 1
    assert parsed[-1].event == ("error" if wire == "messages" else "response.failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("streamed_starts", [False, True])
async def test_snapshot_items_cannot_bypass_an_earlier_unpublished_call(
    wire, streamed_starts
):
    events = _responses_call(status="incomplete", terminal="incomplete")
    pending = events[-1]["response"]["output"][0]
    tail = {
        "type": "message",
        "id": "msg_tail",
        "status": "completed",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": "Dependent tail", "annotations": []}
        ],
    }
    later = {
        **pending,
        "id": "fc_later",
        "call_id": "call_later",
        "status": "completed",
    }
    items = [pending, tail, later]
    events[-1]["response"]["output"] = items
    starts = (
        [
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": {**item, "status": "in_progress"},
            }
            for index, item in enumerate(items)
        ]
        if streamed_starts
        else []
    )
    output, _, bodies = await _collect(
        "responses", wire, [events[0], *starts, events[-1]]
    )
    assert len(bodies) == 1
    assert "call_observed" not in output and "call_later" not in output
    assert "Dependent tail" not in output

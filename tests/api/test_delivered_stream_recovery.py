"""Recovery extends one public response after content has been delivered."""

import asyncio
import json
from copy import deepcopy
from itertools import pairwise

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.failure_policy import RetryableProviderProtocolError
from free_claude_code.providers.request_recovery import RequestCorrections
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.support import create_test_app, provider_manager_for_app
from tests.api.test_hidden_stream_retries import delivered, partial_tool
from tests.api.test_tool_call_buffer import _response
from tests.api.test_tool_call_buffer_transports import _tools
from tests.providers.test_anthropic_messages_transport import Wire, _sse
from tests.providers.test_anthropic_provider import native_body, provider
from tests.providers.test_history_transports import _events_for, _harness
from tests.providers.test_native_tool_arguments import tool_events


@pytest.fixture(autouse=True)
def release_provider_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


def text_events(protocol, text, *, complete=True):
    if protocol == "chat":
        template = _events_for(protocol)[0]
        events = [
            {
                **template,
                "choices": [
                    {"index": 0, "delta": {"content": text}, "finish_reason": None}
                ],
            }
        ]
        if complete:
            events.append(
                {
                    **template,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
        return events
    if protocol == "messages":
        events = [
            _events_for(protocol)[0],
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ]
        if complete:
            events.extend(
                [
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 2},
                    },
                    {"type": "message_stop"},
                ]
            )
        return events
    response = deepcopy(_events_for(protocol)[-1]["response"])
    item = {
        "id": "msg_text",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    response["output"] = [item]
    events = [
        {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {**item, "status": "in_progress", "content": []},
        },
        {
            "type": "response.content_part.added",
            "output_index": 0,
            "item_id": "msg_text",
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "item_id": "msg_text",
            "content_index": 0,
            "delta": text,
        },
    ]
    if complete:
        events.extend(
            [
                {
                    "type": "response.output_text.done",
                    "output_index": 0,
                    "item_id": "msg_text",
                    "content_index": 0,
                    "text": text,
                },
                {
                    "type": "response.content_part.done",
                    "output_index": 0,
                    "item_id": "msg_text",
                    "content_index": 0,
                    "part": item["content"][0],
                },
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": response},
            ]
        )
    return [{**event, "sequence_number": index} for index, event in enumerate(events)]


def public_text(events, wire):
    if wire == "messages":
        return "".join(
            event.data.get("delta", {}).get("text", "")
            + event.data.get("content_block", {}).get("text", "")
            for event in events
        )
    return "".join(
        event.data["delta"]
        for event in events
        if event.event == "response.output_text.delta"
    )


def assert_completed(events, wire):
    assert events[-1].event == (
        "message_stop" if wire == "messages" else "response.completed"
    )
    assert (
        sum(
            event.event
            == ("message_start" if wire == "messages" else "response.created")
            for event in events
        )
        == 1
    )
    assert not any(event.event in {"error", "response.failed"} for event in events)
    if wire == "responses":
        numbers = [event.data["sequence_number"] for event in events]
        assert all(right > left for left, right in pairwise(numbers))
        assert (
            len(
                {
                    event.data["response"]["id"]
                    for event in events
                    if "response" in event.data
                }
            )
            == 1
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_text_cutoff_continues_one_response_and_preserves_tools(protocol, wire):
    def reply(bodies):
        return 200, text_events(
            protocol,
            "Hello " if len(bodies) == 1 else "world.",
            complete=len(bodies) > 1,
        )

    async with _harness(protocol, reply, max_attempts=2) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "greet"}], tools=_tools(wire)), wire
        )
    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert public_text(events, wire) == "Hello world."
    assert len(bodies) == 2
    assert bodies[1]["model"] == bodies[0]["model"]
    assert bodies[1]["tools"] == bodies[0]["tools"]
    assert "Hello " in str(bodies[1])
    if wire == "responses":
        output = events[-1].data["response"]["output"]
        assert (
            "".join(
                part.get("text", "")
                for item in output
                for part in item.get("content", [])
            )
            == "Hello world."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_released_tool_finishes_handoff_without_another_request(protocol, wire):
    def reply(_bodies):
        events = tool_events(protocol, '{"path":"kept"}')
        return 200, events[:-2] if protocol == "messages" else events[:-1]

    async with _harness(protocol, reply, max_attempts=1) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    events = parse_sse_text(result)
    assert len(bodies) == 1
    assert_completed(events, wire)
    if wire == "messages":
        assert events[-2].data["delta"]["stop_reason"] == "tool_use"
    else:
        assert events[-1].data["response"]["output"][0]["call_id"] == "call_probe"


def after_text(protocol, tail):
    events = text_events(protocol, "Before. ")
    events = events[:-1] if protocol != "messages" else events[:-2]
    tail = deepcopy(tail if protocol == "chat" else tail[1:])
    for event in tail:
        field = "index" if protocol == "messages" else "output_index"
        if field in event:
            event[field] += 1
    combined = events + tail
    if protocol == "responses":
        for index, event in enumerate(combined):
            event["sequence_number"] = index
    return combined


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_partial_call_after_public_text_is_discarded_before_continuation(
    protocol, wire
):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, after_text(protocol, partial_tool(protocol))
        assert "abandoned" not in str(bodies[-1])
        return 200, tool_events(protocol, '{"path":"winning"}')

    async with _harness(protocol, reply, max_attempts=2) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert len(bodies) == 2
    assert "call_abandoned" not in result
    assert public_text(events, wire) == "Before. "
    if wire == "messages":
        arguments = "".join(
            e.data.get("delta", {}).get("partial_json", "") for e in events
        )
        assert arguments == '{"path":"winning"}'
    else:
        assert "winning" in result
        calls = [
            item
            for item in events[-1].data["response"]["output"]
            if item["type"] == "function_call"
        ]
        assert len(calls) == 1
        assert calls[0]["arguments"] == '{"path":"winning"}'


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_repeated_interruption_uses_latest_prefix_and_one_budget(protocol, wire):
    def reply(bodies):
        index = len(bodies) - 1
        return 200, text_events(
            protocol, ["One ", "two ", "three."][index], complete=index == 2
        )

    async with _harness(protocol, reply, max_attempts=3) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "count"}]), wire
        )
    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert public_text(events, wire) == "One two three."
    assert len(bodies) == 3
    history = bodies[-1].get("messages", bodies[-1].get("input"))
    assert len(history) == 3
    assert "One two " in str(history)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "candidate,expected",
    [
        ("Hello world.", "Hello world."),
        ("Hel", "Hello Hel"),
        (" world.", "Hello world."),
    ],
)
async def test_exact_overlap_filter_preserves_nonmatching_text(
    protocol, wire, candidate, expected
):
    def reply(bodies):
        return 200, text_events(
            protocol,
            "Hello " if len(bodies) == 1 else candidate,
            complete=len(bodies) > 1,
        )

    async with _harness(protocol, reply, max_attempts=2) as (send, _, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "greet"}]), wire
        )
    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert public_text(events, wire) == expected
    if wire == "responses":
        output = events[-1].data["response"]["output"]
        assert (
            "".join(
                part.get("text", "")
                for item in output
                for part in item.get("content", [])
            )
            == expected
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_exact_replay_without_new_output_is_not_success(protocol, wire):
    async with _harness(
        protocol,
        lambda bodies: (200, text_events(protocol, "Same.", complete=len(bodies) > 1)),
        max_attempts=2,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "greet"}]), wire
        )
    events = parse_sse_text(result)
    assert len(bodies) == 2
    assert public_text(events, wire) == "Same."
    assert events[-1].event == ("error" if wire == "messages" else "response.failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize(
    "prefix,candidate,expected",
    [
        ("The answer is", "answer is 42", "The answer is 42"),
        ("abcabc", "abcabcX", "abcabcX"),
        ("abcabd", "abcX", "abcabdabcX"),
        ("very ", "very good", "very good"),
        ("Hello 世界", "世界!", "Hello 世界!"),
    ],
)
async def test_partial_overlap_across_single_character_chunks(
    protocol, wire, prefix, candidate, expected
):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events(protocol, prefix, complete=False)
        events = text_events(protocol, candidate)
        expanded = []
        for event in events:
            if protocol == "chat" and event["choices"][0]["delta"].get("content"):
                for character in candidate:
                    chunk = deepcopy(event)
                    chunk["choices"][0]["delta"]["content"] = character
                    expanded.append(chunk)
            elif event.get("type") == "content_block_delta":
                expanded.extend(
                    {**event, "delta": {"type": "text_delta", "text": character}}
                    for character in candidate
                )
            elif event.get("type") == "response.output_text.delta":
                expanded.extend(
                    {**event, "delta": character} for character in candidate
                )
            else:
                expanded.append(event)
        return 200, expanded

    async with _harness(protocol, reply, max_attempts=2) as (send, _, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "continue"}]), wire
        )
    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert public_text(events, wire) == expected
    if wire == "responses":
        assert (
            "".join(
                part.get("text", "")
                for item in events[-1].data["response"]["output"]
                for part in item.get("content", [])
            )
            == expected
        )


@pytest.mark.asyncio
async def test_native_passthrough_continuation_closes_original_connection():
    bodies = []
    first = Wire([_sse(*text_events("messages", "Hello ", complete=False))])
    second = Wire([_sse(*text_events("messages", "world."))])

    def reply(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 2:
            assert first.closed
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=first if len(bodies) == 1 else second,
        )

    p = provider(reply)
    try:
        result = await delivered(
            p.stream_native_messages(NativeMessagesRequest(native_body(True))),
            "messages",
        )
        events = parse_sse_text(result)
        assert_completed(events, "messages")
        assert public_text(events, "messages") == "Hello world."
        assert len(bodies) == 2
        assert first.closed and second.closed
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("complete", [False, True])
async def test_retention_limit_does_not_truncate_healthy_output(
    monkeypatch, wire, complete
):
    monkeypatch.setattr(
        "free_claude_code.core.delivered_response.RECOVERY_RETENTION_BYTES", 100
    )
    async with _harness(
        "chat", lambda _: (200, text_events("chat", "x" * 1000, complete=complete))
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "write"}]), wire
        )
    events = parse_sse_text(result)
    assert len(bodies) == 1
    assert public_text(events, wire) == "x" * 1000
    if complete:
        assert_completed(events, wire)
    else:
        assert events[-1].event == (
            "error" if wire == "messages" else "response.failed"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("continuation", [False, True])
async def test_stopped_continuation_cannot_request_another_body_correction(
    monkeypatch, wire, continuation
):
    corrections = []

    def correction(*args, **kwargs):
        corrections.append(True)
        return None

    monkeypatch.setattr(RequestCorrections, "next_body", correction)

    def adapter(kind, payload):
        if kind == "response.completed":
            raise RetryableProviderProtocolError("malformed terminal snapshot")
        return payload

    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events(
                "responses",
                "First " if len(bodies) == 1 else "second",
                complete=not continuation or len(bodies) > 1,
            ),
        ),
    ) as (send, bodies, transport):
        transport._event_adapter_factory = lambda: adapter
        result = await delivered(
            send(wire, [{"role": "user", "content": "write"}]), wire
        )
    assert len(bodies) == (2 if continuation else 1)
    assert corrections == []
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_opaque_reasoning_is_not_regenerated_after_delivery(protocol, wire):
    events = _events_for(protocol)
    if protocol == "chat":
        events = events[:1]
    elif protocol == "messages":
        events = events[:-2]
    else:
        events = events[:-1]
    async with _harness(protocol, lambda _: (200, events)) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "think"}]), wire
        )
    assert len(bodies) == 1
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_hidden_hosted_action_after_text_blocks_continuation(wire):
    events = after_text("responses", partial_tool("responses"))
    events.append(
        {
            "type": "response.output_item.added",
            "sequence_number": 40,
            "output_index": 2,
            "item": {
                "type": "web_search_call",
                "id": "ws_held",
                "status": "in_progress",
            },
        }
    )
    async with _harness("responses", lambda _: (200, events)) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "search"}], tools=_tools(wire)),
            wire,
        )
    assert len(bodies) == 1
    assert "ws_held" not in result
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


@pytest.mark.asyncio
async def test_cancellation_during_continuation_closes_both_attempts():
    entered = asyncio.Event()
    calls = []
    first = Wire([_sse(*text_events("messages", "First ", complete=False))])

    class BlockingWire(httpx.AsyncByteStream):
        closes = 0

        async def __aiter__(self):
            yield _sse(*text_events("messages", "second ", complete=False))
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            self.closes += 1

    second = BlockingWire()

    def reply(request):
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=first if len(calls) == 1 else second,
        )

    p = provider(reply)
    response = await _response(
        "messages", p.stream_native_messages(NativeMessagesRequest(native_body(True)))
    )
    try:

        async def read():
            return [chunk async for chunk in response.body_iterator]

        task = asyncio.create_task(read())
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await response.aclose()
        await response.aclose()
        assert len(calls) == 2
        assert first.closed and second.closes == 1
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_completed_signed_thinking_and_call_can_finish_without_regeneration(wire):
    events = _events_for("messages")[:-2]
    calls = tool_events("messages", '{"path":"kept"}')[1:-2]
    for event in calls:
        if "index" in event:
            event["index"] += 1
    async with _harness(
        "messages", lambda _: (200, events + calls), max_attempts=1
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    assert len(bodies) == 1
    assert_completed(parse_sse_text(result), wire)


def test_http_recovery_then_real_harness_tool_result_has_no_private_history():
    sent = []
    wires = []

    def reply(request):
        body = json.loads(request.content)
        sent.append(body)
        if len(sent) == 3:
            return httpx.Response(
                200,
                json={
                    "type": "message",
                    "model": "primary",
                    "content": [{"type": "text", "text": "result received"}],
                },
            )
        events = (
            after_text("messages", partial_tool("messages"))
            if len(sent) == 1
            else tool_events("messages", '{"path":"winning"}')
        )
        wire = Wire([_sse(*events)])
        wires.append(wire)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    app = create_test_app(
        Settings(MODEL="anthropic/primary", ENABLE_WEB_SERVER_TOOLS=False),
        providers={"anthropic": provider(reply)},
    )
    body = {
        **native_body(True),
        "model": "anthropic/primary",
        "tools": _tools("messages"),
    }
    with TestClient(app) as client:
        first = client.post("/v1/messages", json=body)
        assert first.status_code == 200
        events = parse_sse_text(first.text)
        assert_completed(events, "messages")
        calls = [
            event.data["content_block"]
            for event in events
            if event.event == "content_block_start"
            and event.data["content_block"]["type"] == "tool_use"
        ]
        assert len(calls) == 1
        call = {**calls[0], "input": {"path": "winning"}}
        history = [
            *body["messages"],
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Before. "}, call],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": "actual file contents",
                    }
                ],
            },
        ]
        second = client.post(
            "/v1/messages", json={**body, "stream": False, "messages": history}
        )
        assert second.status_code == 200
        assert second.json()["content"][0]["text"] == "result received"
    assert len(sent) == 3
    assert sent[2]["messages"] == history
    assert "previous provider stream" not in str(sent[2])
    assert all(wire.closed for wire in wires)
    assert provider_manager_for_app(app)._current.active_leases == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_unsigned_thinking_can_continue_into_text(wire):
    first = text_events("chat", "", complete=False)
    first[0]["choices"][0]["delta"] = {
        "reasoning_content": "Look at the request first."
    }
    async with _harness(
        "chat",
        lambda bodies: (
            200,
            first if len(bodies) == 1 else text_events("chat", "Answer."),
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "answer"}]), wire
        )
    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert public_text(events, wire) == "Answer."
    assert "Look at the request first." in str(bodies[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("echo", [False, True])
async def test_retention_limit_during_continuation_keeps_final_output_consistent(
    monkeypatch, wire, echo
):
    monkeypatch.setattr(
        "free_claude_code.core.delivered_response.RECOVERY_RETENTION_BYTES", 1200
    )
    async with _harness(
        "responses",
        lambda bodies: (
            200,
            text_events(
                "responses",
                "First "
                if len(bodies) == 1
                else ("First " if echo else "") + "x" * 2000,
                complete=len(bodies) > 1,
            ),
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "write"}]), wire
        )
    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert public_text(events, wire) == "First " + "x" * 2000
    assert len(bodies) == 2
    if wire == "responses":
        final = events[-1].data["response"]
        assert (
            "".join(
                part.get("text", "")
                for item in final["output"]
                for part in item.get("content", [])
            )
            == "First " + "x" * 2000
        )
        streamed_ids = [
            event.data["item"]["id"]
            for event in events
            if event.event == "response.output_item.done"
        ]
        assert [item["id"] for item in final["output"]] == streamed_ids
        assert final["usage"]["output_tokens"] > 0
    else:
        assert events[-2].data["usage"]["output_tokens"] >= 250


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_partial_replay_without_new_output_is_not_success(wire):
    async with _harness(
        "messages",
        lambda bodies: (
            200,
            text_events(
                "messages",
                "The answer is" if len(bodies) == 1 else "answer is",
                complete=len(bodies) > 1,
            ),
        ),
        max_attempts=2,
    ) as (send, _, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "continue"}]), wire
        )
    events = parse_sse_text(result)
    assert public_text(events, wire) == "The answer is"
    assert events[-1].event == ("error" if wire == "messages" else "response.failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_seeded_text_start_participates_in_overlap_filter(wire):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, text_events("messages", "The answer is", complete=False)
        events = text_events("messages", " 42")
        events[1]["content_block"]["text"] = "answer is"
        return 200, events

    async with _harness("messages", reply, max_attempts=2) as (send, _, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "continue"}]), wire
        )
    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert public_text(events, wire) == "The answer is 42"

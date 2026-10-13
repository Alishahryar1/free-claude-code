"""Physical provider retries use evidence from the public delivery boundary."""

import asyncio
import json
from copy import deepcopy
from itertools import pairwise
from unittest.mock import AsyncMock

import httpx
import pytest

from free_claude_code.api.response_streams import bind_response_lifetime
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.config.nim import NimSettings
from free_claude_code.core.anthropic import aggregate_anthropic_sse_to_message
from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.failure_policy import RetryableProviderProtocolError
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider, native_tool_stream
from free_claude_code.providers.openai_chat.transport import _OpenAIChatStreamAssembler
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_response_streams import _serve
from tests.api.test_tool_call_buffer import _response
from tests.api.test_tool_call_buffer_transports import _tools
from tests.application.test_execution import _routed_request
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_anthropic_messages_transport import Wire, _sse
from tests.providers.test_anthropic_provider import native_body, provider
from tests.providers.test_history_transports import _events_for, _harness, _native
from tests.providers.test_native_tool_arguments import tool_events


@pytest.fixture
def committed_holdback(monkeypatch):
    # Commit on metadata, regardless of whether a presenter emits partial args.
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


def partial_tool(protocol):
    events = tool_events(protocol, '{"path":"old"}')
    events = events[:2] if protocol == "chat" else events[:3]
    if protocol == "chat":
        events[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = (
            '{"path":"abandoned'
        )
    elif protocol == "messages":
        events[2]["delta"]["partial_json"] = '{"path":"abandoned'
    else:
        events[2]["delta"] = '{"path":"abandoned'
    return json.loads(
        json.dumps(events)
        .replace("call_probe", "call_abandoned")
        .replace("resp_probe", "resp_abandoned")
    )


def executed(body, wire):
    async def open_candidate(_index, _target, _continuation):
        return body

    return ProviderExecutor(
        AsyncMock(), progress_timeout_seconds=10
    )._stream_candidates(
        resolved=_routed_request().resolved,
        reasoning=ReasoningPolicy.provider_default(),
        wire_api=wire,
        raw_log_label="HIDDEN_RETRY",
        raw_log_payload=dict,
        request_snapshot=dict,
        ingress_count_name="message_count",
        ingress_count=1,
        request_id="hidden-retry",
        open_candidate=open_candidate,
    )


async def delivered(body, wire):
    response = await _response(wire, executed(body, wire))
    messages = await _serve(response)
    return b"".join(message.get("body", b"") for message in messages).decode()


def assert_winning_tool(result, wire):
    events = parse_sse_text(result)
    assert "call_abandoned" not in result
    if wire == "messages":
        assert sum(event.event == "message_start" for event in events) == 1
        assert events[-1].event == "message_stop"
        assert (
            "".join(
                event.data["delta"]["partial_json"]
                for event in events
                if event.data.get("delta", {}).get("type") == "input_json_delta"
            )
            == '{"path":"winning"}'
        )
    else:
        assert sum(event.event == "response.created" for event in events) == 1
        assert events[-1].event == "response.completed"
        call = events[-1].data["response"]["output"][0]
        assert call["call_id"] == "call_probe"
        assert call["arguments"] == '{"path":"winning"}'
        numbers = [event.data["sequence_number"] for event in events]
        assert all(right > left for left, right in pairwise(numbers))
        ids = {
            event.data["response"]["id"] for event in events if "response" in event.data
        }
        assert len(ids) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_hidden_tool_cutoff_retries_identical_generation(
    protocol, wire, committed_holdback
):
    def reply(bodies):
        return 200, (
            partial_tool(protocol)
            if len(bodies) == 1
            else tool_events(protocol, '{"path":"winning"}')
        )

    async with _harness(protocol, reply) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)),
            wire,
        )
    assert len(bodies) == 2
    assert bodies[0] == bodies[1]
    assert_winning_tool(result, wire)


@pytest.mark.asyncio
async def test_raw_native_messages_hidden_cutoff_retries_and_closes_first():
    bodies = []
    first = Wire([_sse(*partial_tool("messages"))])
    second = Wire([_sse(*tool_events("messages", '{"path":"winning"}'))])

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
        assert len(bodies) == 2
        assert bodies[0] == bodies[1]
        assert first.closed and second.closed
        assert_winning_tool(result, "messages")
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["-", "\u2028", "\u2029", "\x85"])
async def test_raw_native_unicode_metadata_keeps_hidden_retry_eligible(separator):
    abandoned = partial_tool("messages")
    abandoned[0]["message"]["extension"] = "a" + separator + "b"
    first = Wire([_sse(*abandoned)])
    winner = tool_events("messages", '{"path":"winning"}')
    winner[0]["message"]["extension"] = "winner" + separator + "b"
    second = Wire([_sse(*winner)])
    bodies = []

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
        assert len(bodies) == 2 and bodies[0] == bodies[1]
        assert first.closed and second.closed
        assert_winning_tool(result, "messages")
        assert parse_sse_text(result)[0].data["message"]["extension"] == (
            "winner" + separator + "b"
        )
    finally:
        await p.cleanup()


def _nim_factory():
    return NvidiaNimProvider(
        make_provider_config("test", "https://provider.invalid/v1"),
        nim_settings=NimSettings(),
        admission=immediate_admission(max_attempts=3),
    )


def _nim_partial_terminal(reason):
    first = deepcopy(_events_for("chat")[0])
    first["choices"][0]["delta"] = {"role": "assistant"}
    first["choices"][0]["finish_reason"] = None
    stopped = deepcopy(first)
    stopped["choices"][0]["delta"] = {"content": "]<]minimax[>[<tool_call>"}
    stopped["choices"][0]["finish_reason"] = reason
    return [first, stopped]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls", "content_filter"])
async def test_normal_chat_stop_survives_nim_normalization_failure(
    wire, reason, committed_holdback
):
    async with _harness(
        "chat",
        lambda bodies: (
            200,
            _nim_partial_terminal(reason)
            if len(bodies) == 1
            else tool_events("chat", '{"path":"winning"}'),
        ),
        chat_provider_factory=_nim_factory,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)),
            wire,
        )
    assert len(bodies) == 1
    assert "]<]minimax" not in result and "winning" not in result
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_nim_unstopped_normalization_failure_can_retry(wire, committed_holdback):
    async with _harness(
        "chat",
        lambda bodies: (
            200,
            _nim_partial_terminal(None)
            if len(bodies) == 1
            else tool_events("chat", '{"path":"winning"}'),
        ),
        chat_provider_factory=_nim_factory,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)),
            wire,
        )
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert_winning_tool(result, wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_replacement_nim_stop_prevents_third_generation(wire, committed_holdback):
    async with _harness(
        "chat",
        lambda bodies: (
            200,
            _nim_partial_terminal(None if len(bodies) == 1 else "length")
            if len(bodies) <= 2
            else tool_events("chat", '{"path":"winning"}'),
        ),
        chat_provider_factory=_nim_factory,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)),
            wire,
        )
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert "winning" not in result
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


@pytest.mark.asyncio
async def test_concurrent_nim_requests_keep_stop_evidence_independent(
    committed_holdback,
):
    attempts = {"stopped": 0, "retry": 0}

    def reply(bodies):
        label = bodies[-1]["messages"][-1]["content"]
        attempts[label] += 1
        if label == "retry" and attempts[label] > 1:
            return 200, tool_events("chat", '{"path":"winning"}')
        return 200, _nim_partial_terminal("stop" if label == "stopped" else None)

    async with _harness("chat", reply, chat_provider_factory=_nim_factory) as (
        send,
        _,
        _,
    ):
        stopped, winning = await asyncio.gather(
            *[
                delivered(
                    send(
                        "messages",
                        [{"role": "user", "content": label}],
                        tools=_tools("messages"),
                    ),
                    "messages",
                )
                for label in ("stopped", "retry")
            ]
        )
    assert attempts == {"stopped": 1, "retry": 2}
    assert parse_sse_text(stopped)[-1].event == "error"
    assert_winning_tool(winning, "messages")


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_raw_nim_stop_survives_later_null_and_empty_choices(
    wire, committed_holdback, monkeypatch
):
    normalize = native_tool_stream._normalized_chunks

    def suppress_terminal(chunk, choice, *args, **kwargs):
        if choice.finish_reason is not None:
            return []
        return normalize(chunk, choice, *args, **kwargs)

    monkeypatch.setattr(native_tool_stream, "_normalized_chunks", suppress_terminal)
    first, terminal = _nim_partial_terminal("stop")
    terminal["choices"][0]["delta"] = {"content": ""}
    usage = {**deepcopy(first), "choices": []}
    async with _harness(
        "chat",
        lambda _: (200, [first, terminal, deepcopy(first), usage]),
        chat_provider_factory=_nim_factory,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)),
            wire,
        )
    assert len(bodies) == 1
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


def overlapping_tools(protocol):
    complete = json.loads(
        json.dumps(tool_events(protocol, '{"path":"oldA"}'))
        .replace("call_probe", "call_oldA")
        .replace("fc_probe", "fc_oldA")
    )
    partial = json.loads(
        json.dumps(partial_tool(protocol))
        .replace("call_abandoned", "call_oldB")
        .replace("fc_probe", "fc_oldB")
    )
    if protocol == "chat":
        for event in partial:
            event["choices"][0]["delta"]["tool_calls"][0]["index"] = 1
        return [complete[0], partial[0], *complete[1:-1], *partial[1:]]
    index_field = "index" if protocol == "messages" else "output_index"
    for event in partial[1:]:
        event[index_field] = 1
    return [
        complete[0],
        complete[1],
        partial[1],
        *partial[2:],
        *complete[2 : (-2 if protocol == "messages" else -1)],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_complete_call_held_by_partial_call_restarts_on_last_attempt(
    protocol, wire, committed_holdback
):
    async with _harness(
        protocol,
        lambda bodies: (
            200,
            overlapping_tools(protocol)
            if len(bodies) == 1
            else tool_events(protocol, '{"path":"winning"}'),
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert "call_oldA" not in result and "call_oldB" not in result
    assert_winning_tool(result, wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_hidden_failures_exhaust_one_budget_without_salvage(
    protocol, wire, committed_holdback
):
    async with _harness(
        protocol,
        lambda _: (200, overlapping_tools(protocol)),
        max_attempts=2,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert "call_oldA" not in result and "call_oldB" not in result
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_replacement_rejection_before_frames_uses_public_empty_envelope(
    protocol, wire, committed_holdback
):
    def reply(bodies):
        return (
            (200, overlapping_tools(protocol))
            if len(bodies) == 1
            else (
                400,
                {"code": "invalid_request_error", "message": "replacement rejected"},
            )
        )

    async with _harness(protocol, reply) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    assert "call_oldA" not in result and "call_oldB" not in result
    events = parse_sse_text(result)
    assert events[-1].event == ("error" if wire == "messages" else "response.failed")
    if wire == "responses":
        assert events[-1].data["response"]["id"] == events[0].data["response"]["id"]
        assert events[-1].data["response"]["output"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("retryable", [True, False])
async def test_typed_upstream_failure_qualifies_before_wire_error(
    protocol, wire, retryable, committed_holdback
):
    failure = (
        {
            "type": "error",
            "error": {
                "type": "overloaded_error" if retryable else "authentication_error",
                "message": "temporary overload" if retryable else "unauthorized",
            },
        }
        if protocol == "messages"
        else {
            "type": "error",
            "error": {
                "code": "rate_limit_exceeded" if retryable else "invalid_request_error",
                "message": "rate limit exceeded" if retryable else "invalid request",
            },
            "code": "rate_limit_exceeded" if retryable else "invalid_request_error",
            "message": "rate limit exceeded" if retryable else "invalid request",
            "sequence_number": 99,
        }
    )
    async with _harness(
        protocol,
        lambda bodies: (
            200,
            [*partial_tool(protocol), failure]
            if len(bodies) == 1
            else tool_events(protocol, '{"path":"winning"}'),
        ),
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    assert len(bodies) == (2 if retryable else 1)
    if retryable:
        assert bodies[0] == bodies[1]
        assert_winning_tool(result, wire)
    else:
        assert "call_abandoned" not in result
        assert parse_sse_text(result)[-1].event == (
            "error" if wire == "messages" else "response.failed"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("terminal", ["response.completed", "response.incomplete"])
async def test_normal_responses_stop_prevents_retry_when_adapter_raises(
    wire, terminal, committed_holdback
):
    events = [
        *partial_tool("responses"),
        {
            "type": terminal,
            "response": {"status": terminal.split(".")[1], "output": []},
            "sequence_number": 4,
        },
    ]

    def adapter(kind, payload):
        if kind == terminal:
            raise RetryableProviderProtocolError("terminal adapter rejected snapshot")
        return payload

    async with _harness("responses", lambda _: (200, events)) as (
        send,
        bodies,
        transport,
    ):
        transport._event_adapter_factory = lambda: adapter
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    assert len(bodies) == 1
    assert "call_abandoned" not in result
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("stop", ["end_turn", "max_tokens", "refusal", "pause_turn"])
async def test_messages_normal_stop_with_open_hidden_block_cannot_retry(
    wire, stop, committed_holdback
):
    events = [
        *partial_tool("messages"),
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop},
            "usage": {"output_tokens": 10},
        },
    ]
    async with _harness("messages", lambda _: (200, events)) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    assert len(bodies) == 1 and "call_abandoned" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_thinking_hidden_inside_tool_group_can_be_discarded(
    wire, committed_holdback
):
    events = overlapping_tools("messages")
    events += [
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "thinking_delta", "thinking": "hidden old reasoning"},
        },
    ]
    async with _harness(
        "messages",
        lambda bodies: (
            200,
            events
            if len(bodies) == 1
            else tool_events("messages", '{"path":"winning"}'),
        ),
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    assert len(bodies) == 2 and "hidden old reasoning" not in result
    assert_winning_tool(result, wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("reason", ["stop", "length", "tool_calls", "content_filter"])
async def test_normal_chat_finish_survives_later_parser_failure(
    wire, reason, committed_holdback, monkeypatch
):
    feed = _OpenAIChatStreamAssembler.feed

    def rejected_terminal(self, chunk):
        yield from feed(self, chunk)
        if chunk.choices and chunk.choices[0].finish_reason is not None:
            raise RetryableProviderProtocolError("terminal parsing failed")

    monkeypatch.setattr(_OpenAIChatStreamAssembler, "feed", rejected_terminal)
    terminal = tool_events("chat", "")[-1]
    terminal["choices"][0]["finish_reason"] = reason
    async with _harness("chat", lambda _: (200, [*partial_tool("chat"), terminal])) as (
        send,
        bodies,
        _,
    ):
        result = await delivered(
            send(wire, [{"role": "user", "content": "read"}], tools=_tools(wire)), wire
        )
    assert len(bodies) == 1 and "call_abandoned" not in result
    assert parse_sse_text(result)[-1].event == (
        "error" if wire == "messages" else "response.failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["text", "thinking", "server_tool_use", "future_block"]
)
async def test_raw_native_published_activity_prevents_clean_retry(kind):
    events = partial_tool("messages")
    events.insert(
        1,
        {
            "type": "content_block_start",
            "index": 7,
            "content_block": {
                "type": kind,
                "text": "",
                "thinking": "",
                "id": "hosted",
                "name": "search",
                "input": {},
            },
        },
    )
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*events)]),
        )

    p = provider(reply)
    try:
        result = await delivered(
            p.stream_native_messages(NativeMessagesRequest(native_body(True))),
            "messages",
        )
        assert len(bodies) == 1 and f'"type": "{kind}"' in result
        assert "call_abandoned" not in result
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_details", ["missing", "null", "zero", "partial"])
async def test_raw_native_retry_uses_winning_initial_usage_and_final_overrides(
    cache_details,
):
    first = partial_tool("messages")
    first[0]["message"]["usage"] = {
        "input_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 70,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 70,
            "ephemeral_1h_input_tokens": 0,
        },
        "abandoned_extension": {"old": 7},
        "output_tokens": 0,
    }
    winner = tool_events("messages", '{"path":"winning"}')
    winner[0]["message"]["id"] = "msg_winning"
    winner[0]["message"]["usage"] = {
        "input_tokens": 0,
        "cache_read_input_tokens": 70,
        "cache_creation_input_tokens": 0,
        "output_tokens": 1,
        "winning_extension": {"note": "a\u2028b", "counts": [0, None]},
    }
    if cache_details != "missing":
        winner[0]["message"]["usage"]["cache_creation"] = (
            None
            if cache_details == "null"
            else {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}
            if cache_details == "zero"
            else {"ephemeral_1h_input_tokens": 0}
        )
    winner[-2]["usage"] = {"input_tokens": 25, "output_tokens": 10}
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*(first if len(bodies) < 3 else winner))]),
        )

    p = provider(reply, max_attempts=3)
    try:
        result = await delivered(
            p.stream_native_messages(NativeMessagesRequest(native_body(True))),
            "messages",
        )
        assert len(bodies) == 3 and bodies[0] == bodies[1] == bodies[2]
        events = parse_sse_text(result)
        assert events[0].data["message"]["id"] == "msg_winning"
        assert events[0].data["message"]["usage"] == winner[0]["message"]["usage"]
        assert events[-2].data["usage"] == winner[-2]["usage"]

        async def emitted():
            yield result

        message, error, complete = await aggregate_anthropic_sse_to_message(emitted())
        assert complete and error is None
        assert message["usage"] == {
            **winner[0]["message"]["usage"],
            **winner[-2]["usage"],
        }
        assert "abandoned_extension" not in result
    finally:
        await p.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("before_first_read", [False, True])
async def test_raw_native_cancellation_closes_once_without_replacement(
    before_first_read,
):
    reading = asyncio.Event()
    calls = []
    releases = []

    class BlockingWire(httpx.AsyncByteStream):
        closes = 0

        async def __aiter__(self):
            events = partial_tool("messages")
            yield _sse(events[0])
            yield _sse(*events[1:])
            reading.set()
            await asyncio.Event().wait()

        async def aclose(self):
            self.closes += 1

    wire = BlockingWire()

    def reply(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    async def release():
        releases.append(True)

    p = provider(reply)
    try:
        response = await _response(
            "messages",
            p.stream_native_messages(NativeMessagesRequest(native_body(True))),
        )
        await bind_response_lifetime(response, release)
        if not before_first_read:
            iterator = aiter(response.body_iterator)

            async def read():
                return [chunk async for chunk in iterator]

            task = asyncio.create_task(read())
            await asyncio.wait_for(reading.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await response.aclose()
        await response.aclose()
        assert len(calls) == 1 and wire.closes == 1 and releases == [True]
    finally:
        await p.cleanup()


@pytest.mark.asyncio
async def test_hidden_retry_keeps_correction_and_uses_its_remaining_budget(
    committed_holdback,
):
    def reply(bodies):
        if len(bodies) == 1:
            return 400, {
                "code": "invalid_value",
                "param": "input[0].id",
                "message": "Invalid 'input[0].id': 'rs_a:rs_b'. Expected an ID that contains letters, numbers, underscores, or dashes.",
            }
        return 200, partial_tool("responses") if len(bodies) == 2 else tool_events(
            "responses", '{"path":"winning"}'
        )

    async with _harness("responses", reply, max_attempts=3) as (send, bodies, _):
        result = await delivered(
            send(
                "responses",
                [_native("responses"), {"role": "user", "content": "read"}],
                tools=_tools("responses"),
            ),
            "responses",
        )
    assert len(bodies) == 3
    assert bodies[0] != bodies[1] and bodies[1] == bodies[2]
    assert ":" not in bodies[1]["input"][0]["id"]
    assert_winning_tool(result, "responses")

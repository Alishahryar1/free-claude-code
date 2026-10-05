"""A continued response retains corrected requests and counts actual output."""

import json
from copy import deepcopy

import httpx
import pytest

from free_claude_code.core.anthropic.passthrough import NativeMessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.stream_recovery import RecoveryHoldbackBuffer
from tests.api.test_delivered_stream_recovery import (
    assert_completed,
    public_text,
    text_events,
)
from tests.api.test_hidden_stream_retries import delivered
from tests.providers.test_anthropic_messages_transport import Wire, _sse
from tests.providers.test_anthropic_provider import native_body, provider
from tests.providers.test_history_transports import _events_for, _harness, _saved_reply


@pytest.fixture(autouse=True)
def release_provider_events(monkeypatch):
    monkeypatch.setattr(
        "free_claude_code.providers.stream_recovery.RecoveryHoldbackBuffer",
        lambda: RecoveryHoldbackBuffer(max_bytes=1),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_empty_continuation_fails_without_compatibility_padding(protocol, wire):
    def reply(bodies):
        return 200, text_events(
            protocol,
            "Hello" if len(bodies) == 1 else "",
            complete=len(bodies) > 1,
        )

    async with _harness(protocol, reply, max_attempts=3) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "greet"}]), wire
        )

    events = parse_sse_text(result)
    assert len(bodies) == 2
    assert public_text(events, wire) == "Hello"
    assert events[-1].event == ("error" if wire == "messages" else "response.failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_empty_initial_generation_preserves_compatibility_padding(protocol, wire):
    async with _harness(protocol, lambda _: (200, text_events(protocol, ""))) as (
        send,
        bodies,
        _,
    ):
        result = await delivered(
            send(wire, [{"role": "user", "content": "greet"}]), wire
        )

    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert len(bodies) == 1
    padded = protocol == "chat" or (protocol == "responses" and wire == "messages")
    assert public_text(events, wire) == (" " if padded else "")


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_real_provider_whitespace_is_continuation_output(protocol, wire):
    async with _harness(
        protocol,
        lambda bodies: (
            200,
            text_events(
                protocol,
                "Hello" if len(bodies) == 1 else " ",
                complete=len(bodies) > 1,
            ),
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "greet"}]), wire
        )

    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert public_text(events, wire) == "Hello "
    assert len(bodies) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_readable_reasoning_continuation_succeeds_without_padding(wire):
    reasoning = text_events("chat", "")
    reasoning[0]["choices"][0]["delta"] = {
        "reasoning_content": "Consider the greeting."
    }
    async with _harness(
        "chat",
        lambda bodies: (
            200,
            text_events("chat", "Hello", complete=False)
            if len(bodies) == 1
            else reasoning,
        ),
        max_attempts=2,
    ) as (send, bodies, _):
        result = await delivered(
            send(wire, [{"role": "user", "content": "greet"}]), wire
        )

    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert public_text(events, wire) == "Hello"
    assert "Consider the greeting." in result
    assert len(bodies) == 2


def rejected_history_error(protocol, body):
    if protocol == "responses":
        index = next(
            i for i, item in enumerate(body["input"]) if item.get("encrypted_content")
        )
        param = f"input[{index}].encrypted_content"
    else:
        field = "content" if protocol == "messages" else "reasoning_details"
        key = "signature" if protocol == "messages" else "data"
        index, block_index = next(
            (i, j)
            for i, message in enumerate(body["messages"])
            if isinstance(message.get(field), list)
            for j, block in enumerate(message[field])
            if block.get(key) == "opaque-original"
        )
        param = f"messages[{index}].{field}[{block_index}].{key}"
    return {
        "type": "invalid_request_error",
        "code": "invalid_signature"
        if protocol == "messages"
        else "invalid_encrypted_content",
        "param": param,
        "message": "Historical reasoning record expired.",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["chat", "messages", "responses"])
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_history_correction_survives_another_interruption(protocol, wire):
    def reply(bodies):
        if len(bodies) == 1:
            return 200, _events_for(protocol)
        if len(bodies) > 2 and "opaque-original" in json.dumps(bodies[-1]):
            return 400, rejected_history_error(protocol, bodies[-1])
        return 200, text_events(
            protocol,
            "Hello "
            if len(bodies) == 2
            else "world "
            if len(bodies) == 4
            else "again.",
            complete=len(bodies) == 5,
        )

    async with _harness(protocol, reply, max_attempts=4) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "think"}]), wire
        )
        history = [*saved, {"role": "user", "content": "greet"}]
        original = deepcopy(history)
        result = await delivered(send(wire, history), wire)

    assert len(bodies) == 5
    assert history == original
    assert "opaque-original" in json.dumps(bodies[2])
    assert all("opaque-original" not in json.dumps(body) for body in bodies[3:])
    history_field = "input" if protocol == "responses" else "messages"
    assert bodies[4][history_field][:-2] == bodies[3][history_field][:-2]
    assert "Hello world " in str(bodies[4][history_field][-2])
    events = parse_sse_text(result)
    assert_completed(events, wire)
    assert public_text(events, wire) == "Hello world again."


@pytest.mark.asyncio
async def test_removed_history_row_stays_removed_after_another_interruption():
    history = [
        {
            "type": "reasoning",
            "id": "rs_old",
            "encrypted_content": "opaque-original",
            "summary": [],
        },
        {"role": "user", "content": "greet"},
    ]

    def reply(bodies):
        if len(bodies) > 1 and "opaque-original" in json.dumps(bodies[-1]):
            return 400, rejected_history_error("responses", bodies[-1])
        return 200, text_events(
            "responses",
            "Hello "
            if len(bodies) == 1
            else "world "
            if len(bodies) == 3
            else "again.",
            complete=len(bodies) == 4,
        )

    async with _harness("responses", reply, max_attempts=4) as (send, bodies, _):
        result = await delivered(send("responses", history), "responses")

    assert len(bodies) == 4
    assert len(bodies[2]["input"]) == len(bodies[1]["input"]) - 1
    assert bodies[3]["input"][:-2] == bodies[2]["input"][:-2]
    assert_completed(parse_sse_text(result), "responses")
    assert public_text(parse_sse_text(result), "responses") == "Hello world again."


@pytest.mark.asyncio
async def test_native_messages_repeated_continuation_replaces_only_private_tail():
    bodies = []
    wires = [
        Wire([_sse(*text_events("messages", text, complete=index == 2))])
        for index, text in enumerate(["Hello ", "world ", "again."])
    ]

    def reply(request):
        assert all(wire.closed for wire in wires[: len(bodies)])
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=wires[len(bodies) - 1],
        )

    p = provider(reply, max_attempts=3)
    initial = native_body(True)
    try:
        result = await delivered(
            p.stream_native_messages(NativeMessagesRequest(initial)), "messages"
        )
        assert_completed(parse_sse_text(result), "messages")
        assert public_text(parse_sse_text(result), "messages") == "Hello world again."
        assert len(bodies) == 3
        assert bodies[-1]["messages"][:-2] == initial["messages"]
        assert bodies[-1]["messages"][-2] == {
            "role": "assistant",
            "content": "Hello world ",
        }
        assert all(wire.closed for wire in wires)
    finally:
        await p.cleanup()

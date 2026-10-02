"""Reasoning stays complete and attached while answers stream."""

import json

import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.chat_observations import ChatStreamUsage
from free_claude_code.core.history_replay import (
    AssociatedReplayRecord,
    ReplayOrigin,
    decode_replay,
)
from free_claude_code.providers.history_replay import normalize_messages_history
from free_claude_code.providers.openai_chat.reasoning_details import (
    StructuredReasoningStream,
)
from tests.protocol_stream_support import ChatSourceHarness
from tests.providers.test_history_transports import (
    _chat_reasoning_events,
    _harness,
    _saved_reply,
)
from tests.stream_helpers import serialize_events


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_distinct_native_reasoning_survives_structured_summary(wire):
    events = _chat_reasoning_events(
        [
            {
                "reasoning_details": [
                    {
                        "type": "reasoning.summary",
                        "summary": "Short summary.",
                        "index": 0,
                    }
                ]
            },
            {"reasoning_content": "Distinct native thought."},
        ]
    )
    async with _harness("chat", lambda _: (200, events)) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
        await _saved_reply(
            send(wire, [*saved, {"role": "user", "content": "next"}]), wire
        )
    body = json.dumps(bodies[-1])
    assert body.count("Distinct native thought.") == 1
    assert body.count("Short summary.") == 1


@pytest.mark.asyncio
async def test_encrypted_only_reasoning_stays_with_its_answer():
    details = [{"type": "reasoning.encrypted", "data": "opaque", "index": 0}]
    events = _chat_reasoning_events([{"reasoning_details": details}])
    async with _harness("chat", lambda _: (200, events)) as (send, bodies, _):
        saved = await _saved_reply(
            send("responses", [{"role": "user", "content": "hello"}]), "responses"
        )
        assert [item["type"] for item in saved] == ["reasoning", "message"]
        await _saved_reply(
            send("responses", [*saved, {"role": "user", "content": "next"}]),
            "responses",
        )
    assistants = [msg for msg in bodies[-1]["messages"] if msg["role"] == "assistant"]
    assert len(assistants) == 1
    assert assistants[0]["content"] == "17"
    assert assistants[0]["reasoning_details"] == details


@pytest.mark.asyncio
async def test_late_metadata_does_not_overlap_messages_content_blocks():
    events = _chat_reasoning_events(
        [
            {
                "reasoning_content": "Plan.",
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "first", "index": 0}
                ],
            },
            {"content": "Answer."},
            {
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "second", "index": 0}
                ]
            },
        ]
    )
    async with _harness("chat", lambda _: (200, events)) as (send, bodies, _):
        stream = [
            chunk
            async for chunk in send("messages", [{"role": "user", "content": "hello"}])
        ]
        active = None
        for event in parse_sse_text("".join(stream)):
            if event.event == "content_block_start":
                assert active is None
                active = event.data["index"]
            elif event.event == "content_block_delta":
                assert event.data["index"] == active
            elif event.event == "content_block_stop":
                assert event.data["index"] == active
                active = None
        assert active is None
        saved = await _saved_reply(
            send("messages", [{"role": "user", "content": "hello"}]), "messages"
        )
        await _saved_reply(
            send("messages", [*saved, {"role": "user", "content": "next"}]), "messages"
        )
    assert bodies[-1]["messages"][0]["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "firstsecond", "index": 0}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["chat", "messages", "responses"])
@pytest.mark.parametrize("readable", [False, True])
async def test_associated_messages_replay_into_every_transport(destination, readable):
    first = {
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "first", "index": 0}
        ]
    }
    if readable:
        first["reasoning_content"] = "Plan."
    events = _chat_reasoning_events(
        [
            first,
            {"content": "Answer."},
            {
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "second", "index": 0}
                ]
            },
        ]
    )
    async with _harness("chat", lambda _: (200, events)) as (send, _, _):
        saved = await _saved_reply(
            send("messages", [{"role": "user", "content": "hello"}]), "messages"
        )
    history = [*saved, {"role": "user", "content": "next"}]
    before = json.dumps(history)
    async with _harness(destination) as (send, bodies, _):
        await _saved_reply(send("messages", history), "messages")
    assert json.dumps(history) == before
    body = json.dumps(bodies[-1])
    assert "fcc:history" not in body
    assert body.count("Plan.") == int(readable)
    if destination == "chat":
        assert bodies[-1]["messages"][0]["reasoning_details"] == [
            {"type": "reasoning.encrypted", "data": "firstsecond", "index": 0}
        ]
    else:
        assert "firstsecond" not in body


def _messages_writer():
    output = ChatSourceHarness(model="public", input_tokens=1)
    output.replay_origin = ReplayOrigin("test", "chat", "endpoint", "account", "model")
    reasoning = StructuredReasoningStream()
    output.reasoning_replay = reasoning
    return output, reasoning


def _readable(reasoning, output, text="Plan.", data="first"):
    return list(
        reasoning.events(
            {
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": data, "index": 0}
                ]
            },
            output,
            native_reasoning=text,
        )
    )


def _assert_serial(events):
    active = None
    for event in events:
        if event.event == "content_block_start":
            assert active is None
            active = event.data["index"]
        elif event.event in {"content_block_delta", "content_block_stop"}:
            assert event.data["index"] == active
            if event.event == "content_block_stop":
                active = None
    assert active is None


@pytest.mark.parametrize("terminal", ["normal", "length", "tool"])
def test_immediate_text_and_late_replay_survive_successful_completion(terminal):
    output, reasoning = _messages_writer()
    frames = output.start_events() + _readable(reasoning, output)
    frames += output.ensure_text_block()
    frames.append(output.emit_text_delta("Answer."))
    # Text has already been emitted, while upstream metadata and termination are pending.
    early_frames = output.project(frames)
    frames = []
    early = parse_sse_text(serialize_events(early_frames))
    assert any(e.data.get("delta", {}).get("text") == "Answer." for e in early)
    assert not any(e.event == "message_stop" for e in early)
    if terminal == "tool":
        frames += output.close_content_blocks()
        frames.append(output.start_tool_block(0, "call_a", "lookup"))
        frames.extend(output.emit_tool_delta(0, '{"query":"x"}'))
    frames += list(
        reasoning.events(
            {
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "second", "index": 0}
                ]
            },
            output,
            native_reasoning=None,
        )
    )
    frames += output.finish_success(
        stop_reason="max_tokens" if terminal == "length" else "end_turn",
        usage=ChatStreamUsage(input_tokens=1, output_tokens=4),
    )
    assert (
        output.finish_success(
            stop_reason="end_turn",
            usage=ChatStreamUsage(input_tokens=1, output_tokens=4),
        )
        == []
    )
    events = parse_sse_text(serialize_events([*early_frames, *output.project(frames)]))
    _assert_serial(events)
    finals = [
        decode_replay(e.data["content_block"]["data"])
        for e in events
        if e.event == "content_block_start"
        and e.data["content_block"]["type"] == "redacted_thinking"
    ]
    assert len(finals) == 1
    assert isinstance(finals[0], AssociatedReplayRecord)
    assert finals[0].part == "final"
    assert finals[0].native == {
        "reasoning_content": "Plan.",
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "firstsecond", "index": 0}
        ],
    }


def test_request_copy_preserves_routing_and_does_not_join_across_messages():
    output, reasoning = _messages_writer()
    frames = output.start_events() + _readable(reasoning, output)
    frames += output.ensure_text_block()
    frames.append(output.emit_text_delta("answer"))
    frames += output.finish_success(
        stop_reason="end_turn", usage=ChatStreamUsage(input_tokens=1, output_tokens=1)
    )
    events = parse_sse_text(serialize_events(output.project(frames)))
    anchor = next(
        e.data["delta"]["signature"]
        for e in events
        if e.data.get("delta", {}).get("type") == "signature_delta"
    )
    final = next(
        e.data["content_block"]["data"]
        for e in events
        if e.event == "content_block_start"
        and e.data["content_block"]["type"] == "redacted_thinking"
    )
    request = MessagesRequest(
        model="route",
        original_model="original",
        resolved_provider_model="actual",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Plan.", "signature": anchor}
                ],
            },
            {"role": "user", "content": "next"},
            {
                "role": "assistant",
                "content": [{"type": "redacted_thinking", "data": final}],
            },
        ],
    )
    before = request.model_dump_json()
    normalized = normalize_messages_history(request)
    assert normalized.original_model == "original"
    assert normalized.resolved_provider_model == "actual"
    assert len(normalized.messages) == 3
    assert request.model_dump_json() == before
    assert "fcc:history:v2:" not in normalized.model_dump_json()
    assert normalize_messages_history(normalized) == normalized


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("large", [False, True])
async def test_incomplete_structured_replay_prevents_recovery_at_any_output_size(
    wire, large
):
    events = _chat_reasoning_events(
        [
            {
                "reasoning_content": "Original thought.",
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "original", "index": 0}
                ],
            },
            {"content": "x" * (66000 if large else 5)},
        ]
    )[:-1]
    async with _harness("chat", lambda _: (200, events)) as (send, bodies, _):
        output = "".join(
            [
                frame
                async for frame in send(wire, [{"role": "user", "content": "hello"}])
            ]
        )
    parsed = parse_sse_text(output)
    assert len(bodies) == 1
    assert parsed[-1].event == ("error" if wire == "messages" else "response.failed")
    assert not any(
        event.event in {"message_stop", "response.completed"} for event in parsed
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_first_metadata_after_text_does_not_invent_earlier_reasoning(wire):
    events = _chat_reasoning_events(
        [
            {"content": "Earlier answer."},
            {
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "late", "index": 0}
                ]
            },
        ]
    )
    async with _harness("chat", lambda _: (200, events)) as (send, _, _):
        if wire == "messages":
            frames = [
                part
                async for part in send(wire, [{"role": "user", "content": "hello"}])
            ]
            _assert_serial(parse_sse_text(serialize_events(frames)))
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
    assert (
        saved[0]["type"] == "message"
        if wire == "responses"
        else saved[0]["content"][0]["type"] == "text"
    )


@pytest.mark.asyncio
async def test_incomplete_replay_is_not_given_a_synthetic_success_terminal():
    events = _chat_reasoning_events(
        [
            {
                "reasoning_content": "Plan.",
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "opaque", "index": 0}
                ],
            },
            {"content": "x" * 66000},
        ]
    )[:-1]
    async with _harness("chat", lambda _: (200, events)) as (send, bodies, _):
        frames = [
            frame
            async for frame in send("responses", [{"role": "user", "content": "hello"}])
        ]
    parsed = parse_sse_text("".join(frames))
    assert len(bodies) == 1
    assert parsed[-1].event == "response.failed"
    assert not any(event.event == "response.completed" for event in parsed)

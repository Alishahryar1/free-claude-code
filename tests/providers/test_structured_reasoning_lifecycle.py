"""Reasoning stays complete and attached while answers stream."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.history_replay import (
    AssociatedReplayRecord,
    ReplayOrigin,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.history_replay import (
    normalize_messages_history,
    replay_origin,
)
from free_claude_code.providers.openai_chat.reasoning_details import (
    StructuredReasoningStream,
)
from free_claude_code.providers.openai_chat.stream_output import (
    AnthropicChatStreamOutput,
    ChatStreamUsage,
)
from free_claude_code.providers.openai_chat.transport import _OpenAIChatStreamRunner
from tests.providers.test_history_transports import (
    _carrier,
    _chat_reasoning_events,
    _harness,
    _saved_reply,
)
from tests.providers.test_streaming_errors import _recovery_output


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
    assert assistants[0]["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "opaque"}
    ]


@pytest.mark.asyncio
async def test_late_metadata_uses_open_indices_and_closes_final_text_last():
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
        parsed = parse_sse_text("".join(stream))
        _assert_indexed(parsed)
        text_index = max(
            event.data["index"]
            for event in parsed
            if event.event == "content_block_start"
            and event.data["content_block"]["type"] == "text"
        )
        assert [
            event.data["index"]
            for event in parsed
            if event.event == "content_block_stop"
        ][-1] == text_index
        saved = await _saved_reply(
            send("messages", [{"role": "user", "content": "hello"}]), "messages"
        )
        await _saved_reply(
            send("messages", [*saved, {"role": "user", "content": "next"}]), "messages"
        )
    assert bodies[-1]["messages"][0]["reasoning_details"] == [
        {"type": "reasoning.encrypted", "data": "firstsecond"}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["chat", "messages", "responses"])
@pytest.mark.parametrize("readable", [False, True])
async def test_legacy_associated_messages_replay_into_every_transport(
    destination, readable
):
    first: JsonObject = {
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "first", "index": 0}
        ]
    }
    if readable:
        first["reasoning_content"] = "Plan."
    async with _harness("chat") as (_, _, provider):
        origin = replay_origin(
            "OPENROUTER", "chat", "actual-returned", client=provider._client
        )
    final: JsonObject = {
        **first,
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "firstsecond", "index": 0}
        ],
    }
    anchor = encode_replay(AssociatedReplayRecord(origin, first, "group", "anchor"))
    completed = encode_replay(AssociatedReplayRecord(origin, final, "group", "final"))
    saved = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Plan." if readable else "",
                    "signature": anchor,
                },
                {"type": "text", "text": "Answer."},
                {"type": "redacted_thinking", "data": completed},
            ],
        }
    ]
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
    output = AnthropicChatStreamOutput(
        message_id="msg_test", model="public", input_tokens=1
    )
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


def _assert_indexed(events):
    active = set()
    seen = set()
    for event in events:
        if event.event == "content_block_start":
            assert event.data["index"] not in seen
            seen.add(event.data["index"])
            active.add(event.data["index"])
        elif event.event in {"content_block_delta", "content_block_stop"}:
            assert event.data["index"] in active
            if event.event == "content_block_stop":
                active.remove(event.data["index"])
    assert not active


@pytest.mark.parametrize(
    "terminal", ["normal", "length", "failure", "tool", "continuation"]
)
def test_immediate_text_and_late_replay_survive_every_exit(terminal):
    output, reasoning = _messages_writer()
    frames = output.start_events() + _readable(reasoning, output)
    frames += output.ensure_text_block()
    frames.append(output.emit_text_delta("Answer."))
    # Text has already been emitted, while upstream metadata and termination are pending.
    early = parse_sse_text("".join(frames))
    assert any(e.data.get("delta", {}).get("text") == "Answer." for e in early)
    assert not any(e.event == "message_stop" for e in early)
    if terminal == "tool":
        frames += output.close_content_blocks()
        frames.append(output.start_tool_block(0, "call_a", "lookup"))
        frames.append(output.emit_tool_delta(0, '{"query":"x"}'))
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
    if terminal == "continuation":
        frames += output.flush_reasoning_replay()
        frames += output.ensure_reasoning_block()
        frames.append(output.emit_reasoning_delta("Recovery thought."))
        frames += output.ensure_text_block()
        frames.append(output.emit_text_delta(" Continued."))
    if terminal == "failure":
        frames += output.close_unclosed_blocks()
        assert output.close_unclosed_blocks() == []
    else:
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
    events = parse_sse_text("".join(frames))
    _assert_indexed(events)
    signatures = [
        e.data["delta"]["signature"]
        for e in events
        if e.data.get("delta", {}).get("type") == "signature_delta"
    ]
    assert signatures == ([] if terminal == "failure" else ["firstsecond"])
    assert "fcc:history" not in "".join(frames)


def test_request_copy_preserves_routing_and_does_not_join_across_messages():
    origin = ReplayOrigin("test", "chat", "endpoint", "account", "model")
    native: JsonObject = {
        "reasoning_content": "Plan.",
        "reasoning_details": [
            {"type": "reasoning.encrypted", "data": "first", "index": 0}
        ],
    }
    anchor = encode_replay(AssociatedReplayRecord(origin, native, "group", "anchor"))
    final = encode_replay(AssociatedReplayRecord(origin, native, "group", "final"))
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
@pytest.mark.parametrize("committed", [False, True])
async def test_structured_replay_respects_retry_and_continuation_boundary(
    wire, committed
):
    prefix = "x" * 66000 if committed else "partial"
    first = _chat_reasoning_events(
        [
            {
                "reasoning_content": "Original thought.",
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "original", "index": 0}
                ],
            },
            {"content": prefix},
        ]
    )[:-1]
    second = _chat_reasoning_events(
        [
            {
                "reasoning_content": "Recovery thought.",
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "replacement", "index": 0}
                ],
            },
            {"content": "continued"},
        ]
    )
    async with _harness(
        "chat", lambda bodies: (200, first if len(bodies) == 1 else second)
    ) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
    assert len(bodies) == 2
    if committed:
        assert "Original thought." in json.dumps(saved)
        assert 'original"' not in json.dumps(saved)
    else:
        assert "Recovery thought." in json.dumps(saved)
        assert _carrier(saved, wire) == "replacement"
    assert "continued" in json.dumps(saved)
    if not committed:
        assert "Original thought." not in json.dumps(saved)


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
            _assert_indexed(parse_sse_text("".join(frames)))
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
    assert (
        saved[0]["type"] == "message"
        if wire == "responses"
        else saved[0]["content"][0]["type"] == "text"
    )


@pytest.mark.asyncio
async def test_finishing_replay_does_not_make_empty_continuation_successful():
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
    async with _harness("chat", lambda _: (200, events)) as (send, _, _):
        with patch.object(
            _OpenAIChatStreamRunner,
            "_collect_recovery_output",
            new_callable=AsyncMock,
            return_value=_recovery_output(),
        ):
            frames = [
                frame
                async for frame in send(
                    "responses", [{"role": "user", "content": "hello"}]
                )
            ]
    parsed = parse_sse_text("".join(frames))
    assert parsed[-1].event == "response.failed"
    assert not any(event.event == "response.completed" for event in parsed)

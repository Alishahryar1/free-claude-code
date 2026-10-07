"""Native reasoning survives conversion without trailing FCC replay records."""

import asyncio
import json
from copy import deepcopy

import httpx2
import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses.chat_request import (
    build_responses_chat_request,
)
from free_claude_code.core.openai_responses.models import OpenAIResponsesRequest
from free_claude_code.core.openai_responses.provider_input import (
    build_responses_provider_request,
)
from free_claude_code.core.openai_responses.provider_stream import (
    ResponsesProviderStream,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.gemini import GeminiProvider
from free_claude_code.providers.openai_chat.reasoning_details import (
    StructuredReasoningStream,
)
from free_claude_code.providers.openai_chat.stream_output import (
    AnthropicChatStreamOutput,
)
from tests.api.test_hidden_stream_retries import delivered
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_history_transports import (
    _events_for,
    _harness,
    _native,
    _saved_reply,
)


def _chunks(deltas):
    return [
        {
            "id": "chat_native_reasoning",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "actual-returned",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        for delta, finish in deltas
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("thinking", ["", "Plan."])
async def test_plain_reasoning_emits_no_artificial_replay_state(wire, thinking):
    upstream = _chunks(
        [
            ({"reasoning_content": thinking}, None),
            ({"content": "Answer."}, "stop"),
        ]
    )
    async with _harness("chat", lambda _: (200, upstream)) as (send, _, _):
        stream = "".join(
            [
                chunk
                async for chunk in send(wire, [{"role": "user", "content": "hello"}])
            ]
        )
    assert "fcc:history:" not in stream
    events = parse_sse_text(stream)
    if wire == "messages":
        starts = [
            event.data for event in events if event.event == "content_block_start"
        ]
        assert [event["content_block"]["type"] for event in starts] == (
            ["thinking"] if thinking else []
        ) + ["text"]
        stops = [
            event.data["index"]
            for event in events
            if event.event == "content_block_stop"
        ]
        assert stops[-1] == starts[-1]["index"]
    else:
        output = events[-1].data["response"]["output"]
        assert [item["type"] for item in output] == (
            ["reasoning"] if thinking else []
        ) + ["message"]
        assert all(not item.get("encrypted_content") for item in output)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_late_opaque_metadata_finishes_before_answer(wire):
    upstream = _chunks(
        [
            (
                {
                    "reasoning_content": "Plan.",
                    "reasoning_details": [
                        {"type": "reasoning.encrypted", "data": "first", "index": 0}
                    ],
                },
                None,
            ),
            ({"content": "Answer."}, None),
            (
                {
                    "reasoning_details": [
                        {"type": "reasoning.encrypted", "data": "second", "index": 0}
                    ]
                },
                "stop",
            ),
        ]
    )
    async with _harness("chat", lambda _: (200, upstream)) as (send, _, _):
        stream = "".join(
            [
                chunk
                async for chunk in send(wire, [{"role": "user", "content": "hello"}])
            ]
        )
    assert "fcc:history:" not in stream
    events = parse_sse_text(stream)
    if wire == "messages":
        starts = [
            event.data for event in events if event.event == "content_block_start"
        ]
        text_index = next(
            event["index"]
            for event in starts
            if event["content_block"]["type"] == "text"
        )
        assert [
            event.data["index"]
            for event in events
            if event.event == "content_block_stop"
        ][-1] == text_index
        assert [
            event.data["delta"]["signature"]
            for event in events
            if event.data.get("delta", {}).get("type") == "signature_delta"
        ] == ["firstsecond"]
    else:
        done = [
            event.data["item"]
            for event in events
            if event.event == "response.output_item.done"
        ]
        assert [
            item.get("encrypted_content")
            for item in done
            if item["type"] == "reasoning"
        ] == ["firstsecond"]
        assert done[-1]["type"] == "message"


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["messages", "responses"])
async def test_native_reasoning_is_forwarded_without_replacement(protocol):
    async with _harness(protocol) as (send, _, _):
        saved = await _saved_reply(
            send(protocol, [{"role": "user", "content": "hello"}]), protocol
        )
    item = saved[0]["content"][0] if protocol == "messages" else saved[0]
    assert item == _native(protocol)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("signature_first", [False, True])
async def test_google_message_signature_round_trips_in_its_native_field(
    wire, signature_first
):
    upstream = _chunks(
        [
            ({"content": "Pong"}, None),
            (
                {
                    "extra_content": {
                        "google": {"thought_signature": "google-signature"}
                    }
                },
                "stop",
            ),
        ]
    )
    if signature_first:
        signature = upstream[1]["choices"][0]
        signature["finish_reason"] = None
        upstream[0]["choices"][0]["finish_reason"] = "stop"
        upstream.reverse()

    def factory():
        return GeminiProvider(
            make_provider_config(
                api_key="test-key", base_url="https://provider.invalid/v1"
            ),
            admission=immediate_admission(),
        )

    async with _harness(
        "chat", lambda _: (200, upstream), chat_provider_factory=factory
    ) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
        await _saved_reply(
            send(wire, [*saved, {"role": "user", "content": "next"}]), wire
        )
    assistant = [
        message for message in bodies[-1]["messages"] if message["role"] == "assistant"
    ]
    assert len(assistant) == 1
    assert assistant[0]["content"] == "Pong"
    assert (
        assistant[0]["extra_content"]["google"]["thought_signature"]
        == "google-signature"
    )
    assert "reasoning_details" not in assistant[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("provider_kind", ["google", "structured"])
async def test_interrupted_opaque_only_state_keeps_hidden_retry_eligible(
    wire, provider_kind
):
    delta = (
        {"extra_content": {"google": {"thought_signature": "abandoned-signature"}}}
        if provider_kind == "google"
        else {
            "reasoning_details": [
                {
                    "type": "reasoning.encrypted",
                    "data": "abandoned-signature",
                    "index": 0,
                }
            ]
        }
    )
    first = _chunks([(delta, None)])

    class Cutoff(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield ("data: " + json.dumps(first[0]) + "\n\n").encode()
            await asyncio.sleep(0.9)

    def factory():
        return GeminiProvider(
            make_provider_config(
                api_key="test-key", base_url="https://provider.invalid/v1"
            ),
            admission=immediate_admission(max_attempts=2),
        )

    second = _chunks([({"content": "Recovered answer."}, "stop")])
    async with _harness(
        "chat",
        lambda bodies: (200, Cutoff() if len(bodies) == 1 else second),
        chat_provider_factory=factory if provider_kind == "google" else None,
        max_attempts=2,
    ) as (_, bodies, provider):
        if wire == "messages":
            stream = provider.stream_messages(
                MessagesRequest(
                    model="requested", messages=[{"role": "user", "content": "hello"}]
                ),
                reasoning=ReasoningPolicy.off(),
            )
        else:
            stream = provider.stream_responses(
                OpenAIResponsesRequest(model="requested", input="hello"),
                reasoning=ReasoningPolicy.off(),
            )
        raw = await delivered(stream, wire)
    assert len(bodies) == 2
    assert "Recovered answer." in raw
    assert "abandoned-signature" not in raw
    assert "response.failed" not in raw


def test_failure_does_not_publish_unfinished_opaque_state():
    output = AnthropicChatStreamOutput(
        message_id="msg_test", model="public", input_tokens=1
    )
    reasoning = StructuredReasoningStream()
    output.reasoning_replay = reasoning
    frames = output.start_events()
    frames += list(
        reasoning.events(
            {
                "reasoning_details": [
                    {
                        "type": "reasoning.encrypted",
                        "data": "partial-cipher",
                        "index": 0,
                    }
                ]
            },
            output,
            native_reasoning="Plan.",
        )
    )
    frames += output.ensure_text_block()
    frames.append(output.emit_text_delta("Partial answer."))
    output.defer_opaque_reasoning("pending-message-signature")
    frames += output.finish_failure(
        ExecutionFailure(FailureKind.UPSTREAM, 502, "Disconnected", True)
    )
    stream = "".join(frames)
    assert "partial-cipher" not in stream
    assert "pending-message-signature" not in stream
    assert "message_stop" not in stream


@pytest.mark.asyncio
async def test_responses_late_cipher_preserves_thinking_and_final_answer():
    original = _events_for("responses")
    message = {
        "id": "msg_answer",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Pong", "annotations": []}],
    }
    terminal = original[-1]
    terminal["response"]["output"].append(message)
    upstream = [
        *original[:3],
        {
            "type": "response.output_item.added",
            "sequence_number": 3,
            "output_index": 1,
            "item": {**message, "status": "in_progress", "content": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 4,
            "output_index": 1,
            "item_id": "msg_answer",
            "content_index": 0,
            "delta": "Pong",
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 5,
            "output_index": 1,
            "item": message,
        },
        {**original[3], "sequence_number": 6},
        {**terminal, "sequence_number": 7},
    ]
    async with _harness("responses", lambda _: (200, upstream)) as (send, _, _):
        stream = "".join(
            [
                chunk
                async for chunk in send(
                    "messages", [{"role": "user", "content": "hello"}]
                )
            ]
        )
    events = parse_sse_text(stream)
    starts = [event.data for event in events if event.event == "content_block_start"]
    text_index = next(
        event["index"] for event in starts if event["content_block"]["type"] == "text"
    )
    assert [
        event.data["index"] for event in events if event.event == "content_block_stop"
    ][-1] == text_index
    assert [
        event.data["delta"]["signature"]
        for event in events
        if event.data.get("delta", {}).get("type") == "signature_delta"
    ] == ["opaque-original"]


@pytest.mark.asyncio
async def test_responses_retains_distinct_summary_and_full_reasoning():
    upstream = _chunks(
        [
            (
                {
                    "reasoning_details": [
                        {
                            "type": "reasoning.summary",
                            "summary": "Short summary.",
                            "index": 0,
                        }
                    ]
                },
                None,
            ),
            ({"reasoning_content": "Distinct full thought."}, None),
            ({"content": "Answer."}, "stop"),
        ]
    )
    async with _harness("chat", lambda _: (200, upstream)) as (send, _, _):
        saved = await _saved_reply(
            send("responses", [{"role": "user", "content": "hello"}]), "responses"
        )
    reasoning = saved[0]
    assert reasoning["summary"] == [{"type": "summary_text", "text": "Short summary."}]
    assert reasoning["content"] == [
        {"type": "reasoning_text", "text": "Distinct full thought."}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_hiding_readable_thinking_preserves_opaque_replay_state(wire):
    upstream = _chunks(
        [
            (
                {
                    "reasoning_content": "Hidden thought.",
                    "reasoning_details": [
                        {
                            "type": "reasoning.encrypted",
                            "data": "opaque-kept",
                            "index": 0,
                        }
                    ],
                },
                None,
            ),
            ({"content": "Answer."}, "stop"),
        ]
    )
    async with _harness("chat", lambda _: (200, upstream)) as (_, _, provider):
        if wire == "messages":
            stream = provider.stream_messages(
                MessagesRequest(
                    model="requested", messages=[{"role": "user", "content": "hello"}]
                ),
                input_tokens=0,
                reasoning=ReasoningPolicy.off(),
            )
        else:
            stream = provider.stream_responses(
                OpenAIResponsesRequest(model="requested", input="hello"),
                input_tokens=0,
                reasoning=ReasoningPolicy.off(),
            )
        output = "".join([chunk async for chunk in stream])
    assert "Hidden thought." not in output
    assert "opaque-kept" in output


def test_converted_top_level_thought_has_valid_responses_identity():
    body = build_responses_provider_request(
        MessagesRequest(
            model="native",
            messages=[
                {
                    "role": "assistant",
                    "content": "Answer.",
                    "reasoning_content": "Plan.",
                }
            ],
        ),
        reasoning=ReasoningPolicy.on(),
    )
    assert body["input"][0]["id"].startswith("rs_")


def test_responses_summary_maps_to_structured_chat_summary():
    request = OpenAIResponsesRequest(
        model="native",
        input=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Short summary."}],
                "content": [{"type": "reasoning_text", "text": "Full thought."}],
                "encrypted_content": "cipher",
            },
            {"type": "message", "role": "assistant", "content": "Answer."},
        ],
    )
    body = build_responses_chat_request(
        request,
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
        structured_reasoning_details=True,
    ).body
    messages = body["messages"]
    assert isinstance(messages, list)
    assistant = messages[0]
    assert assistant["content"] == "Answer."
    assert assistant["reasoning_details"] == [
        {"type": "reasoning.text", "text": "Full thought.", "signature": "cipher"},
        {"type": "reasoning.summary", "summary": "Short summary."},
    ]
    assert json.dumps(body).count("Full thought.") == 1


def test_existing_responses_tool_delta_does_not_close_final_text():
    stream = ResponsesProviderStream(
        message_id="msg_test", model="native", input_tokens=1
    )
    frames = stream.start()
    frames += stream.feed(
        "response.output_item.added",
        {
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
            }
        },
    )
    frames += stream.feed(
        "response.function_call_arguments.delta", {"item_id": "fc_1", "delta": '{"q":'}
    )
    frames += stream.feed(
        "response.output_text.delta", {"item_id": "msg_answer", "delta": "Answer."}
    )
    frames += stream.feed(
        "response.function_call_arguments.delta", {"item_id": "fc_1", "delta": '"x"}'}
    )
    frames += stream.feed("response.completed", {"response": {}})
    events = parse_sse_text("".join(frames))
    starts = [event.data for event in events if event.event == "content_block_start"]
    text_index = next(
        event["index"] for event in starts if event["content_block"]["type"] == "text"
    )
    assert [
        event.data["index"] for event in events if event.event == "content_block_stop"
    ][-1] == text_index


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_unrecognized_detail_data_is_not_a_reasoning_signature(wire):
    upstream = _chunks(
        [
            (
                {
                    "reasoning_details": [
                        {"type": "provider.usage", "data": "ordinary-extra-data"}
                    ],
                    "content": "Answer.",
                },
                "stop",
            )
        ]
    )
    async with _harness("chat", lambda _: (200, upstream)) as (send, _, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
    assert "ordinary-extra-data" not in str(saved)
    blocks = saved[0]["content"] if wire == "messages" else saved
    assert [block["type"] for block in blocks] == [
        "text" if wire == "messages" else "message"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_fragmented_summary_and_distinct_full_thought_replay_once(wire):
    upstream = _chunks(
        [
            (
                {
                    "reasoning_details": [
                        {"type": "reasoning.summary", "summary": "Short ", "index": 0}
                    ]
                },
                None,
            ),
            (
                {
                    "reasoning_details": [
                        {"type": "reasoning.summary", "summary": "summary.", "index": 0}
                    ]
                },
                None,
            ),
            ({"reasoning_content": "Full thought."}, None),
            ({"content": "Answer."}, "stop"),
        ]
    )
    async with _harness("chat", lambda _: (200, upstream)) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
        await _saved_reply(
            send(wire, [*saved, {"role": "user", "content": "next"}]), wire
        )
    assert json.dumps(saved).count("Short summary.") == 1
    assert json.dumps(saved).count("Full thought.") == 1
    assert json.dumps(bodies[-1]).count("Short summary.") == 1
    assert json.dumps(bodies[-1]).count("Full thought.") == 1


@pytest.mark.asyncio
async def test_hidden_readable_only_cutoff_retries_without_delivered_content():
    first = _chunks(
        [
            (
                {
                    "reasoning_details": [
                        {
                            "type": "reasoning.text",
                            "text": "Hidden thought.",
                            "index": 0,
                        }
                    ]
                },
                None,
            )
        ]
    )

    class Cutoff(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield ("data: " + json.dumps(first[0]) + "\n\n").encode()
            await asyncio.sleep(0.9)

    second = _chunks([({"content": "Recovered answer."}, "stop")])
    async with _harness(
        "chat",
        lambda bodies: (200, Cutoff() if len(bodies) == 1 else second),
        max_attempts=2,
    ) as (_, bodies, provider):
        stream = provider.stream_responses(
            OpenAIResponsesRequest(model="requested", input="hello"),
            reasoning=ReasoningPolicy.off(),
        )
        raw = await delivered(stream, "responses")
    assert len(bodies) == 2
    assert "Recovered answer." in raw
    assert "Hidden thought." not in raw
    assert "response.failed" not in raw
    assert '"type": "reasoning"' not in raw


@pytest.mark.asyncio
@pytest.mark.parametrize("answer_before_done", [False, True])
async def test_completed_full_reasoning_survives_streamed_summary(answer_before_done):
    upstream = deepcopy(_events_for("responses"))
    upstream[3]["item"]["content"] = [
        {"type": "reasoning_text", "text": "Distinct full thought."}
    ]
    if answer_before_done:
        message = {
            "id": "msg_answer",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Answer.", "annotations": []}],
        }
        upstream[3:3] = [
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {**message, "status": "in_progress", "content": []},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_answer",
                "output_index": 1,
                "content_index": 0,
                "delta": "Answer.",
            },
            {"type": "response.output_item.done", "output_index": 1, "item": message},
        ]
        upstream[-1]["response"]["output"].append(message)
        for sequence, event in enumerate(upstream):
            event["sequence_number"] = sequence
    async with _harness("responses", lambda _: (200, upstream)) as (send, bodies, _):
        frames = [
            frame
            async for frame in send("messages", [{"role": "user", "content": "hello"}])
        ]

        async def saved_frames():
            for frame in frames:
                yield frame

        saved = await _saved_reply(saved_frames(), "messages")
        await _saved_reply(
            send("messages", [*saved, {"role": "user", "content": "next"}]), "messages"
        )
    assert json.dumps(saved).count("Find 17.") == 1
    assert json.dumps(saved).count("Distinct full thought.") == 1
    assert json.dumps(bodies[-1]).count("Find 17.") == 1
    assert json.dumps(bodies[-1]).count("Distinct full thought.") == 1
    if answer_before_done:
        events = parse_sse_text("".join(frames))
        text_index = next(
            event.data["index"]
            for event in events
            if event.event == "content_block_start"
            and event.data["content_block"]["type"] == "text"
        )
        assert [
            event.data["index"]
            for event in events
            if event.event == "content_block_stop"
        ][-1] == text_index


@pytest.mark.parametrize("summary", [False, True])
def test_responses_seeded_reasoning_and_fragments_are_completed_once(summary):
    stream = ResponsesProviderStream(
        message_id="msg_test", model="native", input_tokens=1
    )
    field = "summary" if summary else "content"
    part = "summary_text" if summary else "reasoning_text"
    item = {
        "type": "reasoning",
        "id": "rs_seeded",
        field: [{"type": part, "text": "First, "}],
    }
    frames = stream.start()
    frames += stream.feed("response.output_item.added", {"item": item})
    frames += stream.feed(
        "response.reasoning_summary_text.delta"
        if summary
        else "response.reasoning_text.delta",
        {"item_id": "rs_seeded", "delta": "think."},
    )
    frames += stream.feed(
        "response.output_item.done",
        {"item": {**item, field: [{"type": part, "text": "First, think."}]}},
    )
    frames += stream.feed("response.completed", {"response": {}})
    events = parse_sse_text("".join(frames))
    assert (
        "".join(
            event.data["delta"]["thinking"]
            for event in events
            if event.data.get("delta", {}).get("type") == "thinking_delta"
        )
        == "First, think."
    )

"""History crosses the real SDK/HTTP and both public presentation boundaries."""

import json
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import httpx
import httpx2
import pytest

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.core.anthropic import aggregate_anthropic_sse_to_message
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.history_replay import decode_replay, encode_replay
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.open_router import OpenRouterProvider
from tests.application.test_execution import (
    ControlledProvider,
    _routed_request,
    _target,
)
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_anthropic_messages_transport import Endpoint, _events
from tests.providers.test_anthropic_messages_transport import (
    _transport as messages_transport,
)
from tests.providers.test_openai_responses_transport import _client, _completed_response
from tests.providers.test_openai_responses_transport import (
    _transport as responses_transport,
)


def _native(protocol):
    if protocol == "responses":
        return {
            "type": "reasoning",
            "id": "rs_a:rs_b",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": "Find 17."}],
            "encrypted_content": "opaque-original",
            "extension": {"keep": 17},
        }
    if protocol == "messages":
        return {
            "type": "thinking",
            "thinking": "Find 17.",
            "signature": "opaque-original",
            "extension": {"keep": 17},
        }
    return {
        "reasoning_details": [
            {"type": "reasoning.text", "text": "Find 17.", "index": 0},
            {
                "type": "reasoning.encrypted",
                "data": "opaque-original",
                "format": "native-test",
                "index": 1,
                "extension": {"keep": 17},
            },
        ]
    }


def _events_for(protocol):
    native = _native(protocol)
    if protocol == "messages":
        events = _events()
        return [
            events[0],
            {"type": "content_block_start", "index": 0, "content_block": native},
            {"type": "content_block_stop", "index": 0},
            *events[-2:],
        ]
    if protocol == "responses":
        completed = _completed_response(model="actual-returned")
        completed["output"] = [native]
        return [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**completed, "status": "in_progress", "output": []},
            },
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    **native,
                    "status": "in_progress",
                    "encrypted_content": None,
                    "summary": [],
                },
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": 2,
                "item_id": native["id"],
                "output_index": 0,
                "summary_index": 0,
                "delta": "Find 17.",
            },
            {
                "type": "response.output_item.done",
                "sequence_number": 3,
                "output_index": 0,
                "item": native,
            },
            {"type": "response.completed", "sequence_number": 4, "response": completed},
        ]
    return [
        {
            "id": "chat_test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "actual-returned",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        for delta, finish in [
            (
                {"role": "assistant", "reasoning_details": native["reasoning_details"]},
                None,
            ),
            ({"content": "17"}, "stop"),
        ]
    ]


@asynccontextmanager
async def _harness(protocol, responder=None, *, key="a", chat_provider_factory=None):
    bodies: list[dict[str, Any]] = []

    def reply(request):
        bodies.append(json.loads(request.content))
        status, payload = (
            responder(bodies) if responder else (200, _events_for(protocol))
        )
        module = httpx if protocol == "messages" else httpx2
        if status != 200:
            return module.Response(status, json={"error": payload})
        raw = "".join(
            (f"event: {event['type']}\n" if protocol == "messages" else "")
            + f"data: {json.dumps(event)}\n\n"
            for event in payload
        )
        return module.Response(
            200, headers={"content-type": "text/event-stream"}, content=raw
        )

    if protocol == "messages":
        client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        provider = messages_transport(client, immediate_admission(max_attempts=5))
        endpoint = Endpoint()
        endpoint.token = key
        extras = {"endpoint_context": endpoint}
    else:
        client = _client(reply, api_key=key)
        if protocol == "responses":
            provider = responses_transport(client)
        else:
            with patch(
                "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
                return_value=client,
            ):
                provider = (
                    chat_provider_factory()
                    if chat_provider_factory is not None
                    else OpenRouterProvider(
                        make_provider_config(
                            api_key=key, base_url="https://provider.invalid/v1"
                        ),
                        admission=immediate_admission(max_attempts=5),
                    )
                )
        extras = {}

    def stream(wire, history):
        options: dict[str, Any] = dict(
            input_tokens=0,
            request_id="history-test",
            response_model="public-alias",
            reasoning=ReasoningPolicy.on(),
            **extras,
        )
        if protocol == "messages":
            options.pop("input_tokens")
            options["reasoning"] = ReasoningPolicy.provider_default()
        if wire == "responses":
            return provider.stream_responses(
                OpenAIResponsesRequest.model_validate(
                    {"model": "requested", "input": history}
                ),
                **options,
            )
        return provider.stream_messages(
            MessagesRequest.model_validate({"model": "requested", "messages": history}),
            **options,
        )

    try:
        yield stream, bodies, provider
    finally:
        if isinstance(client, httpx.AsyncClient):
            await client.aclose()
        else:
            await client.close()


async def _saved_reply(stream, wire):
    if wire == "messages":
        message, error, saw_stop = await aggregate_anthropic_sse_to_message(stream)
        assert error is None and saw_stop
        return [{"role": "assistant", "content": message["content"]}]
    events = parse_sse_text("".join([event async for event in stream]))
    return events[-1].data["response"]["output"]


def _carrier(history, wire):
    if wire == "responses":
        return next(
            item["encrypted_content"]
            for item in history
            if item.get("type") == "reasoning"
        )
    return next(
        block.get("signature", block.get("data"))
        for block in history[0]["content"]
        if block["type"] in {"thinking", "redacted_thinking"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["responses", "messages", "chat"])
@pytest.mark.parametrize("wire", ["responses", "messages"])
async def test_provider_a_b_a_preserves_exact_native_history(protocol, wire):
    async with _harness(protocol) as (send, bodies, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
        record = decode_replay(_carrier(saved, wire))
        assert record.native == _native(protocol)
        assert record.origin.model == (
            "native" if protocol == "messages" else "actual-returned"
        )
        assert record.origin.model != "public-alias"
        history = json.loads(
            json.dumps([*saved, {"role": "user", "content": "continue"}])
        )
        original = deepcopy(history)
        async with _harness(protocol, key="b") as (foreign, foreign_bodies, _):
            await _saved_reply(foreign(wire, history), wire)
            foreign_wire = json.dumps(foreign_bodies[-1])
            assert (
                "opaque-original" not in foreign_wire
                and "fcc:history" not in foreign_wire
            )
            assert "Find 17." in foreign_wire
        await _saved_reply(send(wire, history), wire)
        if protocol == "responses":
            assert bodies[-1]["input"][0] == _native(protocol)
        elif protocol == "messages":
            assert bodies[-1]["messages"][0]["content"][0] == _native(protocol)
        else:
            assert (
                bodies[-1]["messages"][0]["reasoning_details"]
                == _native(protocol)["reasoning_details"]
            )
        assert history == original


@pytest.mark.asyncio
async def test_legacy_responses_id_then_cipher_rejection_recovers_without_changing_effort():
    wire = "responses"

    def responder(bodies):
        if len(bodies) == 1:
            return 400, {
                "code": "invalid_value",
                "param": "input[0].id",
                "message": "Invalid 'input[0].id': 'rs_a:rs_b'. Expected an ID that contains letters, numbers, underscores, or dashes.",
            }
        if len(bodies) == 2:
            return 400, {
                "code": "invalid_encrypted_content",
                "param": "input[0].encrypted_content",
                "message": "The encrypted content could not be verified.",
            }
        return 200, _events_for("responses")

    history = [
        _native("responses"),
        {"role": "assistant", "content": "17"},
        {"role": "user", "content": "continue"},
    ]
    original = deepcopy(history)
    async with _harness("responses", responder) as (send, bodies, _):
        await _saved_reply(send(wire, history), wire)
        assert len(bodies) == 3
        assert ":" not in bodies[1]["input"][0]["id"]
        assert bodies[1]["input"][0]["encrypted_content"] == "opaque-original"
        assert bodies[2]["input"][0] == {
            "role": "assistant",
            "content": "[Earlier reasoning summary]\nFind 17.",
        }
        assert all(body["reasoning"] == bodies[0]["reasoning"] for body in bodies)
        assert history == original


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["responses", "messages", "chat"])
async def test_explicit_legacy_rejection_corrects_only_history_before_commit(protocol):
    def responder(bodies):
        if len(bodies) == 1:
            return 400, {
                "code": "invalid_encrypted_content"
                if protocol != "messages"
                else "invalid_signature",
                "message": "The encrypted content or signature could not be verified.",
            }
        return 200, _events_for(protocol)

    block = (
        _native("messages")
        if protocol == "messages"
        else {"type": "redacted_thinking", "data": "opaque-original"}
    )
    history = [
        {"role": "assistant", "content": [block, {"type": "text", "text": "17"}]},
        {"role": "user", "content": "continue"},
    ]
    original = deepcopy(history)
    async with _harness(protocol, responder) as (send, bodies, _):
        await _saved_reply(send("messages", history), "messages")
        assert len(bodies) == 2
        assert "opaque-original" in json.dumps(bodies[0])
        assert "opaque-original" not in json.dumps(bodies[1])
        assert "17" in json.dumps(bodies[1])
        assert history == original


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["responses", "messages", "chat"])
@pytest.mark.parametrize("committed", [False, True])
async def test_stream_history_rejection_respects_the_public_commit_boundary(
    protocol, committed
):
    visible = "already visible " * 5000  # Exceeds the existing holdback budget.
    error = {
        "code": "invalid_signature"
        if protocol == "messages"
        else "invalid_encrypted_content",
        "type": "invalid_request_error",
        "message": "The encrypted content or signature could not be verified.",
    }

    def responder(bodies):
        if len(bodies) > 1:
            return 200, _events_for(protocol)
        if protocol == "messages":
            events = _events(visible)[:3] if committed else _events()[:1]
            return 200, [*events, {"type": "error", "error": error}]
        if protocol == "responses":
            events = _events_for(protocol)[:1]
            if committed:
                events.append(
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": 1,
                        "item_id": "text",
                        "content_index": 0,
                        "output_index": 0,
                        "delta": visible,
                        "logprobs": [],
                    }
                )
            return 200, [*events, {**error, "type": "error", "sequence_number": 2}]
        events = []
        if committed:
            chunk = deepcopy(_events_for(protocol)[-1])
            chunk["choices"][0] = {
                "index": 0,
                "delta": {"content": visible},
                "finish_reason": None,
            }
            events.append(chunk)
        return 200, [*events, {"error": error}]

    block = (
        _native("messages")
        if protocol == "messages"
        else {"type": "redacted_thinking", "data": "opaque-original"}
    )
    history = [
        {"role": "assistant", "content": [block, {"type": "text", "text": "17"}]},
        {"role": "user", "content": "continue"},
    ]
    async with _harness(protocol, responder) as (send, bodies, _):
        if committed:
            output = ""
            with pytest.raises(ExecutionFailure):
                async for event in send("messages", history):
                    output += event
            assert visible in output
            assert len(bodies) == 1
        else:
            await _saved_reply(send("messages", history), "messages")
            assert len(bodies) == 2
            assert "opaque-original" not in json.dumps(bodies[1])


@pytest.mark.asyncio
async def test_successful_fallback_stamps_its_own_origin_and_gets_unmodified_input():
    class MutatingPrimary(ControlledProvider):
        async def stream_messages(self, request, input_tokens=0, **kwargs):
            request.messages[0].content = "primary mutation"
            async for event in super().stream_messages(request, input_tokens, **kwargs):
                yield event

    primary = MutatingPrimary(
        [ExecutionFailure(FailureKind.UNAVAILABLE, 503, "unavailable", True)]
    )
    bodies = []

    def reply(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(
                f"data: {json.dumps(event)}\n\n" for event in _events_for("chat")
            ),
        )

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        return_value=_client(reply),
    ):
        provider = OpenRouterProvider(
            make_provider_config(
                api_key="fallback-key", base_url="https://provider.invalid/v1"
            ),
            admission=immediate_admission(),
        )
    routed = _routed_request(_target("open_router", "fallback-model"))
    original = routed.request.model_dump()
    executor = ProviderExecutor(
        lambda name: primary if name == "provider" else provider,
        progress_timeout_seconds=10,
    )
    try:
        saved = await _saved_reply(
            executor.stream_messages(
                routed, raw_log_payload={}, request_id="actual-fallback"
            ),
            "messages",
        )
        record = decode_replay(_carrier(saved, "messages"))
        assert record.origin.provider == "OPENROUTER"
        assert record.origin.model == "actual-returned"
        assert bodies[0]["messages"][0]["content"] == "hello"
        assert routed.request.model_dump() == original
    finally:
        await provider._client.close()


@pytest.mark.asyncio
async def test_history_corrections_exhaust_the_shared_attempt_budget():
    history = [
        {
            "type": "reasoning",
            "id": f"rs_{index}",
            "summary": [],
            "encrypted_content": f"opaque{index}",
        }
        for index in range(7)
    ]
    history += [{"role": "user", "content": "continue"}]
    original = deepcopy(history)

    def responder(bodies):
        return 400, {
            "code": "invalid_encrypted_content",
            "param": "input[0].encrypted_content",
            "message": "The encrypted content could not be verified.",
        }

    async with _harness("responses", responder) as (send, bodies, _):
        with pytest.raises(ExecutionFailure):
            await _saved_reply(send("responses", history), "responses")
        assert len(bodies) == 5
        assert [len(body["input"]) for body in bodies] == [8, 7, 6, 5, 4]
        assert history == original


def _chat_reasoning_events(deltas):
    template = _events_for("chat")[0]
    return [
        {**template, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        for delta in deltas
    ] + [_events_for("chat")[-1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize(
    ("prefix", "suffix", "finish", "has_tool"),
    [
        ("<", "17", "stop", False),
        ("<th", "ink>inline</think>17", "stop", False),
        (
            "<tool_call>",
            "<function=lookup><parameter=query>x</parameter></function></tool_call>",
            "stop",
            True,
        ),
        (
            "● <function=lookup>",
            "<parameter=query>x</parameter></function>",
            "stop",
            True,
        ),
        ("<", None, "tool_calls", True),
        ("<", "", "length", False),
        ("<think>", "", "stop", False),
    ],
    ids=[
        "text",
        "thinking",
        "function-tag",
        "heuristic-tool",
        "native-tool",
        "length",
        "thinking-end",
    ],
)
async def test_chat_buffered_content_preserves_reasoning_through_next_request(
    wire, fragmented, prefix, suffix, finish, has_tool
):
    first = {"type": "reasoning.encrypted", "data": "first", "index": 0}
    second = {
        "type": "reasoning.encrypted",
        "data": "second",
        "index": 0 if fragmented else 1,
    }
    expected = [{**first, "data": "firstsecond"}] if fragmented else [first, second]
    deltas = [
        {"reasoning_content": "Plan.", "reasoning_details": [first]},
        {"content": prefix},
        {"reasoning_details": [second]},
        {"content": suffix}
        if suffix is not None
        else {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_lookup",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"x"}'},
                }
            ]
        },
    ]
    events = _chat_reasoning_events(deltas)
    events[-1]["choices"][0].update(delta={}, finish_reason=finish)
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def send(history):
        if wire == "messages":
            return provider.stream_messages(
                MessagesRequest.model_validate(
                    {
                        "model": "requested",
                        "messages": history,
                        "tools": [{"name": "lookup", "input_schema": schema}],
                    }
                ),
                reasoning=ReasoningPolicy.on(),
            )
        return provider.stream_responses(
            OpenAIResponsesRequest.model_validate(
                {
                    "model": "requested",
                    "input": history,
                    "tools": [
                        {"type": "function", "name": "lookup", "parameters": schema}
                    ],
                }
            ),
            reasoning=ReasoningPolicy.on(),
        )

    async with _harness("chat", lambda bodies: (200, events)) as (_, bodies, provider):
        saved = await _saved_reply(send([{"role": "user", "content": "hello"}]), wire)
        assert decode_replay(_carrier(saved, wire)).native == {
            "reasoning_content": "Plan.",
            "reasoning_details": expected,
        }
        history = deepcopy(saved)
        blocks = saved[0]["content"] if wire == "messages" else saved
        calls = [
            block for block in blocks if block["type"] in {"tool_use", "function_call"}
        ]
        assert len(calls) == int(has_tool)
        for call in calls:
            assert call["name"] == "lookup"
            if wire == "messages":
                assert call["input"] == {"query": "x"}
                history.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call["id"],
                                "content": "result",
                            }
                        ],
                    }
                )
            else:
                assert json.loads(call["arguments"]) == {"query": "x"}
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": "result",
                    }
                )
        history.append({"role": "user", "content": "next"})
        original = deepcopy(history)
        await _saved_reply(send(history), wire)
        assert len(bodies) == 2
        assert [
            detail
            for message in bodies[-1]["messages"]
            for detail in message.get("reasoning_details", [])
        ] == expected
        assert history == original


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
async def test_chat_encrypted_only_completion_does_not_add_blank_text(wire):
    details = [{"type": "reasoning.encrypted", "data": "opaque-only", "index": 0}]
    events = _chat_reasoning_events([{"reasoning_details": details}])
    events[-1]["choices"][0]["delta"] = {}
    async with _harness("chat", lambda bodies: (200, events)) as (send, _, _):
        saved = await _saved_reply(
            send(wire, [{"role": "user", "content": "hello"}]), wire
        )
    assert decode_replay(_carrier(saved, wire)).native == {"reasoning_details": details}
    blocks = saved[0]["content"] if wire == "messages" else saved
    assert [block["type"] for block in blocks] == [
        "redacted_thinking" if wire == "messages" else "reasoning"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("committed", [False, True])
async def test_chat_pending_reasoning_is_finalized_on_failure_or_discarded_on_retry(
    wire, committed
):
    text = "Plan." * (30000 if committed else 1)
    first = {"type": "reasoning.encrypted", "data": "first", "index": 0}
    second = {"type": "reasoning.encrypted", "data": "second", "index": 1}
    events = _chat_reasoning_events(
        [
            {"reasoning_content": text, "reasoning_details": [first]},
            {"content": "<"},
            {"reasoning_details": [second]},
        ]
    )[:-1]
    # A cutoff retries before commitment; a terminal API error closes visible output.
    if committed:
        events.append(
            {"error": {"type": "invalid_request_error", "message": "stream failed"}}
        )

    def responder(bodies):
        return 200, events if len(bodies) == 1 else _events_for("chat")

    async with _harness("chat", responder) as (send, bodies, _):
        stream = send(wire, [{"role": "user", "content": "hello"}])
        if not committed:
            saved = await _saved_reply(stream, wire)
            assert len(bodies) == 2
            assert decode_replay(_carrier(saved, wire)).native == _native("chat")
            return
        if wire == "messages":
            output = ""
            with pytest.raises(ExecutionFailure):
                async for frame in stream:
                    output += frame

            async def replay_frames():
                yield output

            message, _, saw_stop = await aggregate_anthropic_sse_to_message(
                replay_frames()
            )
            assert not saw_stop
            saved = [{"role": "assistant", "content": message["content"]}]
        else:
            frames = [frame async for frame in stream]
            response = parse_sse_text("".join(frames))[-1].data["response"]
            assert response["status"] == "failed"
            saved = response["output"]
        assert len(bodies) == 1
        assert decode_replay(_carrier(saved, wire)).native == {
            "reasoning_content": text,
            "reasoning_details": [first, second],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("destination", ["chat", "messages", "responses"])
@pytest.mark.parametrize("incomplete_saved_record", [False, True])
async def test_chat_plaintext_beside_encrypted_details_survives_switching(
    wire, destination, incomplete_saved_record
):
    text = "The crucial visible reasoning."
    details = [{"type": "reasoning.encrypted", "data": "opaque-only", "index": 0}]
    events = _chat_reasoning_events(
        [{"reasoning_content": text, "reasoning_details": details}]
    )
    async with _harness("chat", lambda bodies: (200, events)) as (source, sent, _):
        saved = await _saved_reply(
            source(wire, [{"role": "user", "content": "hello"}]), wire
        )
        record = decode_replay(_carrier(saved, wire))
        if incomplete_saved_record:
            record.native.pop("reasoning_content", None)
            carrier = encode_replay(record)
            if wire == "responses":
                saved[0]["encrypted_content"] = carrier
            else:
                saved[0]["content"][0]["signature"] = carrier
        else:
            assert record.native["reasoning_content"] == text
        history = json.loads(json.dumps([*saved, {"role": "user", "content": "next"}]))
        original = deepcopy(history)
        async with _harness(destination, key="b") as (foreign, bodies, _):
            await _saved_reply(foreign(wire, history), wire)
            outgoing = json.dumps(bodies[-1])
            assert outgoing.count(text) == 1
            assert "opaque-only" not in outgoing and "fcc:history" not in outgoing
        await _saved_reply(source(wire, history), wire)
        assert sent[-1]["messages"][0]["reasoning_details"] == details
        assert json.dumps(sent[-1]).count(text) == 1
        assert history == original


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["messages", "responses"])
@pytest.mark.parametrize("encrypted", [False, True])
async def test_chat_summary_fragments_remain_readable_and_restore_exactly(
    wire, encrypted
):
    first = {"type": "reasoning.summary", "summary": "Short ", "index": 0}
    second = {"type": "reasoning.summary", "summary": "summary.", "index": 0}
    opaque = {"type": "reasoning.encrypted", "data": "opaque-only", "index": 1}
    details = [{**first, "summary": "Short summary."}, *([opaque] if encrypted else [])]
    events = _chat_reasoning_events(
        [
            {"reasoning_details": [first]},
            {"reasoning_details": [second, *([opaque] if encrypted else [])]},
        ]
    )
    async with _harness("chat", lambda bodies: (200, events)) as (source, sent, _):
        saved = await _saved_reply(
            source(wire, [{"role": "user", "content": "hello"}]), wire
        )
        assert json.dumps(saved).count("Short summary.") == 1
        record = decode_replay(_carrier(saved, wire))
        assert record.native["reasoning_details"] == details
        history = [*saved, {"role": "user", "content": "next"}]
        async with _harness("responses", key="b") as (foreign, bodies, _):
            await _saved_reply(foreign(wire, history), wire)
            assert bodies[-1]["input"][0] == {
                "role": "assistant",
                "content": "[Earlier reasoning summary]\nShort summary.",
            }
        await _saved_reply(source(wire, history), wire)
        assert sent[-1]["messages"][0]["reasoning_details"] == details


@pytest.mark.asyncio
@pytest.mark.parametrize("native_first", [False, True])
async def test_chat_alternate_readable_fields_replay_once(native_first):
    native = {"reasoning_content": "One readable thought."}
    typed = {
        "reasoning_details": [
            {
                "type": "reasoning.text",
                "text": "One readable thought.",
                "index": 0,
            }
        ]
    }
    events = _chat_reasoning_events(
        [native, typed] if native_first else [typed, native]
    )
    async with _harness("chat", lambda bodies: (200, events)) as (source, sent, _):
        saved = await _saved_reply(
            source("messages", [{"role": "user", "content": "hi"}]), "messages"
        )
        assert json.dumps(saved).count("One readable thought.") == 1
        history = [*saved, {"role": "user", "content": "next"}]
        async with _harness("responses", key="b") as (foreign, bodies, _):
            await _saved_reply(foreign("messages", history), "messages")
            assert json.dumps(bodies[-1]).count("One readable thought.") == 1
        await _saved_reply(source("messages", history), "messages")
        assert json.dumps(sent[-1]).count("One readable thought.") == 1


@pytest.mark.asyncio
async def test_chat_reasoning_groups_do_not_inherit_previous_plaintext():
    events = _chat_reasoning_events(
        [
            {
                "reasoning_content": "First thought.",
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "first-", "index": 0}
                ],
            },
            {
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "secret", "index": 0}
                ]
            },
            {"content": "First answer."},
            {
                "reasoning_content": "Second thought.",
                "reasoning_details": [
                    {"type": "reasoning.encrypted", "data": "second-secret", "index": 0}
                ],
            },
        ]
    )
    async with _harness("chat", lambda bodies: (200, events)) as (source, _, _):
        saved = await _saved_reply(
            source("messages", [{"role": "user", "content": "hi"}]), "messages"
        )
    records = [
        decode_replay(block["signature"]).native
        for block in saved[0]["content"]
        if block["type"] == "thinking"
    ]
    assert records == [
        {
            "reasoning_content": "First thought.",
            "reasoning_details": [
                {"type": "reasoning.encrypted", "data": "first-secret", "index": 0}
            ],
        },
        {
            "reasoning_content": "Second thought.",
            "reasoning_details": [
                {"type": "reasoning.encrypted", "data": "second-secret", "index": 0}
            ],
        },
    ]


@pytest.mark.asyncio
async def test_chat_plaintext_matching_multiple_detail_parts_is_not_repeated():
    events = _chat_reasoning_events(
        [
            {
                "reasoning_content": "First part. Second part.",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "First part. ", "index": 0},
                    {"type": "reasoning.text", "text": "Second part.", "index": 1},
                ],
            }
        ]
    )
    async with _harness("chat", lambda bodies: (200, events)) as (source, _, _):
        saved = await _saved_reply(
            source("messages", [{"role": "user", "content": "hi"}]), "messages"
        )
    async with _harness("responses", key="b") as (foreign, bodies, _):
        await _saved_reply(
            foreign("messages", [*saved, {"role": "user", "content": "next"}]),
            "messages",
        )
    outgoing = json.dumps(bodies[-1])
    assert outgoing.count("First part.") == outgoing.count("Second part.") == 1

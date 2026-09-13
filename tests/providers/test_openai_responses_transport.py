"""Contracts for the standard API-key OpenAI Responses transport."""

import asyncio
import json
import re
from collections.abc import Callable, Mapping

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
    thinking_content,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesToolPolicy,
)
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport
from tests.providers.support import REASONING_ON, immediate_admission


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_error", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        {
            "code": "invalid_request_error",
            "message": "Reasoning is mandatory and cannot be disabled.",
        },
        {
            "code": "unsupported_value",
            "param": "reasoning.effort",
            "message": "Value 'none' is not supported",
        },
    ],
    ids=["mandatory", "unsupported_value"],
)
async def test_tolerant_classifier_corrects_required_reasoning(stream_error, error):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("reasoning", {}).get("effort") == "none":
            if stream_error:
                return httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    text=_sse({"type": "error", **error, "sequence_number": 0}),
                )
            return httpx2.Response(400, json={"error": error})
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_text_delta("<severity>0</severity>"), _completed_event()),
        )

    client = _client(handler)
    try:
        output = [
            event
            async for event in _transport(client).stream_messages(
                _request(max_tokens=64, thinking={"type": "disabled"}),
                input_tokens=11,
                request_id="classifier",
                response_model="public",
                reasoning=ReasoningPolicy.prefer_off(),
            )
        ]
        assert text_content(parse_sse_text("".join(output))) == "<severity>0</severity>"
        assert len(bodies) == 2
        assert bodies[0]["reasoning"]["effort"] == "none"
        assert "max_output_tokens" not in bodies[0]
        assert "reasoning" not in bodies[1]
    finally:
        await client.close()


def _request(**overrides: object) -> MessagesRequest:
    payload: dict[str, object] = {
        "model": "upstream-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 123,
        "metadata": {"source": "test"},
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def _completed_response(
    *,
    model: str = "upstream-model",
    input_tokens: int = 8,
    cached_tokens: int = 3,
    output_tokens: int = 2,
) -> dict[str, object]:
    return {
        "id": "resp_test",
        "created_at": 0,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "model": model,
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "background": False,
        "conversation": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "previous_response_id": None,
        "prompt": None,
        "prompt_cache_key": None,
        "reasoning": None,
        "safety_identifier": None,
        "service_tier": "default",
        "status": "completed",
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "top_logprobs": 0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
        "user": None,
        "store": False,
    }


def _text_delta(text: str, *, sequence: int = 0) -> dict[str, object]:
    return {
        "type": "response.output_text.delta",
        "sequence_number": sequence,
        "item_id": "item_text",
        "output_index": 0,
        "content_index": 0,
        "delta": text,
        "logprobs": [],
    }


def _completed_event(*, sequence: int = 1) -> dict[str, object]:
    return {
        "type": "response.completed",
        "sequence_number": sequence,
        "response": _completed_response(),
    }


def _sse(*events: Mapping[str, object]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _client(
    handler: Callable[[httpx2.Request], httpx2.Response],
    *,
    api_key: str = "test-key",
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def _transport(
    client: AsyncOpenAI,
    *,
    max_attempts: int = 5,
    tool_policy: ResponsesToolPolicy = ResponsesToolPolicy(),
):
    return OpenAIResponsesTransport(
        client=client,
        admission=immediate_admission(
            provider_name="TEST_RESPONSES",
            max_attempts=max_attempts,
        ),
        provider_name="TEST_RESPONSES",
        read_timeout_s=120.0,
        log_raw_sse_events=False,
        tool_policy=tool_policy,
    )


async def _collect(
    transport: OpenAIResponsesTransport,
    request: MessagesRequest | None = None,
) -> list[str]:
    return [
        chunk
        async for chunk in transport.stream_messages(
            request or _request(),
            input_tokens=11,
            request_id="req_responses",
            response_model="public-model",
            reasoning=REASONING_ON,
        )
    ]


async def _collect_native(
    transport: OpenAIResponsesTransport,
    request: OpenAIResponsesRequest,
    *,
    reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
) -> list[str]:
    return [
        chunk
        async for chunk in transport.stream_responses(
            request,
            input_tokens=11,
            request_id="req_native_responses",
            response_model="public-model",
            reasoning=reasoning,
        )
    ]


@pytest.mark.asyncio
async def test_default_transport_keeps_native_tool_capabilities() -> None:
    tools = [
        {
            "type": "custom",
            "name": "edit",
            "format": {"type": "text"},
            "defer_loading": True,
            "allowed_callers": ["direct"],
        },
        {"type": "web_search", "search_content_types": ["text", "image"]},
    ]
    history = [
        {"type": "custom_tool_call", "call_id": "c", "name": "edit", "input": "patch"}
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        assert payload["tools"] == tools and payload["input"] == history
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_completed_event()),
        )

    client = _client(handler)
    try:
        await _collect_native(
            _transport(client),
            OpenAIResponsesRequest(model="example", tools=tools, input=history),
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_adaptation_retries_start_with_fresh_event_state() -> None:
    attempts = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        wire = json.loads(request.content)
        assert wire["tools"][0]["type"] == "function"
        created = {
            "type": "response.created",
            "sequence_number": 100 if attempts == 1 else 0,
            "response": {**_completed_response(), "status": "in_progress"},
        }
        text = (
            _sse(created)
            if attempts == 1
            else _sse(
                created, _text_delta("hello", sequence=1), _completed_event(sequence=2)
            )
        )
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=text
        )

    client = _client(handler)
    try:
        transport = _transport(
            client,
            max_attempts=2,
            tool_policy=ResponsesToolPolicy(custom_tools_as_functions=True),
        )
        chunks = await _collect_native(
            transport,
            OpenAIResponsesRequest(
                model="example",
                input="hello",
                tools=[{"type": "custom", "name": "edit"}],
            ),
        )
    finally:
        await client.close()
    events = parse_sse_text("".join(chunks))
    assert attempts == 2
    assert [event.data["sequence_number"] for event in events] == [0, 1, 2]


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_share_tool_identities() -> None:
    started = 0
    both_started = asyncio.Event()

    class ResponseStream(httpx2.AsyncByteStream):
        def __init__(self, content: str) -> None:
            self.content = content

        async def __aiter__(self):
            await asyncio.wait_for(both_started.wait(), timeout=5)
            yield self.content.encode()

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        body = json.loads(request.content)
        call = {
            "type": "function_call",
            "id": "same_item",
            "call_id": "same_call",
            "name": "edit",
            "arguments": '{"input":"patch"}',
            "status": "completed",
        }
        response = {
            **_completed_response(),
            "output": [call],
            "metadata": {"owner": body["input"]},
        }
        events = _sse(
            {
                "type": "response.output_item.added",
                "sequence_number": 0,
                "output_index": 0,
                "item": {**call, "arguments": "", "status": "in_progress"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 1,
                "item_id": "same_item",
                "output_index": 0,
                "delta": call["arguments"],
            },
            {
                "type": "response.output_item.done",
                "sequence_number": 2,
                "output_index": 0,
                "item": call,
            },
            {"type": "response.completed", "sequence_number": 3, "response": response},
        )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ResponseStream(events),
        )

    client = _client(handler)
    transport = _transport(
        client, tool_policy=ResponsesToolPolicy(custom_tools_as_functions=True)
    )
    try:
        chunks = await asyncio.gather(
            *[
                _collect_native(
                    transport,
                    OpenAIResponsesRequest(
                        model="example",
                        input=kind,
                        tools=[{"type": kind, "name": "edit"}],
                    ),
                )
                for kind in ("custom", "function")
            ]
        )
    finally:
        await client.close()
    custom_events, function_events = [
        parse_sse_text("".join(output)) for output in chunks
    ]
    assert custom_events[-1].data["response"]["output"][0]["type"] == "custom_tool_call"
    assert function_events[-1].data["response"]["output"][0]["type"] == "function_call"
    assert not any(
        event.event == "response.function_call_arguments.delta"
        for event in custom_events
    )
    assert any(
        event.event == "response.function_call_arguments.delta"
        for event in function_events
    )


@pytest.mark.asyncio
async def test_native_responses_preserves_request_and_upstream_event_identity() -> None:
    captured: list[dict[str, object]] = []
    created_response = {
        **_completed_response(model="upstream-model"),
        "created_at": 1788587503,
        "status": "in_progress",
        "output": [],
        "usage": None,
    }
    created = {
        "type": "response.created",
        "sequence_number": 0,
        "response": created_response,
    }
    delta = _text_delta("hello", sequence=1)
    completed = {
        **_completed_event(sequence=2),
        "response": {
            **_completed_response(),
            "created_at": 1788587503,
            "completed_at": 1788587504,
        },
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        captured.append(payload)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(created, delta, completed),
        )

    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "upstream-model",
            "input": [{"role": "user", "content": "hello"}],
            "stream": False,
            "store": True,
            "previous_response_id": "resp_previous",
            "future_option": {"enabled": True},
        }
    )
    client = _client(handler)
    try:
        chunks = await _collect_native(_transport(client), request)
    finally:
        await client.close()

    assert captured == [
        {
            "model": "upstream-model",
            "input": [{"role": "user", "content": "hello"}],
            "stream": True,
            "store": False,
            "future_option": {"enabled": True},
        }
    ]
    events = parse_sse_text("".join(chunks))
    assert [event.event for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[0].data["response"]["id"] == "resp_test"
    assert events[0].data["response"]["model"] == "public-model"
    for event in (events[0], events[2]):
        timestamp = event.data["response"]["created_at"]
        assert type(timestamp) is int
        assert timestamp == 1788587503
    assert events[1].data == delta
    assert events[2].data["response"]["model"] == "public-model"
    assert type(events[2].data["response"]["completed_at"]) is int
    assert events[2].data["response"]["completed_at"] == 1788587504


@pytest.mark.asyncio
async def test_native_responses_applies_resolved_reasoning_override() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        captured.append(payload)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_completed_event()),
        )

    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "upstream-model",
            "input": "hello",
            "reasoning": {"effort": "high", "summary": "detailed"},
        }
    )
    client = _client(handler)
    try:
        await _collect_native(
            _transport(client),
            request,
            reasoning=ReasoningPolicy.off(),
        )
    finally:
        await client.close()

    assert captured[0]["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
async def test_native_response_failed_retries_before_public_commitment() -> None:
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        created_response = {
            **_completed_response(),
            "id": f"resp_attempt_{attempts}",
            "status": "in_progress",
            "output": [],
            "usage": None,
        }
        created = {
            "type": "response.created",
            "sequence_number": 0,
            "response": created_response,
        }
        if attempts == 1:
            failed = {
                "type": "response.failed",
                "sequence_number": 1,
                "response": {
                    **created_response,
                    "status": "failed",
                    "error": {
                        "message": "temporary",
                        "type": "server_error",
                        "code": "server_error",
                    },
                },
            }
            events = (created, failed)
        else:
            completed = {
                **_completed_event(sequence=1),
                "response": {
                    **_completed_response(),
                    "id": "resp_attempt_2",
                },
            }
            events = (created, completed)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(*events),
        )

    client = _client(handler)
    request = OpenAIResponsesRequest(model="upstream-model", input="hello")
    try:
        chunks = await _collect_native(_transport(client, max_attempts=2), request)
    finally:
        await client.close()

    body = "".join(chunks)
    events = parse_sse_text(body)
    assert attempts == 2
    assert [event.event for event in events] == [
        "response.created",
        "response.completed",
    ]
    assert "resp_attempt_1" not in body
    assert events[0].data["response"]["id"] == "resp_attempt_2"


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_api", ["messages", "responses"])
async def test_context_failure_event_is_canonical_and_not_retried(
    wire_api: str,
) -> None:
    attempts = 0
    failed = {
        "type": "response.failed",
        "sequence_number": 0,
        "response": {
            **_completed_response(),
            "status": "failed",
            "error": {
                "message": "maximum context reached",
                "type": "invalid_request_error",
                "code": " Context_Length_Exceeded ",
            },
        },
    }

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(failed),
        )

    client = _client(handler)
    try:
        with pytest.raises(ExecutionFailure) as exc_info:
            if wire_api == "messages":
                await _collect(_transport(client, max_attempts=2))
            else:
                await _collect_native(
                    _transport(client, max_attempts=2),
                    OpenAIResponsesRequest(model="upstream-model", input="hello"),
                )
    finally:
        await client.close()

    assert attempts == 1
    assert exc_info.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_api", ["messages", "responses"])
async def test_top_level_context_error_is_canonical_and_not_retried(
    wire_api: str,
) -> None:
    attempts = 0
    failed = {
        "type": "error",
        "sequence_number": 0,
        "code": "context_length_exceeded",
        "message": "maximum context reached",
        "param": None,
    }

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(failed),
        )

    client = _client(handler)
    try:
        with pytest.raises(ExecutionFailure) as exc_info:
            if wire_api == "messages":
                await _collect(_transport(client, max_attempts=2))
            else:
                await _collect_native(
                    _transport(client, max_attempts=2),
                    OpenAIResponsesRequest(model="upstream-model", input="hello"),
                )
    finally:
        await client.close()

    assert attempts == 1
    assert exc_info.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_native_committed_truncation_emits_one_failed_terminal() -> None:
    attempts = 0
    committed = "x" * 70_000
    created = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            **_completed_response(),
            "status": "in_progress",
            "output": [],
            "usage": None,
        },
    }

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(created, _text_delta(committed, sequence=1)),
        )

    client = _client(handler)
    request = OpenAIResponsesRequest(model="upstream-model", input="hello")
    try:
        chunks = await _collect_native(_transport(client, max_attempts=2), request)
    finally:
        await client.close()

    events = parse_sse_text("".join(chunks))
    assert attempts == 1
    assert [event.event for event in events][-1] == "response.failed"
    assert sum(event.event == "response.failed" for event in events) == 1
    assert events[-1].data["response"]["id"] == "resp_test"
    assert events[-1].data["response"]["model"] == "public-model"
    assert events[-1].data["response"]["status"] == "failed"


@pytest.mark.asyncio
async def test_native_response_incomplete_is_a_normal_terminal() -> None:
    incomplete = {
        "type": "response.incomplete",
        "sequence_number": 1,
        "response": {
            **_completed_response(),
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    }
    client = _client(
        lambda _request: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(incomplete),
        )
    )
    request = OpenAIResponsesRequest(model="upstream-model", input="hello")
    try:
        chunks = await _collect_native(_transport(client), request)
    finally:
        await client.close()

    events = parse_sse_text("".join(chunks))
    assert [event.event for event in events] == ["response.incomplete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_api", ["messages", "responses"])
async def test_context_incomplete_reason_is_canonical_failure(wire_api: str) -> None:
    attempts = 0
    incomplete = {
        "type": "response.incomplete",
        "sequence_number": 0,
        "response": {
            **_completed_response(),
            "status": "incomplete",
            "incomplete_details": {"reason": "model_context_window_exceeded"},
        },
    }

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(incomplete),
        )

    client = _client(handler)
    try:
        with pytest.raises(ExecutionFailure) as exc_info:
            if wire_api == "messages":
                await _collect(_transport(client, max_attempts=2))
            else:
                await _collect_native(
                    _transport(client, max_attempts=2),
                    OpenAIResponsesRequest(model="upstream-model", input="hello"),
                )
    finally:
        await client.close()

    assert attempts == 1
    assert exc_info.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_native_response_stops_at_terminal_before_trailing_ping() -> None:
    client = _client(
        lambda _request: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_completed_event(), {"type": "ping"}),
        )
    )
    request = OpenAIResponsesRequest(model="upstream-model", input="hello")
    try:
        chunks = await _collect_native(_transport(client), request)
    finally:
        await client.close()

    events = parse_sse_text("".join(chunks))
    assert [event.event for event in events] == ["response.completed"]


@pytest.mark.asyncio
async def test_standard_responses_preserves_public_fields_and_usage() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        captured.append(payload)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_text_delta("hello"), _completed_event()),
        )

    client = _client(handler)
    try:
        chunks = await _collect(_transport(client))
    finally:
        await client.close()

    assert len(captured) == 1
    assert captured[0]["model"] == "upstream-model"
    assert captured[0]["max_output_tokens"] == 123
    assert captured[0]["metadata"] == {"source": "test"}
    assert captured[0]["store"] is False
    events = parse_sse_text("".join(chunks))
    assert_anthropic_stream_contract(events)
    assert text_content(events) == "hello"
    final_usage = next(
        event.data["usage"] for event in events if event.event == "message_delta"
    )
    assert final_usage == {
        "input_tokens": 5,
        "output_tokens": 2,
        "cache_read_input_tokens": 3,
    }


@pytest.mark.asyncio
async def test_standard_responses_maps_reasoning_and_tool_calls() -> None:
    events: tuple[dict[str, object], ...] = (
        {
            "type": "response.reasoning_summary_text.delta",
            "sequence_number": 0,
            "item_id": "reasoning_1",
            "output_index": 0,
            "summary_index": 0,
            "delta": "thinking",
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "item_tool",
                "call_id": "call_tool",
                "name": "Read",
                "arguments": "",
                "status": "in_progress",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "item_id": "item_tool",
            "output_index": 1,
            "delta": '{"path":"README.md"}',
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 3,
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "item_tool",
                "call_id": "call_tool",
                "name": "Read",
                "arguments": '{"path":"README.md"}',
                "status": "completed",
            },
        },
        _completed_event(sequence=4),
    )

    client = _client(
        lambda _request: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(*events),
        )
    )
    try:
        chunks = await _collect(_transport(client))
    finally:
        await client.close()

    parsed = parse_sse_text("".join(chunks))
    assert_anthropic_stream_contract(parsed)
    assert thinking_content(parsed) == "thinking"
    tool_start = next(
        event.data["content_block"]
        for event in parsed
        if event.event == "content_block_start"
        and event.data["content_block"]["type"] == "tool_use"
    )
    assert tool_start == {
        "type": "tool_use",
        "id": "call_tool",
        "name": "Read",
        "input": {},
    }
    tool_delta = next(
        event.data["delta"]
        for event in parsed
        if event.event == "content_block_delta"
        and event.data["delta"]["type"] == "input_json_delta"
    )
    assert tool_delta["partial_json"] == '{"path":"README.md"}'


@pytest.mark.asyncio
async def test_retryable_open_failure_retries_inside_one_transport() -> None:
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx2.Response(
                500,
                json={"error": {"message": "temporary", "type": "server_error"}},
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_text_delta("recovered"), _completed_event()),
        )

    client = _client(handler)
    try:
        chunks = await _collect(_transport(client))
    finally:
        await client.close()

    assert attempts == 2
    parsed = parse_sse_text("".join(chunks))
    assert_anthropic_stream_contract(parsed)
    assert text_content(parsed) == "recovered"


@pytest.mark.asyncio
async def test_early_truncated_retry_has_one_visible_lifecycle() -> None:
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        body = (
            _sse(_text_delta("discarded"))
            if attempts == 1
            else _sse(_text_delta("kept"), _completed_event())
        )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    client = _client(handler)
    try:
        chunks = await _collect(_transport(client))
    finally:
        await client.close()

    parsed = parse_sse_text("".join(chunks))
    assert attempts == 2
    assert text_content(parsed) == "kept"
    assert "discarded" not in "".join(chunks)
    assert sum(event.event == "message_start" for event in parsed) == 1
    assert sum(event.event == "message_stop" for event in parsed) == 1
    assert_anthropic_stream_contract(parsed)


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_api", ["messages", "responses"])
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize(
    "error_type", [httpx2.ReadError, httpx2.ReadTimeout, httpx2.RemoteProtocolError]
)
async def test_sdk_stream_interruptions_retry_before_commit(
    wire_api: str,
    buffered: bool,
    error_type: type[Exception],
) -> None:
    class InterruptedBody(httpx2.AsyncByteStream):
        async def __aiter__(self):
            if buffered:
                yield _sse(
                    {
                        "type": "response.created",
                        "sequence_number": 0,
                        "response": {**_completed_response(), "status": "in_progress"},
                    }
                ).encode()
            raise error_type("connection interrupted")

    requests = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=InterruptedBody(),
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_text_delta("recovered"), _completed_event()),
        )

    client = _client(handler)
    try:
        transport = _transport(client, max_attempts=2)
        chunks = (
            await _collect(transport)
            if wire_api == "messages"
            else await _collect_native(
                transport, OpenAIResponsesRequest(model="upstream-model", input="hello")
            )
        )
    finally:
        await client.close()
    assert requests == 2
    events = parse_sse_text("".join(chunks))
    terminal = "message_stop" if wire_api == "messages" else "response.completed"
    assert [event.event for event in events].count(terminal) == 1


@pytest.mark.asyncio
async def test_exhausted_5xx_uses_exact_attempt_budget() -> None:
    attempts = 0

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            503,
            json={"error": {"message": "busy", "type": "server_error"}},
        )

    client = _client(handler)
    try:
        with pytest.raises(ExecutionFailure) as exc_info:
            await _collect(_transport(client))
    finally:
        await client.close()

    assert attempts == 5
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_post_commit_truncation_is_not_replayed() -> None:
    attempts = 0
    committed = "x" * 70_000

    def handler(_request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_text_delta(committed)),
        )

    client = _client(handler)
    chunks: list[str] = []
    try:
        with pytest.raises(ExecutionFailure):
            async for chunk in _transport(client).stream_messages(
                _request(),
                input_tokens=1,
                request_id="req_committed",
                response_model="public-model",
                reasoning=REASONING_ON,
            ):
                chunks.extend((chunk,))
    finally:
        await client.close()

    assert attempts == 1
    assert "".join(chunks).count(committed) == 1
    assert "".join(chunks).count("event: message_start") == 1


class _BlockingBody(httpx2.AsyncByteStream):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self):
        self.entered.set()
        await asyncio.Event().wait()
        yield b""

    async def aclose(self) -> None:
        self.closed.set()


@pytest.mark.asyncio
async def test_cancellation_closes_the_sdk_stream() -> None:
    body = _BlockingBody()
    client = _client(
        lambda _request: httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=body,
        )
    )
    task = asyncio.create_task(_collect(_transport(client)))
    await asyncio.wait_for(body.entered.wait(), timeout=5)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(body.closed.wait(), timeout=5)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_startup_rejects_fields_responses_cannot_represent() -> None:
    client = _client(lambda _request: httpx2.Response(500))
    transport = _transport(client)
    request = _request(stop_sequences=["done"])

    try:
        with pytest.raises(InvalidRequestError, match="stop_sequences"):
            transport.stream_messages(
                request,
                input_tokens=0,
                request_id=None,
                response_model=request.model,
                reasoning=REASONING_ON,
            )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_claude_artifact_pattern_is_portable_on_the_sdk_wire():
    captured = []

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_sse(_text_delta("ok"), _completed_event()),
        )

    pattern = r"^[^\p{Cc}\p{Cf}\p{Zl}\p{Zp}]{1,200}$"
    request = _request(
        tools=[
            {
                "name": "Artifact",
                "description": "Test artifact",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "pattern": pattern},
                        "slug": {"type": "string", "pattern": "^[a-z]+$"},
                    },
                },
            }
        ]
    )
    client = _client(handler)
    try:
        await _collect(_transport(client), request)
    finally:
        await client.close()
    assert len(captured) == 1
    properties = captured[0]["tools"][0]["parameters"]["properties"]
    assert properties["slug"]["pattern"] == "^[a-z]+$"
    regex = re.compile(properties["name"]["pattern"])
    assert regex.fullmatch("demo")
    assert not regex.fullmatch("bad\U000e0001name")
    assert request.tools is not None
    assert request.tools[0].input_schema is not None
    assert request.tools[0].input_schema["properties"]["name"]["pattern"] == pattern

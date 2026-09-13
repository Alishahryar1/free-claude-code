"""Native Messages attempts preserve protocol, commitment and resource ownership."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.endpoint import HttpEndpoint
from free_claude_code.providers.http import maybe_await_aclose
from tests.providers.support import immediate_admission


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_error", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        {
            "type": "invalid_request_error",
            "message": "Reasoning is mandatory and cannot be disabled.",
        },
        {
            "type": "invalid_request_error",
            "param": "thinking.type",
            "message": "Value 'disabled' is not supported",
        },
        {
            "type": "invalid_request_error",
            "loc": ["body", "thinking", "type"],
            "message": "Value 'disabled' is not supported",
        },
    ],
    ids=["mandatory", "param", "loc"],
)
async def test_tolerant_classifier_corrects_required_reasoning(stream_error, error):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("thinking", {}).get("type") == "disabled":
            if stream_error:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=Wire([_sse({"type": "error", "error": error})]),
                )
            return httpx.Response(400, json={"error": error})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*_events("<severity>0</severity>"))]),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        output = [
            event
            async for event in _transport(client).stream_messages(
                MessagesRequest.model_validate(
                    {
                        "model": "route",
                        "messages": [{"role": "user", "content": "classify"}],
                        "max_tokens": 64,
                        "thinking": {"type": "disabled"},
                    }
                ),
                endpoint_context=Endpoint(),
                reasoning=ReasoningPolicy.prefer_off(),
            )
        ]
    assert "<severity>0</severity>" in "".join(output)
    assert len(bodies) == 2
    assert bodies[0]["max_tokens"] == 8192
    assert "thinking" not in bodies[1]


class Endpoint:
    def __init__(self) -> None:
        self.refreshes: list[bool] = []
        self.token = "original"
        self.base_url = "https://native.invalid/v1/"

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        self.refreshes.append(force_refresh)
        return HttpEndpoint(
            self.base_url,
            {
                "Authorization": "Bearer fresh"
                if force_refresh
                else f"Bearer {self.token}",
                "Anthropic-Version": "2023-06-01",
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("kind", ["authentication_error", "permission_error"])
async def test_precommit_sse_auth_failure_refreshes_once(
    started: bool, kind: str
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        events = (
            _events()
            if calls > 1
            else [
                *(_events()[:1] if started else []),
                {"type": "error", "error": {"type": kind}},
            ]
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*events)]),
        )

    endpoint = Endpoint()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = parse_sse_text(
            "".join(
                [
                    event
                    async for event in _stream(
                        _transport(client), endpoint, responses=True
                    )
                ]
            )
        )
    assert calls == 2 and endpoint.refreshes == [False, True]
    assert sum(event.event == "response.created" for event in result) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
async def test_unicode_line_characters_remain_inside_sse_text(separator: str) -> None:
    text = f"left{separator}right"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Wire([_sse(*_events(text))]),
            )
        )
    ) as client:
        endpoint = Endpoint()
        output = parse_sse_text(
            "".join(
                [
                    event
                    async for event in _stream(
                        _transport(client), endpoint, responses=True
                    )
                ]
            )
        )
    assert endpoint.refreshes == [False]
    assert output[-1].data["response"]["output"][0]["content"][0]["text"] == text


class Wire(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: list[bytes | Exception],
        *,
        closed: Callable[[], None] | None = None,
        close_gate: asyncio.Event | None = None,
    ) -> None:
        self.chunks = chunks
        self.closed = False
        self._on_close = closed
        self._close_gate = close_gate

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def aclose(self) -> None:
        if self._close_gate is not None:
            await self._close_gate.wait()
        self.closed = True
        if self._on_close is not None:
            self._on_close()


def _sse(*events: JsonObject) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in events
    ).encode()


def _events(text: str = "hello", stop: str = "end_turn") -> list[JsonObject]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "upstream-id",
                "type": "message",
                "model": "native",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
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
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


def _transport(
    client: httpx.AsyncClient, admission: ProviderAdmissionController | None = None
) -> AnthropicMessagesTransport:
    return AnthropicMessagesTransport(
        client=client,
        admission=admission or immediate_admission(max_attempts=2),
        provider_name="TEST",
        replay_scope="test/messages",
        read_timeout_s=3,
    )


def _stream(
    transport: AnthropicMessagesTransport, endpoint: Endpoint, *, responses: bool
) -> AsyncIterator[str]:
    if responses:
        return transport.stream_responses(
            OpenAIResponsesRequest(model="native", input="hi"),
            endpoint_context=endpoint,
            response_model="public",
        )
    return transport.stream_messages(
        MessagesRequest(
            model="native",
            messages=[{"role": "user", "content": "hi"}],
            betas=["test-beta"],
        ),
        endpoint_context=endpoint,
        response_model="public",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("versioned_base", [False, True])
async def test_fragmented_messages_http_keeps_native_path_and_public_identity(
    responses: bool,
    versioned_base: bool,
) -> None:
    raw = _sse(*_events())
    wire = Wire([raw[index : index + 7] for index in range(0, len(raw), 7)])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        endpoint = Endpoint()
        endpoint.base_url = (
            "https://native.invalid/v1/" if versioned_base else "https://native.invalid"
        )
        output = parse_sse_text(
            "".join(
                [
                    event
                    async for event in _stream(
                        _transport(client), endpoint, responses=responses
                    )
                ]
            )
        )
        assert not client.is_closed
    assert wire.closed and len(requests) == 1
    assert requests[0].url.path == "/v1/messages"
    assert requests[0].headers["Authorization"] == "Bearer original"
    assert requests[0].headers["anthropic-version"] == "2023-06-01"
    assert requests[0].headers.get_list("anthropic-version") == ["2023-06-01"]
    body = json.loads(requests[0].content)
    assert body["stream"] is True and body["model"] == "native"
    assert body["max_tokens"] > 0
    if responses:
        assert output[-1].data["response"]["model"] == "public"
        assert output[-1].event == "response.completed"
    else:
        assert requests[0].headers["anthropic-beta"] == "test-beta"
        assert output[0].data["message"]["id"] == "upstream-id"
        assert output[0].data["message"]["model"] == "public"
        assert output[-1].event == "message_stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_unauthorized_refresh_uses_same_budget_and_closes_before_next_request(
    status: int,
) -> None:
    endpoint = Endpoint()
    wires: list[Wire] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if wires:
            assert wires[0].closed
            assert request.headers["Authorization"] == "Bearer fresh"
        wire = Wire([b'{"error":"expired"}' if not wires else _sse(*_events())])
        wires.append(wire)
        return httpx.Response(
            status if len(wires) == 1 else 200,
            headers={"content-type": "text/event-stream"},
            stream=wire,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert [
            item async for item in _stream(_transport(client), endpoint, responses=True)
        ]
    assert endpoint.refreshes == [False, True] and all(wire.closed for wire in wires)


@pytest.mark.asyncio
async def test_repeated_unauthorized_cannot_create_unbounded_auth_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"type": "authentication_error"}})

    endpoint = Endpoint()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ExecutionFailure) as caught:
            _ = [
                event
                async for event in _stream(_transport(client), endpoint, responses=True)
            ]
    assert caught.value.kind is FailureKind.AUTHENTICATION
    assert calls == 2 and endpoint.refreshes == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first",
    [
        b"event: content_block_delta\ndata: {bad}\n\n",
        _sse(*_events()[:3]),
        b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error"}}\n\n',
    ],
)
async def test_early_malformed_truncated_and_overloaded_attempts_retry_invisibly(
    first: bytes,
) -> None:
    calls = 0
    wires: list[Wire] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        wire = Wire([first if calls == 1 else _sse(*_events("final"))])
        wires.append(wire)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = "".join(
            [
                event
                async for event in _stream(
                    _transport(client), Endpoint(), responses=True
                )
            ]
        )
    events = parse_sse_text(result)
    assert calls == 2 and all(wire.closed for wire in wires)
    assert sum(event.event == "response.created" for event in events) == 1
    assert "final" in result and "hello" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("auth", [False, True])
async def test_committed_failure_never_retries_or_emits_success(
    responses: bool,
    auth: bool,
) -> None:
    calls = 0
    wire = Wire(
        [
            _sse(*_events("x" * 70_000)[:3]),
            _sse({"type": "error", "error": {"type": "authentication_error"}})
            if auth
            else httpx.ReadTimeout("stalled"),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    chunks: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        stream = _stream(_transport(client), Endpoint(), responses=responses)
        if responses:
            chunks = [chunk async for chunk in stream]
        else:
            with pytest.raises(ExecutionFailure):
                async for chunk in stream:
                    chunks.append(chunk)
    events = parse_sse_text("".join(chunks))
    assert calls == 1 and wire.closed
    assert not any(
        event.event in {"message_stop", "response.completed"} for event in events
    )
    if responses:
        assert sum(event.event == "response.failed" for event in events) == 1
        assert events[-1].data["response"]["output"][0]["status"] == "incomplete"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", [False, True])
async def test_context_window_stop_is_nonretryable_canonical_failure(
    error_code: bool,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire(
                [
                    _sse(
                        {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "code": "context_length_exceeded",
                            },
                        }
                    )
                    if error_code
                    else _sse(*_events(stop="model_context_window_exceeded"))
                ]
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ExecutionFailure) as caught:
            _ = [
                event
                async for event in _stream(
                    _transport(client), Endpoint(), responses=True
                )
            ]
    assert caught.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED and calls == 1


@pytest.mark.asyncio
async def test_cancelled_consumer_closes_response_before_releasing_admission() -> None:
    gate = asyncio.Event()
    first = Wire([_sse(*_events("x" * 70_000))], close_gate=gate)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert request.headers["Authorization"] == "Bearer new"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=first if calls == 1 else Wire([_sse(*_events())]),
        )

    admission = ProviderAdmissionController(
        provider_name="TEST",
        rate_limit=1_000_000,
        rate_window=1,
        max_concurrency=1,
        max_attempts=2,
        base_delay=0,
        max_delay=0,
        jitter=0,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = _transport(client, admission)
        stream = _stream(transport, Endpoint(), responses=True)
        assert await anext(stream)
        close = asyncio.create_task(maybe_await_aclose(stream))
        next_endpoint = Endpoint()
        next_stream = _stream(transport, next_endpoint, responses=True)
        next_request = asyncio.ensure_future(anext(next_stream))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert calls == 1 and not close.done() and not next_request.done()
        assert next_endpoint.refreshes == []
        next_endpoint.token = "new"
        gate.set()
        await close
        assert await next_request
        await maybe_await_aclose(next_stream)
    assert first.closed and calls == 2


@pytest.mark.asyncio
async def test_task_cancellation_closes_live_http_response() -> None:
    entered = asyncio.Event()

    class BlockingWire(Wire):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield _sse(*_events("x" * 70_000)[:3])
            entered.set()
            await asyncio.Event().wait()

    wire = BlockingWire([])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=wire
            )
        )
    ) as client:
        stream = _stream(_transport(client), Endpoint(), responses=True)

        async def consume() -> None:
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert wire.closed


@pytest.mark.asyncio
async def test_auth_refresh_cannot_exceed_single_attempt_budget() -> None:
    endpoint = Endpoint()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(401, json={"error": "expired"})
        )
    ) as client:
        with pytest.raises(ExecutionFailure):
            _ = [
                event
                async for event in _stream(
                    _transport(client, immediate_admission(max_attempts=1)),
                    endpoint,
                    responses=True,
                )
            ]
    assert endpoint.refreshes == [False]


@pytest.mark.asyncio
async def test_startup_rejects_invalid_conversion_without_request_io() -> None:
    async with httpx.AsyncClient() as client:
        transport = _transport(client)
        with pytest.raises(InvalidRequestError):
            transport.stream_responses(
                OpenAIResponsesRequest(
                    model="native", input=[{"type": "input_file", "file_id": "remote"}]
                ),
                endpoint_context=Endpoint(),
            )

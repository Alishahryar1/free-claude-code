"""Client-owned request lifetime contracts at the pure-ASGI boundary."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import cast

import pytest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import AdminRuntimePort, ApiServices
from free_claude_code.api.request_lifetime import (
    ClientRequestLifetimeMiddleware,
)
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import (
    ProviderPort,
    RequestRuntimeLease,
    RequestRuntimePort,
    TaskController,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy


def _http_scope(
    path: str = "/v1/messages",
    *,
    method: str = "POST",
) -> Scope:
    return cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": None,
            "server": None,
        },
    )


async def _unused_send(_message: Message) -> None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        _http_scope("/health"),
        _http_scope("/v1/messages", method="OPTIONS"),
        _http_scope("/admin/api/code/sessions/session/turns"),
        _http_scope("/admin/api/code/sessions/session/stop"),
        cast(Scope, {"type": "lifespan", "asgi": {"version": "3.0"}}),
    ],
)
async def test_other_scopes_delegate_with_original_callables(
    scope: Scope,
) -> None:
    observed: tuple[Scope, Receive, Send] | None = None

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(_message: Message) -> None:
        return None

    async def app(
        received_scope: Scope,
        received_receive: Receive,
        received_send: Send,
    ) -> None:
        nonlocal observed
        observed = (received_scope, received_receive, received_send)

    await ClientRequestLifetimeMiddleware(cast(ASGIApp, app))(
        scope,
        receive,
        send,
    )

    assert observed == (scope, receive, send)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/v1/messages",
        "/v1/responses",
        "/admin/api/code/folder-picker",
    ],
)
async def test_request_body_frames_are_relayed_once_in_order(path: str) -> None:
    frames: asyncio.Queue[Message] = asyncio.Queue()
    receiver_closed = asyncio.Event()
    first = {
        "type": "http.request",
        "body": b"first",
        "more_body": True,
    }
    second = {
        "type": "http.request",
        "body": b"second",
        "more_body": False,
    }
    await frames.put(first)
    await frames.put(second)

    async def receive() -> Message:
        if not frames.empty():
            return frames.get_nowait()
        try:
            await asyncio.Event().wait()
        finally:
            receiver_closed.set()
        raise AssertionError("unreachable")

    received: list[Message] = []

    async def app(_scope: Scope, app_receive: Receive, _send: Send) -> None:
        received.append(await app_receive())
        received.append(await app_receive())

    await ClientRequestLifetimeMiddleware(cast(ASGIApp, app))(
        _http_scope(path),
        receive,
        _unused_send,
    )

    assert received == [first, second]
    assert receiver_closed.is_set()


@pytest.mark.asyncio
async def test_receive_pump_buffers_only_a_bounded_number_of_frames() -> None:
    receive_calls = 0
    app_started = asyncio.Event()
    stop_app = asyncio.Event()

    async def receive() -> Message:
        nonlocal receive_calls
        receive_calls += 1
        return {
            "type": "http.request",
            "body": str(receive_calls).encode(),
            "more_body": True,
        }

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        app_started.set()
        await stop_app.wait()

    request = asyncio.create_task(
        ClientRequestLifetimeMiddleware(cast(ASGIApp, app))(
            _http_scope(),
            receive,
            _unused_send,
        )
    )
    await app_started.wait()
    for _ in range(5):
        await asyncio.sleep(0)

    assert receive_calls == 2

    stop_app.set()
    await request


type DisconnectScenario = Callable[
    [Scope, Receive, Send],
    Awaitable[None],
]


async def _run_disconnect_scenario(
    app: DisconnectScenario,
    ready: asyncio.Event,
    *,
    path: str = "/v1/messages",
) -> list[Message]:
    allow_disconnect = asyncio.Event()
    body_consumed = asyncio.Event()
    sent: list[Message] = []
    first_receive = True

    async def receive() -> Message:
        nonlocal first_receive
        if first_receive:
            first_receive = False
            body_consumed.set()
            return {
                "type": "http.request",
                "body": b"{}",
                "more_body": False,
            }
        await allow_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    request = asyncio.create_task(
        ClientRequestLifetimeMiddleware(app)(
            _http_scope(path),
            receive,
            send,
        )
    )
    await body_consumed.wait()
    await ready.wait()
    allow_disconnect.set()
    await request
    return sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/v1/messages", "/v1/responses", "/admin/api/code/folder-picker"]
)
async def test_disconnect_before_response_start_cancels_application(path: str) -> None:
    app_started = asyncio.Event()
    app_closed = asyncio.Event()

    async def app(_scope: Scope, app_receive: Receive, _send: Send) -> None:
        await app_receive()
        app_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            app_closed.set()

    sent = await _run_disconnect_scenario(app, app_started, path=path)

    assert sent == []
    assert app_closed.is_set()


@pytest.mark.asyncio
async def test_disconnect_after_response_start_cancels_silent_body() -> None:
    body_started = asyncio.Event()
    app_closed = asyncio.Event()

    async def app(_scope: Scope, app_receive: Receive, send: Send) -> None:
        await app_receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {
                "type": "http.response.body",
                "body": b"first",
                "more_body": True,
            }
        )
        body_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            app_closed.set()

    sent = await _run_disconnect_scenario(app, body_started)

    assert [message["type"] for message in sent] == [
        "http.response.start",
        "http.response.body",
    ]
    assert app_closed.is_set()


@pytest.mark.asyncio
async def test_disconnect_preserves_application_cleanup_failure() -> None:
    app_started = asyncio.Event()

    async def app(_scope: Scope, app_receive: Receive, _send: Send) -> None:
        await app_receive()
        app_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await _run_disconnect_scenario(app, app_started)


@pytest.mark.asyncio
async def test_application_error_wins_and_receive_pump_is_closed() -> None:
    receive_closed = asyncio.Event()

    async def receive() -> Message:
        try:
            await asyncio.Event().wait()
        finally:
            receive_closed.set()
        raise AssertionError("unreachable")

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        raise RuntimeError("application failed")

    with pytest.raises(RuntimeError, match="application failed"):
        await ClientRequestLifetimeMiddleware(cast(ASGIApp, app))(
            _http_scope(),
            receive,
            _unused_send,
        )

    assert receive_closed.is_set()


@pytest.mark.asyncio
async def test_receive_error_cancels_application_and_is_preserved() -> None:
    app_closed = asyncio.Event()

    async def receive() -> Message:
        raise RuntimeError("receive failed")

    async def app(_scope: Scope, app_receive: Receive, _send: Send) -> None:
        try:
            await app_receive()
        finally:
            app_closed.set()

    with pytest.raises(RuntimeError, match="receive failed"):
        await ClientRequestLifetimeMiddleware(cast(ASGIApp, app))(
            _http_scope(),
            receive,
            _unused_send,
        )

    assert app_closed.is_set()


@pytest.mark.asyncio
async def test_outer_cancellation_drains_both_owned_tasks() -> None:
    app_started = asyncio.Event()
    app_cleanup_started = asyncio.Event()
    app_closed = asyncio.Event()
    receive_started = asyncio.Event()
    receive_cleanup_started = asyncio.Event()
    receive_closed = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def receive() -> Message:
        receive_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            receive_cleanup_started.set()
            await allow_cleanup.wait()
            receive_closed.set()
        raise AssertionError("unreachable")

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        app_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            app_cleanup_started.set()
            await allow_cleanup.wait()
            app_closed.set()

    request = asyncio.create_task(
        ClientRequestLifetimeMiddleware(cast(ASGIApp, app))(
            _http_scope(),
            receive,
            _unused_send,
        )
    )
    await app_started.wait()
    await receive_started.wait()

    request.cancel()
    await app_cleanup_started.wait()
    await receive_cleanup_started.wait()
    request.cancel()
    allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert app_closed.is_set()
    assert receive_closed.is_set()


class _ControlledProvider:
    def __init__(self, chunks: tuple[str, ...]) -> None:
        self._chunks = chunks
        self.blocked = asyncio.Event()
        self.closed = asyncio.Event()
        self.request_ids: list[str] = []

    async def stream_messages(
        self,
        _request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        del input_tokens, response_model, reasoning
        self.request_ids.append(request_id)
        try:
            for chunk in self._chunks:
                yield chunk
            self.blocked.set()
            await asyncio.Event().wait()
        finally:
            self.closed.set()

    async def stream_responses(
        self,
        _request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        del input_tokens, response_model, reasoning
        self.request_ids.append(request_id)
        try:
            for chunk in self._chunks:
                yield chunk
            self.blocked.set()
            await asyncio.Event().wait()
        finally:
            self.closed.set()


class _Lease:
    generation_id = 1

    def __init__(self, settings: Settings, provider: ProviderPort) -> None:
        self.settings = settings
        self.model_infos: tuple[ProviderModelInfo, ...] = ()
        self._provider = provider
        self.release_calls = 0

    def is_provider_cached(self, provider_id: str) -> bool:
        del provider_id
        return True

    def resolve_provider(self, provider_id: str) -> ProviderPort:
        assert provider_id == "nvidia_nim"
        return self._provider

    async def release(self) -> None:
        self.release_calls += 1


class _Requests:
    def __init__(self, lease: _Lease) -> None:
        self._lease = lease

    async def acquire(
        self, *, include_model_infos: bool = False
    ) -> RequestRuntimeLease:
        del include_model_infos
        return self._lease

    def current_settings(self) -> Settings:
        return self._lease.settings

    def cached_model_info(
        self,
        provider_id: str,
        model_id: str,
    ) -> ProviderModelInfo | None:
        del provider_id, model_id
        return None

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        return ()


def _message_start() -> str:
    return format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_lifetime",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )


def _response_created() -> str:
    payload = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {
            "id": "resp_lifetime",
            "object": "response",
            "model": "nvidia_nim/test-model",
            "status": "in_progress",
            "output": [],
            "error": None,
            "usage": None,
        },
    }
    return f"event: response.created\ndata: {json.dumps(payload)}\n\n"


def _partial_message_stream() -> tuple[str, ...]:
    return (
        _message_start(),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            },
        ),
    )


async def _disconnect_real_app(
    *,
    path: str,
    payload: dict[str, object],
    provider: _ControlledProvider,
) -> tuple[list[Message], _Lease]:
    settings = Settings(provider_progress_timeout=60.0)
    lease = _Lease(settings, cast(ProviderPort, provider))
    requests = _Requests(lease)
    app = create_app(
        ApiServices(
            requests=cast(RequestRuntimePort, requests),
            admin=cast(AdminRuntimePort, object()),
            tasks=cast(TaskController, object()),
        )
    )
    body = json.dumps(payload).encode()
    scope = _http_scope(path)
    scope["headers"] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    allow_disconnect = asyncio.Event()
    sent: list[Message] = []
    request_pending = True

    async def receive() -> Message:
        nonlocal request_pending
        if request_pending:
            request_pending = False
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }
        await allow_disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    async def run_app() -> None:
        await app(scope, receive, send)

    request = asyncio.create_task(run_app())
    await asyncio.wait_for(provider.blocked.wait(), timeout=1)
    allow_disconnect.set()
    await asyncio.wait_for(request, timeout=1)
    await asyncio.wait_for(provider.closed.wait(), timeout=1)
    return sent, lease


def _messages_payload(*, stream: bool) -> dict[str, object]:
    return {
        "model": "nvidia_nim/test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        "stream": stream,
    }


def _responses_payload() -> dict[str, object]:
    return {
        "model": "nvidia_nim/test-model",
        "input": "Hello",
        "max_output_tokens": 32,
        "stream": True,
    }


@pytest.mark.asyncio
async def test_real_messages_disconnect_before_first_event_releases_lease() -> None:
    provider = _ControlledProvider(())

    sent, lease = await _disconnect_real_app(
        path="/v1/messages",
        payload=_messages_payload(stream=True),
        provider=provider,
    )

    assert sent == []
    assert provider.request_ids[0].startswith("req_")
    assert lease.release_calls == 1


@pytest.mark.asyncio
async def test_real_messages_stream_false_disconnect_discards_partial_body() -> None:
    provider = _ControlledProvider(_partial_message_stream())

    sent, lease = await _disconnect_real_app(
        path="/v1/messages",
        payload=_messages_payload(stream=False),
        provider=provider,
    )

    assert sent == []
    assert lease.release_calls == 1


@pytest.mark.asyncio
async def test_real_messages_post_start_disconnect_has_no_terminal_error() -> None:
    provider = _ControlledProvider(_partial_message_stream())

    sent, lease = await _disconnect_real_app(
        path="/v1/messages",
        payload=_messages_payload(stream=True),
        provider=provider,
    )

    wire = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    ).decode()
    assert any(message["type"] == "http.response.start" for message in sent)
    assert "event: error" not in wire
    assert "message_stop" not in wire
    assert lease.release_calls == 1


@pytest.mark.asyncio
async def test_real_responses_post_start_disconnect_has_no_failed_event() -> None:
    provider = _ControlledProvider((_response_created(),))

    sent, lease = await _disconnect_real_app(
        path="/v1/responses",
        payload=_responses_payload(),
        provider=provider,
    )

    wire = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    ).decode()
    assert any(message["type"] == "http.response.start" for message in sent)
    assert "event: response.created" in wire
    assert "response.failed" not in wire
    assert lease.release_calls == 1

"""Final inference outcomes include failures hidden inside HTTP 200 streams."""

import asyncio
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from free_claude_code.api.request_ids import RequestCorrelationMiddleware
from free_claude_code.api.request_outcomes import RequestOutcomeMiddleware
from free_claude_code.config.settings import Settings
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.request_outcomes import (
    current_request_outcome,
    record_request_route,
)
from tests.api.support import create_test_app
from tests.api.test_request_lifetime import _http_scope


@pytest.fixture
def outcomes():
    records = []
    sink = logger.add(
        lambda message: records.append(message.record),
        level="INFO",
        filter=lambda record: record["extra"].get("event") == "request.completed",
    )
    try:
        yield records
    finally:
        logger.remove(sink)


class OutcomeProvider:
    def __init__(self, result):
        self.result = result
        self.started = asyncio.Event()

    async def stream_messages(self, request, **kwargs):
        async for chunk in self._stream("messages"):
            yield chunk

    async def stream_responses(self, request, **kwargs):
        async for chunk in self._stream("responses"):
            yield chunk

    async def _stream(self, wire_api):
        failure = ExecutionFailure(
            FailureKind.RATE_LIMIT, 429, "private error text", True
        )
        if self.result == "wait_before_start":
            self.started.set()
            await asyncio.Event().wait()
        if self.result == "pre_start":
            raise failure
        if wire_api == "responses":
            yield 'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_test","object":"response","status":"in_progress","output":[]}}\n\n'
        else:
            yield 'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[]}}\n\n'
        if self.result == "exception":
            raise failure
        if self.result == "wait_after_start":
            self.started.set()
            await asyncio.Event().wait()
        if self.result == "wire_error":
            event = "error" if wire_api == "messages" else "response.failed"
            data = {"type": event, "error": {"type": "rate_limit_error"}}
            if wire_api == "responses":
                data = {
                    "type": event,
                    "response": {"status": "failed", "error": data["error"]},
                }
            frame = f"event: {event}\ndata: {json.dumps(data)}\n\n"
            # Framing must work even when a wire event is split between chunks.
            yield frame[:12]
            yield frame[12:]
        else:
            yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'


@pytest.mark.parametrize("wire_api", ["messages", "responses"])
@pytest.mark.parametrize("result", ["success", "pre_start", "exception", "wire_error"])
def test_one_final_outcome_for_streamed_inference(outcomes, wire_api, result):
    payload = {"model": "nvidia_nim/test-model", "stream": True}
    if wire_api == "messages":
        payload.update(
            max_tokens=64, messages=[{"role": "user", "content": "private input"}]
        )
    else:
        payload["input"] = "private input"
    with (
        patch(
            "free_claude_code.api.routes.resolve_provider",
            return_value=OutcomeProvider(result),
        ),
        TestClient(create_test_app(Settings())) as client,
    ):
        response = client.post(f"/v1/{wire_api}", json=payload)
    assert response.status_code == (429 if result == "pre_start" else 200)
    assert len(outcomes) == 1
    record = outcomes[0]
    fields = record["extra"]
    assert record["level"].name == "INFO"
    assert fields["request_id"] == response.headers["request-id"]
    assert fields["provider_id"] == "nvidia_nim"
    assert fields["model"] == "test-model"
    assert fields["wire_api"] == wire_api
    assert fields["status_code"] == response.status_code
    assert fields["outcome"] == ("success" if result == "success" else "failure")
    assert fields["duration_ms"] >= 0
    assert (
        fields["failure_reason"]
        == {
            "success": None,
            "pre_start": "rate_limit",
            "exception": "rate_limit",
            "wire_error": "rate_limit_error",
        }[result]
    )
    assert "private input" not in str(record)
    assert "private error text" not in str(record)


def test_fallback_records_selected_provider_without_logging_failed_attempt(outcomes):
    settings = Settings(model_fallbacks=["open_router/fallback-model"])
    providers = {
        "nvidia_nim": OutcomeProvider("pre_start"),
        "open_router": OutcomeProvider("success"),
    }
    with (
        patch(
            "free_claude_code.api.routes.resolve_provider",
            side_effect=lambda name, **kwargs: providers[name],
        ),
        TestClient(create_test_app(settings)) as client,
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "nvidia_nim/test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
    assert response.status_code == 200
    assert len(outcomes) == 1
    fields = outcomes[0]["extra"]
    assert (
        fields["provider_id"],
        fields["model"],
        fields["outcome"],
        fields["failure_reason"],
    ) == (
        "open_router",
        "fallback-model",
        "success",
        None,
    )


@pytest.mark.parametrize("result", ["success", "exception", "wire_error"])
def test_non_streaming_messages_have_one_outcome(outcomes, result):
    with (
        patch(
            "free_claude_code.api.routes.resolve_provider",
            return_value=OutcomeProvider(result),
        ),
        TestClient(create_test_app(Settings())) as client,
        patch(
            "free_claude_code.api.request_outcomes.monotonic", side_effect=[10.0, 12.5]
        ),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "nvidia_nim/test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
    assert response.status_code == (200 if result == "success" else 429)
    assert len(outcomes) == 1
    assert outcomes[0]["extra"]["duration_ms"] == 2500
    assert outcomes[0]["extra"]["outcome"] == (
        "success" if result == "success" else "failure"
    )


@pytest.mark.parametrize(
    "payload,status,reason",
    [
        ({}, 422, "http_422"),
        (
            {"model": "nvidia_nim/test-model", "input": "hello", "stream": False},
            400,
            "invalid_request",
        ),
    ],
)
def test_rejections_before_execution_are_recorded(outcomes, payload, status, reason):
    with TestClient(create_test_app(Settings())) as client:
        response = client.post("/v1/responses", json=payload)
    assert response.status_code == status
    assert len(outcomes) == 1
    fields = outcomes[0]["extra"]
    assert fields["outcome"] == "failure"
    assert fields["failure_reason"] == reason
    assert fields["provider_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_api", ["messages", "responses"])
@pytest.mark.parametrize("stage", ["wait_before_start", "wait_after_start"])
async def test_disconnect_logs_cancellation_after_cleanup(outcomes, wire_api, stage):
    provider = OutcomeProvider(stage)
    frames = asyncio.Queue()
    payload = {"model": "nvidia_nim/test-model", "stream": True}
    payload.update(
        {"messages": [{"role": "user", "content": "hello"}]}
        if wire_api == "messages"
        else {"input": "hello"}
    )
    await frames.put({"type": "http.request", "body": json.dumps(payload).encode()})
    sent = []

    async def send(message):
        sent.append(message)

    with patch("free_claude_code.api.routes.resolve_provider", return_value=provider):
        app = create_test_app(Settings())
        scope = _http_scope(f"/v1/{wire_api}")
        scope["headers"] = [(b"content-type", b"application/json")]
        task = asyncio.create_task(app(scope, frames.get, send))
        try:
            async with asyncio.timeout(5):
                await provider.started.wait()
                assert outcomes == []
                await frames.put({"type": "http.disconnect"})
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert len(outcomes) == 1
    fields = outcomes[0]["extra"]
    assert fields["outcome"] == "cancelled"
    assert fields["status_code"] == (200 if stage == "wait_after_start" else None)
    assert fields["provider_id"] == "nvidia_nim"
    assert current_request_outcome.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,method",
    [
        ("/health", "GET"),
        ("/v1/models", "GET"),
        ("/admin/api/status", "GET"),
        ("/v1/messages/count_tokens", "POST"),
        ("/v1/messages", "OPTIONS"),
    ],
)
async def test_other_endpoints_do_not_emit_inference_outcomes(outcomes, path, method):
    async def app(scope, receive, send):
        assert current_request_outcome.get() is None
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"{}"})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    await RequestOutcomeMiddleware(app)(_http_scope(path, method=method), receive, send)
    assert outcomes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending", ["send_failure", "late_disconnect", "exception", "cancelled"]
)
async def test_delivery_outcomes_preserve_control_flow(outcomes, ending):
    async def app(scope, receive, send):
        if ending == "exception":
            raise ValueError("private exception message")
        if ending == "cancelled":
            raise asyncio.CancelledError()
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"{}"})
        await receive()

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if ending == "send_failure" and message["type"] == "http.response.body":
            raise OSError("connection closed")

    middleware = RequestCorrelationMiddleware(RequestOutcomeMiddleware(app))
    if ending == "late_disconnect":
        await middleware(_http_scope(), receive, send)
    else:
        with pytest.raises(
            {
                "exception": ValueError,
                "cancelled": asyncio.CancelledError,
                "send_failure": OSError,
            }[ending]
        ):
            await middleware(_http_scope(), receive, send)
    assert len(outcomes) == 1
    fields = outcomes[0]["extra"]
    assert (
        fields["outcome"]
        == {
            "send_failure": "cancelled",
            "late_disconnect": "success",
            "exception": "failure",
            "cancelled": "cancelled",
        }[ending]
    )
    assert fields["failure_reason"] == ("ValueError" if ending == "exception" else None)
    assert "private exception message" not in str(outcomes)
    assert current_request_outcome.get() is None


@pytest.mark.asyncio
async def test_wire_observation_preserves_split_utf8_body(outcomes):
    frame = 'event: error\r\ndata: {"type":"error","error":{"type":"api_error","message":"☃"}}\r\n\r\n'.encode()
    chunks = [frame[index : index + 2] for index in range(0, len(frame), 2)]
    sent = []

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for chunk in chunks:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await RequestCorrelationMiddleware(RequestOutcomeMiddleware(app))(
        _http_scope(), receive, send
    )
    assert b"".join(message.get("body", b"") for message in sent) == frame
    assert len(outcomes) == 1
    assert outcomes[0]["extra"]["outcome"] == "failure"
    assert outcomes[0]["extra"]["failure_reason"] == "api_error"
    assert "☃" not in str(outcomes)


@pytest.mark.asyncio
async def test_overlapping_requests_keep_separate_outcomes(outcomes):
    ready = asyncio.Event()
    count = 0

    async def app(scope, receive, send):
        nonlocal count
        name = scope["query_string"].decode()
        record_request_route(name, f"{name}-model")
        count += 1
        if count == 2:
            ready.set()
        await ready.wait()
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"{}"})

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        pass

    middleware = RequestCorrelationMiddleware(RequestOutcomeMiddleware(app))
    scopes = [_http_scope(), _http_scope()]
    for scope, name in zip(scopes, [b"first", b"second"], strict=True):
        scope["query_string"] = name
    async with asyncio.timeout(5):
        await asyncio.gather(*(middleware(scope, receive, send) for scope in scopes))
    assert len(outcomes) == 2
    assert {
        (row["extra"]["provider_id"], row["extra"]["model"]) for row in outcomes
    } == {
        ("first", "first-model"),
        ("second", "second-model"),
    }
    assert len({row["extra"]["request_id"] for row in outcomes}) == 2
    assert current_request_outcome.get() is None

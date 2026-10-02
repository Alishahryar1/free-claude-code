"""Native commitment, cleanup, and count/continue workflows across API layers."""

import asyncio
import json
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from tests.api.support import create_test_app, provider_manager_for_app
from tests.api.test_native_token_count import count_body
from tests.api.test_request_lifetime import _http_scope
from tests.providers.test_anthropic_messages_transport import Wire, _events, _sse
from tests.providers.test_anthropic_provider import native_body, provider


def test_only_leading_pings_then_all_candidates_fail_returns_json_error():
    calls = []
    wires = []

    def handle(request):
        calls.append(json.loads(request.content)["model"])
        wire = Wire(
            [
                _sse(
                    {"type": "ping"},
                    {
                        "type": "error",
                        "error": {"type": "overloaded_error", "message": "busy"},
                        "request_id": "upstream-evidence",
                    },
                )
            ]
        )
        wires.append(wire)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    app = create_test_app(
        Settings(MODEL="anthropic/primary", MODEL_FALLBACKS=["anthropic/backup"]),
        providers={"anthropic": provider(handle)},
    )
    with TestClient(app) as client:
        response = client.post("/v1/messages", json=native_body(True))
    assert response.status_code == 529, response.text
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-should-retry"] == "false"
    assert response.json()["error"]["type"] == "overloaded_error"
    assert "upstream-evidence" in response.text
    assert calls == ["primary", "primary", "backup", "backup"]
    assert all(wire.closed for wire in wires)
    assert provider_manager_for_app(app)._current.active_leases == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [False, True])
async def test_leading_ping_timeout_or_disconnect_releases_all_resources(disconnect):
    entered = asyncio.Event()
    calls = []

    class Pings(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            for _ in range(100):
                entered.set()
                yield _sse({"type": "ping"})
                await asyncio.sleep(0.005)
            raise httpx.ReadError("No message started")

        async def aclose(self):
            self.closed = True

    wire = Pings()

    def handle(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    p = provider(handle)
    p._admission = ProviderAdmissionController(
        provider_name="native-recovery-test",
        rate_limit=1_000_000,
        rate_window=1.0,
        max_concurrency=1,
        max_attempts=2,
        base_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
    )
    app = create_test_app(
        Settings(
            MODEL="anthropic/primary",
            MODEL_FALLBACKS=["anthropic/backup"],
            provider_progress_timeout=0.05,
        ),
        providers={"anthropic": p},
    )
    manager = provider_manager_for_app(app)
    try:
        if disconnect:
            body = json.dumps(native_body(True)).encode()
            scope = _http_scope()
            scope["headers"] = [(b"content-type", b"application/json")]
            sent = []
            pending = True

            async def receive():
                nonlocal pending
                if pending:
                    pending = False
                    return {"type": "http.request", "body": body, "more_body": False}
                await entered.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                sent.append(message)

            await asyncio.wait_for(app(scope, receive, send), 2)
            assert sent == []
        else:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                response = await asyncio.wait_for(
                    client.post("/v1/messages", json=native_body(True)), 2
                )
            assert response.status_code == 504, response.text
            assert response.json()["error"]["type"] == "timeout_error"
            assert response.headers["x-should-retry"] == "false"
        assert len(calls) == 1
        assert wire.closed
        assert manager._current.active_leases == 0
        # With concurrency one, a new operation can finish only if the old permit was released.
        execution = p._admission.start_execution()

        async def next_call():
            return "released"

        assert (
            await asyncio.wait_for(
                execution.run_call(
                    next_call, operation_kind=ProviderOperationKind.GENERATION
                ),
                1,
            )
            == "released"
        )
    finally:
        await manager.close()


def test_count_generate_and_continue_native_tool_history_with_recovery():
    sent = []
    result_block = {
        "type": "future_hosted_result",
        "tool_use_id": "srvtoolu_new",
        "content": {"payload": "hosted tool response " * 30},
    }
    events = _events(stop="pause_turn")
    events[1]["content_block"] = result_block
    del events[2]

    def handle(request):
        body = json.loads(request.content)
        sent.append(body)
        if len(sent) == 1:
            wire = Wire([_sse({"type": "ping"}), httpx.ReadError("retry before start")])
        elif body.get("stream"):
            wire = Wire([_sse({"type": "ping"}, *events)])
        else:
            return httpx.Response(
                200,
                json={"type": "message", "model": "primary", "content": [result_block]},
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    app = create_test_app(
        Settings(MODEL="anthropic/primary", ENABLE_WEB_SERVER_TOOLS=False),
        providers={"anthropic": provider(handle)},
    )
    body = {**count_body(), "max_tokens": 512, "stream": True}
    with TestClient(app) as client:
        with patch(
            "free_claude_code.runtime.provider_manager.ProviderGenerationLease.resolve_provider",
            side_effect=AssertionError("local count"),
        ):
            initial_count = client.post("/v1/messages/count_tokens", json=body)
        response = client.post("/v1/messages", json=body)
        assert initial_count.status_code == 200, initial_count.text
        assert response.status_code == 200, response.text
        output = parse_sse_text(response.text)
        assert output[0].event == "message_start"
        assert output[0].data["message"]["model"] == body["model"]
        returned = next(
            event.data["content_block"]
            for event in output
            if event.event == "content_block_start"
        )
        assert returned == result_block
        history = body["messages"]
        assert isinstance(history, list)
        continuation = {
            **body,
            "stream": False,
            "messages": [
                *history,
                {"role": "assistant", "content": [returned]},
                {"role": "user", "content": "continue"},
            ],
        }
        with patch(
            "free_claude_code.runtime.provider_manager.ProviderGenerationLease.resolve_provider",
            side_effect=AssertionError("local count"),
        ):
            next_count = client.post("/v1/messages/count_tokens", json=continuation)
        continued = client.post("/v1/messages", json=continuation)
    assert next_count.status_code == 200, next_count.text
    assert next_count.json()["input_tokens"] > initial_count.json()["input_tokens"]
    assert continued.status_code == 200, continued.text
    assert continued.json()["content"] == [result_block]
    assert continued.json()["model"] == body["model"]
    assert sent == [
        {**body, "model": "primary"},
        {**body, "model": "primary"},
        {**continuation, "model": "primary"},
    ]
    assert provider_manager_for_app(app)._current.active_leases == 0

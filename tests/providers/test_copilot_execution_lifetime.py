"""Provider policy observes failures through the application execution boundary."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import patch

import httpx
import pytest

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.web_tools.responses import ResponsesWebTools
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses.web_history import prepare_web_history
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.core.sse import parse_sse_text
from free_claude_code.providers.github_copilot.types import (
    CopilotAuthenticationRequired,
    CopilotEgress,
    CopilotUnavailable,
)
from tests.application.test_execution import (
    ResponsesFakeProvider,
    _routed_responses_request,
    _target,
)
from tests.application.test_responses_web_workflow import WebClient, request
from tests.providers.test_anthropic_messages_transport import _events, _sse
from tests.providers.test_github_copilot_provider import Harness, collect


def executor_stream(
    harness,
    *,
    timeout=60,
    web_client=None,
    tools: tuple[JsonObject, ...] | list[JsonObject] = (),
    fallback=None,
):
    routed = _routed_responses_request()
    routed = replace(
        routed,
        request=routed.request.model_copy(update={"model": harness.runtime.name}),
        resolved=replace(
            routed.resolved,
            primary=_target("github_copilot", harness.runtime.name),
            fallbacks=(_target("fallback", "fallback-model"),) if fallback else (),
        ),
        reasoning=DEFAULT_REASONING_POLICY,
    )
    routed.request.tools = list(tools)
    if web_client is not None:
        routed.request.tools.append({"type": "web_search"})

    async def resolve(provider_id):
        if provider_id == "fallback":
            assert harness.provider._active == 0
            assert not harness.auth.is_connected()
            return fallback
        return harness.provider

    return ProviderExecutor(
        resolve,
        progress_timeout_seconds=timeout,
        responses_token_counter=lambda _: 1,
        responses_web_tools=ResponsesWebTools(web_client),
    ).stream_responses(routed, raw_log_payload={}, request_id="copilot-lifetime")


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
@pytest.mark.parametrize("status", [200, 400, 401])
async def test_executor_preserves_copilot_account_failure_policy(
    tmp_path, egress, status
):
    harness = Harness(tmp_path, egress)
    harness.statuses = [status, status] if status == 401 else [status]
    try:
        if status == 200:
            assert "ok" in await collect(executor_stream(harness))
        else:
            with pytest.raises(ExecutionFailure) as raised:
                await collect(executor_stream(harness))
            assert raised.value.status_code == status
        assert harness.auth.is_connected() is (status != 401)
        state = json.loads((tmp_path / "copilot.json").read_text(encoding="utf-8"))
        assert state["enabled"] is (status != 401)
        if status == 401:
            message = harness.auth.status().message
            assert message is not None and "Reconnect" in message
        assert len(harness.seen) == (2 if status == 401 else 1)
        assert harness.provider._active == 0
        assert all(wire.closed for wire in harness.wires)
    finally:
        await harness.provider.cleanup()
        await harness.auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
async def test_copilot_disconnects_before_executor_selects_fallback(tmp_path, egress):
    harness = Harness(tmp_path, egress)
    harness.statuses = [401, 401]
    fallback = ResponsesFakeProvider()
    try:
        assert "response.completed" in await collect(
            executor_stream(harness, fallback=fallback)
        )
        assert len(fallback.stream_calls) == 1
        assert len(harness.seen) == 2
        assert not harness.auth.is_connected()
    finally:
        await harness.provider.cleanup()
        await harness.auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("authentication", [False, True])
async def test_executor_preserves_sdk_entry_and_refresh_failure_policy(
    tmp_path, monkeypatch, egress, refresh, authentication
):
    harness = Harness(tmp_path, egress)
    error = (CopilotAuthenticationRequired if authentication else CopilotUnavailable)(
        "SDK unavailable"
    )
    try:
        if refresh:
            async with harness.auth.lease(harness.runtime.name):
                pass

            async def rejected_endpoint():
                raise error

            monkeypatch.setattr(
                harness.runtime.sessions[0], "endpoint", rejected_endpoint
            )
            harness.statuses = [401]
        else:
            harness.runtime.model_error = error
        with pytest.raises(ExecutionFailure) as raised:
            await collect(executor_stream(harness))
        assert raised.value.status_code == (401 if authentication else 503)
        assert harness.auth.is_connected() is not authentication
        assert len(harness.seen) == int(refresh)
        assert harness.provider._active == 0
    finally:
        await harness.provider.cleanup()
        await harness.auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("egress", list(CopilotEgress))
async def test_executor_repeated_cancellation_drains_before_disconnect_and_shutdown(
    tmp_path, egress
):
    harness = Harness(tmp_path, egress)
    harness.block_read = harness.block_close = True
    operation = asyncio.create_task(collect(executor_stream(harness)))
    cleanup = []
    try:
        async with asyncio.timeout(5):
            while not harness.wires:
                await asyncio.sleep(0)
            wire = harness.wires[0]
            await wire.read_entered.wait()
            operation.cancel()
            await wire.close_entered.wait()
            operation.cancel()
            cleanup = [
                asyncio.create_task(harness.auth.disconnect()),
                asyncio.create_task(harness.provider.cleanup()),
            ]
            for _ in range(8):
                await asyncio.sleep(0)
            assert not operation.done() and not any(task.done() for task in cleanup)
            assert harness.provider._active == 1
            assert not harness.runtime.sessions[0].closed
            wire.close_gate.set()
            with pytest.raises(asyncio.CancelledError):
                await operation
            await asyncio.gather(*cleanup)
            assert wire.closed and harness.provider._active == 0
    finally:
        operation.cancel()
        for wire in harness.wires:
            wire.read_gate.set()
            wire.close_gate.set()
        await asyncio.gather(operation, *cleanup, return_exceptions=True)
        await harness.provider.cleanup()
        await harness.auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("through_executor", [False, True])
@pytest.mark.parametrize("status", [200, 401])
async def test_owned_client_close_error_preserves_result_and_auth_state(
    tmp_path, through_executor, status
):
    harness = Harness(tmp_path, CopilotEgress.MESSAGES)
    harness.statuses = [status, status]
    original_close = httpx.AsyncClient.aclose
    closed = []

    async def failing_close(client):
        await original_close(client)
        closed.append(client)
        raise RuntimeError("request client close failed")

    try:
        stream = executor_stream(harness) if through_executor else harness.stream(True)
        with patch.object(httpx.AsyncClient, "aclose", failing_close):
            if status == 200:
                assert "ok" in await collect(stream)
            else:
                with pytest.raises(ExecutionFailure) as raised:
                    await collect(stream)
                assert raised.value.status_code == 401
        assert len(closed) == 1 and closed[0].is_closed
        assert harness.auth.is_connected() is (status != 401)
        assert harness.provider._active == 0
    finally:
        await harness.provider.cleanup()
        await harness.auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("through_executor", [False, True])
async def test_owned_client_close_error_preserves_external_cancellation(
    tmp_path, through_executor
):
    harness = Harness(tmp_path, CopilotEgress.MESSAGES)
    harness.block_read = True
    original_close = httpx.AsyncClient.aclose

    async def failing_close(client):
        await original_close(client)
        raise RuntimeError("request client close failed")

    stream = executor_stream(harness) if through_executor else harness.stream(True)
    operation = asyncio.create_task(collect(stream))
    try:
        with patch.object(httpx.AsyncClient, "aclose", failing_close):
            async with asyncio.timeout(5):
                while not harness.wires:
                    await asyncio.sleep(0)
                await harness.wires[0].read_entered.wait()
                operation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await operation
        assert harness.auth.is_connected()
        assert harness.provider._active == 0
        assert all(wire.closed for wire in harness.wires)
    finally:
        operation.cancel()
        for wire in harness.wires:
            wire.read_gate.set()
        await asyncio.gather(operation, return_exceptions=True)
        await harness.provider.cleanup()
        await harness.auth.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["authentication", "timeout", "cancel", "complete"])
@pytest.mark.parametrize("cleanup_error", [False, True])
async def test_web_continuation_preserves_outcome_through_owned_client_cleanup(
    tmp_path, monkeypatch, outcome, cleanup_error
):
    harness = Harness(tmp_path, CopilotEgress.MESSAGES)
    web_client = WebClient()
    timeouts = []
    original_close = httpx.AsyncClient.aclose
    ordinary_messages = harness.messages

    def controlled_timeout(deadline):
        timeout = asyncio.timeout(None)
        timeouts.append(timeout)
        return timeout

    async def close(client):
        await original_close(client)
        if cleanup_error:
            raise RuntimeError("request client close failed")

    def messages(incoming):
        if harness.seen:
            harness.messages_content = _sse(*_events("Finished."))
            harness.block_read = outcome in {"timeout", "cancel"}
            if outcome == "authentication":
                harness.statuses = [401]
        else:
            name = json.loads(incoming.content)["tools"][0]["name"]
            harness.messages_content = _sse(
                _events()[0],
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "search",
                        "name": name,
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"action":"search","query":"docs"}',
                    },
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 4},
                },
                {"type": "message_stop"},
            )
        return ordinary_messages(incoming)

    monkeypatch.setattr(harness.provider._messages_pool, "handler", messages)
    monkeypatch.setattr(
        "free_claude_code.application.execution.asyncio.timeout_at", controlled_timeout
    )
    events = []

    async def consume():
        async for frame in executor_stream(harness, web_client=web_client):
            events.extend(parse_sse_text(frame))

    operation = asyncio.create_task(consume())
    try:
        with patch.object(httpx.AsyncClient, "aclose", close):
            async with asyncio.timeout(5):
                if outcome in {"timeout", "cancel"}:
                    while len(harness.wires) < 2:
                        await asyncio.sleep(0)
                    await harness.wires[-1].read_entered.wait()
                    if outcome == "timeout":
                        timeouts[-1].reschedule(asyncio.get_running_loop().time())
                    else:
                        operation.cancel()
                if outcome == "cancel":
                    with pytest.raises(asyncio.CancelledError):
                        await operation
                else:
                    await operation
        terminals = [
            event.data["response"]
            for event in events
            if event.event
            in {"response.completed", "response.failed", "response.incomplete"}
        ]
        if outcome == "cancel":
            assert terminals == []
        else:
            assert len(terminals) == 1
            final = terminals[0]
            assert final["status"] == (
                "completed" if outcome == "complete" else "failed"
            )
            if outcome != "complete":
                assert (
                    final["error"]["type"]
                    == {
                        "authentication": "authentication_error",
                        "timeout": "timeout_error",
                    }[outcome]
                )
            assert final["output"][0]["type"] == "web_search_call"
            assert final["output"][0]["status"] == "completed"
            retained = prepare_web_history(request(input=final["output"]))
            assert len(retained.records) == 1
        assert web_client.searches == ["docs"]
        assert harness.auth.is_connected() is (outcome != "authentication")
        assert harness.provider._active == 0
        assert all(wire.closed for wire in harness.wires)
    finally:
        operation.cancel()
        for wire in harness.wires:
            wire.read_gate.set()
        await asyncio.gather(operation, return_exceptions=True)
        await harness.provider.cleanup()
        await harness.auth.close()

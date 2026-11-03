import asyncio
import json
from unittest.mock import MagicMock

import pytest

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.ports import ProviderPort
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import Message, MessagesRequest
from free_claude_code.core.failures import ExecutionFailure, FailureKind


def policy_settings(**updates):
    return Settings.model_validate(
        {
            "ROUTING_MODE": "auto",
            "ROUTING_FAILOVER_ENABLED": True,
            "ROUTING_CANDIDATES": json.dumps(
                [
                    {
                        "model": f"ollama/{name}",
                        "context_tokens": 32000,
                        "supports_tools": True,
                        "supports_thinking": True,
                    }
                    for name in ["first", "second"]
                ]
            ),
            **updates,
        }
    )


def request():
    return MessagesRequest(
        model="claude-sonnet-4",
        max_tokens=100,
        messages=[Message(role="user", content="private")],
    )


def stream(settings, emit, calls, *, startup_wait=False):
    provider = MagicMock(spec=ProviderPort)
    provider.stream_messages.side_effect = emit

    async def resolve(provider_id, key):
        calls.append(provider_id)
        if startup_wait:
            await asyncio.Event().wait()
        return provider

    routed = ModelRouter(settings).resolve_messages_request(request())
    return ProviderExecutor(resolve, progress_timeout_seconds=60).stream_messages(
        routed, raw_log_payload=lambda: {}, request_id="policy-test"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_policy_only_tries_alternate_before_any_event(committed):
    models, providers = [], []

    async def emit(req, **kwargs):
        models.append(req.model)
        if req.model == "first":
            if committed:
                yield ": heartbeat\n\n"
            raise ExecutionFailure(FailureKind.RATE_LIMIT, 429, "temporary", True)
        yield "event: message_stop\ndata: {}\n\n"

    if committed:
        with pytest.raises(ExecutionFailure):
            _ = [chunk async for chunk in stream(policy_settings(), emit, providers)]
        assert models == ["first"]
    else:
        _ = [chunk async for chunk in stream(policy_settings(), emit, providers)]
        assert models == ["first", "second"]


@pytest.mark.asyncio
async def test_nontransient_failure_does_not_use_fallback():
    models, providers = [], []

    async def emit(req, **kwargs):
        models.append(req.model)
        raise ExecutionFailure(FailureKind.AUTHENTICATION, 401, "invalid", False)
        yield ""

    with pytest.raises(ExecutionFailure):
        _ = [chunk async for chunk in stream(policy_settings(), emit, providers)]
    assert models == ["first"]


@pytest.mark.asyncio
async def test_total_policy_deadline_includes_provider_initialization():
    async def emit(req, **kwargs):
        yield ""

    providers = []
    settings = policy_settings(ROUTING_DEADLINE_SECONDS=0.01)
    with pytest.raises(ExecutionFailure) as raised:
        _ = [
            chunk
            async for chunk in stream(settings, emit, providers, startup_wait=True)
        ]
    assert raised.value.kind is FailureKind.TIMEOUT
    assert providers == ["ollama"]


@pytest.mark.asyncio
async def test_first_event_timeout_closes_attempt_before_next_candidate():
    closed, models, providers = [], [], []

    async def emit(req, **kwargs):
        models.append(req.model)
        try:
            if req.model == "first":
                await asyncio.Event().wait()
            yield "event: message_stop\ndata: {}\n\n"
        finally:
            closed.append(req.model)

    settings = policy_settings(ROUTING_FIRST_EVENT_TIMEOUT_SECONDS=0.01)
    _ = [chunk async for chunk in stream(settings, emit, providers)]
    assert models == closed == ["first", "second"]


def test_strict_provider_context_disables_hidden_recovery():
    from free_claude_code.core.stream_delivery import bind_strict_routing
    from free_claude_code.providers.stream_recovery import (
        RecoveryController,
        RecoveryFailureAction,
        can_retry_undelivered_stream,
    )

    with bind_strict_routing(True):
        recovery = RecoveryController()
        decision = recovery.advance_failure(
            retryable=True,
            stream_opened=True,
            generated_output=True,
            complete_tool_salvageable=False,
            attempts_remaining=2,
        )
        assert decision.action is RecoveryFailureAction.FINAL_ERROR
        assert not can_retry_undelivered_stream(
            None, retryable=True, attempts_remaining=2, normal_stop_seen=False
        )


@pytest.mark.asyncio
async def test_closes_yielded_policy_iterator_before_releasing_provider():
    closed = []

    async def emit(req, **kwargs):
        try:
            yield ": first\n\n"
            yield ": second\n\n"
        finally:
            closed.append(req.model)

    body = stream(policy_settings(), emit, [])
    assert await anext(body) == ": first\n\n"
    await body.aclose()
    assert closed == ["first"]


@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.asyncio
async def test_expired_absolute_deadline_rejects_buffered_event_or_eof(tail):
    async def emit(req, **kwargs):
        yield ": first\n\n"
        if tail:
            yield ": buffered\n\n"

    body = stream(policy_settings(ROUTING_DEADLINE_SECONDS=0.01), emit, [])
    try:
        assert await anext(body) == ": first\n\n"
        await asyncio.sleep(0.02)
        with pytest.raises(ExecutionFailure) as raised:
            await anext(body)
        assert raised.value.kind is FailureKind.TIMEOUT
    finally:
        await body.aclose()


def test_strict_context_blocks_shared_public_recovery_path(monkeypatch):
    from types import SimpleNamespace

    from free_claude_code.core.stream_delivery import bind_strict_routing
    from free_claude_code.core.stream_recovery import ContinuationSeed, RecoveryAction
    from free_claude_code.providers.continuation import public_recovery

    monkeypatch.setattr(
        "free_claude_code.providers.continuation.assess_recovery",
        lambda **kwargs: SimpleNamespace(
            action=RecoveryAction.CONTINUE,
            seed=ContinuationSeed("partial", ""),
            reason="test",
        ),
    )
    execution = MagicMock()
    execution.delivery.content_released = True
    request = MagicMock()
    request.build.return_value = {}
    with bind_strict_routing(True):
        assert (
            public_recovery(
                execution,
                body={},
                request=request,
                retryable=True,
                normal_stop_seen=False,
                source=MagicMock(),
            )
            is None
        )

"""Discovery, account binding and response-owned endpoint lease contracts."""

import asyncio
import time
from dataclasses import replace

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.providers.github_copilot.broker import CopilotBroker
from free_claude_code.providers.github_copilot.types import (
    CopilotAuthenticationRequired,
    CopilotEgress,
    CopilotIdentity,
    CopilotUnavailable,
)
from free_claude_code.providers.history_replay import replay_origin
from tests.providers.copilot_support import FakeRuntime, model


@pytest.mark.asyncio
async def test_cold_discovery_coalesces_and_cancelled_waiter_does_not_cancel_owner() -> (
    None
):
    runtime = FakeRuntime()
    runtime.model_gate.clear()
    broker = CopilotBroker(runtime)
    first = asyncio.create_task(broker.snapshot())
    await runtime.model_entered.wait()
    second = asyncio.create_task(broker.snapshot())
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    runtime.model_gate.set()
    assert list(await second) == ["model"]
    assert runtime.start_calls == runtime.model_calls == 1
    await broker.close()


@pytest.mark.asyncio
async def test_refresh_failure_preserves_last_good_snapshot() -> None:
    runtime = FakeRuntime()
    broker = CopilotBroker(runtime)
    previous = await broker.snapshot()
    runtime.model_error = CopilotUnavailable("offline")
    with pytest.raises(CopilotUnavailable):
        await broker.snapshot(refresh=True)
    assert await broker.snapshot() is previous
    await broker.close()


@pytest.mark.asyncio
async def test_concurrent_leases_reuse_session_and_refresh_known_expiry() -> None:
    runtime = FakeRuntime()
    broker = CopilotBroker(runtime)
    async with broker.lease("model") as first, broker.lease("model") as second:
        assert runtime.session_calls == 1
        session = runtime.sessions[0]
        await first.endpoint()
        await second.endpoint()
        assert session.endpoint_calls == 1
        session.value = replace(session.value, expires_at=time.time() + 10)
        await first.endpoint(force_refresh=True)
        await second.endpoint()
        assert session.endpoint_calls == 3
    await broker.close()
    assert session.closed


@pytest.mark.asyncio
async def test_replay_origin_survives_copilot_token_refresh() -> None:
    runtime = FakeRuntime()
    broker = CopilotBroker(runtime)
    async with broker.lease("model") as lease:
        first = await lease.endpoint()
        session = runtime.sessions[0]
        session.value = replace(
            session.value,
            http=replace(
                session.value.http,
                api_key="refreshed-token",
                headers={"Authorization": "Bearer refreshed-token"},
            ),
        )
        second = await lease.endpoint(force_refresh=True)
        assert replay_origin("copilot", "messages", "a", endpoint=first).accepts(
            replay_origin("copilot", "messages", "b", endpoint=second)
        )
    await broker.close()


@pytest.mark.asyncio
async def test_identity_change_and_removed_models_cannot_publish_new_credentials() -> (
    None
):
    runtime = FakeRuntime()
    broker = CopilotBroker(runtime)
    async with broker.lease("model") as lease:
        runtime.current_identity = CopilotIdentity("github.com", "other")
        with pytest.raises(CopilotAuthenticationRequired):
            await lease.endpoint()
        runtime.current_identity = CopilotIdentity("github.com", "octocat")
        runtime.available = (model("other-model"),)
        await broker.snapshot(refresh=True)
        with pytest.raises(InvalidRequestError):
            await lease.endpoint()
    with pytest.raises(InvalidRequestError):
        async with broker.lease("model"):
            pytest.fail("removed model was leased")
    await broker.close()


@pytest.mark.asyncio
async def test_lease_rejects_protocol_change_or_expired_credential() -> None:
    runtime = FakeRuntime()
    broker = CopilotBroker(runtime)
    async with broker.lease("model") as lease:
        session = runtime.sessions[0]
        session.value = replace(session.value, egress=CopilotEgress.MESSAGES)
        with pytest.raises(CopilotUnavailable, match="protocol"):
            await lease.endpoint(force_refresh=True)
        session.value = replace(session.value, expires_at=time.time() - 10)
        with pytest.raises(CopilotUnavailable, match="expired"):
            await lease.endpoint(force_refresh=True)
    await broker.close()


@pytest.mark.asyncio
async def test_cancelled_close_waits_for_active_lease_and_released_lease_is_unusable() -> (
    None
):
    runtime = FakeRuntime()
    broker = CopilotBroker(runtime)
    async with broker.lease("model") as lease:
        task = asyncio.create_task(broker.close())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and runtime.close_calls == 0
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.close_calls == 1
    with pytest.raises(CopilotUnavailable, match="closed"):
        await lease.endpoint()


@pytest.mark.asyncio
async def test_cancelled_session_creation_is_drained_before_runtime_closes() -> None:
    runtime = FakeRuntime()
    runtime.session_gate.clear()
    broker = CopilotBroker(runtime)

    async def request() -> None:
        async with broker.lease("model"):
            pytest.fail("cancelled request admitted")

    request_task = asyncio.create_task(request())
    await runtime.session_entered.wait()
    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task
    close = asyncio.create_task(broker.close())
    await asyncio.sleep(0)
    assert runtime.close_calls == 0
    runtime.session_gate.set()
    await close
    assert runtime.sessions[0].closed


@pytest.mark.asyncio
async def test_account_change_during_discovery_rejects_snapshot() -> None:
    runtime = FakeRuntime()
    runtime.model_gate.clear()
    broker = CopilotBroker(runtime)
    task = asyncio.create_task(broker.snapshot())
    await runtime.model_entered.wait()
    runtime.current_identity = CopilotIdentity("github.com", "other")
    runtime.model_gate.set()
    with pytest.raises(CopilotAuthenticationRequired):
        await task
    assert not broker.current_models
    await broker.close()


@pytest.mark.asyncio
async def test_confirmed_empty_discovery_revokes_old_model_credentials() -> None:
    runtime = FakeRuntime()
    broker = CopilotBroker(runtime)
    try:
        async with broker.lease("model") as lease:
            session = runtime.sessions[0]
            runtime.available = ()
            with pytest.raises(CopilotUnavailable, match="no enabled Copilot models"):
                await broker.snapshot(refresh=True)
            previous_endpoint_calls = session.endpoint_calls
            with pytest.raises(InvalidRequestError):
                await lease.endpoint(force_refresh=True)
            assert session.endpoint_calls == previous_endpoint_calls
        assert not broker.current_models
        with pytest.raises((CopilotUnavailable, InvalidRequestError)):
            async with broker.lease("model"):
                pytest.fail("empty discovery left a removed model leasable")
    finally:
        await broker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity", [None, CopilotIdentity("github.com", "other-account")]
)
async def test_cached_discovery_rejects_changed_or_signed_out_native_identity(
    identity: CopilotIdentity | None,
) -> None:
    runtime = FakeRuntime()
    broker = CopilotBroker(runtime)
    try:
        assert list(await broker.snapshot()) == ["model"]
        runtime.current_identity = identity
        with pytest.raises(CopilotAuthenticationRequired):
            await broker.snapshot(refresh=False)
    finally:
        await broker.close()

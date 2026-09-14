"""FCC opt-in and interactive login over isolated native-account test doubles."""

import asyncio
import json
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
)
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.github_copilot.auth import CopilotAuthManager
from free_claude_code.providers.github_copilot.login import DeviceChallenge
from free_claude_code.providers.github_copilot.types import (
    CopilotIdentity,
    CopilotUnavailable,
)
from tests.providers.copilot_support import FakeRuntime

DEVICE = ConnectedAccountLoginMode.DEVICE


async def settled(manager: CopilotAuthManager) -> None:
    async with asyncio.timeout(2):
        while manager.status().state is ConnectedAccountState.CONNECTING:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_existing_native_profile_connects_persists_only_safe_state_and_restarts_lazily(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    path = tmp_path / "state.json"
    manager = CopilotAuthManager(
        state_path=path, runtime_factory=AsyncMock(side_effect=lambda: runtime)
    )
    assert not manager.is_connected() and runtime.start_calls == 0
    await manager.start_login(DEVICE)
    await settled(manager)
    assert manager.is_connected() and manager.status().display_identity == "@octocat"
    state = json.loads(path.read_text())
    assert set(state) == {"schema_version", "enabled", "identity", "revision"}
    assert state["enabled"] is True
    await manager.close()
    replacement = FakeRuntime()
    restarted = CopilotAuthManager(
        state_path=path, runtime_factory=AsyncMock(side_effect=lambda: replacement)
    )
    assert restarted.is_connected() and replacement.start_calls == 0
    assert list(await restarted.models()) == ["model"]
    await restarted.disconnect()
    assert json.loads(path.read_text())["enabled"] is False
    assert replacement.close_calls == 1
    await restarted.close()


@pytest.mark.asyncio
async def test_device_login_challenge_and_fresh_validation_after_cli_finishes(
    tmp_path: Path,
) -> None:
    signed_out, signed_in = FakeRuntime(), FakeRuntime()
    signed_out.current_identity = None
    runtimes = iter((signed_out, signed_in))
    release = asyncio.Event()

    async def login(callback: Callable[[DeviceChallenge], None]) -> None:
        callback(DeviceChallenge("https://github.com/login/device", "ABCD-1234"))
        await release.wait()

    manager = CopilotAuthManager(
        state_path=tmp_path / "state.json",
        runtime_factory=AsyncMock(side_effect=lambda: next(runtimes)),
        login_flow=login,
    )
    await manager.start_login(DEVICE)
    async with asyncio.timeout(2):
        while manager.status().user_code is None:
            await asyncio.sleep(0)
    status = manager.status()
    assert not status.connected and status.user_code == "ABCD-1234"
    assert (
        status.supported_login_modes == (DEVICE,)
        and status.default_login_mode is DEVICE
    )
    assert signed_out.close_calls == 1
    release.set()
    await settled(manager)
    assert manager.is_connected() and signed_in.model_calls == 1
    await manager.close()


@pytest.mark.asyncio
async def test_concurrent_connect_coalesces_and_cancel_prevents_stale_publication(
    tmp_path: Path,
) -> None:
    signed_out, signed_in = FakeRuntime(), FakeRuntime()
    signed_out.current_identity = None
    runtimes = iter((signed_out, signed_in))
    entered = asyncio.Event()

    async def login(_callback: Callable[[DeviceChallenge], None]) -> None:
        entered.set()
        # Simulate a native grant that finishes as cancellation arrives.
        with suppress(asyncio.CancelledError):
            await asyncio.Event().wait()

    manager = CopilotAuthManager(
        state_path=tmp_path / "state.json",
        runtime_factory=AsyncMock(side_effect=lambda: next(runtimes)),
        login_flow=login,
    )
    first, second = await asyncio.gather(
        manager.start_login(DEVICE), manager.start_login(DEVICE)
    )
    assert first.attempt_id == second.attempt_id
    await entered.wait()
    await manager.cancel_login()
    assert not manager.is_connected()
    assert signed_in.close_calls == 1
    assert manager.status().state is ConnectedAccountState.DISCONNECTED
    await manager.close()


@pytest.mark.asyncio
async def test_login_timeout_cleans_up_and_reports_local_deadline(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    runtime.current_identity = None
    exited = asyncio.Event()

    async def login(_callback: Callable[[DeviceChallenge], None]) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    manager = CopilotAuthManager(
        state_path=tmp_path / "state.json",
        runtime_factory=AsyncMock(side_effect=lambda: runtime),
        login_flow=login,
        login_timeout_s=0.02,
    )
    await manager.start_login(DEVICE)
    await settled(manager)
    assert exited.is_set() and not manager.is_connected()
    assert "FCC's time limit" in (manager.status().message or "")
    await manager.close()


@pytest.mark.asyncio
async def test_profile_change_invalidates_revision_before_credentials_leave_lease(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    manager = CopilotAuthManager(
        state_path=tmp_path / "state.json",
        runtime_factory=AsyncMock(side_effect=lambda: runtime),
    )
    await manager.start_login(DEVICE)
    await settled(manager)
    revision = manager.status().revision
    async with manager.lease("model") as lease:
        runtime.current_identity = CopilotIdentity("github.com", "other")
        with pytest.raises(ExecutionFailure) as error:
            await lease.endpoint()
        assert error.value.status_code == 401
        assert not manager.is_connected() and manager.status().revision > revision
        assert runtime.close_calls == 0
    await manager.close()
    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_disconnect_invalidates_new_admission_and_waits_for_response_lease(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime()
    manager = CopilotAuthManager(
        state_path=tmp_path / "state.json",
        runtime_factory=AsyncMock(side_effect=lambda: runtime),
    )
    await manager.start_login(DEVICE)
    await settled(manager)
    async with manager.lease("model"):
        task = asyncio.create_task(manager.disconnect())
        await asyncio.sleep(0)
        assert not manager.is_connected() and not task.done()
        with pytest.raises(ExecutionFailure):
            async with manager.lease("model"):
                pytest.fail("new request admitted after disconnect")
    await task
    assert runtime.close_calls == 1
    await manager.close()


@pytest.mark.asyncio
async def test_failed_persistence_never_reports_connected(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    runtime.model_gate.clear()
    path = tmp_path / "state.json"
    manager = CopilotAuthManager(
        state_path=path, runtime_factory=AsyncMock(side_effect=lambda: runtime)
    )
    await manager.start_login(DEVICE)
    await runtime.model_entered.wait()
    path.unlink()
    path.mkdir()
    runtime.model_gate.set()
    await settled(manager)
    assert (
        not manager.is_connected()
        and manager.status().state is ConnectedAccountState.ERROR
    )
    assert runtime.close_calls == 1
    await manager.close()


@pytest.mark.asyncio
async def test_unsupported_login_mode_does_not_start_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    manager = CopilotAuthManager(
        state_path=tmp_path / "state.json",
        runtime_factory=AsyncMock(side_effect=lambda: runtime),
    )
    with pytest.raises(InvalidRequestError):
        await manager.start_login(ConnectedAccountLoginMode.BROWSER)
    assert runtime.start_calls == 0
    await manager.close()


@pytest.mark.asyncio
async def test_cancel_reports_and_retains_failed_runtime_cleanup(
    tmp_path: Path,
) -> None:
    class Runtime(FakeRuntime):
        fail_close = True

        async def close(self) -> None:
            if self.fail_close:
                raise CopilotUnavailable("Owned runtime could not be stopped.")
            await super().close()

    runtime = Runtime()
    runtime.model_gate.clear()
    manager = CopilotAuthManager(
        state_path=tmp_path / "state.json",
        runtime_factory=AsyncMock(side_effect=lambda: runtime),
    )
    await manager.start_login(DEVICE)
    await runtime.model_entered.wait()
    cancellation = asyncio.create_task(manager.cancel_login())
    await asyncio.sleep(0)
    runtime.model_gate.set()
    try:
        status = await cancellation
        assert status.state is ConnectedAccountState.ERROR
        assert "cleanup" in (status.message or "").lower()
        assert not status.connected
    finally:
        runtime.fail_close = False
        await manager.close()
    assert runtime.close_calls == 1

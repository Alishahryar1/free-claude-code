import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.integrations.discovery import LocalInstallations
from free_claude_code.runtime.integrations.service import IntegrationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.runtime.test_application_runtime import TrackingFactory, _prepared, _settings


@pytest.mark.asyncio
async def test_awaitable_restart_callback_is_not_executed_under_apply_lock(tmp_path):
    manager = ProviderRuntimeManager(_settings("nvidia_nim/old"))
    configuration = AsyncMock(spec=ConfigurationService)
    prepared = _prepared(
        _settings("nvidia_nim/old", port=9090), tmp_path, pending_fields=("PORT",)
    )
    configuration.prepare.return_value = prepared
    configuration.commit.return_value = prepared.applied_response()
    callback_tasks = []

    async def close_on_restart():
        callback_tasks.append(asyncio.current_task())
        await runtime.close()

    # A dynamically supplied callback can evade static checking by returning a coroutine.
    callback = MagicMock(side_effect=close_on_restart)
    runtime = ApplicationRuntime(
        manager,
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=configuration,
        transcriber=None,
        restart_callback=callback,
    )
    with patch(
        "free_claude_code.runtime.application.check_credentials",
        AsyncMock(return_value=()),
    ):
        apply = asyncio.create_task(runtime.apply_admin_config({"PORT": "9090"}))
        try:
            done, _ = await asyncio.wait({apply}, timeout=2)
            assert apply in done, "restart callback deadlocked with runtime.close"
            result = apply.result()
            assert result["applied"] is True
            restart = result["restart"]
            assert isinstance(restart, dict)
            assert restart["automatic"] is False
            assert runtime._pending_fields == ["PORT"]
            assert callback_tasks == []
            callback.assert_called_once_with()
        finally:
            for task in callback_tasks:
                task.cancel()
            apply.cancel()
            await asyncio.gather(apply, return_exceptions=True)
            await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [(), ("PORT",)])
@pytest.mark.parametrize("fail", [False, True])
async def test_cancelled_commit_settles_before_shutdown(tmp_path, pending, fail):
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/old"), runtime_factory=factory
    )
    configuration = AsyncMock(spec=ConfigurationService)
    prepared = _prepared(_settings("nvidia_nim/new"), tmp_path, pending_fields=pending)
    configuration.prepare.return_value = prepared
    entered, release = asyncio.Event(), asyncio.Event()

    async def commit(_prepared):
        entered.set()
        await release.wait()
        if fail:
            raise OSError("disk full")
        return prepared.applied_response()

    configuration.commit.side_effect = commit
    restart = MagicMock(return_value=None)
    runtime = ApplicationRuntime(
        manager,
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=configuration,
        transcriber=None,
        restart_callback=restart,
    )
    with patch(
        "free_claude_code.runtime.application.check_credentials",
        AsyncMock(return_value=()),
    ):
        apply = asyncio.create_task(runtime.apply_admin_config({}))
        await asyncio.wait_for(entered.wait(), 5)
        apply.cancel()
        await asyncio.sleep(0)
        apply.cancel()
        close = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert not apply.done()
        assert not close.done()
        assert factory.runtimes[0].cleanup_calls == 0
        release.set()
        with pytest.raises(asyncio.CancelledError) as error:
            await asyncio.wait_for(apply, 5)
        if fail:
            assert isinstance(error.value.__cause__, OSError)
        assert manager.current_generation_id == (2 if not pending and not fail else 1)
        assert restart.call_count == (1 if pending and not fail else 0)
        assert await asyncio.wait_for(close, 5)
        with pytest.raises(ApplicationUnavailableError):
            await runtime.apply_admin_config({})
    configuration.commit.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_required", [False, True])
@pytest.mark.parametrize("cancel_during_apply", [False, True])
@pytest.mark.parametrize("previous_cancellation", [False, True])
async def test_cancellation_at_finalization_handoff_prevents_persistence(
    monkeypatch, restart_required, cancel_during_apply, previous_cancellation
):
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    store = ManagedConfigStore()
    store.initialize()
    initial = store.read()
    original_bytes = initial.path.read_bytes()
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(initial.settings, runtime_factory=factory)
    restart = MagicMock(return_value=None)
    runtime = ApplicationRuntime(
        manager,
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=ConfigurationService(store),
        transcriber=None,
        restart_callback=restart,
    )
    updates = {"PORT": "8123"} if restart_required else {"MODEL": "nvidia_nim/new"}

    async def run_apply():
        if previous_cancellation:
            caller = asyncio.current_task()
            assert caller is not None
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.sleep(0)
        return await runtime.apply_admin_config(updates)

    async def check_credentials(*_args):
        if cancel_during_apply:
            # Queue cancellation before finalization, with the caller's wakeup after it.
            asyncio.get_running_loop().call_soon(apply.cancel)
        return ()

    with (
        patch(
            "free_claude_code.runtime.application.check_credentials", check_credentials
        ),
        patch.object(manager, "_refresh_generation_in_background", AsyncMock()),
        patch.object(store, "commit", wraps=store.commit) as commit,
    ):
        apply = asyncio.create_task(run_apply())
        try:
            if cancel_during_apply:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(apply, 5)
                commit.assert_not_called()
                assert initial.path.read_bytes() == original_bytes
                assert manager.current_generation_id == 1
                assert runtime._pending_fields == []
                restart.assert_not_called()
                assert all(item.cleanup_calls == 1 for item in factory.runtimes[1:])
            else:
                assert (await asyncio.wait_for(apply, 5))["applied"] is True
                commit.assert_called_once()
                assert manager.current_generation_id == (1 if restart_required else 2)
                assert restart.call_count == int(restart_required)
        finally:
            apply.cancel()
            await asyncio.gather(apply, return_exceptions=True)
            await runtime.close()


@pytest.mark.asyncio
async def test_cancellation_waiting_for_replacement_does_not_persist(tmp_path):
    manager = ProviderRuntimeManager(_settings("nvidia_nim/old"))
    configuration = AsyncMock(spec=ConfigurationService)
    configuration.prepare.return_value = _prepared(
        _settings("nvidia_nim/new"), tmp_path
    )
    runtime = ApplicationRuntime(
        manager,
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=configuration,
        transcriber=None,
    )
    with patch(
        "free_claude_code.runtime.application.check_credentials",
        AsyncMock(return_value=()),
    ):
        async with manager._replace_lock:
            apply = asyncio.create_task(runtime.apply_admin_config({}))
            while not manager._replace_lock._waiters:
                await asyncio.sleep(0)
            apply.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(apply, 5)
        configuration.commit.assert_not_awaited()
        assert manager.current_generation_id == 1
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_worker_commit_retains_ownership_against_queued_apply(
    monkeypatch, shutdown
):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    store = ManagedConfigStore()
    store.initialize()
    manager = ProviderRuntimeManager(_settings("nvidia_nim/old"))
    restart = MagicMock(return_value=None)
    runtime = ApplicationRuntime(
        manager,
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=ConfigurationService(store),
        transcriber=None,
        restart_callback=restart,
    )
    entered, release = threading.Event(), threading.Event()
    original = store.commit
    writes = []

    def blocked_commit(values):
        entered.set()
        assert release.wait(5)
        writes.append(values["LOG_LEVEL"])
        original(values)

    async def wait_for_commit():
        while not entered.is_set():
            await asyncio.sleep(0)

    with patch.object(store, "commit", blocked_commit):
        first = asyncio.create_task(
            runtime.apply_admin_config({"LOG_LEVEL": "WARNING"})
        )
        second = None
        try:
            await asyncio.wait_for(wait_for_commit(), 5)
            first.cancel()
            second = asyncio.create_task(
                runtime.apply_admin_config({"LOG_LEVEL": "ERROR"})
            )
            await asyncio.sleep(0)
            assert runtime._config_lock.locked()
            assert not second.done()
            assert writes == []
            if shutdown:
                runtime.begin_shutdown()
            first.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first, 5)
            if shutdown:
                with pytest.raises(ApplicationUnavailableError):
                    await asyncio.wait_for(second, 5)
            else:
                assert (await asyncio.wait_for(second, 5))["applied"]
            assert writes == (["WARNING"] if shutdown else ["WARNING", "ERROR"])
            assert store.read().settings.log_level == writes[-1]
            assert restart.call_count == len(writes)
        finally:
            release.set()
            await asyncio.gather(
                first, *([second] if second is not None else []), return_exceptions=True
            )
            await runtime.close()

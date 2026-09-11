import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config import loader
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.paths import config_lock_path
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.integrations.discovery import LocalInstallations
from free_claude_code.runtime.integrations.service import IntegrationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.runtime.test_application_runtime import _settings


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_shutdown_drains_initialization_worker(cancel, fail):
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled_errors = []
    loop.set_exception_handler(lambda _loop, context: unhandled_errors.append(context))
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    store = ManagedConfigStore()
    manager = ProviderRuntimeManager(_settings("nvidia_nim/old"))
    runtime = ApplicationRuntime(
        manager,
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=ConfigurationService(store),
        transcriber=None,
    )
    consolidate = loader.consolidate_managed_config

    def blocked_consolidation(env):
        entered.set()
        try:
            assert release.wait(5)
            if fail:
                raise OSError("private-initialization-detail")
            return consolidate(env)
        finally:
            finished.set()

    async def wait_for_worker():
        while not entered.is_set():
            await asyncio.sleep(0)

    with (
        patch.object(loader, "consolidate_managed_config", blocked_consolidation),
        patch.object(manager, "warm_referenced_model_cache", AsyncMock()) as warm,
    ):
        start = asyncio.create_task(runtime.start())
        close = None
        try:
            await asyncio.wait_for(wait_for_worker(), 5)
            if cancel:
                start.cancel()
                await asyncio.sleep(0)
                start.cancel()
            close = asyncio.create_task(runtime.close())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not start.done()
            assert not close.done()
            assert not runtime.is_closed
            assert not finished.is_set()
            assert not store.path.exists()
            release.set()
            error = (
                asyncio.CancelledError
                if cancel
                else OSError
                if fail
                else ApplicationUnavailableError
            )
            with pytest.raises(error):
                await asyncio.wait_for(start, 5)
            assert await asyncio.wait_for(close, 5)
            assert finished.is_set()
            assert unhandled_errors == []
            assert store.path.exists() is not fail
            warm.assert_not_awaited()
            lock = InterprocessFileLock(config_lock_path())
            try:
                assert lock.acquire()
            finally:
                lock.release()
        finally:
            release.set()
            await asyncio.gather(
                start, *([close] if close else []), return_exceptions=True
            )
            await runtime.close()
            loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_closed_runtime_cannot_start_another_configuration_writer():
    store = ManagedConfigStore()
    runtime = ApplicationRuntime(
        ProviderRuntimeManager(_settings("nvidia_nim/old")),
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=ConfigurationService(store),
        transcriber=None,
    )
    await runtime.close()
    with pytest.raises(ApplicationUnavailableError):
        await runtime.start()
    assert not store.path.exists()

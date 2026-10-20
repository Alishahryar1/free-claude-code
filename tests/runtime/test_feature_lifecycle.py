import asyncio
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from free_claude_code.runtime.sqlite_database import SQLiteDatabase
from tests.runtime.test_application_runtime import (
    TrackingFactory,
    TrackingMessagingRuntime,
    TrackingTranscriber,
    _settings,
)


@pytest.mark.asyncio
async def test_close_cancels_both_feature_owners_before_acquiring_config_lock():
    manager = ProviderRuntimeManager(_settings("nvidia_nim/model"))
    database = AsyncMock(spec=SQLiteDatabase)
    import_started = asyncio.Event()

    async def start_database():
        import_started.set()
        await asyncio.Event().wait()

    database.start.side_effect = start_database
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=None,
        database=database,
    )
    close = None
    try:
        async with runtime._config_lock:
            await runtime.start()
            await asyncio.wait_for(import_started.wait(), 5)
            refresh = runtime._integrations._claude_update.task
            assert refresh is not None and not refresh.done()
            messaging_tasks = tuple(runtime._messaging._startup_tasks)
            assert len(messaging_tasks) == 2
            assert all(not task.done() for task in messaging_tasks)
            close = asyncio.create_task(runtime.close())
            results = await asyncio.wait_for(
                asyncio.gather(refresh, *messaging_tasks, return_exceptions=True), 5
            )
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            assert not close.done()
            database.close.assert_not_awaited()
        assert await asyncio.wait_for(close, 5)
        database.close.assert_awaited_once()
    finally:
        if close is not None:
            await close
        await runtime.close()


@pytest.mark.asyncio
async def test_draining_runtime_rejects_new_feature_work():
    runtime = ApplicationRuntime(
        ProviderRuntimeManager(_settings("nvidia_nim/model")),
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=None,
    )
    try:
        runtime.begin_shutdown()
        runtime._integrations.catalog_changed()
        assert runtime._integrations._vscode_update.task is None
        with pytest.raises(ApplicationUnavailableError):
            await runtime.refresh_claude_vscode()
        with pytest.raises(ApplicationUnavailableError):
            runtime._messaging.start()
        assert runtime._messaging._startup_tasks == []
    finally:
        assert await runtime.close()


@pytest.mark.asyncio
async def test_failed_messaging_drain_retains_shared_dependencies_until_retry():
    events = []
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/model"), runtime_factory=factory
    )
    code = AsyncMock(spec=CodeService)
    database = AsyncMock(spec=SQLiteDatabase)
    code.close.side_effect = lambda: events.append("code.close")
    database.close.side_effect = lambda: events.append("database.close")
    transcriber = TrackingTranscriber(events)
    runtime = ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=transcriber,
        database=database,
        code_service=code,
    )
    runtime._messaging._messaging_runtime = TrackingMessagingRuntime(
        events, fail_quiesce_once=True
    )
    try:
        assert not await runtime.close()
        assert not runtime.is_closed
        code.close.assert_not_awaited()
        database.close.assert_not_awaited()
        assert transcriber.close_calls == 0
        assert factory.runtimes[0].cleanup_calls == 0
        assert await runtime.close()
        assert events == [
            "messaging.quiesce",
            "messaging.quiesce",
            "messaging.close",
            "code.close",
            "database.close",
            "transcriber.close",
        ]
        assert factory.runtimes[0].cleanup_calls == 1
    finally:
        await runtime.close()

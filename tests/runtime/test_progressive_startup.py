"""Behavioral checks for progressively available, owned startup work."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.readiness import InitializationWait
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.settings import Settings
from free_claude_code.core.async_tasks import run_sync_owned
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.runtime.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.codex_catalog import CodexModelCatalogPublisher
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


def _settings():
    return Settings().model_copy(
        update={
            "model": "nvidia_nim/one",
            "nvidia_nim_api_key": "test",
            "groq_api_key": "test",
            "messaging_platform": "none",
        }
    )


@pytest.mark.asyncio
async def test_http_and_admin_serve_while_selected_catalog_is_held():
    entered, release = asyncio.Event(), asyncio.Event()
    provider = MagicMock(spec=BaseProvider)

    async def list_models():
        entered.set()
        await release.wait()
        return frozenset({ProviderModelInfo("one")})

    provider.list_model_infos = AsyncMock(side_effect=list_models)

    async def construct(_provider_id, _settings):
        return provider

    manager = ProviderRuntimeManager(
        _settings(),
        runtime_factory=lambda settings: ProviderRuntime(
            settings, provider_constructor=construct
        ),
    )
    runtime = ApplicationRuntime(
        manager,
        configuration=ConfigurationService(ManagedConfigStore()),
        transcriber=None,
    )
    app = create_app(ApiServices(manager, runtime, runtime))
    try:
        await asyncio.wait_for(runtime.start(), 1)
        await asyncio.wait_for(entered.wait(), 1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            assert (await client.get("/health")).status_code == 200
            status = await client.get("/admin/api/status")
            assert status.status_code == 200
            assert status.json()["startup"]["catalog"] == "starting"
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_independent_provider_and_cancelled_waiter_share_initialization():
    entered, release = asyncio.Event(), asyncio.Event()
    slow, fast = MagicMock(spec=BaseProvider), MagicMock(spec=BaseProvider)

    async def slow_models():
        entered.set()
        await release.wait()
        return frozenset({ProviderModelInfo("slow")})

    slow.list_model_infos = AsyncMock(side_effect=slow_models)
    fast.list_model_infos = AsyncMock(
        return_value=frozenset({ProviderModelInfo("one")})
    )

    async def construct(provider_id, _settings):
        return slow if provider_id == "groq" else fast

    manager = ProviderRuntimeManager(
        _settings(),
        runtime_factory=lambda settings: ProviderRuntime(
            settings, provider_constructor=construct
        ),
    )
    manager.start_model_list_refresh()
    leases = [await manager.acquire() for _ in range(3)]
    waiting = [
        asyncio.create_task(lease.resolve_provider("groq")) for lease in leases[:2]
    ]
    try:
        await entered.wait()
        assert await leases[2].resolve_provider("nvidia_nim") is fast
        assert leases[2].model_info("nvidia_nim", "one") is not None
        waiting[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting[0]
        assert not waiting[1].done()
        release.set()
        assert await waiting[1] is slow
        assert slow.list_model_infos.await_count == 1
    finally:
        release.set()
        await asyncio.gather(*waiting, return_exceptions=True)
        for lease in leases:
            await lease.release()
        await manager.close()


@pytest.mark.asyncio
async def test_wait_timeout_does_not_cancel_initializer_and_settled_work_still_works():
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    wait = InitializationWait(0.01)
    try:
        with pytest.raises(ApplicationUnavailableError, match="still starting"):
            await wait.wait(task)
        assert not task.done()
        release.set()
        await task
        assert await wait.wait(task) is True
    finally:
        release.set()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh", [False, True])
async def test_failed_discovery_construction_recovers_in_same_generation(refresh):
    provider = MagicMock(spec=BaseProvider)
    provider.list_model_infos = AsyncMock(
        return_value=frozenset({ProviderModelInfo("one")})
    )
    attempts = 0

    async def construct(_id, _settings):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("construction failed")
        return provider

    manager = ProviderRuntimeManager(
        _settings(),
        runtime_factory=lambda settings: ProviderRuntime(
            settings, provider_constructor=construct
        ),
    )
    generation = manager.current_generation_id
    try:
        result = await manager.refresh_provider("nvidia_nim")
        assert result.failed_provider_ids == ("nvidia_nim",)
        if refresh:
            result = await manager.refresh_provider("nvidia_nim")
            assert result.refreshed_provider_ids == ("nvidia_nim",)
        async with await manager.acquire() as lease:
            assert await lease.resolve_provider("nvidia_nim") is provider
            assert (lease.model_info("nvidia_nim", "one") is not None) is refresh
        assert manager.current_generation_id == generation
        assert attempts == 2
    finally:
        await manager.close()
    provider.cleanup.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_recovery_wait_uses_remaining_budget_without_orphaning_waiter(
    monkeypatch, cancel
):
    entered, release = asyncio.Event(), asyncio.Event()
    provider = MagicMock(spec=BaseProvider)
    attempts = 0
    construction = None

    async def construct(_id, _settings):
        nonlocal attempts, construction
        attempts += 1
        if attempts == 1:
            raise RuntimeError("construction failed")
        construction = asyncio.current_task()
        entered.set()
        await release.wait()
        return provider

    manager = ProviderRuntimeManager(
        _settings(),
        runtime_factory=lambda settings: ProviderRuntime(
            settings, provider_constructor=construct
        ),
    )
    wait = InitializationWait(1 if cancel else 0.01)
    monkeypatch.setattr(
        "free_claude_code.runtime.provider_manager.InitializationWait", lambda: wait
    )
    lease = await manager.acquire()
    waiting = None
    try:
        result = await manager.refresh_provider("nvidia_nim")
        assert result.failed_provider_ids == ("nvidia_nim",)
        existing_tasks = asyncio.all_tasks()
        waiting = asyncio.create_task(lease.resolve_provider("nvidia_nim"))
        await asyncio.wait_for(entered.wait(), 1)
        if cancel:
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        else:
            with pytest.raises(ApplicationUnavailableError, match="still starting"):
                await asyncio.wait_for(asyncio.shield(waiting), 1)
        assert construction is not None and not construction.done()
        assert asyncio.all_tasks() <= existing_tasks | {construction}
        release.set()
        await construction
        wait.remaining = 0
        assert await lease.resolve_provider("nvidia_nim") is provider
        assert attempts == 2
    finally:
        release.set()
        if waiting is not None:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
        await lease.release()
        await manager.close()


@pytest.mark.asyncio
async def test_cancellation_drains_the_actual_worker():
    entered, release = threading.Event(), threading.Event()

    def worker():
        entered.set()
        release.wait()

    task = asyncio.create_task(run_sync_owned(worker))
    try:
        await asyncio.to_thread(entered.wait)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_late_retired_discovery_cannot_replace_current_metadata():
    entered, release = asyncio.Event(), asyncio.Event()
    providers = []

    def factory(settings):
        provider = MagicMock(spec=BaseProvider)
        providers.append(provider)
        ordinal = len(providers)

        async def models():
            if ordinal == 1:
                entered.set()
                await release.wait()
            return frozenset({ProviderModelInfo(f"model-{ordinal}")})

        provider.list_model_infos = AsyncMock(side_effect=models)

        async def construct(_provider_id, _settings):
            return provider

        return ProviderRuntime(settings, provider_constructor=construct)

    manager = ProviderRuntimeManager(_settings(), runtime_factory=factory)
    old = await manager.acquire()
    waiting = asyncio.create_task(old.resolve_provider("nvidia_nim"))
    new = None
    try:
        await entered.wait()
        await manager.replace(_settings(), commit=AsyncMock())
        new = await manager.acquire()
        await new.resolve_provider("nvidia_nim")
        release.set()
        await waiting
        assert old.model_info("nvidia_nim", "model-1") is not None
        assert new.model_info("nvidia_nim", "model-2") is not None
        assert manager.cached_model_info("nvidia_nim", "model-1") is None
    finally:
        release.set()
        await asyncio.gather(waiting, return_exceptions=True)
        await old.release()
        if new:
            await new.release()
        await manager.close()


@pytest.mark.asyncio
async def test_complete_catalog_publication_preserves_prior_file_until_all_attempts(
    tmp_path,
):
    entered, release = asyncio.Event(), asyncio.Event()
    path = tmp_path / "catalog.json"
    path.write_text("prior complete catalog", encoding="utf-8")
    provider = MagicMock(spec=BaseProvider)

    async def models():
        entered.set()
        await release.wait()
        return frozenset({ProviderModelInfo("new-model")})

    provider.list_model_infos = AsyncMock(side_effect=models)

    async def construct(_provider_id, _settings):
        return provider

    manager = ProviderRuntimeManager(
        _settings(),
        runtime_factory=lambda settings: ProviderRuntime(
            settings, provider_constructor=construct
        ),
        model_catalog_publisher=CodexModelCatalogPublisher(path),
    )
    waiter = asyncio.create_task(manager.wait_for_catalog())
    try:
        await entered.wait()
        assert not waiter.done()
        assert path.read_text(encoding="utf-8") == "prior complete catalog"
        release.set()
        catalog = await waiter
        assert catalog.cached_model_info("nvidia_nim", "new-model") is not None
        assert "new-model" in path.read_text(encoding="utf-8")
    finally:
        release.set()
        await asyncio.gather(waiter, return_exceptions=True)
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("retained_request", [False, True])
@pytest.mark.parametrize("catalog_file", [False, True])
async def test_catalog_wait_follows_replacement(
    tmp_path, retained_request, catalog_file
):
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    def runtime(settings):
        nonlocal count
        count += 1
        old = count == 1
        provider = MagicMock(spec=BaseProvider)

        async def models():
            if old:
                entered.set()
                await release.wait()
            return frozenset({ProviderModelInfo("old" if old else "current")})

        provider.list_model_infos = AsyncMock(side_effect=models)

        async def construct(_id, _settings):
            return provider

        return ProviderRuntime(settings, provider_constructor=construct)

    path = tmp_path / "catalog.json"
    manager = ProviderRuntimeManager(
        _settings(),
        runtime_factory=runtime,
        model_catalog_publisher=CodexModelCatalogPublisher(path),
    )
    lease = await manager.acquire() if retained_request else None
    request = (
        asyncio.create_task(lease.resolve_provider("nvidia_nim")) if lease else None
    )
    waiting = asyncio.create_task(
        manager.wait_for_catalog_file(InitializationWait())
        if catalog_file
        else manager.wait_for_catalog()
    )
    try:
        await entered.wait()
        settings = _settings().model_copy(update={"model": "nvidia_nim/current"})
        await manager.replace(settings, commit=AsyncMock())
        await manager.wait_for_catalog()
        result = await asyncio.wait_for(asyncio.shield(waiting), 1)
        if catalog_file:
            assert isinstance(result, int)
            assert result == manager.current_generation_id
            assert "nvidia_nim/current" in path.read_text(encoding="utf-8")
        else:
            assert not isinstance(result, int)
            assert result.settings is settings
            assert result.cached_model_info("nvidia_nim", "current") is not None
        if request is not None:
            assert lease is not None
            assert not request.done()
            release.set()
            await request
            assert lease.model_info("nvidia_nim", "old") is not None
    finally:
        release.set()
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        if request is not None:
            await asyncio.gather(request, return_exceptions=True)
        if lease is not None:
            await lease.release()
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("catalog_file", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_catalog_waiter_exit_preserves_shared_discovery(
    monkeypatch, catalog_file, cancel
):
    existing_tasks = asyncio.all_tasks()
    entered, release = asyncio.Event(), asyncio.Event()
    provider = MagicMock(spec=BaseProvider)

    async def models():
        entered.set()
        await release.wait()
        return frozenset({ProviderModelInfo("one")})

    provider.list_model_infos = AsyncMock(side_effect=models)

    async def construct(_id, _settings):
        return provider

    manager = ProviderRuntimeManager(
        _settings(),
        runtime_factory=lambda settings: ProviderRuntime(
            settings, provider_constructor=construct
        ),
    )
    wait = InitializationWait(1 if cancel else 0.01)
    monkeypatch.setattr(
        "free_claude_code.runtime.provider_manager.InitializationWait", lambda: wait
    )
    waiting = asyncio.create_task(
        manager.wait_for_catalog_file(wait)
        if catalog_file
        else manager.wait_for_catalog()
    )
    try:
        await entered.wait()
        if cancel:
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        else:
            with pytest.raises(ApplicationUnavailableError, match="still starting"):
                await waiting
        monkeypatch.setattr(
            "free_claude_code.runtime.provider_manager.InitializationWait",
            InitializationWait,
        )
        release.set()
        snapshot = await manager.wait_for_catalog()
        assert snapshot.cached_model_info("nvidia_nim", "one") is not None
    finally:
        release.set()
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await manager.close()
    assert asyncio.all_tasks() <= existing_tasks

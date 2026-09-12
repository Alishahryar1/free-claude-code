import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import RequestRuntimePort
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


class FakeRuntime(ProviderRuntime):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cleanup_calls = 0
        self.cleanup_error: Exception | None = None
        self.cleanup_started: asyncio.Event | None = None
        self.cleanup_release: asyncio.Event | None = None
        self.provider = MagicMock()
        self.provider.list_model_infos = AsyncMock(return_value=frozenset())

    def is_cached(self, provider_id: str) -> bool:
        return provider_id == "cached"

    def resolve_provider(self, provider_id: str) -> BaseProvider:
        return cast(BaseProvider, self.provider)

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.cleanup_started is not None:
            self.cleanup_started.set()
        if self.cleanup_release is not None:
            await self.cleanup_release.wait()
        if self.cleanup_error is not None:
            raise self.cleanup_error


class RuntimeFactory:
    def __init__(self) -> None:
        self.runtimes: list[FakeRuntime] = []
        self.error: Exception | None = None

    def __call__(self, settings: Settings) -> ProviderRuntime:
        if self.error is not None:
            raise self.error
        runtime = FakeRuntime(settings)
        self.runtimes.append(runtime)
        return runtime


class RecordingModelCatalogPublisher:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.snapshots: list[tuple[str, tuple[str, ...]]] = []

    def ensure_exists(self, runtime: RequestRuntimePort) -> None:
        self._record("ensure_exists", runtime)

    def publish(self, runtime: RequestRuntimePort) -> None:
        self._record("publish", runtime)

    def _record(self, event: str, runtime: RequestRuntimePort) -> None:
        self.events.append(event)
        self.snapshots.append(
            (
                runtime.current_settings().model,
                tuple(info.model_id for info in runtime.cached_prefixed_model_infos()),
            )
        )


def _settings(model: str) -> Settings:
    return Settings().model_copy(
        update={"model": model, "nvidia_nim_api_key": "test-key"}
    )


@pytest.mark.asyncio
async def test_startup_generation_lease_and_shutdown_close_exactly_once() -> None:
    factory = RuntimeFactory()
    settings = _settings("nvidia_nim/one")
    manager = ProviderRuntimeManager(settings, runtime_factory=factory)

    lease = await manager.acquire()

    assert lease.generation_id == 1
    assert lease.settings is settings
    assert lease.model_infos == ()
    assert lease.is_provider_cached("cached") is True
    assert lease.resolve_provider("nvidia_nim") is factory.runtimes[0].provider
    await lease.release()
    await lease.release()
    await manager.close()
    await manager.close()

    assert factory.runtimes[0].cleanup_calls == 1
    with pytest.raises(ApplicationUnavailableError, match="shutting down"):
        await manager.acquire()


@pytest.mark.asyncio
async def test_connected_provider_refresh_and_eviction_are_targeted() -> None:
    factory = RuntimeFactory()
    connected: set[str] = set()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
        connected_provider_ids=lambda: tuple(connected),
    )
    factory.runtimes[0].provider.list_model_infos = AsyncMock(
        return_value=frozenset(
            {ProviderModelInfo(model_id="gpt-visible", supports_thinking=True)}
        )
    )

    connected.add("openai")
    result = await manager.connected_provider_changed("openai", connected=True)

    assert result.refreshed_provider_ids == ("openai",)
    assert manager.cached_model_ids()["openai"] == frozenset({"gpt-visible"})

    connected.clear()
    assert "openai" not in manager.cached_model_ids()

    await manager.connected_provider_changed("openai", connected=False)

    assert "openai" not in manager.cached_model_ids()
    await manager.close()


@pytest.mark.asyncio
async def test_catalog_publication_tracks_warm_refresh_and_direct_cache() -> None:
    factory = RuntimeFactory()
    publisher = RecordingModelCatalogPublisher()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
        model_catalog_publisher=publisher,
    )
    factory.runtimes[0].provider.list_model_infos = AsyncMock(
        return_value=frozenset({ProviderModelInfo("warm-model")})
    )

    await manager.warm_referenced_model_cache()
    manager.start_model_list_refresh()
    refresh_task = manager._refresh_task
    assert refresh_task is not None
    await refresh_task
    manager.cache_model_infos(
        "nvidia_nim",
        {ProviderModelInfo("tested-model")},
    )

    assert publisher.events == ["ensure_exists", "publish", "publish"]
    assert publisher.snapshots == [
        ("nvidia_nim/one", ("nvidia_nim/warm-model",)),
        (
            "nvidia_nim/one",
            (
                "nvidia_nim/warm-model",
                "lmstudio/warm-model",
                "llamacpp/warm-model",
                "ollama/warm-model",
            ),
        ),
        (
            "nvidia_nim/one",
            (
                "nvidia_nim/tested-model",
                "lmstudio/warm-model",
                "llamacpp/warm-model",
                "ollama/warm-model",
            ),
        ),
    ]

    await manager.close()
    assert publisher.events == ["ensure_exists", "publish", "publish"]


@pytest.mark.asyncio
async def test_failed_startup_discovery_still_ensures_a_fresh_catalog() -> None:
    factory = RuntimeFactory()
    publisher = RecordingModelCatalogPublisher()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/configured"),
        runtime_factory=factory,
        model_catalog_publisher=publisher,
    )
    factory.runtimes[0].provider.list_model_infos = AsyncMock(
        side_effect=RuntimeError("upstream unavailable")
    )

    result = await manager.warm_referenced_model_cache()

    assert result.failed_provider_ids == ("nvidia_nim",)
    assert publisher.events == ["ensure_exists"]
    assert publisher.snapshots == [("nvidia_nim/configured", ())]
    await manager.close()


@pytest.mark.asyncio
async def test_catalog_publication_tracks_connected_account_changes() -> None:
    factory = RuntimeFactory()
    publisher = RecordingModelCatalogPublisher()
    connected: set[str] = set()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
        connected_provider_ids=lambda: tuple(connected),
        model_catalog_publisher=publisher,
    )
    factory.runtimes[0].provider.list_model_infos = AsyncMock(
        return_value=frozenset({ProviderModelInfo("gpt-connected")})
    )

    connected.add("openai")
    await manager.connected_provider_changed("openai", connected=True)
    connected.clear()
    await manager.connected_provider_changed("openai", connected=False)

    assert publisher.events == ["publish", "publish"]
    assert publisher.snapshots == [
        ("nvidia_nim/one", ("openai/gpt-connected",)),
        ("nvidia_nim/one", ()),
    ]
    await manager.close()


@pytest.mark.asyncio
async def test_catalog_publication_tracks_replacement_and_its_refresh() -> None:
    factory = RuntimeFactory()
    publisher = RecordingModelCatalogPublisher()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
        model_catalog_publisher=publisher,
    )

    await manager.replace(
        _settings("nvidia_nim/two"),
        commit=AsyncMock(),
    )
    refresh_task = manager._refresh_task
    assert refresh_task is not None
    await refresh_task

    assert publisher.events == ["publish", "publish"]
    assert publisher.snapshots == [
        ("nvidia_nim/two", ()),
        ("nvidia_nim/two", ()),
    ]
    await manager.close()


@pytest.mark.asyncio
async def test_catalog_publication_failure_is_warning_only_and_secret_safe() -> None:
    factory = RuntimeFactory()
    secret = "private-catalog-write-detail"
    publisher = MagicMock()
    publisher.ensure_exists.side_effect = PermissionError(secret)
    publisher.publish.side_effect = PermissionError(secret)
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
        model_catalog_publisher=publisher,
    )

    with patch("free_claude_code.runtime.provider_manager.logger.warning") as warning:
        await manager.warm_referenced_model_cache()
        manager.cache_model_infos(
            "nvidia_nim",
            {ProviderModelInfo("tested-model")},
        )

    assert warning.call_count == 2
    log_blob = " ".join(
        str(value)
        for call in warning.call_args_list
        for value in (*call.args, *call.kwargs.values())
    )
    assert "PermissionError" in log_blob
    assert secret not in log_blob
    await manager.close()


@pytest.mark.asyncio
async def test_replacement_keeps_leased_generation_open_until_final_release() -> None:
    factory = RuntimeFactory()
    first_settings = _settings("nvidia_nim/one")
    second_settings = _settings("nvidia_nim/two")
    manager = ProviderRuntimeManager(first_settings, runtime_factory=factory)
    old_lease = await manager.acquire()
    committed: list[str] = []

    generation_id = await manager.replace(
        second_settings,
        commit=AsyncMock(side_effect=lambda: committed.append("persisted")),
    )
    new_lease = await manager.acquire()

    assert generation_id == 2
    assert committed == ["persisted"]
    assert new_lease.generation_id == 2
    assert new_lease.settings is second_settings
    assert factory.runtimes[0].cleanup_calls == 0
    await new_lease.release()
    await old_lease.release()
    assert factory.runtimes[0].cleanup_calls == 1
    await manager.close()
    assert factory.runtimes[1].cleanup_calls == 1


@pytest.mark.asyncio
async def test_hot_replacement_owns_admission_per_provider_generation() -> None:
    first_settings = _settings("nvidia_nim/one")
    second_settings = _settings("nvidia_nim/two")
    clients: list[MagicMock] = []

    def create_client(*_args: object, **_kwargs: object) -> MagicMock:
        client = MagicMock()
        client.close = AsyncMock()
        clients.append(client)
        return client

    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI",
        side_effect=create_client,
    ):
        manager = ProviderRuntimeManager(first_settings)
        old_lease = await manager.acquire()
        old_provider = old_lease.resolve_provider("nvidia_nim")
        refresh = AsyncMock()

        with patch.object(manager, "_refresh_generation", refresh):
            await manager.replace(second_settings, commit=AsyncMock())
            new_lease = await manager.acquire()
            new_provider = new_lease.resolve_provider("nvidia_nim")
            await asyncio.sleep(0)

            assert isinstance(old_provider, NvidiaNimProvider)
            assert isinstance(new_provider, NvidiaNimProvider)
            assert new_provider is not old_provider
            assert new_provider._admission is not old_provider._admission
            assert old_lease.resolve_provider("nvidia_nim") is old_provider
            clients[0].close.assert_not_awaited()

            await new_lease.release()
            await old_lease.release()

            clients[0].close.assert_awaited_once()
            clients[1].close.assert_not_awaited()
            refresh.assert_awaited_once()
            await manager.close()

        clients[1].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_replacement_closes_unleased_generation_immediately() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )

    await manager.replace(
        _settings("nvidia_nim/two"),
        commit=AsyncMock(),
    )

    assert factory.runtimes[0].cleanup_calls == 1
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_replacement_does_not_cancel_owned_generation_cleanup() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    refresh_started = asyncio.Event()
    factory.runtimes[0].cleanup_started = cleanup_started
    factory.runtimes[0].cleanup_release = cleanup_release

    async def refresh(*_args: object, **_kwargs: object) -> None:
        refresh_started.set()
        await asyncio.Event().wait()

    with patch.object(manager, "_refresh_generation", side_effect=refresh):
        replace_task = asyncio.create_task(
            manager.replace(
                _settings("nvidia_nim/two"),
                commit=AsyncMock(),
            )
        )
        await cleanup_started.wait()
        await refresh_started.wait()

        replace_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replace_task

        retired = manager._retired[1]
        assert manager.current_generation_id == 2
        assert retired.cleanup_task is not None
        assert not retired.cleanup_task.cancelled()
        assert factory.runtimes[0].cleanup_calls == 1

        close_task = asyncio.create_task(manager.close())
        await asyncio.sleep(0)
        assert not close_task.done()
        cleanup_release.set()
        await close_task

    assert factory.runtimes[0].cleanup_calls == 1
    assert factory.runtimes[1].cleanup_calls == 1
    assert manager._retired == {}


@pytest.mark.asyncio
async def test_cancelled_final_lease_release_keeps_owned_cleanup_running() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    lease = await manager.acquire()
    await manager.replace(
        _settings("nvidia_nim/two"),
        commit=AsyncMock(),
    )
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    factory.runtimes[0].cleanup_started = cleanup_started
    factory.runtimes[0].cleanup_release = cleanup_release

    release_task = asyncio.create_task(lease.release())
    await cleanup_started.wait()
    release_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await release_task

    retired = manager._retired[1]
    assert retired.active_leases == 0
    assert retired.cleanup_task is not None
    assert not retired.cleanup_task.cancelled()
    assert factory.runtimes[0].cleanup_calls == 1

    close_task = asyncio.create_task(manager.close())
    await asyncio.sleep(0)
    assert not close_task.done()
    cleanup_release.set()
    await close_task

    assert factory.runtimes[0].cleanup_calls == 1
    assert manager._retired == {}


@pytest.mark.asyncio
async def test_hot_cleanup_failure_keeps_published_replacement() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    factory.runtimes[0].cleanup_error = RuntimeError("cleanup failed")

    generation_id = await manager.replace(
        _settings("nvidia_nim/two"),
        commit=AsyncMock(),
    )

    assert generation_id == 2
    assert manager.current_generation_id == 2
    assert factory.runtimes[0].cleanup_calls == 1
    assert 1 in manager._retired

    factory.runtimes[0].cleanup_error = None
    await manager.close()

    assert factory.runtimes[0].cleanup_calls == 2
    assert manager._retired == {}


@pytest.mark.asyncio
async def test_candidate_construction_failure_preserves_current_generation() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    factory.error = RuntimeError("cannot construct")

    with pytest.raises(RuntimeError, match="cannot construct"):
        await manager.replace(
            _settings("nvidia_nim/two"),
            commit=AsyncMock(),
        )

    assert manager.current_generation_id == 1
    assert manager.current_settings().model == "nvidia_nim/one"
    assert factory.runtimes[0].cleanup_calls == 0
    await manager.close()


@pytest.mark.asyncio
async def test_failed_candidate_cleanup_is_retried_at_shutdown() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )

    async def fail_commit() -> None:
        factory.runtimes[1].cleanup_error = RuntimeError("private cleanup detail")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        await manager.replace(
            _settings("nvidia_nim/two"),
            commit=fail_commit,
        )

    assert manager.current_generation_id == 1
    assert factory.runtimes[0].cleanup_calls == 0
    assert factory.runtimes[1].cleanup_calls == 1
    assert manager._unpublished == {factory.runtimes[1]}

    manager.cache_model_infos("lmstudio", {ProviderModelInfo("cached")})
    with pytest.raises(
        RuntimeError,
        match="One or more provider runtimes failed to close",
    ) as exc_info:
        await manager.close()

    assert "private cleanup detail" not in str(exc_info.value)
    assert manager._closed is False
    assert manager._unpublished == {factory.runtimes[1]}
    assert manager.cached_model_ids() == {"lmstudio": frozenset({"cached"})}

    factory.runtimes[1].cleanup_error = None
    await manager.close()

    assert factory.runtimes[0].cleanup_calls == 1
    assert factory.runtimes[1].cleanup_calls == 3
    assert manager._unpublished == set()
    assert manager.cached_model_ids() == {}
    assert manager._closed is True


@pytest.mark.asyncio
async def test_later_replacement_retries_failed_unpublished_candidate() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )

    async def fail_commit() -> None:
        factory.runtimes[1].cleanup_error = RuntimeError("cleanup failed")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        await manager.replace(
            _settings("nvidia_nim/two"),
            commit=fail_commit,
        )

    factory.runtimes[1].cleanup_error = None
    generation_id = await manager.replace(
        _settings("nvidia_nim/three"),
        commit=AsyncMock(),
    )

    assert generation_id == 2
    assert factory.runtimes[1].cleanup_calls == 2
    assert manager._unpublished == set()
    await manager.close()


@pytest.mark.asyncio
async def test_cancelled_candidate_cleanup_remains_owned_until_shutdown() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def fail_commit() -> None:
        candidate = factory.runtimes[1]
        candidate.cleanup_started = cleanup_started
        candidate.cleanup_release = cleanup_release
        raise OSError("disk full")

    replace_task = asyncio.create_task(
        manager.replace(
            _settings("nvidia_nim/two"),
            commit=fail_commit,
        )
    )
    await cleanup_started.wait()
    replace_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await replace_task

    assert manager._unpublished == {factory.runtimes[1]}
    assert factory.runtimes[1].cleanup_calls == 1

    cleanup_release.set()
    await manager.close()

    assert factory.runtimes[1].cleanup_calls == 2
    assert manager._unpublished == set()


@pytest.mark.asyncio
async def test_concurrent_replacements_are_serialized_in_call_order() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    cancel_calls = 0
    original_cancel = manager._cancel_refresh

    async def controlled_cancel() -> None:
        nonlocal cancel_calls
        cancel_calls += 1
        if cancel_calls == 1:
            first_entered.set()
            await release_first.wait()
        await original_cancel()

    with patch.object(manager, "_cancel_refresh", side_effect=controlled_cancel):
        first = asyncio.create_task(
            manager.replace(
                _settings("nvidia_nim/two"),
                commit=AsyncMock(),
            )
        )
        await first_entered.wait()
        second = asyncio.create_task(
            manager.replace(
                _settings("nvidia_nim/three"),
                commit=AsyncMock(),
            )
        )
        await asyncio.sleep(0)
        assert len(factory.runtimes) == 1
        assert not second.done()
        release_first.set()
        assert await asyncio.gather(first, second) == [2, 3]

    assert manager.current_settings().model == "nvidia_nim/three"
    assert [runtime.cleanup_calls for runtime in factory.runtimes[:2]] == [1, 1]
    await manager.close()


@pytest.mark.asyncio
async def test_shutdown_waits_for_active_lease_then_rejects_new_work() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    lease = await manager.acquire()

    close_task = asyncio.create_task(manager.close())
    await asyncio.sleep(0)

    assert not close_task.done()
    with pytest.raises(ApplicationUnavailableError, match="shutting down"):
        await manager.acquire()
    await lease.release()
    await close_task
    assert factory.runtimes[0].cleanup_calls == 1


@pytest.mark.asyncio
async def test_cancelled_shutdown_retains_generation_for_retry() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    lease = await manager.acquire()
    close_task = asyncio.create_task(manager.close())
    await asyncio.sleep(0)

    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    with pytest.raises(ApplicationUnavailableError, match="shutting down"):
        await manager.acquire()
    with pytest.raises(ApplicationUnavailableError, match="shutting down"):
        await manager.replace(_settings("nvidia_nim/two"), commit=AsyncMock())
    assert factory.runtimes[0].cleanup_calls == 0

    await lease.release()
    await manager.close()

    assert factory.runtimes[0].cleanup_calls == 1
    assert manager._closed is True


@pytest.mark.asyncio
async def test_cancelled_shutdown_reuses_the_same_owned_cleanup_task() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_calls = 0

    async def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_started.set()
        await cleanup_release.wait()

    with patch.object(factory.runtimes[0], "cleanup", side_effect=cleanup):
        close_task = asyncio.create_task(manager.close())
        await cleanup_started.wait()

        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        assert manager._closed is False
        assert manager._retired

        retry_task = asyncio.create_task(manager.close())
        await asyncio.sleep(0)
        assert not retry_task.done()
        cleanup_release.set()
        await retry_task

    assert cleanup_calls == 1
    assert manager._retired == {}
    assert manager._closed is True


@pytest.mark.asyncio
async def test_failed_shutdown_cleanup_is_retryable() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    manager.cache_model_infos("lmstudio", {ProviderModelInfo("cached")})
    factory.runtimes[0].cleanup_error = RuntimeError("private provider detail")

    with pytest.raises(
        RuntimeError,
        match="One or more provider runtimes failed to close",
    ) as exc_info:
        await manager.close()

    assert "private provider detail" not in str(exc_info.value)
    assert manager._closed is False
    assert 1 in manager._retired
    assert manager.cached_model_ids() == {"lmstudio": frozenset({"cached"})}
    assert factory.runtimes[0].cleanup_calls == 1

    factory.runtimes[0].cleanup_error = None
    await manager.close()

    assert factory.runtimes[0].cleanup_calls == 2
    assert manager._retired == {}
    assert manager.cached_model_ids() == {}
    assert manager._closed is True


@pytest.mark.asyncio
async def test_application_catalog_survives_generation_replacement() -> None:
    factory = RuntimeFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/one"),
        runtime_factory=factory,
    )
    manager.cache_model_infos(
        "lmstudio",
        {ProviderModelInfo("persisted", supports_thinking=True)},
    )

    await manager.replace(
        _settings("nvidia_nim/two"),
        commit=AsyncMock(),
    )

    assert manager.cached_model_ids() == {"lmstudio": frozenset({"persisted"})}
    assert manager.cached_model_info("lmstudio", "persisted") == ProviderModelInfo(
        "persisted",
        supports_thinking=True,
    )
    await manager.close()


@pytest.mark.asyncio
async def test_generation_lease_keeps_its_model_metadata_after_replacement() -> None:
    factory = RuntimeFactory()
    first_settings = _settings("open_router/one").model_copy(
        update={"open_router_api_key": "open-router-key"}
    )
    manager = ProviderRuntimeManager(first_settings, runtime_factory=factory)
    manager.cache_model_infos(
        "open_router",
        {
            ProviderModelInfo(
                "old-model",
                context_window_tokens=32_000,
                max_output_tokens=4_096,
            )
        },
    )
    lease = await manager.acquire(include_model_infos=True)

    await manager.replace(
        _settings("nvidia_nim/two"),
        commit=AsyncMock(),
    )

    assert lease.model_infos == (
        ProviderModelInfo(
            "open_router/old-model",
            context_window_tokens=32_000,
            max_output_tokens=4_096,
        ),
    )
    await lease.release()
    await manager.close()


@pytest.mark.asyncio
async def test_replacement_prunes_and_rejects_removed_remote_provider_cache() -> None:
    factory = RuntimeFactory()
    first_settings = _settings("open_router/one").model_copy(
        update={"open_router_api_key": "open-router-key"}
    )
    manager = ProviderRuntimeManager(first_settings, runtime_factory=factory)
    manager.cache_model_infos("open_router", {ProviderModelInfo("old-model")})

    await manager.replace(
        _settings("nvidia_nim/two"),
        commit=AsyncMock(),
    )
    manager.cache_model_infos("open_router", {ProviderModelInfo("late-old-model")})

    assert "open_router" not in manager.cached_model_ids()
    await manager.close()


@pytest.mark.asyncio
async def test_generation_lifecycle_traces_contain_minimal_correlation_fields() -> None:
    factory = RuntimeFactory()

    with patch("free_claude_code.runtime.provider_manager.trace_event") as trace:
        manager = ProviderRuntimeManager(
            _settings("nvidia_nim/one"),
            runtime_factory=factory,
        )
        lease = await manager.acquire()
        await manager.replace(
            _settings("nvidia_nim/two"),
            commit=AsyncMock(),
            reason="test_replace",
        )
        await lease.release()
        await manager.close()

    events = [call.kwargs for call in trace.call_args_list]
    names = [event["event"] for event in events]
    assert names == [
        "provider_generation.published",
        "provider_generation.published",
        "provider_generation.retired",
        "provider_generation.closed",
        "provider_generation.retired",
        "provider_generation.closed",
    ]
    assert events[1]["generation_id"] == 2
    assert events[1]["previous_generation_id"] == 1
    assert events[1]["reason"] == "test_replace"
    assert events[2]["active_leases"] == 1
    assert events[3]["forced"] is False

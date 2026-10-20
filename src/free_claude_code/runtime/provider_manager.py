"""Single-owner provider generations and progressively available model metadata."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Protocol

from loguru import logger

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import (
    ProviderModelInfo,
    ProviderModelRefreshResult,
)
from free_claude_code.application.ports import ModelCatalogPort, ModelCatalogSnapshot
from free_claude_code.application.readiness import InitializationWait
from free_claude_code.config.model_refs import split_provider_model_ref
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.core.async_tasks import run_sync_owned
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.token_estimation import initialize_token_estimation
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission_policy import ProviderAdmissionLimits
from free_claude_code.providers.admission_registry import ProviderAdmissionRegistry
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.model_listing import model_infos_from_ids
from free_claude_code.providers.runtime.discovery import (
    _provider_query_failure_reason,
    is_optional_local_failure,
    is_optional_local_provider,
    model_cache_provider_ids_for_settings,
    model_list_provider_ids_for_settings,
    referenced_provider_ids,
)
from free_claude_code.providers.runtime.model_cache import ProviderModelCache
from free_claude_code.providers.runtime.runtime import ProviderRuntime

ProviderRuntimeFactory = Callable[
    [Settings, ProviderAdmissionRegistry], ProviderRuntime
]
ConnectedProviderIds = Callable[[], tuple[str, ...]]
CommitConfig = Callable[[], Awaitable[None]]

LOCAL_DISCOVERY_TIMEOUT_SECONDS = 5.0


class ModelCatalogPublisher(Protocol):
    """Publish a frozen model inventory without reading loop-owned mutable state."""

    def publish(self, runtime: ModelCatalogPort) -> None: ...


@dataclass(slots=True, eq=False)
class _ProviderGeneration:
    generation_id: int
    settings: Settings
    runtime: ProviderRuntime
    cache: ProviderModelCache
    active_leases: int = 0
    retired: bool = False
    closed: bool = False
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    replaced: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_task: asyncio.Task[bool] | None = None
    catalog_tasks: dict[str, asyncio.Task[ProviderModelRefreshResult]] = field(
        default_factory=dict
    )
    initialized: set[str] = field(default_factory=set)
    initial_ids: tuple[str, ...] = ()
    catalog_started: bool = False
    initial_complete: bool = False
    refresh_task: asyncio.Task[ProviderModelRefreshResult] | None = None
    file_state: str = "starting"

    def __post_init__(self) -> None:
        for definition in self.settings.custom_providers:
            if definition.model_ids:
                self.cache.cache_model_infos(
                    definition.provider_id, model_infos_from_ids(definition.model_ids)
                )
                self.initialized.add(definition.provider_id)
        self.drained.set()

    def snapshot(self) -> ModelCatalogSnapshot:
        return ModelCatalogSnapshot(
            self.settings, self.cache.cached_prefixed_model_infos()
        )


class ProviderGenerationLease:
    """Retain one generation and freeze metadata only after its provider is ready."""

    def __init__(
        self, manager: ProviderRuntimeManager, generation: _ProviderGeneration
    ) -> None:
        self._manager = manager
        self._generation = generation
        self._wait = InitializationWait()
        self._model_infos: dict[str, dict[str, ProviderModelInfo]] = {}
        self._released = False

    @property
    def generation_id(self) -> int:
        return self._generation.generation_id

    @property
    def settings(self) -> Settings:
        return self._generation.settings

    def is_provider_cached(self, provider_id: str, api_key_number: int = 1) -> bool:
        return self._generation.runtime.is_cached(provider_id, api_key_number)

    async def wait_for_token_estimation(self) -> None:
        self._manager._ensure_open()
        await self._wait.wait(self._manager._ensure_encoder())

    async def resolve_provider(
        self, provider_id: str, api_key_number: int = 1
    ) -> BaseProvider:
        self._manager._ensure_open()
        generation = self._generation
        if provider_id not in generation.initialized:
            task = self._manager._catalog_task(generation, provider_id)
            await self._wait.wait(task)
        if generation.runtime.is_cached(provider_id, api_key_number):
            provider = await generation.runtime.resolve_provider(
                provider_id, api_key_number
            )
        else:
            waiting = asyncio.create_task(
                generation.runtime.resolve_provider(provider_id, api_key_number)
            )
            try:
                provider = await self._wait.wait(waiting)
            finally:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
        if provider_id not in self._model_infos:
            self._model_infos[provider_id] = {
                info.model_id: replace(info, model_id=f"{provider_id}/{info.model_id}")
                for info in generation.cache.provider_infos(provider_id)
            }
        return provider

    def model_info(self, provider_id: str, model_id: str) -> ProviderModelInfo | None:
        return self._model_infos.get(provider_id, {}).get(model_id)

    async def release(self) -> None:
        if not self._released:
            self._released = True
            await self._manager._release(self._generation)

    async def __aenter__(self) -> ProviderGenerationLease:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.release()


class ProviderRuntimeManager:
    """Own provider lifetimes, shared initialization and current catalog publication."""

    def __init__(
        self,
        settings: Settings,
        *,
        runtime_factory: ProviderRuntimeFactory = ProviderRuntime,
        connected_provider_ids: ConnectedProviderIds = tuple,
        model_catalog_publisher: ModelCatalogPublisher | None = None,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._admission_registry = ProviderAdmissionRegistry(
            ProviderAdmissionLimits.from_settings(settings)
        )
        self._connected_provider_ids = connected_provider_ids
        self._model_catalog_publisher = model_catalog_publisher
        self._catalog_changed: Callable[[], None] | None = None
        self._replace_lock = asyncio.Lock()
        self._publication_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._next_generation_id = 2
        self._retired: dict[int, _ProviderGeneration] = {}
        self._unpublished: set[ProviderRuntime] = set()
        self._publication_task: asyncio.Task[None] | None = None
        self._published_version: tuple[int, int] | None = None
        self._encoder_task: asyncio.Task[object] | None = None
        self._catalog_revision = 0
        self._closing = False
        self._closed = False
        self._current = _ProviderGeneration(
            generation_id=1,
            settings=settings,
            runtime=runtime_factory(settings, self._admission_registry),
            cache=ProviderModelCache(
                model_cache_provider_ids_for_settings(
                    settings, connected_provider_ids()
                )
            ),
        )
        self._trace_published(self._current, previous=None, reason="startup")

    @property
    def current_generation_id(self) -> int:
        return self._current.generation_id

    def set_catalog_changed_callback(self, callback: Callable[[], None] | None) -> None:
        """Register a nonblocking notification for application-owned consumers."""
        self._catalog_changed = callback

    def _ensure_open(self) -> None:
        if self._closing or self._closed:
            raise ApplicationUnavailableError("Provider runtime is shutting down.")

    def _ensure_encoder(self) -> asyncio.Task[object]:
        if self._encoder_task is None:
            self._encoder_task = asyncio.create_task(
                run_sync_owned(initialize_token_estimation)
            )
        return self._encoder_task

    async def acquire(self) -> ProviderGenerationLease:
        self._ensure_open()
        generation = self._current
        generation.active_leases += 1
        generation.drained.clear()
        return ProviderGenerationLease(self, generation)

    def current_settings(self) -> Settings:
        return self._current.settings

    def _synchronize_model_cache_scope(self) -> None:
        self._current.cache.set_available_providers(
            model_cache_provider_ids_for_settings(
                self._current.settings, self._connected_provider_ids()
            )
        )

    def cached_model_ids(self) -> dict[str, frozenset[str]]:
        self._synchronize_model_cache_scope()
        return self._current.cache.cached_model_ids()

    def cached_model_info(
        self, provider_id: str, model_id: str
    ) -> ProviderModelInfo | None:
        self._synchronize_model_cache_scope()
        return self._current.cache.cached_model_info(provider_id, model_id)

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        self._synchronize_model_cache_scope()
        return self._current.cache.cached_prefixed_model_infos()

    def catalog_snapshot(self) -> ModelCatalogSnapshot:
        self._ensure_open()
        self._synchronize_model_cache_scope()
        return self._current.snapshot()

    def cache_model_infos(
        self, provider_id: str, model_infos: Iterable[ProviderModelInfo]
    ) -> None:
        self._cache_result(self._current, provider_id, model_infos)

    def _cache_result(
        self,
        generation: _ProviderGeneration,
        provider_id: str,
        infos: Iterable[ProviderModelInfo],
    ) -> None:
        before = generation.cache.provider_infos(provider_id)
        generation.cache.cache_model_infos(provider_id, infos)
        if generation is self._current and before != generation.cache.provider_infos(
            provider_id
        ):
            self._catalog_revision += 1
            self._catalog_updated()

    def _catalog_task(
        self,
        generation: _ProviderGeneration,
        provider_id: str,
        *,
        refresh: bool = False,
    ) -> asyncio.Task[ProviderModelRefreshResult]:
        task = generation.catalog_tasks.get(provider_id)
        if task is None or (refresh and task.done()):
            task = asyncio.create_task(self._discover_provider(generation, provider_id))
            generation.catalog_tasks[provider_id] = task
        return task

    async def _discover_provider(
        self, generation: _ProviderGeneration, provider_id: str
    ) -> ProviderModelRefreshResult:
        definition = generation.settings.custom_provider(provider_id)
        if definition is not None and definition.model_ids:
            generation.initialized.add(provider_id)
            return ProviderModelRefreshResult(refreshed_provider_ids=(provider_id,))
        try:
            provider = await generation.runtime.resolve_provider(provider_id)
            from free_claude_code.providers.admission import (
                quiet_discovery_recovery_logs,
            )

            descriptor = PROVIDER_CATALOG.get(provider_id)
            timeout = (
                LOCAL_DISCOVERY_TIMEOUT_SECONDS
                if descriptor is not None and descriptor.local
                else None
            )
            with quiet_discovery_recovery_logs(
                is_optional_local_provider(generation.settings, provider_id)
            ):
                async with asyncio.timeout(timeout):
                    infos = await provider.list_model_infos()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = _provider_query_failure_reason(exc)
            log = (
                logger.debug
                if is_optional_local_failure(generation.settings, provider_id, message)
                else logger.warning
            )
            log(
                "Provider model discovery skipped: provider={} reason={}",
                provider_id,
                message,
            )
            result = ProviderModelRefreshResult(
                failed_provider_ids=(provider_id,),
                failure_messages={provider_id: message},
            )
        else:
            self._cache_result(generation, provider_id, infos)
            logger.info(
                "Provider model discovery cached: provider={} models={}",
                provider_id,
                len(infos),
            )
            result = ProviderModelRefreshResult(refreshed_provider_ids=(provider_id,))
        generation.initialized.add(provider_id)
        return result

    def _start_pass(
        self, generation: _ProviderGeneration, *, refresh: bool = False
    ) -> asyncio.Task[ProviderModelRefreshResult]:
        if not generation.catalog_started:
            generation.catalog_started = True
            generation.initial_ids = tuple(
                dict.fromkeys(
                    (
                        *referenced_provider_ids(generation.settings),
                        *model_list_provider_ids_for_settings(
                            generation.settings, self._connected_provider_ids()
                        ),
                    )
                )
            )
            if generation is self._current:
                self._catalog_updated()
        elif (
            refresh
            and generation is self._current
            and self._published_version
            != (generation.generation_id, self._catalog_revision)
        ):
            self._request_publication()
        if generation.refresh_task is not None and not generation.refresh_task.done():
            return generation.refresh_task
        for provider_id in generation.initial_ids:
            self._catalog_task(generation, provider_id, refresh=refresh)
        generation.refresh_task = asyncio.create_task(self._finish_pass(generation))
        return generation.refresh_task

    async def _finish_pass(
        self, generation: _ProviderGeneration
    ) -> ProviderModelRefreshResult:
        while True:
            tasks = tuple(
                self._catalog_task(generation, provider_id)
                for provider_id in generation.initial_ids
            )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            if tasks != tuple(
                generation.catalog_tasks[provider_id]
                for provider_id in generation.initial_ids
            ):
                continue
            for result in results:
                if isinstance(result, asyncio.CancelledError):
                    raise result
            generation.initial_complete = True
            if generation.initial_complete and tasks == tuple(
                generation.catalog_tasks[provider_id]
                for provider_id in generation.initial_ids
            ):
                break
        failures = {
            provider: message
            for result in results
            if isinstance(result, ProviderModelRefreshResult)
            for provider, message in result.failure_messages.items()
        }
        failed_ids = tuple(
            provider
            for result in results
            if isinstance(result, ProviderModelRefreshResult)
            for provider in result.failed_provider_ids
            if not is_optional_local_failure(
                generation.settings, provider, failures.get(provider)
            )
        )
        return ProviderModelRefreshResult(
            refreshed_provider_ids=tuple(
                provider
                for result in results
                if isinstance(result, ProviderModelRefreshResult)
                for provider in result.refreshed_provider_ids
            ),
            failed_provider_ids=failed_ids,
            failure_messages={provider: failures[provider] for provider in failed_ids},
        )

    def start_model_list_refresh(self) -> None:
        self._ensure_open()
        self._ensure_encoder()
        self._start_pass(self._current)

    async def refresh_model_list_cache(self) -> ProviderModelRefreshResult:
        self._ensure_open()
        async with self._replace_lock:
            task = self._start_pass(self._current, refresh=True)
        return await asyncio.shield(task)

    async def refresh_provider(self, provider_id: str) -> ProviderModelRefreshResult:
        self._ensure_open()
        generation = self._current
        generation.active_leases += 1
        generation.drained.clear()
        try:
            task = self._catalog_task(generation, provider_id, refresh=True)
            return await asyncio.shield(task)
        finally:
            await self._release(generation)

    async def wait_for_model_catalog(self, model: str) -> ModelCatalogSnapshot:
        """Wait only for a selection's metadata, following settings replacement."""
        wait = InitializationWait()
        while True:
            snapshot = self.catalog_snapshot()
            try:
                provider_id, _ = split_provider_model_ref(model)
            except ValueError:
                return snapshot
            generation = self._current
            if provider_id not in (
                *referenced_provider_ids(generation.settings),
                *model_list_provider_ids_for_settings(
                    generation.settings, self._connected_provider_ids()
                ),
            ):
                return snapshot
            if provider_id not in generation.initialized:
                try:
                    await self._wait_for_catalog_work(
                        generation, self._catalog_task(generation, provider_id), wait
                    )
                except ApplicationUnavailableError:
                    if generation is self._current:
                        raise
                    continue
            if generation is self._current:
                return self.catalog_snapshot()

    async def _wait_for_catalog_work[T](
        self,
        generation: _ProviderGeneration,
        task: asyncio.Task[T],
        wait: InitializationWait,
    ) -> None:
        if generation is not self._current:
            return
        if task.done():
            await wait.wait(task)
            return
        replaced = asyncio.create_task(generation.replaced.wait())
        settled = asyncio.create_task(
            asyncio.wait((task, replaced), return_when=asyncio.FIRST_COMPLETED)
        )
        try:
            await wait.wait(settled)
            if generation is self._current:
                await wait.wait(task)
        finally:
            # These waits belong to the caller; discovery belongs to its generation.
            settled.cancel()
            replaced.cancel()
            await asyncio.gather(settled, replaced, return_exceptions=True)

    async def wait_for_catalog_file(self, wait: InitializationWait) -> tuple[int, int]:
        while True:
            self._ensure_open()
            generation = self._current
            version = (generation.generation_id, self._catalog_revision)
            if self._published_version != version:
                try:
                    await self._wait_for_catalog_work(
                        generation, self._request_publication(), wait
                    )
                except ApplicationUnavailableError:
                    if generation is self._current:
                        raise
                    continue
            if generation is not self._current:
                continue
            if generation.file_state == "failed":
                raise ApplicationUnavailableError(
                    "Could not publish the Codex model catalog. Check file permissions and retry."
                )
            if self._published_version == (
                generation.generation_id,
                self._catalog_revision,
            ):
                return self._published_version

    def catalog_status(self) -> JsonObject:
        generation = self._current
        providers: JsonObject = {}
        errors: JsonObject = {}
        for provider_id in dict.fromkeys(
            (*generation.initial_ids, *generation.catalog_tasks)
        ):
            task = generation.catalog_tasks.get(provider_id)
            if task is None or not task.done():
                state = "starting"
            elif task.cancelled() or task.exception() is not None:
                state = "failed"
            else:
                result = task.result()
                message = result.failure_messages.get(provider_id)
                state = "failed" if result.failed_provider_ids else "ready"
                if is_optional_local_failure(generation.settings, provider_id, message):
                    state = "unavailable"
                elif message:
                    errors[provider_id] = message
            providers[provider_id] = state
        return {
            "generation_id": generation.generation_id,
            "catalog_revision": self._catalog_revision,
            "catalog": "ready" if generation.initial_complete else "starting",
            "catalog_file": generation.file_state,
            "providers": providers,
            "provider_errors": errors,
        }

    async def connected_provider_changed(
        self, provider_id: str, *, connected: bool
    ) -> ProviderModelRefreshResult:
        async with self._replace_lock, self._publication_lock:
            self._ensure_open()
            generation = self._current
            self._start_pass(generation)
            generation.initial_ids = tuple(
                p for p in generation.initial_ids if p != provider_id
            )
            task = generation.catalog_tasks.pop(provider_id, None)
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            generation.initialized.discard(provider_id)
            generation.cache.remove_provider(provider_id)
            if connected:
                generation.cache.add_provider(provider_id)
                generation.initial_ids += (provider_id,)
                self._catalog_task(generation, provider_id)
            generation.initial_complete = False
            generation.file_state = "starting"
            self._catalog_revision += 1
            self._start_pass(generation)
            self._catalog_updated()
        return ProviderModelRefreshResult()

    def _catalog_updated(self) -> None:
        if self._closing or not self._current.catalog_started:
            return
        self._request_publication()
        if self._catalog_changed is not None and not self._closing:
            self._catalog_changed()

    def _request_publication(self) -> asyncio.Task[None]:
        self._current.file_state = "starting"
        if self._publication_task is None or self._publication_task.done():
            self._publication_task = asyncio.create_task(
                self._publish_catalog(), name="fcc-model-catalog"
            )
        return self._publication_task

    async def _publish_catalog(self) -> None:
        while not self._closing:
            async with self._publication_lock:
                if self._closing:
                    return
                generation = self._current
                version = (generation.generation_id, self._catalog_revision)
                try:
                    publisher = self._model_catalog_publisher
                    if publisher is not None and self._published_version != version:
                        snapshot = generation.snapshot()
                        await run_sync_owned(partial(publisher.publish, snapshot))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    generation.file_state = "failed"
                    logger.warning(
                        "Model catalog publication failed: exc_type={}",
                        type(exc).__name__,
                    )
                    if version[1] == self._catalog_revision:
                        return
                else:
                    self._published_version = version
                    if version[1] == self._catalog_revision:
                        generation.file_state = "ready"
                        return
                generation.file_state = "starting"

    async def replace(
        self, settings: Settings, *, commit: CommitConfig, reason: str = "admin_apply"
    ) -> int:
        async with self._replace_lock:
            self._ensure_open()
            await self._retry_unpublished_cleanup()
            limits = ProviderAdmissionLimits.from_settings(settings)
            candidate_id = self._next_generation_id
            candidate_runtime: ProviderRuntime | None = None
            try:
                candidate_runtime = self._runtime_factory(
                    settings, self._admission_registry
                )
                await commit()
            except BaseException as exc:
                trace_event(
                    stage="runtime",
                    event="provider_generation.replace_failed",
                    source="runtime",
                    current_generation_id=self._current.generation_id,
                    candidate_generation_id=candidate_id,
                    reason=reason,
                    exc_type=type(exc).__name__,
                )
                if candidate_runtime is not None:
                    await self._cleanup_unpublished(candidate_runtime)
                raise
            self._next_generation_id += 1
            assert candidate_runtime is not None
            async with self._publication_lock:
                previous = self._current
                cache = previous.cache.copy(
                    model_cache_provider_ids_for_settings(
                        settings, self._connected_provider_ids()
                    )
                )
                for provider_id in cache.cached_model_ids():
                    descriptor = PROVIDER_CATALOG.get(provider_id)
                    if (
                        descriptor is not None
                        and descriptor.local
                        and any(
                            attr is not None
                            and getattr(previous.settings, attr)
                            != getattr(settings, attr)
                            for attr in (
                                descriptor.base_url_attr,
                                descriptor.proxy_attr,
                            )
                        )
                    ):
                        cache.remove_provider(provider_id)
                        cache.add_provider(provider_id)
                for definition in settings.custom_providers:
                    old = previous.settings.custom_provider(definition.provider_id)
                    if old is None or not definition.same_inventory(old):
                        cache.remove_provider(definition.provider_id)
                        cache.add_provider(definition.provider_id)
                candidate = _ProviderGeneration(
                    generation_id=candidate_id,
                    settings=settings,
                    runtime=candidate_runtime,
                    cache=cache,
                )
                self._admission_registry.reconfigure(limits)
                self._current = candidate
                self._catalog_revision += 1
                previous.retired = True
                previous.replaced.set()
                self._retired[previous.generation_id] = previous
                self._trace_published(candidate, previous=previous, reason=reason)
                self._trace_retired(previous, reason=reason)
                self._start_pass(candidate)
            if previous.active_leases == 0:
                await self._close_generation(previous, forced=False)
            return candidate.generation_id

    def begin_shutdown(self) -> None:
        self._closing = True
        for generation in (self._current, *self._retired.values()):
            if generation.refresh_task is not None:
                generation.refresh_task.cancel()
            for task in generation.catalog_tasks.values():
                task.cancel()

    async def _cancel_generation_work(self, generation: _ProviderGeneration) -> None:
        tasks = [*generation.catalog_tasks.values()]
        if generation.refresh_task is not None:
            tasks.append(generation.refresh_task)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self.begin_shutdown()
            async with self._replace_lock:
                current = self._current
                if not current.retired:
                    current.retired = True
                    self._retired[current.generation_id] = current
                    self._trace_retired(current, reason="shutdown")
                generations = tuple(self._retired.values())
            await asyncio.gather(
                *(self._cancel_generation_work(g) for g in generations)
            )
            if self._publication_task is not None:
                await asyncio.gather(self._publication_task, return_exceptions=True)
            if self._encoder_task is not None:
                self._encoder_task.cancel()
                await asyncio.gather(self._encoder_task, return_exceptions=True)
            await asyncio.gather(*(g.drained.wait() for g in generations))
            generation_results = await asyncio.gather(
                *(self._close_generation(g, forced=False) for g in generations)
            )
            unpublished_closed = await self._retry_unpublished_cleanup()
            if not all(generation_results) or not unpublished_closed:
                raise RuntimeError("One or more provider runtimes failed to close.")
            self._current.cache.clear()
            self._admission_registry.close()
            self._closed = True

    def _prune_admission(self) -> None:
        runtimes = (
            self._current.runtime,
            *(generation.runtime for generation in self._retired.values()),
            *self._unpublished,
        )
        self._admission_registry.retain_custom(
            definition.provider_id
            for runtime in runtimes
            for definition in runtime.settings.custom_providers
        )

    async def _release(self, generation: _ProviderGeneration) -> None:
        if generation.active_leases <= 0:
            return
        generation.active_leases -= 1
        if generation.active_leases == 0:
            generation.drained.set()
            if generation.retired and not self._closing:
                await self._close_generation(generation, forced=False)

    async def _cleanup_unpublished(self, runtime: ProviderRuntime) -> bool:
        self._unpublished.add(runtime)
        try:
            await runtime.cleanup()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Unpublished provider generation cleanup failed: exc_type={}",
                type(exc).__name__,
            )
            return False
        self._unpublished.discard(runtime)
        self._prune_admission()
        return True

    async def _retry_unpublished_cleanup(self) -> bool:
        all_closed = True
        for runtime in tuple(self._unpublished):
            if not await self._cleanup_unpublished(runtime):
                all_closed = False
        return all_closed

    async def _close_generation(
        self,
        generation: _ProviderGeneration,
        *,
        forced: bool,
    ) -> bool:
        if generation.closed:
            return True
        if generation.active_leases != 0:
            return False
        task = generation.cleanup_task
        if task is None:
            task = asyncio.create_task(
                self._run_generation_cleanup(generation, forced=forced),
                name=f"provider-generation-cleanup-{generation.generation_id}",
            )
            generation.cleanup_task = task
        return await asyncio.shield(task)

    async def _run_generation_cleanup(
        self,
        generation: _ProviderGeneration,
        *,
        forced: bool,
    ) -> bool:
        task = asyncio.current_task()
        try:
            try:
                await self._cancel_generation_work(generation)
                await generation.runtime.cleanup()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Provider generation cleanup failed: generation_id={} exc_type={}",
                    generation.generation_id,
                    type(exc).__name__,
                )
                return False

            generation.closed = True
            self._retired.pop(generation.generation_id, None)
            self._prune_admission()
            trace_event(
                stage="runtime",
                event="provider_generation.closed",
                source="runtime",
                generation_id=generation.generation_id,
                active_leases=generation.active_leases,
                forced=forced,
                outcome="ok",
            )
            return True
        finally:
            if not generation.closed and generation.cleanup_task is task:
                generation.cleanup_task = None

    @staticmethod
    def _trace_published(
        generation: _ProviderGeneration,
        *,
        previous: _ProviderGeneration | None,
        reason: str,
    ) -> None:
        trace_event(
            stage="runtime",
            event="provider_generation.published",
            source="runtime",
            generation_id=generation.generation_id,
            previous_generation_id=(
                previous.generation_id if previous is not None else None
            ),
            reason=reason,
        )

    @staticmethod
    def _trace_retired(generation: _ProviderGeneration, *, reason: str) -> None:
        trace_event(
            stage="runtime",
            event="provider_generation.retired",
            source="runtime",
            generation_id=generation.generation_id,
            active_leases=generation.active_leases,
            reason=reason,
        )

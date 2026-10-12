"""One closable generation of lazily constructed provider clients."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from functools import partial
from typing import TYPE_CHECKING

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.settings import Settings
from free_claude_code.core.async_tasks import run_sync_owned
from free_claude_code.providers.admission_registry import ProviderAdmissionRegistry
from free_claude_code.providers.base import BaseProvider

if TYPE_CHECKING:
    from .factory import ProviderFactory

type ProviderConstructor = Callable[
    [str, Settings, ProviderAdmissionRegistry, int], Awaitable[BaseProvider]
]
type ProviderLoader = Callable[[], ProviderFactory]
type ProviderClientKey = tuple[str, int]


def _load_constructor(
    provider_id: str,
    provider_loaders: Mapping[str, ProviderLoader],
    settings: Settings,
    api_key_number: int,
) -> Callable[[Settings, ProviderAdmissionRegistry], BaseProvider]:
    from .factory import prepare_provider

    return prepare_provider(
        provider_id,
        provider_loaders,
        settings.custom_provider(provider_id)
        if provider_id.startswith("custom_")
        else None,
        api_key_number=api_key_number,
    )


async def create_provider(
    provider_id: str,
    settings: Settings,
    admission_registry: ProviderAdmissionRegistry,
    api_key_number: int = 1,
    *,
    provider_loaders: Mapping[str, ProviderLoader] | None = None,
) -> BaseProvider:
    constructor = await run_sync_owned(
        partial(
            _load_constructor,
            provider_id,
            provider_loaders or {},
            settings,
            api_key_number,
        )
    )
    return constructor(settings, admission_registry)


class ProviderRuntime:
    """Own provider instances for one immutable settings snapshot."""

    def __init__(
        self,
        settings: Settings,
        admission_registry: ProviderAdmissionRegistry,
        providers: MutableMapping[ProviderClientKey, BaseProvider] | None = None,
        *,
        provider_constructor: ProviderConstructor = create_provider,
    ) -> None:
        self.settings = settings
        self._admission_registry = admission_registry
        self._providers = providers if providers is not None else {}
        self._provider_constructor = provider_constructor
        self._creations: dict[ProviderClientKey, asyncio.Task[BaseProvider]] = {}
        self._closing = False

    def is_cached(self, provider_id: str, api_key_number: int = 1) -> bool:
        """Return whether this provider/key client is already cached."""
        return (provider_id, api_key_number) in self._providers

    async def resolve_provider(
        self, provider_id: str, api_key_number: int = 1
    ) -> BaseProvider:
        """Return an existing provider or create it lazily."""
        if self._closing:
            raise ApplicationUnavailableError("Provider runtime is shutting down.")
        key = (provider_id, api_key_number)
        if key in self._providers:
            return self._providers[key]
        task = self._creations.get(key)
        if task is None or task.done():
            task = asyncio.create_task(self._construct(key))
            self._creations[key] = task
            task.add_done_callback(partial(self._creation_done, key))
        return await asyncio.shield(task)

    def _creation_done(
        self, key: ProviderClientKey, task: asyncio.Task[BaseProvider]
    ) -> None:
        if self._creations.get(key) is task:
            del self._creations[key]
        if not task.cancelled():
            task.exception()  # Observe failures even when every caller stopped waiting.

    async def _construct(self, key: ProviderClientKey) -> BaseProvider:
        provider_id, api_key_number = key
        provider = await self._provider_constructor(
            provider_id, self.settings, self._admission_registry, api_key_number
        )
        self._providers[key] = provider
        return provider

    async def cleanup(self) -> None:
        """Release every provider client constructed by this generation."""
        self._closing = True
        creations = tuple(self._creations.values())
        for task in creations:
            if not task.done():
                task.cancel()
        await asyncio.gather(*creations, return_exceptions=True)
        self._creations.clear()
        errors: list[Exception] = []
        for provider_id, provider in list(self._providers.items()):
            try:
                await provider.cleanup()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(exc)
            else:
                self._providers.pop(provider_id, None)
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("One or more provider cleanups failed", errors)

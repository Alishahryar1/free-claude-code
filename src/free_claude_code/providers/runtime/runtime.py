"""One closable generation of lazily constructed provider clients."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from functools import partial
from typing import TYPE_CHECKING

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.config.settings import Settings
from free_claude_code.core.async_tasks import run_sync_owned
from free_claude_code.providers.base import BaseProvider

if TYPE_CHECKING:
    from .factory import ProviderFactory

type ProviderConstructor = Callable[[str, Settings], Awaitable[BaseProvider]]
type ProviderLoader = Callable[[], ProviderFactory]


def _load_constructor(
    provider_id: str, provider_loaders: Mapping[str, ProviderLoader], settings: Settings
) -> Callable[[Settings], BaseProvider]:
    from .factory import prepare_provider

    return prepare_provider(
        provider_id,
        provider_loaders,
        settings.custom_provider(provider_id)
        if provider_id.startswith("custom_")
        else None,
    )


async def create_provider(
    provider_id: str,
    settings: Settings,
    *,
    provider_loaders: Mapping[str, ProviderLoader] | None = None,
) -> BaseProvider:
    constructor = await run_sync_owned(
        partial(_load_constructor, provider_id, provider_loaders or {}, settings)
    )
    return constructor(settings)


class ProviderRuntime:
    """Own provider instances for one immutable settings snapshot."""

    def __init__(
        self,
        settings: Settings,
        providers: MutableMapping[str, BaseProvider] | None = None,
        *,
        provider_constructor: ProviderConstructor = create_provider,
    ) -> None:
        self.settings = settings
        self._providers = providers if providers is not None else {}
        self._provider_constructor = provider_constructor
        self._creations: dict[str, asyncio.Task[BaseProvider]] = {}
        self._closing = False

    def is_cached(self, provider_id: str) -> bool:
        """Return whether a provider for this id is already cached."""
        return provider_id in self._providers

    async def resolve_provider(self, provider_id: str) -> BaseProvider:
        """Return an existing provider or create it lazily."""
        if self._closing:
            raise ApplicationUnavailableError("Provider runtime is shutting down.")
        if provider_id in self._providers:
            return self._providers[provider_id]
        task = self._creations.get(provider_id)
        if task is None or task.done():
            task = asyncio.create_task(self._construct(provider_id))
            self._creations[provider_id] = task
            task.add_done_callback(partial(self._creation_done, provider_id))
        return await asyncio.shield(task)

    def _creation_done(
        self, provider_id: str, task: asyncio.Task[BaseProvider]
    ) -> None:
        if self._creations.get(provider_id) is task:
            del self._creations[provider_id]
        if not task.cancelled():
            task.exception()  # Observe failures even when every caller stopped waiting.

    async def _construct(self, provider_id: str) -> BaseProvider:
        provider = await self._provider_constructor(provider_id, self.settings)
        self._providers[provider_id] = provider
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

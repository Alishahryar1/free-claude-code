"""Provider model-list discovery and background refresh."""

import asyncio
from collections.abc import Callable

import httpx
from loguru import logger

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import (
    ProviderModelInfo,
    ProviderModelRefreshResult,
)
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
    ProviderDescriptor,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.model_listing import ModelListResponseError

from .config import has_provider_configuration
from .model_cache import ProviderModelCache

ProviderResolver = Callable[[str], BaseProvider]


def _provider_query_failure_reason(exc: BaseException, settings: Settings) -> str:
    """Return a concise model-list query failure reason for user-facing logs."""
    if isinstance(exc, ModelListResponseError):
        return f"malformed model-list response: {exc.message}"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"query failure: HTTP {exc.response.status_code}"
    if isinstance(exc, ApplicationUnavailableError):
        return f"query failure: {exc.message}"
    if isinstance(exc, ExecutionFailure) and settings.log_api_error_tracebacks:
        return f"query failure: {exc.message}"
    return f"query failure: {type(exc).__name__}"


def referenced_provider_ids(settings: Settings) -> tuple[str, ...]:
    """Return unique provider ids referenced by configured chat models."""
    return tuple(
        dict.fromkeys(ref.provider_id for ref in configured_chat_model_refs(settings))
    )


def model_cache_provider_ids_for_settings(
    settings: Settings,
    connected_provider_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return providers whose model metadata is valid for these settings."""
    configured = tuple(
        provider_id
        for provider_id, descriptor in PROVIDER_CATALOG.items()
        if has_provider_configuration(descriptor, settings)
    )
    available = set(configured) | set(connected_provider_ids)
    return tuple(
        provider_id for provider_id in PROVIDER_CATALOG if provider_id in available
    )


def _self_sufficient_local(descriptor: ProviderDescriptor) -> bool:
    """Return whether a local server can be probed with zero configuration.

    Keyless locals advertising built-in defaults (Ollama, LM Studio,
    llama.cpp) expose their OpenAI-compatible model lists without any user
    settings, so they belong in best-effort discovery (#1296).
    """
    return (
        descriptor.local
        and descriptor.static_credential is not None
        and descriptor.default_base_url is not None
    )


def model_list_provider_ids_for_settings(
    settings: Settings,
    connected_provider_ids: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return providers worth discovering for this process configuration."""
    referenced_ids = referenced_provider_ids(settings)
    included = set(
        model_cache_provider_ids_for_settings(settings, connected_provider_ids)
    )
    for provider_id, descriptor in PROVIDER_CATALOG.items():
        if (
            descriptor.local
            and provider_id not in included
            and (provider_id in referenced_ids or _self_sufficient_local(descriptor))
        ):
            included.add(provider_id)
    return tuple(
        provider_id for provider_id in SUPPORTED_PROVIDER_IDS if provider_id in included
    )


class ProviderModelDiscovery:
    """Refresh provider model-list metadata for one provider runtime."""

    def __init__(
        self,
        settings: Settings,
        provider_resolver: ProviderResolver,
        model_cache: ProviderModelCache,
        connected_provider_ids: tuple[str, ...] = (),
    ) -> None:
        self._settings = settings
        self._provider_resolver = provider_resolver
        self._model_cache = model_cache
        self._connected_provider_ids = connected_provider_ids

    async def warm_referenced_model_cache(self) -> ProviderModelRefreshResult:
        """Synchronously cache model metadata for routed providers."""
        return await self._refresh_model_infos(referenced_provider_ids(self._settings))

    async def refresh_model_list_cache(
        self, *, only_missing: bool = False
    ) -> ProviderModelRefreshResult:
        """Best-effort refresh of model lists for usable providers."""
        provider_ids = model_list_provider_ids_for_settings(
            self._settings, self._connected_provider_ids
        )
        if only_missing:
            provider_ids = tuple(
                provider_id
                for provider_id in provider_ids
                if not self._model_cache.has_provider(provider_id)
                and not self._model_cache.discovery_in_cooldown(provider_id)
            )
        return await self._refresh_model_infos(provider_ids)

    async def refresh_provider(self, provider_id: str) -> ProviderModelRefreshResult:
        """Refresh exactly one dynamically changed provider."""

        return await self._refresh_model_infos((provider_id,))

    async def _refresh_model_infos(
        self, provider_ids: tuple[str, ...]
    ) -> ProviderModelRefreshResult:
        failed_provider_ids: list[str] = []
        tasks: dict[str, asyncio.Task[frozenset[ProviderModelInfo]]] = {}
        for provider_id in provider_ids:
            try:
                provider = self._provider_resolver(provider_id)
            except Exception as exc:
                self._log_discovery_failure(provider_id, exc)
                self._record_discovery_failure(provider_id)
                failed_provider_ids.append(provider_id)
                continue
            tasks[provider_id] = asyncio.create_task(provider.list_model_infos())

        refreshed_provider_ids: list[str] = []
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for (provider_id, _task), result in zip(
                tasks.items(), results, strict=True
            ):
                if isinstance(result, BaseException):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    self._log_discovery_failure(provider_id, result)
                    self._record_discovery_failure(provider_id)
                    failed_provider_ids.append(provider_id)
                    continue
                self._model_cache.cache_model_infos(provider_id, result)
                refreshed_provider_ids.append(provider_id)
                logger.info(
                    "Provider model discovery cached: provider={} models={}",
                    provider_id,
                    len(result),
                )

        return ProviderModelRefreshResult(
            refreshed_provider_ids=tuple(refreshed_provider_ids),
            failed_provider_ids=tuple(failed_provider_ids),
        )

    def _record_discovery_failure(self, provider_id: str) -> None:
        """Pace repeat probes of offline keyless locals (cooldown); remotes stay every-cycle."""
        descriptor = PROVIDER_CATALOG.get(provider_id)
        if descriptor is not None and descriptor.local:
            self._model_cache.mark_discovery_failure(provider_id)

    def _log_discovery_failure(self, provider_id: str, exc: BaseException) -> None:
        logger.warning(
            "Provider model discovery skipped: provider={} reason={}",
            provider_id,
            _provider_query_failure_reason(exc, self._settings),
        )

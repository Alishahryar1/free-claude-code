"""Provider model-list metadata cache."""

import time
from collections.abc import Iterable
from dataclasses import replace

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import SUPPORTED_PROVIDER_IDS

# Retry pacing for offline keyless local servers (Ollama/LM Studio/llama.cpp):
# failed discovery probes back off for this long before the next attempt.
LOCAL_DISCOVERY_RETRY_COOLDOWN_S: float = 60.0


class ProviderModelCache:
    """Store provider model metadata for instant model-list responses."""

    def __init__(
        self,
        available_provider_ids: Iterable[str] = SUPPORTED_PROVIDER_IDS,
    ) -> None:
        self._available_provider_ids = frozenset(available_provider_ids)
        self._model_infos_by_provider: dict[str, dict[str, ProviderModelInfo]] = {}
        self._discovery_failures: dict[str, float] = {}

    def cache_model_infos(
        self, provider_id: str, model_infos: Iterable[ProviderModelInfo]
    ) -> None:
        """Store provider model metadata by raw provider model id."""
        if provider_id not in self._available_provider_ids:
            return
        clean_infos = {
            info.model_id: info for info in model_infos if info.model_id.strip()
        }
        self._model_infos_by_provider[provider_id] = clean_infos
        self._discovery_failures.pop(provider_id, None)

    def set_available_providers(self, provider_ids: Iterable[str]) -> None:
        """Replace the provider scope and discard entries outside it."""
        self._available_provider_ids = frozenset(provider_ids)
        self._model_infos_by_provider = {
            provider_id: infos
            for provider_id, infos in self._model_infos_by_provider.items()
            if provider_id in self._available_provider_ids
        }
        self._discovery_failures = {
            provider_id: attempted_at
            for provider_id, attempted_at in self._discovery_failures.items()
            if provider_id in self._available_provider_ids
        }

    def add_provider(self, provider_id: str) -> None:
        """Make one dynamically authenticated provider cacheable."""

        self._available_provider_ids = self._available_provider_ids | {provider_id}

    def remove_provider(self, provider_id: str) -> None:
        """Evict one provider and stop accepting its discovered metadata."""

        self._available_provider_ids = self._available_provider_ids - {provider_id}
        self._model_infos_by_provider.pop(provider_id, None)
        self._discovery_failures.pop(provider_id, None)

    def cached_model_ids(self) -> dict[str, frozenset[str]]:
        """Return cached raw provider model ids by provider."""
        return {
            provider_id: frozenset(infos)
            for provider_id, infos in self._model_infos_by_provider.items()
        }

    def has_provider(self, provider_id: str) -> bool:
        """Return whether this provider has any cached model-list result."""
        return provider_id in self._model_infos_by_provider

    def mark_discovery_failure(self, provider_id: str) -> None:
        """Record a failed discovery attempt to pace local-server retries."""
        self._discovery_failures[provider_id] = time.monotonic()

    def discovery_in_cooldown(self, provider_id: str) -> bool:
        """Return whether a recent failure still blocks re-probing this provider."""
        attempted_at = self._discovery_failures.get(provider_id)
        if attempted_at is None:
            return False
        elapsed = time.monotonic() - attempted_at
        if elapsed >= LOCAL_DISCOVERY_RETRY_COOLDOWN_S:
            del self._discovery_failures[provider_id]
            return False
        return True

    def cached_model_info(
        self, provider_id: str, model_id: str
    ) -> ProviderModelInfo | None:
        """Return the complete cached record for one provider model."""
        return self._model_infos_by_provider.get(provider_id, {}).get(model_id)

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        """Return cached provider models with user-selectable prefixed ids."""
        infos: list[ProviderModelInfo] = []
        for provider_id in SUPPORTED_PROVIDER_IDS:
            provider_infos = self._model_infos_by_provider.get(provider_id, {})
            infos.extend(
                replace(info, model_id=f"{provider_id}/{info.model_id}")
                for info in sorted(
                    provider_infos.values(), key=lambda item: item.model_id
                )
            )
        return tuple(infos)

    def clear(self) -> None:
        """Clear all cached model metadata."""
        self._model_infos_by_provider.clear()

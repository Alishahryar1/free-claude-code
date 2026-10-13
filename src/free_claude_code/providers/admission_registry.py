"""Loop-owned admission policy and controller lifetimes, independent of clients."""

from collections.abc import Iterable
from hashlib import sha256
from typing import TYPE_CHECKING

from free_claude_code.providers.admission_capacity import ProviderCapacityLimiter
from free_claude_code.providers.admission_policy import ProviderAdmissionLimits

if TYPE_CHECKING:
    from free_claude_code.providers.admission import ProviderAdmissionController


class ProviderAdmissionRegistry:
    """Share provider capacity and retain recovery by the actual credential."""

    def __init__(self, limits: ProviderAdmissionLimits) -> None:
        self._limits = limits
        self._controllers: dict[
            str, dict[bytes | None, ProviderAdmissionController]
        ] = {}
        self._capacities: dict[str, ProviderCapacityLimiter] = {}
        self._closed = False

    def get(
        self, provider_id: str, *, api_key: str | None = None
    ) -> ProviderAdmissionController:
        if self._closed:
            raise RuntimeError("Provider admission registry is closed")
        if provider_id not in self._controllers:
            self._controllers[provider_id] = {}
            self._capacities[provider_id] = ProviderCapacityLimiter(self._limits)
        credential = sha256(api_key.encode()).digest() if api_key else None
        controllers = self._controllers[provider_id]
        if credential not in controllers:
            # Provider preparation loads SDKs in a worker before reaching here.
            from free_claude_code.providers.admission import ProviderAdmissionController

            controllers[credential] = ProviderAdmissionController(
                provider_name=provider_id,
                rate_limit=self._limits.rate_limit,
                rate_window=self._limits.rate_window,
                max_concurrency=self._limits.max_concurrency,
                capacity=self._capacities[provider_id],
            )
        return controllers[credential]

    def reconfigure(self, limits: ProviderAdmissionLimits) -> None:
        """Publish validated limits without yielding to waiting requests."""
        self._limits = limits
        for capacity in self._capacities.values():
            capacity.reconfigure(limits)

    def retain_custom(self, provider_ids: Iterable[str]) -> None:
        retained = set(provider_ids)
        for provider_id in tuple(self._controllers):
            if provider_id.startswith("custom_") and provider_id not in retained:
                del self._controllers[provider_id]
                del self._capacities[provider_id]

    def close(self) -> None:
        """Forget state only after client generations have drained and closed."""
        self._closed = True
        self._controllers.clear()
        self._capacities.clear()

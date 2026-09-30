"""Loop-owned admission policy and controller lifetimes, independent of clients."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from free_claude_code.providers.admission_policy import ProviderAdmissionLimits

if TYPE_CHECKING:
    from free_claude_code.providers.admission import ProviderAdmissionController


class ProviderAdmissionRegistry:
    """One protection budget per configured provider for one runtime manager."""

    def __init__(self, limits: ProviderAdmissionLimits) -> None:
        self._limits = limits
        self._controllers: dict[str, ProviderAdmissionController] = {}
        self._closed = False

    def get(self, provider_id: str) -> ProviderAdmissionController:
        if self._closed:
            raise RuntimeError("Provider admission registry is closed")
        if provider_id not in self._controllers:
            # Provider preparation loads SDKs in a worker before reaching here.
            from free_claude_code.providers.admission import ProviderAdmissionController

            self._controllers[provider_id] = ProviderAdmissionController(
                provider_name=provider_id,
                rate_limit=self._limits.rate_limit,
                rate_window=self._limits.rate_window,
                max_concurrency=self._limits.max_concurrency,
            )
        return self._controllers[provider_id]

    def reconfigure(self, limits: ProviderAdmissionLimits) -> None:
        """Publish validated limits without yielding to waiting requests."""
        self._limits = limits
        for controller in self._controllers.values():
            controller.reconfigure(limits)

    def retain_custom(self, provider_ids: Iterable[str]) -> None:
        retained = set(provider_ids)
        for provider_id in tuple(self._controllers):
            if provider_id.startswith("custom_") and provider_id not in retained:
                del self._controllers[provider_id]

    def close(self) -> None:
        """Forget state only after client generations have drained and closed."""
        self._closed = True
        self._controllers.clear()

"""Admission limits specific to the Antigravity connected provider."""

from free_claude_code.providers.admission_policy import ProviderAdmissionLimits
from free_claude_code.providers.admission_registry import ProviderAdmissionRegistry


def test_antigravity_registry_pins_single_concurrency_and_attempt() -> None:
    registry = ProviderAdmissionRegistry(ProviderAdmissionLimits(7, 11, 4))

    controller = registry.get("antigravity")

    assert controller._limits == ProviderAdmissionLimits(7, 11, 1)
    assert controller.start_execution().max_attempts == 1

    registry.reconfigure(ProviderAdmissionLimits(9, 13, 8))

    assert controller._limits == ProviderAdmissionLimits(9, 13, 1)
    assert controller.start_execution().max_attempts == 1

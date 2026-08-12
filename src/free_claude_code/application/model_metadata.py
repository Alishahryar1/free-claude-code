"""Application-owned model metadata."""

from dataclasses import dataclass, field

from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningCapability


@dataclass(frozen=True, slots=True)
class ProviderModelPricing:
    """Per-token pricing metadata advertised by a provider (USD per 1K tokens)."""

    input: float | None = None
    output: float | None = None


@dataclass(frozen=True, slots=True)
class ProviderModelInfo:
    """Provider model metadata used to shape the application model catalog.

    Only ``model_id`` is required. All capability/limit/pricing fields are
    optional and populated only when the upstream provider exposes them, so
    providers that return a bare ``/models`` list keep working unchanged.
    """

    model_id: str
    supports_thinking: bool | None = None
    input_modalities: frozenset[ModelInputModality] | None = None
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    reasoning_capability: ReasoningCapability = ReasoningCapability.UNKNOWN
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    pricing: ProviderModelPricing | None = None


@dataclass(frozen=True, slots=True)
class ProviderModelRefreshResult:
    """Per-provider outcome of one model-catalog refresh."""

    refreshed_provider_ids: tuple[str, ...] = ()
    failed_provider_ids: tuple[str, ...] = ()

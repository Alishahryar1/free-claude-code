"""Model-list response construction for FCC clients."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import RequestRuntimePort
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import (
    gateway_model_id,
    no_thinking_gateway_model_id,
)
from free_claude_code.core.model_capabilities import ModelInputModality

DISCOVERED_MODEL_CREATED_AT = "1970-01-01T00:00:00Z"
_INFERENCE_IDLE_TIMEOUT_MARGIN_SECONDS = 60
_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class ModelCatalogView(StrEnum):
    """Client-specific projections of the application model inventory."""

    CLAUDE = "claude"
    MESSAGES = "messages"
    RESPONSES = "responses"


class MuseModelLimits(BaseModel):
    context: int | None = None
    output: int | None = None


class MuseModelMetadata(BaseModel):
    """Muse's native model visibility and capability metadata."""

    name: str
    is_hidden: Literal[False] = False
    reasoning: bool | None = None
    limit: MuseModelLimits | None = None


class MuseMetadataEnvelope(BaseModel):
    muse_code: MuseModelMetadata = Field(serialization_alias="muse-code")


class ModelPricing(BaseModel):
    """Per-1K-token pricing (USD) advertised for a model, when known."""

    input: float | None = None
    output: float | None = None


class ModelResponse(BaseModel):
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "free-claude-code"
    created_at: str
    display_name: str
    id: str
    type: Literal["model"] = "model"
    provider_model_ref: str | None = None
    api_backend: Literal["responses"] | None = Field(
        default=None, serialization_alias="apiBackend"
    )
    max_retries: Literal[0] | None = Field(
        default=None, serialization_alias="maxRetries"
    )
    supports_reasoning_effort: bool | None = Field(
        default=None, serialization_alias="supportsReasoningEffort"
    )
    supports_reasoning: bool | None = Field(
        default=None, serialization_alias="supportsReasoning"
    )
    input_modalities: tuple[ModelInputModality, ...] | None = Field(
        default=None, serialization_alias="inputModalities"
    )
    context_window_tokens: int | None = Field(
        default=None, serialization_alias="contextWindow"
    )
    max_output_tokens: int | None = Field(
        default=None, serialization_alias="maxCompletionTokens"
    )
    reasoning_efforts: tuple[str, ...] | None = Field(
        default=None, serialization_alias="reasoningEfforts"
    )
    inference_idle_timeout_seconds: int | None = Field(
        default=None, serialization_alias="inferenceIdleTimeoutSecs"
    )
    metadata: MuseMetadataEnvelope | None = None
    # Optional capability/pricing metadata, surfaced only when the
    # upstream provider advertised it (omitted from the payload when None).
    capabilities: list[str] | None = None
    pricing: ModelPricing | None = None


class ModelsListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelResponse]
    first_id: str | None
    has_more: bool
    last_id: str | None


SUPPORTED_CLAUDE_MODELS = [
    ModelResponse(
        id="claude-fable-5",
        display_name="Claude Fable 5",
        created_at="2026-06-09T00:00:00Z",
    ),
    ModelResponse(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-haiku-4-20250514",
        display_name="Claude Haiku 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        created_at="2024-02-29T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        created_at="2024-10-22T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        created_at="2024-03-07T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        created_at="2024-10-22T00:00:00Z",
    ),
]


@dataclass(frozen=True, slots=True)
class _InventoryModel:
    provider_model_ref: str
    supports_thinking: bool | None
    input_modalities: frozenset[ModelInputModality] | None
    context_window_tokens: int | None
    max_output_tokens: int | None
    metadata: ProviderModelInfo | None = None


def build_models_list_response(
    settings: Settings,
    runtime: RequestRuntimePort,
    *,
    view: ModelCatalogView = ModelCatalogView.CLAUDE,
) -> ModelsListResponse:
    """Return the application model inventory in the requested client view."""
    if view is ModelCatalogView.CLAUDE:
        return _build_claude_models_response(settings, runtime)
    return _build_direct_models_response(settings, runtime, view=view)


def build_muse_models_list_response(
    settings: Settings, runtime: RequestRuntimePort
) -> ModelsListResponse:
    """Keep routable IDs while making them visible to Muse's native picker."""
    catalog = build_models_list_response(
        settings, runtime, view=ModelCatalogView.RESPONSES
    )
    for model in catalog.data:
        limits = (
            MuseModelLimits(
                context=model.context_window_tokens,
                output=model.max_output_tokens,
            )
            if model.context_window_tokens is not None
            or model.max_output_tokens is not None
            else None
        )
        model.metadata = MuseMetadataEnvelope(
            muse_code=MuseModelMetadata(
                name=model.display_name,
                reasoning=model.supports_reasoning,
                limit=limits,
            )
        )
    return catalog


def _build_claude_models_response(
    settings: Settings, runtime: RequestRuntimePort
) -> ModelsListResponse:
    """Preserve the established Claude-compatible catalog exactly."""
    models: list[ModelResponse] = []
    seen: set[str] = set()

    for ref in configured_chat_model_refs(settings):
        cached_info = runtime.cached_model_info(ref.provider_id, ref.model_id)
        _append_provider_model_variants(
            models,
            seen,
            ref.model_ref,
            supports_thinking=(
                cached_info.supports_thinking if cached_info is not None else None
            ),
            metadata=cached_info,
        )

    for model_info in runtime.cached_prefixed_model_infos():
        _append_provider_model_variants(
            models,
            seen,
            model_info.model_id,
            supports_thinking=model_info.supports_thinking,
            metadata=model_info,
        )

    for model in SUPPORTED_CLAUDE_MODELS:
        _append_unique_model(models, seen, model)

    return ModelsListResponse(
        data=models,
        first_id=models[0].id if models else None,
        has_more=False,
        last_id=models[-1].id if models else None,
    )


def _build_direct_models_response(
    settings: Settings,
    runtime: RequestRuntimePort,
    *,
    view: ModelCatalogView,
) -> ModelsListResponse:
    models: list[ModelResponse] = []
    timeout_seconds = (
        _responses_inference_idle_timeout_seconds(settings.provider_progress_timeout)
        if view is ModelCatalogView.RESPONSES
        else None
    )

    for inventory_model in _collect_inventory(settings, runtime):
        provider_model_ref = inventory_model.provider_model_ref
        allows_reasoning = inventory_model.supports_thinking is not False
        model_id = (
            provider_model_ref
            if allows_reasoning
            else no_thinking_gateway_model_id(provider_model_ref)
        )
        models.append(
            ModelResponse(
                id=model_id,
                display_name=(
                    provider_model_ref
                    if allows_reasoning
                    else f"{provider_model_ref} (no thinking)"
                ),
                created_at=DISCOVERED_MODEL_CREATED_AT,
                provider_model_ref=provider_model_ref,
                api_backend=(
                    "responses" if view is ModelCatalogView.RESPONSES else None
                ),
                max_retries=0 if view is ModelCatalogView.RESPONSES else None,
                supports_reasoning=inventory_model.supports_thinking,
                input_modalities=_serialize_input_modalities(
                    inventory_model.input_modalities
                ),
                context_window_tokens=inventory_model.context_window_tokens,
                max_output_tokens=inventory_model.max_output_tokens,
                supports_reasoning_effort=(
                    allows_reasoning if view is ModelCatalogView.RESPONSES else None
                ),
                reasoning_efforts=(
                    _REASONING_EFFORTS
                    if view is ModelCatalogView.RESPONSES and allows_reasoning
                    else None
                ),
                inference_idle_timeout_seconds=timeout_seconds,
                capabilities=_capabilities_response(
                    inventory_model.metadata, strip_thinking=not allows_reasoning
                ),
                pricing=_pricing_response(inventory_model.metadata),
            )
        )

    return ModelsListResponse(
        data=models,
        first_id=models[0].id if models else None,
        has_more=False,
        last_id=models[-1].id if models else None,
    )


def _collect_inventory(
    settings: Settings, runtime: RequestRuntimePort
) -> tuple[_InventoryModel, ...]:
    inventory: list[_InventoryModel] = []
    seen: set[str] = set()

    for ref in configured_chat_model_refs(settings):
        if ref.model_ref in seen:
            continue
        seen.add(ref.model_ref)
        model_info = runtime.cached_model_info(ref.provider_id, ref.model_id)
        inventory.append(
            _InventoryModel(
                provider_model_ref=ref.model_ref,
                supports_thinking=(
                    model_info.supports_thinking if model_info is not None else None
                ),
                input_modalities=(
                    model_info.input_modalities if model_info is not None else None
                ),
                context_window_tokens=(
                    model_info.context_window_tokens if model_info is not None else None
                ),
                max_output_tokens=(
                    model_info.max_output_tokens if model_info is not None else None
                ),
                metadata=model_info,
            )
        )

    for model_info in runtime.cached_prefixed_model_infos():
        if model_info.model_id in seen:
            continue
        seen.add(model_info.model_id)
        inventory.append(
            _InventoryModel(
                provider_model_ref=model_info.model_id,
                supports_thinking=model_info.supports_thinking,
                input_modalities=model_info.input_modalities,
                context_window_tokens=model_info.context_window_tokens,
                max_output_tokens=model_info.max_output_tokens,
                metadata=model_info,
            )
        )

    return tuple(inventory)


def _serialize_input_modalities(
    modalities: frozenset[ModelInputModality] | None,
) -> tuple[ModelInputModality, ...] | None:
    if modalities is None:
        return None
    return tuple(modality for modality in ModelInputModality if modality in modalities)


def _responses_inference_idle_timeout_seconds(provider_progress_timeout: float) -> int:
    return math.ceil(provider_progress_timeout) + _INFERENCE_IDLE_TIMEOUT_MARGIN_SECONDS


def _pricing_response(metadata: ProviderModelInfo | None) -> ModelPricing | None:
    if metadata is None or metadata.pricing is None:
        return None
    return ModelPricing(
        input=metadata.pricing.input,
        output=metadata.pricing.output,
    )


def _capabilities_response(
    metadata: ProviderModelInfo | None, *, strip_thinking: bool = False
) -> list[str] | None:
    # Routes that force reasoning off must never advertise thinking support.
    if metadata is None or not metadata.capabilities:
        return None
    capabilities = list(metadata.capabilities)
    if strip_thinking and "thinking" in capabilities:
        capabilities = [cap for cap in capabilities if cap != "thinking"]
    return capabilities or None


def _discovered_model_response(
    model_id: str,
    *,
    display_name: str,
    metadata: ProviderModelInfo | None = None,
    strip_thinking: bool = False,
) -> ModelResponse:
    """Build a discovered-model response, attaching optional upstream metadata."""
    return ModelResponse(
        id=model_id,
        display_name=display_name,
        created_at=DISCOVERED_MODEL_CREATED_AT,
        capabilities=_capabilities_response(metadata, strip_thinking=strip_thinking),
        pricing=_pricing_response(metadata),
    )


def _append_unique_model(
    models: list[ModelResponse], seen: set[str], model: ModelResponse
) -> None:
    if model.id in seen:
        return
    seen.add(model.id)
    models.append(model)


def _append_provider_model_variants(
    models: list[ModelResponse],
    seen: set[str],
    provider_model_ref: str,
    *,
    supports_thinking: bool | None = None,
    metadata: ProviderModelInfo | None = None,
) -> None:
    if supports_thinking is not False:
        _append_unique_model(
            models,
            seen,
            _discovered_model_response(
                gateway_model_id(provider_model_ref),
                display_name=provider_model_ref,
                metadata=metadata,
            ),
        )
    _append_unique_model(
        models,
        seen,
        _discovered_model_response(
            no_thinking_gateway_model_id(provider_model_ref),
            display_name=f"{provider_model_ref} (no thinking)",
            metadata=metadata,
            strip_thinking=True,
        ),
    )

"""Model-list response construction for FCC clients."""

import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from free_claude_code.application.model_catalog import ModelCatalog, read_model_catalog
from free_claude_code.application.ports import ModelCatalogPort
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import (
    desktop_model_id,
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
    CLAUDE_DESKTOP = "claude-desktop"
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


class ModelsListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelResponse]
    first_id: str | None
    has_more: bool
    last_id: str | None
    default_model_id: str | None = None


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


def build_models_list_response(
    settings: Settings,
    runtime: ModelCatalogPort,
    *,
    view: ModelCatalogView = ModelCatalogView.CLAUDE,
) -> ModelsListResponse:
    """Return the application model inventory in the requested client view."""
    catalog = read_model_catalog(runtime, settings)
    if view in {ModelCatalogView.CLAUDE, ModelCatalogView.CLAUDE_DESKTOP}:
        return _build_claude_models_response(
            catalog, desktop=view is ModelCatalogView.CLAUDE_DESKTOP
        )
    return _build_direct_models_response(settings, catalog, view=view)


def build_muse_models_list_response(
    settings: Settings, runtime: ModelCatalogPort
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
    catalog: ModelCatalog, *, desktop: bool = False
) -> ModelsListResponse:
    """Keep shortcuts first and provider variants together in catalog order."""
    models = list(SUPPORTED_CLAUDE_MODELS)
    for model in catalog.models:
        ref = model.provider_model_ref
        if model.supports_reasoning is not False:
            models.append(
                _discovered_model_response(
                    desktop_model_id(ref) if desktop else gateway_model_id(ref),
                    display_name=ref,
                )
            )
        models.append(
            _discovered_model_response(
                desktop_model_id(ref, no_thinking=True)
                if desktop
                else no_thinking_gateway_model_id(ref),
                display_name=f"{ref} (no thinking)",
            )
        )
    return ModelsListResponse(
        data=models,
        first_id=models[0].id if models else None,
        has_more=False,
        last_id=models[-1].id if models else None,
    )


def _build_direct_models_response(
    settings: Settings,
    catalog: ModelCatalog,
    *,
    view: ModelCatalogView,
) -> ModelsListResponse:
    models: list[ModelResponse] = []
    timeout_seconds = (
        _responses_inference_idle_timeout_seconds(settings.provider_progress_timeout)
        if view is ModelCatalogView.RESPONSES
        else None
    )

    for model in catalog.models:
        allows_reasoning = model.supports_reasoning is not False
        models.append(
            ModelResponse(
                id=model.wire_slug,
                display_name=model.display_name,
                created_at=DISCOVERED_MODEL_CREATED_AT,
                provider_model_ref=model.provider_model_ref,
                api_backend=(
                    "responses" if view is ModelCatalogView.RESPONSES else None
                ),
                max_retries=0 if view is ModelCatalogView.RESPONSES else None,
                supports_reasoning=model.supports_reasoning,
                input_modalities=_serialize_input_modalities(model.input_modalities),
                context_window_tokens=model.context_window_tokens,
                max_output_tokens=model.max_output_tokens,
                supports_reasoning_effort=(
                    allows_reasoning if view is ModelCatalogView.RESPONSES else None
                ),
                reasoning_efforts=(
                    _REASONING_EFFORTS
                    if view is ModelCatalogView.RESPONSES and allows_reasoning
                    else None
                ),
                inference_idle_timeout_seconds=timeout_seconds,
            )
        )

    return ModelsListResponse(
        data=models,
        default_model_id=catalog.default_model_id,
        first_id=models[0].id if models else None,
        has_more=False,
        last_id=models[-1].id if models else None,
    )


def _serialize_input_modalities(
    modalities: frozenset[ModelInputModality] | None,
) -> tuple[ModelInputModality, ...] | None:
    if modalities is None:
        return None
    return tuple(modality for modality in ModelInputModality if modality in modalities)


def _responses_inference_idle_timeout_seconds(provider_progress_timeout: float) -> int:
    return math.ceil(provider_progress_timeout) + _INFERENCE_IDLE_TIMEOUT_MARGIN_SECONDS


def _discovered_model_response(model_id: str, *, display_name: str) -> ModelResponse:
    return ModelResponse(
        id=model_id,
        display_name=display_name,
        created_at=DISCOVERED_MODEL_CREATED_AT,
    )

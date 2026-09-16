"""Application model inventory and presentation order, independent of clients."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from free_claude_code.config.model_refs import (
    configured_chat_model_refs,
    split_provider_model_ref,
)
from free_claude_code.core.gateway_model_ids import no_thinking_gateway_model_id
from free_claude_code.core.model_capabilities import ModelInputModality

from .model_metadata import ProviderModelInfo

if TYPE_CHECKING:
    from free_claude_code.config.settings import Settings

    from .ports import ModelCatalogPort


@dataclass(frozen=True, slots=True)
class CatalogModel:
    """One exact FCC model identity and its reported capabilities."""

    wire_slug: str
    provider_model_ref: str
    display_name: str
    supports_reasoning: bool | None
    input_modalities: frozenset[ModelInputModality] | None = None
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """An ordered inventory with selection independent of list position."""

    models: tuple[CatalogModel, ...]
    default_model_id: str


def model_order_key(provider_model_ref: str) -> tuple[str, str, str, str]:
    provider, model = split_provider_model_ref(provider_model_ref)
    return provider.casefold(), model.casefold(), provider, model


def read_model_catalog(
    runtime: ModelCatalogPort, settings: Settings | None = None
) -> ModelCatalog:
    """Merge configured and cached models without discovery or provider leases."""
    settings = settings if settings is not None else runtime.current_settings()
    models: dict[str, CatalogModel] = {}
    for ref in configured_chat_model_refs(settings):
        models[ref.model_ref] = _catalog_model(
            ref.model_ref, runtime.cached_model_info(ref.provider_id, ref.model_id)
        )
    for info in runtime.cached_prefixed_model_infos():
        if info.model_id not in models:
            models[info.model_id] = _catalog_model(info.model_id, info)
    return ModelCatalog(
        models=tuple(
            sorted(
                models.values(),
                key=lambda model: model_order_key(model.provider_model_ref),
            )
        ),
        default_model_id=models[settings.model].wire_slug,
    )


def _catalog_model(ref: str, info: ProviderModelInfo | None) -> CatalogModel:
    info = info if info is not None else ProviderModelInfo(ref)
    no_thinking = info.supports_thinking is False
    return CatalogModel(
        wire_slug=no_thinking_gateway_model_id(ref) if no_thinking else ref,
        provider_model_ref=ref,
        display_name=f"{ref} (no thinking)" if no_thinking else ref,
        supports_reasoning=info.supports_thinking,
        input_modalities=info.input_modalities,
        context_window_tokens=info.context_window_tokens,
        max_output_tokens=info.max_output_tokens,
    )


def catalog_wire_slug_for_ref(
    models: Sequence[CatalogModel], provider_model_ref: str | None
) -> str | None:
    """Resolve an explicit selection, retaining a missing or unset reference."""
    if not provider_model_ref:
        return provider_model_ref
    for model in models:
        if model.provider_model_ref == provider_model_ref:
            return model.wire_slug
    return provider_model_ref

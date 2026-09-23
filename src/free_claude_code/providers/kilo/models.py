"""Kilo.ai model-catalog interpretation."""

from collections.abc import Mapping, Sequence
from typing import Any

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.providers.model_listing import (
    extract_tool_capable_model_infos,
    model_list_items,
)


def extract_kilo_model_infos(
    payload: Any, *, provider_name: str
) -> frozenset[ProviderModelInfo]:
    """Return chat/tool Kilo models with their advertised reasoning support."""
    tool_models = extract_tool_capable_model_infos(
        payload,
        provider_name=provider_name,
    )
    excluded_ids = {
        model_id
        for item in model_list_items(payload, provider_name=provider_name)
        if (model_id := _model_id(item)) is not None
        and (_produces_images(item) or _requires_responses_api(item))
    }
    return frozenset(info for info in tool_models if info.model_id not in excluded_ids)


def _produces_images(item: Any) -> bool:
    architecture = _field(item, "architecture")
    modalities = _string_values(_field(architecture, "output_modalities"))
    return modalities is not None and "image" in modalities


def _requires_responses_api(item: Any) -> bool:
    opencode = _field(item, "opencode")
    provider = _field(opencode, "ai_sdk_provider")
    return isinstance(provider, str) and provider.casefold() == "openai"


def _model_id(item: Any) -> str | None:
    value = _field(item, "id")
    return value if isinstance(value, str) and value.strip() else None


def _string_values(value: Any) -> set[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return None
    return {item.casefold() for item in value if isinstance(item, str)}


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    value = getattr(item, name, None)
    if value is not None:
        return value
    extra = getattr(item, "model_extra", None)
    return extra.get(name) if isinstance(extra, Mapping) else None

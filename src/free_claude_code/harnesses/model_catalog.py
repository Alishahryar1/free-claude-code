"""Shared FCC model-catalog projection for native clients."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.model_capabilities import ModelInputModality


@dataclass(frozen=True, slots=True)
class ClientModel:
    """One direct FCC model reference suitable for a client-side catalog."""

    wire_slug: str
    provider_model_ref: str
    display_name: str
    supports_reasoning: bool | None
    input_modalities: frozenset[ModelInputModality] | None = None
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None


def client_models_from_response(
    models_response: Mapping[str, JsonValue],
) -> tuple[ClientModel, ...]:
    """Project an FCC `/v1/models` response into direct client model records."""

    models: list[ClientModel] = []
    seen_slugs: set[str] = set()

    for candidate in _catalog_candidates(models_response):
        if candidate.wire_slug in seen_slugs:
            continue
        seen_slugs.add(candidate.wire_slug)
        models.append(candidate)

    return tuple(models)


def catalog_wire_slug_for_ref(
    models: Sequence[ClientModel],
    provider_model_ref: str | None,
) -> str | None:
    """Return the slug a client catalog advertises for one configured ref.

    A model the gateway reports as non-thinking is advertised under its
    no-thinking slug, not its bare provider ref, so a client told to select the
    bare ref would not find it in the catalog it was given.
    """

    if not provider_model_ref:
        return provider_model_ref

    for model in models:
        if model.provider_model_ref == provider_model_ref:
            return model.wire_slug
    return provider_model_ref


def _catalog_candidates(
    models_response: Mapping[str, JsonValue],
) -> list[ClientModel]:
    data = models_response.get("data")
    if not isinstance(data, list):
        return []

    candidates: list[ClientModel] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        model_id = _nonempty_string(item.get("id"))
        if model_id is None:
            continue
        provider_model_ref = _provider_model_ref(item.get("provider_model_ref"))
        if provider_model_ref is None:
            continue
        candidates.append(
            ClientModel(
                wire_slug=model_id,
                provider_model_ref=provider_model_ref,
                display_name=_nonempty_string(item.get("display_name")) or model_id,
                supports_reasoning=_optional_boolean(item.get("supportsReasoning")),
                input_modalities=_input_modalities(item.get("inputModalities")),
                context_window_tokens=_optional_positive_int(item.get("contextWindow")),
                max_output_tokens=_optional_positive_int(
                    item.get("maxCompletionTokens")
                ),
            )
        )
    return candidates


def _nonempty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _optional_boolean(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_positive_int(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _input_modalities(value: object) -> frozenset[ModelInputModality] | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        modalities = frozenset(ModelInputModality(item) for item in value)
    except TypeError, ValueError:
        return None
    if ModelInputModality.TEXT not in modalities:
        return None
    return modalities


def _provider_model_ref(value: object) -> str | None:
    ref = _nonempty_string(value)
    if ref is None:
        return None
    provider_id, separator, model_id = ref.partition("/")
    if not separator or not provider_id or not model_id:
        return None
    return ref

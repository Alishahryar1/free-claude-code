"""Provider-prefixed model reference helpers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .constants import DEFAULT_MODEL

RETIRED_PROVIDER_IDS = frozenset({"github_models"})


def is_retired_model_ref(model_ref: str) -> bool:
    """Recognize complete references owned by a retired provider."""

    provider, separator, model = model_ref.strip().partition("/")
    return provider in RETIRED_PROVIDER_IDS and bool(separator and model.strip())


def parse_model_fallbacks(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return tuple(part.strip() for part in value.split(","))
    if isinstance(value, list | tuple) and not value:
        return None
    return value


def normalize_retired_model_settings(
    values: Mapping[str, str], *, preserve_empty_overrides: bool
) -> dict[str, str]:
    """Repair one source without changing its precedence or unrelated validation."""

    normalized = dict(values)
    for key in ("MODEL", "MODEL_FABLE", "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"):
        if is_retired_model_ref(normalized.get(key, "")):
            if preserve_empty_overrides:
                normalized[key] = DEFAULT_MODEL if key == "MODEL" else ""
            else:
                normalized.pop(key)
    fallbacks = parse_model_fallbacks(normalized.get("MODEL_FALLBACKS"))
    if isinstance(fallbacks, tuple) and all(fallbacks):
        retained = tuple(ref for ref in fallbacks if not is_retired_model_ref(ref))
        if retained != fallbacks:
            if retained or preserve_empty_overrides:
                normalized["MODEL_FALLBACKS"] = ",".join(retained)
            else:
                normalized.pop("MODEL_FALLBACKS")
    return normalized


@dataclass(frozen=True, slots=True)
class ConfiguredChatModelRef:
    """A unique configured chat model reference."""

    model_ref: str
    provider_id: str
    model_id: str


class ChatModelConfig(Protocol):
    model: str
    model_fable: str | None
    model_opus: str | None
    model_sonnet: str | None
    model_haiku: str | None
    model_fallbacks: tuple[str, ...] | None


def split_provider_model_ref(model_ref: str) -> tuple[str, str]:
    """Split one complete ``provider/model`` reference."""

    provider_id, separator, model_id = model_ref.partition("/")
    if not separator or not provider_id or not model_id:
        raise ValueError("Model reference must contain provider and model names.")
    return provider_id, model_id


def parse_provider_type(model_ref: str) -> str:
    """Extract provider type from any 'provider/model' string."""

    return split_provider_model_ref(model_ref)[0]


def parse_model_name(model_ref: str) -> str:
    """Extract model name from any 'provider/model' string."""

    return split_provider_model_ref(model_ref)[1]


def configured_chat_model_refs(
    settings: ChatModelConfig,
) -> tuple[ConfiguredChatModelRef, ...]:
    """Return unique configured chat provider/model refs."""

    model_refs = dict.fromkeys(
        model_ref
        for model_ref in (
            settings.model,
            settings.model_fable,
            settings.model_opus,
            settings.model_sonnet,
            settings.model_haiku,
            *(settings.model_fallbacks or ()),
        )
        if model_ref is not None
    )

    return tuple(
        ConfiguredChatModelRef(
            model_ref=model_ref,
            provider_id=parse_provider_type(model_ref),
            model_id=parse_model_name(model_ref),
        )
        for model_ref in model_refs
    )

"""Provider-prefixed model reference helpers."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Protocol

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
)

from .constants import DEFAULT_MODEL
from .reasoning import ReasoningPreference

RETIRED_PROVIDER_IDS = frozenset({"github_models"})


class ModelFallback(BaseModel):
    """One ordered fallback's model, reasoning preference, and key number."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    reasoning: ReasoningPreference = ReasoningPreference.INHERIT
    api_key_number: int = Field(default=1, strict=True, ge=1)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        split_provider_model_ref(value)
        return value


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
        if value.lstrip().startswith("["):
            try:
                value = json.loads(value)
            except ValueError:
                raise ValueError(
                    "MODEL_FALLBACKS must contain a valid JSON array."
                ) from None
        else:
            value = value.split(",")
    if isinstance(value, list | tuple):
        return (
            tuple({"model": item} if isinstance(item, str) else item for item in value)
            or None
        )
    return value


type ModelFallbacks = Annotated[
    tuple[ModelFallback, ...] | None, BeforeValidator(parse_model_fallbacks)
]
_FALLBACKS = TypeAdapter(ModelFallbacks)


def decode_model_fallbacks(value: object) -> tuple[ModelFallback, ...] | None:
    return _FALLBACKS.validate_python(value)


def encode_model_fallbacks(value: tuple[ModelFallback, ...]) -> str:
    return json.dumps(
        [entry.model_dump(mode="json") for entry in value],
        ensure_ascii=False,
        separators=(",", ":"),
    )


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
    try:
        fallbacks = decode_model_fallbacks(normalized.get("MODEL_FALLBACKS"))
    except ValueError:
        return normalized  # Settings reports the original validation error.
    if fallbacks:
        retained = tuple(
            entry for entry in fallbacks if not is_retired_model_ref(entry.model)
        )
        if retained:
            normalized["MODEL_FALLBACKS"] = encode_model_fallbacks(retained)
        elif preserve_empty_overrides:
            normalized["MODEL_FALLBACKS"] = ""
        else:
            normalized.pop("MODEL_FALLBACKS", None)
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
    model_fallbacks: tuple[ModelFallback, ...] | None


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
            *(entry.model for entry in settings.model_fallbacks or ()),
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

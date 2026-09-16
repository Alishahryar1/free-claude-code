"""Process-local Aider configuration for FCC model routing."""

import re
from dataclasses import dataclass, field

from free_claude_code.application.model_catalog import CatalogModel, model_order_key
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality

AIDER_API_KEY_ENV_PREFIX = "FCC_AIDER_PROXY_AUTH_"
_AIDER_API_KEY_ENV_PATTERN = re.compile(rf"{AIDER_API_KEY_ENV_PREFIX}[A-Z0-9]+")


@dataclass(frozen=True, slots=True)
class AiderConfig:
    """Secret-free model settings and metadata for one Aider process."""

    settings: list[JsonObject] = field(repr=False)
    metadata: JsonObject = field(repr=False)


def build_aider_config(
    models: tuple[CatalogModel, ...],
    *,
    messages_url: str,
    api_key_env: str,
    launch_id: str,
) -> AiderConfig:
    """Project a non-empty FCC Messages catalog into Aider's file contracts."""

    if not models:
        raise ValueError("Aider requires at least one routable FCC model")
    if _AIDER_API_KEY_ENV_PATTERN.fullmatch(api_key_env) is None:
        raise ValueError("invalid Aider proxy-auth environment variable name")

    by_name = {model.wire_slug: model for model in models}
    for model in models:
        by_name.setdefault(f"anthropic/{model.wire_slug}", model)
    settings: list[JsonObject] = []
    metadata: JsonObject = {}
    for name, model in sorted(
        by_name.items(),
        key=lambda item: (
            model_order_key(item[1].provider_model_ref),
            item[0] != item[1].wire_slug,
            item[0],
        ),
    ):
        entry: JsonObject = {
            "name": name,
            "weak_model_name": name,
            "editor_model_name": name,
            "extra_params": {
                "model": f"anthropic/{model.wire_slug}",
                "api_base": messages_url,
                "api_key": f"os.environ/{api_key_env}",
                "extra_headers": {"x-fcc-launch-id": launch_id},
            },
        }
        if model.supports_reasoning is not None:
            entry["accepts_settings"] = (
                ["reasoning_effort"] if model.supports_reasoning else []
            )
        settings.append(entry)
        metadata[name] = _model_metadata(model)
    return AiderConfig(settings=settings, metadata=metadata)


def _model_metadata(model: CatalogModel) -> JsonObject:
    metadata: JsonObject = {
        "litellm_provider": "anthropic",
        "mode": "chat",
    }
    if model.input_modalities is not None:
        metadata["supports_vision"] = ModelInputModality.IMAGE in model.input_modalities
    if model.context_window_tokens is not None:
        metadata["max_input_tokens"] = model.context_window_tokens
    if model.max_output_tokens is not None:
        metadata["max_output_tokens"] = model.max_output_tokens
    return metadata

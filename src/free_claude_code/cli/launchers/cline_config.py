"""Process-local Cline CLI configuration for FCC model routing."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from free_claude_code.application.model_catalog import CatalogModel
from free_claude_code.config.server_urls import proxy_v1_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality

# Cline's released session gateway only instantiates built-in providers. FCC
# replaces this process-local Responses provider's endpoint and catalog instead
# of creating a custom provider that the picker can see but inference cannot.
CLINE_PROVIDER_ID = "openai-native"


@dataclass(frozen=True, slots=True)
class ClineConfig:
    """Private provider settings and model registry for one Cline process."""

    providers: JsonObject = field(repr=False)
    models: JsonObject = field(repr=False)


def build_cline_config(
    models: tuple[CatalogModel, ...],
    *,
    default_model_id: str,
    proxy_root_url: str,
    auth_token: str,
    launch_id: str,
    now: datetime | None = None,
) -> ClineConfig:
    """Translate a non-empty FCC model snapshot into Cline's file contracts."""

    if not models:
        raise ValueError("Cline requires at least one routable FCC model")

    timestamp = (
        (now or datetime.now(UTC)).astimezone(UTC).isoformat().replace("+00:00", "Z")
    )
    provider_settings: JsonObject = {
        "provider": CLINE_PROVIDER_ID,
        "apiKey": auth_token,
        "headers": {"x-fcc-launch-id": launch_id},
        "model": default_model_id,
        "protocol": "openai-responses",
        "baseUrl": proxy_v1_url(proxy_root_url),
        "capabilities": ["streaming", "tools"],
    }
    model_entries: JsonObject = {
        model.wire_slug: _model_entry(model) for model in models
    }

    return ClineConfig(
        providers={
            "version": 1,
            "modes": {},
            "lastUsedProvider": CLINE_PROVIDER_ID,
            "providers": {
                CLINE_PROVIDER_ID: {
                    "settings": provider_settings,
                    "updatedAt": timestamp,
                    "tokenSource": "manual",
                }
            },
        },
        models={
            "version": 1,
            "providers": {
                CLINE_PROVIDER_ID: {
                    "provider": {
                        "name": "Free Claude Code",
                        "baseUrl": proxy_v1_url(proxy_root_url),
                        "defaultModelId": default_model_id,
                        "protocol": "openai-responses",
                        "client": "openai",
                        "capabilities": ["streaming", "tools"],
                    },
                    "models": model_entries,
                }
            },
        },
    )


def _model_entry(model: CatalogModel) -> JsonObject:
    supports_reasoning = model.supports_reasoning is not False
    supports_vision = (
        model.input_modalities is not None
        and ModelInputModality.IMAGE in model.input_modalities
    )
    capabilities = ["streaming", "tools"]
    if supports_reasoning:
        capabilities.append("reasoning")
    if supports_vision:
        capabilities.append("images")

    entry: JsonObject = {
        "name": model.display_name,
        "capabilities": capabilities,
        "supportsReasoning": supports_reasoning,
        "apiFormat": "openai-responses",
    }
    if model.input_modalities is not None:
        entry.update(
            {
                "supportsVision": supports_vision,
                "inputModalities": [
                    modality.value
                    for modality in ModelInputModality
                    if modality in model.input_modalities
                ],
                "outputModalities": ["text"],
            }
        )
    if model.context_window_tokens is not None:
        entry["contextWindow"] = model.context_window_tokens
    if model.max_output_tokens is not None:
        entry["maxTokens"] = model.max_output_tokens
    return entry

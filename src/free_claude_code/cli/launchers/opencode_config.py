"""Process-local OpenCode v1 configuration for FCC model routing."""

from dataclasses import dataclass

from free_claude_code.config.server_urls import proxy_v1_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses.model_catalog import ClientModel

OPENCODE_API_KEY_ENV = "FCC_OPENCODE_API_KEY"
OPENCODE_PROVIDER_ID = "free-claude-code"


@dataclass(frozen=True, slots=True)
class OpenCodeConfig:
    """Secret-free file and overlay configuration for one OpenCode process."""

    file: JsonObject
    overlay: JsonObject


def build_opencode_config(
    models: tuple[ClientModel, ...], *, proxy_root_url: str
) -> OpenCodeConfig:
    """Translate a non-empty FCC model snapshot into OpenCode v1 config."""

    if not models:
        raise ValueError("OpenCode requires at least one routable FCC model")

    model_config: JsonObject = {
        model.wire_slug: _model_config(model) for model in models
    }
    provider_config: JsonObject = {
        "name": "Free Claude Code",
        "npm": "@ai-sdk/openai",
        "options": {
            "baseURL": proxy_v1_url(proxy_root_url),
            "apiKey": f"{{env:{OPENCODE_API_KEY_ENV}}}",
        },
    }
    default_model = f"{OPENCODE_PROVIDER_ID}/{models[0].wire_slug}"

    return OpenCodeConfig(
        file={
            "provider": {
                OPENCODE_PROVIDER_ID: {
                    **provider_config,
                    "models": model_config,
                }
            }
        },
        overlay={
            "provider": {OPENCODE_PROVIDER_ID: provider_config},
            "enabled_providers": [OPENCODE_PROVIDER_ID],
            "disabled_providers": [],
            "model": default_model,
            "small_model": default_model,
        },
    )


def _model_config(model: ClientModel) -> JsonObject:
    config: JsonObject = {
        "name": model.display_name,
        "reasoning": model.supports_reasoning is not False,
    }
    if model.input_modalities is not None:
        config["modalities"] = {
            "input": [
                modality.value
                for modality in ModelInputModality
                if modality in model.input_modalities
            ]
        }
    if model.context_window_tokens is not None or model.max_output_tokens is not None:
        # OpenCode requires both fields and uses zero when either limit is unknown.
        config["limit"] = {
            "context": (
                model.context_window_tokens
                if model.context_window_tokens is not None
                else 0
            ),
            "output": (
                model.max_output_tokens if model.max_output_tokens is not None else 0
            ),
        }
    return config

"""Process-local OpenCode v2 configuration for FCC model routing."""

from dataclasses import dataclass

from free_claude_code.application.model_catalog import CatalogModel
from free_claude_code.config.server_urls import proxy_v1_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality

OPENCODE_API_KEY_ENV = "FCC_OPENCODE_API_KEY"
OPENCODE_PROVIDER_ID = "free-claude-code"


@dataclass(frozen=True, slots=True)
class OpenCodeConfig:
    """Secret-free file and overlay configuration for one OpenCode process."""

    file: JsonObject
    overlay: JsonObject


def build_opencode_config(
    models: tuple[CatalogModel, ...], *, default_model_id: str, proxy_root_url: str
) -> OpenCodeConfig:
    """Translate a non-empty FCC model snapshot into OpenCode v2 config."""

    if not models:
        raise ValueError("OpenCode requires at least one routable FCC model")

    model_config: JsonObject = {
        model.wire_slug: _model_config(model) for model in models
    }
    provider_config: JsonObject = {
        "name": "Free Claude Code",
        "package": "@opencode/ai/providers/openai/responses",
        "settings": {
            "baseURL": proxy_v1_url(proxy_root_url),
            "apiKey": f"{{env:{OPENCODE_API_KEY_ENV}}}",
        },
    }
    default_model = f"{OPENCODE_PROVIDER_ID}/{default_model_id}"
    # OpenCode's buffer is global, so it must also fit the smallest selectable
    # model. Leave its native 20K default alone when every known window fits it.
    buffer = min(
        (
            model.context_window_tokens // 4
            for model in models
            if model.context_window_tokens is not None
        ),
        default=20000,
    )
    compaction: JsonObject = (
        {"compaction": {"buffer": buffer}} if buffer < 20000 else {}
    )

    return OpenCodeConfig(
        file={
            "providers": {
                OPENCODE_PROVIDER_ID: {
                    **provider_config,
                    "models": model_config,
                }
            }
        },
        overlay={
            **compaction,
            "providers": {OPENCODE_PROVIDER_ID: provider_config},
            "experimental": {
                "policies": [
                    {"action": "provider.use", "resource": "*", "effect": "deny"},
                    {
                        "action": "provider.use",
                        "resource": OPENCODE_PROVIDER_ID,
                        "effect": "allow",
                    },
                ]
            },
            "model": default_model,
            "agents": {"title": {"model": default_model}},
        },
    )


def _model_config(model: CatalogModel) -> JsonObject:
    config: JsonObject = {"name": model.display_name}
    if model.input_modalities is not None:
        config["capabilities"] = {
            "tools": True,
            "input": [
                modality.value
                for modality in ModelInputModality
                if modality in model.input_modalities
            ],
            "output": ["text"],
        }
    limits: JsonObject = {}
    if model.context_window_tokens is not None:
        limits["context"] = model.context_window_tokens
    if model.max_output_tokens is not None:
        limits["output"] = model.max_output_tokens
    elif model.context_window_tokens is not None:
        # An omitted output limit inherits 32K in OpenCode, which can consume
        # the entire known window. This fallback is an OpenCode runtime budget.
        limits["output"] = max(1, min(4096, model.context_window_tokens // 4))
    if limits:
        config["limit"] = limits
    return config

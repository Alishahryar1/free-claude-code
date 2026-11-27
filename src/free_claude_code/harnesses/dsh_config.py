"""DeepSeek Harness provider metadata and process-local CLI overlays."""

from pathlib import Path

from free_claude_code.application.model_catalog import (
    CatalogModel,
    context_window_for_client,
)
from free_claude_code.config.server_urls import proxy_v1_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality

DSH_API_KEY_ENV = "FCC_DSH_API_KEY"
DSH_ENV_PREFIX = "FCC_DSH_"
DSH_PROVIDER_ID = "free-claude-code"

_MAX_TIMER_DELAY_MS = 2_147_483_647
_REASONING_EFFORTS: JsonObject = {
    "off": "none",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def build_dsh_launch_config(
    models: tuple[CatalogModel, ...],
    *,
    default_model_id: str,
    proxy_root_url: str,
    credentials_path: Path,
) -> tuple[JsonObject, ...]:
    """Translate FCC's catalog into an isolated DSH overlay."""

    provider = build_dsh_provider(
        models,
        proxy_root_url=proxy_root_url,
        credential_ref=DSH_API_KEY_ENV,
    )
    return (
        _configured_row(
            "credentials",
            "@deepseek-ai/dsh-credentials-local",
            {"path": str(credentials_path), "watch": False},
        ),
        _configured_row(
            "llm-pi-ai",
            "@deepseek-ai/dsh-llm-pi-ai",
            {"providers": {DSH_PROVIDER_ID: provider}},
        ),
        _configured_row(
            "agent-default-model",
            "@deepseek-ai/dsh-agent-default-model",
            {"provider": DSH_PROVIDER_ID, "model": default_model_id},
        ),
        _disabled_row("llm-deepseek", "@deepseek-ai/dsh-llm-deepseek-api-key"),
        _disabled_row("llm-deepseek-account", "@deepseek-ai/dsh-llm-deepseek-account"),
        _disabled_row(
            "web-search-deepseek",
            "@deepseek-ai/dsh-web-search-deepseek",
        ),
        _disabled_row("tool-web", "@deepseek-ai/dsh-tool-web"),
    )


def build_dsh_provider(
    models: tuple[CatalogModel, ...],
    *,
    proxy_root_url: str,
    credential_ref: str,
) -> JsonObject:
    """Build the same FCC Responses route for temporary and persistent clients."""
    if not models:
        raise ValueError("DeepSeek Harness requires at least one routable FCC model")

    return {
        "displayName": "Free Claude Code",
        "apiKeyEnv": credential_ref,
        "api": "openai-responses",
        "baseURL": proxy_v1_url(proxy_root_url),
        "models": [_model_profile(model) for model in models],
        "defaultInput": ["text"],
        "retryPolicy": {"mode": "normal", "maxRetries": 0},
        "streamIdleTimeoutMs": _MAX_TIMER_DELAY_MS,
    }


def _model_profile(model: CatalogModel) -> JsonObject:
    profile: JsonObject = {
        "id": model.wire_slug,
        "name": model.display_name,
    }
    profile["reasoningEfforts"] = (
        dict(_REASONING_EFFORTS) if model.supports_reasoning is not False else False
    )
    if model.input_modalities is not None:
        profile["input"] = [
            modality.value
            for modality in ModelInputModality
            if modality in model.input_modalities
        ]
    profile["contextWindow"] = context_window_for_client(model)
    if model.max_output_tokens is not None:
        profile["maxTokens"] = model.max_output_tokens
    return profile


def _configured_row(row_id: str, name: str, config: JsonObject) -> JsonObject:
    return {"id": row_id, "name": name, "config": config}


def _disabled_row(row_id: str, name: str) -> JsonObject:
    return {"id": row_id, "name": name, "disabled": True}

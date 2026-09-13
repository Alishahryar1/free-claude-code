"""Process-local DeepSeek Harness configuration for FCC model routing."""

import math
from pathlib import Path

from free_claude_code.config.server_urls import proxy_v1_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses.model_catalog import ClientModel

DSH_API_KEY_ENV = "FCC_DSH_API_KEY"
DSH_ENV_PREFIX = "FCC_DSH_"
DSH_PROVIDER_ID = "free-claude-code"

_MAX_TIMER_DELAY_MS = 2_147_483_647
_STREAM_TIMEOUT_MARGIN_SECONDS = 60
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
    models: tuple[ClientModel, ...],
    *,
    proxy_root_url: str,
    settings_path: Path,
    credentials_path: Path,
    provider_progress_timeout: float,
) -> tuple[JsonObject, ...]:
    """Translate FCC's catalog and timeout into an isolated DSH overlay."""

    if not models:
        raise ValueError("DeepSeek Harness requires at least one routable FCC model")

    stream_idle_timeout_ms = _stream_idle_timeout_ms(provider_progress_timeout)
    selected_model = models[0].wire_slug
    provider: JsonObject = {
        "displayName": "Free Claude Code",
        "apiKeyEnv": DSH_API_KEY_ENV,
        "api": "openai-responses",
        "baseURL": proxy_v1_url(proxy_root_url),
        "models": [_model_profile(model) for model in models],
        "defaultInput": ["text"],
        "retryPolicy": {"mode": "normal", "maxRetries": 0},
        "streamIdleTimeoutMs": stream_idle_timeout_ms,
    }
    return (
        _configured_row(
            "settings",
            "@deepseek-ai/dsh-settings-file",
            {"path": str(settings_path), "watch": False},
        ),
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
            {"provider": DSH_PROVIDER_ID, "model": selected_model},
        ),
        _disabled_row("llm-deepseek", "@deepseek-ai/dsh-llm-deepseek"),
        _disabled_row(
            "web-search-deepseek",
            "@deepseek-ai/dsh-web-search-deepseek",
        ),
        _disabled_row("tool-web", "@deepseek-ai/dsh-tool-web"),
    )


def _stream_idle_timeout_ms(provider_progress_timeout: float) -> int:
    if not math.isfinite(provider_progress_timeout) or provider_progress_timeout <= 0:
        raise ValueError("PROVIDER_PROGRESS_TIMEOUT must be a positive finite value")

    timeout_ms = math.ceil(
        (provider_progress_timeout + _STREAM_TIMEOUT_MARGIN_SECONDS) * 1000
    )
    if timeout_ms > _MAX_TIMER_DELAY_MS:
        raise ValueError(
            "PROVIDER_PROGRESS_TIMEOUT is too large for DeepSeek Harness; "
            "lower it so the DSH stream timeout stays within Node's timer limit"
        )
    return timeout_ms


def _model_profile(model: ClientModel) -> JsonObject:
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
    if model.context_window_tokens is not None:
        profile["contextWindow"] = model.context_window_tokens
    if model.max_output_tokens is not None:
        profile["maxTokens"] = model.max_output_tokens
    return profile


def _configured_row(row_id: str, name: str, config: JsonObject) -> JsonObject:
    return {"id": row_id, "name": name, "config": config}


def _disabled_row(row_id: str, name: str) -> JsonObject:
    return {"id": row_id, "name": name, "disabled": True}

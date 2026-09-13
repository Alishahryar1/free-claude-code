"""Contracts for the process-local DeepSeek Harness configuration."""

import json
import math
from pathlib import Path

import pytest

from free_claude_code.cli.launchers.dsh_config import build_dsh_launch_config
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses.model_catalog import ClientModel


def _models() -> tuple[ClientModel, ...]:
    return (
        ClientModel(
            wire_slug="nvidia_nim/vendor/model",
            provider_model_ref="nvidia_nim/vendor/model",
            display_name="Nested model",
            supports_reasoning=True,
            input_modalities=frozenset(
                {ModelInputModality.TEXT, ModelInputModality.IMAGE}
            ),
            context_window_tokens=131072,
            max_output_tokens=8192,
        ),
        ClientModel(
            wire_slug="claude-3-freecc-no-thinking/open_router/plain-model",
            provider_model_ref="open_router/plain-model",
            display_name="No-thinking model",
            supports_reasoning=False,
            input_modalities=frozenset({ModelInputModality.TEXT}),
            max_output_tokens=4096,
        ),
        ClientModel(
            wire_slug="future_provider/unknown-model",
            provider_model_ref="future_provider/unknown-model",
            display_name="Unknown model",
            supports_reasoning=None,
        ),
    )


def _row_by_id(patch: tuple[dict, ...], row_id: str) -> dict:
    return next(row for row in patch if row["id"] == row_id)


def test_dsh_config_pins_responses_models_retries_and_private_state(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    credentials_path = tmp_path / ".credentials.yaml"
    launch = build_dsh_launch_config(
        _models(),
        proxy_root_url="http://127.0.0.1:9191/",
        settings_path=settings_path,
        credentials_path=credentials_path,
        provider_progress_timeout=600.0,
    )

    settings = _row_by_id(launch, "settings")
    assert settings == {
        "id": "settings",
        "name": "@deepseek-ai/dsh-settings-file",
        "config": {"path": str(settings_path), "watch": False},
    }
    credentials = _row_by_id(launch, "credentials")
    assert credentials == {
        "id": "credentials",
        "name": "@deepseek-ai/dsh-credentials-local",
        "config": {"path": str(credentials_path), "watch": False},
    }

    llm = _row_by_id(launch, "llm-pi-ai")
    provider = llm["config"]["providers"]["free-claude-code"]
    assert provider == {
        "displayName": "Free Claude Code",
        "apiKeyEnv": "FCC_DSH_API_KEY",
        "api": "openai-responses",
        "baseURL": "http://127.0.0.1:9191/v1",
        "models": [
            {
                "id": "nvidia_nim/vendor/model",
                "name": "Nested model",
                "reasoningEfforts": {
                    "off": "none",
                    "minimal": "minimal",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "xhigh",
                    "max": "max",
                },
                "input": ["text", "image"],
                "contextWindow": 131072,
                "maxTokens": 8192,
            },
            {
                "id": "claude-3-freecc-no-thinking/open_router/plain-model",
                "name": "No-thinking model",
                "reasoningEfforts": False,
                "input": ["text"],
                "maxTokens": 4096,
            },
            {
                "id": "future_provider/unknown-model",
                "name": "Unknown model",
                "reasoningEfforts": {
                    "off": "none",
                    "minimal": "minimal",
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "xhigh": "xhigh",
                    "max": "max",
                },
            },
        ],
        "defaultInput": ["text"],
        "retryPolicy": {"mode": "normal", "maxRetries": 0},
        "streamIdleTimeoutMs": 660_000,
    }
    assert _row_by_id(launch, "agent-default-model") == {
        "id": "agent-default-model",
        "name": "@deepseek-ai/dsh-agent-default-model",
        "config": {
            "provider": "free-claude-code",
            "model": "nvidia_nim/vendor/model",
        },
    }

    for row_id, package in (
        ("llm-deepseek", "@deepseek-ai/dsh-llm-deepseek"),
        ("web-search-deepseek", "@deepseek-ai/dsh-web-search-deepseek"),
        ("tool-web", "@deepseek-ai/dsh-tool-web"),
    ):
        assert _row_by_id(launch, row_id) == {
            "id": row_id,
            "name": package,
            "disabled": True,
        }

    serialized = json.dumps(launch)
    assert "proxy-token" not in serialized
    for unsupported in ("compat", "headers", "telemetry"):
        assert unsupported not in serialized


def test_dsh_config_rounds_fractional_progress_timeout_up() -> None:
    launch = build_dsh_launch_config(
        _models(),
        proxy_root_url="http://127.0.0.1:9191",
        settings_path=Path("settings.yaml"),
        credentials_path=Path("credentials.yaml"),
        provider_progress_timeout=0.0001,
    )

    provider = _row_by_id(launch, "llm-pi-ai")["config"]["providers"][
        "free-claude-code"
    ]
    assert provider["streamIdleTimeoutMs"] == 60_001


def test_dsh_config_rejects_empty_catalog() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_dsh_launch_config(
            (),
            proxy_root_url="http://127.0.0.1:9191",
            settings_path=Path("settings.yaml"),
            credentials_path=Path("credentials.yaml"),
            provider_progress_timeout=600,
        )


@pytest.mark.parametrize("timeout", [0.0, -1.0, math.inf, math.nan])
def test_dsh_config_rejects_invalid_progress_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        build_dsh_launch_config(
            _models(),
            proxy_root_url="http://127.0.0.1:9191",
            settings_path=Path("settings.yaml"),
            credentials_path=Path("credentials.yaml"),
            provider_progress_timeout=timeout,
        )


def test_dsh_config_rejects_timeout_beyond_node_timer_limit() -> None:
    with pytest.raises(ValueError, match="too large"):
        build_dsh_launch_config(
            _models(),
            proxy_root_url="http://127.0.0.1:9191",
            settings_path=Path("settings.yaml"),
            credentials_path=Path("credentials.yaml"),
            provider_progress_timeout=2_147_424,
        )

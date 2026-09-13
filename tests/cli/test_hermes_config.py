"""Contracts for the process-local Hermes configuration."""

import json

import pytest

from free_claude_code.cli.launchers.hermes_config import (
    build_hermes_managed_config,
)
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


def test_hermes_config_pins_responses_catalog_and_fallbacks() -> None:
    managed = build_hermes_managed_config(
        (_models()[1], _models()[0], _models()[2]),
        proxy_root_url="http://127.0.0.1:9191/",
        nonce="a1b2c3",
    )

    assert managed.provider_ref == "custom:fcc-a1b2c3"
    assert managed.key_env == "FCC_HERMES_A1B2C3"
    assert managed.default_model == (
        "claude-3-freecc-no-thinking/open_router/plain-model"
    )
    assert managed.config["providers"] == {
        "fcc-a1b2c3": {
            "name": "Free Claude Code",
            "api": "http://127.0.0.1:9191/v1",
            "key_env": "FCC_HERMES_A1B2C3",
            "extra_headers": {
                "Authorization": "Bearer ${FCC_HERMES_A1B2C3}",
            },
            "transport": "codex_responses",
            "default_model": "claude-3-freecc-no-thinking/open_router/plain-model",
            "models": {
                "nvidia_nim/vendor/model": {},
                "claude-3-freecc-no-thinking/open_router/plain-model": {},
                "future_provider/unknown-model": {},
            },
            "discover_models": False,
        }
    }
    assert managed.config["model"] == {
        "provider": "custom:fcc-a1b2c3",
        "default": "claude-3-freecc-no-thinking/open_router/plain-model",
        "base_url": "",
        "api_key": "",
        "api_mode": "codex_responses",
        "default_headers": {"x-fcc-launch-id": "a1b2c3"},
    }
    assert managed.config["fallback_providers"] == []
    assert managed.config["fallback_model"] == []
    assert managed.config["model_overrides"] == {
        "custom": {
            "nvidia_nim/vendor/model": {
                "supports_reasoning": True,
                "supports_vision": True,
                "context_window": 131072,
                "max_output_tokens": 8192,
            },
            "claude-3-freecc-no-thinking/open_router/plain-model": {
                "supports_reasoning": False,
                "supports_vision": False,
                "max_output_tokens": 4096,
            },
        }
    }

    auxiliary = managed.config["auxiliary"]
    assert isinstance(auxiliary, dict)
    assert set(auxiliary) == {
        "vision",
        "web_extract",
        "compression",
        "skills_hub",
        "approval",
        "mcp",
        "title_generation",
        "memory_query_rewrite",
        "tts_audio_tags",
        "triage_specifier",
        "kanban_decomposer",
        "profile_describer",
        "goal_judge",
        "curator",
        "monitor",
        "background_review",
        "moa_reference",
        "moa_aggregator",
    }
    for task, task_config in auxiliary.items():
        assert isinstance(task_config, dict)
        expected = {
            "provider": "custom:fcc-a1b2c3",
            "model": "",
            "base_url": "",
            "api_key": "",
            "extra_body": {},
            "fallback_chain": [],
        }
        if task == "title_generation":
            expected["prefer_fast_model"] = False
        assert task_config == expected

    serialized = json.dumps(managed.config)
    assert "proxy-token" not in serialized
    assert "transient_retries" not in serialized
    assert "timeout" not in serialized


def test_hermes_config_uses_first_model_by_default() -> None:
    managed = build_hermes_managed_config(
        _models(),
        proxy_root_url="http://127.0.0.1:9191",
        nonce="ABC123",
    )

    assert managed.default_model == "nvidia_nim/vendor/model"
    assert managed.key_env == "FCC_HERMES_ABC123"


def test_hermes_config_rejects_empty_catalog() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_hermes_managed_config(
            (),
            proxy_root_url="http://127.0.0.1:9191",
            nonce="abc123",
        )


@pytest.mark.parametrize("nonce", ["", "not-valid", "spaces here"])
def test_hermes_config_rejects_unsafe_nonce(nonce: str) -> None:
    with pytest.raises(ValueError, match="alphanumeric"):
        build_hermes_managed_config(
            _models(),
            proxy_root_url="http://127.0.0.1:9191",
            nonce=nonce,
        )

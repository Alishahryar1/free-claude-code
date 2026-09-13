"""Contracts for Aider's process-local FCC model files."""

import json

import pytest

from free_claude_code.cli.launchers.aider_config import build_aider_config
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
            wire_slug="ollama_cloud/qwen3-coder:480b",
            provider_model_ref="ollama_cloud/qwen3-coder:480b",
            display_name="Colon model",
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


def test_aider_config_projects_messages_route_and_canonical_catalog() -> None:
    config = build_aider_config(
        _models(),
        messages_url="http://127.0.0.1:9191/v1/messages",
        api_key_env="FCC_AIDER_PROXY_AUTH_A1B2C3",
        launch_id="launch-a",
    )

    expected_metadata = {
        "nvidia_nim/vendor/model": {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_vision": True,
            "max_input_tokens": 131072,
            "max_output_tokens": 8192,
        },
        "ollama_cloud/qwen3-coder:480b": {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_vision": False,
            "max_output_tokens": 4096,
        },
        "future_provider/unknown-model": {
            "litellm_provider": "anthropic",
            "mode": "chat",
        },
    }
    entries = {entry["name"]: entry for entry in config.settings}
    assert len(entries) == len(config.metadata) == 6
    for wire_name, metadata in expected_metadata.items():
        for name in (wire_name, f"anthropic/{wire_name}"):
            assert config.metadata[name] == metadata
            entry = entries[name]
            assert entry["weak_model_name"] == name
            assert entry["editor_model_name"] == name
            assert entry["extra_params"] == {
                "model": f"anthropic/{wire_name}",
                "api_base": "http://127.0.0.1:9191/v1/messages",
                "api_key": "os.environ/FCC_AIDER_PROXY_AUTH_A1B2C3",
                "extra_headers": {"x-fcc-launch-id": "launch-a"},
            }
    assert entries["nvidia_nim/vendor/model"]["accepts_settings"] == [
        "reasoning_effort"
    ]
    assert entries["ollama_cloud/qwen3-coder:480b"]["accepts_settings"] == []
    assert "accepts_settings" not in entries["future_provider/unknown-model"]

    serialized = json.dumps({"settings": config.settings, "metadata": config.metadata})
    assert json.loads(serialized) == {
        "settings": config.settings,
        "metadata": config.metadata,
    }
    assert "proxy-token" not in serialized
    for fabricated_key in (
        "context_window",
        "max_tokens",
        "input_cost_per_token",
        "output_cost_per_token",
        "edit_format",
    ):
        assert fabricated_key not in serialized


def test_aider_config_rejects_empty_catalog() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_aider_config(
            (),
            messages_url="http://127.0.0.1:9191/v1/messages",
            api_key_env="FCC_AIDER_PROXY_AUTH_A1B2C3",
            launch_id="launch-a",
        )


def test_aider_catalog_ids_take_precedence_over_generated_transport_spellings() -> None:
    models = (
        ClientModel("provider/model", "provider/model", "First", None),
        ClientModel(
            "anthropic/provider/model", "anthropic/provider/model", "Second", None
        ),
    )
    config = build_aider_config(
        models,
        messages_url="http://localhost:8182/v1/messages",
        api_key_env="FCC_AIDER_PROXY_AUTH_COLLISION",
        launch_id="launch-a",
    )
    entries = {entry["name"]: entry for entry in config.settings}
    extra = entries["anthropic/provider/model"]["extra_params"]
    assert isinstance(extra, dict)
    assert extra["model"] == "anthropic/anthropic/provider/model"
    assert len(entries) == 3


@pytest.mark.parametrize(
    "api_key_env",
    [
        "",
        "FCC_AIDER_PROXY_AUTH_",
        "ANTHROPIC_API_KEY",
        "FCC_AIDER_PROXY_AUTH_lowercase",
        "FCC_AIDER_PROXY_AUTH_BAD-NAME",
        "FCC_AIDER_PROXY_AUTH_BAD/NAME",
    ],
)
def test_aider_config_rejects_invalid_api_key_environment_name(
    api_key_env: str,
) -> None:
    with pytest.raises(ValueError, match="environment"):
        build_aider_config(
            _models(),
            messages_url="http://127.0.0.1:9191/v1/messages",
            api_key_env=api_key_env,
            launch_id="launch-a",
        )

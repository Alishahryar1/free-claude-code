"""OpenCode launcher contract tests."""

import json
from pathlib import Path

import pytest

from free_claude_code.cli.launchers.opencode_config import build_opencode_config
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.harnesses.model_catalog import ClientModel


def test_opencode_config_uses_responses_sdk_and_only_known_metadata() -> None:
    config = build_opencode_config(
        (
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
                context_window_tokens=65536,
            ),
            ClientModel(
                wire_slug="future_provider/unknown-model",
                provider_model_ref="future_provider/unknown-model",
                display_name="Unknown model",
                supports_reasoning=None,
            ),
            ClientModel(
                wire_slug="future_provider/output-only",
                provider_model_ref="future_provider/output-only",
                display_name="Output-only model",
                supports_reasoning=None,
                max_output_tokens=4096,
            ),
        ),
        proxy_root_url="http://127.0.0.1:9191",
    )

    provider = config.file["provider"]
    assert isinstance(provider, dict)
    fcc = provider["free-claude-code"]
    assert isinstance(fcc, dict)
    assert fcc["npm"] == "@ai-sdk/openai"
    assert fcc["options"] == {
        "baseURL": "http://127.0.0.1:9191/v1",
        "apiKey": "{env:FCC_OPENCODE_API_KEY}",
    }
    assert fcc["models"] == {
        "nvidia_nim/vendor/model": {
            "name": "Nested model",
            "reasoning": True,
            "modalities": {"input": ["text", "image"]},
            "limit": {"context": 131072, "output": 8192},
        },
        "claude-3-freecc-no-thinking/open_router/plain-model": {
            "name": "No-thinking model",
            "reasoning": False,
            "modalities": {"input": ["text"]},
            "limit": {"context": 65536, "output": 0},
        },
        "future_provider/unknown-model": {
            "name": "Unknown model",
            "reasoning": True,
        },
        "future_provider/output-only": {
            "name": "Output-only model",
            "reasoning": True,
            "limit": {"context": 0, "output": 4096},
        },
    }
    assert config.overlay == {
        "provider": {
            "free-claude-code": {
                "name": "Free Claude Code",
                "npm": "@ai-sdk/openai",
                "options": {
                    "baseURL": "http://127.0.0.1:9191/v1",
                    "apiKey": "{env:FCC_OPENCODE_API_KEY}",
                },
            }
        },
        "enabled_providers": ["free-claude-code"],
        "disabled_providers": [],
        "model": "free-claude-code/nvidia_nim/vendor/model",
        "small_model": "free-claude-code/nvidia_nim/vendor/model",
    }
    serialized = json.dumps(config.file | config.overlay)
    assert "proxy-token" not in serialized
    assert "attachment" not in serialized


def test_opencode_config_rejects_empty_model_catalog() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_opencode_config((), proxy_root_url="http://127.0.0.1:9191")


def test_opencode_child_receives_private_catalog_and_overlay(launch_capture) -> None:
    from tests.cli.test_launcher_workflow import launch

    def inspect(command, env):
        config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        overlay = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        provider = config["provider"]["free-claude-code"]
        assert provider["options"]["baseURL"] == "http://127.0.0.1:8182/v1"
        assert "nvidia_nim/catalog-model:variant" in provider["models"]
        assert overlay["enabled_providers"] == ["free-claude-code"]
        assert env["FCC_OPENCODE_API_KEY"] == "launcher-test-token"

    launch_capture.on_start = inspect
    launch("opencode", ["models"])


@pytest.mark.parametrize("key", ("OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT"))
def test_existing_opencode_process_configuration_is_not_replaced(
    key, launch_capture, monkeypatch
) -> None:
    from tests.cli.test_launcher_workflow import launch

    monkeypatch.setenv(key, "user-owned-config")
    launch("opencode", [], exit_code=1)
    assert not launch_capture.commands

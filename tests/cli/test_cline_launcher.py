"""Contract tests for the installed `fcc-cline` launcher."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from free_claude_code.cli.launchers.cline_config import (
    CLINE_PROVIDER_ID,
    build_cline_config,
)
from free_claude_code.cli.launchers.model_catalog import ClientModel
from free_claude_code.core.model_capabilities import ModelInputModality


def test_cline_config_uses_responses_and_only_known_metadata() -> None:
    config = build_cline_config(
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
        ),
        proxy_root_url="http://127.0.0.1:9191/",
        auth_token="proxy-token",
        launch_id="launch-a",
        now=datetime(2026, 8, 15, 12, 30, tzinfo=UTC),
    )

    assert config.providers == {
        "version": 1,
        "modes": {},
        "lastUsedProvider": "openai-native",
        "providers": {
            "openai-native": {
                "settings": {
                    "provider": "openai-native",
                    "apiKey": "proxy-token",
                    "headers": {"x-fcc-launch-id": "launch-a"},
                    "model": "nvidia_nim/vendor/model",
                    "protocol": "openai-responses",
                    "baseUrl": "http://127.0.0.1:9191/v1",
                    "capabilities": ["streaming", "tools"],
                },
                "updatedAt": "2026-08-15T12:30:00Z",
                "tokenSource": "manual",
            }
        },
    }
    assert config.models == {
        "version": 1,
        "providers": {
            "openai-native": {
                "provider": {
                    "name": "Free Claude Code",
                    "baseUrl": "http://127.0.0.1:9191/v1",
                    "defaultModelId": "nvidia_nim/vendor/model",
                    "protocol": "openai-responses",
                    "client": "openai",
                    "capabilities": ["streaming", "tools"],
                },
                "models": {
                    "nvidia_nim/vendor/model": {
                        "name": "Nested model",
                        "capabilities": [
                            "streaming",
                            "tools",
                            "reasoning",
                            "images",
                        ],
                        "supportsReasoning": True,
                        "supportsVision": True,
                        "inputModalities": ["text", "image"],
                        "outputModalities": ["text"],
                        "contextWindow": 131072,
                        "maxTokens": 8192,
                        "apiFormat": "openai-responses",
                    },
                    "claude-3-freecc-no-thinking/open_router/plain-model": {
                        "name": "No-thinking model",
                        "capabilities": ["streaming", "tools"],
                        "supportsReasoning": False,
                        "supportsVision": False,
                        "inputModalities": ["text"],
                        "outputModalities": ["text"],
                        "contextWindow": 65536,
                        "apiFormat": "openai-responses",
                    },
                    "future_provider/unknown-model": {
                        "name": "Unknown model",
                        "capabilities": ["streaming", "tools", "reasoning"],
                        "supportsReasoning": True,
                        "apiFormat": "openai-responses",
                    },
                },
            }
        },
    }
    serialized = json.dumps(config.providers | config.models)
    assert "reasoningEffort" not in serialized
    assert "proxy-token" not in repr(config)


def test_cline_config_rejects_empty_model_catalog() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_cline_config(
            (),
            proxy_root_url="http://127.0.0.1:9191",
            auth_token="proxy-token",
            launch_id="launch-a",
        )


def test_cline_native_configuration_is_available_to_child(launch_capture) -> None:
    from tests.cli.test_launcher_workflow import launch

    launch_ids = []
    hub_paths = []

    def inspect(command, env):
        providers_path = Path(env["CLINE_PROVIDER_SETTINGS_PATH"])
        providers = json.loads(providers_path.read_text())
        models = json.loads(providers_path.with_name("models.json").read_text())
        settings = providers["providers"][CLINE_PROVIDER_ID]["settings"]
        assert settings["baseUrl"] == "http://127.0.0.1:8182/v1"
        assert settings["apiKey"] == "launcher-test-token"
        launch_ids.append(settings["headers"]["x-fcc-launch-id"])
        assert launch_ids[-1]
        assert settings["model"] in models["providers"][CLINE_PROVIDER_ID]["models"]
        assert env["CLINE_SESSION_BACKEND_MODE"] == "local"
        assert command[:3] == ["cline", "--provider", "openai-native"]
        hub_path = Path(env["CLINE_HUB_DISCOVERY_PATH"])
        assert hub_path.parent.name == "hub"
        assert hub_path.name == "production.json"
        # Hub discovery must resolve inside the same ephemeral tree as the
        # private provider files. The hub/ directory is pre-created so the
        # child can record discovery state; the file itself must not exist.
        assert hub_path.parent.parent == providers_path.parent.parent
        assert hub_path.parent.is_dir()
        assert not hub_path.exists()
        hub_paths.append(hub_path)

    launch_capture.on_start = inspect
    args = ["--model", "native-selection", "--data-dir", "native-data-dir"]
    launch("cline", args)
    assert launch_capture.commands[0][-len(args) :] == args
    launch("cline", [])
    assert launch_ids[0] != launch_ids[1]
    assert hub_paths[0] != hub_paths[1]
    assert all(not hub_path.exists() for hub_path in hub_paths)


def test_cline_hub_discovery_path_overrides_stale_inherited_value(
    launch_capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.cli.test_launcher_workflow import launch

    monkeypatch.setenv("CLINE_HUB_DISCOVERY_PATH", "stale-user-hub-discovery.json")
    dir_existed_at_start = []

    def inspect(command, env):
        hub_dir = Path(env["CLINE_HUB_DISCOVERY_PATH"]).parent
        dir_existed_at_start.append(hub_dir.is_dir())

    launch_capture.on_start = inspect
    launch("cline", [])
    hub_path = Path(launch_capture.environments[0]["CLINE_HUB_DISCOVERY_PATH"])
    # The stale user-level discovery file must be replaced by the isolated
    # ephemeral path, never inherited by the fcc-cline child. The hub/
    # directory is pre-created at launch (checked via on_start, since the
    # ephemeral tree is cleaned up when the launcher exits); only the file
    # itself must not be pre-created.
    assert hub_path.name == "production.json"
    assert hub_path.parent.name == "hub"
    assert str(hub_path) != "stale-user-hub-discovery.json"
    assert dir_existed_at_start == [True]
    assert not hub_path.exists()

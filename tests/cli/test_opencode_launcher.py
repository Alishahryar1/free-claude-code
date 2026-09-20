"""OpenCode launcher contract tests."""

import json
from pathlib import Path

import pytest

from free_claude_code.application.model_catalog import CatalogModel
from free_claude_code.cli.launchers.opencode_config import build_opencode_config
from free_claude_code.core.model_capabilities import ModelInputModality


def test_opencode_config_uses_native_responses_and_model_budgets() -> None:
    config = build_opencode_config(
        (
            CatalogModel(
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
            CatalogModel(
                wire_slug="claude-3-freecc-no-thinking/open_router/plain-model",
                provider_model_ref="open_router/plain-model",
                display_name="No-thinking model",
                supports_reasoning=False,
                input_modalities=frozenset({ModelInputModality.TEXT}),
                context_window_tokens=65536,
            ),
            CatalogModel(
                wire_slug="future_provider/unknown-model",
                provider_model_ref="future_provider/unknown-model",
                display_name="Unknown model",
                supports_reasoning=None,
            ),
            CatalogModel(
                wire_slug="future_provider/output-only",
                provider_model_ref="future_provider/output-only",
                display_name="Output-only model",
                supports_reasoning=None,
                max_output_tokens=4096,
            ),
        ),
        default_model_id="nvidia_nim/vendor/model",
        proxy_root_url="http://127.0.0.1:9191",
    )

    provider = config.file["providers"]
    assert isinstance(provider, dict)
    fcc = provider["free-claude-code"]
    assert isinstance(fcc, dict)
    assert fcc["package"] == "@opencode/ai/providers/openai/responses"
    assert fcc["settings"] == {
        "baseURL": "http://127.0.0.1:9191/v1",
        "apiKey": "{env:FCC_OPENCODE_API_KEY}",
    }
    assert fcc["models"] == {
        "nvidia_nim/vendor/model": {
            "name": "Nested model",
            "capabilities": {
                "tools": True,
                "input": ["text", "image"],
                "output": ["text"],
            },
            "limit": {"context": 131072, "output": 8192},
        },
        "claude-3-freecc-no-thinking/open_router/plain-model": {
            "name": "No-thinking model",
            "capabilities": {"tools": True, "input": ["text"], "output": ["text"]},
            "limit": {"context": 65536, "output": 4096},
        },
        "future_provider/unknown-model": {
            "name": "Unknown model",
        },
        "future_provider/output-only": {
            "name": "Output-only model",
            "limit": {"output": 4096},
        },
    }
    assert config.overlay == {
        "compaction": {"buffer": 16384},
        "providers": {
            "free-claude-code": {
                "name": "Free Claude Code",
                "package": "@opencode/ai/providers/openai/responses",
                "settings": {
                    "baseURL": "http://127.0.0.1:9191/v1",
                    "apiKey": "{env:FCC_OPENCODE_API_KEY}",
                },
            }
        },
        "experimental": {
            "policies": [
                {"action": "provider.use", "resource": "*", "effect": "deny"},
                {
                    "action": "provider.use",
                    "resource": "free-claude-code",
                    "effect": "allow",
                },
            ]
        },
        "model": "free-claude-code/nvidia_nim/vendor/model",
        "agents": {"title": {"model": "free-claude-code/nvidia_nim/vendor/model"}},
    }
    serialized = json.dumps(config.file | config.overlay)
    assert "proxy-token" not in serialized
    assert "attachment" not in serialized


@pytest.mark.parametrize(
    ("context", "output", "expected_limits", "expected_buffer"),
    [
        (32768, None, {"context": 32768, "output": 4096}, 8192),
        (8192, None, {"context": 8192, "output": 2048}, 2048),
        (16384, 1024, {"context": 16384, "output": 1024}, 4096),
        (131072, 65536, {"context": 131072, "output": 65536}, None),
        (200000, None, {"context": 200000, "output": 4096}, None),
        (None, 4096, {"output": 4096}, None),
        (None, None, None, None),
    ],
)
def test_opencode_context_reserves_fit_small_models(
    context: int | None,
    output: int | None,
    expected_limits: dict[str, int] | None,
    expected_buffer: int | None,
) -> None:
    config = build_opencode_config(
        (
            CatalogModel(
                wire_slug="test/model",
                provider_model_ref="test/model",
                display_name="Test model",
                supports_reasoning=None,
                context_window_tokens=context,
                max_output_tokens=output,
            ),
        ),
        default_model_id="test/model",
        proxy_root_url="http://127.0.0.1:8182",
    )
    provider = config.file["providers"]
    assert isinstance(provider, dict)
    fcc = provider["free-claude-code"]
    assert isinstance(fcc, dict)
    models = fcc["models"]
    assert isinstance(models, dict)
    model = models["test/model"]
    assert isinstance(model, dict)
    assert model.get("limit") == expected_limits
    assert config.overlay.get("compaction") == (
        {"buffer": expected_buffer} if expected_buffer is not None else None
    )


def test_opencode_config_rejects_empty_model_catalog() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_opencode_config(
            (),
            default_model_id="nvidia_nim/vendor/model",
            proxy_root_url="http://127.0.0.1:9191",
        )


def test_opencode_child_receives_private_catalog_and_overlay(launch_capture) -> None:
    from tests.cli.test_launcher_workflow import launch

    def inspect(command, env):
        config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        overlay = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        provider = config["providers"]["free-claude-code"]
        assert provider["settings"]["baseURL"] == "http://127.0.0.1:8182/v1"
        assert "nvidia_nim/catalog-model:variant" in provider["models"]
        assert overlay["experimental"]["policies"][-1]["resource"] == "free-claude-code"
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


@pytest.mark.parametrize(
    "args, expected",
    [
        ([], ["--standalone"]),
        (["run", "hello"], ["run", "hello", "--standalone"]),
        (["models"], ["models", "--standalone"]),
        (["session", "list"], ["session", "list", "--standalone"]),
        (
            ["run", "--", "--standalone=false"],
            ["run", "--standalone", "--", "--standalone=false"],
        ),
    ],
)
def test_opencode_owns_a_private_server_without_rewriting_positionals(
    args, expected, launch_capture
) -> None:
    from tests.cli.test_launcher_workflow import launch

    launch("opencode", args)
    assert launch_capture.commands[0][1:] == expected


@pytest.mark.parametrize(
    "option",
    [
        "--standalone",
        "--no-standalone",
        "--standalone=false",
        "--standalone=0",
        "--no-standalone=false",
    ],
)
def test_opencode_rejects_caller_control_of_server_lifetime(
    option, launch_capture, capsys
) -> None:
    from tests.cli.test_launcher_workflow import launch

    launch("opencode", ["run", option, "false", "hello"], exit_code=1)
    assert not launch_capture.commands
    assert "FCC manages OpenCode standalone mode" in capsys.readouterr().err

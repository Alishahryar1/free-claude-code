"""Hermes's native managed overlay and activation check."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from free_claude_code.config.paths import launcher_temp_dir_path
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


def test_hermes_uses_native_defaults_and_keeps_user_commands(
    launch_capture: LaunchCapture,
) -> None:
    launch_ids = []

    def inspect(command, env):
        config = json.loads(
            (Path(env["HERMES_MANAGED_DIR"]) / "config.yaml").read_text()
        )
        provider = env["HERMES_INFERENCE_PROVIDER"]
        assert provider == config["model"]["provider"]
        assert env["HERMES_INFERENCE_MODEL"] == config["model"]["default"]
        (entry,) = config["providers"].values()
        assert entry["transport"] == "codex_responses"
        assert entry["api"] == "http://127.0.0.1:8182/v1"
        assert env[entry["key_env"]] == "launcher-test-token"
        launch_ids.append(config["model"]["default_headers"]["x-fcc-launch-id"])
        assert launch_ids[-1]
        assert all(aux["provider"] == provider for aux in config["auxiliary"].values())
        assert "launcher-test-token" not in json.dumps(config)

    launch_capture.on_start = inspect
    args = [
        "--profile",
        "native-profile",
        "--model",
        "native-model",
        "--oneshot",
        "hello",
    ]
    launch("hermes", args)
    assert launch_capture.commands == [["hermes", *args]]
    assert launch_capture.probes[-1] == [
        "hermes",
        "config",
        "get",
        "model.provider",
        "--json",
    ]
    launch("hermes", [])
    assert launch_ids[0] != launch_ids[1]


@pytest.mark.parametrize("output", ['"other-provider"', "{}", "", "not json"])
def test_hermes_failed_activation_cleans_configuration(
    output: str, launch_capture: LaunchCapture
) -> None:
    original = launch_capture.probe

    def probe(command, **kwargs):
        if "config" in command:
            return subprocess.CompletedProcess(command, 0, output, "")
        return original(command, **kwargs)

    with patch.object(subprocess, "run", side_effect=probe):
        launch("hermes", [], exit_code=1)
    assert not launch_capture.commands
    assert list(launcher_temp_dir_path().iterdir()) == []


def test_existing_hermes_managed_policy_is_not_replaced(
    launch_capture: LaunchCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_MANAGED_DIR", "organization-policy")
    launch("hermes", [], exit_code=1)
    assert not launch_capture.commands


def test_hermes_activation_timeout_stops_and_cleans(
    launch_capture: LaunchCapture,
) -> None:
    original = launch_capture.probe

    def probe(command, **kwargs):
        if "config" in command:
            assert kwargs["timeout"] == 15.0
            raise subprocess.TimeoutExpired(command, 15)
        return original(command, **kwargs)

    with patch.object(subprocess, "run", side_effect=probe):
        launch("hermes", [], exit_code=1)
    assert not launch_capture.commands
    assert list(launcher_temp_dir_path().iterdir()) == []

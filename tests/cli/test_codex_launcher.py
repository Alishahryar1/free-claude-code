"""Codex's CLI catalog and external credential integration."""

import tomllib
from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers import codex, runner
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


def test_external_credential_command_needs_no_native_launch(
    launch_capture: LaunchCapture, capsys: pytest.CaptureFixture[str]
) -> None:
    with patch.object(
        runner, "launch_harness", side_effect=AssertionError("must not launch")
    ):
        codex.launch(["--print-proxy-auth-token"])
    assert capsys.readouterr() == ("launcher-test-token\n", "")
    assert not launch_capture.commands
    assert not launch_capture.probes
    assert not launch_capture.requests


def test_codex_selects_the_advertised_slug_from_opaque_catalog_data(
    launch_capture: LaunchCapture,
) -> None:
    launch_capture.catalog = {
        "default_model_id": "arbitrary-wire-id",
        "data": [
            {
                "id": "arbitrary-wire-id",
                "provider_model_ref": "nvidia_nim/catalog-model:variant",
                "supportsReasoning": False,
            }
        ],
    }
    launch("codex", ["exec", "--model", "native-choice", "hello"])
    command = launch_capture.commands[0]
    config = tomllib.loads(
        "\n".join(command[i + 1] for i, arg in enumerate(command) if arg == "-c")
    )
    assert config["model"] == "arbitrary-wire-id"
    assert config["model_providers"]["fcc"]["wire_api"] == "responses"
    assert config["model_providers"]["fcc"]["base_url"] == "http://127.0.0.1:8182/v1"
    assert config["model_providers"]["fcc"]["auth"] == {
        "command": "fcc-codex",
        "args": ["--print-proxy-auth-token"],
    }
    assert command[-4:] == ["exec", "--model", "native-choice", "hello"]


def test_codex_scrubs_parent_task_identity_and_preserves_native_home(
    launch_capture: LaunchCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", "native-codex-home")
    monkeypatch.setenv("CODEX_THREAD_ID", "parent-task")
    monkeypatch.setenv("OPENAI_SOMETHING", "old-route")
    launch("codex", [])
    env = launch_capture.environments[0]
    assert env["CODEX_HOME"] == "native-codex-home"
    assert "CODEX_THREAD_ID" not in env
    assert not any(key.startswith("OPENAI_") for key in env)

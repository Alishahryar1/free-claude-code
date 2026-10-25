"""Antigravity uses explicit native setup and the shared FCC launcher lifecycle."""

import json
import os
from unittest.mock import patch

import pytest

from free_claude_code.cli.launchers import agy, common
from free_claude_code.harnesses import antigravity_integration as integration
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


def test_default_model_is_selected_without_replacing_native_home_or_history(
    launch_capture: LaunchCapture, monkeypatch
):
    for key, value in {
        "GEMINI_API_KEY": "personal-key",
        "GOOGLE_GEMINI_BASE_URL": "https://personal.invalid",
        "HOME": "normal-home",
        "USERPROFILE": "normal-profile",
    }.items():
        monkeypatch.setenv(key, value)
    parent = dict(os.environ)
    launch("agy", ["--print", "hello"])
    assert launch_capture.commands == [
        ["agy", "--model", "nvidia_nim/catalog-model:variant", "--print", "hello"]
    ]
    env = launch_capture.environments[0]
    assert env["GEMINI_API_KEY"] == "launcher-test-token"
    assert env["GOOGLE_GEMINI_BASE_URL"] == "http://127.0.0.1:8182"
    assert env["HOME"] == parent["HOME"] and env["USERPROFILE"] == parent["USERPROFILE"]
    assert dict(os.environ) == parent
    for path in (integration.config_path(), integration.state_path()):
        assert "launcher-test-token" not in path.read_text()


@pytest.mark.parametrize(
    "args",
    [
        ["models"],
        ["mcp", "list"],
        ["plugin", "list"],
        ["help", "models"],
        ["--conversation", "native-id", "--model", "provider/explicit:free"],
        ["--continue"],
        ["--print", "literal --disconnect"],
    ],
)
def test_native_subcommands_and_session_arguments_keep_native_scope(
    args, launch_capture
):
    launch("agy", args)
    expected = (
        ["agy", *args]
        if not args[0].startswith("-")
        else ["agy", "--model", "nvidia_nim/catalog-model:variant", *args]
    )
    assert launch_capture.commands == [expected]
    assert "--dangerously-skip-permissions" not in expected


def test_connect_is_explicit_and_runs_native_catalog_verification(launch_capture):
    integration.disconnect(integration.config_path(), integration.state_path())
    launch("agy", ["--connect"])
    assert launch_capture.commands == [["agy", "models"]]
    assert (
        json.loads(integration.config_path().read_text())["modelProvider"] == "gemini"
    )


def test_unconnected_launch_does_not_silently_change_provider(launch_capture, capsys):
    integration.disconnect(integration.config_path(), integration.state_path())
    before = integration.config_path().read_bytes()
    launch("agy", [], exit_code=1)
    assert not launch_capture.commands
    assert integration.config_path().read_bytes() == before
    assert "fcc-agy --connect" in capsys.readouterr().err


def test_disconnect_works_without_binary_settings_server_or_catalog(launch_capture):
    with patch.object(
        common.shutil, "which", side_effect=AssertionError("no binary needed")
    ):
        agy.launch(["--disconnect"])
    assert not launch_capture.requests and not launch_capture.probes
    assert not launch_capture.commands
    assert not integration.state_path().exists()
    assert "modelProvider" not in json.loads(integration.config_path().read_text())


@pytest.mark.parametrize("output", ["1.4.6", "1.4.7-beta.1", "unknown"])
def test_unsupported_native_version_stops_before_fcc(output, launch_capture):
    launch_capture.versions["agy"] = output
    launch("agy", [], exit_code=126)
    assert not launch_capture.requests and not launch_capture.commands

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


@pytest.mark.parametrize("args", [["--connect"], ["--disconnect"]])
@pytest.mark.parametrize("native", ["missing", "old", "supported"])
def test_old_connection_commands_point_to_admin_without_mutation(
    args, launch_capture, capsys, monkeypatch, native
):
    monkeypatch.setenv("PORT", "4567")
    if native == "missing":
        monkeypatch.setattr(common.shutil, "which", lambda name: None)
    elif native == "old":
        launch_capture.versions["agy"] = "1.4.6"
    before = (
        integration.config_path().read_bytes(),
        integration.state_path().read_bytes(),
    )
    launch("agy", args, exit_code=1)
    assert not launch_capture.commands
    assert not launch_capture.requests and not launch_capture.probes
    error = capsys.readouterr().err
    assert "http://127.0.0.1:4567/admin/integrations" in error
    assert "fcc-server" in error
    assert (
        integration.config_path().read_bytes(),
        integration.state_path().read_bytes(),
    ) == before


@pytest.mark.parametrize("host, expected", [("0.0.0.0", "127.0.0.1"), ("::1", "[::1]")])
def test_unconnected_launch_precedes_catalog_and_links_actual_admin(
    launch_capture, capsys, monkeypatch, host, expected
):
    monkeypatch.setenv("HOST", host)
    monkeypatch.setenv("PORT", "4567")
    launch_capture.catalog_error = ValueError("catalog unavailable")
    integration.disconnect(integration.config_path(), integration.state_path())
    before = integration.config_path().read_bytes()
    launch("agy", [], exit_code=1)
    assert not launch_capture.commands
    assert integration.config_path().read_bytes() == before
    assert [request.full_url for request in launch_capture.requests] == [
        f"http://{expected}:4567/health"
    ]
    error = capsys.readouterr().err
    assert "disconnected" in error.lower()
    assert f"http://{expected}:4567/admin/integrations" in error
    assert "catalog unavailable" not in error


def test_disconnect_during_catalog_refresh_stops_launch(launch_capture, capsys):
    fetch = agy.fetch_proxy_model_catalog

    def disconnect(*args, **kwargs):
        result = fetch(*args, **kwargs)
        integration.disconnect(integration.config_path(), integration.state_path())
        return result

    with patch.object(agy, "fetch_proxy_model_catalog", disconnect):
        launch("agy", [], exit_code=1)
    assert not launch_capture.commands
    assert not integration.state_path().exists()
    assert "modelProvider" not in json.loads(integration.config_path().read_text())
    assert "http://127.0.0.1:8182/admin/integrations" in capsys.readouterr().err


def test_old_connection_guidance_does_not_leak_settings_errors(launch_capture, capsys):
    with patch.object(agy, "get_settings", side_effect=ValueError("sensitive-value")):
        launch("agy", ["--disconnect"], exit_code=1)
    error = capsys.readouterr().err
    assert "fcc-server" in error and "sensitive-value" not in error
    assert not launch_capture.commands and not launch_capture.probes


@pytest.mark.parametrize("output", ["1.4.6", "1.4.7-beta.1", "unknown"])
def test_unsupported_native_version_stops_before_fcc(output, launch_capture):
    launch_capture.versions["agy"] = output
    launch("agy", [], exit_code=126)
    assert not launch_capture.requests and not launch_capture.commands

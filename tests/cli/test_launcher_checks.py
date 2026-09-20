"""One bootstrap contract, with only native probe syntax varying by harness."""

import importlib
import subprocess
from unittest.mock import patch
from urllib.error import URLError

import pytest

from free_claude_code.cli.launchers import common, runner
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import HARNESSES, launch


@pytest.mark.parametrize(
    ("name", "output", "compatible"),
    [
        ("opencode", "opencode v2.0.9", True),
        ("opencode", "opencode version v2.0.10+build", True),
        ("opencode", "opencode version 2.0.10", True),
        ("opencode", "v2.0.10", True),
        ("opencode", "2.0.10", True),
        ("opencode", "1.18.18", False),
        ("opencode", "opencode version 1.19.0+build", False),
        ("opencode", "1.18.17", False),
        ("opencode", "1.18.18-beta.1", False),
        ("opencode", "opencode v2.0.10-beta.1", False),
        ("opencode", "opencode v3.0.0", False),
        ("opencode", "opaque upstream version output", False),
        ("cline", "cline 3.0.55", True),
        ("cline", "v3.1.0", True),
        ("cline", "3.0.54", False),
        ("cline", "3.1.0-beta", False),
        ("hermes", "Hermes Agent v0.20.4", True),
        ("hermes", "Hermes Agent 0.20.3", False),
        ("dsh", "dsh v0.1.0-rc.8", True),
        ("dsh", "0.1.0-rc.7", False),
        ("dsh", "0.1.0-rc.9", False),
        ("grok", '{"currentVersion":"1.0.5"}', True),
        ("grok", '{"currentVersion":"1.1.0+build (build info)"}', True),
        ("grok", '{"currentVersion":"1.0.4"}', False),
        ("grok", '{"currentVersion":"1.0.5-beta"}', False),
        ("grok", '{"currentVersion":3}', False),
        ("grok", "[]", False),
        ("muse", "Muse Code 0.2.1", True),
        ("muse", "Muse Code 1.0.3 (1.0.3-R2198.1)", True),
        ("muse", "Muse Code 0.2.0", False),
        ("pi", "0.80.4", True),
        ("pi", "0.85.0", True),
        ("pi", "0.80.3", False),
        ("pi", "0.80.4-beta", False),
        ("pi", "--extension --models", False),
    ],
)
def test_native_compatibility_precedes_fcc_setup(
    name: str, output: str, compatible: bool, launch_capture: LaunchCapture
) -> None:
    launch_capture.versions[name] = output
    launch(name, ["--help"], exit_code=23 if compatible else 126)
    assert bool(launch_capture.commands) is compatible
    assert bool(launch_capture.requests) is compatible


@pytest.mark.parametrize(
    "name", ("pi", "opencode", "cline", "hermes", "dsh", "grok", "muse")
)
@pytest.mark.parametrize(
    "error", (OSError("cannot execute"), subprocess.TimeoutExpired("probe", 5))
)
def test_native_probe_failure_stops_before_fcc(
    name: str, error: Exception, launch_capture: LaunchCapture
) -> None:
    with patch.object(subprocess, "run", side_effect=error):
        launch(name, [], exit_code=126)
    assert not launch_capture.requests
    assert not launch_capture.commands


@pytest.mark.parametrize("name", HARNESSES)
def test_missing_binary_does_not_read_settings(
    name: str, launch_capture: LaunchCapture, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch.object(common.shutil, "which", return_value=None),
        patch.object(runner, "get_settings") as settings,
    ):
        launch(name, [], exit_code=127)
    settings.assert_not_called()
    assert not launch_capture.requests
    assert "Install" in capsys.readouterr().err


@pytest.mark.parametrize("name", HARNESSES)
def test_settings_are_loaded_once_for_a_launch(
    name: str, launch_capture: LaunchCapture
) -> None:
    with patch.object(runner, "get_settings", wraps=runner.get_settings) as settings:
        launch(name, [])
    settings.assert_called_once_with()


def test_empty_token_stops_before_network(launch_capture: LaunchCapture) -> None:
    settings = runner.get_settings().model_copy(update={"proxy_auth_token": "  "})
    with patch.object(runner, "get_settings", return_value=settings):
        launch("claude", [], exit_code=1)
    assert not launch_capture.requests


@pytest.mark.parametrize(
    "name", tuple(name for name in HARNESSES if name not in {"claude", "pi"})
)
@pytest.mark.parametrize(
    "payload", ({}, {"data": []}, {"data": [{"id": "no provider reference"}]})
)
def test_unusable_catalog_prevents_every_catalog_dependent_child(
    name: str, payload, launch_capture: LaunchCapture
) -> None:
    launch_capture.catalog = payload
    launch(name, [], exit_code=1)
    assert not launch_capture.commands


def test_setup_error_reports_once_and_redacts_token(
    launch_capture: LaunchCapture, capsys: pytest.CaptureFixture[str]
) -> None:
    launch_capture.catalog_error = URLError("rejected launcher-test-token")
    launch("codex", [], exit_code=1)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.count("Could not prepare model catalog") == 1
    assert "launcher-test-token" not in output.err
    assert "rejected [redacted]" in output.err


def test_programming_errors_are_not_disguised_as_setup_failures(
    launch_capture: LaunchCapture,
) -> None:
    launch_capture.catalog_error = AssertionError("programming error")
    with pytest.raises(AssertionError, match="programming error"):
        importlib.import_module("free_claude_code.cli.launchers.codex").launch([])

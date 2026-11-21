"""Native command ownership and retirement of previous Antigravity connections."""

import json
import os
from pathlib import Path

import pytest

from free_claude_code.harnesses import antigravity
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch
from tests.harnesses.antigravity_support import write_previous_connection


def test_default_model_and_credentials_keep_native_home(
    launch_capture: LaunchCapture, monkeypatch: pytest.MonkeyPatch
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
    command = launch_capture.commands[0]
    assert command[:2] == ["agy", "--settings"]
    assert command[3:] == [
        "--model",
        "nvidia_nim/catalog-model:variant",
        "--print",
        "hello",
    ]
    env = launch_capture.environments[0]
    assert env["GEMINI_API_KEY"] == "launcher-test-token"
    assert env["GOOGLE_GEMINI_BASE_URL"] == "http://127.0.0.1:8182"
    assert env["HOME"] == parent["HOME"] and env["USERPROFILE"] == parent["USERPROFILE"]
    assert dict(os.environ) == parent
    assert (
        not antigravity.config_path().exists() and not antigravity.state_path().exists()
    )


@pytest.mark.parametrize(
    "args",
    [
        ["models"],
        ["mcp", "list"],
        ["plugin", "list"],
        ["help", "models"],
        ["--conversation", "native-id", "--model", "provider/explicit:free"],
        ["--continue"],
        ["--print", "literal --settings={} and --disconnect"],
    ],
)
def test_native_arguments_follow_global_settings(
    args: list[str], launch_capture: LaunchCapture
):
    launch("agy", args)
    command = launch_capture.commands[0]
    assert command[:2] == ["agy", "--settings"]
    assert command[3:] == (
        args
        if not args[0].startswith("-")
        else ["--model", "nvidia_nim/catalog-model:variant", *args]
    )
    assert "--dangerously-skip-permissions" not in command


def test_old_connection_is_retired_before_native_start(launch_capture: LaunchCapture):
    path, state = antigravity.config_path(), antigravity.state_path()
    original = write_previous_connection(path, state)

    def inspect(command: object, env: object):
        assert json.loads(path.read_text()) == original
        assert not state.exists()

    launch_capture.on_start = inspect
    launch("agy", [])
    before = path.read_bytes()
    launch("agy", [])
    assert path.read_bytes() == before


def test_catalog_failure_does_not_retire_connection(launch_capture: LaunchCapture):
    path, state = antigravity.config_path(), antigravity.state_path()
    write_previous_connection(path, state)
    before = path.read_bytes(), state.read_bytes()
    launch_capture.catalog_error = ValueError("catalog unavailable")
    launch("agy", [], exit_code=1)
    assert (path.read_bytes(), state.read_bytes()) == before
    assert not launch_capture.commands


def test_cleanup_failure_stops_launch_and_keeps_record(
    launch_capture: LaunchCapture, capsys: pytest.CaptureFixture[str]
):
    path, state = antigravity.config_path(), antigravity.state_path()
    write_previous_connection(path, state)
    path.write_text("{broken")
    before = path.read_bytes(), state.read_bytes()
    launch("agy", [], exit_code=1)
    assert not launch_capture.commands
    assert (path.read_bytes(), state.read_bytes()) == before
    assert "/admin/integrations" not in capsys.readouterr().err


def test_native_start_failure_removes_temporary_settings(launch_capture: LaunchCapture):
    files = []

    def fail(command: list[str], env: object):
        path = Path(command[command.index("--settings") + 1])
        assert path.exists()
        files.append(path)
        raise PermissionError("cannot start executable")

    launch_capture.on_start = fail
    launch("agy", [], exit_code=1)
    assert files and all(not path.exists() for path in files)


@pytest.mark.parametrize("output", ["1.4.7", "1.5.8", "1.5.9-beta.1", "unknown"])
def test_unsupported_native_version_stops_before_fcc(
    output: str, launch_capture: LaunchCapture
):
    launch_capture.versions["agy"] = output
    launch("agy", [], exit_code=126)
    assert not launch_capture.requests and not launch_capture.commands

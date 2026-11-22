"""Claude owns permission options and prompt interpretation."""

import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses.claude import claude_proxy_values
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch
from tests.wire_payload import decode_object, object_payload

_FCC_ENV = claude_proxy_values("http://127.0.0.1:8182", "launcher-test-token")


def _capture_settings(launch_capture: LaunchCapture) -> list[JsonObject]:
    """Record each settings file while Claude would be reading it."""
    captured: list[JsonObject] = []

    def inspect(command: list[str], env: Mapping[str, str]) -> None:
        assert command[1] == "--settings"
        path = Path(command[2])
        captured.append(decode_object(path.read_text(encoding="utf-8")))

    launch_capture.on_start = inspect
    return captured


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--permission-mode", "auto", "fix tests"],
        ["--permission-mode=acceptEdits", "fix tests"],
        ["--dangerously-skip-permissions", "fix tests"],
        ["--permission-mode"],
        ["--", "--permission-mode=auto"],
        ["explain --permission-mode auto"],
    ],
)
def test_claude_owns_permission_selection(
    args: list[str], launch_capture: LaunchCapture
) -> None:
    launch("claude", args)
    [command] = launch_capture.commands
    assert command[:2] == ["claude", "--settings"]
    assert command[3:] == args
    env = launch_capture.environments[0]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8182"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "launcher-test-token"
    assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert env["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "1"
    assert env["CLAUDE_CODE_AUTO_MODE_SERVER"] == "0"


@pytest.mark.parametrize("retry_value", [None, "1"])
def test_claude_launch_extends_waiting_without_changing_retries(
    launch_capture: LaunchCapture,
    monkeypatch: pytest.MonkeyPatch,
    retry_value: str | None,
) -> None:
    expected = {
        "CLAUDE_ENABLE_STREAM_WATCHDOG": "0",
        "CLAUDE_ENABLE_BYTE_WATCHDOG": "0",
        "API_FORCE_IDLE_TIMEOUT": "0",
        "API_TIMEOUT_MS": "2147483647",
    }
    for name in expected:
        monkeypatch.setenv(name, "1")
    retry_keys = (
        "CLAUDE_CODE_MAX_RETRIES",
        "CLAUDE_CODE_RETRY_WATCHDOG",
        "CLAUDE_CODE_NONSTREAMING_TIMEOUT_RETRIES",
    )
    for name in retry_keys:
        if retry_value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, retry_value)
    monkeypatch.setenv("BASH_DEFAULT_TIMEOUT_MS", "120000")

    launch("claude", [])

    env = launch_capture.environments[0]
    assert {name: env[name] for name in expected} == expected
    assert env["BASH_DEFAULT_TIMEOUT_MS"] == "120000"
    for name in retry_keys:
        if retry_value is None:
            assert name not in env
        else:
            assert env[name] == retry_value
    for name in expected:
        assert os.environ[name] == "1"


def test_claude_settings_files_cannot_redirect_the_fcc_connection(
    launch_capture: LaunchCapture,
) -> None:
    # Claude Code applies settings-file env over the process environment, so a
    # stale ANTHROPIC_BASE_URL in ~/.claude/settings.json would bypass FCC.
    captured = _capture_settings(launch_capture)
    launch("claude", ["fix tests"])
    assert captured == [{"env": dict(_FCC_ENV)}]
    settings_path = Path(launch_capture.commands[0][2])
    assert not settings_path.exists()


@pytest.mark.parametrize("form", ["separate", "equals", "file"])
def test_claude_user_settings_flag_is_merged_under_fcc_connection(
    form: str, launch_capture: LaunchCapture, tmp_path: Path
) -> None:
    user_settings: JsonObject = {
        "model": "user-model",
        "env": {"ANTHROPIC_BASE_URL": "https://elsewhere.invalid", "KEEP": "1"},
    }
    value = json.dumps(user_settings)
    if form == "file":
        path = tmp_path / "user-settings.json"
        path.write_text(value, encoding="utf-8")
        value = str(path)
    flag = [f"--settings={value}"] if form == "equals" else ["--settings", value]
    captured = _capture_settings(launch_capture)

    launch("claude", ["--verbose", *flag, "fix tests"])

    [command] = launch_capture.commands
    assert command[3:] == ["--verbose", "fix tests"]
    [settings] = captured
    assert settings["model"] == "user-model"
    assert object_payload(settings["env"]) == {"KEEP": "1", **_FCC_ENV}


def test_claude_last_user_settings_flag_wins_like_claude(
    launch_capture: LaunchCapture,
) -> None:
    captured = _capture_settings(launch_capture)
    launch(
        "claude",
        ["--settings", '{"model": "first"}', "--settings", '{"model": "second"}'],
    )
    assert captured[0]["model"] == "second"
    assert launch_capture.commands[0][3:] == []


def test_claude_settings_after_separator_stays_prompt_text(
    launch_capture: LaunchCapture,
) -> None:
    captured = _capture_settings(launch_capture)
    args = ["--", "--settings", '{"model": "literal"}']
    launch("claude", args)
    assert launch_capture.commands[0][3:] == args
    assert captured == [{"env": dict(_FCC_ENV)}]


@pytest.mark.parametrize(
    "value", ["[1]", '{"env": ["x"]}', "missing-settings.json", "{not json"]
)
def test_claude_unreadable_user_settings_stop_launch(
    value: str, launch_capture: LaunchCapture
) -> None:
    launch("claude", ["--settings", value], exit_code=1)
    assert not launch_capture.commands

"""Claude owns permission options and prompt interpretation."""

import os

import pytest

from free_claude_code.harnesses.claude import CLAUDE_PROVIDER_SWITCHES
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


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
    assert launch_capture.commands == [["claude", *args]]
    env = launch_capture.environments[0]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8182"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "launcher-test-token"
    assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert env["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "1"
    assert env["CLAUDE_CODE_AUTO_MODE_SERVER"] == "0"
    # Settings-file provider, endpoint and auth values cannot bypass FCC.
    assert env["CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST"] == "1"


@pytest.mark.parametrize("name", CLAUDE_PROVIDER_SWITCHES)
def test_claude_launch_turns_off_inherited_provider_switches(
    name: str, launch_capture: LaunchCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Host-managed mode ignores settings that would turn these off again.
    monkeypatch.setenv(name, "1")
    launch("claude", [])
    assert launch_capture.environments[0][name] == "0"


def test_claude_launch_keeps_non_provider_use_preferences(
    launch_capture: LaunchCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_USE_POWERSHELL_TOOL", "1")
    launch("claude", [])
    assert launch_capture.environments[0]["CLAUDE_CODE_USE_POWERSHELL_TOOL"] == "1"


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

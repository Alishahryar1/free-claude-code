"""Claude owns permission options and prompt interpretation."""

import pytest

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


def test_claude_waits_for_fcc_recovery_and_stall_detection(
    launch_capture: LaunchCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited = {
        "API_TIMEOUT_MS": "1000",
        "API_FORCE_IDLE_TIMEOUT": "1",
        "CLAUDE_ENABLE_STREAM_WATCHDOG": "1",
        "CLAUDE_ENABLE_BYTE_WATCHDOG": "1",
        "CLAUDE_CODE_MAX_RETRIES": "5",
        "CLAUDE_CODE_NONSTREAMING_TIMEOUT_RETRIES": "3",
        "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "0",
        "CLAUDE_CODE_EFFORT_LEVEL": "high",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)

    launch("claude", [])

    env = launch_capture.environments[0]
    assert env["API_TIMEOUT_MS"] == "2147483647"
    assert env["API_FORCE_IDLE_TIMEOUT"] == "0"
    assert env["CLAUDE_ENABLE_STREAM_WATCHDOG"] == "0"
    assert env["CLAUDE_ENABLE_BYTE_WATCHDOG"] == "0"
    assert env["CLAUDE_CODE_MAX_RETRIES"] == "0"
    assert env["CLAUDE_CODE_NONSTREAMING_TIMEOUT_RETRIES"] == "0"
    assert env["CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"] == "1"
    assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "high"

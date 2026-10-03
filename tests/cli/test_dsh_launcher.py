"""DSH keeps its native profile grammar and receives one FCC patch."""

import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import pytest

from free_claude_code.cli.launchers import runner
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ["dsh", "web", "--patch", "<patch>"]),
        (
            ["web", "--port", "8000"],
            ["dsh", "web", "--patch", "<patch>", "--port", "8000"],
        ),
        (["headless", "hello"], ["dsh", "headless", "--patch", "<patch>", "hello"]),
        (
            ["--profile", "headless", "hello"],
            ["dsh", "--patch", "<patch>", "--profile", "headless", "hello"],
        ),
        (
            ["--profile=headless", "hello"],
            ["dsh", "--patch", "<patch>", "--profile=headless", "hello"],
        ),
        (
            ["--profile", "future-native-profile", "--patch", "user.patch.yml"],
            [
                "dsh",
                "--patch",
                "<patch>",
                "--profile",
                "future-native-profile",
                "--patch",
                "user.patch.yml",
            ],
        ),
        (
            ["--patch", "user.patch.yml", "--profile", "future-native-profile"],
            [
                "dsh",
                "--patch",
                "<patch>",
                "--patch",
                "user.patch.yml",
                "--profile",
                "future-native-profile",
            ],
        ),
        (
            ["--", "--profile", "app-value"],
            [
                "dsh",
                "--profile",
                "web",
                "--patch",
                "<patch>",
                "--",
                "--profile",
                "app-value",
            ],
        ),
        (
            ["--", "--help"],
            ["dsh", "--profile", "web", "--patch", "<patch>", "--", "--help"],
        ),
    ],
)
def test_dsh_attaches_patch_without_overriding_native_profile(
    args: list[str], expected: list[str], launch_capture: LaunchCapture
) -> None:
    def inspect(command, env):
        path = Path(command[command.index("--patch") + 1])
        patch = json.loads(path.read_text())
        serialized = json.dumps(patch)
        assert "openai-responses" in serialized
        assert "http://127.0.0.1:8182/v1" in serialized
        assert "nvidia_nim/catalog-model:variant" in serialized
        assert not (path.parent / "settings.yaml").exists()
        assert json.loads((path.parent / ".credentials.yaml").read_text()) == {}
        assert env["FCC_DSH_API_KEY"] == "launcher-test-token"
        assert env["DSH_TELEMETRY_DISABLED"] == "1"
        assert "launcher-test-token" not in serialized

    launch_capture.on_start = inspect
    launch("dsh", args)
    command = launch_capture.commands[0]
    command[command.index("--patch") + 1] = "<patch>"
    assert command == expected


@pytest.mark.parametrize(
    "args",
    [
        ["--port", "9000", "--profile", "app-value"],
        ["--future-app-option", "--profile=headless"],
        ["--", "--profile", "headless"],
    ],
)
def test_app_arguments_cannot_select_the_launch_profile(
    args: list[str], launch_capture: LaunchCapture
) -> None:
    launch("dsh", args)
    command = launch_capture.commands[0]
    prefix = command[: -len(args)]
    assert "web" in prefix
    assert prefix.count("--profile") == 1
    assert command[-len(args) :] == args


@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["-h"],
        ["--version"],
        ["-V"],
        ["plugin", "--profile", "web", "list"],
        ["web", "--dump-default-config"],
        ["--patch", "user.patch.yml", "--help"],
    ],
)
def test_control_commands_do_not_require_fcc(
    args: list[str], launch_capture: LaunchCapture
) -> None:
    launch_capture.health_error = URLError("FCC is stopped")
    with patch.object(runner, "get_settings") as settings:
        launch("dsh", args)
    settings.assert_not_called()
    assert not launch_capture.requests
    assert launch_capture.commands == [["dsh", *args]]
    assert "FCC_DSH_API_KEY" not in launch_capture.environments[0]


@pytest.mark.parametrize(
    "args",
    [
        ["web", "--help"],
        ["headless", "prompt", "--version"],
        ["--profile", "web", "--port", "9000", "--dump-default-config"],
        ["web", "--dump-config"],
        ["web", "--dump-config-schema"],
        ["--profile", "--version"],
        ["--patch", "--dump-default-config"],
    ],
)
def test_profile_operations_receive_fcc_configuration(
    args: list[str], launch_capture: LaunchCapture
) -> None:
    launch("dsh", args)
    assert len(launch_capture.requests) == 2
    assert "--patch" in launch_capture.commands[0]

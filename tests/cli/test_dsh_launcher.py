"""DSH keeps its native profile grammar and receives one FCC patch."""

import json
from pathlib import Path

import pytest

from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["web", "--port", "8000"],
        ["--profile", "headless", "--prompt", "hello"],
        ["--profile", "future-native-profile", "--patch", "user.patch.yml"],
        ["--", "--help"],
    ],
)
def test_dsh_attaches_patch_without_classifying_native_profiles(
    args: list[str], launch_capture: LaunchCapture
) -> None:
    def inspect(command, env):
        path = Path(command[command.index("--patch") + 1])
        patch = json.loads(path.read_text())
        serialized = json.dumps(patch)
        assert "openai-responses" in serialized
        assert "http://127.0.0.1:8182/v1" in serialized
        assert "nvidia_nim/catalog-model:variant" in serialized
        assert json.loads((path.parent / "settings.yaml").read_text()) == {}
        assert json.loads((path.parent / ".credentials.yaml").read_text()) == {}
        assert env["FCC_DSH_API_KEY"] == "launcher-test-token"
        assert env["DSH_TELEMETRY_DISABLED"] == "1"
        assert "launcher-test-token" not in serialized

    launch_capture.on_start = inspect
    launch("dsh", args)
    command = launch_capture.commands[0]
    if not args or args[0] == "web":
        assert command[:3] == ["dsh", "web", "--patch"]
        assert command[4:] == args[1:]
    else:
        assert command[:4] == ["dsh", "--profile", "web", "--patch"]
        assert command[5:] == args

"""Pi keeps its native extension and catalog ownership."""

from pathlib import Path
from unittest.mock import patch

from free_claude_code.cli.launchers import pi
from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


def test_pi_registers_bundled_extension_without_a_python_catalog(
    launch_capture: LaunchCapture,
) -> None:
    launch("pi", ["--model", "native-selection", "--print", "hello"])
    command = launch_capture.commands[0]
    assert command[:2] == ["pi", "-e"]
    assert Path(command[2]) == pi.pi_extension_path()
    assert command[3:5] == ["--models", "free-claude-code/**"]
    assert launch_capture.environments[0]["FCC_PI_API_KEY"] == "launcher-test-token"
    assert launch_capture.environments[0]["FCC_PI_BASE_URL"] == "http://127.0.0.1:8182"
    assert len(launch_capture.requests) == 1


def test_missing_bundled_extension_prevents_launch(
    launch_capture: LaunchCapture, tmp_path: Path
) -> None:
    with patch.object(pi, "pi_extension_path", return_value=tmp_path / "absent.ts"):
        launch("pi", [], exit_code=1)
    assert not launch_capture.commands


def test_pi_install_hints_use_official_platform_installers() -> None:
    assert "https://pi.dev/install.ps1" in pi.pi_install_hint("win32")
    assert "https://pi.dev/install.sh" in pi.pi_install_hint("linux")

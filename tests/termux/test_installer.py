import os
import subprocess
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[2] / "scripts" / "install-termux.sh"


def test_termux_installer_passes_posix_shell_syntax_check():
    result = subprocess.run(
        ["sh", "-n", str(INSTALLER)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_termux_installer_help_is_available_without_running_installs():
    result = subprocess.run(
        ["sh", str(INSTALLER), "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "private Termux Python environment" in result.stdout
    assert "FCC configuration" in result.stdout


def test_termux_installer_rejects_non_termux_before_install_actions():
    result = subprocess.run(
        ["sh", str(INSTALLER)],
        capture_output=True,
        check=False,
        text=True,
        env={"HOME": os.environ.get("HOME", "/tmp"), "PATH": os.environ["PATH"]},
    )

    assert result.returncode != 0
    assert "must run inside the Termux app" in result.stderr

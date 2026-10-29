"""Fork installers must install and update from the fork source."""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts/install.sh"


def shell_dry_run(*, version: str | None = None) -> str:
    environment = os.environ.copy()
    if version is not None:
        environment["FCC_INSTALL_VERSION"] = version
    result = subprocess.run(
        ["sh", str(INSTALLER), "--dry-run"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_shell_installer_uses_fork_main_by_default():
    output = shell_dry_run()

    assert (
        "free-claude-code @ "
        "git+https://github.com/mridwanfirdaus2603/free-claude-code.git@main"
    ) in output
    assert "free-claude-code==<selected-release>" not in output


def test_shell_installer_version_pin_uses_fork_tag():
    output = shell_dry_run(version="1.2.3")

    assert (
        "free-claude-code @ "
        "git+https://github.com/mridwanfirdaus2603/free-claude-code.git@v1.2.3"
    ) in output


def test_update_launchers_install_from_fork_main():
    unix_updater = (ROOT / "scripts/update/fcc-update").read_text()
    windows_updater = (ROOT / "scripts/update/fcc-update.cmd").read_text()

    for updater in (unix_updater, windows_updater):
        assert (
            "raw.githubusercontent.com/"
            "mridwanfirdaus2603/free-claude-code/main/scripts/install"
        ) in updater
        assert "Alishahryar1/free-claude-code" not in updater
        assert "UV_LINK_MODE=copy" in updater or "UV_LINK_MODE = 'copy'" in updater


def test_powershell_installer_uses_fork_source():
    installer = (ROOT / "scripts/install.ps1").read_text()

    assert '$FccSourceRepository = "mridwanfirdaus2603/free-claude-code"' in installer
    assert "git+https://github.com/${FccSourceRepository}.git@${FccSourceRef}" in installer

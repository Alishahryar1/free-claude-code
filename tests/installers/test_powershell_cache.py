"""Installer subprocesses disable disk cache writes without disabling discovery."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell cache behavior")
@pytest.mark.parametrize("shell_name", ["powershell", "pwsh"])
def test_powershell_cache_is_disabled_in_children(
    tmp_path: Path, shell_name: str, powershell_module_paths
) -> None:
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is not installed")
    assert os.environ["PSMODULEANALYSISCACHEPATH"] == "NUL"
    child = tmp_path / "child.ps1"
    child.write_text(
        'if ($env:PSModuleAnalysisCachePath -ne "NUL") { throw "Cache enabled" }\n'
        "Get-Command Get-Item -ErrorAction Stop | Out-Null\n"
        "Get-Command fcc_nonexistent_cache_probe -ErrorAction SilentlyContinue | Out-Null\n"
        'Write-Output "discovery complete"\n'
        "exit 0\n"
    )
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "& $env:FCC_TEST_SHELL -NoProfile -NonInteractive -File $env:FCC_TEST_CHILD; exit $LASTEXITCODE",
        ],
        cwd=tmp_path,
        env=os.environ
        | {
            "PSMODULEPATH": powershell_module_paths[shell],
            "FCC_TEST_SHELL": shell,
            "FCC_TEST_CHILD": str(child),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "discovery complete"

"""Keep PowerShell's background module cache inside each test's storage."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell cache behavior")
@pytest.mark.parametrize("shell_name", ["powershell", "pwsh"])
def test_powershell_module_cache_does_not_leak_into_working_directory(
    tmp_path: Path, shell_name: str
) -> None:
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is not installed")

    working_directory = tmp_path / "working-directory"
    profile = tmp_path / "profile"
    local_app_data = tmp_path / "local-app-data"
    for directory in (working_directory, profile, local_app_data):
        directory.mkdir()

    cache_value = os.environ.get("PSMODULEANALYSISCACHEPATH")
    if cache_value:
        assert Path(cache_value).is_absolute()
        assert Path(cache_value).is_relative_to(tmp_path)
    leaked_cache = (
        working_directory
        / "Microsoft"
        / "Windows"
        / "PowerShell"
        / "ModuleAnalysisCache"
    )
    watched_cache = Path(cache_value) if cache_value else leaked_cache
    result = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            """$ErrorActionPreference = 'Stop'
Get-Command fcc_nonexistent_cache_probe_command -ErrorAction SilentlyContinue
$deadline = [DateTime]::UtcNow.AddSeconds(20)
while (-not (Test-Path -LiteralPath $env:FCC_TEST_WATCHED_CACHE) -and [DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Milliseconds 100
}
""",
        ],
        cwd=working_directory,
        env=os.environ
        | {
            "USERPROFILE": str(profile),
            "LOCALAPPDATA": str(local_app_data),
            "FCC_TEST_WATCHED_CACHE": str(watched_cache),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert not (working_directory / "Microsoft").exists(), "PowerShell leaked its cache"
    assert cache_value, "PowerShell test processes need an isolated cache path"
    assert watched_cache.is_file(), "PowerShell did not exercise its cache writer"
    assert watched_cache.stat().st_size > 0

"""Isolate caches created by installer and lifecycle test subprocesses."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_powershell_module_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A redirected USERPROFILE can make Windows PowerShell's default cache path
    # relative to the checkout, even when LOCALAPPDATA points at a temporary dir.
    monkeypatch.setenv(
        "PSModuleAnalysisCachePath", str(tmp_path / "powershell-module-cache")
    )


@pytest.fixture(scope="session")
def powershell_module_paths(tmp_path_factory):
    if os.name != "nt":
        return {}
    root = tmp_path_factory.mktemp("powershell-modules")
    paths = {}
    for shell in dict.fromkeys(
        path for name in ("pwsh", "powershell") if (path := shutil.which(name))
    ):
        path = root / Path(shell).stem
        # A private path avoids Windows PowerShell adding machine-wide module
        # directories back when it recognizes its default system module path.
        # Junctions keep the bundled modules intact without copying their DLLs.
        subprocess.run(
            [
                shell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "New-Item -ItemType Junction -Path $env:FCC_TEST_MODULE_PATH "
                "-Target (Join-Path $PSHOME 'Modules') | Out-Null",
            ],
            env=os.environ | {"FCC_TEST_MODULE_PATH": str(path)},
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        paths[shell] = str(path)
    return paths

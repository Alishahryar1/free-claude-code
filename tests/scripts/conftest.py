"""Isolate caches created by installer and lifecycle test subprocesses."""

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

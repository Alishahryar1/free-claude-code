"""Keep caches from CI-script subprocess tests outside the checkout."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_powershell_module_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(
        "PSModuleAnalysisCachePath", str(tmp_path / "powershell-module-cache")
    )

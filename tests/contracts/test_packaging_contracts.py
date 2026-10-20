import re
import subprocess
import sys
from pathlib import Path


def test_pyproject_first_party_packages_match_packaged_roots() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"known-first-party = \[(?P<items>[^\]]+)\]", pyproject)

    assert match is not None
    configured = {
        item.strip().strip('"')
        for item in match.group("items").split(",")
        if item.strip()
    }
    expected = {"free_claude_code", "smoke"}
    assert configured == expected


def test_standard_install_can_load_nim_transcription_client() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from riva.client import Auth, ASRService, RecognitionConfig; "
            "RecognitionConfig(language_code='en-US', max_alternatives=1)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

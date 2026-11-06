import os
import stat
from pathlib import Path

import pytest

from free_claude_code.harnesses.config_file import atomic_write_text


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission modes are unavailable")
def test_private_write_does_not_restore_a_public_mode(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o644)

    atomic_write_text(path, "new", private=True)

    assert path.read_text(encoding="utf-8") == "new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "nt", reason="Exercises Windows replacement retry")
def test_failed_windows_replacements_preserve_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text("original", encoding="utf-8")

    def denied_replace(self, target):
        raise PermissionError("replacement denied")

    monkeypatch.setattr(Path, "replace", denied_replace)

    with pytest.raises(PermissionError, match="replacement denied"):
        atomic_write_text(path, "replacement")

    assert path.read_text(encoding="utf-8") == "original"


@pytest.mark.skipif(os.name != "nt", reason="Exercises Windows replacement retry")
def test_read_only_windows_file_is_replaced_and_remains_read_only(
    tmp_path, monkeypatch
):
    path = tmp_path / "settings.json"
    path.write_text("original", encoding="utf-8")
    path.chmod(stat.S_IREAD)
    assert not path.stat().st_mode & stat.S_IWRITE

    original_replace = Path.replace
    attempts = 0

    def fail_first_replace(self, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("target is read-only")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_first_replace)

    atomic_write_text(path, "replacement")

    assert attempts == 2
    assert path.read_text(encoding="utf-8") == "replacement"
    assert not path.stat().st_mode & stat.S_IWRITE

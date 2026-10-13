import json
from pathlib import Path

import pytest

from free_claude_code.harnesses import claude_onboarding


def test_missing_state_status_does_not_create_files(tmp_path):
    path = tmp_path / "missing" / ".claude.json"
    assert claude_onboarding.status(path) == {
        "completed": False,
        "paths": {"claude_state": str(path.resolve())},
    }
    assert not path.parent.exists()
    assert claude_onboarding.complete(path) == {"completed": True, "changed": True}
    assert json.loads(path.read_text()) == {"hasCompletedOnboarding": True}


@pytest.mark.parametrize("flag", [None, False, 1, "true", True])
def test_complete_preserves_values_and_is_idempotent(tmp_path, flag):
    path = tmp_path / ".claude.json"
    path.write_text(
        json.dumps(
            {"theme": "dark", "keep": "\U0001f600", "hasCompletedOnboarding": flag}
        )
    )
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert claude_onboarding.status(path)["completed"] is (flag is True)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert claude_onboarding.complete(path) == {
        "completed": True,
        "changed": flag is not True,
    }
    assert json.loads(path.read_text()) == {
        "theme": "dark",
        "keep": "\U0001f600",
        "hasCompletedOnboarding": True,
    }
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert claude_onboarding.complete(path) == {"completed": True, "changed": False}
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("source", ["{invalid", "[]", '{"a":1,"a":2}', '{"a":NaN}'])
def test_invalid_state_is_never_overwritten(tmp_path, source):
    path = tmp_path / ".claude.json"
    path.write_text(source)
    for operation in (claude_onboarding.status, claude_onboarding.complete):
        with pytest.raises(ValueError):
            operation(path)
        assert path.read_text() == source


def test_failed_write_preserves_original_and_removes_temporary(tmp_path, monkeypatch):
    path = tmp_path / ".claude.json"
    path.write_text('{"theme":"dark"}')

    def fail(self, target):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(PermissionError):
        claude_onboarding.complete(path)
    assert path.read_text() == '{"theme":"dark"}'
    assert list(tmp_path.iterdir()) == [path]


def test_symlink_is_preserved(tmp_path):
    target, link = tmp_path / "state.json", tmp_path / ".claude.json"
    target.write_text('{"theme":"dark"}')
    try:
        link.symlink_to(target.name)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or privilege")
        raise
    assert claude_onboarding.status(link)["paths"] == {
        "claude_state": str(target.resolve())
    }
    assert claude_onboarding.complete(link) == {"completed": True, "changed": True}
    assert link.is_symlink()
    assert json.loads(target.read_text()) == {
        "theme": "dark",
        "hasCompletedOnboarding": True,
    }


def test_default_state_path(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert claude_onboarding.state_path() == tmp_path / ".claude.json"

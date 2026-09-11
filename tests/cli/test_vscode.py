import json
from pathlib import Path

import pytest

from free_claude_code.cli import vscode

URL = "http://127.0.0.1:8000"
TOKEN = "test-integration-token"
ENV = "claudeCode.environmentVariables"
LOGIN = "claudeCode.disableLoginPrompt"


def operate(path, connected=None):
    return vscode.configure(path, URL, TOKEN, connected)


def test_connect_merges_jsonc_and_disconnect_preserves_unrelated_values(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        """{
      // User accepts losing this comment.
      "editor.fontSize": 16,
      "claudeCode.environmentVariables": [
        {"name": "KEEP", "value": "yes"},
        {"name": "ANTHROPIC_AUTH_TOKEN", "value": "old", "extra": "keep"},
      ],
    }""",
        encoding="utf-8",
    )
    assert operate(path, True) == {"connected": True}
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["editor.fontSize"] == 16
    entries = {entry["name"]: entry for entry in saved[ENV]}
    assert set(entries) == {
        "KEEP",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
        "DISABLE_AUTOUPDATER",
        "DISABLE_FEEDBACK_COMMAND",
        "DISABLE_ERROR_REPORTING",
    }
    assert entries["ANTHROPIC_AUTH_TOKEN"] == {
        "name": "ANTHROPIC_AUTH_TOKEN",
        "value": TOKEN,
        "extra": "keep",
    }
    assert entries["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]["value"] == "190000"
    assert "//" not in path.read_text().replace("http://", "")
    assert operate(path, False) == {"connected": False}
    assert json.loads(path.read_text()) == {
        "editor.fontSize": 16,
        ENV: [{"name": "KEEP", "value": "yes"}],
    }


def test_missing_file_and_idempotent_operations(tmp_path):
    path = tmp_path / "Code" / "User" / "settings.json"
    assert operate(path) == {"connected": False}
    assert operate(path, False) == {"connected": False}
    assert not path.parent.exists()
    operate(path, True)
    before = path.stat().st_mtime_ns
    operate(path, True)
    assert path.stat().st_mtime_ns == before
    operate(path, False)
    assert json.loads(path.read_text()) == {}
    before = path.stat().st_mtime_ns
    operate(path, False)
    assert path.stat().st_mtime_ns == before


@pytest.mark.parametrize("url", ["http://localhost:8000/", URL, "http://[::1]:8000"])
def test_manual_setup_only_requires_connection_fields(tmp_path, url):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                LOGIN: True,
                ENV: [
                    {"name": "ANTHROPIC_BASE_URL", "value": url},
                    {"name": "ANTHROPIC_AUTH_TOKEN", "value": TOKEN},
                    {
                        "name": "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY",
                        "value": "1",
                    },
                ],
            }
        )
    )
    assert operate(path) == {"connected": True}


@pytest.mark.parametrize(
    "field,value",
    [
        ("ANTHROPIC_BASE_URL", "http://localhost:9000"),
        ("ANTHROPIC_BASE_URL", "http://localhost:bad"),
        ("ANTHROPIC_BASE_URL", "http://localhost:8000/?other=1"),
        ("ANTHROPIC_BASE_URL", "http://user@localhost:8000"),
        ("ANTHROPIC_AUTH_TOKEN", "other"),
        ("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "0"),
    ],
)
def test_partial_or_different_setup_is_disconnected(tmp_path, field, value):
    path = tmp_path / "settings.json"
    operate(path, True)
    data = json.loads(path.read_text())
    next(entry for entry in data[ENV] if entry["name"] == field)["value"] = value
    path.write_text(json.dumps(data))
    assert operate(path) == {"connected": False}
    operate(path, False)
    assert json.loads(path.read_text()) == {}


@pytest.mark.parametrize(
    "source",
    [
        "{invalid",
        "[]",
        '{"a": 1, "a": 2}',
        '{"x": NaN}',
        '{"claudeCode.environmentVariables": {}}',
        '{"claudeCode.environmentVariables": [null]}',
        '{"claudeCode.environmentVariables": [{"name": "X", "value": 3}]}',
        '{"claudeCode.environmentVariables": ['
        '{"name": "ANTHROPIC_AUTH_TOKEN", "value": "a"},'
        '{"name": "ANTHROPIC_AUTH_TOKEN", "value": "b"}]}',
    ],
)
def test_invalid_settings_are_never_overwritten(tmp_path, source):
    path = tmp_path / "settings.json"
    path.write_text(source)
    for action in (None, True, False):
        with pytest.raises(ValueError):
            operate(path, action)
        assert path.read_text() == source


@pytest.mark.parametrize("readonly", [False, True])
def test_failed_replace_preserves_original_and_removes_tempfile(
    tmp_path, monkeypatch, readonly
):
    path = tmp_path / "settings.json"
    path.write_text('{"keep": true}')
    if readonly:
        path.chmod(0o444)

    def fail_replace(self, target):
        raise PermissionError("test write failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    try:
        with pytest.raises(OSError):
            operate(path, True)
        assert path.read_text() == '{"keep": true}'
        assert list(tmp_path.iterdir()) == [path]
    finally:
        for item in tmp_path.iterdir():
            item.chmod(0o600)


@pytest.mark.parametrize(
    "platform,suffix",
    [
        ("win32", "AppData/Roaming/Code/User/settings.json"),
        ("darwin", "Library/Application Support/Code/User/settings.json"),
        ("linux", ".config/Code/User/settings.json"),
    ],
)
def test_standard_global_path(tmp_path, monkeypatch, platform, suffix):
    monkeypatch.setattr(vscode.sys, "platform", platform)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert vscode.settings_path() == tmp_path / suffix


@pytest.mark.parametrize(
    "platform,variable", [("win32", "APPDATA"), ("linux", "XDG_CONFIG_HOME")]
)
def test_config_root_override(tmp_path, monkeypatch, platform, variable):
    monkeypatch.setattr(vscode.sys, "platform", platform)
    monkeypatch.setenv(variable, str(tmp_path))
    assert vscode.settings_path() == tmp_path / "Code/User/settings.json"

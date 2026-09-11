import json
from pathlib import Path

import pytest

from free_claude_code.cli import vscode

URL = "http://127.0.0.1:8000"
TOKEN = "test-integration-token"
ENV = "claudeCode.environmentVariables"
LOGIN = "claudeCode.disableLoginPrompt"


def operate(path, connected=None):
    return vscode.configure(path, path.parent / ".claude.json", URL, TOKEN, connected)


def test_connect_completes_onboarding_and_disconnect_preserves_state(tmp_path):
    path = tmp_path / "settings.json"
    state_path = tmp_path / ".claude.json"
    state_path.write_text('{"theme":"dark","hasCompletedOnboarding":false}')
    assert operate(path, True) == {"connected": True}
    assert json.loads(state_path.read_text()) == {
        "theme": "dark",
        "hasCompletedOnboarding": True,
    }
    saved = state_path.read_bytes()
    modified = state_path.stat().st_mtime_ns
    operate(path, True)
    assert state_path.stat().st_mtime_ns == modified
    assert operate(path, False) == {"connected": False}
    assert state_path.read_bytes() == saved


@pytest.mark.parametrize("flag", [None, False, 1, "true"])
def test_connected_requires_completed_onboarding(tmp_path, flag):
    path = tmp_path / "settings.json"
    state_path = tmp_path / ".claude.json"
    operate(path, True)
    state_path.write_text(json.dumps({"hasCompletedOnboarding": flag}))
    assert operate(path) == {"connected": False}
    operate(path, True)
    assert json.loads(state_path.read_text())["hasCompletedOnboarding"] is True
    assert operate(path) == {"connected": True}


def test_missing_onboarding_state_is_read_only_until_connect(tmp_path):
    path = tmp_path / "settings.json"
    state_path = tmp_path / ".claude.json"
    operate(path, True)
    state_path.unlink()
    assert operate(path) == {"connected": False}
    assert not state_path.exists()
    assert operate(path, False) == {"connected": False}
    assert not state_path.exists()
    operate(path, True)
    assert json.loads(state_path.read_text()) == {"hasCompletedOnboarding": True}


@pytest.mark.parametrize("source", ["{invalid", "[]", '{"a":1,"a":2}', '{"a":NaN}'])
def test_invalid_state_blocks_connect_but_not_disconnect(tmp_path, source):
    path = tmp_path / "settings.json"
    state_path = tmp_path / ".claude.json"
    operate(path, True)
    settings_before = path.read_bytes()
    state_path.write_text(source)
    for action in (None, True):
        with pytest.raises(ValueError):
            operate(path, action)
        assert path.read_bytes() == settings_before
        assert state_path.read_text() == source
    assert operate(path, False) == {"connected": False}
    assert json.loads(path.read_text()) == {}
    assert state_path.read_text() == source


def test_invalid_vscode_settings_do_not_update_onboarding(tmp_path):
    path = tmp_path / "settings.json"
    state_path = tmp_path / ".claude.json"
    path.write_text('{"claudeCode.environmentVariables":{}}')
    state_path.write_text('{"hasCompletedOnboarding":false}')
    with pytest.raises(ValueError):
        operate(path, True)
    assert state_path.read_text() == '{"hasCompletedOnboarding":false}'


@pytest.mark.parametrize("failed_file", ["state", "settings"])
def test_save_failure_preserves_completed_steps_and_retry_completes(
    tmp_path, monkeypatch, failed_file
):
    path = tmp_path / "settings.json"
    state_path = tmp_path / ".claude.json"
    path.write_text('{"keep":true}')
    original_replace = Path.replace

    def fail_replace(self, target):
        if target == (state_path if failed_file == "state" else path):
            raise PermissionError("test failure")
        return original_replace(self, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError):
            operate(path, True)
    if failed_file == "state":
        assert not state_path.exists()
    else:
        assert json.loads(state_path.read_text())["hasCompletedOnboarding"] is True
    assert path.read_text() == '{"keep":true}'
    assert operate(path) == {"connected": False}
    assert operate(path, True) == {"connected": True}


def test_onboarding_symlink_and_escaped_unicode_are_preserved(tmp_path):
    path = tmp_path / "settings.json"
    state_path = tmp_path / ".claude.json"
    target = tmp_path / "state.json"
    target.write_text(json.dumps({"keep": "\U0001f600"}))
    try:
        state_path.symlink_to(Path("state.json"))
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or privilege")
        raise
    assert operate(path, True) == {"connected": True}
    assert state_path.is_symlink()
    assert json.loads(target.read_text()) == {
        "keep": "\U0001f600",
        "hasCompletedOnboarding": True,
    }


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


@pytest.mark.parametrize("connected", [True, False])
def test_escaped_unicode_survives_connect_and_disconnect(tmp_path, connected):
    path = tmp_path / "settings.json"
    data = {"keep": "\U0001f600"}
    if not connected:
        operate(path, True)
        data.update(json.loads(path.read_text(encoding="utf-8")))
    path.write_text(json.dumps(data), encoding="utf-8")
    assert operate(path, connected) == {"connected": connected}
    assert json.loads(path.read_text(encoding="utf-8"))["keep"] == "\U0001f600"


@pytest.mark.parametrize("connected", [True, False])
@pytest.mark.parametrize("target_exists", [True, False])
def test_settings_symlink_is_preserved(tmp_path, connected, target_exists):
    target = tmp_path / "dotfiles" / "settings.json"
    link = tmp_path / "settings.json"
    if target_exists:
        target.parent.mkdir()
        target.write_text('{"keep": true}')
        if not connected:
            operate(target, True)
    try:
        link.symlink_to(Path("dotfiles/settings.json"))
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or privilege")
        raise
    assert operate(link, connected) == {"connected": connected}
    assert link.is_symlink()
    if connected or target_exists:
        assert target.is_file()
        saved = json.loads(target.read_text())
        assert saved.get(LOGIN, False) is connected
        if target_exists:
            assert saved["keep"] is True
    else:
        assert not target.exists()


@pytest.mark.parametrize("url", ["http://localhost:8000/", URL, "http://[::1]:8000"])
def test_manual_setup_only_requires_connection_fields(tmp_path, url):
    path = tmp_path / "settings.json"
    (tmp_path / ".claude.json").write_text('{"hasCompletedOnboarding":true}')
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
    state_path = tmp_path / ".claude.json"
    state_path.write_text('{"hasCompletedOnboarding":true}')
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
        assert set(tmp_path.iterdir()) == {path, state_path}
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

import tomllib
from pathlib import Path

import pytest

from free_claude_code.cli import codex_integration

URL = "http://127.0.0.1:8082"


def operate(path, connected=None):
    return codex_integration.configure(
        path, path.parent / "catalog.json", URL, connected
    )


def test_connect_disconnect_preserve_model_comments_and_other_providers(tmp_path):
    path = tmp_path / "config.toml"
    original = '# My preferences\nmodel = "my-choice" # Keep this\n\n[model_providers.other]\nname = "Other"\n'
    path.write_text(original)
    assert operate(path, True)["connected"] is True
    saved = path.read_text()
    assert '# My preferences\nmodel = "my-choice" # Keep this' in saved
    data = tomllib.loads(saved)
    assert data["model"] == "my-choice"
    assert data["model_provider"] == "fcc"
    assert Path(data["model_catalog_json"]) == tmp_path / "catalog.json"
    assert data["model_providers"]["fcc"] == {
        "name": "Free Claude Code",
        "base_url": URL + "/v1",
        "wire_api": "responses",
        "auth": {"command": "fcc-codex", "args": ["--print-proxy-auth-token"]},
    }
    assert not (tmp_path / "catalog.json").exists()
    assert operate(path, False)["connected"] is False
    assert tomllib.loads(path.read_text()) == {
        "model": "my-choice",
        "model_providers": {"other": {"name": "Other"}},
    }
    assert '# My preferences\nmodel = "my-choice" # Keep this' in path.read_text()


def test_missing_file_noops_and_unset_model(tmp_path):
    path = tmp_path / "codex" / "config.toml"
    assert operate(path)["connected"] is False
    assert operate(path, False)["connected"] is False
    assert not path.parent.exists()
    result = operate(path, True)
    assert result == {"connected": True, "paths": {"codex_config": str(path.resolve())}}
    assert "model" not in tomllib.loads(path.read_text())
    before = path.stat().st_mtime_ns
    operate(path, True)
    assert path.stat().st_mtime_ns == before
    operate(path, False)
    before = path.stat().st_mtime_ns
    operate(path, False)
    assert path.stat().st_mtime_ns == before


def test_inline_tables_merge_without_losing_extra_values(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'model_providers = {fcc = {name = "Old", extra = 7, auth = {command = "old", keep = true}}, other = {name = "Other"}} # Keep comment\n'
    )
    operate(path, True)
    data = tomllib.loads(path.read_text())
    assert data["model_providers"]["fcc"]["extra"] == 7
    assert data["model_providers"]["fcc"]["auth"]["keep"] is True
    assert "# Keep comment" in path.read_text()
    assert operate(path)["connected"] is True
    operate(path, False)
    assert tomllib.loads(path.read_text())["model_providers"] == {
        "other": {"name": "Other"}
    }


def test_manual_connection_ignores_model_and_display_name(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'model = "another-model"\nmodel_provider = "fcc"\n'
        f'model_catalog_json = "{(tmp_path / "catalog.json").as_posix()}"\n'
        '[model_providers.fcc]\nname = "My FCC"\nbase_url = "http://localhost:8082/v1/"\nwire_api = "responses"\n'
        '[model_providers.fcc.auth]\ncommand = "fcc-codex"\nargs = ["--print-proxy-auth-token"]\n'
    )
    before = path.read_bytes()
    assert operate(path)["connected"] is True
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "content",
    [
        "model_providers = {}\n",
        'model_providers = {fcc = {name = "Old"}}\n',
        '[model_providers.fcc]\nname = "Old"\n[other]\nkeep = true\n[model_providers.other]\nname = "Other"\n[model_providers.fcc.auth]\ncommand = "old"\n',
    ],
)
def test_valid_table_layouts_can_be_updated(tmp_path, content):
    path = tmp_path / "config.toml"
    path.write_text(content)
    original = tomllib.loads(content)
    assert operate(path, True)["connected"] is True
    assert tomllib.loads(path.read_text())["model_providers"]["fcc"]["auth"][
        "args"
    ] == ["--print-proxy-auth-token"]
    assert operate(path, False)["connected"] is False
    if "other" in original:
        result = tomllib.loads(path.read_text())
        assert result["other"] == original["other"]
        assert (
            result["model_providers"]["other"] == original["model_providers"]["other"]
        )


@pytest.mark.parametrize(
    "old,new",
    [
        ('model_provider = "fcc"', 'model_provider = "other"'),
        ('wire_api = "responses"', 'wire_api = "chat"'),
        (
            'base_url = "http://127.0.0.1:8082/v1"',
            'base_url = "http://127.0.0.1:8083/v1"',
        ),
        ('command = "fcc-codex"', 'command = "other"'),
        ("--print-proxy-auth-token", "--other"),
        ("catalog.json", "another-catalog.json"),
    ],
)
def test_changed_connection_fields_are_detected(tmp_path, old, new):
    path = tmp_path / "config.toml"
    operate(path, True)
    content = path.read_text()
    assert old in content
    path.write_text(content.replace(old, new))
    assert operate(path)["connected"] is False


def test_disconnect_keeps_global_selections_for_another_provider(tmp_path):
    path = tmp_path / "config.toml"
    operate(path, True)
    path.write_text(
        path.read_text()
        .replace('model_provider = "fcc"', 'model_provider = "other"')
        .replace("catalog.json", "other.json")
    )
    operate(path, False)
    data = tomllib.loads(path.read_text())
    assert data["model_provider"] == "other"
    assert Path(data["model_catalog_json"]) == tmp_path / "other.json"
    assert "fcc" not in data.get("model_providers", {})


@pytest.mark.parametrize(
    "content",
    [
        "[invalid",
        'model_provider="fcc"\nmodel_provider="other"',
        'model_providers = "bad"',
        "[model_providers]\nfcc = []",
        '[model_providers.fcc]\nauth = "bad"',
    ],
)
def test_invalid_documents_are_not_rewritten(tmp_path, content):
    path = tmp_path / "config.toml"
    path.write_text(content)
    for action in (None, True, False):
        with pytest.raises(ValueError):
            operate(path, action)
        assert path.read_text() == content


def test_atomic_failure_preserves_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('model = "keep"')

    def fail(self, target):
        raise PermissionError("failure")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        operate(path, True)
    assert path.read_text() == 'model = "keep"'
    assert list(tmp_path.iterdir()) == [path]


def test_symlink_unicode_and_path_display(tmp_path):
    target = tmp_path / "actual.toml"
    target.write_text('model = "模型" # Keep 😀\n', encoding="utf-8")
    path = tmp_path / "config.toml"
    try:
        path.symlink_to(target.name)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or privilege")
        raise
    assert operate(path, True)["paths"]["codex_config"] == str(target.resolve())
    assert path.is_symlink()
    assert tomllib.loads(target.read_text(encoding="utf-8"))["model"] == "模型"
    operate(path, False)
    assert path.is_symlink()
    assert "# Keep 😀" in target.read_text(encoding="utf-8")


def test_native_config_path_honors_codex_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert codex_integration.config_path() == tmp_path / ".codex/config.toml"
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "custom"))
    assert codex_integration.config_path() == tmp_path / "custom/config.toml"

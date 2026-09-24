import json
import subprocess

import pytest

from free_claude_code.harnesses import devin_acp_integration as devin

CONFIG_PATH = devin.config_path


@pytest.fixture
def installed(tmp_path, monkeypatch):
    scripts = tmp_path / "FCC scripts"
    scripts.mkdir()
    launcher = scripts / (
        "fcc-opencode.exe" if devin.sys.platform == "win32" else "fcc-opencode"
    )
    launcher.touch(mode=0o755)
    native = tmp_path / "native" / "opencode.exe"
    native.parent.mkdir()
    native.touch(mode=0o755)
    monkeypatch.setattr(devin.sysconfig, "get_path", lambda name: str(scripts))
    monkeypatch.setattr(devin.shutil, "which", lambda name: str(native))
    monkeypatch.setattr(
        devin.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 0, "opencode v2.0.10", ""),
    )
    monkeypatch.setenv("PATH", str(native.parent) + devin.os.pathsep + "other-bin")
    return launcher, native


def test_connect_preserves_registry_and_disconnect_needs_no_binary(
    tmp_path, installed, monkeypatch
):
    path = tmp_path / "registry.json"
    other = {"id": "other", "future": {"data": 3}}
    original = {
        "version": "1.0.0",
        "agents": [other],
        "extensions": ["keep"],
        "future": True,
    }
    path.write_text(json.dumps(original))
    assert devin.configure(path, True)["connected"] is True
    document = json.loads(path.read_text())
    entry = document["agents"][1]
    binary = entry["distribution"]["binary"][devin.platform_key()]
    assert binary["cmd"] == str(installed[0])
    assert binary["args"] == ["acp"]
    assert (
        binary["env"]["PATH"].split(devin.os.pathsep).count(str(installed[1].parent))
        == 1
    )
    assert document["agents"][0] == other
    before = path.read_bytes()
    assert devin.configure(path)["connected"] is True
    assert not devin.refresh_connected(path)
    assert path.read_bytes() == before
    monkeypatch.setattr(devin.shutil, "which", lambda name: None)
    installed[0].unlink()
    assert devin.configure(path, False)["connected"] is False
    assert json.loads(path.read_text()) == original


def test_absent_status_disconnect_and_refresh_do_not_create_files(tmp_path):
    path = tmp_path / "missing" / "registry.json"
    assert not devin.configure(path)["connected"]
    assert not devin.configure(path, False)["connected"]
    assert not devin.refresh_connected(path)
    assert not path.exists()


@pytest.mark.parametrize(
    "source",
    [
        "{",
        "[]",
        '{"agents": {}}',
        '{"agents": [], "agents": []}',
        '{"x": NaN}',
        '{"agents": [{"id": "fcc-opencode"}]}',
        '{"agents": [{"id": "fcc-opencode"}, {"id": "fcc-opencode"}]}',
    ],
)
def test_invalid_or_unowned_registry_is_not_modified(tmp_path, source, installed):
    path = tmp_path / "registry.json"
    path.write_text(source)
    for operation in (None, True, False):
        with pytest.raises(ValueError):
            devin.configure(path, operation)
        assert path.read_text() == source


def test_refresh_preserves_custom_fields_and_disconnect_after_failed_refresh(
    tmp_path, installed, monkeypatch
):
    path = tmp_path / "registry.json"
    devin.configure(path, True)
    document = json.loads(path.read_text())
    entry = document["agents"][0]
    binary = entry["distribution"]["binary"][devin.platform_key()]
    binary["cmd"] = "old-fcc"
    binary["env"]["CUSTOM"] = "keep"
    entry["custom"] = [1]
    path.write_text(json.dumps(document))
    assert devin.refresh_connected(path)
    updated = json.loads(path.read_text())["agents"][0]
    assert updated["custom"] == [1]
    assert (
        updated["distribution"]["binary"][devin.platform_key()]["env"]["CUSTOM"]
        == "keep"
    )
    monkeypatch.setattr(devin.shutil, "which", lambda name: None)
    before = path.read_bytes()
    with pytest.raises(devin.SetupError):
        devin.refresh_connected(path)
    assert path.read_bytes() == before
    assert devin.configure(path)["connected"]
    assert not devin.configure(path, False)["connected"]


def test_failed_write_leaves_existing_registry(tmp_path, installed, monkeypatch):
    path = tmp_path / "registry.json"
    path.write_text('{"agents": []}')
    before = path.read_bytes()

    def fail(*a):
        raise PermissionError("read-only")

    monkeypatch.setattr(devin, "atomic_write_text", fail)
    with pytest.raises(PermissionError):
        devin.configure(path, True)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "system, relative",
    [
        ("win32", "AppData/Roaming/Code/User/acp/registry.json"),
        ("darwin", ".windsurf/acp/registry.json"),
        ("linux", ".windsurf/acp/registry.json"),
    ],
)
def test_documented_paths(system, relative, tmp_path, monkeypatch):
    monkeypatch.setattr(devin.sys, "platform", system)
    monkeypatch.setattr(devin.Path, "home", lambda: tmp_path)
    assert CONFIG_PATH() == tmp_path / relative

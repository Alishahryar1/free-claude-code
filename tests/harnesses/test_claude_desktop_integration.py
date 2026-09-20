import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from free_claude_code.harnesses import claude_desktop_integration as desktop
from free_claude_code.harnesses.claude_desktop_integration import (
    check_unmanaged,
    config_root,
)

URL = "http://127.0.0.1:8000"
TOKEN = "desktop-test-token"


@pytest.mark.parametrize(
    "platform,variable,suffix",
    [
        ("win32", "LOCALAPPDATA", "local/Claude-3p"),
        ("darwin", None, "Library/Application Support/Claude-3p"),
        ("linux", "XDG_CONFIG_HOME", "local/Claude-3p"),
    ],
)
def test_platform_config_root(monkeypatch, tmp_path, platform, variable, suffix):
    monkeypatch.setattr(desktop.sys, "platform", platform)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    if variable:
        monkeypatch.setenv(variable, str(tmp_path / "local"))
    assert config_root() == tmp_path / suffix


@pytest.mark.parametrize("state", ["absent", "empty", "policy", "unreadable"])
def test_windows_policy_boundary(monkeypatch, state):
    opened = []

    def open_key(hive, path):
        opened.append((hive, path))
        if state == "absent":
            raise FileNotFoundError
        if state == "unreadable":
            raise PermissionError
        return nullcontext("test-key")

    monkeypatch.setattr(desktop.sys, "platform", "win32")
    monkeypatch.setitem(
        desktop.sys.modules,
        "winreg",
        SimpleNamespace(
            HKEY_LOCAL_MACHINE=1,
            HKEY_CURRENT_USER=2,
            OpenKey=open_key,
            QueryInfoKey=lambda key: (0, int(state == "policy"), 0),
        ),
    )
    if state in {"policy", "unreadable"}:
        with pytest.raises(desktop.ManagedDesktopError):
            check_unmanaged()
    else:
        check_unmanaged()
        assert opened == [
            (1, r"SOFTWARE\Policies\Claude"),
            (2, r"SOFTWARE\Policies\Claude"),
        ]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def profile(root):
    return root / "configLibrary" / f"{desktop.FCC_ID}.json"


def meta(root):
    return root / "configLibrary" / "_meta.json"


def mode(root):
    return root / "claude_desktop_config.json"


def test_fresh_connection_and_disconnect_preserve_native_library(tmp_path):
    root = tmp_path / "desktop"
    assert desktop.configure(root, URL, TOKEN)["connected"] is False
    assert not root.exists()
    assert desktop.refresh_connected(root, URL, TOKEN) is False
    assert not root.exists()
    assert desktop.configure(root, URL, TOKEN, True)["connected"] is True
    assert read(meta(root))["appliedId"] == desktop.FCC_ID
    assert read(mode(root))["deploymentMode"] == "3p"
    assert read(profile(root))["inferenceGatewayApiKey"] == TOKEN
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    assert desktop.refresh_connected(root, URL, TOKEN) is False
    desktop.configure(root, URL, TOKEN, True)
    assert {p: p.read_bytes() for p in root.rglob("*.json")} == before
    assert desktop.configure(root, URL, TOKEN, False)["connected"] is False
    assert not profile(root).exists()
    assert read(mode(root))["deploymentMode"] == "1p"
    assert read(meta(root))["entries"] == [
        {"id": desktop.DEFAULT_ID, "name": "Default"}
    ]
    assert read(root / "configLibrary" / f"{desktop.DEFAULT_ID}.json") == {}
    after = {p: p.read_bytes() for p in root.rglob("*.json")}
    desktop.configure(root, URL, TOKEN, False)
    assert {p: p.read_bytes() for p in root.rglob("*.json")} == after


def test_refresh_rotates_credentials_without_reactivating(tmp_path):
    desktop.configure(tmp_path, URL, TOKEN, True)
    assert desktop.refresh_connected(tmp_path, "http://localhost:9000", "new-token")
    assert desktop.configure(tmp_path, "http://localhost:9000", "new-token")[
        "connected"
    ]
    write(mode(tmp_path), {"deploymentMode": "1p", "unrelated": True})
    before = profile(tmp_path).read_bytes()
    assert not desktop.refresh_connected(tmp_path, URL, TOKEN)
    assert profile(tmp_path).read_bytes() == before
    assert read(mode(tmp_path))["unrelated"] is True


def test_refresh_repairs_registered_profile_header_without_changing_selection(tmp_path):
    desktop.configure(tmp_path, URL, TOKEN, True)
    config = read(profile(tmp_path))
    config["inferenceCustomHeaders"] = {"X-Other": "preserved"}
    write(profile(tmp_path), config)
    metadata = meta(tmp_path).read_bytes()
    assert not desktop.configure(tmp_path, URL, TOKEN)["connected"]
    assert desktop.refresh_connected(tmp_path, URL, TOKEN)
    assert desktop.configure(tmp_path, URL, TOKEN)["connected"]
    assert meta(tmp_path).read_bytes() == metadata
    assert read(profile(tmp_path))["inferenceCustomHeaders"] == {
        "X-Other": "preserved",
        "X-FCC-Model-View": "claude-desktop",
    }


@pytest.mark.parametrize("failed_write", [1, 2, 3])
def test_disconnect_last_entry_partial_write_is_retryable(
    tmp_path, monkeypatch, failed_write
):
    desktop.configure(tmp_path, URL, TOKEN, True)
    write(
        meta(tmp_path),
        {
            "appliedId": desktop.FCC_ID,
            "entries": [{"id": desktop.FCC_ID, "name": "FCC"}],
        },
    )
    (tmp_path / "configLibrary" / f"{desktop.DEFAULT_ID}.json").unlink()
    atomic = desktop.atomic_write_text
    calls = 0

    def fail(path, content):
        nonlocal calls
        calls += 1
        if calls == failed_write:
            raise PermissionError("test")
        atomic(path, content)

    monkeypatch.setattr(desktop, "atomic_write_text", fail)
    with pytest.raises(PermissionError):
        desktop.configure(tmp_path, URL, TOKEN, False)
    assert profile(tmp_path).exists()
    if failed_write > 1:
        assert read(mode(tmp_path))["deploymentMode"] == "1p"
    monkeypatch.setattr(desktop, "atomic_write_text", atomic)
    assert not desktop.configure(tmp_path, URL, TOKEN, False)["connected"]
    assert not profile(tmp_path).exists()
    assert read(meta(tmp_path))["appliedId"] == desktop.DEFAULT_ID


def test_other_profiles_and_unknown_settings_are_preserved(tmp_path):
    other = "11111111-1111-4111-8111-111111111111"
    other_path = tmp_path / "configLibrary" / f"{other}.json"
    write(other_path, {"inferenceProvider": "gateway", "secret": "other-secret"})
    write(
        meta(tmp_path),
        {"entries": [{"id": other, "name": "Other"}], "appliedId": other, "extra": 1},
    )
    write(mode(tmp_path), {"deploymentMode": "3p", "keep": "yes"})
    before = other_path.read_bytes()
    desktop.configure(tmp_path, URL, TOKEN, True)
    config = read(profile(tmp_path))
    config["keep"] = True
    config["inferenceCustomHeaders"]["X-Other"] = "value"
    config["inferenceModels"] = ["old-model"]
    write(profile(tmp_path), config)
    desktop.refresh_connected(tmp_path, URL, TOKEN)
    assert read(profile(tmp_path))["keep"] is True
    assert read(profile(tmp_path))["inferenceCustomHeaders"]["X-Other"] == "value"
    assert "inferenceModels" not in read(profile(tmp_path))
    metadata = read(meta(tmp_path))
    metadata["appliedId"] = other
    metadata["hybridPointer"] = {"bootstrapUrl": "https://example.test"}
    write(meta(tmp_path), metadata)
    assert not desktop.refresh_connected(tmp_path, URL, TOKEN)
    desktop.configure(tmp_path, URL, TOKEN, False)
    assert read(meta(tmp_path))["appliedId"] == other
    assert "hybridPointer" in read(meta(tmp_path))
    assert read(meta(tmp_path))["extra"] == 1
    assert read(mode(tmp_path)) == {"deploymentMode": "3p", "keep": "yes"}
    assert other_path.read_bytes() == before


@pytest.mark.parametrize("failed_write", [1, 2, 3, 4])
def test_connect_partial_write_is_retryable(tmp_path, monkeypatch, failed_write):
    atomic = desktop.atomic_write_text
    calls = 0

    def fail(path, content):
        nonlocal calls
        calls += 1
        if calls == failed_write:
            raise PermissionError("test")
        atomic(path, content)

    monkeypatch.setattr(desktop, "atomic_write_text", fail)
    with pytest.raises(PermissionError):
        desktop.configure(tmp_path, URL, TOKEN, True)
    assert not mode(tmp_path).exists()
    monkeypatch.setattr(desktop, "atomic_write_text", atomic)
    assert desktop.configure(tmp_path, URL, TOKEN, True)["connected"]


@pytest.mark.parametrize("missing_header", [False, True])
def test_disconnect_recovers_orphan_after_delete_failure(
    tmp_path, monkeypatch, missing_header
):
    desktop.configure(tmp_path, URL, TOKEN, True)
    if missing_header:
        config = read(profile(tmp_path))
        config.pop("inferenceCustomHeaders")
        write(profile(tmp_path), config)
    unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path == profile(tmp_path):
            raise PermissionError("test")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    with pytest.raises(PermissionError):
        desktop.configure(tmp_path, URL, TOKEN, False)
    assert read(mode(tmp_path))["deploymentMode"] == "1p"
    assert read(meta(tmp_path))["appliedId"] != desktop.FCC_ID
    monkeypatch.setattr(Path, "unlink", unlink)
    desktop.configure(tmp_path, URL, TOKEN, False)
    assert not profile(tmp_path).exists()


def test_disconnect_last_entry_creates_default(tmp_path):
    desktop.configure(tmp_path, URL, TOKEN, True)
    write(
        meta(tmp_path),
        {
            "appliedId": desktop.FCC_ID,
            "entries": [{"id": desktop.FCC_ID, "name": "FCC"}],
        },
    )
    (tmp_path / "configLibrary" / f"{desktop.DEFAULT_ID}.json").unlink()
    status = desktop.configure(tmp_path, URL, TOKEN)
    paths = status["paths"]
    assert isinstance(paths, dict)
    assert "default_profile" in paths
    desktop.configure(tmp_path, URL, TOKEN, False)
    assert read(meta(tmp_path))["appliedId"] == desktop.DEFAULT_ID


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {},
        {"entries": []},
        {"entries": [{"id": desktop.FCC_ID, "name": "FCC"}], "appliedId": []},
        {"entries": [{"id": desktop.FCC_ID, "name": "FCC"}], "appliedId": {}},
        {
            "entries": [{"id": "../../other", "name": "Other"}],
            "appliedId": "../../other",
        },
    ],
)
def test_invalid_metadata_is_never_overwritten(tmp_path, metadata):
    write(meta(tmp_path), metadata)
    before = meta(tmp_path).read_bytes()
    with pytest.raises(ValueError):
        desktop.configure(tmp_path, URL, TOKEN, True)
    assert meta(tmp_path).read_bytes() == before
    assert not profile(tmp_path).exists()


def test_reserved_profile_collision_is_not_overwritten(tmp_path):
    write(profile(tmp_path), {"inferenceProvider": "bedrock"})
    with pytest.raises(ValueError):
        desktop.configure(tmp_path, URL, TOKEN, True)
    assert read(profile(tmp_path)) == {"inferenceProvider": "bedrock"}


@pytest.mark.parametrize("url,token", [("http://192.0.2.1:8000", TOKEN), (URL, "")])
def test_invalid_gateway_does_not_write(tmp_path, url, token):
    with pytest.raises(ValueError):
        desktop.configure(tmp_path, url, token, True)
    assert not list(tmp_path.rglob("*.json"))

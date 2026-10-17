import genericpath
import json
import os
import tomllib
from pathlib import Path

import pytest

from free_claude_code.harnesses import (
    claude_desktop_integration,
    claude_integration,
    codex_integration,
    vscode_chat_integration,
)
from free_claude_code.harnesses.dsh_config import DSH_PROVIDER_ID
from free_claude_code.harnesses.integration_cleanup import (
    IntegrationCleanupError,
    IntegrationPaths,
    disconnect_integrations,
)

URL = "http://127.0.0.1:8082"
TOKEN = "cleanup-test-token"


def paths_at(root: Path) -> IntegrationPaths:
    return IntegrationPaths(
        root / "fcc/.env",
        root / "vscode/settings.json",
        root / "vscode/chatLanguageModels.json",
        root / "codex/config.toml",
        root / "fcc/catalog.json",
        root / "jetbrains/acp.json",
        root / "desktop",
        root / "fcc/desktop-disconnect.json",
        root / "dsh",
    )


@pytest.fixture
def paths(tmp_path, monkeypatch):
    paths = paths_at(tmp_path)
    paths.env.parent.mkdir()
    paths.env.write_text(f"PORT=8082\nANTHROPIC_AUTH_TOKEN={TOKEN}\nMODEL=invalid\n")
    for key in ("HOST", "PORT", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(claude_desktop_integration, "check_unmanaged", lambda: None)
    return paths


def test_cleanup_disconnects_native_connections_and_preserves_unrelated_settings(paths):
    paths.claude_vscode.parent.mkdir()
    paths.claude_vscode.write_text('{"editor.fontSize": 17}')
    claude_integration.configure(paths.claude_vscode, URL, TOKEN, True)
    paths.codex.parent.mkdir()
    paths.codex.write_text(
        '# Keep comment\nmodel="my-model"\n[model_providers.other]\nname="Other"\n'
    )
    codex_integration.configure(paths.codex, paths.codex_catalog, URL, True)
    paths.vscode_chat.write_text('[{"vendor":"other","name":"Keep"}]')
    vscode_chat_integration.configure(paths.vscode_chat, URL, TOKEN, [])
    paths.jetbrains.parent.mkdir()
    paths.jetbrains.write_text(
        json.dumps(
            {
                "agent_servers": {
                    "Claude Code (FCC)": {"env": {"FCC_JETBRAINS_ACP": "1"}},
                    "Other": {"command": "keep"},
                }
            }
        )
    )
    claude_desktop_integration.configure(
        paths.desktop, URL, TOKEN, True, disconnect_path=paths.desktop_disconnect
    )
    mode = paths.desktop / "claude_desktop_config.json"
    mode.write_text(json.dumps({"deploymentMode": "3p", "keep": True}))
    profile = paths.dsh / "profiles/desktop/cordis.patch.yml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        f"- id: llm-pi-ai\n  config:\n    providers:\n      {DSH_PROVIDER_ID}:\n        baseURL: http://127.0.0.1:8082/v1\n      other:\n        keep: true\n- id: agent-default-model\n  config:\n    provider: {DSH_PROVIDER_ID}\n    model: test\n"
    )
    credentials = paths.dsh / ".credentials.yaml"
    credentials.write_text(
        f"version: 1\nrefs:\n  FCC_DSH_DESKTOP_API_KEY: {TOKEN}\n  OTHER: keep\n"
    )

    disconnect_integrations(paths)

    assert json.loads(paths.claude_vscode.read_text()) == {"editor.fontSize": 17}
    assert tomllib.loads(paths.codex.read_text()) == {
        "model": "my-model",
        "model_providers": {"other": {"name": "Other"}},
    }
    assert "# Keep comment" in paths.codex.read_text()
    assert json.loads(paths.vscode_chat.read_text()) == [
        {"vendor": "other", "name": "Keep"}
    ]
    assert json.loads(paths.jetbrains.read_text()) == {
        "agent_servers": {"Other": {"command": "keep"}}
    }
    assert json.loads(mode.read_text()) == {"deploymentMode": "1p", "keep": True}
    assert not (
        paths.desktop / f"configLibrary/{claude_desktop_integration.FCC_ID}.json"
    ).exists()
    assert not paths.desktop_disconnect.exists()
    assert f"      {DSH_PROVIDER_ID}:" not in profile.read_text()
    assert "        keep: true" in profile.read_text()
    assert "FCC_DSH_DESKTOP_API_KEY" not in credentials.read_text()
    assert "OTHER: keep" in credentials.read_text()
    assert paths.env.exists()
    tracked = [
        paths.claude_vscode,
        paths.codex,
        paths.vscode_chat,
        paths.jetbrains,
        mode,
        profile,
        credentials,
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tracked}
    disconnect_integrations(paths)
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tracked
    }


def test_missing_native_configurations_create_no_files(tmp_path):
    disconnect_integrations(paths_at(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("change", ["token", "url"])
def test_manually_changed_vscode_connection_is_preserved(paths, change):
    claude_integration.configure(paths.claude_vscode, URL, TOKEN, True)
    before = paths.claude_vscode.read_text().replace(
        TOKEN if change == "token" else URL,
        "another-token" if change == "token" else "https://other.example",
    )
    paths.claude_vscode.write_text(before)
    with pytest.raises(IntegrationCleanupError, match="Claude Code in VS Code"):
        disconnect_integrations(paths)
    assert paths.claude_vscode.read_text() == before
    assert paths.env.exists()


@pytest.mark.parametrize(
    ("key", "value", "connected_url"),
    [
        ("PORT", "9000", "http://127.0.0.1:9000"),
        ("HOST", "192.0.2.20", "http://192.0.2.20:8082"),
    ],
)
@pytest.mark.parametrize("keep_override", [False, True])
def test_cleanup_of_connection_from_process_override(
    paths, monkeypatch, key, value, connected_url, keep_override
):
    monkeypatch.setenv(key, value)
    claude_integration.configure(paths.claude_vscode, connected_url, TOKEN, True)
    vscode_chat_integration.configure(paths.vscode_chat, connected_url, TOKEN, [])
    before = paths.claude_vscode.read_bytes()
    chat_before = paths.vscode_chat.read_bytes()
    env_before = paths.env.read_bytes()
    if keep_override:
        disconnect_integrations(paths)
        assert json.loads(paths.claude_vscode.read_text()) == {}
        assert json.loads(paths.vscode_chat.read_text()) == []
    else:
        monkeypatch.delenv(key)
        with pytest.raises(IntegrationCleanupError, match="Claude Code in VS Code"):
            disconnect_integrations(paths)
        assert paths.claude_vscode.read_bytes() == before
        assert paths.vscode_chat.read_bytes() == chat_before
    assert paths.env.read_bytes() == env_before


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("claudeCode.disableLoginPrompt", False),
        ("claudeCode.disableLoginPrompt", None),
        ("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "0"),
        ("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", None),
    ],
)
def test_unmarked_vscode_settings_are_preserved_without_blocking(paths, key, value):
    claude_integration.configure(
        paths.claude_vscode, "https://other.example", "unrelated-token", True
    )
    document = json.loads(paths.claude_vscode.read_text())
    if key == "claudeCode.disableLoginPrompt":
        if value is None:
            document.pop(key)
        else:
            document[key] = value
    else:
        entries = document["claudeCode.environmentVariables"]
        if value is None:
            document["claudeCode.environmentVariables"] = [
                entry for entry in entries if entry["name"] != key
            ]
        else:
            next(entry for entry in entries if entry["name"] == key)["value"] = value
    paths.claude_vscode.write_text(json.dumps(document))
    before = paths.claude_vscode.read_bytes()
    disconnect_integrations(paths)
    assert paths.claude_vscode.read_bytes() == before


@pytest.mark.parametrize("change", ["login", "discovery", "both"])
@pytest.mark.parametrize("remove", [False, True])
@pytest.mark.parametrize("mismatch", [None, "url", "token"])
def test_vscode_cleanup_uses_gateway_identity_without_optional_flags(
    paths, change, remove, mismatch
):
    claude_integration.configure(paths.claude_vscode, URL, TOKEN, True)
    document = json.loads(paths.claude_vscode.read_text())
    document["editor.fontSize"] = 17
    if change in {"login", "both"}:
        if remove:
            document.pop("claudeCode.disableLoginPrompt")
        else:
            document["claudeCode.disableLoginPrompt"] = False
    entries = document["claudeCode.environmentVariables"]
    if change in {"discovery", "both"}:
        discovery = next(
            entry
            for entry in entries
            if entry["name"] == "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"
        )
        if remove:
            entries.remove(discovery)
        else:
            discovery["value"] = "0"
    if mismatch is not None:
        name, value = (
            ("ANTHROPIC_BASE_URL", "https://other.example")
            if mismatch == "url"
            else ("ANTHROPIC_AUTH_TOKEN", "changed-token")
        )
        next(entry for entry in entries if entry["name"] == name)["value"] = value
    paths.claude_vscode.write_text(json.dumps(document))
    before = paths.claude_vscode.read_bytes()

    if mismatch is None:
        disconnect_integrations(paths)
        assert json.loads(paths.claude_vscode.read_text()) == {"editor.fontSize": 17}
    else:
        with pytest.raises(IntegrationCleanupError, match="Claude Code in VS Code"):
            disconnect_integrations(paths)
        assert paths.claude_vscode.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("env", "connection settings"),
        ("desktop", "Claude Desktop"),
        ("desktop_disconnect", "Claude Desktop"),
        ("dsh", "DSH Desktop"),
    ],
)
@pytest.mark.parametrize("failure", [PermissionError, OSError])
def test_cleanup_fails_when_path_access_cannot_be_confirmed(
    paths, monkeypatch, field, message, failure
):
    blocked = getattr(paths, field)
    if field == "desktop":
        claude_desktop_integration.configure(
            paths.desktop, URL, TOKEN, True, disconnect_path=paths.desktop_disconnect
        )
    elif field == "dsh":
        paths.dsh.mkdir()
        (paths.dsh / ".credentials.yaml").write_text(
            f"version: 1\nrefs:\n  FCC_DSH_DESKTOP_API_KEY: {TOKEN}\n"
        )
    elif field == "desktop_disconnect":
        blocked.write_text(
            json.dumps(
                {"root": str(paths.desktop.resolve()), "return_to_sign_in": True}
            )
        )
    before = {
        path: path.read_bytes()
        for path in paths.env.parent.parent.rglob("*")
        if path.is_file()
    }
    original_stat = os.stat

    def inaccessible(path, *args, **kwargs):
        if Path(path) == blocked:
            raise failure("private-filesystem-detail")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "stat", inaccessible)
        # Exercise POSIX existence semantics on Windows too, where exists uses
        # a separate native fast path instead of os.stat.
        patch.setattr(os.path, "exists", genericpath.exists)
        with pytest.raises(IntegrationCleanupError, match=message) as error:
            disconnect_integrations(paths)
        assert "private-filesystem-detail" not in str(error.value)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    "connected_url",
    ["http://localhost:8082/", "http://[::1]:8082", "http://127.0.0.1:8082/"],
)
def test_equivalent_loopback_connection_is_disconnected(paths, connected_url):
    claude_integration.configure(paths.claude_vscode, connected_url, TOKEN, True)
    disconnect_integrations(paths)
    assert json.loads(paths.claude_vscode.read_text()) == {}


def test_missing_identity_does_not_claim_gateway_settings(paths):
    claude_integration.configure(paths.claude_vscode, URL, TOKEN, True)
    paths.env.unlink()
    before = paths.claude_vscode.read_bytes()
    with pytest.raises(IntegrationCleanupError, match="Claude Code in VS Code"):
        disconnect_integrations(paths)
    assert paths.claude_vscode.read_bytes() == before


@pytest.mark.parametrize("content", ["PORT=invalid", "ANTHROPIC_AUTH_TOKEN="])
def test_invalid_connection_settings_abort_without_modifying_apps(paths, content):
    claude_integration.configure(paths.claude_vscode, URL, TOKEN, True)
    paths.env.write_text(content)
    before = paths.claude_vscode.read_bytes()
    with pytest.raises(IntegrationCleanupError, match="connection settings"):
        disconnect_integrations(paths)
    assert paths.claude_vscode.read_bytes() == before
    assert paths.env.read_text() == content


def test_invalid_settings_error_does_not_disclose_source_text(paths):
    paths.claude_vscode.parent.mkdir()
    paths.claude_vscode.write_text('{"secret": "cleanup-secret", invalid}')
    with pytest.raises(IntegrationCleanupError) as error:
        disconnect_integrations(paths)
    assert "cleanup-secret" not in str(error.value)
    assert paths.env.exists()


@pytest.mark.parametrize(
    "error", [claude_desktop_integration.ManagedDesktopError, PermissionError]
)
def test_desktop_policy_or_access_failure_preserves_recovery_data(
    paths, monkeypatch, error
):
    paths.desktop.mkdir()

    def fail():
        raise error("private-policy-data")

    monkeypatch.setattr(claude_desktop_integration, "check_unmanaged", fail)
    with pytest.raises(IntegrationCleanupError, match="Claude Desktop") as result:
        disconnect_integrations(paths)
    assert "private-policy-data" not in str(result.value)
    assert paths.env.exists()


def test_orphan_dsh_credential_is_removed_without_creating_profile(paths):
    paths.dsh.mkdir()
    credentials = paths.dsh / ".credentials.yaml"
    credentials.write_text(
        f"version: 1\nrefs:\n  FCC_DSH_DESKTOP_API_KEY: {TOKEN}\n  OTHER: keep\n"
    )
    disconnect_integrations(paths)
    assert "FCC_DSH_DESKTOP_API_KEY" not in credentials.read_text()
    assert "OTHER: keep" in credentials.read_text()
    assert not (paths.dsh / "profiles").exists()

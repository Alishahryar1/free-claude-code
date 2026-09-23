import json
import tomllib

import pytest

from free_claude_code.harnesses import claude_integration, codex_integration

URL = "http://127.0.0.1:8000"
TOKEN = "test-integration-token"
# Original connection markers, deliberately independent of today's desired settings.
OLD_CLAUDE = {
    "editor.fontSize": 15,
    "claudeCode.disableLoginPrompt": True,
    "claudeCode.environmentVariables": [
        {"name": "ANTHROPIC_BASE_URL", "value": "http://localhost:8000"},
        {"name": "ANTHROPIC_AUTH_TOKEN", "value": TOKEN},
        {"name": "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY", "value": "1"},
        {"name": "KEEP_ME", "value": "user-value"},
    ],
}
OLD_CODEX = """model = "my-model" # Preserve my choice
model_provider = "fcc"
model_catalog_json = "/old/catalog.json"
[model_providers.fcc]
base_url = "http://localhost:8000/v1"
[model_providers.other]
base_url = "https://example.com/v1"
"""


def test_refresh_upgrades_original_claude_directly_and_only_once(tmp_path):
    settings, state = tmp_path / "settings.json", tmp_path / ".claude.json"
    settings.write_text(json.dumps(OLD_CLAUDE))
    state.write_text('{"theme":"dark"}')
    assert (
        claude_integration.configure(settings, state, URL, TOKEN)["connected"] is False
    )
    assert claude_integration.refresh_connected(settings, state, URL, TOKEN) is True
    saved = json.loads(settings.read_text())
    env = {
        entry["name"]: entry["value"]
        for entry in saved["claudeCode.environmentVariables"]
    }
    assert env["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "1"
    assert env["CLAUDE_CODE_AUTO_MODE_SERVER"] == "0"
    assert env["KEEP_ME"] == "user-value"
    assert saved["editor.fontSize"] == 15
    assert json.loads(state.read_text()) == {
        "theme": "dark",
        "hasCompletedOnboarding": True,
    }
    assert (
        claude_integration.configure(settings, state, URL, TOKEN)["connected"] is True
    )
    before = [path.stat().st_mtime_ns for path in (settings, state)]
    assert claude_integration.refresh_connected(settings, state, URL, TOKEN) is False
    assert [path.stat().st_mtime_ns for path in (settings, state)] == before


@pytest.mark.parametrize("change", ["missing", "disconnected", "url", "token"])
def test_refresh_does_not_claim_unrecognized_claude(tmp_path, change):
    settings, state = tmp_path / "settings.json", tmp_path / ".claude.json"
    document = json.loads(json.dumps(OLD_CLAUDE))
    if change == "disconnected":
        document.pop("claudeCode.disableLoginPrompt")
    elif change in {"url", "token"}:
        document["claudeCode.environmentVariables"][0 if change == "url" else 1][
            "value"
        ] = "other"
    if change != "missing":
        settings.write_text(json.dumps(document))
    before = settings.read_bytes() if settings.exists() else None
    # Unrelated Claude state is not even parsed when the integration is not ours.
    state.write_text("{invalid")
    assert claude_integration.refresh_connected(settings, state, URL, TOKEN) is False
    assert (settings.read_bytes() if settings.exists() else None) == before
    assert state.read_text() == "{invalid"


def test_refresh_claude_retry_finishes_partial_write(tmp_path, monkeypatch):
    settings, state = tmp_path / "settings.json", tmp_path / ".claude.json"
    settings.write_text(json.dumps(OLD_CLAUDE))
    before = settings.read_bytes()
    write = claude_integration.atomic_write_text

    def fail_settings(path, content):
        if path == settings:
            raise PermissionError("denied")
        write(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(claude_integration, "atomic_write_text", fail_settings)
        with pytest.raises(PermissionError):
            claude_integration.refresh_connected(settings, state, URL, TOKEN)
    assert settings.read_bytes() == before
    assert json.loads(state.read_text())["hasCompletedOnboarding"] is True
    assert claude_integration.refresh_connected(settings, state, URL, TOKEN) is True
    assert (
        claude_integration.configure(settings, state, URL, TOKEN)["connected"] is True
    )


def test_refresh_codex_merges_latest_settings_without_changing_model(tmp_path):
    path, catalog = tmp_path / "config.toml", tmp_path / "catalog.json"
    path.write_text(OLD_CODEX)
    assert codex_integration.recognizes_connection(path, URL) is True
    assert codex_integration.refresh_connected(path, catalog, URL) is True
    saved = tomllib.loads(path.read_text())
    assert saved["model"] == "my-model"
    assert "# Preserve my choice" in path.read_text()
    assert saved["model_catalog_json"] == str(catalog.resolve())
    assert saved["model_providers"]["fcc"]["auth"] == {
        "command": "fcc-codex",
        "args": ["--print-proxy-auth-token"],
    }
    assert saved["model_providers"]["other"] == {"base_url": "https://example.com/v1"}
    assert codex_integration.configure(path, catalog, URL)["connected"] is True
    before = path.stat().st_mtime_ns
    assert codex_integration.refresh_connected(path, catalog, URL) is False
    assert path.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "source",
    [
        None,
        "",
        OLD_CODEX.replace('model_provider = "fcc"', 'model_provider = "other"'),
        OLD_CODEX.replace("localhost:8000", "localhost:9999"),
    ],
)
def test_refresh_codex_leaves_unrecognized_settings_alone(tmp_path, source):
    path = tmp_path / "config.toml"
    if source is not None:
        path.write_text(source)
    before = path.read_bytes() if path.exists() else None
    assert codex_integration.recognizes_connection(path, URL) is False
    assert (
        codex_integration.refresh_connected(path, tmp_path / "catalog.json", URL)
        is False
    )
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize("integration", ["claude", "codex"])
def test_refresh_rejects_malformed_input_without_writing(tmp_path, integration):
    path = tmp_path / "settings"
    path.write_text("{invalid")
    with pytest.raises(ValueError):
        if integration == "claude":
            claude_integration.refresh_connected(
                path, tmp_path / ".claude.json", URL, TOKEN
            )
        else:
            codex_integration.refresh_connected(path, tmp_path / "catalog.json", URL)
    assert path.read_text() == "{invalid"
    assert list(tmp_path.iterdir()) == [path]

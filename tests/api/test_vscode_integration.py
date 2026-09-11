import json

import pytest
from fastapi.testclient import TestClient

from free_claude_code.cli import vscode
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app, runtime_for_app

ROOT = "/admin/api/integrations/claude-vscode"


@pytest.fixture
def integration(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(vscode, "settings_path", lambda: path)
    monkeypatch.setattr(vscode, "claude_state_path", lambda: tmp_path / ".claude.json")
    app = create_test_app(
        Settings(host="0.0.0.0", port=4321, proxy_auth_token="integration-secret")
    )
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client, path, runtime_for_app(app)


def test_routes_read_actual_file_connect_and_disconnect(integration):
    client, path, _ = integration
    assert client.get(ROOT).json()["connected"] is False
    response = client.post(f"{ROOT}/connect")
    assert response.status_code == 200
    assert response.json() == {"connected": True}
    assert response.headers["cache-control"] == "no-store"
    data = json.loads(path.read_text())
    environment = {
        entry["name"]: entry["value"]
        for entry in data["claudeCode.environmentVariables"]
    }
    assert environment["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:4321"
    assert environment["ANTHROPIC_AUTH_TOKEN"] == "integration-secret"
    assert "integration-secret" not in response.text
    assert client.get(ROOT).json()["connected"] is True
    state_path = path.parent / ".claude.json"
    assert json.loads(state_path.read_text())["hasCompletedOnboarding"] is True
    state_path.write_text('{"hasCompletedOnboarding":false}')
    assert client.get(ROOT).json()["connected"] is False
    assert client.post(f"{ROOT}/connect").json() == {"connected": True}
    path.write_text("{}")
    assert client.get(ROOT).json()["connected"] is False
    assert client.post(f"{ROOT}/disconnect").json() == {"connected": False}
    assert json.loads(state_path.read_text())["hasCompletedOnboarding"] is True


@pytest.mark.parametrize("field", ["vscode_settings", "claude_state"])
def test_status_reports_resolved_file_paths_without_creating_files(integration, field):
    client, path, _ = integration
    state_path = path.parent / ".claude.json"
    result = client.get(ROOT)
    assert result.json()["paths"] == {
        "vscode_settings": str(path.resolve()),
        "claude_state": str(state_path.resolve()),
    }
    assert result.headers["cache-control"] == "no-store"
    assert not path.exists()
    assert not state_path.exists()
    link = path if field == "vscode_settings" else state_path
    target = path.parent / "actual-settings.json"
    target.write_text("{}")
    try:
        link.symlink_to(target.name)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or privilege")
        raise
    assert client.get(ROOT).json()["paths"][field] == str(target.resolve())


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect"])
@pytest.mark.parametrize("target", ["settings", "state"])
def test_invalid_file_returns_safe_uncached_error(integration, action, target):
    client, path, _ = integration
    source = '{"integration-secret": malformed}'
    if target == "state":
        path = path.parent / ".claude.json"
    path.write_text(source)
    response = client.request("GET" if not action else "POST", ROOT + action)
    expected_error = target == "settings" or action != "/disconnect"
    assert response.status_code == (400 if expected_error else 200)
    assert response.headers["cache-control"] == "no-store"
    assert "integration-secret" not in response.text
    if expected_error:
        assert response.json()["detail"]
    assert path.read_text() == source


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect"])
@pytest.mark.parametrize(
    "headers", [{"Host": "evil.test"}, {"Origin": "https://evil.test"}]
)
def test_local_admin_security_applies(integration, action, headers):
    client, path, _ = integration
    response = client.request(
        "GET" if not action else "POST", ROOT + action, headers=headers
    )
    assert response.status_code == 403
    assert not path.exists()


def test_remote_client_is_rejected(integration):
    client, path, _ = integration
    with TestClient(
        client.app, base_url="http://127.0.0.1", client=("192.0.2.1", 50000)
    ) as remote:
        assert remote.post(f"{ROOT}/connect").status_code == 403
    assert not path.exists()


def test_io_failure_returns_safe_error(integration, monkeypatch):
    client, _, _ = integration

    def fail(*args):
        raise PermissionError("integration-secret")

    monkeypatch.setattr(vscode, "configure", fail)
    response = client.post(f"{ROOT}/connect")
    assert response.status_code == 503
    assert "integration-secret" not in response.text


def test_pending_restart_and_shutdown_do_not_write(integration):
    client, path, runtime = integration
    runtime._pending_fields = ["PORT"]
    assert client.post(f"{ROOT}/connect").status_code == 503
    runtime._pending_fields = []
    runtime.begin_shutdown()
    assert client.post(f"{ROOT}/connect").status_code == 503
    assert not path.exists()

import json

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.harnesses import antigravity_integration as native
from tests.api.support import create_test_app, runtime_for_app

ROOT = "/admin/api/integrations/antigravity"


@pytest.fixture
def integration():
    app = create_test_app(Settings(proxy_auth_token="integration-secret"))
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client, runtime_for_app(app)


def test_admin_connection_preserves_native_settings_and_recognizes_existing_record(
    integration,
):
    client, _ = integration
    path = native.config_path()
    assert client.get(ROOT).json()["connected"] is False
    assert not path.exists() and not native.state_path().exists()
    original = {
        "modelProvider": "personal",
        "model": "selected",
        "customModelsConfig": {"customModels": {"own": {"modelName": "native"}}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(original))
    response = client.post(ROOT + "/connect")
    assert response.status_code == 200
    assert response.json()["connected"] is True
    assert response.headers["cache-control"] == "no-store"
    assert (
        "integration-secret"
        not in response.text + path.read_text() + native.state_path().read_text()
    )
    assert client.get(ROOT).json()["connected"] is True
    assert client.post(ROOT + "/connect").json()["connected"] is True
    assert client.post(ROOT + "/disconnect").json()["connected"] is False
    assert json.loads(path.read_text()) == original
    assert not native.state_path().exists()


def test_status_and_disconnect_do_not_require_catalog(integration, monkeypatch):
    client, runtime = integration
    assert client.post(ROOT + "/connect").status_code == 200

    def fail():
        raise AssertionError("catalog must not be used")

    monkeypatch.setattr(
        runtime._integrations.provider_manager, "catalog_snapshot", fail
    )
    assert client.get(ROOT).json()["connected"] is True
    assert client.post(ROOT + "/disconnect").json()["connected"] is False


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect"])
@pytest.mark.parametrize("target", ["settings", "record"])
def test_invalid_files_return_safe_errors_without_mutation(integration, action, target):
    client, _ = integration
    assert client.post(ROOT + "/connect").status_code == 200
    path = native.config_path() if target == "settings" else native.state_path()
    path.write_text('{"integration-secret": invalid}')
    before = path.read_bytes()
    result = client.request("POST" if action else "GET", ROOT + action)
    assert result.status_code == 400
    assert "integration-secret" not in result.text
    assert result.headers["cache-control"] == "no-store"
    assert path.read_bytes() == before


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect"])
@pytest.mark.parametrize(
    "headers", [{"Host": "evil.test"}, {"Origin": "https://evil.test"}]
)
def test_local_admin_security(integration, action, headers):
    client, _ = integration
    response = client.request(
        "POST" if action else "GET", ROOT + action, headers=headers
    )
    assert response.status_code == 403
    assert not native.state_path().exists()


def test_remote_admin_rejected(integration):
    client, _ = integration
    with TestClient(
        client.app, base_url="http://127.0.0.1", client=("192.0.2.1", 50000)
    ) as remote:
        assert remote.post(ROOT + "/connect").status_code == 403
    assert not native.state_path().exists()


def test_io_error_is_safe_and_retryable(integration, monkeypatch):
    client, _ = integration

    def fail(*args, **kwargs):
        raise PermissionError("integration-secret")

    with monkeypatch.context() as patch:
        patch.setattr(native, "configure", fail)
        response = client.post(ROOT + "/connect")
        assert response.status_code == 503
        assert "integration-secret" not in response.text
    assert client.post(ROOT + "/connect").json()["connected"] is True

"""The DSH desktop card persists through the existing admin/service boundary."""

import json

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.harnesses import dsh_desktop_integration as desktop
from tests.api.support import create_test_app

ROOT = "/admin/api/integrations/dsh-desktop"


@pytest.fixture
def client():
    profile = desktop.config_home() / "profiles/desktop"
    profile.mkdir(parents=True)
    (profile / "package.json").write_text(
        json.dumps(
            {
                "dsh": {
                    "profile": {
                        "bundles": ["@deepseek-ai/dsh-base", "@deepseek-ai/dsh-web-app"]
                    }
                },
            }
        )
    )
    app = create_test_app(
        Settings(host="0.0.0.0", port=4321, proxy_auth_token="desktop-secret")
    )
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client


def test_configure_status_disconnect_hide_credentials(client):
    before = client.get(ROOT)
    assert before.status_code == 200
    assert not before.json()["connected"]
    response = client.post(ROOT + "/connect")
    assert response.status_code == 200, response.text
    assert response.json()["connected"]
    assert "desktop-secret" not in response.text
    status = client.get(ROOT)
    assert status.json()["connected"]
    assert status.headers["cache-control"] == "no-store"
    assert "desktop-secret" not in status.text
    response = client.post(ROOT + "/disconnect")
    assert response.status_code == 200
    assert not response.json()["connected"]
    assert not client.get(ROOT).json()["configured"]


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect", "/refresh"])
def test_routes_reject_foreign_origins(client, action):
    response = client.request(
        "POST" if action else "GET",
        ROOT + action,
        headers={"Origin": "https://evil.test"},
    )
    assert response.status_code == 403
    assert not (desktop.config_home() / ".credentials.yaml").exists()


def test_invalid_yaml_does_not_leak_its_secret_line(client):
    path = desktop.config_home() / ".credentials.yaml"
    path.write_text("refs: {KEY: secret-one, KEY: secret-two}")
    response = client.post(ROOT + "/connect")
    assert response.status_code == 400
    assert "secret-one" not in response.text
    assert "secret-two" not in response.text


def test_missing_native_app_returns_setup_guidance(client):
    (desktop.config_home() / "profiles/desktop/package.json").unlink()
    response = client.post(ROOT + "/connect")
    assert response.status_code == 400
    assert "open" in response.json()["detail"].lower()
    assert not (desktop.config_home() / ".credentials.yaml").exists()

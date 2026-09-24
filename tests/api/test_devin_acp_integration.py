import time

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.harnesses import devin_acp_integration as devin
from tests.api.support import create_test_app
from tests.harnesses.test_devin_acp_integration import installed as installed

ROOT = "/admin/api/integrations/devin-acp"


@pytest.fixture
def client(installed):
    with TestClient(
        create_test_app(Settings(proxy_auth_token="secret")),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as client:
        yield client


def test_connect_readonly_status_and_disconnect_after_failed_refresh(client, installed):
    response = client.get(ROOT)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["connected"] is False
    assert not devin.config_path().exists()
    response = client.post(ROOT + "/connect")
    assert response.status_code == 200
    assert response.json()["connected"]
    assert "secret" not in response.text
    before = devin.config_path().read_bytes()
    assert b"secret" not in before
    assert client.get(ROOT).json()["connected"]
    assert devin.config_path().read_bytes() == before
    installed[0].unlink()
    assert client.post(ROOT + "/refresh").status_code == 200
    for _ in range(100):
        state = client.get(ROOT).json()
        if state["update"]["state"] != "starting":
            break
        time.sleep(0.01)
    assert state["update"]["state"] == "failed"
    assert state["connected"]
    assert client.post(ROOT + "/disconnect").json()["connected"] is False


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect", "/refresh"])
def test_foreign_origin_is_rejected(client, action):
    response = client.request(
        "POST" if action else "GET",
        ROOT + action,
        headers={"Origin": "https://evil.test"},
    )
    assert response.status_code == 403
    assert not devin.config_path().exists()


def test_malformed_configuration_is_a_safe_error(client):
    path = devin.config_path()
    path.parent.mkdir(parents=True)
    path.write_text('{"agents":[{"id":"fcc-opencode","distribution":{}}]}')
    before = path.read_bytes()
    assert client.post(ROOT + "/connect").status_code == 400
    assert path.read_bytes() == before

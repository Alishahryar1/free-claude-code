import json
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.harnesses import jetbrains_acp_integration as jb
from tests.api.support import create_test_app
from tests.harnesses.test_jetbrains_acp_integration import install

ROOT = "/admin/api/integrations/jetbrains-acp"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jb, "config_path", lambda: tmp_path / "acp.json")
    monkeypatch.setattr(jb, "registry_path", lambda: tmp_path / "installed.json")
    monkeypatch.setattr(jb, "system_root", lambda: tmp_path / "systems")
    monkeypatch.setattr(
        jb.subprocess,
        "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 0, "v24.19.0", ""),
    )
    with TestClient(
        create_test_app(Settings(proxy_auth_token="secret")),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as client:
        yield client


def test_api_connect_failure_retry_disconnect_and_safe_status(client):
    status = client.get(ROOT)
    assert status.status_code == 200
    assert status.json()["connected"] is False
    assert status.headers["cache-control"] == "no-store"
    assert client.post(ROOT + "/connect").status_code == 400
    install(jb.registry_path(), jb.system_root())
    response = client.post(ROOT + "/connect")
    assert response.status_code == 200
    assert response.json()["connected"] is True
    assert "secret" not in response.text
    before = jb.config_path().read_bytes()
    assert client.get(ROOT).json()["connected"] is True
    assert jb.config_path().read_bytes() == before
    jb.registry_path().unlink()
    assert client.post(ROOT + "/refresh").status_code == 200
    # Wait for the background result; a failure must not block Disconnect.
    for _ in range(100):
        state = client.get(ROOT).json()
        if state["update"]["state"] != "starting":
            break
        time.sleep(0.01)
    assert state["update"]["state"] == "failed"
    assert state["connected"] is True
    assert client.post(ROOT + "/disconnect").json()["connected"] is False
    assert json.loads(jb.config_path().read_text()) == {"agent_servers": {}}


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect", "/refresh"])
def test_routes_reject_other_origins(client, action):
    response = client.request(
        "POST" if action else "GET",
        ROOT + action,
        headers={"Origin": "https://evil.test"},
    )
    assert response.status_code == 403
    assert not jb.config_path().exists()

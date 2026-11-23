import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.harnesses import jetbrains_acp_integration as jb
from tests.api.support import create_test_app
from tests.harnesses.test_jetbrains_acp_integration import install
from tests.wire_payload import object_payload, response_object

ROOT = "/admin/api/integrations/jetbrains-acp"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(jb, "config_path", lambda: tmp_path / "acp.json")
    monkeypatch.setattr(jb, "registry_path", lambda: tmp_path / "installed.json")
    monkeypatch.setattr(jb, "system_root", lambda: tmp_path / "systems")

    def run_probe(
        args: Sequence[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "v24.19.0", "")

    monkeypatch.setattr(jb.subprocess, "run", run_probe)
    with TestClient(
        create_test_app(Settings(proxy_auth_token="secret")),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as client:
        yield client


def test_api_connect_failure_retry_disconnect_and_safe_status(
    client: TestClient,
) -> None:
    status = client.get(ROOT)
    assert status.status_code == 200
    assert response_object(status)["connected"] is False
    assert status.headers["cache-control"] == "no-store"
    assert client.post(ROOT + "/connect").status_code == 400
    install(jb.registry_path(), jb.system_root())
    response = client.post(ROOT + "/connect")
    assert response.status_code == 200
    assert response_object(response)["connected"] is True
    assert "secret" not in response.text
    before = jb.config_path().read_bytes()
    assert response_object(client.get(ROOT))["connected"] is True
    assert jb.config_path().read_bytes() == before
    jb.registry_path().unlink()
    assert client.post(ROOT + "/refresh").status_code == 200
    # Wait for the background result; a failure must not block Disconnect.
    for _ in range(100):
        state = response_object(client.get(ROOT))
        if object_payload(state["update"])["state"] != "starting":
            break
        time.sleep(0.01)
    assert object_payload(state["update"])["state"] == "failed"
    assert state["connected"] is True
    assert response_object(client.post(ROOT + "/disconnect"))["connected"] is False
    assert json.loads(jb.config_path().read_text()) == {"agent_servers": {}}


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect", "/refresh"])
def test_routes_reject_other_origins(client: TestClient, action: str) -> None:
    response = client.request(
        "POST" if action else "GET",
        ROOT + action,
        headers={"Origin": "https://evil.test"},
    )
    assert response.status_code == 403
    assert not jb.config_path().exists()

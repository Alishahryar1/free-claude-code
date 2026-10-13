import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from free_claude_code.harnesses import claude_integration, claude_onboarding
from tests.api.support import create_test_app, runtime_for_app

ROOT = "/admin/api/integrations/claude-onboarding"
VSCODE = "/admin/api/integrations/claude-vscode"


@pytest.fixture
def onboarding():
    app = create_test_app()
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client, runtime_for_app(app), claude_onboarding.state_path()


def test_passive_status_and_independent_completion(onboarding):
    client, runtime, path = onboarding
    runtime.provider_manager.wait_for_catalog = AsyncMock(
        side_effect=AssertionError("No model discovery needed")
    )
    runtime._integrations._claude_update.state = "failed"
    runtime._integrations._claude_update.message = "VS Code unavailable"
    result = client.get(ROOT)
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    assert result.json() == {
        "completed": False,
        "paths": {"claude_state": str(path.resolve())},
    }
    assert not path.exists()
    response = client.post(f"{ROOT}/complete")
    assert response.status_code == 200
    assert response.json() == {"completed": True, "changed": True}
    assert response.headers["cache-control"] == "no-store"
    assert json.loads(path.read_text())["hasCompletedOnboarding"] is True
    assert not claude_integration.settings_path().exists()
    assert client.get(ROOT).json()["completed"] is True
    before = path.stat().st_mtime_ns
    assert client.post(f"{ROOT}/complete").json() == {
        "completed": True,
        "changed": False,
    }
    assert path.stat().st_mtime_ns == before
    assert runtime._integrations._claude_update.state == "failed"
    assert runtime._integrations._claude_update.message == "VS Code unavailable"
    runtime.provider_manager.wait_for_catalog.assert_not_awaited()


@pytest.mark.parametrize("action", ["", "/complete"])
def test_invalid_onboarding_is_scoped_and_does_not_change_connection(
    onboarding, action
):
    client, _, path = onboarding
    assert client.post(f"{VSCODE}/connect").status_code == 200
    source = '{"onboarding-secret": invalid}'
    path.write_text(source)
    response = client.request("POST" if action else "GET", ROOT + action)
    assert response.status_code == 400
    assert response.headers["cache-control"] == "no-store"
    assert "onboarding-secret" not in response.text
    assert ".claude.json" in response.json()["detail"]
    assert path.read_text() == source
    assert client.get(VSCODE).json()["connected"] is True


@pytest.mark.parametrize("action", ["", "/complete"])
@pytest.mark.parametrize(
    "headers", [{"Host": "evil.test"}, {"Origin": "https://evil.test"}]
)
def test_loopback_admin_guard(onboarding, action, headers):
    client, _, path = onboarding
    assert (
        client.request(
            "POST" if action else "GET", ROOT + action, headers=headers
        ).status_code
        == 403
    )
    assert not path.exists()


def test_remote_client_is_rejected(onboarding):
    client, _, path = onboarding
    with TestClient(
        client.app, base_url="http://127.0.0.1", client=("192.0.2.1", 50000)
    ) as remote:
        assert remote.get(ROOT).status_code == 403
        assert remote.post(f"{ROOT}/complete").status_code == 403
    assert not path.exists()


def test_pending_restart_and_shutdown_do_not_write(onboarding):
    client, runtime, path = onboarding
    runtime._pending_fields = ["PORT"]
    assert client.post(f"{ROOT}/complete").status_code == 503
    runtime._pending_fields = []
    runtime.begin_shutdown()
    assert client.post(f"{ROOT}/complete").status_code == 503
    assert not path.exists()


@pytest.mark.parametrize("action", ["", "/complete"])
def test_io_failure_returns_safe_error(onboarding, monkeypatch, action):
    client, _, path = onboarding

    def fail(*args):
        raise PermissionError("onboarding-secret")

    monkeypatch.setattr(claude_onboarding, "complete" if action else "status", fail)
    response = client.request("POST" if action else "GET", ROOT + action)
    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"
    assert "onboarding-secret" not in response.text
    assert not path.exists()

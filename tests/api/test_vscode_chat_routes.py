import json

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.settings import Settings
from free_claude_code.harnesses import vscode_chat_integration as vscode
from tests.api.support import create_test_app, runtime_for_app

ROOT = "/admin/api/integrations/vscode-chat"


@pytest.fixture
def integration():
    app = create_test_app(
        Settings(host="0.0.0.0", port=4321, proxy_auth_token="integration-secret")
    )
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client, vscode.config_path(), runtime_for_app(app)


def test_connect_disconnect_and_doctor_status(integration):
    from free_claude_code.runtime.diagnostics import integration_report

    client, path, runtime = integration
    result = client.get(ROOT)
    assert result.json()["paths"]["vscode_models"] == str(path.resolve())
    assert not path.exists()
    result = client.post(ROOT + "/connect")
    assert result.status_code == 200
    assert result.json()["connected"]
    assert result.headers["cache-control"] == "no-store"
    assert "integration-secret" not in result.text
    group = json.loads(path.read_text())[0]
    # VS Code resolves group apiKey through its secret store, even for plain text.
    # Explicit model headers reach the Messages endpoint without that lookup.
    assert "apiKey" not in group
    headers = group["models"][0]["requestHeaders"]
    assert headers == {"x-api-key": "integration-secret"}
    assert client.head("/v1/messages", headers=headers).status_code == 204
    assert group["models"][0]["url"] == "http://127.0.0.1:4321/v1/messages"
    assert integration_report(runtime.settings)["vscode-chat"] == {
        "status": "available",
        "connected": True,
    }
    assert not client.post(ROOT + "/disconnect").json()["connected"]


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect", "/refresh"])
@pytest.mark.parametrize(
    "headers", [{"Host": "evil.test"}, {"Origin": "https://evil.test"}]
)
def test_local_admin_security(integration, action, headers):
    client, path, _ = integration
    assert (
        client.request(
            "GET" if not action else "POST", ROOT + action, headers=headers
        ).status_code
        == 403
    )
    assert not path.exists()


def test_bad_configuration_errors_do_not_leak_credentials(integration):
    client, path, _ = integration
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"integration-secret":bad}')
    for action in ("", "/connect", "/disconnect"):
        response = client.request("GET" if not action else "POST", ROOT + action)
        assert response.status_code == 400
        assert "integration-secret" not in response.text
        assert response.headers["cache-control"] == "no-store"
    assert path.read_text() == '{"integration-secret":bad}'


def test_pending_restart_and_shutdown_reject_writes(integration):
    client, path, runtime = integration
    runtime._pending_fields = ["PORT"]
    assert client.post(ROOT + "/connect").status_code == 503
    runtime._pending_fields = []
    runtime.begin_shutdown()
    assert client.post(ROOT + "/disconnect").status_code == 503
    assert not path.exists()

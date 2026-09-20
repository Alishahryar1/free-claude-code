import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import desktop_model_id
from free_claude_code.harnesses import claude_desktop_integration as desktop
from tests.api.support import create_test_app

ROOT = "/admin/api/integrations/claude-desktop"


@pytest.fixture
def client():
    app = create_test_app(
        Settings(host="0.0.0.0", port=4321, proxy_auth_token="desktop-secret")
    )
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client


def test_desktop_routes_connect_disconnect_and_hide_credentials(client):
    root = desktop.config_root()
    response = client.get(ROOT)
    assert response.json()["connected"] is False
    assert response.headers["cache-control"] == "no-store"
    assert not root.exists()
    response = client.post(ROOT + "/connect")
    assert response.status_code == 200
    assert response.json()["connected"] is True
    assert "desktop-secret" not in response.text
    profile = root / "configLibrary" / f"{desktop.FCC_ID}.json"
    assert (
        json.loads(profile.read_text())["inferenceGatewayBaseUrl"]
        == "http://127.0.0.1:4321"
    )
    before = {path: path.stat().st_mtime_ns for path in root.rglob("*.json")}
    assert client.get(ROOT).json()["connected"] is True
    assert before == {path: path.stat().st_mtime_ns for path in root.rglob("*.json")}
    assert client.post(ROOT + "/disconnect").json()["connected"] is False
    assert not profile.exists()


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect", "/refresh"])
def test_desktop_admin_routes_reject_other_origins(client, action):
    response = client.request(
        "POST" if action else "GET",
        ROOT + action,
        headers={"Origin": "https://evil.test"},
    )
    assert response.status_code == 403
    assert not desktop.config_root().exists()


def test_managed_desktop_error_is_safe(client, monkeypatch):
    def managed():
        raise desktop.ManagedDesktopError("private-policy-secret")

    monkeypatch.setattr(desktop, "check_unmanaged", managed)
    response = client.post(ROOT + "/connect")
    assert response.status_code == 400
    assert "private-policy-secret" not in response.text
    assert "organization" in response.json()["detail"]


def test_desktop_model_view_preserves_labels_and_query_precedence():
    ref = "nvidia_nim/nemotron-3.5"
    settings = Settings().model_copy(
        update={
            "model": ref,
            "model_opus": None,
            "model_haiku": None,
            "model_fallbacks": (),
            "proxy_auth_enabled": False,
        }
    )
    with TestClient(create_test_app(settings)) as client:
        headers = {"X-FCC-Model-View": "claude-desktop"}
        payload = client.get("/v1/models", headers=headers).json()
        models = [row for row in payload["data"] if "/" in row["id"]]
        assert [row["id"] for row in models] == [
            desktop_model_id(ref),
            desktop_model_id(ref, no_thinking=True),
        ]
        assert [row["display_name"] for row in models] == [ref, ref + " (no thinking)"]
        regular = client.get("/v1/models").json()
        assert client.get("/v1/models?view=claude", headers=headers).json() == regular
        assert (
            client.get(
                "/v1/models", headers={"X-FCC-Model-View": "invalid"}
            ).status_code
            == 422
        )
    route = ModelRouter(settings).resolve(models[0]["id"])
    assert route.primary.provider_model_ref == ref
    assert (
        ModelRouter(settings).resolve(models[1]["id"]).reasoning_preference.value
        == "off"
    )


@pytest.mark.parametrize(
    "model",
    ["claude-fcc/", "claude-fcc/zz", "claude-fcc", desktop_model_id("unknown/model")],
)
def test_invalid_desktop_model_never_routes_to_default(model):
    with pytest.raises(InvalidRequestError):
        ModelRouter(Settings()).resolve(model)


@pytest.mark.parametrize("no_thinking", [False, True])
def test_desktop_token_count_routes_to_original_provider(no_thinking):
    ref = "groq/desktop-model"
    app = create_test_app(Settings(proxy_auth_enabled=False))
    with (
        TestClient(app) as client,
        patch("free_claude_code.api.routes.get_token_count", return_value=7),
        patch("free_claude_code.api.handlers.token_count.trace_event") as trace,
    ):
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": desktop_model_id(ref, no_thinking=no_thinking),
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert response.status_code == 200
    assert response.json()["input_tokens"] == 7
    routed = next(
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs["stage"] == "routing"
    )
    assert routed["provider_model_ref"] == ref

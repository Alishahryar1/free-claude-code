import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import desktop_model_id
from free_claude_code.harnesses import claude_desktop_integration as desktop
from tests.api.support import create_test_app
from tests.wire_payload import (
    array_payload,
    decode_object,
    object_payload,
    response_object,
    response_payload,
    string_payload,
)

ROOT = "/admin/api/integrations/claude-desktop"


def test_pending_disconnect_survives_runtime_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(host="127.0.0.1", port=4321, proxy_auth_token="original")
    root = desktop.config_root()
    profile = root / "configLibrary" / f"{desktop.FCC_ID}.json"
    unlink = Path.unlink

    def fail(path: Path, missing_ok: bool = False):
        if path == profile:
            raise PermissionError("test")
        return unlink(path, missing_ok=missing_ok)

    with TestClient(
        create_test_app(settings),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as client:
        assert client.post(ROOT + "/connect").status_code == 200
        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail)
            assert client.post(ROOT + "/disconnect").status_code == 503
        assert response_object(client.get(ROOT))["disconnect_pending"] is True
    before = profile.read_bytes()
    with TestClient(
        create_test_app(settings.model_copy(update={"proxy_auth_token": "rotated"})),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as client:
        status = response_object(client.get(ROOT))
        assert status["disconnect_pending"] is True
        assert profile.read_bytes() == before
        refused = client.post(ROOT + "/connect")
        assert refused.status_code == 400
        assert "Finish disconnecting" in string_payload(
            response_object(refused)["detail"]
        )
        result = response_object(client.post(ROOT + "/disconnect"))
        assert result["connected"] is False
        assert result["disconnect_pending"] is False
        assert not profile.exists()


@pytest.fixture
def client():
    app = create_test_app(
        Settings(host="0.0.0.0", port=4321, proxy_auth_token="desktop-secret")
    )
    with TestClient(
        app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
    ) as client:
        yield client


def test_desktop_routes_connect_disconnect_and_hide_credentials(
    client: TestClient,
) -> None:
    root = desktop.config_root()
    response = client.get(ROOT)
    assert response_object(response)["connected"] is False
    assert response.headers["cache-control"] == "no-store"
    assert not root.exists()
    response = client.post(ROOT + "/connect")
    assert response.status_code == 200
    assert response_object(response)["connected"] is True
    assert "desktop-secret" not in response.text
    profile = root / "configLibrary" / f"{desktop.FCC_ID}.json"
    assert (
        decode_object(profile.read_text())["inferenceGatewayBaseUrl"]
        == "http://127.0.0.1:4321"
    )
    config = decode_object(profile.read_text())
    models = [desktop_model_id("nvidia_nim/one")]
    config["inferenceModels"] = models
    profile.write_text(json.dumps(config))
    before = {path: path.stat().st_mtime_ns for path in root.rglob("*.json")}
    response = client.get(ROOT)
    assert response_object(response)["connected"] is True
    assert "desktop-secret" not in response.text
    assert decode_object(profile.read_text())["inferenceModels"] == models
    assert before == {path: path.stat().st_mtime_ns for path in root.rglob("*.json")}
    assert response_object(client.post(ROOT + "/disconnect"))["connected"] is False
    assert not profile.exists()


@pytest.mark.parametrize("action", ["", "/connect", "/disconnect", "/refresh"])
def test_desktop_admin_routes_reject_other_origins(
    client: TestClient, action: str
) -> None:
    response = client.request(
        "POST" if action else "GET",
        ROOT + action,
        headers={"Origin": "https://evil.test"},
    )
    assert response.status_code == 403
    assert not desktop.config_root().exists()


def test_managed_desktop_error_is_safe(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def managed():
        raise desktop.ManagedDesktopError("private-policy-secret")

    monkeypatch.setattr(desktop, "check_unmanaged", managed)
    response = client.post(ROOT + "/connect")
    assert response.status_code == 400
    assert "private-policy-secret" not in response.text
    assert "organization" in response.json()["detail"]


def test_desktop_model_view_preserves_labels_and_query_precedence() -> None:
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
        payload = response_object(client.get("/v1/models", headers=headers))
        models = [
            row
            for row in array_payload(payload["data"])
            if "/" in string_payload(object_payload(row)["id"])
        ]
        assert [object_payload(row)["id"] for row in models] == [
            desktop_model_id(ref),
            desktop_model_id(ref, no_thinking=True),
        ]
        assert [object_payload(row)["display_name"] for row in models] == [
            ref,
            ref + " (no thinking)",
        ]
        regular = response_payload(client.get("/v1/models"))
        assert (
            response_payload(client.get("/v1/models?view=claude", headers=headers))
            == regular
        )
        assert (
            client.get(
                "/v1/models", headers={"X-FCC-Model-View": "invalid"}
            ).status_code
            == 422
        )
    route = ModelRouter(settings).resolve(
        string_payload(object_payload(models[0])["id"])
    )
    assert route.primary.provider_model_ref == ref
    assert (
        ModelRouter(settings)
        .resolve(string_payload(object_payload(models[1])["id"]))
        .reasoning_preference.value
        == "off"
    )


@pytest.mark.parametrize(
    "model",
    ["claude-fcc/", "claude-fcc/zz", "claude-fcc", desktop_model_id("unknown/model")],
)
def test_invalid_desktop_model_never_routes_to_default(model: str) -> None:
    with pytest.raises(InvalidRequestError):
        ModelRouter(Settings()).resolve(model)


@pytest.mark.parametrize("no_thinking", [False, True])
def test_desktop_token_count_routes_to_original_provider(no_thinking: bool) -> None:
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


def test_pending_windows_migration_reports_recovery_without_writes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    legacy = desktop.legacy_windows_root()
    assert legacy is not None
    legacy.mkdir()
    history = legacy / "history.json"
    history.write_text('{"chat":"keep"}')
    response = client.post(ROOT + "/connect")
    assert response.status_code == 400
    detail = response_object(response)["detail"]
    assert "Launch Claude Desktop" in string_payload(detail)
    assert "quit" in string_payload(detail)
    assert "retry Connect" in string_payload(detail)
    assert not desktop.config_root().exists()
    assert history.read_text() == '{"chat":"keep"}'

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_settings
from config.settings import Settings


def test_anthropic_auth_token_required_and_accepts_x_api_key():
    client = TestClient(app)
    settings = Settings()
    settings.anthropic_auth_token = "s3cr3t"
    app.dependency_overrides[get_settings] = lambda: settings

    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("api.routes.get_token_count", return_value=1):
        # No header -> 401
        r = client.post("/v1/messages/count_tokens", json=payload)
        assert r.status_code == 401

        # X-API-Key header -> 200
        r = client.post(
            "/v1/messages/count_tokens", json=payload, headers={"X-API-Key": "s3cr3t"}
        )
        assert r.status_code == 200
        assert r.json()["input_tokens"] == 1

    app.dependency_overrides.clear()


def test_anthropic_auth_token_accepts_bearer_authorization():
    client = TestClient(app)
    settings = Settings()
    settings.anthropic_auth_token = "b3artoken"
    app.dependency_overrides[get_settings] = lambda: settings

    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("api.routes.get_token_count", return_value=2):
        # Authorization Bearer -> 200
        r = client.post(
            "/v1/messages/count_tokens",
            json=payload,
            headers={"Authorization": "Bearer b3artoken"},
        )
        assert r.status_code == 200
        assert r.json()["input_tokens"] == 2

    app.dependency_overrides.clear()

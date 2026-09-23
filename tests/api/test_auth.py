from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.api.dependencies import get_settings
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app


@pytest.fixture
def app():
    return create_test_app(Settings())


def test_anthropic_post_routes_accept_x_api_key(app):
    client = TestClient(app)
    settings = Settings(proxy_auth_enabled=True, proxy_auth_token="s3cr3t")
    app.dependency_overrides[get_settings] = lambda: settings

    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with (
        patch("free_claude_code.api.routes.get_token_count", return_value=1),
        patch(
            "free_claude_code.api.routes._create_messages_response",
            new_callable=AsyncMock,
            return_value={"accepted": True},
        ),
    ):
        count_response = client.post(
            "/v1/messages/count_tokens",
            json=payload,
            headers={"X-API-Key": "s3cr3t"},
        )
        messages_response = client.post(
            "/v1/messages",
            json={**payload, "max_tokens": 16},
            headers={"X-API-Key": "s3cr3t"},
        )

    assert count_response.status_code == 200
    assert count_response.json()["input_tokens"] == 1
    assert messages_response.status_code == 200
    assert messages_response.json() == {"accepted": True}

    app.dependency_overrides.clear()


def test_anthropic_probe_routes_accept_x_api_key(app):
    client = TestClient(app)
    settings = Settings(proxy_auth_enabled=True, proxy_auth_token="probe-token")
    app.dependency_overrides[get_settings] = lambda: settings

    for path in ("/v1/messages", "/v1/messages/count_tokens"):
        for method in (client.head, client.options):
            response = method(path, headers={"X-API-Key": "probe-token"})
            assert response.status_code == 204
            assert response.headers["Allow"] == "POST, HEAD, OPTIONS"

    app.dependency_overrides.clear()


def test_anthropic_routes_still_reject_anthropic_auth_token_only(app):
    client = TestClient(app)
    settings = Settings(proxy_auth_enabled=True, proxy_auth_token="s3cr3t")
    app.dependency_overrides[get_settings] = lambda: settings

    for path in ("/v1/messages", "/v1/messages/count_tokens"):
        for method in (client.head, client.options):
            response = method(path, headers={"anthropic-auth-token": "s3cr3t"})
            assert response.status_code == 401

    app.dependency_overrides.clear()


def test_messages_auth_gives_authorization_precedence_over_x_api_key(app):
    client = TestClient(app)
    settings = Settings(proxy_auth_enabled=True, proxy_auth_token="b3artoken")
    app.dependency_overrides[get_settings] = lambda: settings

    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch(
        "free_claude_code.api.routes._create_messages_response",
        new_callable=AsyncMock,
        return_value={"accepted": True},
    ):
        r = client.post(
            "/v1/messages",
            json={**payload, "max_tokens": 16},
            headers={
                "Authorization": "Bearer b3artoken",
                "X-API-Key": "stale-anthropic-key",
                "anthropic-auth-token": "stale-proxy-token",
            },
        )
        assert r.status_code == 200
        assert r.json() == {"accepted": True}

        r = client.post(
            "/v1/messages",
            json={**payload, "max_tokens": 16},
            headers={
                "Authorization": "Bearer wrong",
                "X-API-Key": "b3artoken",
            },
        )
        assert r.status_code == 401
        assert r.json() == {"detail": "Invalid proxy authentication token"}

    app.dependency_overrides.clear()


def test_x_api_key_remains_rejected_on_non_messages_routes(app):
    client = TestClient(app)
    settings = Settings(proxy_auth_enabled=True, proxy_auth_token="route-token")
    app.dependency_overrides[get_settings] = lambda: settings

    for method, path in (
        (client.head, "/v1/responses"),
        (client.get, "/v1/models"),
        (client.get, "/"),
    ):
        response = method(path, headers={"X-API-Key": "route-token"})
        assert response.status_code == 401

    app.dependency_overrides.clear()


def test_proxy_auth_token_normalizes_configured_whitespace(app):
    client = TestClient(app)
    settings = Settings(
        proxy_auth_enabled=True,
        proxy_auth_token="  spaced-token  \n",
    )
    app.dependency_overrides[get_settings] = lambda: settings

    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("free_claude_code.api.routes.get_token_count", return_value=3):
        r = client.post(
            "/v1/messages/count_tokens",
            json=payload,
            headers={"Authorization": "Bearer spaced-token"},
        )
        assert r.status_code == 200
        assert r.json()["input_tokens"] == 3

    app.dependency_overrides.clear()


def test_proxy_auth_token_applies_to_model_catalog_endpoints(app):
    client = TestClient(app)
    settings = Settings(proxy_auth_enabled=True, proxy_auth_token="models-token")
    app.dependency_overrides[get_settings] = lambda: settings

    for path in ("/v1/models", "/muse-code/models"):
        r = client.get(path)
        assert r.status_code == 401
        assert r.headers["x-request-id"] == r.headers["request-id"]
        assert "x-should-retry" not in r.headers

        r = client.get(path, headers={"Authorization": "Bearer models-token"})
        assert r.status_code == 200
        assert "data" in r.json()

    app.dependency_overrides.clear()


def test_root_get_requires_auth_but_root_probes_are_public(app):
    client = TestClient(app)
    settings = Settings(proxy_auth_enabled=True, proxy_auth_token="root-token")
    app.dependency_overrides[get_settings] = lambda: settings

    response = client.get("/")
    assert response.status_code == 401

    head = client.head("/")
    assert head.status_code == 204
    assert head.headers["Allow"] == "GET, HEAD, OPTIONS"

    options = client.options("/")
    assert options.status_code == 204
    assert options.headers["Allow"] == "GET, HEAD, OPTIONS"

    app.dependency_overrides.clear()

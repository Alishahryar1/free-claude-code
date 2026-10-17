from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.api import admin_routes
from free_claude_code.config.admin.state import ConfigValueState
from tests.api.support import create_test_app, runtime_for_app


@pytest.fixture
def local_probe(monkeypatch):
    requests: list[httpx.Request] = []
    async_client = httpx.AsyncClient
    clock = iter((10.0, 10.0125))
    monkeypatch.setattr(admin_routes, "perf_counter", lambda: next(clock))

    def probe(handler, provider_id="lmstudio"):
        def reply(request):
            requests.append(request)
            return handler(request)

        def client(*args, **kwargs):
            return async_client(*args, transport=httpx.MockTransport(reply), **kwargs)

        monkeypatch.setattr(admin_routes.httpx, "AsyncClient", client)
        app = create_test_app()
        response = TestClient(
            app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
        ).get(f"/admin/api/providers/{provider_id}/local-status")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        return response, requests

    return probe


@pytest.mark.parametrize(
    ("provider_id", "env_key", "provider_url", "probe_path"),
    (
        ("lmstudio", "LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1", "/models"),
        ("llamacpp", "LLAMACPP_BASE_URL", "http://127.0.0.1:8080/v1", "/models"),
        ("ollama", "OLLAMA_BASE_URL", "http://127.0.0.1:11434", "/api/tags"),
    ),
)
def test_local_probe_reports_the_requested_endpoint_and_elapsed_time(
    monkeypatch, local_probe, provider_id, env_key, provider_url, probe_path
):
    monkeypatch.setenv(env_key, provider_url)
    response, requests = local_probe(lambda _: httpx.Response(200), provider_id)

    assert [(request.method, str(request.url)) for request in requests] == [
        ("GET", f"{provider_url}{probe_path}")
    ]
    assert response.json() == {
        "provider_id": provider_id,
        "status": "reachable",
        "label": "Reachable",
        "base_url": provider_url,
        "status_code": 200,
        "probe_path": probe_path,
        "latency_ms": pytest.approx(12.5),
        "error_type": None,
    }


@pytest.mark.parametrize("status_code", (200, 204, 302, 401, 404, 429, 503))
def test_local_probe_preserves_http_status_without_returning_response_content(
    local_probe, status_code
):
    private = "private-response-credential-marker"
    response, requests = local_probe(
        lambda _: httpx.Response(
            status_code,
            text=private,
            headers={"x-private-token": private},
        )
    )

    assert len(requests) == 1
    payload = response.json()
    reachable = 200 <= status_code < 300
    assert payload["status"] == ("reachable" if reachable else "offline")
    assert payload["label"] == ("Reachable" if reachable else "Offline")
    assert payload["status_code"] == status_code
    assert payload["error_type"] == (None if reachable else "http_error")
    assert payload["probe_path"] == "/models"
    assert payload["latency_ms"] == pytest.approx(12.5)
    assert private not in response.text


@pytest.mark.parametrize(
    ("failure", "error_type"),
    (
        (httpx.ReadTimeout, "timeout"),
        (httpx.ConnectError, "connection_error"),
        (httpx.ProxyError, "connection_error"),
        (httpx.InvalidURL, "invalid_url"),
        (httpx.UnsupportedProtocol, "invalid_url"),
        (httpx.ReadError, "transport_error"),
        (RuntimeError, "unexpected_error"),
    ),
)
def test_local_probe_returns_safe_failure_categories(local_probe, failure, error_type):
    private = "CREDENTIAL[unrecognized-format-987654321]"

    def fail(_request):
        raise failure(f"Request failed with {private}")

    response, requests = local_probe(fail)

    assert len(requests) == 1
    payload = response.json()
    assert payload["status"] == "offline"
    assert payload["label"] == "Offline"
    assert payload["error_type"] == error_type
    assert payload["probe_path"] == "/models"
    assert payload["latency_ms"] == pytest.approx(12.5)
    assert payload["message"]
    assert private not in response.text
    assert failure.__name__ not in response.text


@pytest.mark.parametrize("provider_url", ("", " \t "))
def test_missing_local_url_has_no_network_probe_or_elapsed_time(provider_url):
    app = create_test_app()
    runtime = runtime_for_app(app)
    values = {
        "LM_STUDIO_BASE_URL": ConfigValueState(value=provider_url, source="default")
    }
    with (
        patch.object(runtime, "admin_values", AsyncMock(return_value=values)),
        patch.object(admin_routes.httpx, "AsyncClient") as client,
        patch.object(admin_routes, "perf_counter") as clock,
    ):
        response = TestClient(
            app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)
        ).get("/admin/api/providers/lmstudio/local-status")

    assert response.status_code == 200
    assert response.json() == {
        "provider_id": "lmstudio",
        "status": "missing_url",
        "label": "Missing URL",
        "base_url": provider_url,
        "probe_path": "/models",
        "latency_ms": None,
        "error_type": None,
    }
    client.assert_not_called()
    clock.assert_not_called()


def test_remote_admin_cannot_trigger_a_local_provider_probe():
    with patch.object(admin_routes.httpx, "AsyncClient") as client:
        response = TestClient(
            create_test_app(),
            base_url="http://127.0.0.1",
            client=("203.0.113.10", 50000),
        ).get("/admin/api/providers/lmstudio/local-status")

    assert response.status_code == 403
    client.assert_not_called()

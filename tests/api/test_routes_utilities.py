from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.api.ports import ApiServices
from free_claude_code.application.ports import StopResult
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app
from tests.web_tools_support import StubWebToolsClient


@pytest.fixture
def app():
    return create_test_app(Settings())


@pytest.fixture
def client(app):
    return TestClient(app)


def test_create_message_empty_messages_returns_400(client):
    """POST /v1/messages with messages: [] returns 400 invalid_request_error."""
    payload = {
        "model": "claude-3-sonnet",
        "max_tokens": 100,
        "messages": [],
    }
    response = client.post("/v1/messages", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data.get("type") == "error"
    assert data.get("error", {}).get("type") == "invalid_request_error"
    assert "cannot be empty" in data.get("error", {}).get("message", "")


def test_count_tokens_empty_messages_returns_400(client):
    """POST /v1/messages/count_tokens with messages: [] matches messages validation."""
    payload = {"model": "claude-3-sonnet", "messages": []}
    response = client.post("/v1/messages/count_tokens", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data.get("type") == "error"
    assert data.get("error", {}).get("type") == "invalid_request_error"
    assert "cannot be empty" in data.get("error", {}).get("message", "")


def test_count_tokens_endpoint(client):
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("free_claude_code.api.routes.get_token_count", return_value=5):
        response = client.post("/v1/messages/count_tokens", json=payload)

    assert response.status_code == 200
    assert response.json()["input_tokens"] == 5


@pytest.mark.parametrize("prefix", ["", "anthropic/", "claude-3-freecc-no-thinking/"])
def test_count_tokens_retired_model_uses_configured_default(prefix):
    test_app = create_test_app(
        Settings(MODEL="groq/default", MODEL_OPUS="deepseek/opus")
    )
    with (
        patch("free_claude_code.api.routes.get_token_count", return_value=5),
        patch("free_claude_code.api.handlers.token_count.trace_event") as trace,
    ):
        response = TestClient(test_app).post(
            "/v1/messages/count_tokens",
            json={
                "model": f"{prefix}github_models/vendor/opus",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    assert response.json()["input_tokens"] == 5
    routed = next(
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs["stage"] == "routing"
    )
    assert routed["provider_model_ref"] == "groq/default"
    assert routed["gateway_model"] == f"{prefix}github_models/vendor/opus"


def test_count_tokens_error_returns_500(client):
    """When get_token_count raises, count_tokens returns 500."""
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch(
        "free_claude_code.api.routes.get_token_count",
        side_effect=RuntimeError("token error"),
    ):
        response = client.post("/v1/messages/count_tokens", json=payload)

    assert response.status_code == 500
    assert "token error" in response.json()["detail"]


def test_stop_cli_with_messaging_workflow(app, client):
    session_control = MagicMock()
    session_control.stop_all = AsyncMock(return_value=StopResult(cancelled_count=3))
    services = app.state.services
    app.state.services = ApiServices(
        requests=services.requests,
        admin=services.admin,
        tasks=session_control,
        web_tools=StubWebToolsClient(),
    )

    response = client.post("/stop")

    assert response.status_code == 200
    assert response.json()["cancelled_count"] == 3
    session_control.stop_all.assert_awaited_once()


def test_stop_cli_fallback_to_manager(app, client):
    session_control = MagicMock()
    session_control.stop_all = AsyncMock(return_value=StopResult(source="cli_manager"))
    services = app.state.services
    app.state.services = ApiServices(
        requests=services.requests,
        admin=services.admin,
        tasks=session_control,
        web_tools=StubWebToolsClient(),
    )

    response = client.post("/stop")

    assert response.status_code == 200
    assert response.json()["source"] == "cli_manager"
    session_control.stop_all.assert_awaited_once()

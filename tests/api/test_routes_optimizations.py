from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_settings
from config.settings import Settings


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_settings():
    settings = Settings()
    settings.fast_prefix_detection = True
    settings.enable_network_probe_mock = True
    settings.enable_title_generation_skip = True
    return settings


def test_create_message_fast_prefix_detection(client, mock_settings):
    app.dependency_overrides[get_settings] = lambda: mock_settings

    payload = {
        "model": "claude-3-sonnet",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "What is the prefix?"}],
    }

    with (
        patch(
            "api.optimization_handlers.is_prefix_detection_request",
            return_value=(True, "/ask"),
        ),
        patch(
            "api.optimization_handlers.extract_command_prefix",
            return_value="/ask",
        ),
    ):
        response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "/ask" in data["content"][0]["text"]

    app.dependency_overrides.clear()


def test_create_message_quota_check_mock(client, mock_settings):
    app.dependency_overrides[get_settings] = lambda: mock_settings

    payload = {
        "model": "claude-3-sonnet",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "quota check"}],
    }

    with patch("api.optimization_handlers.is_quota_check_request", return_value=True):
        response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    assert "Quota check passed" in response.json()["content"][0]["text"]

    app.dependency_overrides.clear()


def test_create_message_title_generation_skip(client, mock_settings):
    app.dependency_overrides[get_settings] = lambda: mock_settings

    payload = {
        "model": "claude-3-sonnet",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "generate title"}],
    }

    with patch(
        "api.optimization_handlers.is_title_generation_request", return_value=True
    ):
        response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    assert "Conversation" in response.json()["content"][0]["text"]

    app.dependency_overrides.clear()


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


def test_count_tokens_endpoint(client):
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("api.routes.get_token_count", return_value=5):
        response = client.post("/v1/messages/count_tokens", json=payload)

    assert response.status_code == 200
    assert response.json()["input_tokens"] == 5


def test_count_tokens_error_returns_500(client):
    """When get_token_count raises, count_tokens returns 500."""
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("api.routes.get_token_count", side_effect=RuntimeError("token error")):
        response = client.post("/v1/messages/count_tokens", json=payload)

    assert response.status_code == 500
    assert "token error" in response.json()["detail"]


# NOTE: /stop endpoint is disabled — see api/routes.py for details.
# def test_stop_cli_with_handler(client):
#     mock_handler = MagicMock()
#     mock_handler.stop_all_tasks = AsyncMock(return_value=3)
#     app.state.message_handler = mock_handler
#
#     response = client.post("/stop")
#
#     assert response.status_code == 200
#     assert response.json()["cancelled_count"] == 3
#     mock_handler.stop_all_tasks.assert_called_once()
#
#     if hasattr(app.state, "message_handler"):
#         del app.state.message_handler
#
#
# def test_stop_cli_fallback_to_manager(client):
#     if hasattr(app.state, "message_handler"):
#         del app.state.message_handler
#
#     mock_manager = MagicMock()
#     mock_manager.stop_all = AsyncMock()
#     app.state.cli_manager = mock_manager
#
#     response = client.post("/stop")
#
#     assert response.status_code == 200
#     assert response.json()["source"] == "cli_manager"
#     mock_manager.stop_all.assert_called_once()
#
#     if hasattr(app.state, "cli_manager"):
#         del app.state.cli_manager

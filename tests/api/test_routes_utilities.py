from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.applications import FastAPI
from fastapi.testclient import TestClient

from free_claude_code.api.dependencies import app_services
from free_claude_code.application.ports import StopResult
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject
from tests.api.support import create_test_app
from tests.wire_payload import object_payload, response_object, string_payload


@pytest.fixture
def app():
    return create_test_app(Settings())


@pytest.fixture
def client(app: FastAPI):
    return TestClient(app)


def test_create_message_empty_messages_returns_400(client: TestClient) -> None:
    """POST /v1/messages with messages: [] returns 400 invalid_request_error."""
    payload: JsonObject = {
        "model": "claude-3-sonnet",
        "max_tokens": 100,
        "messages": [],
    }
    response = client.post("/v1/messages", json=payload)
    assert response.status_code == 400
    data = response_object(response)
    assert data.get("type") == "error"
    assert object_payload(data.get("error", {})).get("type") == "invalid_request_error"
    assert "cannot be empty" in string_payload(
        object_payload(data.get("error", {})).get("message", "")
    )


def test_count_tokens_empty_messages_returns_400(client: TestClient) -> None:
    """POST /v1/messages/count_tokens with messages: [] matches messages validation."""
    payload: JsonObject = {"model": "claude-3-sonnet", "messages": []}
    response = client.post("/v1/messages/count_tokens", json=payload)
    assert response.status_code == 400
    data = response_object(response)
    assert data.get("type") == "error"
    assert object_payload(data.get("error", {})).get("type") == "invalid_request_error"
    assert "cannot be empty" in string_payload(
        object_payload(data.get("error", {})).get("message", "")
    )


def test_count_tokens_endpoint(client: TestClient) -> None:
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "hello"}],
    }

    with patch("free_claude_code.api.routes.get_token_count", return_value=5):
        response = client.post("/v1/messages/count_tokens", json=payload)

    assert response.status_code == 200
    assert response.json()["input_tokens"] == 5


@pytest.mark.parametrize("prefix", ["", "anthropic/", "claude-3-freecc-no-thinking/"])
@pytest.mark.parametrize(
    "ref,target",
    [
        ("github_models/vendor/opus", "groq/default"),
        (
            "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
            "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
        ),
    ],
)
def test_count_tokens_retired_model_uses_replacement(
    prefix: str, ref: str, target: str
) -> None:
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
                "model": f"{prefix}{ref}",
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
    assert routed["provider_model_ref"] == target
    assert routed["gateway_model"] == f"{prefix}{ref}"


def test_count_tokens_error_returns_500(client: TestClient) -> None:
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


def test_stop_cli_with_messaging_workflow(app: FastAPI, client: TestClient) -> None:
    session_control = MagicMock()
    session_control.stop_all = AsyncMock(return_value=StopResult(cancelled_count=3))
    app.state.services = replace(app_services(app), tasks=session_control)

    response = client.post("/stop")

    assert response.status_code == 200
    assert response_object(response)["cancelled_count"] == 3
    session_control.stop_all.assert_awaited_once()


def test_stop_cli_fallback_to_manager(app: FastAPI, client: TestClient) -> None:
    session_control = MagicMock()
    session_control.stop_all = AsyncMock(return_value=StopResult(source="cli_manager"))
    app.state.services = replace(app_services(app), tasks=session_control)

    response = client.post("/stop")

    assert response.status_code == 200
    assert response_object(response)["source"] == "cli_manager"
    session_control.stop_all.assert_awaited_once()

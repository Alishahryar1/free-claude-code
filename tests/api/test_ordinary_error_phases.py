"""Ordinary ingress, routing, readiness, and validation error contracts."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.errors import (
    ApplicationError,
    ApplicationUnavailableError,
    InvalidRequestError,
    UnknownProviderError,
)
from free_claude_code.config.settings import Settings
from free_claude_code.providers.open_router import OpenRouterProvider
from tests.api.support import create_test_app
from tests.providers.support import immediate_admission, make_provider_config

_PRODUCT_REQUESTS = (
    (
        "messages",
        "/v1/messages",
        {
            "model": "open_router/test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ),
    (
        "responses",
        "/v1/responses",
        {
            "model": "open_router/test-model",
            "input": "hello",
        },
    ),
)


def _settings(**updates: object) -> Settings:
    return Settings().model_copy(
        update={
            "proxy_auth_enabled": False,
            "log_api_error_tracebacks": False,
            **updates,
        }
    )


@pytest.mark.parametrize("mode", ["messages_stream", "messages_nonstream", "responses"])
def test_conversion_error_returns_http_400_without_generation(mode):
    payload = {"model": "open_router/test-model"}
    if mode == "responses":
        payload["input"] = [
            {"type": "reasoning", "encrypted_content": "fcc:history:v2:unsupported"}
        ]
        path = "/v1/responses"
    else:
        payload["stream"] = mode == "messages_stream"
        payload["messages"] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "redacted_thinking", "data": "fcc:history:v2:unsupported"}
                ],
            }
        ]
        path = "/v1/messages"
    sdk = MagicMock()
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI", return_value=sdk
    ):
        provider = OpenRouterProvider(
            make_provider_config(
                api_key="test", base_url="https://provider.invalid/v1"
            ),
            admission=immediate_admission(),
        )
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(create_test_app(_settings())) as client,
    ):
        response = client.post(path, json=payload)
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "replay" in response.json()["error"]["message"]
    assert "x-should-retry" not in response.headers
    sdk.chat.completions.create.assert_not_called()


def _assert_ordinary_protocol_error(
    response: httpx.Response,
    *,
    wire_api: str,
    status_code: int,
    error_type: str,
    message: str,
) -> None:
    assert response.status_code == status_code
    assert response.headers["content-type"].startswith("application/json")
    assert "x-should-retry" not in response.headers

    request_id = response.headers["request-id"]
    assert request_id.startswith("req_")
    if wire_api == "responses":
        assert response.headers["x-request-id"] == request_id
        assert response.json() == {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        }
        return

    assert "x-request-id" not in response.headers
    assert response.json() == {
        "type": "error",
        "error": {"type": error_type, "message": message},
        "request_id": request_id,
    }


@pytest.mark.parametrize(
    "error_type",
    [InvalidRequestError, UnknownProviderError, ApplicationUnavailableError],
)
def test_application_errors_share_one_protocol_neutral_base(
    error_type: type[ApplicationError],
) -> None:
    assert isinstance(error_type("failure"), ApplicationError)


@pytest.mark.parametrize(
    ("wire_api", "path", "payload"),
    _PRODUCT_REQUESTS,
    ids=("messages", "responses"),
)
def test_missing_provider_credential_is_protocol_specific_503_without_terminal_header(
    wire_api: str,
    path: str,
    payload: dict[str, object],
) -> None:
    message = (
        "OPENROUTER_API_KEY is not set. Add it in the Admin UI. "
        "Get a key at https://openrouter.ai/keys"
    )
    app = create_test_app(
        _settings(
            model="open_router/test-model",
            open_router_api_key="",
        )
    )

    with TestClient(app) as client:
        response = client.post(path, json=payload)

    _assert_ordinary_protocol_error(
        response,
        wire_api=wire_api,
        status_code=503,
        error_type="api_error",
        message=message,
    )


@pytest.mark.parametrize(
    ("wire_api", "path", "payload"),
    _PRODUCT_REQUESTS,
    ids=("messages", "responses"),
)
def test_runtime_acquisition_failure_is_protocol_specific_503_without_terminal_header(
    wire_api: str,
    path: str,
    payload: dict[str, object],
) -> None:
    message = "Provider runtime is shutting down."
    app = create_test_app(_settings())
    acquire = AsyncMock(side_effect=ApplicationUnavailableError(message))

    with (
        patch.object(app.state.services.requests, "acquire", new=acquire),
        TestClient(app) as client,
    ):
        response = client.post(path, json=payload)

    acquire.assert_awaited_once()
    _assert_ordinary_protocol_error(
        response,
        wire_api=wire_api,
        status_code=503,
        error_type="api_error",
        message=message,
    )


@pytest.mark.parametrize(
    ("wire_api", "path", "payload"),
    _PRODUCT_REQUESTS,
    ids=("messages", "responses"),
)
def test_unknown_provider_is_protocol_specific_400_without_terminal_header(
    wire_api: str,
    path: str,
    payload: dict[str, object],
) -> None:
    message = "Unknown provider_type: 'unknown'."
    app = create_test_app(_settings())

    with (
        patch(
            "free_claude_code.api.routes.resolve_provider",
            side_effect=UnknownProviderError(message),
        ),
        TestClient(app) as client,
    ):
        response = client.post(path, json=payload)

    _assert_ordinary_protocol_error(
        response,
        wire_api=wire_api,
        status_code=400,
        error_type="invalid_request_error",
        message=message,
    )


@pytest.mark.parametrize(
    ("wire_api", "path", "payload"),
    _PRODUCT_REQUESTS,
    ids=("messages", "responses"),
)
def test_startup_rejection_is_protocol_specific_400_without_terminal_header(
    wire_api: str,
    path: str,
    payload: dict[str, object],
) -> None:
    message = "bad tool shape"
    provider = MagicMock()
    stream = (
        provider.stream_responses
        if wire_api == "responses"
        else provider.stream_messages
    )
    stream.side_effect = InvalidRequestError(message)
    app = create_test_app(_settings())

    with (
        patch(
            "free_claude_code.api.routes.resolve_provider",
            return_value=provider,
        ),
        TestClient(app) as client,
    ):
        response = client.post(path, json=payload)

    stream = (
        provider.stream_responses
        if wire_api == "responses"
        else provider.stream_messages
    )
    stream.assert_called_once()
    _assert_ordinary_protocol_error(
        response,
        wire_api=wire_api,
        status_code=400,
        error_type="invalid_request_error",
        message=message,
    )


@pytest.mark.parametrize(
    ("wire_api", "path", "payload"),
    _PRODUCT_REQUESTS,
    ids=("messages", "responses"),
)
@pytest.mark.parametrize(
    ("headers", "detail"),
    [
        ({}, "Missing proxy authentication token"),
        (
            {"authorization": "Bearer wrong"},
            "Invalid proxy authentication token",
        ),
    ],
    ids=("missing", "invalid"),
)
def test_proxy_auth_preserves_ingress_detail_contract(
    wire_api: str,
    path: str,
    payload: dict[str, object],
    headers: dict[str, str],
    detail: str,
) -> None:
    app = create_test_app(_settings(proxy_auth_enabled=True, proxy_auth_token="secret"))

    with TestClient(app) as client:
        response = client.post(path, json=payload, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": detail}
    assert response.headers["content-type"].startswith("application/json")
    assert "x-should-retry" not in response.headers
    request_id = response.headers["request-id"]
    assert request_id.startswith("req_")
    if wire_api == "responses":
        assert response.headers["x-request-id"] == request_id
    else:
        assert "x-request-id" not in response.headers

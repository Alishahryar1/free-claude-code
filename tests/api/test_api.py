from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_provider
from providers.nvidia_nim import NvidiaNimProvider

# Mock provider
mock_provider = MagicMock(spec=NvidiaNimProvider)

# Track stream_response calls for test_model_mapping
_stream_response_calls = []


async def _mock_stream_response(*args, **kwargs):
    """Minimal async generator for streaming tests."""
    _stream_response_calls.append((args, kwargs))
    yield "event: message_start\ndata: {}\n\n"
    yield "[DONE]\n\n"


mock_provider.stream_response = _mock_stream_response


def override_get_provider():
    return mock_provider


app.dependency_overrides[get_provider] = override_get_provider
client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_create_message_stream():
    """Create message returns streaming response."""
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": True,
    }
    response = client.post("/v1/messages", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = b"".join(response.iter_bytes())
    assert b"message_start" in content or b"event:" in content


def test_model_mapping():
    # Test Haiku mapping
    _stream_response_calls.clear()
    payload_haiku = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": True,
    }
    client.post("/v1/messages", json=payload_haiku)
    assert len(_stream_response_calls) == 1
    args = _stream_response_calls[0][0]
    assert args[0].model != "claude-3-haiku-20240307"
    assert args[0].original_model == "claude-3-haiku-20240307"


def test_error_fallbacks():
    from providers.exceptions import (
        AuthenticationError,
        OverloadedError,
        RateLimitError,
    )

    base_payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        "stream": True,
    }

    def _raise_auth(*args, **kwargs):
        raise AuthenticationError("Invalid Key")

    def _raise_rate_limit(*args, **kwargs):
        raise RateLimitError("Too Many Requests")

    def _raise_overloaded(*args, **kwargs):
        raise OverloadedError("Server Overloaded")

    # 1. Authentication Error (401)
    mock_provider.stream_response = _raise_auth
    response = client.post("/v1/messages", json=base_payload)
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"

    # 2. Rate Limit (429)
    mock_provider.stream_response = _raise_rate_limit
    response = client.post("/v1/messages", json=base_payload)
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"

    # 3. Overloaded (529)
    mock_provider.stream_response = _raise_overloaded
    response = client.post("/v1/messages", json=base_payload)
    assert response.status_code == 529
    assert response.json()["error"]["type"] == "overloaded_error"

    # Reset for subsequent tests
    mock_provider.stream_response = _mock_stream_response


def test_generic_exception_returns_500():
    """Non-ProviderError exceptions are caught and returned as HTTPException(500)."""

    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("unexpected crash")

    mock_provider.stream_response = _raise_runtime
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    assert response.status_code == 500
    mock_provider.stream_response = _mock_stream_response


def test_generic_exception_with_status_code():
    """Generic exception with status_code attribute uses that status (getattr fallback)."""

    class ExceptionWithStatus(RuntimeError):
        def __init__(self, msg: str, status_code: int = 500):
            super().__init__(msg)
            self.status_code = status_code

    def _raise_with_status(*args, **kwargs):
        raise ExceptionWithStatus("bad gateway", 502)

    mock_provider.stream_response = _raise_with_status
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    assert response.status_code == 502
    mock_provider.stream_response = _mock_stream_response


def test_count_tokens_endpoint():
    """count_tokens endpoint returns token count."""
    response = client.post(
        "/v1/messages/count_tokens",
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert "input_tokens" in response.json()


# NOTE: /stop endpoint is disabled — see api/routes.py for details.
# def test_stop_endpoint_no_handler_no_cli_503():
#     """POST /stop without handler or cli_manager returns 503."""
#     # Ensure no handler or cli_manager on app state
#     if hasattr(app.state, "message_handler"):
#         delattr(app.state, "message_handler")
#     if hasattr(app.state, "cli_manager"):
#         delattr(app.state, "cli_manager")
#     response = client.post("/stop")
#     assert response.status_code == 503

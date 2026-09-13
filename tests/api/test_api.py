from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.config.constants import DEFAULT_MODEL
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from tests.api.support import create_test_app
from tests.providers.support import immediate_admission, make_provider_config


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefix", ["", "anthropic/", "claude-3-freecc-no-thinking/"])
def test_retired_model_ingress_calls_default_provider(client, stream, prefix):
    original = f"{prefix}github_models/vendor/opus"
    _stream_messages_calls.clear()
    response = client.post(
        "/v1/messages",
        json={
            "model": original,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
        },
    )
    assert response.status_code == 200
    assert len(_stream_messages_calls) == 1
    args, kwargs = _stream_messages_calls[0]
    assert args[0].model == DEFAULT_MODEL.split("/", 1)[1]
    assert kwargs["response_model"] == original
    if prefix == "claude-3-freecc-no-thinking/":
        assert kwargs["reasoning"] == ReasoningPolicy.off()


@pytest.fixture
def app():
    return create_test_app(Settings())


# Mock provider
mock_provider = MagicMock(spec=NvidiaNimProvider)

# Track stream_messages calls for test_model_mapping
_stream_messages_calls: list = []


async def _mock_stream_messages(*args, **kwargs):
    """Minimal async generator for streaming tests."""
    _stream_messages_calls.append((args, kwargs))
    yield "event: message_start\ndata: {}\n\n"
    yield "[DONE]\n\n"


async def _mock_pre_start_rate_limit(*args, **kwargs):
    """Provider stream that fails before any downstream-visible SSE chunk."""
    _stream_messages_calls.append((args, kwargs))
    raise ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="upstream is busy",
        retryable=True,
    )
    yield "unreachable"


async def _mock_empty_stream(*args, **kwargs):
    """Provider stream that completes without a protocol frame."""
    _stream_messages_calls.append((args, kwargs))
    if False:
        yield "unreachable"


def _terminal_json_error(response, *, status_code: int):
    assert response.status_code == status_code
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-should-retry"] == "false"
    request_id = response.headers["request-id"]
    payload = response.json()
    assert payload["request_id"] == request_id
    return payload["error"]


mock_provider.stream_messages = _mock_stream_messages


@pytest.fixture
def client(app):
    """HTTP client with provider resolution stubbed; patch only for this file."""
    with (
        patch(
            "free_claude_code.api.routes.resolve_provider",
            return_value=mock_provider,
        ),
        TestClient(app) as test_client,
    ):
        yield test_client


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["request-id"].startswith("req_")


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.headers["request-id"].startswith("req_")


def test_models_list(client: TestClient):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["has_more"] is False
    ids = [item["id"] for item in data["data"]]
    assert "claude-sonnet-4-20250514" in ids
    assert data["first_id"] == ids[0]
    assert data["last_id"] == ids[-1]
    assert response.headers["x-request-id"] == response.headers["request-id"]


def test_probe_endpoints_return_204_with_allow_headers(client: TestClient):
    responses = [
        client.head("/"),
        client.options("/"),
        client.head("/health"),
        client.options("/health"),
        client.head("/v1/messages"),
        client.options("/v1/messages"),
        client.head("/v1/messages/count_tokens"),
        client.options("/v1/messages/count_tokens"),
    ]

    for response in responses:
        assert response.status_code == 204
        assert "Allow" in response.headers


def test_create_message_stream(client: TestClient):
    """Create message returns streaming response."""
    _stream_messages_calls.clear()
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
    assert _stream_messages_calls[0][1]["request_id"] == response.headers["request-id"]


def test_auto_mode_classifier_without_stream_returns_json(client: TestClient):
    """Claude side queries omit stream and require one complete Message object."""
    _stream_messages_calls.clear()
    payload = {
        "model": "claude-opus-4-8",
        "max_tokens": 64,
        "system": (
            "You are a security monitor. Respond with <block>yes</block> "
            "or <block>no</block>."
        ),
        "messages": [
            {
                "role": "user",
                "content": "<transcript>\nBash curl example.com\n</transcript>",
            }
        ],
    }

    response = client.post("/v1/messages?beta=true", json=payload)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["type"] == "message"
    assert body["usage"] == {"input_tokens": 0, "output_tokens": 0}
    routed_request = _stream_messages_calls[0][0][0]
    assert routed_request.stream is False
    assert _stream_messages_calls[0][1]["reasoning"] == ReasoningPolicy.prefer_off()


def test_create_message_ingress_error_has_request_id_without_terminal_header(
    client: TestClient,
):
    response = client.post(
        "/v1/messages",
        json={"model": "test", "messages": [], "max_tokens": 10, "stream": True},
    )

    assert response.status_code == 400
    assert "x-should-retry" not in response.headers
    assert response.json()["request_id"] == response.headers["request-id"]


def test_create_message_rejects_unportable_image_as_invalid_request() -> None:
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        return_value=MagicMock(),
    ):
        provider = OpenAIChatProvider(
            make_provider_config(
                api_key="test-key",
                base_url="https://provider.invalid/v1",
            ),
            profile=OpenAIChatProfile(
                OpenAIChatRequestPolicy(
                    provider_name="TEST_CHAT",
                    reasoning_replay=ReasoningReplayMode.DISABLED,
                ),
                NO_REASONING,
            ),
            admission=immediate_admission(provider_name="TEST_CHAT"),
        )
    test_app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(test_app) as test_client,
    ):
        response = test_client.post(
            "/v1/messages",
            json={
                "model": "nvidia_nim/test-model",
                "max_tokens": 32,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "file", "file_id": "file_1"},
                            }
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "file" in response.json()["error"]["message"]


def test_create_message_schema_validation_has_request_id_without_terminal_header(
    client: TestClient,
):
    response = client.post(
        "/v1/messages",
        json={"model": "test", "messages": "not-a-list"},
    )

    assert response.status_code == 422
    assert response.headers["request-id"].startswith("req_")
    assert "x-should-retry" not in response.headers


def test_create_message_pre_start_provider_error_returns_terminal_json(
    client: TestClient,
):
    """Pre-start provider failures keep status without enabling client retries."""
    mock_provider.stream_messages = _mock_pre_start_rate_limit
    _stream_messages_calls.clear()
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": True,
    }

    with (
        patch("free_claude_code.api.response_streams.trace_event") as trace,
        patch("free_claude_code.application.execution.trace_event") as execution_trace,
    ):
        response = client.post("/v1/messages", json=payload)

    error = _terminal_json_error(response, status_code=429)
    assert error == {"type": "rate_limit_error", "message": "upstream is busy"}
    request_id = response.headers["request-id"]
    assert _stream_messages_calls[0][1]["request_id"] == request_id
    route_trace = next(
        call.kwargs
        for call in execution_trace.call_args_list
        if call.kwargs.get("event") == "free_claude_code.api.route.resolved"
    )
    assert route_trace["request_id"] == request_id
    terminal_trace = next(
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event")
        == "free_claude_code.api.response.terminal_execution_error"
    )
    assert terminal_trace == {
        "stage": "egress",
        "event": "free_claude_code.api.response.terminal_execution_error",
        "source": "api",
        "wire_api": "messages",
        "request_id": request_id,
        "status_code": 429,
        "error_type": "rate_limit_error",
        "client_should_retry": False,
        "exc_type": "ExecutionFailure",
        "failure_kind": "rate_limit",
        "provider_retryable": True,
    }
    mock_provider.stream_messages = _mock_stream_messages


def test_create_message_preserves_system_role_messages(client: TestClient):
    """Create message preserves latest-client system message placement."""
    mock_provider.stream_messages = _mock_stream_messages
    _stream_messages_calls.clear()
    payload = {
        "model": "claude-3-sonnet",
        "messages": [
            {"role": "user", "content": "context"},
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Hi"},
        ],
        "max_tokens": 100,
        "stream": True,
    }

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    routed_request = _stream_messages_calls[0][0][0]
    assert [message.role for message in routed_request.messages] == [
        "user",
        "system",
        "user",
    ]
    assert routed_request.messages[1].content == "system prompt"
    assert routed_request.system is None


def test_model_mapping(client: TestClient):
    # Test Haiku mapping
    _stream_messages_calls.clear()
    payload_haiku = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": True,
    }
    client.post("/v1/messages", json=payload_haiku)
    assert len(_stream_messages_calls) == 1
    args = _stream_messages_calls[0][0]
    kwargs = _stream_messages_calls[0][1]
    assert args[0].model != "claude-3-haiku-20240307"
    assert kwargs["reasoning"] == ReasoningPolicy.provider_default()


@pytest.mark.parametrize(
    ("failure", "expected_type"),
    [
        (
            ExecutionFailure(
                FailureKind.AUTHENTICATION, 401, "Invalid Key", retryable=False
            ),
            "authentication_error",
        ),
        (
            ExecutionFailure(
                FailureKind.INVALID_REQUEST,
                400,
                "Invalid request api_key=SECRET useful detail",
                retryable=False,
            ),
            "invalid_request_error",
        ),
        (
            ExecutionFailure(
                FailureKind.RATE_LIMIT, 429, "Too Many Requests", retryable=True
            ),
            "rate_limit_error",
        ),
        (
            ExecutionFailure(
                FailureKind.OVERLOADED, 529, "Server Overloaded", retryable=True
            ),
            "overloaded_error",
        ),
        (
            ExecutionFailure(
                FailureKind.UPSTREAM, 503, "Upstream failed", retryable=True
            ),
            "api_error",
        ),
    ],
)
def test_provider_execution_errors_preserve_status_and_type(
    client: TestClient,
    failure: ExecutionFailure,
    expected_type: str,
):
    base_payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        "stream": True,
    }

    def _raise_provider_error(*args, **kwargs):
        raise failure

    try:
        mock_provider.stream_messages = _raise_provider_error
        response = client.post("/v1/messages", json=base_payload)
        error = _terminal_json_error(response, status_code=failure.status_code)
        assert error["type"] == expected_type
        assert "SECRET" not in error["message"]
        if expected_type == "invalid_request_error":
            assert "useful detail" in error["message"]
    finally:
        mock_provider.stream_messages = _mock_stream_messages


def test_empty_provider_stream_returns_terminal_json(client: TestClient):
    mock_provider.stream_messages = _mock_empty_stream
    try:
        response = client.post(
            "/v1/messages",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "stream": True,
            },
        )
        error = _terminal_json_error(response, status_code=500)
        assert error["type"] == "api_error"
        assert error["message"] == "Stream ended before emitting a response."
    finally:
        mock_provider.stream_messages = _mock_stream_messages


def test_generic_stream_exception_returns_terminal_json(client: TestClient):
    """Unexpected provider execution failures return detailed terminal JSON."""

    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("unexpected crash")

    mock_provider.stream_messages = _raise_runtime
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    error = _terminal_json_error(response, status_code=500)
    assert error["type"] == "api_error"
    assert error["message"] == "unexpected crash"
    mock_provider.stream_messages = _mock_stream_messages


def test_generic_stream_exception_with_status_code_returns_terminal_json(
    client: TestClient,
):
    """Ad-hoc status_code attrs do not become retryable HTTP responses."""

    class ExceptionWithStatus(RuntimeError):
        def __init__(self, msg: str, status_code: int = 500):
            super().__init__(msg)
            self.status_code = status_code

    def _raise_with_status(*args, **kwargs):
        raise ExceptionWithStatus("bad gateway", 502)

    mock_provider.stream_messages = _raise_with_status
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    error = _terminal_json_error(response, status_code=500)
    assert error["type"] == "api_error"
    assert error["message"] == "bad gateway"
    mock_provider.stream_messages = _mock_stream_messages


def test_generic_stream_exception_empty_message_returns_non_empty_error(
    client: TestClient,
):
    """Exceptions with empty __str__ still return a readable HTTP detail."""

    class SilentError(RuntimeError):
        def __str__(self):
            return ""

    def _raise_silent(*args, **kwargs):
        raise SilentError()

    mock_provider.stream_messages = _raise_silent
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    error = _terminal_json_error(response, status_code=500)
    assert error["type"] == "api_error"
    assert error["message"] != ""
    mock_provider.stream_messages = _mock_stream_messages


def test_count_tokens_endpoint(client: TestClient):
    """count_tokens endpoint returns token count."""
    response = client.post(
        "/v1/messages/count_tokens",
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert "input_tokens" in response.json()
    assert response.headers["request-id"].startswith("req_")


def test_stop_endpoint_no_workflow_no_cli_503(app, client: TestClient):
    """POST /stop without messaging workflow or cli_manager returns 503."""
    # Ensure no messaging workflow or cli_manager on app state
    if hasattr(app.state, "messaging_workflow"):
        delattr(app.state, "messaging_workflow")
    if hasattr(app.state, "cli_manager"):
        delattr(app.state, "cli_manager")
    response = client.post("/stop")
    assert response.status_code == 503

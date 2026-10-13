"""Raw provider failure classification into the canonical neutral model."""

import ssl
from collections.abc import Callable
from dataclasses import asdict, dataclass

import httpx
import httpx2
import openai
import pytest

from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    attach_upstream_error_body,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.failure_policy import (
    ProviderRecoveryExhausted,
    classify_provider_failure,
    is_retryable_provider_error,
    is_retryable_stream_error,
    retryable_upstream_status,
)
from free_claude_code.providers.stream_recovery import TruncatedProviderStreamError


def _openai_status_error(
    error_type: type[openai.APIStatusError],
    *,
    status_code: int,
    message: str,
    body: object | None = None,
    headers: dict[str, str] | None = None,
) -> openai.APIStatusError:
    request = httpx2.Request("POST", "https://provider.test/v1/chat/completions")
    response = httpx2.Response(status_code, request=request, headers=headers)
    return error_type(
        message,
        response=response,
        body=body or {"error": {"message": message}},
    )


def _statusless_openai_error(message: str, body: object | None) -> openai.APIError:
    return openai.APIError(
        message,
        request=httpx2.Request("POST", "https://provider.test/v1/chat/completions"),
        body=body,
    )


def _http_status_error(
    status_code: int,
    message: str,
    *,
    body: object | None = None,
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.test/v1/messages")
    response = httpx.Response(
        status_code,
        request=request,
        json=body or {"error": {"message": message, "api_key": "SECRET"}},
    )
    return httpx.HTTPStatusError(message, request=request, response=response)


class _CodedError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__("provider request failed")
        self.code = code


def test_stream_retry_classification_distinguishes_protocol_and_status() -> None:
    assert is_retryable_stream_error(TruncatedProviderStreamError("truncated"))
    assert is_retryable_stream_error(httpx.ReadError("disconnected"))
    assert is_retryable_stream_error(_http_status_error(503, "unavailable"))
    assert not is_retryable_stream_error(_http_status_error(400, "bad request"))


@pytest.mark.parametrize(
    "error_name",
    [
        "ReadError",
        "ReadTimeout",
        "ConnectError",
        "ConnectTimeout",
        "WriteError",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
    ],
)
def test_http_client_network_errors_have_the_same_failure_policy(
    error_name: str,
) -> None:
    errors = [
        getattr(module, error_name)("connection interrupted")
        for module in (httpx, httpx2)
    ]
    failures = [
        classify_provider_failure(
            error, provider_name="TEST", read_timeout_s=10, request_id="request"
        )
        for error in errors
    ]
    assert asdict(failures[0]) == asdict(failures[1])
    assert is_retryable_provider_error(errors[0]) == is_retryable_provider_error(
        errors[1]
    )
    assert is_retryable_stream_error(errors[0]) == is_retryable_stream_error(errors[1])


def test_stream_retry_classification_only_accepts_post_open_timeouts() -> None:
    request = httpx.Request("POST", "https://provider.test/v1/messages")

    assert is_retryable_stream_error(httpx.ReadTimeout("read", request=request))
    assert not is_retryable_stream_error(
        httpx.ConnectTimeout("connect", request=request)
    )
    assert not is_retryable_stream_error(httpx.WriteTimeout("write", request=request))
    assert not is_retryable_stream_error(httpx.PoolTimeout("pool", request=request))


@pytest.mark.parametrize("read_timeout_s", [None, 120])
def test_ssl_want_read_error_uses_stream_failure_policy(
    read_timeout_s: float | None,
) -> None:
    error = ssl.SSLWantReadError("the operation did not complete")

    assert is_retryable_stream_error(error)
    assert is_retryable_provider_error(error)
    failure = classify_provider_failure(
        error, provider_name="NIM", read_timeout_s=read_timeout_s, request_id="request"
    )
    assert failure.kind is FailureKind.UNAVAILABLE
    assert failure.status_code == 502
    assert failure.retryable
    assert "Could not read the provider response." in failure.message
    assert "timed out" not in failure.message


@pytest.mark.parametrize(
    ("message", "body"),
    (
        (
            "stream embedded error",
            {"error": {"message": "internal failure", "code": 500}},
        ),
        (
            "stream embedded error",
            {
                "error": {
                    "message": "internal failure",
                    "type": "internal_server_error",
                }
            },
        ),
        (
            "ResourceExhausted: limit reached while generating response",
            {"error": {"message": "ResourceExhausted: limit reached"}},
        ),
    ),
)
def test_stream_retry_classification_accepts_statusless_transients(
    message: str,
    body: object,
) -> None:
    assert is_retryable_stream_error(_statusless_openai_error(message, body))


def test_stream_retry_classification_rejects_openai_bad_request() -> None:
    assert not is_retryable_stream_error(
        _openai_status_error(
            openai.BadRequestError,
            status_code=400,
            message="bad request",
        )
    )


def test_http_413_status_wins_over_rate_limit_markers() -> None:
    error = _openai_status_error(
        openai.APIStatusError,
        status_code=413,
        message="Request too large for token rate limit",
        body={
            "error": {
                "message": "Request requires 55940 tokens but limit is 8000",
                "type": "tokens",
                "code": "rate_limit_exceeded",
            }
        },
        headers={"retry-after": "67", "x-should-retry": "false"},
    )

    assert not is_retryable_provider_error(error)
    assert not is_retryable_stream_error(error)
    assert retryable_upstream_status(error) is None

    failure = classify_provider_failure(
        error,
        provider_name="GROQ",
        read_timeout_s=60.0,
        request_id="req_too_large",
    )

    assert failure.kind is FailureKind.INVALID_REQUEST
    assert failure.status_code == 413
    assert failure.retryable is False
    assert "Provider rejected the request as too large." in failure.message
    assert "Request requires 55940 tokens but limit is 8000" in failure.message
    assert "Request ID: req_too_large" in failure.message


@pytest.mark.parametrize(
    "error",
    [
        _http_status_error(413, "request too large"),
        _statusless_openai_error(
            "rate limit exceeded",
            {
                "error": {
                    "status": 413,
                    "code": "rate_limit_exceeded",
                    "message": "request too large",
                }
            },
        ),
    ],
    ids=["httpx_status", "structured_body_status"],
)
def test_http_413_sources_are_terminal_invalid_requests(error: Exception) -> None:
    assert not is_retryable_provider_error(error)
    assert not is_retryable_stream_error(error)
    assert retryable_upstream_status(error) is None

    failure = classify_provider_failure(
        error,
        provider_name="TEST_PROVIDER",
        read_timeout_s=60.0,
        request_id="req_413",
    )

    assert failure.kind is FailureKind.INVALID_REQUEST
    assert failure.status_code == 413
    assert failure.retryable is False
    assert "Provider rejected the request as too large." in failure.message


@pytest.mark.parametrize(
    "error",
    [
        _CodedError(" Context_Length_Exceeded "),
        _openai_status_error(
            openai.BadRequestError,
            status_code=400,
            message="maximum context reached",
            body={"error": {"code": "context_length_exceeded"}},
        ),
        _statusless_openai_error(
            "maximum context reached",
            {"type": "context_length_exceeded"},
        ),
        _http_status_error(
            500,
            "maximum context reached",
            body={"error": {"type": "context_length_exceeded"}},
        ),
    ],
    ids=["exception_code", "sdk_nested_code", "sdk_root_type", "http_body_type"],
)
def test_structured_context_window_signals_are_canonical_and_terminal(
    error: Exception,
) -> None:
    assert not is_retryable_provider_error(error)
    assert not is_retryable_stream_error(error)
    assert retryable_upstream_status(error) is None

    failure = classify_provider_failure(
        error,
        provider_name="TEST_PROVIDER",
        read_timeout_s=60.0,
        request_id="req_context",
    )

    assert failure.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED
    assert failure.status_code == 400
    assert failure.retryable is False
    assert "Provider input exceeds the model context window." in failure.message


@pytest.mark.parametrize(
    "error",
    [
        _openai_status_error(
            openai.BadRequestError,
            status_code=400,
            message="context_length_exceeded",
        ),
        _http_status_error(
            413,
            "Request too large for model context_length_exceeded",
            body={"error": {"code": "request_too_large"}},
        ),
        _http_status_error(
            413,
            "Request requires 26206 tokens but TPM limit is 8000",
            body={
                "error": {
                    "type": "tokens",
                    "code": "rate_limit_exceeded",
                    "message": "Request too large for the tokens-per-minute limit",
                }
            },
        ),
    ],
    ids=["message_only", "request_too_large", "groq_tpm"],
)
def test_ambiguous_large_request_signals_are_not_context_exhaustion(
    error: Exception,
) -> None:
    failure = classify_provider_failure(
        error,
        provider_name="TEST_PROVIDER",
        read_timeout_s=60.0,
        request_id=None,
    )

    assert failure.kind is FailureKind.INVALID_REQUEST


def test_nonstandard_capacity_status_remains_retryable() -> None:
    error = _openai_status_error(
        openai.APIStatusError,
        status_code=498,
        message="capacity exceeded",
        body={"error": {"code": "capacity_exceeded"}},
    )

    assert retryable_upstream_status(error) == 503
    failure = classify_provider_failure(
        error,
        provider_name="GROQ",
        read_timeout_s=60.0,
        request_id="req_capacity",
    )
    assert failure.kind is FailureKind.OVERLOADED
    assert failure.retryable is True


@dataclass(frozen=True, slots=True)
class _ClassificationCase:
    name: str
    error: Callable[[], Exception]
    kind: FailureKind
    status_code: int
    retryable: bool
    message: str | None = None


_CASES = (
    _ClassificationCase(
        "openai_authentication",
        lambda: _openai_status_error(
            openai.AuthenticationError,
            status_code=401,
            message="Unauthorized",
        ),
        FailureKind.AUTHENTICATION,
        401,
        False,
    ),
    _ClassificationCase(
        "openai_permission_denied",
        lambda: _openai_status_error(
            openai.PermissionDeniedError,
            status_code=403,
            message="Forbidden",
        ),
        FailureKind.PERMISSION,
        403,
        False,
    ),
    _ClassificationCase(
        "generic_openai_401",
        lambda: _openai_status_error(
            openai.APIStatusError,
            status_code=401,
            message="Unauthorized",
        ),
        FailureKind.AUTHENTICATION,
        401,
        False,
    ),
    _ClassificationCase(
        "generic_openai_402",
        lambda: _openai_status_error(
            openai.APIStatusError,
            status_code=402,
            message="Inference credits exhausted",
        ),
        FailureKind.PERMISSION,
        402,
        False,
        "Provider requires payment or additional credits. Add credits or resolve billing.",
    ),
    _ClassificationCase(
        "generic_openai_403",
        lambda: _openai_status_error(
            openai.APIStatusError,
            status_code=403,
            message="Forbidden",
        ),
        FailureKind.PERMISSION,
        403,
        False,
    ),
    _ClassificationCase(
        "openai_rate_limit",
        lambda: _openai_status_error(
            openai.RateLimitError,
            status_code=429,
            message="Too many requests",
        ),
        FailureKind.RATE_LIMIT,
        429,
        True,
    ),
    _ClassificationCase(
        "openai_bad_request",
        lambda: _openai_status_error(
            openai.BadRequestError,
            status_code=400,
            message="bad tool shape",
        ),
        FailureKind.INVALID_REQUEST,
        400,
        False,
    ),
    _ClassificationCase(
        "openai_overload_marker",
        lambda: _openai_status_error(
            openai.InternalServerError,
            status_code=500,
            message="No capacity available",
        ),
        FailureKind.OVERLOADED,
        529,
        True,
    ),
    _ClassificationCase(
        "openai_generic_503_preserved",
        lambda: _openai_status_error(
            openai.InternalServerError,
            status_code=503,
            message="generic server failure",
        ),
        FailureKind.UPSTREAM,
        503,
        True,
    ),
    _ClassificationCase(
        "statusless_openai_rate_limit_body",
        lambda: _statusless_openai_error(
            "stream embedded error",
            {"error": {"message": "too many requests", "code": 429}},
        ),
        FailureKind.RATE_LIMIT,
        429,
        True,
    ),
    _ClassificationCase(
        "statusless_openai_overload_body",
        lambda: _statusless_openai_error(
            "ResourceExhausted: limit reached",
            {"error": {"message": "ResourceExhausted: limit reached"}},
        ),
        FailureKind.OVERLOADED,
        529,
        True,
    ),
    _ClassificationCase(
        "statusless_openai_unknown_is_not_retryable",
        lambda: _statusless_openai_error(
            "stream embedded error",
            {"error": {"message": "unknown provider failure"}},
        ),
        FailureKind.UPSTREAM,
        500,
        False,
    ),
    _ClassificationCase(
        "statusless_openai_tokenrouter_bad_request",
        lambda: _statusless_openai_error(
            "stream embedded error",
            {
                "object": "error",
                "message": (
                    "Invalid request: Disaggregated request received without "
                    "bootstrap room id"
                ),
                "type": "BAD_REQUEST",
                "param": None,
                "code": 400,
            },
        ),
        FailureKind.UPSTREAM,
        500,
        False,
    ),
    _ClassificationCase(
        "openai_insufficient_user_quota",
        lambda: _openai_status_error(
            openai.PermissionDeniedError,
            status_code=403,
            message="insufficient user quota",
            body={
                "error": {
                    "message": "insufficient user quota",
                    "code": "insufficient_user_quota",
                }
            },
        ),
        FailureKind.PERMISSION,
        403,
        False,
    ),
    _ClassificationCase(
        "http_401_maps_authentication",
        lambda: _http_status_error(401, "Unauthorized"),
        FailureKind.AUTHENTICATION,
        401,
        False,
    ),
    _ClassificationCase(
        "http_402_maps_billing_permission",
        lambda: _http_status_error(402, "Inference credits exhausted"),
        FailureKind.PERMISSION,
        402,
        False,
        "Provider requires payment or additional credits. Add credits or resolve billing.",
    ),
    _ClassificationCase(
        "http_403_maps_permission",
        lambda: _http_status_error(403, "Forbidden"),
        FailureKind.PERMISSION,
        403,
        False,
    ),
    _ClassificationCase(
        "http_502_keeps_overload_quirk",
        lambda: _http_status_error(502, "Bad gateway"),
        FailureKind.OVERLOADED,
        529,
        True,
    ),
    _ClassificationCase(
        "http_599_preserves_status",
        lambda: _http_status_error(599, "Upstream failure"),
        FailureKind.UPSTREAM,
        599,
        True,
    ),
    _ClassificationCase(
        "http_405_is_not_retryable",
        lambda: _http_status_error(405, "Wrong endpoint"),
        FailureKind.UPSTREAM,
        405,
        False,
    ),
    _ClassificationCase(
        "read_timeout_keeps_pre_start_status",
        lambda: httpx.ReadTimeout(
            "",
            request=httpx.Request("POST", "https://provider.test/v1/messages"),
        ),
        FailureKind.TIMEOUT,
        502,
        True,
    ),
    _ClassificationCase(
        "openai_connection_error_keeps_status",
        lambda: openai.APIConnectionError(
            request=httpx2.Request("POST", "https://provider.test/v1/chat/completions")
        ),
        FailureKind.UNAVAILABLE,
        500,
        True,
    ),
    _ClassificationCase(
        "unknown_exception_keeps_gateway_status",
        lambda: RuntimeError("unexpected provider failure"),
        FailureKind.UPSTREAM,
        502,
        False,
    ),
)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_raw_provider_failure_maps_to_canonical_failure(
    case: _ClassificationCase,
) -> None:
    failure = classify_provider_failure(
        case.error(),
        provider_name="TEST_PROVIDER",
        read_timeout_s=30.0,
        request_id="req_classification",
    )

    assert isinstance(failure, ExecutionFailure)
    assert failure.kind is case.kind
    assert failure.status_code == case.status_code
    assert failure.retryable is case.retryable
    assert failure.message.strip()
    assert "Request ID: req_classification" in failure.message
    assert "SECRET" not in failure.message
    if case.message is not None:
        assert case.message in failure.message


def test_classification_preserves_useful_body_while_redacting_credentials() -> None:
    error = _http_status_error(
        400,
        "unsupported model format authorization: Bearer AUTH_SECRET",
    )

    failure = classify_provider_failure(
        error,
        provider_name="LOCAL",
        read_timeout_s=60.0,
        request_id="req_body",
    )

    assert failure.kind is FailureKind.INVALID_REQUEST
    assert failure.status_code == 400
    assert "Upstream provider LOCAL returned HTTP 400." in failure.message
    assert "unsupported model format" in failure.message
    assert "Request ID: req_body" in failure.message
    assert "AUTH_SECRET" not in failure.message
    assert "SECRET" not in failure.message


def test_auth_failure_preserves_model_error_body_instead_of_masking_it() -> None:
    error = _openai_status_error(
        openai.AuthenticationError,
        status_code=401,
        message="Unauthorized",
        body={
            "type": "error",
            "error": {
                "type": "ModelError",
                "message": ("Model qwen3.7-max is not supported for format oa-compat"),
            },
        },
    )

    failure = classify_provider_failure(
        error,
        provider_name="OPENCODE_GO",
        read_timeout_s=60.0,
        request_id="req_model",
    )

    assert failure.kind is FailureKind.AUTHENTICATION
    assert failure.status_code == 401
    assert "Category: ModelError" in failure.message
    assert "Provider authentication failed. Check API key." in failure.message
    assert "Model qwen3.7-max is not supported for format oa-compat" in failure.message
    assert "Request ID: req_model" in failure.message


def test_permission_failure_preserves_actionable_upstream_diagnostic() -> None:
    error = _openai_status_error(
        openai.PermissionDeniedError,
        status_code=403,
        message="Forbidden",
        body={
            "status": 403,
            "title": "Forbidden",
            "detail": "Authorization failed",
        },
    )

    failure = classify_provider_failure(
        error,
        provider_name="NIM",
        read_timeout_s=60.0,
        request_id="req_permission",
    )

    assert failure.kind is FailureKind.PERMISSION
    assert failure.status_code == 403
    assert failure.retryable is False
    assert not is_retryable_provider_error(error)
    assert "Upstream provider NIM returned HTTP 403." in failure.message
    assert "Category: permission" in failure.message
    assert (
        "Mapped message: Provider denied access. "
        "Check credential permissions and model access."
    ) in failure.message
    assert '"detail":"Authorization failed"' in failure.message
    assert "Request ID: req_permission" in failure.message


def test_empty_http_error_body_is_reported_explicitly() -> None:
    request = httpx.Request("POST", "https://provider.test/v1/messages")
    response = httpx.Response(500, request=request, content=b"")
    error = httpx.HTTPStatusError(
        "Server Error",
        request=request,
        response=response,
    )

    failure = classify_provider_failure(
        error,
        provider_name="EMPTY",
        read_timeout_s=30.0,
        request_id="req_empty",
    )

    assert failure.kind is FailureKind.UPSTREAM
    assert failure.status_code == 500
    assert "Upstream provider EMPTY returned HTTP 500." in failure.message
    assert "(empty upstream error body)" in failure.message


def test_http_405_diagnostic_names_rejected_upstream_endpoint() -> None:
    failure = classify_provider_failure(
        _http_status_error(405, "Method Not Allowed"),
        provider_name="LOCAL",
        read_timeout_s=30.0,
        request_id="req_405",
    )

    assert failure.kind is FailureKind.UPSTREAM
    assert failure.status_code == 405
    assert (
        "Upstream provider LOCAL rejected the request method or endpoint (HTTP 405)."
        in failure.message
    )
    assert "Request ID: req_405" in failure.message


def test_connection_cause_chain_is_redacted_and_capped() -> None:
    request = httpx2.Request("POST", "https://provider.test/v1/chat/completions")
    error = openai.APIConnectionError(request=request)
    error.__cause__ = httpx2.ConnectError(
        "connect failed authorization: Bearer CAUSE_SECRET "
        + "x" * (ERROR_DETAIL_DISPLAY_CAP_BYTES + 10),
        request=request,
    )

    failure = classify_provider_failure(
        error,
        provider_name="NIM",
        read_timeout_s=30.0,
        request_id="req_cause",
    )

    assert "Caused by:" in failure.message
    assert "ConnectError: connect failed authorization: <redacted>" in failure.message
    assert "CAUSE_SECRET" not in failure.message
    assert f"truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes" in failure.message
    assert "Request ID: req_cause" in failure.message


def test_attached_streamed_error_body_remains_bounded() -> None:
    request = httpx.Request("POST", "https://provider.test/v1/messages")
    response = httpx.Response(500, request=request, content=b"")
    error = httpx.HTTPStatusError(
        "Server Error",
        request=request,
        response=response,
    )
    attach_upstream_error_body(
        error,
        "x" * (ERROR_DETAIL_DISPLAY_CAP_BYTES + 10),
    )

    failure = classify_provider_failure(
        error,
        provider_name="LONG",
        read_timeout_s=30.0,
        request_id="req_long",
    )

    assert f"truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes" in failure.message
    assert "x" * 100 in failure.message


def test_shared_recovery_exhaustion_preserves_last_provider_failure() -> None:
    raw_error = _http_status_error(503, "provider still unavailable")
    exhausted = ProviderRecoveryExhausted(raw_error)

    failure = classify_provider_failure(
        exhausted,
        provider_name="SHARED",
        read_timeout_s=30.0,
        request_id="req_exhausted",
    )

    assert failure.kind is FailureKind.OVERLOADED
    assert failure.status_code == 529
    assert "provider still unavailable" in failure.message
    assert "Request ID: req_exhausted" in failure.message
    assert retryable_upstream_status(exhausted) is None
    assert not is_retryable_provider_error(exhausted)

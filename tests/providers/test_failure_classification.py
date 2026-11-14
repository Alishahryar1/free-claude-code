"""One table that pins how every upstream exception shape is classified.

Each row states the full observable contract for one exception: the canonical
failure a client receives, the transient status used for backoff labels, whether
admission may retry it, whether an already-open stream may replay it, and the
credential status used for authentication refresh.
"""

import ssl
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import httpx
import httpx2
import openai
import pytest

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.failure_policy import (
    ProviderRecoveryExhausted,
    RetryableProviderProtocolError,
    classify_provider_failure,
    is_retryable_provider_error,
    is_retryable_stream_error,
    provider_authentication_status,
    retryable_upstream_status,
)

_URL = "https://provider.test/v1/chat/completions"


@dataclass(frozen=True, slots=True)
class Observed:
    kind: FailureKind
    status_code: int
    retryable: bool
    transient_status: int | None
    retryable_on_open: bool
    retryable_mid_stream: bool
    auth_status: int | None


def _observe(exc: Exception) -> Observed:
    failure = classify_provider_failure(
        exc, provider_name="TEST", read_timeout_s=None, request_id=None
    )
    return Observed(
        kind=failure.kind,
        status_code=failure.status_code,
        retryable=failure.retryable,
        transient_status=retryable_upstream_status(exc),
        retryable_on_open=is_retryable_provider_error(exc),
        retryable_mid_stream=is_retryable_stream_error(exc),
        auth_status=provider_authentication_status(exc),
    )


def _openai_status(
    error_type: type[openai.APIStatusError], status_code: int, message: str
) -> Callable[[], Exception]:
    def build() -> Exception:
        request = httpx2.Request("POST", _URL)
        response = httpx2.Response(status_code, request=request)
        return error_type(
            message, response=response, body={"error": {"message": message}}
        )

    return build


def _statusless(message: str, body: object) -> Callable[[], Exception]:
    def build() -> Exception:
        return openai.APIError(message, request=httpx2.Request("POST", _URL), body=body)

    return build


def _http_status(
    module: ModuleType, status_code: int, body: object | None = None
) -> Callable[[], Exception]:
    def build() -> Exception:
        request = module.Request("POST", _URL)
        response = module.Response(
            status_code,
            request=request,
            json=body or {"error": {"message": "upstream said no"}},
        )
        return module.HTTPStatusError(
            "status error", request=request, response=response
        )

    return build


def _transport(
    error_type: type[Exception], message: str = "connection interrupted"
) -> Callable[[], Exception]:
    return lambda: error_type(message)


class _CodedError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__("provider request failed")
        self.code = code


class _CodedReadTimeout(httpx.ReadTimeout):
    def __init__(self, code: str) -> None:
        super().__init__("read timed out")
        self.code = code


def _o(
    kind: FailureKind,
    status_code: int,
    retryable: bool,
    *,
    transient: int | None = None,
    on_open: bool | None = None,
    mid_stream: bool | None = None,
    auth: int | None = None,
) -> Observed:
    on_open = retryable if on_open is None else on_open
    return Observed(
        kind=kind,
        status_code=status_code,
        retryable=retryable,
        transient_status=transient,
        retryable_on_open=on_open,
        retryable_mid_stream=on_open if mid_stream is None else mid_stream,
        auth_status=auth,
    )


K = FailureKind

_ROWS: dict[str, tuple[Callable[[], Exception], Observed]] = {
    # Canonical failures pass through unchanged.
    "execution_failure_retryable": (
        lambda: ExecutionFailure(K.UPSTREAM, 503, "unavailable", True),
        _o(K.UPSTREAM, 503, True, transient=503),
    ),
    "execution_failure_terminal": (
        lambda: ExecutionFailure(K.AUTHENTICATION, 401, "bad key", False),
        _o(K.AUTHENTICATION, 401, False, auth=401),
    ),
    # Exhausted recovery reports its cause but never retries again.
    "recovery_exhausted": (
        lambda: ProviderRecoveryExhausted(
            _openai_status(openai.RateLimitError, 429, "slow down")()
        ),
        _o(K.RATE_LIMIT, 429, True, on_open=False),
    ),
    "protocol_error": (
        lambda: RetryableProviderProtocolError("truncated"),
        _o(K.UPSTREAM, 502, True),
    ),
    # OpenAI SDK status errors.
    "openai_401": (
        _openai_status(openai.AuthenticationError, 401, "bad key"),
        _o(K.AUTHENTICATION, 401, False, auth=401),
    ),
    "openai_403": (
        _openai_status(openai.PermissionDeniedError, 403, "denied"),
        _o(K.PERMISSION, 403, False, auth=403),
    ),
    "openai_403_with_overload_text": (
        _openai_status(openai.PermissionDeniedError, 403, "limit reached"),
        _o(K.PERMISSION, 403, False, transient=503, auth=403),
    ),
    "openai_400": (
        _openai_status(openai.BadRequestError, 400, "bad request"),
        _o(K.INVALID_REQUEST, 400, False),
    ),
    "openai_429": (
        _openai_status(openai.RateLimitError, 429, "slow down"),
        _o(K.RATE_LIMIT, 429, True, transient=429),
    ),
    "openai_500": (
        _openai_status(openai.InternalServerError, 500, "server fault"),
        _o(K.UPSTREAM, 500, True, transient=500),
    ),
    "openai_503_overloaded": (
        _openai_status(openai.InternalServerError, 503, "overloaded"),
        _o(K.OVERLOADED, 529, True, transient=503),
    ),
    "openai_402_billing": (
        _openai_status(openai.APIStatusError, 402, "payment required"),
        _o(K.PERMISSION, 402, False),
    ),
    "openai_404": (
        _openai_status(openai.NotFoundError, 404, "no such model"),
        _o(K.UPSTREAM, 404, False),
    ),
    "openai_413_beats_rate_limit_text": (
        _openai_status(openai.APIStatusError, 413, "too large for rate limit"),
        _o(K.INVALID_REQUEST, 413, False),
    ),
    "openai_498_capacity": (
        _openai_status(openai.APIStatusError, 498, "capacity"),
        _o(K.OVERLOADED, 529, True, transient=503),
    ),
    "openai_401_with_rate_limit_text": (
        _openai_status(openai.APIStatusError, 401, "rate limit exceeded"),
        _o(K.AUTHENTICATION, 401, False, transient=429, auth=401),
    ),
    # Statusless OpenAI errors carry their meaning in the body or text.
    "statusless_body_429": (
        _statusless("stream error", {"error": {"code": 429}}),
        _o(K.RATE_LIMIT, 429, True, transient=429),
    ),
    "statusless_overload_text": (
        _statusless(
            "ResourceExhausted: limit reached",
            {"error": {"message": "ResourceExhausted: limit reached"}},
        ),
        _o(K.OVERLOADED, 529, True, transient=503),
    ),
    "statusless_body_500": (
        _statusless("stream error", {"error": {"code": 500}}),
        _o(K.UPSTREAM, 500, True, transient=500),
    ),
    "statusless_invalid_request_type": (
        _statusless("stream error", {"error": {"type": "invalid_request_error"}}),
        _o(K.INVALID_REQUEST, 400, False),
    ),
    "statusless_invalid_api_key": (
        _statusless("stream error", {"error": {"code": "invalid_api_key"}}),
        _o(K.AUTHENTICATION, 401, False, auth=401),
    ),
    "statusless_unknown": (
        _statusless("stream error", {"error": {"message": "mystery"}}),
        _o(K.UPSTREAM, 500, False),
    ),
    "openai_timeout": (
        lambda: openai.APITimeoutError(request=httpx2.Request("POST", _URL)),
        _o(K.TIMEOUT, 500, True),
    ),
    "openai_connection": (
        lambda: openai.APIConnectionError(request=httpx2.Request("POST", _URL)),
        _o(K.UNAVAILABLE, 500, True),
    ),
    # Raw httpx status errors.
    "httpx_400": (_http_status(httpx, 400), _o(K.INVALID_REQUEST, 400, False)),
    "httpx_401": (_http_status(httpx, 401), _o(K.AUTHENTICATION, 401, False, auth=401)),
    "httpx_402": (_http_status(httpx, 402), _o(K.PERMISSION, 402, False)),
    "httpx_403": (_http_status(httpx, 403), _o(K.PERMISSION, 403, False, auth=403)),
    "httpx_404": (_http_status(httpx, 404), _o(K.UPSTREAM, 404, False)),
    "httpx_413": (_http_status(httpx, 413), _o(K.INVALID_REQUEST, 413, False)),
    "httpx_429": (_http_status(httpx, 429), _o(K.RATE_LIMIT, 429, True, transient=429)),
    "httpx_500": (_http_status(httpx, 500), _o(K.UPSTREAM, 500, True, transient=500)),
    "httpx_503": (_http_status(httpx, 503), _o(K.OVERLOADED, 529, True, transient=503)),
    "httpx_500_context_window": (
        _http_status(httpx, 500, {"error": {"code": "context_length_exceeded"}}),
        _o(K.CONTEXT_WINDOW_EXCEEDED, 400, False),
    ),
    # Raw httpx2 status errors.
    "httpx2_401": (
        _http_status(httpx2, 401),
        _o(K.AUTHENTICATION, 401, False, auth=401),
    ),
    "httpx2_402": (_http_status(httpx2, 402), _o(K.PERMISSION, 402, False)),
    "httpx2_404": (_http_status(httpx2, 404), _o(K.UPSTREAM, 404, False)),
    "httpx2_429": (
        _http_status(httpx2, 429),
        _o(K.RATE_LIMIT, 429, True, transient=429),
    ),
    "httpx2_503": (
        _http_status(httpx2, 503),
        _o(K.OVERLOADED, 529, True, transient=503),
    ),
    # Transport failures.
    "httpx_read_timeout": (_transport(httpx.ReadTimeout), _o(K.TIMEOUT, 502, True)),
    "httpx2_read_timeout": (_transport(httpx2.ReadTimeout), _o(K.TIMEOUT, 502, True)),
    "httpx_connect_timeout": (
        _transport(httpx.ConnectTimeout),
        _o(K.TIMEOUT, 502, True, mid_stream=False),
    ),
    "httpx2_connect_timeout": (
        _transport(httpx2.ConnectTimeout),
        _o(K.TIMEOUT, 502, True, mid_stream=False),
    ),
    "httpx_write_timeout": (
        _transport(httpx.WriteTimeout),
        _o(K.TIMEOUT, 502, True, mid_stream=False),
    ),
    "httpx_pool_timeout": (
        _transport(httpx.PoolTimeout),
        _o(K.TIMEOUT, 502, True, mid_stream=False),
    ),
    "httpx_connect_error": (
        _transport(httpx.ConnectError),
        _o(K.UNAVAILABLE, 502, True),
    ),
    "httpx_read_error": (_transport(httpx.ReadError), _o(K.UNAVAILABLE, 502, True)),
    "httpx_remote_protocol": (
        _transport(httpx.RemoteProtocolError),
        _o(K.UPSTREAM, 502, True),
    ),
    "httpx_local_protocol": (
        _transport(httpx.LocalProtocolError),
        _o(K.UPSTREAM, 502, False),
    ),
    "builtin_timeout": (_transport(TimeoutError), _o(K.TIMEOUT, 502, True)),
    "ssl_want_read": (
        _transport(ssl.SSLWantReadError),
        _o(K.UNAVAILABLE, 502, True),
    ),
    # Unknown exceptions are judged by their text.
    "generic_rate_limit_text": (
        _transport(RuntimeError, "rate limit exceeded"),
        _o(K.RATE_LIMIT, 429, True, transient=429),
    ),
    "generic_overload_text": (
        _transport(RuntimeError, "model overloaded"),
        _o(K.OVERLOADED, 529, True, transient=503),
    ),
    "generic_unknown": (_transport(RuntimeError, "boom"), _o(K.UPSTREAM, 502, False)),
    # Structured context-window codes are terminal on any exception type.
    "coded_context_window": (
        lambda: _CodedError("context_length_exceeded"),
        _o(K.CONTEXT_WINDOW_EXCEEDED, 400, False),
    ),
    "coded_context_window_on_read_timeout": (
        lambda: _CodedReadTimeout("context_length_exceeded"),
        _o(K.CONTEXT_WINDOW_EXCEEDED, 400, False),
    ),
}


@pytest.mark.parametrize(("build", "expected"), _ROWS.values(), ids=_ROWS.keys())
def test_failure_classification_matrix(
    build: Callable[[], Exception], expected: Observed
) -> None:
    assert _observe(build()) == expected


def test_transient_statuses_are_always_backoff_eligible() -> None:
    for build, _expected in _ROWS.values():
        status = _observe(build()).transient_status
        assert status is None or status == 429 or 500 <= status <= 599

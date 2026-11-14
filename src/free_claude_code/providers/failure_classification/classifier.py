"""Classify one raw upstream exception into one failure verdict."""

import ssl
from dataclasses import replace

import httpx
import httpx2
import openai

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.canonical_failures import (
    CONTEXT_WINDOW_EXCEEDED_MESSAGE,
    OVERLOADED_MESSAGE,
)
from free_claude_code.providers.recovery_errors import (
    ProviderRecoveryExhausted,
    RetryableProviderProtocolError,
)

from .evidence import (
    auth_status,
    is_backoff_status,
    is_invalid_request,
    reported_status,
    reports_context_window_exceeded,
    reports_overload,
    transient_status,
)
from .verdict import FailureVerdict

_AUTHENTICATION_MESSAGE = "Provider authentication failed. Check API key."
_PERMISSION_MESSAGE = (
    "Provider denied access. Check credential permissions and model access."
)
_BILLING_MESSAGE = (
    "Provider requires payment or additional credits. Add credits or resolve billing."
)
_RATE_LIMIT_MESSAGE = "Provider rate limit reached. Please retry shortly."
_INVALID_REQUEST_MESSAGE = "Invalid request sent to provider."
_REQUEST_TOO_LARGE_MESSAGE = "Provider rejected the request as too large."
_UNAVAILABLE_MESSAGE = "Provider is temporarily unavailable. Please retry."
_API_FAILED_MESSAGE = "Provider API request failed."

_RETRYABLE_TRANSPORT_ERRORS = (
    TimeoutError,
    ssl.SSLWantReadError,
    httpx.TimeoutException,
    httpx2.TimeoutException,
    httpx.RemoteProtocolError,
    httpx2.RemoteProtocolError,
    httpx.NetworkError,
    httpx2.NetworkError,
    RetryableProviderProtocolError,
)

type _Canonical = tuple[FailureKind, int, bool, str | None]


def classify(exc: Exception) -> FailureVerdict:
    """Return the one verdict that describes a raw upstream failure."""
    if isinstance(exc, ProviderRecoveryExhausted):
        # Exhausted recovery keeps its cause's meaning but never retries again.
        cause = classify(exc.last_error)
        return replace(
            cause,
            retryable=False,
            retryable_mid_stream=False,
            transient_status=None,
            auth_status=None,
        )
    if isinstance(exc, ExecutionFailure):
        return _from_execution_failure(exc)

    transient = transient_status(exc)
    credential = auth_status(exc)
    kind, status_code, retryable, message = _canonical(exc, transient, credential)
    return FailureVerdict(
        kind=kind,
        status_code=status_code,
        retryable=retryable,
        message=message,
        # Connection-setup timeouts cannot arise on an open stream.
        retryable_mid_stream=retryable and not _is_connection_setup_timeout(exc),
        transient_status=transient,
        auth_status=credential,
    )


def _from_execution_failure(failure: ExecutionFailure) -> FailureVerdict:
    status = failure.status_code
    return FailureVerdict(
        kind=failure.kind,
        status_code=status,
        retryable=failure.retryable,
        message=failure.message,
        retryable_mid_stream=failure.retryable,
        transient_status=(
            status if failure.retryable and is_backoff_status(status) else None
        ),
        auth_status=status if status in {401, 403} else None,
    )


def _canonical(
    exc: Exception, transient: int | None, credential: int | None
) -> _Canonical:
    if reports_context_window_exceeded(exc):
        return (
            FailureKind.CONTEXT_WINDOW_EXCEEDED,
            400,
            False,
            CONTEXT_WINDOW_EXCEEDED_MESSAGE,
        )
    if is_invalid_request(exc):
        return _INVALID_REQUEST
    if reported_status(exc) == 413:
        return (FailureKind.INVALID_REQUEST, 413, False, _REQUEST_TOO_LARGE_MESSAGE)
    if credential == 401:
        return (FailureKind.AUTHENTICATION, 401, False, _AUTHENTICATION_MESSAGE)
    if credential == 403:
        return (FailureKind.PERMISSION, 403, False, _PERMISSION_MESSAGE)
    if isinstance(exc, openai.APIError):
        return _openai_canonical(exc, transient)
    if isinstance(exc, httpx.HTTPStatusError | httpx2.HTTPStatusError):
        return _http_status_canonical(exc.response.status_code)
    return _transport_canonical(exc, transient)


def _openai_canonical(exc: openai.APIError, transient: int | None) -> _Canonical:
    if isinstance(exc, openai.RateLimitError):
        return _RATE_LIMITED
    if isinstance(exc, openai.BadRequestError):
        return _INVALID_REQUEST
    if isinstance(exc, openai.APITimeoutError):
        return (FailureKind.TIMEOUT, 500, True, _API_FAILED_MESSAGE)
    if isinstance(exc, openai.APIConnectionError):
        return (FailureKind.UNAVAILABLE, 500, True, _API_FAILED_MESSAGE)
    if isinstance(exc, openai.InternalServerError):
        if reports_overload(exc):
            return _OVERLOADED
        status = transient or exc.status_code
        if not 500 <= status <= 599:
            status = 500
        return (FailureKind.UPSTREAM, status, True, _upstream_message(status))
    if transient == 429:
        return _RATE_LIMITED
    if reports_overload(exc):
        return _OVERLOADED
    status = transient or getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = 500
    if status == 402:
        return (FailureKind.PERMISSION, 402, False, _BILLING_MESSAGE)
    return (
        FailureKind.UPSTREAM,
        status,
        transient is not None,
        _upstream_message(status),
    )


def _http_status_canonical(status: int) -> _Canonical:
    if status == 402:
        return (FailureKind.PERMISSION, 402, False, _BILLING_MESSAGE)
    if status == 429:
        return _RATE_LIMITED
    if status in (502, 503, 504):
        return _OVERLOADED
    return (
        FailureKind.UPSTREAM,
        status,
        is_backoff_status(status),
        _upstream_message(status),
    )


def _transport_canonical(exc: Exception, transient: int | None) -> _Canonical:
    if isinstance(exc, TimeoutError | httpx.TimeoutException | httpx2.TimeoutException):
        kind = FailureKind.TIMEOUT
    elif isinstance(
        exc, ssl.SSLWantReadError | httpx.NetworkError | httpx2.NetworkError
    ):
        kind = FailureKind.UNAVAILABLE
    elif transient == 429:
        return _RATE_LIMITED
    elif reports_overload(exc):
        return _OVERLOADED
    else:
        kind = FailureKind.UPSTREAM
    retryable = transient is not None or isinstance(exc, _RETRYABLE_TRANSPORT_ERRORS)
    return (kind, 502, retryable, None)


def _is_connection_setup_timeout(exc: Exception) -> bool:
    return isinstance(
        exc, httpx.TimeoutException | httpx2.TimeoutException
    ) and not isinstance(exc, httpx.ReadTimeout | httpx2.ReadTimeout)


def _upstream_message(status: int) -> str:
    return _UNAVAILABLE_MESSAGE if status in (502, 503, 504) else _API_FAILED_MESSAGE


_INVALID_REQUEST: _Canonical = (
    FailureKind.INVALID_REQUEST,
    400,
    False,
    _INVALID_REQUEST_MESSAGE,
)
_RATE_LIMITED: _Canonical = (FailureKind.RATE_LIMIT, 429, True, _RATE_LIMIT_MESSAGE)
_OVERLOADED: _Canonical = (FailureKind.OVERLOADED, 529, True, OVERLOADED_MESSAGE)

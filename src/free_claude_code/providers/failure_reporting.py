"""Turn a classified provider failure into the final client-facing failure."""

import math
import ssl
import time
from dataclasses import replace

import httpx
import httpx2

from free_claude_code.core.diagnostics import (
    extract_upstream_error_detail,
    format_execution_failure_message,
    safe_exception_message,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.canonical_failures import ProviderFailureOverride
from free_claude_code.providers.failure_classification import classify
from free_claude_code.providers.recovery_errors import ProviderRecoveryExhausted


def finalize_provider_failure(
    exc: Exception,
    *,
    provider_name: str,
    read_timeout_s: float | None,
    request_id: str | None,
    provider_failure_override: ProviderFailureOverride | None = None,
    upstream_evidence: bool = False,
) -> ExecutionFailure:
    """Return one detailed canonical failure after provider retries are exhausted.

    ``upstream_evidence`` also reports the upstream error body of a failure the
    provider stream already classified, for clients that speak its protocol.
    """
    remaining = 0
    if isinstance(exc, ProviderRecoveryExhausted):
        exc, retry_not_before = exc.last_error, exc.retry_not_before
        if retry_not_before is not None:
            remaining = math.ceil(retry_not_before - time.monotonic())
    if isinstance(exc, ExecutionFailure):
        message = _with_cooldown(exc.message, remaining)
        if upstream_evidence:
            failure = replace(exc, message=message)
            message = format_execution_failure_message(
                failure,
                extract_upstream_error_detail(exc),
                upstream_name=provider_name,
                request_id=request_id,
            )
            return replace(failure, message=message)
        request_id_line = f"Request ID: {request_id}" if request_id else None
        if request_id_line and request_id_line not in message:
            message = f"{message}\n\n{request_id_line}"
        return replace(exc, message=message)

    failure = (
        provider_failure_override(exc)
        if provider_failure_override is not None
        else None
    )
    if failure is None:
        verdict = classify(exc)
        failure = ExecutionFailure(
            kind=verdict.kind,
            status_code=verdict.status_code,
            message=verdict.message
            or _describe_transport_error(exc, read_timeout_s=read_timeout_s),
            retryable=verdict.retryable,
        )
    failure = replace(failure, message=_with_cooldown(failure.message, remaining))
    message = format_execution_failure_message(
        failure,
        extract_upstream_error_detail(exc),
        upstream_name=provider_name,
        request_id=request_id,
    )
    return replace(failure, message=message)


def _with_cooldown(message: str, remaining: int) -> str:
    if remaining <= 0:
        return message
    message = message.removesuffix(" Please retry shortly.").removesuffix(
        " Please retry."
    )
    unit = "second" if remaining == 1 else "seconds"
    return f"{message} Retry this provider route in {remaining} {unit}."


def _describe_transport_error(exc: Exception, *, read_timeout_s: float | None) -> str:
    """Map raw transport exception types to stable customer-facing wording."""
    if isinstance(exc, httpx.ReadTimeout | httpx2.ReadTimeout):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Provider request timed out."
    if isinstance(
        exc,
        httpx.ConnectTimeout
        | httpx2.ConnectTimeout
        | httpx.ConnectError
        | httpx2.ConnectError,
    ):
        return "Could not connect to provider."
    if isinstance(exc, httpx.RemoteProtocolError | httpx2.RemoteProtocolError):
        return "Provider connection was interrupted before a response was received."
    if isinstance(exc, ssl.SSLWantReadError):
        return "Could not read the provider response."
    if isinstance(exc, TimeoutError):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Request timed out."
    return safe_exception_message(exc)

"""Anthropic error types and envelopes."""

from typing import Any

from free_claude_code.core.diagnostics import redact_sensitive_error_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind

_ANTHROPIC_ERROR_STATUS_CODES = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}

_FAILURE_ERROR_TYPES = {
    FailureKind.INVALID_REQUEST: "invalid_request_error",
    FailureKind.CONTEXT_WINDOW_EXCEEDED: "invalid_request_error",
    FailureKind.AUTHENTICATION: "authentication_error",
    FailureKind.PERMISSION: "permission_error",
    FailureKind.RATE_LIMIT: "rate_limit_error",
    FailureKind.OVERLOADED: "overloaded_error",
    FailureKind.TIMEOUT: "api_error",
    FailureKind.UPSTREAM: "api_error",
    FailureKind.UNAVAILABLE: "api_error",
}


def anthropic_error_type_for_failure(
    failure: FailureKind | ExecutionFailure,
) -> str:
    """Map neutral failure semantics to an Anthropic wire type."""
    if isinstance(failure, ExecutionFailure):
        if failure.kind == FailureKind.PERMISSION and failure.status_code == 402:
            return "billing_error"
        if failure.kind == FailureKind.INVALID_REQUEST:
            if failure.status_code == 404:
                return "not_found_error"
            if failure.status_code == 413:
                return "request_too_large"
        if failure.kind == FailureKind.TIMEOUT and failure.status_code == 504:
            return "timeout_error"
        kind = failure.kind
    else:
        kind = failure
    return _FAILURE_ERROR_TYPES[kind]


def anthropic_error_payload(
    *,
    error_type: str,
    message: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Return one Anthropic-compatible JSON error envelope."""
    payload: dict[str, Any] = {
        "type": "error",
        "error": {
            "type": error_type,
            "message": redact_sensitive_error_text(message),
        },
    }
    if request_id:
        payload["request_id"] = request_id
    return payload


def anthropic_failure_payload(
    failure: ExecutionFailure,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Serialize a canonical execution failure as an Anthropic JSON error."""
    return anthropic_error_payload(
        error_type=anthropic_error_type_for_failure(failure),
        message=_anthropic_failure_message(failure),
        request_id=request_id,
    )


def _anthropic_failure_message(failure: ExecutionFailure) -> str:
    """Add client-recognized wording only at the Anthropic wire boundary."""
    if failure.kind != FailureKind.CONTEXT_WINDOW_EXCEEDED:
        return failure.message
    if failure.message.lstrip().lower().startswith("prompt is too long"):
        return failure.message
    return f"prompt is too long\n\n{failure.message}"


def anthropic_status_for_error_type(error_type: str) -> int:
    """Return the standard HTTP status for an Anthropic error type."""
    return _ANTHROPIC_ERROR_STATUS_CODES.get(error_type, 500)

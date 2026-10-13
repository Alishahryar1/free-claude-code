"""Canonical execution failures and their protocol-owned wire mappings."""

import pytest

from free_claude_code.core.anthropic.errors import (
    anthropic_error_type_for_failure,
    anthropic_failure_payload,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses.errors import (
    openai_error_type_for_failure,
    openai_failure_payload,
)


@pytest.mark.parametrize(
    ("kind", "anthropic_type", "openai_type"),
    [
        (FailureKind.INVALID_REQUEST, "invalid_request_error", "invalid_request_error"),
        (
            FailureKind.CONTEXT_WINDOW_EXCEEDED,
            "invalid_request_error",
            "invalid_request_error",
        ),
        (FailureKind.AUTHENTICATION, "authentication_error", "authentication_error"),
        (FailureKind.PERMISSION, "permission_error", "permission_error"),
        (FailureKind.RATE_LIMIT, "rate_limit_error", "rate_limit_error"),
        (FailureKind.OVERLOADED, "overloaded_error", "overloaded_error"),
        # Existing finalized transport timeouts are exposed as api_error; the
        # semantic kind becomes more precise without changing that wire contract.
        (FailureKind.TIMEOUT, "api_error", "api_error"),
        (FailureKind.UPSTREAM, "api_error", "api_error"),
        (FailureKind.UNAVAILABLE, "api_error", "api_error"),
    ],
)
def test_each_protocol_owns_failure_kind_to_wire_type_mapping(
    kind: FailureKind,
    anthropic_type: str,
    openai_type: str,
) -> None:
    assert anthropic_error_type_for_failure(kind) == anthropic_type
    assert openai_error_type_for_failure(kind) == openai_type


def test_finalized_status_502_timeout_keeps_existing_api_error_wire_type() -> None:
    failure = ExecutionFailure(
        kind=FailureKind.TIMEOUT,
        status_code=502,
        message="Provider request timed out.",
        retryable=True,
    )

    assert anthropic_failure_payload(failure)["error"]["type"] == "api_error"
    assert openai_failure_payload(failure)["error"]["type"] == "api_error"


def test_each_protocol_adds_its_client_compaction_trigger() -> None:
    failure = ExecutionFailure(
        kind=FailureKind.CONTEXT_WINDOW_EXCEEDED,
        status_code=400,
        message="Provider input exceeds its context window.\n\nRequest ID: req_context",
        retryable=False,
    )

    anthropic_error = anthropic_failure_payload(failure)["error"]
    openai_error = openai_failure_payload(failure)["error"]

    assert anthropic_error == {
        "type": "invalid_request_error",
        "message": (
            "prompt is too long\n\nProvider input exceeds its context window."
            "\n\nRequest ID: req_context"
        ),
    }
    assert openai_error == {
        "message": (
            "Provider input exceeds its context window.\n\nRequest ID: req_context"
        ),
        "type": "invalid_request_error",
        "param": None,
        "code": "context_length_exceeded",
    }

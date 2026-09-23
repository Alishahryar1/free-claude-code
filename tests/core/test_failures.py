"""Canonical, protocol-neutral execution failure contracts."""

from dataclasses import FrozenInstanceError, fields, is_dataclass

import pytest

from free_claude_code.core.failures import (
    ExecutionFailure,
    FailureKind,
    find_execution_failure,
)


def test_failure_kind_has_only_protocol_neutral_semantics() -> None:
    assert tuple(FailureKind) == (
        FailureKind.INVALID_REQUEST,
        FailureKind.CONTEXT_WINDOW_EXCEEDED,
        FailureKind.AUTHENTICATION,
        FailureKind.PERMISSION,
        FailureKind.RATE_LIMIT,
        FailureKind.OVERLOADED,
        FailureKind.TIMEOUT,
        FailureKind.UPSTREAM,
        FailureKind.UNAVAILABLE,
    )
    assert tuple(kind.value for kind in FailureKind) == (
        "invalid_request",
        "context_window_exceeded",
        "authentication",
        "permission",
        "rate_limit",
        "overloaded",
        "timeout",
        "upstream",
        "unavailable",
    )


def test_execution_failure_is_the_direct_frozen_slotted_exception() -> None:
    failure = ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="Provider rate limit reached.",
        retryable=True,
    )

    assert is_dataclass(failure)
    assert tuple(field.name for field in fields(failure)) == (
        "kind",
        "status_code",
        "message",
        "retryable",
    )
    assert ExecutionFailure.__slots__ == (
        "kind",
        "status_code",
        "message",
        "retryable",
    )
    assert str(failure) == "Provider rate limit reached."
    assert failure.args == ("Provider rate limit reached.",)

    with pytest.raises(ExecutionFailure) as raised:
        raise failure

    assert raised.value is failure
    with pytest.raises(FrozenInstanceError):
        failure.status_code = 500


def test_execution_failure_uses_exception_identity_not_value_equality() -> None:
    first = ExecutionFailure(
        kind=FailureKind.UPSTREAM,
        status_code=500,
        message="same",
        retryable=True,
    )
    second = ExecutionFailure(
        kind=FailureKind.UPSTREAM,
        status_code=500,
        message="same",
        retryable=True,
    )

    assert first is not second
    assert first != second


def test_find_execution_failure_recurses_through_nested_groups() -> None:
    failure = ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="provider is busy",
        retryable=True,
    )
    grouped = ExceptionGroup(
        "stream and cleanup failed",
        [
            RuntimeError("cleanup failed"),
            ExceptionGroup("provider failed", [failure]),
        ],
    )

    assert find_execution_failure(failure) is failure
    assert find_execution_failure(grouped) is failure


def test_find_execution_failure_leaves_unrelated_groups_unclassified() -> None:
    grouped = BaseExceptionGroup(
        "unrelated failures",
        [RuntimeError("socket closed"), KeyboardInterrupt()],
    )

    assert find_execution_failure(grouped) is None

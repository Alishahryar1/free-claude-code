"""Canonical, protocol-neutral execution failure contracts."""

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

import pytest

from free_claude_code.core.failures import (
    ExecutionFailure,
    FailureKind,
    find_execution_failure,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.stream_recovery import (
    SourceRecoverySnapshot,
    StreamFailureContext,
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
        "stream_context",
    )
    assert ExecutionFailure.__slots__ == (
        "kind",
        "status_code",
        "message",
        "retryable",
        "stream_context",
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


def test_failure_context_survives_replacement_and_group_lookup_without_aliasing() -> (
    None
):
    error: JsonObject = {"param": None}
    payload: JsonObject = {"type": "response.failed", "response": {"error": error}}
    context = StreamFailureContext(
        SourceRecoverySnapshot(unsafe_reason="native_content", can_handoff=False),
        payload,
    )
    original = ExecutionFailure(FailureKind.UPSTREAM, 502, "cut", True, context)
    changed = replace(original, message="upstream cut")
    found = find_execution_failure(ExceptionGroup("wrapped", [changed]))
    assert found is changed and found.stream_context is context
    error["param"] = "changed"
    assert context.responses_failure_payload == {
        "type": "response.failed",
        "response": {"error": {"param": None}},
    }
    assert changed.kind is FailureKind.UPSTREAM and changed.status_code == 502
    assert str(original) == "cut" and str(changed) == "upstream cut"
    with pytest.raises(FrozenInstanceError):
        changed.stream_context = None

"""Canonical provider failures and the wire fields that signal them."""

from collections.abc import Callable, Mapping

from free_claude_code.core.failures import ExecutionFailure, FailureKind

type ProviderFailureOverride = Callable[[Exception], ExecutionFailure | None]
"""A provider hook that maps its own raw failures before generic classification."""

OVERLOADED_MESSAGE = "Provider is currently overloaded. Please retry."
CONTEXT_WINDOW_EXCEEDED_MESSAGE = "Provider input exceeds the model context window."
_CONTEXT_WINDOW_ERROR_CODE = "context_length_exceeded"
_CONTEXT_WINDOW_FINISH_REASON = "model_context_window_exceeded"


def overloaded_provider_failure() -> ExecutionFailure:
    """Return the canonical provider-overload meaning and stable wording."""
    return ExecutionFailure(FailureKind.OVERLOADED, 529, OVERLOADED_MESSAGE, True)


def context_window_exceeded_provider_failure(
    message: str = CONTEXT_WINDOW_EXCEEDED_MESSAGE,
) -> ExecutionFailure:
    """Return the canonical non-retryable provider context-window failure."""
    return ExecutionFailure(FailureKind.CONTEXT_WINDOW_EXCEEDED, 400, message, False)


def is_context_window_error_code(value: object) -> bool:
    """Return whether one structured provider discriminator means context exhaustion."""
    return (
        isinstance(value, str)
        and value.strip().casefold() == _CONTEXT_WINDOW_ERROR_CODE
    )


def is_context_window_finish_reason(value: object) -> bool:
    """Return whether one terminal provider reason means context exhaustion."""
    return (
        isinstance(value, str)
        and value.strip().casefold() == _CONTEXT_WINDOW_FINISH_REASON
    )


def reports_context_window_incomplete(
    event_type: str,
    payload: Mapping[str, object],
) -> bool:
    """Return whether one Responses incomplete terminal reports context exhaustion."""
    if event_type != "response.incomplete":
        return False
    response = payload.get("response")
    if not isinstance(response, Mapping):
        return False
    details = response.get("incomplete_details")
    return isinstance(details, Mapping) and is_context_window_finish_reason(
        details.get("reason")
    )

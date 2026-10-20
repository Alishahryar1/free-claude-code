"""Request-scoped diagnostic fields shared by routing and HTTP delivery."""

from contextvars import ContextVar
from dataclasses import dataclass

from .failures import find_execution_failure


@dataclass
class RequestOutcome:
    """Mutable fields shared with the request's response and cleanup tasks."""

    provider_id: str | None = None
    model: str | None = None
    failure_reason: str | None = None


current_request_outcome: ContextVar[RequestOutcome | None] = ContextVar(
    "request_outcome", default=None
)


def record_request_route(provider_id: str, model: str) -> None:
    outcome = current_request_outcome.get()
    if outcome is not None:
        outcome.provider_id = provider_id
        outcome.model = model


def record_request_failure(reason: str) -> None:
    outcome = current_request_outcome.get()
    if outcome is not None and outcome.failure_reason is None:
        outcome.failure_reason = reason


def record_request_exception(error: BaseException) -> None:
    failure = find_execution_failure(error)
    record_request_failure(failure.kind.value if failure else type(error).__name__)

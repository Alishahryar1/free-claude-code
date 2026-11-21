"""The single result of classifying one upstream failure."""

from dataclasses import dataclass

from free_claude_code.core.failures import FailureKind


@dataclass(frozen=True, slots=True)
class FailureVerdict:
    """Everything provider execution needs to know about one raw failure.

    ``kind``, ``status_code`` and ``retryable`` are the canonical failure a
    client receives. ``message`` is its stable wording, or ``None`` when the
    wording must describe the transport exception itself.
    """

    kind: FailureKind
    status_code: int
    retryable: bool
    message: str | None
    retryable_mid_stream: bool
    """Whether an already-open stream may replay or recover after this failure."""
    transient_status: int | None
    """A 429 or 5xx status the failure reports, for backoff labels and traces."""
    auth_status: int | None
    """A 401 or 403 the failure reports, for credential refresh."""

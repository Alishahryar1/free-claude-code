"""What a failed platform call does next."""

from dataclasses import dataclass
from enum import StrEnum

MAX_ATTEMPTS = 3
"""Calls one delivery may spend on network errors and rate limits together."""


class FailureKind(StrEnum):
    """What went wrong, as the platform adapter read it."""

    NETWORK = "network"
    """The call was lost on the way; sending it again may work."""
    RATE_LIMITED = "rate_limited"
    """The platform asked to wait before calling again."""
    UNMODIFIED = "unmodified"
    """An edit changed nothing; the message already reads that way."""
    GONE = "gone"
    """The message to edit or delete no longer exists or cannot be touched."""
    UNPARSEABLE = "unparseable"
    """The platform could not parse the message's formatting."""
    REJECTED = "rejected"
    """The platform refused the request for any other reason."""


@dataclass(frozen=True, slots=True)
class DeliveryFailure:
    kind: FailureKind
    retry_after: float = 0.0
    """Seconds the platform asked to wait; set for RATE_LIMITED."""


@dataclass(frozen=True, slots=True)
class CallState:
    attempts: int
    """Calls spent from the budget, the failed one included."""
    formatted: bool
    """Whether the call still sends formatted text."""
    suppress_known: bool
    """Whether unmodified and gone messages count as delivered."""


@dataclass(frozen=True, slots=True)
class Ignore:
    """Treat the call as delivered."""


@dataclass(frozen=True, slots=True)
class RetryPlain:
    """Send again without formatting; this does not spend the budget."""


@dataclass(frozen=True, slots=True)
class Retry:
    """Wait ``delay`` seconds, then call again, spending one more attempt."""

    delay: float


@dataclass(frozen=True, slots=True)
class RateLimited:
    """Out of attempts while rate limited; queued work waits ``retry_after``."""

    retry_after: float


@dataclass(frozen=True, slots=True)
class Raise:
    """Fail the call with its error."""


type Decision = Ignore | RetryPlain | Retry | RateLimited | Raise


def decide(failure: DeliveryFailure, state: CallState) -> Decision:
    """Decide what one failed platform call does next."""
    match failure.kind:
        case FailureKind.UNMODIFIED | FailureKind.GONE:
            return Ignore() if state.suppress_known else Raise()
        case FailureKind.UNPARSEABLE:
            return RetryPlain() if state.formatted else Raise()
        case FailureKind.RATE_LIMITED:
            if state.attempts < MAX_ATTEMPTS:
                return Retry(failure.retry_after)
            return RateLimited(failure.retry_after)
        case FailureKind.NETWORK:
            if state.attempts < MAX_ATTEMPTS:
                return Retry(2.0 ** (state.attempts - 1))
            return Raise()
        case FailureKind.REJECTED:
            return Raise()

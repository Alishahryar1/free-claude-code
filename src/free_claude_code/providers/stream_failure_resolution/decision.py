"""Pure retry rules for a failed stream attempt.

Delivery is described by ``content_released``: ``None`` when the request has no
public delivery, otherwise whether any content has reached the client.
"""

from enum import StrEnum


class NextStep(StrEnum):
    """What follows once admission has seen the failure."""

    RETRY = "retry"
    RECOVER = "recover"


def is_hidden_stop(*, normal_stop_seen: bool, content_released: bool | None) -> bool:
    """A model stop the client has not seen proves upstream accepted the attempt."""
    return normal_stop_seen and content_released is False


def can_restart(
    *,
    content_released: bool | None,
    retryable: bool,
    attempts_remaining: int,
    normal_stop_seen: bool,
) -> bool:
    """A clean restart is safe only before content and a normal model stop."""
    return (
        content_released is False
        and retryable
        and attempts_remaining > 0
        and not normal_stop_seen
    )


def next_step(
    *,
    attempted: bool,
    admission_retry_allowed: bool | None,
    retryable: bool,
    content_released: bool | None,
    attempts_remaining: int,
    normal_stop_seen: bool,
) -> NextStep:
    """Retry when the owner of the verdict allows it; otherwise recover.

    Before acceptance admission judges the failure and its verdict is final.
    ``admission_retry_allowed`` is ``None`` once the attempt was accepted; then
    only delivery evidence can justify a clean restart.
    """
    if not attempted:
        return NextStep.RECOVER
    if admission_retry_allowed is not None:
        return NextStep.RETRY if admission_retry_allowed else NextStep.RECOVER
    if can_restart(
        content_released=content_released,
        retryable=retryable,
        attempts_remaining=attempts_remaining,
        normal_stop_seen=normal_stop_seen,
    ):
        return NextStep.RETRY
    return NextStep.RECOVER

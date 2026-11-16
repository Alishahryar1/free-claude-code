"""Pure rules for what follows a failed stream attempt.

Delivery is described by ``content_released``: ``None`` when the request has no
public delivery, otherwise whether any content has reached the client.
"""

from enum import StrEnum


class Stage(StrEnum):
    """How far the failed attempt got."""

    BEFORE_ATTEMPT = "before_attempt"
    """Preparation failed; no provider call was made or counted."""
    OPEN = "open"
    """An attempt was opened, whether or not the provider accepted it."""


class Repair(StrEnum):
    """A change that may let the same request succeed on another attempt."""

    REFRESH = "refresh"
    CORRECT = "correct"


class NextStep(StrEnum):
    """What follows once repairs and admission have been consulted."""

    RETRY = "retry"
    RECOVER = "recover"
    FAIL = "fail"


def is_hidden_stop(*, normal_stop_seen: bool, content_released: bool | None) -> bool:
    """A model stop the client has not seen proves upstream accepted the attempt."""
    return normal_stop_seen and content_released is False


def repairs(
    *,
    stage: Stage,
    auth_status: int | None,
    can_refresh: bool,
    refreshed: bool,
    content_released: bool | None,
    attempt_content_released: bool,
    normal_stop_seen: bool,
    continuing: bool,
) -> tuple[Repair, ...]:
    """Repairs worth asking admission to authorize, in the order to try them.

    A credential refresh is allowed once, and only before the client has seen
    content. Correcting the request is allowed while this attempt's content
    can still be replaced. A model stop proves upstream accepted the request,
    so nothing is repaired after one unless earlier content was delivered by
    a generation that is still replaceable.
    """
    if stage is Stage.BEFORE_ATTEMPT:
        return ()
    if (
        normal_stop_seen
        and content_released is not None
        and (not content_released or continuing)
    ):
        return ()
    found: list[Repair] = []
    if (
        can_refresh
        and auth_status in {401, 403}
        and not refreshed
        and not content_released
    ):
        found.append(Repair.REFRESH)
    if not attempt_content_released:
        found.append(Repair.CORRECT)
    return tuple(found)


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
    stage: Stage,
    admission_retry_allowed: bool | None,
    retryable: bool,
    content_released: bool | None,
    attempts_remaining: int,
    normal_stop_seen: bool,
) -> NextStep:
    """Retry when the owner of the verdict allows it, recover, or fail.

    A failure before any attempt spent nothing, so neither admission nor
    public recovery may repeat it. Before acceptance admission judges the
    failure and its verdict is final. ``admission_retry_allowed`` is ``None``
    once the attempt was accepted; then only delivery evidence can justify a
    clean restart.
    """
    if stage is Stage.BEFORE_ATTEMPT:
        return NextStep.FAIL
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

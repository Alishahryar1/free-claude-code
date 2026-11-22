"""What follows a failed stream attempt, decided from its evidence alone."""

from dataclasses import dataclass

import pytest

from free_claude_code.providers.stream_failure_resolution import (
    NextStep,
    Stage,
    can_restart,
    next_step,
)


@dataclass(frozen=True, slots=True)
class Row:
    stage: Stage = Stage.OPEN
    admission: bool | None = None
    retryable: bool = True
    released: bool | None = False
    attempts_remaining: int = 1
    normal_stop: bool = False


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        # A failure before any attempt is never repeated, whatever was delivered.
        (Row(stage=Stage.BEFORE_ATTEMPT), NextStep.FAIL),
        (Row(stage=Stage.BEFORE_ATTEMPT, released=True), NextStep.FAIL),
        (Row(stage=Stage.BEFORE_ATTEMPT, admission=True), NextStep.FAIL),
        (Row(admission=True), NextStep.RETRY),
        (Row(admission=False), NextStep.RECOVER),
        # Admission's verdict is final even when delivery evidence would allow it.
        (Row(admission=False, retryable=True), NextStep.RECOVER),
        (Row(), NextStep.RETRY),
        (Row(retryable=False), NextStep.RECOVER),
        (Row(released=True), NextStep.RECOVER),
        (Row(released=None), NextStep.RECOVER),
        (Row(attempts_remaining=0), NextStep.RECOVER),
        (Row(normal_stop=True), NextStep.RECOVER),
    ],
)
def test_next_step(row: Row, expected: NextStep) -> None:
    assert (
        next_step(
            stage=row.stage,
            admission_retry_allowed=row.admission,
            retryable=row.retryable,
            content_released=row.released,
            attempts_remaining=row.attempts_remaining,
            normal_stop_seen=row.normal_stop,
        )
        is expected
    )


@pytest.mark.parametrize("released", [None, False, True])
@pytest.mark.parametrize("normal_stop", [False, True])
@pytest.mark.parametrize("retryable", [False, True])
@pytest.mark.parametrize("attempts", [0, 1])
def test_restart_requires_undelivered_retryable_unstopped_stream(
    released: bool | None, normal_stop: bool, retryable: bool, attempts: int
) -> None:
    assert can_restart(
        content_released=released,
        retryable=retryable,
        attempts_remaining=attempts,
        normal_stop_seen=normal_stop,
    ) == (released is False and not normal_stop and retryable and attempts > 0)

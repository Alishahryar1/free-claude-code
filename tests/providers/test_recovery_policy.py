"""Recovery policy decides from values alone: no event loop, clock or controller."""

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from free_claude_code.providers.recovery_policy import (
    EpisodeView,
    Exhaust,
    FailedAttempt,
    FailFromEpisode,
    GiveUp,
    OpenEpisode,
    RearmProbe,
    RecoveryTiming,
    StartCooldown,
    Wait,
    on_failure,
    retry_after_seconds,
    retry_delay,
)
from free_claude_code.providers.recovery_policy.decision import Transition

TIMING = RecoveryTiming(base_delay=2.0, max_delay=60.0, jitter=1.0)
NOW = 100.0


def _failed(
    attempt: int = 1, *, can_retry: bool = True, retry_after: int | None = None
) -> FailedAttempt:
    return FailedAttempt(attempt=attempt, can_retry=can_retry, retry_after=retry_after)


def _open(*, probing: bool = False, leader_idle: bool = False) -> EpisodeView:
    return EpisodeView(
        terminal_until=None,
        reports_cooldown=False,
        probing=probing,
        leader_idle=leader_idle,
    )


def _closed(until: float, *, cooldown: bool) -> EpisodeView:
    return EpisodeView(
        terminal_until=until,
        reports_cooldown=cooldown,
        probing=False,
        leader_idle=False,
    )


@pytest.mark.parametrize(
    ("failed", "episode", "expected"),
    [
        pytest.param(_failed(), None, OpenEpisode(102.5, 2.5), id="open"),
        pytest.param(
            _failed(3), _open(probing=True), RearmProbe(108.5, 8.5), id="rearm"
        ),
        pytest.param(_failed(), _open(), Wait(), id="wait"),
        pytest.param(
            _failed(can_retry=False),
            None,
            Exhaust(102.5, open_episode=True),
            id="exhaust-without-episode",
        ),
        pytest.param(
            _failed(2, can_retry=False),
            _open(probing=True),
            Exhaust(104.5, open_episode=False),
            id="exhaust-probe",
        ),
        pytest.param(
            _failed(can_retry=False), _open(), GiveUp(), id="give-up-outside-probe"
        ),
        pytest.param(
            _failed(retry_after=5),
            _closed(150, cooldown=False),
            FailFromEpisode(),
            id="closed-episode",
        ),
        pytest.param(
            _failed(retry_after=7), None, OpenEpisode(107.0, 7.0), id="hint-lengthens"
        ),
        pytest.param(
            _failed(retry_after=60), None, OpenEpisode(160.0, 60.0), id="hint-at-max"
        ),
        pytest.param(
            _failed(retry_after=600),
            None,
            StartCooldown(
                700.0,
                open_episode=True,
                keep_episode_error=False,
                fail_idle_leader=False,
            ),
            id="cooldown-opens",
        ),
        pytest.param(
            _failed(retry_after=600),
            _open(leader_idle=True),
            StartCooldown(
                700.0,
                open_episode=False,
                keep_episode_error=False,
                fail_idle_leader=True,
            ),
            id="cooldown-fails-idle-leader",
        ),
        pytest.param(
            _failed(retry_after=600, can_retry=False),
            _open(probing=True),
            StartCooldown(
                700.0,
                open_episode=False,
                keep_episode_error=False,
                fail_idle_leader=False,
            ),
            id="cooldown-spares-in-flight-leader",
        ),
        pytest.param(
            _failed(retry_after=600),
            _closed(300, cooldown=True),
            StartCooldown(
                700.0,
                open_episode=False,
                keep_episode_error=False,
                fail_idle_leader=False,
            ),
            id="cooldown-extends",
        ),
        pytest.param(
            _failed(retry_after=120),
            _closed(700, cooldown=True),
            FailFromEpisode(),
            id="cooldown-covered",
        ),
        pytest.param(
            _failed(retry_after=65),
            _closed(170, cooldown=False),
            StartCooldown(
                170.0,
                open_episode=False,
                keep_episode_error=True,
                fail_idle_leader=False,
            ),
            id="cooldown-inside-exhausted-window",
        ),
        pytest.param(
            _failed(retry_after=65),
            _closed(150, cooldown=False),
            StartCooldown(
                165.0,
                open_episode=False,
                keep_episode_error=False,
                fail_idle_leader=False,
            ),
            id="cooldown-outlasts-exhausted-window",
        ),
    ],
)
def test_on_failure(
    failed: FailedAttempt, episode: EpisodeView | None, expected: Transition
) -> None:
    assert on_failure(failed, episode, now=NOW, timing=TIMING, noise=0.5) == expected


@pytest.mark.parametrize(
    ("attempt", "retry_after", "noise", "expected"),
    [
        (1, None, 0.0, 2.0),
        (2, None, 0.0, 4.0),
        (6, None, 0.0, 60.0),
        (9, None, 0.75, 60.75),
        (0, None, 0.0, 2.0),
        (1, 9.0, 0.5, 9.0),
        (1, 1.0, 0.5, 2.5),
    ],
)
def test_retry_delay(
    attempt: int, retry_after: float | None, noise: float, expected: float
) -> None:
    assert retry_delay(attempt, retry_after, TIMING, noise=noise) == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_delay": -1, "max_delay": 1, "jitter": 0}, "base_delay"),
        ({"base_delay": 2, "max_delay": 1, "jitter": 0}, "max_delay"),
        ({"base_delay": 0, "max_delay": 0, "jitter": -1}, "jitter"),
    ],
)
def test_timing_rejects_invalid_bounds(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RecoveryTiming(**kwargs)


NOW_UTC = datetime(2026, 1, 1, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("7", 7.0),
        (" 2.5 ", 2.5),
        ("-3", 0.0),
        (format_datetime(NOW_UTC + timedelta(seconds=90), usegmt=True), 90.0),
        ("Thu, 01 Jan 2026 12:01:00", 60.0),
        (format_datetime(NOW_UTC - timedelta(seconds=90), usegmt=True), 0.0),
        ("invalid", None),
        ("nan", None),
        ("inf", None),
        ("", None),
        (None, None),
        (7, None),
    ],
)
def test_retry_after_seconds(value: int | str | None, expected: float | None) -> None:
    assert retry_after_seconds(value, now=NOW_UTC) == expected

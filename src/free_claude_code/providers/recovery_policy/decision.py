"""What one retryable failure does to its model's recovery episode."""

from dataclasses import dataclass

from .timing import RecoveryTiming, retry_delay


@dataclass(frozen=True, slots=True)
class FailedAttempt:
    """The failed attempt, as the execution that made it sees it."""

    attempt: int
    """Attempts the execution has started, this one included."""
    can_retry: bool
    """Whether the execution has budget left for another attempt."""
    retry_after: float | None
    """Seconds the upstream asked to wait, if it said."""


@dataclass(frozen=True, slots=True)
class EpisodeView:
    """The model's recovery episode at the moment of the failure."""

    terminal_until: float | None
    """When a closed episode reopens; None while it is still recovering."""
    reports_cooldown: bool
    """Whether a closed episode was closed by a Retry-After cooldown."""
    probing: bool
    """Whether the failed attempt is this episode's live probe."""
    leader_idle: bool
    """Whether a leader is waiting between attempts, with no call in flight."""


@dataclass(frozen=True, slots=True)
class FailFromEpisode:
    """A closed episode already covers this failure; fail with its outcome."""


@dataclass(frozen=True, slots=True)
class StartCooldown:
    """Close the episode until ``until`` and fail everyone waiting on it."""

    until: float
    open_episode: bool
    """Open an episode first; none exists yet."""
    keep_episode_error: bool
    """Report the closed episode's error; its window already covers the hint."""
    fail_idle_leader: bool
    """Fail the leader too; it is between attempts, not mid-probe."""


@dataclass(frozen=True, slots=True)
class OpenEpisode:
    """Open an episode led by the failing execution, probing at ``ready_at``."""

    ready_at: float
    delay: float


@dataclass(frozen=True, slots=True)
class RearmProbe:
    """The probe failed; probe again at ``ready_at``."""

    ready_at: float
    delay: float


@dataclass(frozen=True, slots=True)
class Wait:
    """Another execution leads; wait for its probe."""


@dataclass(frozen=True, slots=True)
class Exhaust:
    """The leader is out of attempts; close the episode until ``until``."""

    until: float
    open_episode: bool
    """Open an episode first; none exists yet."""


@dataclass(frozen=True, slots=True)
class GiveUp:
    """Out of attempts, but another execution owns the episode; leave it be."""


type Transition = (
    FailFromEpisode | StartCooldown | OpenEpisode | RearmProbe | Wait | Exhaust | GiveUp
)


def on_failure(
    failed: FailedAttempt,
    episode: EpisodeView | None,
    *,
    now: float,
    timing: RecoveryTiming,
    noise: float,
) -> Transition:
    """Decide what a retryable failure does to its model's recovery episode.

    ``noise`` is the caller's jitter draw, between 0 and ``timing.jitter``.
    """
    retry_after = failed.retry_after
    if retry_after is not None and retry_after > timing.max_delay:
        return _cooldown(episode, until=now + retry_after)
    if episode is not None and episode.terminal_until is not None:
        return FailFromEpisode()

    delay = retry_delay(failed.attempt, retry_after, timing, noise=noise)
    if failed.can_retry:
        if episode is None:
            return OpenEpisode(ready_at=now + delay, delay=delay)
        if episode.probing:
            return RearmProbe(ready_at=now + delay, delay=delay)
        return Wait()
    if episode is None or episode.probing:
        return Exhaust(until=now + delay, open_episode=episode is None)
    return GiveUp()


def _cooldown(episode: EpisodeView | None, *, until: float) -> Transition:
    keep_episode_error = False
    if (
        episode is not None
        and episode.terminal_until is not None
        and episode.terminal_until >= until
    ):
        if episode.reports_cooldown:
            return FailFromEpisode()
        until = episode.terminal_until
        keep_episode_error = True
    return StartCooldown(
        until=until,
        open_episode=episode is None,
        keep_episode_error=keep_episode_error,
        fail_idle_leader=episode is not None and episode.leader_idle,
    )

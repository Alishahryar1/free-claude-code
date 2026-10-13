"""Provider-owned stream holdback and recovery decisions."""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum, StrEnum, auto

from free_claude_code.core.async_tasks import wait_owned
from free_claude_code.core.stream_delivery import StreamDeliveryState

from .failure_policy import RetryableProviderProtocolError

EARLY_HOLDBACK_SECONDS = 0.75
RECOVERY_BUFFER_MAX_BYTES = 65_536


class TruncatedProviderStreamError(RetryableProviderProtocolError):
    """An upstream stream ended without its required terminal marker."""


class RecoveryFailureAction(StrEnum):
    """How one provider stream should respond to an upstream failure."""

    EARLY_RETRY = "early_retry"
    MIDSTREAM_RECOVERY = "midstream_recovery"
    FINAL_ERROR = "final_error"


class HoldbackSignal(Enum):
    """Wake a transport to release its buffer without cancelling its read."""

    EXPIRED = auto()


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Failure decision for one provider stream attempt."""

    action: RecoveryFailureAction
    retryable: bool
    committed: bool
    has_buffered: bool


class RecoveryHoldbackBuffer:
    """Briefly retain SSE so early cutoffs can be retried invisibly."""

    def __init__(
        self,
        *,
        holdback_seconds: float = EARLY_HOLDBACK_SECONDS,
        max_bytes: int = RECOVERY_BUFFER_MAX_BYTES,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._holdback_seconds = holdback_seconds
        self._max_bytes = max_bytes
        self._now = now or time.monotonic
        self._events: list[str] = []
        self._bytes = 0
        self._started_at: float | None = None
        self.committed = False

    def push(self, event: str) -> list[str]:
        if self.committed:
            return [event]
        if self._started_at is None:
            self._started_at = self._now()
        self._events.append(event)
        self._bytes += len(event.encode("utf-8", errors="replace"))
        if (
            self._bytes >= self._max_bytes
            or self._now() - self._started_at >= self._holdback_seconds
        ):
            return self.flush()
        return []

    def flush(self) -> list[str]:
        if self.committed:
            return []
        self.committed = True
        events = self._events
        self._events = []
        self._bytes = 0
        self._started_at = None
        return events

    def discard(self) -> None:
        self._events = []
        self._bytes = 0
        self._started_at = None

    @property
    def has_buffered(self) -> bool:
        return bool(self._events)

    @property
    def remaining_delay(self) -> float | None:
        if self.committed or self._started_at is None:
            return None
        return max(0.0, self._holdback_seconds - (self._now() - self._started_at))


class _HoldbackReader[EventT](AsyncIterator[EventT | HoldbackSignal]):
    """Borrow a source and retain only the read interrupted by a buffer deadline."""

    def __init__(
        self, source: AsyncIterator[EventT], remaining_delay: Callable[[], float | None]
    ) -> None:
        self._source = source
        self._remaining_delay = remaining_delay
        self._pending: asyncio.Task[EventT] | None = None
        self._deadline_checked = False

    def __aiter__(self) -> _HoldbackReader[EventT]:
        return self

    async def __anext__(self) -> EventT | HoldbackSignal:
        pending = self._pending
        delay = self._remaining_delay()
        if pending is not None and pending.done():
            self._pending = None
            self._deadline_checked = delay == 0
            return pending.result()
        # Check one ready event at expiry, then release even if it was metadata.
        if delay == 0 and self._deadline_checked:
            return HoldbackSignal.EXPIRED
        if pending is None:
            if delay is None:
                return await anext(self._source)
            pending = self._pending = asyncio.create_task(self._read())
        await asyncio.wait({pending}, timeout=delay)
        if pending.done():
            self._pending = None
            self._deadline_checked = self._remaining_delay() == 0
            return pending.result()
        return HoldbackSignal.EXPIRED

    async def _read(self) -> EventT:
        return await anext(self._source)

    async def aclose(self) -> None:
        pending, self._pending = self._pending, None
        if pending is None:
            return
        if not pending.done():
            pending.cancel()

        async def drain() -> None:
            await asyncio.gather(pending, return_exceptions=True)

        await wait_owned(asyncio.create_task(drain()))


class RecoveryController:
    """Own commit-boundary holdback for one provider stream lifecycle."""

    def __init__(self, delivery: StreamDeliveryState | None = None) -> None:
        self._holdback = RecoveryHoldbackBuffer()
        self._delivery = delivery

    @property
    def committed(self) -> bool:
        return self._holdback.committed

    @property
    def has_buffered(self) -> bool:
        return self._holdback.has_buffered

    @property
    def remaining_delay(self) -> float | None:
        return self._holdback.remaining_delay

    @asynccontextmanager
    async def read_stream[EventT](
        self, source: AsyncIterator[EventT]
    ) -> AsyncIterator[_HoldbackReader[EventT]]:
        reader = _HoldbackReader(source, lambda: self.remaining_delay)
        try:
            yield reader
        finally:
            await reader.aclose()

    def push(self, event: str) -> list[str]:
        return self._holdback.push(event)

    def flush(self) -> list[str]:
        return self._holdback.flush()

    def discard(self) -> None:
        self._holdback.discard()

    def flush_uncommitted(self, decision: RecoveryDecision) -> list[str]:
        if not decision.committed and decision.has_buffered:
            return self.flush()
        return []

    def advance_failure(
        self,
        *,
        retryable: bool,
        stream_opened: bool,
        generated_output: bool,
        complete_tool_salvageable: bool,
        attempts_remaining: int,
        normal_stop_seen: bool = False,
    ) -> RecoveryDecision:
        committed = self._holdback.committed
        has_buffered = self._holdback.has_buffered
        retry_available = attempts_remaining > 0
        reserve_last_attempt_for_recovery = generated_output and attempts_remaining == 1

        public_hidden = (
            self._delivery is not None and not self._delivery.content_released
        )

        if (
            stream_opened
            and can_retry_undelivered_stream(
                self._delivery,
                retryable=retryable,
                attempts_remaining=attempts_remaining,
                normal_stop_seen=normal_stop_seen,
            )
        ) or (
            not public_hidden
            and retryable
            and retry_available
            and stream_opened
            and not committed
            and not complete_tool_salvageable
            and not reserve_last_attempt_for_recovery
        ):
            self._holdback.discard()
            self._holdback = RecoveryHoldbackBuffer()
            return RecoveryDecision(
                action=RecoveryFailureAction.EARLY_RETRY,
                retryable=True,
                committed=False,
                has_buffered=has_buffered,
            )

        if (
            not public_hidden
            and retryable
            and generated_output
            and (retry_available or complete_tool_salvageable)
        ):
            return RecoveryDecision(
                action=RecoveryFailureAction.MIDSTREAM_RECOVERY,
                retryable=True,
                committed=committed,
                has_buffered=has_buffered,
            )

        return RecoveryDecision(
            action=RecoveryFailureAction.FINAL_ERROR,
            retryable=retryable,
            committed=committed,
            has_buffered=has_buffered,
        )


def can_retry_undelivered_stream(
    delivery: StreamDeliveryState | None,
    *,
    retryable: bool,
    attempts_remaining: int,
    normal_stop_seen: bool,
) -> bool:
    """A clean restart is safe only before content and a normal model stop."""
    return (
        delivery is not None
        and not delivery.content_released
        and retryable
        and attempts_remaining > 0
        and not normal_stop_seen
    )

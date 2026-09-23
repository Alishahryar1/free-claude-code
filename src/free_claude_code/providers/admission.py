"""Provider-owned admission, concurrency, and coordinated retry lifecycle."""

import asyncio
import math
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import TypeVar

from loguru import logger

from free_claude_code.core.rate_limit import StrictSlidingWindowLimiter
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.failure_policy import (
    ProviderFailureOverride,
    ProviderRecoveryExhausted,
    is_retryable_provider_error,
    retryable_upstream_status,
)

T = TypeVar("T")

UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS = 5
DEFAULT_UPSTREAM_BASE_DELAY = 2.0
DEFAULT_UPSTREAM_MAX_DELAY = 60.0
DEFAULT_UPSTREAM_JITTER = 1.0


class ProviderOperationKind(StrEnum):
    """Safe trace category for one physical provider call."""

    MODEL_DISCOVERY = "model_discovery"
    GENERATION = "generation"
    CONTINUATION = "continuation"
    TOOL_REPAIR = "tool_repair"


class ProviderExecutionState(StrEnum):
    """Terminal state of one logical provider operation."""

    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ProviderCorrectionAction(StrEnum):
    """Whether one deterministic request correction may consume another attempt."""

    RETRY = "retry"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class ProviderFailureDecision:
    """Immutable provider qualification and execution-budget decision."""

    retryable: bool
    retry_allowed: bool


@dataclass(frozen=True, slots=True)
class _AttemptClaim:
    execution_id: str
    ordinal: int
    operation_kind: ProviderOperationKind


class ProviderExecution:
    """Own one logical provider operation and its physical-attempt budget."""

    def __init__(
        self,
        controller: ProviderAdmissionController,
        *,
        max_attempts: int,
        request_id: str | None,
    ) -> None:
        self._controller = controller
        self._execution_id = str(uuid.uuid4())
        self._max_attempts = max_attempts
        self._request_id = request_id
        self._attempts_started = 0
        self._active_claim: _AttemptClaim | None = None
        self._last_failure: Exception | None = None
        self._state = ProviderExecutionState.ACTIVE

    @property
    def execution_id(self) -> str:
        return self._execution_id

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @property
    def request_id(self) -> str | None:
        return self._request_id

    @property
    def attempts_started(self) -> int:
        return self._attempts_started

    @property
    def can_attempt(self) -> bool:
        return (
            self._state is ProviderExecutionState.ACTIVE
            and self._attempts_started < self._max_attempts
        )

    @property
    def attempts_remaining(self) -> int:
        if self._state is not ProviderExecutionState.ACTIVE:
            return 0
        return max(0, self._max_attempts - self._attempts_started)

    @property
    def state(self) -> ProviderExecutionState:
        return self._state

    @property
    def last_failure(self) -> Exception | None:
        return self._last_failure

    async def open_attempt(
        self,
        operation_kind: ProviderOperationKind,
    ) -> ProviderAttempt:
        """Open the sole active physical call for this execution."""
        return await self._controller._open_attempt(self, operation_kind)

    async def run_call(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        operation_kind: ProviderOperationKind,
        provider_failure_override: ProviderFailureOverride | None = None,
    ) -> T:
        """Run a callable that performs exactly one provider call per invocation."""
        try:
            while self.can_attempt:
                attempt = await self.open_attempt(operation_kind)
                try:
                    result = await fn()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    decision = await attempt.fail(
                        error,
                        provider_failure_override=provider_failure_override,
                    )
                    if not decision.retry_allowed:
                        raise
                else:
                    await attempt.accept()
                    self.succeed()
                    return result
                finally:
                    await attempt.aclose()
        except asyncio.CancelledError:
            self.abandon()
            raise
        except Exception as error:
            self.fail(error)
            raise

        if self._last_failure is not None:
            self.fail(self._last_failure)
            raise self._last_failure
        self.abandon()
        raise RuntimeError("provider execution ended without an attempt outcome")

    def succeed(self) -> None:
        """Mark the complete logical provider operation successful."""
        if self._state is ProviderExecutionState.ACTIVE:
            self._state = ProviderExecutionState.SUCCEEDED

    def fail(self, error: Exception) -> None:
        """Mark the complete logical provider operation failed."""
        if self._state is ProviderExecutionState.ACTIVE:
            self._last_failure = error
            self._state = ProviderExecutionState.FAILED

    def abandon(self) -> None:
        """Mark an unfinished logical provider operation abandoned."""
        if self._state is ProviderExecutionState.ACTIVE:
            self._state = ProviderExecutionState.ABANDONED

    def _claim_attempt(
        self,
        operation_kind: ProviderOperationKind,
    ) -> _AttemptClaim:
        if not self.can_attempt:
            raise RuntimeError("provider execution is terminal or exhausted")
        if self._active_claim is not None:
            raise RuntimeError("provider execution already has an active attempt")
        self._attempts_started += 1
        claim = _AttemptClaim(
            execution_id=self._execution_id,
            ordinal=self._attempts_started,
            operation_kind=operation_kind,
        )
        self._active_claim = claim
        return claim

    def _close_attempt(self, claim: _AttemptClaim) -> None:
        if self._active_claim == claim:
            self._active_claim = None

    def _record_failure(self, error: Exception) -> None:
        if self._state is ProviderExecutionState.ACTIVE:
            self._last_failure = error

    def _fail_recovery(self, error: Exception) -> None:
        self.fail(error)

    def _terminal_failure(self) -> Exception | None:
        if self._state is ProviderExecutionState.FAILED:
            return self._last_failure
        return None


@dataclass(frozen=True, slots=True)
class _GatePermit:
    generation: int | None
    probe: bool


@dataclass(slots=True)
class _RecoveryEpisode:
    generation: int
    leader: ProviderExecution | None
    ready_at: float
    last_error: Exception
    waiters: set[ProviderExecution] = field(default_factory=set)
    probe_active: bool = False
    terminal_until: float | None = None


class ProviderAttempt:
    """One admitted upstream attempt and its held concurrency slot."""

    def __init__(
        self,
        controller: ProviderAdmissionController,
        execution: ProviderExecution,
        permit: _GatePermit,
        claim: _AttemptClaim,
    ) -> None:
        self._controller = controller
        self._execution = execution
        self._permit = permit
        self._claim = claim
        self._resolved = False
        self._accepted = False
        self._closed = False

    @property
    def accepted(self) -> bool:
        """Return whether upstream acceptance has resolved this attempt."""
        return self._accepted

    async def __aenter__(self) -> ProviderAttempt:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc, traceback
        await self.aclose()

    async def accept(self) -> None:
        """Record a valid upstream response and close a recovery episode if probing."""
        if self._resolved or self._closed:
            return
        await self._controller._attempt_accepted(self._execution, self._permit)
        self._resolve("accepted", accepted=True)

    async def correct(self, error: Exception) -> ProviderCorrectionAction:
        """Resolve one deterministic request correction without provider backoff."""
        if self._resolved or self._closed:
            return ProviderCorrectionAction.FINAL
        self._execution._record_failure(error)
        status = _exception_status(error)
        if self._execution.can_attempt:
            await self._controller._attempt_corrected(self._execution, self._permit)
            self._resolve("corrected", status=status, error=error)
            return ProviderCorrectionAction.RETRY
        await self._controller._attempt_rejected(self._execution, self._permit)
        self._execution.fail(error)
        self._resolve("rejected", status=status, error=error)
        return ProviderCorrectionAction.FINAL

    async def fail(
        self,
        error: Exception,
        *,
        provider_failure_override: ProviderFailureOverride | None = None,
    ) -> ProviderFailureDecision:
        """Classify one failure and return its immutable retry decision."""
        if self._resolved or self._closed:
            return ProviderFailureDecision(retryable=False, retry_allowed=False)
        effective_error = (
            provider_failure_override(error)
            if provider_failure_override is not None
            else None
        )
        if effective_error is None:
            effective_error = error
        status = retryable_upstream_status(effective_error)
        retryable = is_retryable_provider_error(effective_error)
        self._execution._record_failure(error)
        if not retryable:
            await self._controller._attempt_rejected(self._execution, self._permit)
            self._execution.fail(error)
            self._resolve("rejected", status=status, error=effective_error)
            return ProviderFailureDecision(retryable=False, retry_allowed=False)
        await self._controller._attempt_failed(
            self._execution,
            self._permit,
            error=error,
            status=status,
        )
        retry_allowed = self._execution.can_attempt
        if not retry_allowed:
            self._execution.fail(error)
        self._resolve(
            "retryable_failure",
            status=status,
            error=effective_error,
            retry_allowed=retry_allowed,
        )
        return ProviderFailureDecision(
            retryable=True,
            retry_allowed=retry_allowed,
        )

    async def aclose(self) -> None:
        """Release attempt ownership and its concurrency slot exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            if not self._resolved:
                try:
                    await asyncio.shield(
                        self._controller._attempt_abandoned(
                            self._execution,
                            self._permit,
                        )
                    )
                finally:
                    self._resolve("abandoned")
        finally:
            self._execution._close_attempt(self._claim)
            self._controller._release_concurrency()

    def _resolve(
        self,
        outcome: str,
        *,
        accepted: bool = False,
        status: int | None = None,
        error: BaseException | None = None,
        retry_allowed: bool | None = None,
    ) -> None:
        if self._resolved:
            return
        self._resolved = True
        self._accepted = accepted
        trace_event(
            stage="provider",
            event="provider.attempt.resolved",
            source="provider",
            provider=self._controller._provider_name,
            request_id=self._execution.request_id,
            execution_id=self._execution.execution_id,
            operation_kind=self._claim.operation_kind.value,
            attempt=self._claim.ordinal,
            max_attempts=self._execution.max_attempts,
            probe=self._permit.probe,
            recovery_generation=self._permit.generation,
            outcome=outcome,
            status_code=status,
            exc_type=(None if error is None else type(error).__name__),
            retry_allowed=retry_allowed,
        )


class ProviderAdmissionController:
    """Coordinate one provider's rate, concurrency, and recovery state.

    Normal attempts pass through a strict sliding window and concurrency bulkhead.
    The first shared transient failure opens one recovery episode. Exactly one
    logical execution owns its half-open probes; concurrent callers wait for that
    episode instead of starting independent retry loops.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
        max_attempts: int = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
        base_delay: float = DEFAULT_UPSTREAM_BASE_DELAY,
        max_delay: float = DEFAULT_UPSTREAM_MAX_DELAY,
        jitter: float = DEFAULT_UPSTREAM_JITTER,
    ) -> None:
        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")
        if base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if max_delay < base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if jitter < 0:
            raise ValueError("jitter must be >= 0")

        self._provider_name = provider_name
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        self._proactive_limiter = StrictSlidingWindowLimiter(
            rate_limit, float(rate_window)
        )
        self._concurrency_sem = asyncio.Semaphore(max_concurrency)
        self._condition = asyncio.Condition()
        self._episode: _RecoveryEpisode | None = None
        self._next_generation = 1
        logger.info(
            "Provider admission initialized for {} ({} req / {}s, "
            "max_concurrency={}, max_attempts={})",
            provider_name,
            rate_limit,
            rate_window,
            max_concurrency,
            max_attempts,
        )

    def start_execution(
        self,
        *,
        request_id: str | None = None,
    ) -> ProviderExecution:
        """Create the sole lifecycle owner for one logical provider operation."""
        return ProviderExecution(
            self,
            max_attempts=self._max_attempts,
            request_id=request_id,
        )

    async def _open_attempt(
        self,
        execution: ProviderExecution,
        operation_kind: ProviderOperationKind,
    ) -> ProviderAttempt:
        """Wait for provider admission and hold one active-operation slot."""
        if not isinstance(operation_kind, ProviderOperationKind):
            raise TypeError("operation_kind must be a ProviderOperationKind")
        if (terminal_error := execution._terminal_failure()) is not None:
            raise ProviderRecoveryExhausted(terminal_error)
        if not execution.can_attempt:
            raise RuntimeError("provider execution is terminal or exhausted")
        if execution._active_claim is not None:
            raise RuntimeError("provider execution already has an active attempt")

        while True:
            permit = await self._wait_for_gate(execution)
            slot_acquired = False
            claim: _AttemptClaim | None = None
            try:
                admitted = await self._proactive_limiter.acquire_if(
                    lambda permit=permit: self._permit_is_current(execution, permit)
                )
                if not admitted:
                    await self._abandon_probe_permit(execution, permit)
                    continue
                await self._concurrency_sem.acquire()
                slot_acquired = True
                if not self._permit_is_current(execution, permit):
                    self._concurrency_sem.release()
                    slot_acquired = False
                    await self._abandon_probe_permit(execution, permit)
                    continue
                claim = execution._claim_attempt(operation_kind)
                attempt = ProviderAttempt(self, execution, permit, claim)
                trace_event(
                    stage="provider",
                    event="provider.attempt.started",
                    source="provider",
                    provider=self._provider_name,
                    request_id=execution.request_id,
                    execution_id=execution.execution_id,
                    operation_kind=operation_kind.value,
                    attempt=claim.ordinal,
                    max_attempts=execution.max_attempts,
                    probe=permit.probe,
                    recovery_generation=permit.generation,
                )
                return attempt
            except BaseException:
                if claim is not None:
                    execution._close_attempt(claim)
                if slot_acquired:
                    self._concurrency_sem.release()
                await self._abandon_probe_permit(execution, permit)
                raise

    async def _wait_for_gate(self, execution: ProviderExecution) -> _GatePermit:
        while True:
            if (terminal_error := execution._terminal_failure()) is not None:
                raise ProviderRecoveryExhausted(terminal_error)
            sleep_delay: float | None = None
            claimed_generation: int | None = None
            async with self._condition:
                episode = self._episode
                if episode is None:
                    return _GatePermit(generation=None, probe=False)

                now = time.monotonic()
                if episode.terminal_until is not None:
                    if now < episode.terminal_until:
                        raise ProviderRecoveryExhausted(episode.last_error)
                    episode = self._start_recovery_episode(
                        leader=execution,
                        ready_at=now,
                        last_error=episode.last_error,
                    )

                if episode.leader is None:
                    episode.leader = execution
                    episode.waiters.discard(execution)
                if episode.leader is execution:
                    if episode.probe_active:
                        await self._condition.wait()
                        continue
                    sleep_delay = max(0.0, episode.ready_at - now)
                    claimed_generation = episode.generation
                    if sleep_delay == 0:
                        episode.probe_active = True
                        return self._probe_permit(execution, episode)
                else:
                    episode.waiters.add(execution)
                    try:
                        await self._condition.wait()
                    except asyncio.CancelledError:
                        current = self._episode
                        if (
                            current is not None
                            and current.generation == episode.generation
                        ):
                            current.waiters.discard(execution)
                        raise
                    continue

            if sleep_delay is None or claimed_generation is None:
                continue
            try:
                if sleep_delay > 0:
                    logger.warning(
                        "Provider {} recovery active, waiting {:.1f}s for one probe",
                        self._provider_name,
                        sleep_delay,
                    )
                    await asyncio.sleep(sleep_delay)
                async with self._condition:
                    episode = self._episode
                    if (
                        episode is not None
                        and episode.generation == claimed_generation
                        and episode.terminal_until is None
                        and episode.leader is execution
                        and not episode.probe_active
                    ):
                        episode.probe_active = True
                        return self._probe_permit(execution, episode)
            except asyncio.CancelledError:
                await self._abandon_waiting_leader(execution, claimed_generation)
                raise

    def _probe_permit(
        self,
        execution: ProviderExecution,
        episode: _RecoveryEpisode,
    ) -> _GatePermit:
        trace_event(
            stage="provider",
            event="provider.recovery.probe",
            source="provider",
            provider=self._provider_name,
            request_id=execution.request_id,
            generation=episode.generation,
            attempt=execution.attempts_started + 1,
            max_attempts=execution.max_attempts,
        )
        return _GatePermit(generation=episode.generation, probe=True)

    def _permit_is_current(
        self,
        execution: ProviderExecution,
        permit: _GatePermit,
    ) -> bool:
        episode = self._episode
        if not permit.probe:
            return episode is None
        return (
            episode is not None
            and episode.generation == permit.generation
            and episode.terminal_until is None
            and episode.leader is execution
            and episode.probe_active
        )

    async def _attempt_accepted(
        self,
        execution: ProviderExecution,
        permit: _GatePermit,
    ) -> None:
        if not permit.probe:
            return
        async with self._condition:
            episode = self._matching_probe(execution, permit)
            if episode is None:
                return
            self._episode = None
            self._condition.notify_all()
        trace_event(
            stage="provider",
            event="provider.recovery.closed",
            source="provider",
            provider=self._provider_name,
            request_id=execution.request_id,
            generation=permit.generation,
            attempt=execution.attempts_started,
            outcome="success",
        )

    async def _attempt_corrected(
        self,
        execution: ProviderExecution,
        permit: _GatePermit,
    ) -> None:
        if not permit.probe:
            return
        async with self._condition:
            episode = self._matching_probe(execution, permit)
            if episode is None:
                return
            episode.probe_active = False
            episode.ready_at = time.monotonic()
            self._condition.notify_all()

    async def _attempt_rejected(
        self,
        execution: ProviderExecution,
        permit: _GatePermit,
    ) -> None:
        """Close a probe episode when upstream responds with a final rejection."""
        if not permit.probe:
            return
        async with self._condition:
            episode = self._matching_probe(execution, permit)
            if episode is None:
                return
            self._episode = None
            self._condition.notify_all()
        trace_event(
            stage="provider",
            event="provider.recovery.closed",
            source="provider",
            provider=self._provider_name,
            request_id=execution.request_id,
            generation=permit.generation,
            attempt=execution.attempts_started,
            outcome="rejected",
        )

    async def _attempt_abandoned(
        self,
        execution: ProviderExecution,
        permit: _GatePermit,
    ) -> None:
        if not permit.probe:
            return
        async with self._condition:
            episode = self._matching_probe(execution, permit)
            if episode is None:
                return
            episode.leader = None
            episode.probe_active = False
            episode.ready_at = time.monotonic()
            self._condition.notify_all()

    async def _attempt_failed(
        self,
        execution: ProviderExecution,
        permit: _GatePermit,
        *,
        error: Exception,
        status: int | None,
    ) -> None:
        can_retry = execution.can_attempt
        delay = self._retry_delay(error, execution.attempts_started)
        became_leader = False
        exhausted_episode = False

        async with self._condition:
            episode = self._episode
            matching_probe = self._matching_probe(execution, permit)
            if can_retry:
                if episode is None:
                    episode = self._start_recovery_episode(
                        leader=execution,
                        ready_at=time.monotonic() + delay,
                        last_error=error,
                    )
                    became_leader = True
                elif matching_probe is not None:
                    episode.last_error = error
                    episode.probe_active = False
                    episode.ready_at = time.monotonic() + delay
                    became_leader = True
                elif episode.terminal_until is not None:
                    execution._fail_recovery(episode.last_error)
                else:
                    episode.waiters.add(execution)
                self._condition.notify_all()
            elif episode is None or matching_probe is not None:
                terminal_delay = self._retry_delay(
                    error,
                    execution.attempts_started,
                )
                if episode is None:
                    episode = self._start_recovery_episode(
                        leader=None,
                        ready_at=time.monotonic(),
                        last_error=error,
                        request_id=execution.request_id,
                    )
                episode.last_error = error
                episode.leader = None
                episode.probe_active = False
                episode.terminal_until = time.monotonic() + terminal_delay
                for waiter in episode.waiters:
                    waiter._fail_recovery(error)
                episode.waiters.clear()
                exhausted_episode = True
                self._condition.notify_all()

        label = self._failure_label(status, error)
        if became_leader:
            logger.warning(
                "{}, attempt {}/{} failed; one provider recovery probe in {:.1f}s",
                label,
                execution.attempts_started,
                execution.max_attempts,
                delay,
            )
            trace_event(
                stage="provider",
                event="provider.retry.scheduled",
                source="provider",
                provider=self._provider_name,
                request_id=execution.request_id,
                status_code=status,
                exc_type=type(error).__name__,
                attempt=execution.attempts_started,
                max_attempts=execution.max_attempts,
                delay_s=round(delay, 3),
                coordinated=True,
            )
        elif can_retry:
            trace_event(
                stage="provider",
                event="provider.retry.coalesced",
                source="provider",
                provider=self._provider_name,
                request_id=execution.request_id,
                status_code=status,
                exc_type=type(error).__name__,
                attempt=execution.attempts_started,
                max_attempts=execution.max_attempts,
            )
        else:
            logger.warning(
                "{} retry exhausted (attempts={})",
                label,
                execution.attempts_started,
            )
            trace_event(
                stage="provider",
                event="provider.retry.exhausted",
                source="provider",
                provider=self._provider_name,
                request_id=execution.request_id,
                status_code=status,
                exc_type=type(error).__name__,
                attempts=execution.attempts_started,
                episode_exhausted=exhausted_episode,
            )

    def _start_recovery_episode(
        self,
        *,
        leader: ProviderExecution | None,
        ready_at: float,
        last_error: Exception,
        request_id: str | None = None,
    ) -> _RecoveryEpisode:
        episode = _RecoveryEpisode(
            generation=self._next_generation,
            leader=leader,
            ready_at=ready_at,
            last_error=last_error,
        )
        self._next_generation += 1
        self._episode = episode
        trace_event(
            stage="provider",
            event="provider.recovery.opened",
            source="provider",
            provider=self._provider_name,
            request_id=(leader.request_id if leader is not None else request_id),
            generation=episode.generation,
        )
        return episode

    def _matching_probe(
        self,
        execution: ProviderExecution,
        permit: _GatePermit,
    ) -> _RecoveryEpisode | None:
        episode = self._episode
        if (
            not permit.probe
            or episode is None
            or episode.generation != permit.generation
            or episode.leader is not execution
            or not episode.probe_active
        ):
            return None
        return episode

    async def _abandon_probe_permit(
        self,
        execution: ProviderExecution,
        permit: _GatePermit,
    ) -> None:
        if not permit.probe:
            return
        async with self._condition:
            episode = self._matching_probe(execution, permit)
            if episode is None:
                return
            episode.leader = None
            episode.probe_active = False
            episode.ready_at = time.monotonic()
            self._condition.notify_all()

    async def _abandon_waiting_leader(
        self,
        execution: ProviderExecution,
        generation: int,
    ) -> None:
        async with self._condition:
            episode = self._episode
            if (
                episode is None
                or episode.generation != generation
                or episode.leader is not execution
                or episode.probe_active
            ):
                return
            episode.leader = None
            self._condition.notify_all()

    def _release_concurrency(self) -> None:
        self._concurrency_sem.release()

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        exponent = max(0, attempt - 1)
        backoff = min(self._base_delay * (2**exponent), self._max_delay)
        backoff += random.uniform(0, self._jitter)
        retry_after = _retry_after_seconds(error)
        return max(backoff, retry_after or 0.0)

    @staticmethod
    def _failure_label(status: int | None, error: Exception) -> str:
        if status == 429:
            return "Rate limited (429)"
        if status is not None:
            return f"Upstream server error ({status})"
        return f"Provider transient error ({type(error).__name__})"


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if not isinstance(value, str) or not value.strip():
        return None
    stripped = value.strip()
    try:
        seconds = float(stripped)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(stripped)
        except TypeError, ValueError, OverflowError:
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - datetime.now(UTC)).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _exception_status(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None

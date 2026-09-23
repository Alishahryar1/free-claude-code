"""Platform-neutral voice note helpers."""

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .models import MessageScope


async def _await_owned_task[T](
    task: asyncio.Task[T],
) -> tuple[T, asyncio.CancelledError | None]:
    """Finish an owned task before returning any caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    current = asyncio.current_task()
    while True:
        cancelling_before = current.cancelling() if current is not None else 0
        try:
            return await asyncio.shield(task), cancellation
        except asyncio.CancelledError as exc:
            if current is None or (
                current.cancelling() <= cancelling_before and task.done()
            ):
                raise
            cancellation = cancellation or exc


class Transcriber(Protocol):
    """Consumer-owned voice transcription boundary."""

    async def transcribe(self, file_path: Path) -> str: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PendingVoiceClaim:
    """Opaque ownership token for one pending voice-note generation."""

    scope: MessageScope
    voice_message_id: str
    claim_id: str


_current_voice_claim: ContextVar[PendingVoiceClaim | None] = ContextVar(
    "current_voice_claim",
    default=None,
)


class VoiceHandoffOutcome(Enum):
    """Exclusive outcome of publishing one transcribed voice message."""

    REJECTED = "rejected"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class VoiceCancellationResult:
    """Released ownership for one successfully cancelled user voice note."""

    scope: MessageScope
    voice_message_id: str
    status_message_id: str | None
    delete_message_ids: frozenset[str]


@dataclass(slots=True)
class _PendingVoice:
    claim: PendingVoiceClaim
    status_message_id: str | None = None
    handoff_task: asyncio.Task[None] | None = None


class PendingVoiceRegistry:
    """Own atomic reservation, cancellation, and handoff of voice notes."""

    def __init__(self) -> None:
        self._pending: dict[tuple[MessageScope, str], _PendingVoice] = {}
        self._lock = asyncio.Lock()
        self._active_cancellations: dict[PendingVoiceClaim, int] = {}

    async def reserve(
        self,
        scope: MessageScope,
        voice_message_id: str,
    ) -> PendingVoiceClaim | None:
        async with self._lock:
            key = (scope, voice_message_id)
            if key in self._pending:
                return None
            claim = PendingVoiceClaim(
                scope=scope,
                voice_message_id=voice_message_id,
                claim_id=uuid4().hex,
            )
            self._pending[key] = _PendingVoice(claim=claim)
            return claim

    async def bind_status(
        self,
        claim: PendingVoiceClaim,
        status_message_id: str,
    ) -> bool:
        async with self._lock:
            entry = self._entry_for_claim(claim)
            if entry is None:
                return False
            if entry.status_message_id is not None:
                return entry.status_message_id == status_message_id
            status_key = (claim.scope, status_message_id)
            existing = self._pending.get(status_key)
            if existing is not None and existing is not entry:
                return False
            entry.status_message_id = status_message_id
            self._pending[status_key] = entry
            return True

    async def handoff(
        self,
        claim: PendingVoiceClaim,
        callback: Callable[[], Awaitable[None]],
    ) -> VoiceHandoffOutcome:
        """Run a published handoff while retaining its cancellable ownership."""
        async with self._lock:
            entry = self._entry_for_claim(claim)
            if (
                entry is None
                or entry.status_message_id is None
                or entry.handoff_task is not None
            ):
                return VoiceHandoffOutcome.REJECTED
            task = asyncio.create_task(
                self._run_callback(claim, callback),
                name=f"voice-handoff-{claim.claim_id}",
            )
            entry.handoff_task = task

        current = asyncio.current_task()
        cancelling_before = current.cancelling() if current is not None else 0
        try:
            await asyncio.shield(task)
        except BaseException as error:
            caller_cancelled = (
                isinstance(error, asyncio.CancelledError)
                and current is not None
                and (current.cancelling() > cancelling_before or not task.done())
            )
            if caller_cancelled:
                task.cancel()
                child_error: BaseException | None = None
                try:
                    await self._drain((task,))
                except BaseException as drained_error:
                    child_error = drained_error
                await self._finish_ownership(entry)
                if child_error is not None:
                    raise child_error from None
                raise error from None

            completed, cancellation = await self._finish_ownership(entry)
            if cancellation is not None and not self._is_fatal(error):
                raise cancellation from None
            if completed or self._is_fatal(error):
                raise error
            return VoiceHandoffOutcome.CANCELLED

        completed, cancellation = await self._finish_ownership(entry)
        if cancellation is not None:
            raise cancellation
        if completed:
            return VoiceHandoffOutcome.COMPLETED
        return VoiceHandoffOutcome.CANCELLED

    async def discard(self, claim: PendingVoiceClaim) -> bool:
        async with self._lock:
            entry = self._entry_for_claim(claim)
            if entry is None:
                return False
            self._remove(entry)
            task = entry.handoff_task
        cancellation = await self._cancel_and_drain(task)
        if cancellation is not None:
            raise cancellation
        return True

    async def cancel(
        self, scope: MessageScope, reply_id: str
    ) -> VoiceCancellationResult | None:
        current_claim = _current_voice_claim.get()
        self._protect_claim(current_claim)
        try:
            async with self._lock:
                entry = self._pending.get((scope, reply_id))
                if entry is None or self._is_excluded(entry, current_claim):
                    return None
                self._remove(entry)
                task = entry.handoff_task
                result = self._cancellation_result(entry, reply_id)
            cancellation = await self._cancel_and_drain(task)
            if cancellation is not None:
                raise cancellation
            return result
        finally:
            self._unprotect_claim(current_claim)

    async def cancel_all(self) -> tuple[VoiceCancellationResult, ...]:
        """Cancel every unique pending voice note and drain published handoffs."""
        return await self._cancel_matching_scope(None)

    async def cancel_scope(
        self, scope: MessageScope
    ) -> tuple[VoiceCancellationResult, ...]:
        """Cancel every unique pending voice note in one platform chat."""
        return await self._cancel_matching_scope(scope)

    async def _cancel_matching_scope(
        self,
        scope: MessageScope | None,
    ) -> tuple[VoiceCancellationResult, ...]:
        current_claim = _current_voice_claim.get()
        self._protect_claim(current_claim)
        try:
            async with self._lock:
                entries = tuple(
                    {
                        entry.claim: entry
                        for (entry_scope, _reference_id), entry in self._pending.items()
                        if (scope is None or entry_scope == scope)
                        and not self._is_excluded(entry, current_claim)
                    }.values()
                )
                for entry in entries:
                    self._remove(entry)

            tasks = tuple(
                task for entry in entries if (task := entry.handoff_task) is not None
            )
            for task in tasks:
                task.cancel()
            cancellation = await self._drain(tasks)
            if cancellation is not None:
                raise cancellation
            return tuple(self._cancellation_result(entry) for entry in entries)
        finally:
            self._unprotect_claim(current_claim)

    async def _finish_ownership(
        self,
        entry: _PendingVoice,
    ) -> tuple[bool, asyncio.CancelledError | None]:
        finish_task = asyncio.create_task(
            self._complete_if_owned(entry),
            name=f"voice-handoff-finish-{entry.claim.claim_id}",
        )
        return await _await_owned_task(finish_task)

    async def _complete_if_owned(self, entry: _PendingVoice) -> bool:
        async with self._lock:
            if self._entry_for_claim(entry.claim) is not entry:
                return False
            self._remove(entry)
            return True

    @staticmethod
    async def _run_callback(
        claim: PendingVoiceClaim,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        token = _current_voice_claim.set(claim)
        try:
            await callback()
        finally:
            _current_voice_claim.reset(token)

    @staticmethod
    async def _cancel_and_drain(
        task: asyncio.Task[None] | None,
    ) -> asyncio.CancelledError | None:
        if task is None or task is asyncio.current_task():
            return None
        task.cancel()
        return await PendingVoiceRegistry._drain((task,))

    @staticmethod
    async def _drain(
        tasks: tuple[asyncio.Task[None], ...],
    ) -> asyncio.CancelledError | None:
        if not tasks:
            return None
        drain_task = asyncio.create_task(
            PendingVoiceRegistry._consume_results(tasks),
            name="voice-handoff-drain",
        )
        _, cancellation = await _await_owned_task(drain_task)
        return cancellation

    @staticmethod
    async def _consume_results(tasks: tuple[asyncio.Task[None], ...]) -> None:
        fatal_error: BaseException | None = None
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError, Exception:
                continue
            except BaseException as error:
                fatal_error = fatal_error or error
        if fatal_error is not None:
            raise fatal_error

    @staticmethod
    def _is_fatal(error: BaseException) -> bool:
        return not isinstance(error, (asyncio.CancelledError, Exception))

    @staticmethod
    def _cancellation_result(
        entry: _PendingVoice,
        reference_id: str | None = None,
    ) -> VoiceCancellationResult:
        delete_message_ids = {entry.claim.voice_message_id}
        if reference_id is not None and reference_id != entry.claim.voice_message_id:
            delete_message_ids.clear()
        if entry.status_message_id is not None:
            delete_message_ids.add(entry.status_message_id)
        return VoiceCancellationResult(
            scope=entry.claim.scope,
            voice_message_id=entry.claim.voice_message_id,
            status_message_id=entry.status_message_id,
            delete_message_ids=frozenset(delete_message_ids),
        )

    def _entry_for_claim(self, claim: PendingVoiceClaim) -> _PendingVoice | None:
        entry = self._pending.get((claim.scope, claim.voice_message_id))
        if entry is None or entry.claim != claim:
            return None
        return entry

    def _is_excluded(
        self,
        entry: _PendingVoice,
        current_claim: PendingVoiceClaim | None,
    ) -> bool:
        return (
            entry.claim == current_claim
            or self._active_cancellations.get(entry.claim, 0) > 0
        )

    def _protect_claim(self, claim: PendingVoiceClaim | None) -> None:
        if claim is None:
            return
        self._active_cancellations[claim] = self._active_cancellations.get(claim, 0) + 1

    def _unprotect_claim(self, claim: PendingVoiceClaim | None) -> None:
        if claim is None:
            return
        remaining = self._active_cancellations[claim] - 1
        if remaining:
            self._active_cancellations[claim] = remaining
        else:
            self._active_cancellations.pop(claim)

    def _remove(self, entry: _PendingVoice) -> None:
        voice_key = (entry.claim.scope, entry.claim.voice_message_id)
        if self._pending.get(voice_key) is entry:
            self._pending.pop(voice_key)
        if entry.status_message_id is None:
            return
        status_key = (entry.claim.scope, entry.status_message_id)
        if self._pending.get(status_key) is entry:
            self._pending.pop(status_key)

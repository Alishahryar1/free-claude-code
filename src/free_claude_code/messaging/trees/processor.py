"""Task execution for claims returned by messaging tree aggregates."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from loguru import logger

from ..safe_diagnostics import format_exception_for_log
from .runtime import MessageTree
from .transitions import CancellationReason, NodeClaim, QueueEntry

NodeProcessor = Callable[[NodeClaim], Awaitable[None]]
QueueUpdateCallback = Callable[[tuple[QueueEntry, ...]], Awaitable[None]]
NodeStartedCallback = Callable[[NodeClaim], Awaitable[None]]
ClaimFailureCallback = Callable[[NodeClaim], Awaitable[None]]
ClaimFinishedCallback = Callable[[MessageTree, NodeClaim], Awaitable[None]]


@dataclass(slots=True)
class _TaskSlot:
    tree: MessageTree
    claim: NodeClaim
    task: asyncio.Task[None] | None = None
    runner_started: bool = False
    transitioned: bool = False
    recovery_task: asyncio.Task[None] | None = None
    cancellation_requested: bool = False
    cancellation_reason: CancellationReason | None = None


@dataclass(frozen=True, slots=True)
class CancelledTask:
    """Task handle plus whether the node runner owns cancellation UI."""

    task: asyncio.Task[None]
    runner_started: bool


class TreeQueueProcessor:
    """Own asyncio tasks while MessageTree owns scheduling state."""

    def __init__(
        self,
        node_processor: NodeProcessor,
        *,
        claim_failure_callback: ClaimFailureCallback,
        claim_finished_callback: ClaimFinishedCallback,
        queue_update_callback: QueueUpdateCallback | None = None,
        node_started_callback: NodeStartedCallback | None = None,
        log_messaging_error_details: bool = False,
    ) -> None:
        self._node_processor = node_processor
        self._claim_failure_callback = claim_failure_callback
        self._claim_finished_callback = claim_finished_callback
        self._queue_update_callback = queue_update_callback
        self._node_started_callback = node_started_callback
        self._log_messaging_error_details = log_messaging_error_details
        self._tasks: dict[str, _TaskSlot] = {}
        self._completion_failures: list[Exception] = []
        self._idle = asyncio.Event()
        self._idle.set()

    @staticmethod
    def _key(claim: NodeClaim) -> str:
        return claim.claim_id

    def launch(
        self,
        tree: MessageTree,
        claim: NodeClaim,
        *,
        announce_started: bool = False,
        queue: tuple[QueueEntry, ...] = (),
    ) -> None:
        """Attach a task synchronously before another coroutine can cancel it."""
        key = self._key(claim)
        if key in self._tasks:
            raise RuntimeError(f"Claim {key} already has a task")
        slot = _TaskSlot(tree=tree, claim=claim)
        self._tasks[key] = slot
        self._idle.clear()
        ownership_ready = asyncio.Event()
        claim_runner = self._run_claim(
            slot,
            ownership_ready=ownership_ready,
            announce_started=announce_started,
            queue=queue,
        )
        try:
            task = asyncio.create_task(
                claim_runner,
                name=(f"messaging-claim-{claim.identity.root_id}-{claim.claim_id[:8]}"),
            )
        except BaseException:
            claim_runner.close()
            if self._tasks.get(key) is slot:
                self._tasks.pop(key)
            if not self._tasks:
                self._idle.set()
            raise
        slot.task = task
        task.add_done_callback(lambda _task, claim_key=key: self._task_done(claim_key))
        ownership_ready.set()

    def _task_done(self, key: str) -> None:
        """Recover a claim if its task was cancelled before entering its body."""
        slot = self._tasks.get(key)
        if slot is None or slot.transitioned or slot.recovery_task is not None:
            return
        slot.recovery_task = asyncio.create_task(
            self._recover_unentered_task(slot),
            name=f"messaging-claim-recovery-{key[:8]}",
        )

    async def _recover_unentered_task(self, slot: _TaskSlot) -> None:
        task = slot.task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if not slot.transitioned:
            await self._finish_and_continue(slot)

    async def _notify_queue_updated(self, queue: tuple[QueueEntry, ...]) -> None:
        if self._queue_update_callback is None:
            return
        try:
            await self._queue_update_callback(queue)
        except Exception as exc:
            logger.warning(
                "Queue update callback failed: {}",
                format_exception_for_log(
                    exc,
                    log_full_message=self._log_messaging_error_details,
                ),
            )

    async def notify_queue_updated(self, queue: tuple[QueueEntry, ...]) -> None:
        """Publish a transition-owned queue snapshot."""
        await self._notify_queue_updated(queue)

    async def _notify_node_started(self, claim: NodeClaim) -> None:
        if self._node_started_callback is None:
            return
        try:
            await self._node_started_callback(claim)
        except Exception as exc:
            logger.warning(
                "Node started callback failed: {}",
                format_exception_for_log(
                    exc,
                    log_full_message=self._log_messaging_error_details,
                ),
            )

    async def _run_claim(
        self,
        slot: _TaskSlot,
        *,
        ownership_ready: asyncio.Event,
        announce_started: bool,
        queue: tuple[QueueEntry, ...],
    ) -> None:
        await ownership_ready.wait()
        claim = slot.claim
        try:
            if announce_started:
                await self._notify_node_started(claim)
                await self._notify_queue_updated(queue)
            if slot.cancellation_requested:
                if slot.cancellation_reason is None:
                    raise asyncio.CancelledError
                raise asyncio.CancelledError(slot.cancellation_reason)
            slot.runner_started = True
            await self._node_processor(claim)
        except asyncio.CancelledError:
            logger.info("Task for node {} was cancelled", claim.node.node_id)
            raise
        except Exception as exc:
            logger.error(
                "Error processing node {}: {}",
                claim.node.node_id,
                format_exception_for_log(
                    exc,
                    log_full_message=self._log_messaging_error_details,
                ),
            )
            await self._claim_failure_callback(claim)
        finally:
            if not slot.transitioned:
                await self._finish_and_continue(slot)

    async def _finish_and_continue(self, slot: _TaskSlot) -> None:
        current = asyncio.current_task()
        if current is not None:
            while current.cancelling():
                current.uncancel()
        try:
            while True:
                try:
                    await self._claim_finished_callback(slot.tree, slot.claim)
                    slot.transitioned = True
                    break
                except asyncio.CancelledError:
                    if current is not None:
                        while current.cancelling():
                            current.uncancel()
                    continue
        except Exception as exc:
            self._completion_failures.append(exc)
            logger.error(
                "Claim completion callback failed for node {}: {}",
                slot.claim.node.node_id,
                format_exception_for_log(
                    exc,
                    log_full_message=self._log_messaging_error_details,
                ),
            )
        finally:
            key = self._key(slot.claim)
            if self._tasks.get(key) is slot:
                self._tasks.pop(key)
            if not self._tasks:
                self._idle.set()

    def cancel(
        self,
        claim: NodeClaim,
        reason: CancellationReason | None,
    ) -> CancelledTask | None:
        """Cancel exactly the task bound to one aggregate claim."""
        slot = self._tasks.get(self._key(claim))
        if slot is None:
            return None
        slot.cancellation_requested = True
        slot.cancellation_reason = reason
        task = slot.task
        if task is None or task.done():
            return None
        if reason is None:
            task.cancel()
        else:
            task.cancel(reason)

        if slot.runner_started:
            return CancelledTask(task=task, runner_started=True)

        if slot.recovery_task is None:
            slot.recovery_task = asyncio.create_task(
                self._recover_unentered_task(slot),
                name=f"messaging-claim-recovery-{claim.claim_id[:8]}",
            )
        return CancelledTask(task=slot.recovery_task, runner_started=False)

    def task_count(self) -> int:
        """Return the number of attached claims for observability."""
        return len(self._tasks)

    async def wait_idle(self) -> None:
        """Wait for every task and hand completion failures to the caller once."""
        await self._idle.wait()
        if not self._completion_failures:
            return
        failures = self._completion_failures
        self._completion_failures = []
        if len(failures) == 1:
            raise failures[0]
        raise ExceptionGroup("Messaging claim completion failures", failures)


__all__ = ["CancelledTask", "TreeQueueProcessor"]

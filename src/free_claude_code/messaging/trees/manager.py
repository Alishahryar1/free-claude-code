"""Public facade for atomic messaging tree aggregates."""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from loguru import logger

from ..models import IncomingMessage, MessageScope
from .identity import TreeIdentity
from .node import MessageNode, MessageState
from .processor import (
    CancelledTask,
    NodeProcessor,
    NodeStartedCallback,
    QueueUpdateCallback,
    TreeQueueProcessor,
)
from .repository import TreeRepository
from .runtime import MessageTree
from .snapshot import ConversationSnapshot, TreeSnapshot
from .transitions import (
    AdmissionRejection,
    CancellationEffect,
    CancellationReason,
    CancellationResult,
    CancellationUiOwner,
    FailureResult,
    MessageSubtreeRemovalResult,
    NodeClaim,
    NodeUiTarget,
    NodeView,
    QueueDecision,
    ReplyTarget,
    TreeCancellation,
)

CANCEL_TASK_DRAIN_TIMEOUT_S = 5.0


async def _finish_transition[T](awaitable: Coroutine[Any, Any, T]) -> T:
    """Deliver a transition result even if its caller is cancelled mid-commit."""
    task = asyncio.create_task(awaitable)
    current = asyncio.current_task()
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()
            if current is not None:
                while current.cancelling():
                    current.uncancel()


async def _drain_cancelled_tasks(tasks: list[asyncio.Task[None]]) -> None:
    """Wait briefly for cancelled claims to finish aggregate cleanup."""
    if not tasks:
        return
    done, pending = await asyncio.wait(
        set(tasks),
        timeout=CANCEL_TASK_DRAIN_TIMEOUT_S,
    )
    if pending:
        logger.warning(
            "Timed out waiting for {} cancelled messaging task(s) to finish cleanup",
            len(pending),
        )
    for task in done:
        if task.cancelled():
            continue
        try:
            task.result()
        except Exception as exc:
            logger.debug(
                "Cancelled messaging task finished with {}", type(exc).__name__
            )


class TreeQueueManager:
    """Locate aggregates and coordinate tasks without exposing mutable trees."""

    def __init__(
        self,
        node_processor: NodeProcessor,
        *,
        queue_update_callback: QueueUpdateCallback | None = None,
        node_started_callback: NodeStartedCallback | None = None,
        unexpected_failure_callback: Callable[[FailureResult], None] | None = None,
        log_messaging_error_details: bool = False,
        _repository: TreeRepository | None = None,
        _restored_snapshot: ConversationSnapshot | None = None,
        _restored_stale_targets: tuple[NodeUiTarget, ...] = (),
    ) -> None:
        self._repository = _repository or TreeRepository()
        self._lock = asyncio.Lock()
        self._processor = TreeQueueProcessor(
            node_processor,
            claim_failure_callback=self._handle_processor_failure,
            claim_finished_callback=self._finish_claim,
            queue_update_callback=queue_update_callback,
            node_started_callback=node_started_callback,
            log_messaging_error_details=log_messaging_error_details,
        )
        self._restored_snapshot = _restored_snapshot
        self._restored_stale_targets = _restored_stale_targets
        self._unexpected_failure_callback = unexpected_failure_callback
        logger.info("TreeQueueManager initialized")

    @property
    def restored_snapshot(self) -> ConversationSnapshot | None:
        return self._restored_snapshot

    @property
    def restored_stale_targets(self) -> tuple[NodeUiTarget, ...]:
        return self._restored_stale_targets

    async def admit(
        self,
        incoming: IncomingMessage,
        status_message_id: str,
        *,
        parent_reference_id: str | None = None,
    ) -> QueueDecision:
        """Publish one admission before its processor can begin."""
        node_id = str(incoming.message_id)
        scope = incoming.scope
        prompt = incoming.text or ""
        async with self._lock:
            duplicate_tree = self._repository.get_tree_for_reference(scope, node_id)
            if duplicate_tree is None:
                duplicate_tree = self._repository.get_tree_for_reference(
                    scope,
                    status_message_id,
                )
            if duplicate_tree is not None:
                return await duplicate_tree.enqueue_or_claim(node_id)

            tree: MessageTree | None = None
            resolved_parent: ReplyTarget | None = None
            if parent_reference_id is not None:
                tree = self._repository.get_tree_for_reference(
                    scope, parent_reference_id
                )
                if tree is not None:
                    resolved_parent = await tree.resolve_reply(parent_reference_id)

                if tree is None or resolved_parent is None:
                    return QueueDecision(
                        claim=None,
                        position=None,
                        snapshot=None,
                        rejection=AdmissionRejection.PARENT_REMOVED,
                    )

            if tree is not None and resolved_parent is not None:
                decision = await tree.add_and_enqueue(
                    node_id,
                    scope,
                    prompt,
                    status_message_id,
                    resolved_parent.node_id,
                    resolved_parent.reference_id,
                )
                self._repository.register_node(
                    identity=tree.identity,
                    node_id=node_id,
                    status_message_id=status_message_id,
                )
                logger.info("Added node {} to tree {}", node_id, tree.identity)
            else:
                root = MessageNode(
                    node_id=node_id,
                    scope=scope,
                    prompt=prompt,
                    status_message_id=status_message_id,
                    state=MessageState.PENDING,
                )
                tree = MessageTree(root)
                decision = await tree.enqueue_or_claim(node_id)
                self._repository.add_tree(
                    tree,
                    node_id=node_id,
                    status_message_id=status_message_id,
                )
                logger.info("Created new tree {}", tree.identity)

        if decision.claim is not None:
            self._processor.launch(tree, decision.claim)
        return decision

    async def resolve_reply(
        self,
        scope: MessageScope,
        reference_id: str,
    ) -> ReplyTarget | None:
        """Resolve a scoped node or status-message reference."""
        async with self._lock:
            tree = self._repository.get_tree_for_reference(scope, reference_id)
            return await tree.resolve_reply(reference_id) if tree is not None else None

    async def resolve_node_id(
        self,
        scope: MessageScope,
        reference_id: str,
    ) -> str | None:
        target = await self.resolve_reply(scope, reference_id)
        return target.node_id if target is not None else None

    async def get_node(
        self,
        scope: MessageScope,
        reference_id: str,
    ) -> NodeView | None:
        """Return an immutable node read model."""
        async with self._lock:
            tree = self._repository.get_tree_for_reference(scope, reference_id)
            if tree is None:
                return None
            target = await tree.resolve_reply(reference_id)
            return await tree.node_view(target.node_id) if target is not None else None

    async def record_session(
        self,
        claim: NodeClaim,
        session_id: str,
    ) -> TreeSnapshot | None:
        async with self._lock:
            tree = self._repository.get_tree(claim.identity)
            return (
                await tree.record_session(claim.claim_id, session_id)
                if tree is not None
                else None
            )

    async def complete_claim(
        self,
        claim: NodeClaim,
        session_id: str | None,
    ) -> TreeSnapshot | None:
        async with self._lock:
            tree = self._repository.get_tree(claim.identity)
            return (
                await tree.complete_claim(claim.claim_id, session_id)
                if tree is not None
                else None
            )

    async def fail_claim(
        self,
        claim: NodeClaim,
        *,
        propagate: bool = True,
    ) -> FailureResult:
        return await self._fail_claim(claim, propagate=propagate)

    async def _fail_claim(
        self,
        claim: NodeClaim,
        *,
        propagate: bool,
    ) -> FailureResult:
        async with self._lock:
            tree = self._repository.get_tree(claim.identity)
            result = (
                await tree.fail_claim(
                    claim.claim_id,
                    propagate=propagate,
                )
                if tree is not None
                else FailureResult(affected=(), queue_update=None, snapshot=None)
            )
        if result.queue_update is not None:
            await self._processor.notify_queue_updated(result.queue_update)
        return result

    async def _handle_processor_failure(
        self,
        claim: NodeClaim,
    ) -> None:
        """Route an escaped runner failure through manager-owned effects."""
        result = await self._fail_claim(claim, propagate=True)
        if self._unexpected_failure_callback is None:
            return
        try:
            self._unexpected_failure_callback(result)
        except Exception as exc:
            logger.warning(
                "Unexpected messaging failure callback failed: {}",
                type(exc).__name__,
            )

    async def _finish_claim(self, tree: MessageTree, claim: NodeClaim) -> None:
        """Serialize successor task publication with aggregate detachment."""
        async with self._lock:
            tree_is_published = self._repository.get_tree(claim.identity) is tree
            completion = await tree.finish_and_claim_next(claim.claim_id)
            if tree_is_published and completion.next_claim is not None:
                self._processor.launch(
                    tree,
                    completion.next_claim,
                    announce_started=True,
                    queue=completion.queue,
                )

    @staticmethod
    def _external_effects(
        transition: TreeCancellation,
        cancelled_task: CancelledTask | None,
    ) -> tuple[CancellationEffect, ...]:
        active_node_id = (
            transition.active_claim.node.node_id
            if transition.active_claim is not None
            else None
        )
        return tuple(
            CancellationEffect(
                node=node,
                ui_owner=(
                    CancellationUiOwner.RUNNER
                    if node.node_id == active_node_id
                    and cancelled_task is not None
                    and cancelled_task.runner_started
                    else CancellationUiOwner.WORKFLOW
                ),
            )
            for node in transition.nodes
        )

    async def _current_snapshot(
        self,
        identity: TreeIdentity,
        expected_tree: MessageTree,
    ) -> TreeSnapshot | None:
        async with self._lock:
            tree = self._repository.get_tree(identity)
            if tree is not expected_tree:
                return None
            return await tree.snapshot()

    async def cancel_node(
        self,
        scope: MessageScope,
        node_id: str,
        *,
        reason: CancellationReason | None = None,
    ) -> CancellationResult:
        return await _finish_transition(self._cancel_node(scope, node_id, reason))

    async def _cancel_node(
        self,
        scope: MessageScope,
        node_id: str,
        reason: CancellationReason | None,
    ) -> CancellationResult:
        async with self._lock:
            tree = self._repository.get_tree_for_reference(scope, node_id)
            if tree is None:
                return CancellationResult()
            target = await tree.resolve_reply(node_id)
            if target is None:
                return CancellationResult()
            transition = await tree.cancel_node(target.node_id)
            cancelled_task = (
                self._processor.cancel(transition.active_claim, reason)
                if transition.active_claim is not None
                else None
            )

        if transition.queue_update is not None:
            await self._processor.notify_queue_updated(transition.queue_update)
        if cancelled_task is not None:
            await _drain_cancelled_tasks([cancelled_task.task])
        snapshot = await self._current_snapshot(tree.identity, tree)
        return CancellationResult(
            effects=self._external_effects(transition, cancelled_task),
            snapshots=(snapshot,) if snapshot is not None else (),
        )

    async def cancel_all(
        self,
        *,
        reason: CancellationReason | None = None,
    ) -> CancellationResult:
        return await _finish_transition(self._cancel_all(reason))

    async def _cancel_all(
        self,
        reason: CancellationReason | None,
    ) -> CancellationResult:
        transitions: list[
            tuple[MessageTree, TreeCancellation, CancelledTask | None]
        ] = []
        async with self._lock:
            for tree in self._repository.trees():
                transition = await tree.cancel_all()
                cancelled_task = (
                    self._processor.cancel(transition.active_claim, reason)
                    if transition.active_claim is not None
                    else None
                )
                transitions.append((tree, transition, cancelled_task))

        for _tree, transition, _task in transitions:
            if transition.queue_update is not None:
                await self._processor.notify_queue_updated(transition.queue_update)
        await _drain_cancelled_tasks(
            [
                cancelled.task
                for _tree, _transition, cancelled in transitions
                if cancelled is not None
            ]
        )

        effects: list[CancellationEffect] = []
        snapshots: list[TreeSnapshot] = []
        for tree, transition, cancelled_task in transitions:
            effects.extend(self._external_effects(transition, cancelled_task))
            snapshot = await self._current_snapshot(tree.identity, tree)
            if snapshot is not None:
                snapshots.append(snapshot)
        return CancellationResult(effects=tuple(effects), snapshots=tuple(snapshots))

    async def clear_scope(
        self,
        scope: MessageScope,
        *,
        reason: CancellationReason | None = None,
    ) -> CancellationResult:
        """Atomically detach one chat's trees, then drain their active tasks."""
        return await _finish_transition(self._clear_scope(scope, reason))

    async def _clear_scope(
        self,
        scope: MessageScope,
        reason: CancellationReason | None,
    ) -> CancellationResult:
        transitions: list[tuple[TreeCancellation, CancelledTask | None]] = []
        async with self._lock:
            for tree in self._repository.trees():
                if tree.identity.scope != scope:
                    continue
                transition = await tree.cancel_all()
                cancelled_task = (
                    self._processor.cancel(transition.active_claim, reason)
                    if transition.active_claim is not None
                    else None
                )
                transitions.append((transition, cancelled_task))
                self._repository.remove_tree(tree.identity)

        await _drain_cancelled_tasks(
            [
                cancelled.task
                for _transition, cancelled in transitions
                if cancelled is not None
            ]
        )
        effects: list[CancellationEffect] = []
        for transition, cancelled_task in transitions:
            effects.extend(self._external_effects(transition, cancelled_task))
        return CancellationResult(effects=tuple(effects))

    async def remove_message_subtree(
        self,
        scope: MessageScope,
        reference_id: str,
        *,
        reason: CancellationReason | None = None,
    ) -> MessageSubtreeRemovalResult:
        """Atomically cancel, detach, and unindex one literal reply subtree."""
        return await _finish_transition(
            self._remove_message_subtree(scope, reference_id, reason)
        )

    async def _remove_message_subtree(
        self,
        scope: MessageScope,
        reference_id: str,
        reason: CancellationReason | None,
    ) -> MessageSubtreeRemovalResult:
        async with self._lock:
            tree = self._repository.get_tree_for_reference(scope, reference_id)
            if tree is None:
                return MessageSubtreeRemovalResult(
                    cancellation=CancellationResult(),
                    removed_tree_identity=None,
                    delete_message_ids=frozenset(),
                    tree_matched=False,
                )
            target = await tree.resolve_reply(reference_id)
            if target is None:
                return MessageSubtreeRemovalResult(
                    cancellation=CancellationResult(),
                    removed_tree_identity=None,
                    delete_message_ids=frozenset(),
                    tree_matched=False,
                )
            identity = tree.identity
            transition = await tree.remove_message_subtree(target.reference_id)
            lookup_ids = set(transition.removed_message_ids)
            if transition.removed_entire_tree:
                self._repository.remove_tree(identity)
            else:
                self._repository.unregister_references(identity, lookup_ids)
            cancelled_task = (
                self._processor.cancel(transition.cancellation.active_claim, reason)
                if transition.cancellation.active_claim is not None
                else None
            )

        if transition.cancellation.queue_update is not None:
            await self._processor.notify_queue_updated(
                transition.cancellation.queue_update
            )
        if cancelled_task is not None:
            await _drain_cancelled_tasks([cancelled_task.task])
        snapshot = (
            None
            if transition.removed_entire_tree
            else await self._current_snapshot(identity, tree)
        )
        cancellation = CancellationResult(
            effects=self._external_effects(
                transition.cancellation,
                cancelled_task,
            ),
            snapshots=(snapshot,) if snapshot is not None else (),
        )
        return MessageSubtreeRemovalResult(
            cancellation=cancellation,
            removed_tree_identity=(
                identity if transition.removed_entire_tree else None
            ),
            delete_message_ids=transition.removed_message_ids,
            tree_matched=True,
        )

    def get_tree_count(self) -> int:
        return self._repository.tree_count()

    def task_count(self) -> int:
        return self._processor.task_count()

    async def wait_idle(self) -> None:
        """Wait until every processor-owned claim has finished cleanup."""
        await self._processor.wait_idle()

    async def get_message_ids_for_chat(self, platform: str, chat_id: str) -> set[str]:
        async with self._lock:
            message_ids: set[str] = set()
            for tree in self._repository.trees():
                message_ids.update(await tree.message_ids_for_chat(platform, chat_id))
            return message_ids

    async def snapshot(self) -> ConversationSnapshot:
        async with self._lock:
            trees: dict[TreeIdentity, TreeSnapshot] = {}
            for tree in self._repository.trees():
                trees[tree.identity] = await tree.snapshot()
            return ConversationSnapshot(trees=trees)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ConversationSnapshot,
        node_processor: NodeProcessor,
        *,
        queue_update_callback: QueueUpdateCallback | None = None,
        node_started_callback: NodeStartedCallback | None = None,
        unexpected_failure_callback: Callable[[FailureResult], None] | None = None,
        log_messaging_error_details: bool = False,
    ) -> TreeQueueManager:
        repository, normalized, stale_targets = TreeRepository.from_snapshot(snapshot)
        return cls(
            node_processor,
            queue_update_callback=queue_update_callback,
            node_started_callback=node_started_callback,
            unexpected_failure_callback=unexpected_failure_callback,
            log_messaging_error_details=log_messaging_error_details,
            _repository=repository,
            _restored_snapshot=normalized,
            _restored_stale_targets=stale_targets,
        )


__all__ = ["TreeQueueManager"]

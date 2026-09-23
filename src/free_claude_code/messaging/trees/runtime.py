"""Atomic runtime aggregate for one messaging conversation tree."""

import asyncio
from dataclasses import dataclass
from uuid import uuid4

from loguru import logger

from ..models import MessageScope
from .graph import MessageTreeGraph
from .identity import TreeIdentity
from .node import MessageNode, MessageReferenceKind, MessageState
from .queue import MessageNodeQueue
from .snapshot import TreeSnapshot
from .transitions import (
    AdmissionRejection,
    CompletionResult,
    FailureResult,
    MessageSubtreeRemoval,
    NodeClaim,
    NodeUiTarget,
    NodeView,
    QueueDecision,
    QueueEntry,
    ReplyTarget,
    TreeCancellation,
)


@dataclass(slots=True)
class _ActiveClaim:
    """Runtime execution identity kept separate from the node's UI state."""

    claim: NodeClaim
    cancellation_requested: bool = False


class MessageTree:
    """Own graph, queue, claim identity, and every concurrency invariant."""

    def __init__(
        self,
        root_node: MessageNode,
        *,
        graph: MessageTreeGraph | None = None,
    ) -> None:
        self._graph = graph or MessageTreeGraph(root_node)
        self._queue = MessageNodeQueue()
        self._lock = asyncio.Lock()
        self._active: _ActiveClaim | None = None
        self._restored_snapshot: TreeSnapshot | None = None
        self._restored_stale_targets: tuple[NodeUiTarget, ...] = ()
        logger.debug("Created MessageTree with root {}", self.root_id)

    @property
    def root_id(self) -> str:
        return self._graph.root_id

    @property
    def identity(self) -> TreeIdentity:
        return self._graph.identity

    @property
    def restored_snapshot(self) -> TreeSnapshot | None:
        """Normalized startup snapshot captured before the tree is published."""
        return self._restored_snapshot

    @property
    def restored_stale_targets(self) -> tuple[NodeUiTarget, ...]:
        """UI targets normalized from runnable to interrupted on restore."""
        return self._restored_stale_targets

    def _ui_target(self, node: MessageNode) -> NodeUiTarget:
        if node.status_message_id is None:
            raise ValueError("Runnable node has no status message")
        return NodeUiTarget(
            scope=node.scope,
            node_id=node.node_id,
            status_message_id=node.status_message_id,
        )

    def _queue_entries(self) -> tuple[QueueEntry, ...]:
        entries: list[QueueEntry] = []
        for node_id in self._queue.items():
            node = self._graph.get_node(node_id)
            if node is None or node.state is not MessageState.PENDING:
                continue
            entries.append(
                QueueEntry(node=self._ui_target(node), position=len(entries) + 1)
            )
        return tuple(entries)

    def _claim(self, node: MessageNode) -> NodeClaim:
        node.update_state(MessageState.IN_PROGRESS)
        claim = NodeClaim(
            identity=self.identity,
            claim_id=uuid4().hex,
            node=self._ui_target(node),
            prompt=node.prompt,
            parent_session_id=self._graph.get_parent_session_id(node.node_id),
        )
        self._active = _ActiveClaim(claim=claim)
        return claim

    def _enqueue_or_claim(self, node_id: str) -> QueueDecision:
        node = self._graph.get_node(node_id)
        if node is None or node.state is not MessageState.PENDING:
            return QueueDecision(
                claim=None,
                position=None,
                snapshot=None,
                rejection=AdmissionRejection.DUPLICATE,
            )

        if self._active is None:
            claim = self._claim(node)
            return QueueDecision(
                claim=claim,
                position=None,
                snapshot=self._graph.snapshot(),
            )

        if not self._queue.put(node_id):
            return QueueDecision(
                claim=None,
                position=None,
                snapshot=None,
                rejection=AdmissionRejection.DUPLICATE,
            )

        position = self._queue.qsize()
        logger.info("Queued node {}, position {}", node_id, position)
        return QueueDecision(
            claim=None,
            position=position,
            snapshot=self._graph.snapshot(),
        )

    async def enqueue_or_claim(self, node_id: str) -> QueueDecision:
        """Atomically reject, queue, or exclusively claim an existing node."""
        async with self._lock:
            return self._enqueue_or_claim(node_id)

    async def add_and_enqueue(
        self,
        node_id: str,
        scope: MessageScope,
        prompt: str,
        status_message_id: str,
        parent_id: str,
        parent_reference_id: str,
    ) -> QueueDecision:
        """Atomically add a reply and admit it to this tree."""
        async with self._lock:
            self._graph.add_node(
                node_id=node_id,
                scope=scope,
                prompt=prompt,
                status_message_id=status_message_id,
                parent_id=parent_id,
                parent_reference_id=parent_reference_id,
            )
            return self._enqueue_or_claim(node_id)

    async def finish_and_claim_next(self, claim_id: str) -> CompletionResult:
        """Release only the matching claim and atomically select its successor."""
        async with self._lock:
            if self._active is None or self._active.claim.claim_id != claim_id:
                return CompletionResult(
                    next_claim=None,
                    queue=self._queue_entries(),
                )

            self._active = None
            next_claim: NodeClaim | None = None
            while node_id := self._queue.pop():
                node = self._graph.get_node(node_id)
                if node is not None and node.state is MessageState.PENDING:
                    next_claim = self._claim(node)
                    break

            return CompletionResult(
                next_claim=next_claim,
                queue=self._queue_entries(),
            )

    async def cancel_node(
        self,
        node_id: str,
    ) -> TreeCancellation:
        """Atomically cancel one active, queued, or stale runnable node."""
        async with self._lock:
            node = self._graph.get_node(node_id)
            active_claim = (
                self._active.claim
                if self._active is not None
                and self._active.claim.node.node_id == node_id
                else None
            )
            if active_claim is not None:
                active = self._active
                if active is not None:
                    active.cancellation_requested = True
            if node is None:
                return TreeCancellation(
                    nodes=(),
                    active_claim=active_claim,
                    queue_update=None,
                )

            queue_changed = self._queue.remove(node_id)
            cancelled_nodes: tuple[NodeUiTarget, ...] = ()
            if node.state in (MessageState.PENDING, MessageState.IN_PROGRESS):
                node.mark_error()
                cancelled_nodes = (self._ui_target(node),)
            elif node.state is MessageState.ERROR and active_claim is not None:
                cancelled_nodes = (self._ui_target(node),)
            return TreeCancellation(
                nodes=cancelled_nodes,
                active_claim=active_claim,
                queue_update=self._queue_entries() if queue_changed else None,
            )

    async def cancel_all(
        self,
    ) -> TreeCancellation:
        """Atomically cancel every runnable node present at the transition."""
        async with self._lock:
            cancelled_nodes: list[NodeUiTarget] = []
            seen: set[str] = set()
            active_claim: NodeClaim | None = None

            if self._active is not None:
                active_claim = self._active.claim
                self._active.cancellation_requested = True
                active_node = self._graph.get_node(active_claim.node.node_id)
                if active_node is not None and active_node.state in (
                    MessageState.PENDING,
                    MessageState.IN_PROGRESS,
                ):
                    active_node.mark_error()
                    seen.add(active_node.node_id)
                    cancelled_nodes.append(self._ui_target(active_node))
                elif active_node is not None:
                    seen.add(active_node.node_id)
                    if active_node.state is MessageState.ERROR:
                        cancelled_nodes.append(self._ui_target(active_node))

            queued_ids = self._queue.drain()
            for node_id in queued_ids:
                node = self._graph.get_node(node_id)
                if node is None or node.state not in (
                    MessageState.PENDING,
                    MessageState.IN_PROGRESS,
                ):
                    continue
                node.mark_error()
                seen.add(node_id)
                cancelled_nodes.append(self._ui_target(node))

            for node in self._graph.all_nodes():
                if node.node_id in seen or node.state not in (
                    MessageState.PENDING,
                    MessageState.IN_PROGRESS,
                ):
                    continue
                node.mark_error()
                cancelled_nodes.append(self._ui_target(node))

            return TreeCancellation(
                nodes=tuple(cancelled_nodes),
                active_claim=active_claim,
                queue_update=() if queued_ids else None,
            )

    async def remove_message_subtree(
        self,
        reference_id: str,
    ) -> MessageSubtreeRemoval:
        """Atomically cancel and detach one literal platform reply subtree."""
        async with self._lock:
            resolved = self._graph.resolve_reference(reference_id)
            reference_ids = tuple(self._graph.get_reference_descendants(reference_id))
            if resolved is None or not reference_ids:
                empty = TreeCancellation(
                    nodes=(),
                    active_claim=None,
                    queue_update=None,
                )
                return MessageSubtreeRemoval(
                    cancellation=empty,
                    removed_message_ids=frozenset(),
                    removed_entire_tree=False,
                )

            owner, reference_kind = resolved
            removed_node_ids = {
                candidate
                for candidate in reference_ids
                if self._graph.get_node(candidate) is not None
            }
            affected_node_ids = set(removed_node_ids)
            if reference_kind is MessageReferenceKind.STATUS:
                affected_node_ids.add(owner.node_id)
            active_claim = (
                self._active.claim
                if self._active is not None
                and self._active.claim.node.node_id in affected_node_ids
                else None
            )
            if active_claim is not None:
                active = self._active
                if active is not None:
                    active.cancellation_requested = True
            cancelled_nodes: list[NodeUiTarget] = []
            queue_changed = False

            for node_id in affected_node_ids:
                node = self._graph.get_node(node_id)
                if node is None:
                    continue
                queue_changed = self._queue.remove(node_id) or queue_changed
                if node.state in (MessageState.PENDING, MessageState.IN_PROGRESS):
                    target = self._ui_target(node)
                    node.mark_error()
                    cancelled_nodes.append(target)
                elif (
                    node.state is MessageState.ERROR
                    and active_claim is not None
                    and active_claim.node.node_id == node_id
                ):
                    cancelled_nodes.append(self._ui_target(node))

            if reference_kind is MessageReferenceKind.STATUS:
                self._graph.clear_status(owner.node_id)
            removed_entire_tree = self.root_id in removed_node_ids
            self._graph.remove_nodes(removed_node_ids)
            cancellation = TreeCancellation(
                nodes=tuple(cancelled_nodes),
                active_claim=active_claim,
                queue_update=self._queue_entries() if queue_changed else None,
            )
            return MessageSubtreeRemoval(
                cancellation=cancellation,
                removed_message_ids=frozenset(reference_ids),
                removed_entire_tree=removed_entire_tree,
            )

    async def record_session(
        self, claim_id: str, session_id: str
    ) -> TreeSnapshot | None:
        """Record a real CLI session only for the currently active claim."""
        async with self._lock:
            if (
                self._active is None
                or self._active.claim.claim_id != claim_id
                or self._active.cancellation_requested
            ):
                return None
            node = self._graph.get_node(self._active.claim.node.node_id)
            if node is None or node.state is not MessageState.IN_PROGRESS:
                return None
            node.update_state(MessageState.IN_PROGRESS, session_id=session_id)
            return self._graph.snapshot()

    async def complete_claim(
        self, claim_id: str, session_id: str | None
    ) -> TreeSnapshot | None:
        """Mark the currently active claim complete."""
        async with self._lock:
            if (
                self._active is None
                or self._active.claim.claim_id != claim_id
                or self._active.cancellation_requested
            ):
                return None
            node = self._graph.get_node(self._active.claim.node.node_id)
            if node is None or node.state not in (
                MessageState.IN_PROGRESS,
                MessageState.ERROR,
            ):
                return None
            node.update_state(MessageState.COMPLETED, session_id=session_id)
            return self._graph.snapshot()

    async def fail_claim(
        self,
        claim_id: str,
        *,
        propagate: bool,
    ) -> FailureResult:
        """Atomically fail the active claim and its pending descendants."""
        async with self._lock:
            if (
                self._active is None
                or self._active.claim.claim_id != claim_id
                or self._active.cancellation_requested
            ):
                return FailureResult(affected=(), queue_update=None, snapshot=None)
            node = self._graph.get_node(self._active.claim.node.node_id)
            if node is None:
                return FailureResult(affected=(), queue_update=None, snapshot=None)

            affected: list[NodeUiTarget] = []
            queue_changed = False
            if node.state is not MessageState.COMPLETED:
                if node.state is not MessageState.ERROR:
                    node.mark_error()
                    affected.append(self._ui_target(node))

                if propagate:
                    for descendant_id in self._graph.get_descendants(node.node_id)[1:]:
                        child = self._graph.get_node(descendant_id)
                        if child is None or child.state is not MessageState.PENDING:
                            continue
                        child.mark_error()
                        queue_changed = (
                            self._queue.remove(child.node_id) or queue_changed
                        )
                        affected.append(self._ui_target(child))

            return FailureResult(
                affected=tuple(affected),
                queue_update=self._queue_entries() if queue_changed else None,
                snapshot=self._graph.snapshot(),
            )

    async def resolve_reply(self, reference_id: str) -> ReplyTarget | None:
        """Resolve a node/status reference without exposing the mutable graph."""
        async with self._lock:
            resolved = self._graph.resolve_reference(reference_id)
            if resolved is None:
                return None
            node, reference_kind = resolved
            return ReplyTarget(
                node_id=node.node_id,
                reference_id=reference_id,
                reference_kind=reference_kind,
                queue_position=(self._queue.qsize() + 1)
                if self._active is not None
                else None,
            )

    async def node_view(self, node_id: str) -> NodeView | None:
        """Return a copied node read model."""
        async with self._lock:
            node = self._graph.get_node(node_id)
            if node is None:
                return None
            return NodeView(
                identity=self.identity,
                node_id=node.node_id,
                state=node.state,
                parent_id=node.parent_id,
                session_id=node.session_id,
            )

    async def snapshot(self) -> TreeSnapshot:
        """Capture a detached persistence snapshot under the aggregate lock."""
        async with self._lock:
            return self._graph.snapshot()

    async def message_ids_for_chat(self, platform: str, chat_id: str) -> set[str]:
        """Copy every prompt and FCC status belonging to one platform chat."""
        async with self._lock:
            if self.identity.scope.platform != str(platform) or (
                self.identity.scope.chat_id != str(chat_id)
            ):
                return set()
            return self._graph.all_reference_ids()

    @classmethod
    def from_snapshot(cls, snapshot: TreeSnapshot) -> MessageTree:
        """Restore and reconcile interrupted nodes before publishing the tree."""
        graph = MessageTreeGraph.from_snapshot(snapshot)
        tree = cls(graph.get_root(), graph=graph)
        stale_targets: list[NodeUiTarget] = []
        for node in graph.all_nodes():
            if node.state in (MessageState.PENDING, MessageState.IN_PROGRESS):
                stale_targets.append(tree._ui_target(node))
                node.mark_error()
        tree._restored_stale_targets = tuple(stale_targets)
        tree._restored_snapshot = graph.snapshot()
        return tree


__all__ = ["MessageTree"]

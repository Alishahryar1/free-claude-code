"""Tree-Based Message Queue Manager - Refactored.

Coordinates data access, async processing, and error handling.
Uses TreeRepository for data, TreeQueueProcessor for async logic.
"""

import asyncio
from datetime import datetime, timezone
from typing import Callable, Awaitable, List, Optional

from .models import IncomingMessage
from .tree_data import MessageState, MessageNode, MessageTree
from .tree_repository import TreeRepository
from .tree_processor import TreeQueueProcessor
from loguru import logger

# Backward compatibility: re-export moved classes
__all__ = [
    "TreeQueueManager",
    "MessageState",
    "MessageNode",
    "MessageTree",
]


class TreeQueueManager:
    """
    Manages multiple message trees. Facade that coordinates components.

    Each new conversation creates a new tree.
    Replies to existing messages add nodes to existing trees.

    Components:
        - TreeRepository: Data access layer
        - TreeQueueProcessor: Async queue processing
    """

    def __init__(
        self,
        queue_update_callback: Optional[
            Callable[[MessageTree], Awaitable[None]]
        ] = None,
        node_started_callback: Optional[
            Callable[[MessageTree, str], Awaitable[None]]
        ] = None,
    ):
        self._repository = TreeRepository()
        self._processor = TreeQueueProcessor(
            queue_update_callback=queue_update_callback,
            node_started_callback=node_started_callback,
        )
        self._lock = asyncio.Lock()

        logger.info("TreeQueueManager initialized")

    async def create_tree(
        self,
        node_id: str,
        incoming: IncomingMessage,
        status_message_id: str,
    ) -> MessageTree:
        """
        Create a new tree with a root node.

        Args:
            node_id: ID for the root node
            incoming: The incoming message
            status_message_id: Bot's status message ID

        Returns:
            The created MessageTree
        """
        async with self._lock:
            root_node = MessageNode(
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_message_id,
                state=MessageState.PENDING,
            )

            tree = MessageTree(root_node)
            self._repository.add_tree(node_id, tree)

            logger.info(f"Created new tree with root {node_id}")
            return tree

    async def add_to_tree(
        self,
        parent_node_id: str,
        node_id: str,
        incoming: IncomingMessage,
        status_message_id: str,
    ) -> tuple[MessageTree, MessageNode]:
        """
        Add a reply as a child node to an existing tree.

        Args:
            parent_node_id: ID of the parent message
            node_id: ID for the new node
            incoming: The incoming reply message
            status_message_id: Bot's status message ID

        Returns:
            Tuple of (tree, new_node)
        """
        async with self._lock:
            if not self._repository.has_node(parent_node_id):
                raise ValueError(f"Parent node {parent_node_id} not found in any tree")

            tree = self._repository.get_tree_for_node(parent_node_id)
            if not tree:
                raise ValueError(f"Parent node {parent_node_id} not found in any tree")

        # Add node (tree has its own lock) - outside manager lock to avoid deadlock
        node = await tree.add_node(
            node_id=node_id,
            incoming=incoming,
            status_message_id=status_message_id,
            parent_id=parent_node_id,
        )

        async with self._lock:
            self._repository.register_node(node_id, tree.root_id)

        logger.info(f"Added node {node_id} to tree {tree.root_id}")
        return tree, node

    def get_tree(self, root_id: str) -> Optional[MessageTree]:
        """Get a tree by its root ID."""
        return self._repository.get_tree(root_id)

    def get_tree_for_node(self, node_id: str) -> Optional[MessageTree]:
        """Get the tree containing a given node."""
        return self._repository.get_tree_for_node(node_id)

    def get_node(self, node_id: str) -> Optional[MessageNode]:
        """Get a node from any tree."""
        return self._repository.get_node(node_id)

    def resolve_parent_node_id(self, msg_id: str) -> Optional[str]:
        """Resolve a message ID to the actual parent node ID."""
        return self._repository.resolve_parent_node_id(msg_id)

    def is_tree_busy(self, root_id: str) -> bool:
        """Check if a tree is currently processing."""
        return self._repository.is_tree_busy(root_id)

    def is_node_tree_busy(self, node_id: str) -> bool:
        """Check if the tree containing a node is busy."""
        return self._repository.is_node_tree_busy(node_id)

    async def enqueue(
        self,
        node_id: str,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> bool:
        """
        Enqueue a node for processing.

        If the tree is not busy, processing starts immediately.
        If busy, the message is queued.

        Args:
            node_id: Node to process
            processor: Async function to process the node

        Returns:
            True if queued, False if processing immediately
        """
        tree = self._repository.get_tree_for_node(node_id)
        if not tree:
            logger.error(f"No tree found for node {node_id}")
            return False

        return await self._processor.enqueue_and_start(tree, node_id, processor)

    def get_queue_size(self, node_id: str) -> int:
        """Get queue size for the tree containing a node."""
        return self._repository.get_queue_size(node_id)

    def get_pending_children(self, node_id: str) -> List[MessageNode]:
        """Get all pending child nodes (recursively) of a given node."""
        return self._repository.get_pending_children(node_id)

    async def mark_node_error(
        self,
        node_id: str,
        error_message: str,
        propagate_to_children: bool = True,
    ) -> List[MessageNode]:
        """
        Mark a node as ERROR and optionally propagate to pending children.

        Args:
            node_id: The node to mark as error
            error_message: Error description
            propagate_to_children: If True, also mark pending children as error

        Returns:
            List of all nodes marked as error (including children)
        """
        tree = self._repository.get_tree_for_node(node_id)
        if not tree:
            return []

        affected = []
        node = tree.get_node(node_id)
        if node:
            await tree.update_state(
                node_id, MessageState.ERROR, error_message=error_message
            )
            affected.append(node)

        if propagate_to_children:
            pending_children = self._repository.get_pending_children(node_id)
            for child in pending_children:
                await tree.update_state(
                    child.node_id,
                    MessageState.ERROR,
                    error_message=f"Parent failed: {error_message}",
                )
                affected.append(child)

        return affected

    def cancel_tree(self, root_id: str) -> List[MessageNode]:
        """
        Cancel all queued and in-progress messages in a tree.

        Updates node states to ERROR and returns list of affected nodes
        that were actually active or in the current processing queue.
        """
        tree = self._repository.get_tree(root_id)
        if not tree:
            return []

        cancelled_nodes = []

        # 1. Cancel running task
        if tree.cancel_current_task():
            current_id = tree.current_node_id
            if current_id:
                node = tree.get_node(current_id)
                if node and node.state not in (
                    MessageState.COMPLETED,
                    MessageState.ERROR,
                ):
                    node.state = MessageState.ERROR
                    node.error_message = "Cancelled by user"
                    cancelled_nodes.append(node)

        # 2. Drain queue and mark nodes as cancelled
        queue_nodes = tree.drain_queue_and_mark_cancelled()
        cancelled_nodes.extend(queue_nodes)
        cancelled_ids = {n.node_id for n in cancelled_nodes}

        # 3. Cleanup: Mark ANY other PENDING or IN_PROGRESS nodes as ERROR
        cleanup_count = 0
        for node in tree.all_nodes():
            if (
                node.state in (MessageState.PENDING, MessageState.IN_PROGRESS)
                and node.node_id not in cancelled_ids
            ):
                node.state = MessageState.ERROR
                node.error_message = "Stale task cleaned up"
                cleanup_count += 1

        tree.reset_processing_state()

        if cancelled_nodes:
            logger.info(
                f"Cancelled {len(cancelled_nodes)} active nodes in tree {root_id}"
            )
        if cleanup_count:
            logger.info(f"Cleaned up {cleanup_count} stale nodes in tree {root_id}")

        return cancelled_nodes

    async def cancel_node(self, node_id: str) -> List[MessageNode]:
        """
        Cancel a single node (queued or in-progress) without affecting other nodes.

        - If the node is currently running, cancels the current asyncio task.
        - If the node is queued, removes it from the queue.
        - Marks the node as ERROR with "Cancelled by user".

        Returns:
            List containing the cancelled node if it was cancellable, else empty list.
        """
        tree = self._repository.get_tree_for_node(node_id)
        if not tree:
            return []

        async with tree.with_lock():
            node = tree.get_node(node_id)
            if not node:
                return []

            if node.state in (MessageState.COMPLETED, MessageState.ERROR):
                return []

            if tree.is_current_node(node_id):
                self._processor.cancel_current(tree)

            try:
                tree.remove_from_queue(node_id)
            except Exception:
                logger.debug(
                    "Failed to remove node from queue; will rely on state=ERROR"
                )

            node.state = MessageState.ERROR
            node.error_message = "Cancelled by user"
            node.completed_at = datetime.now(timezone.utc)

            return [node]

    async def cancel_all(self) -> List[MessageNode]:
        """Cancel all messages in all trees (async wrapper)."""
        async with self._lock:
            return self.cancel_all_sync()

    def cancel_all_sync(self) -> List[MessageNode]:
        """
        Cancel all messages in all trees (synchronous/locked version).
        NOTE: Must be called with self._lock held.
        """
        all_cancelled = []
        for root_id in list(self._repository.tree_ids()):
            all_cancelled.extend(self.cancel_tree(root_id))
        return all_cancelled

    def cleanup_stale_nodes(self) -> int:
        """
        Mark any PENDING or IN_PROGRESS nodes in all trees as ERROR.
        Used on startup to reconcile restored state.
        """
        count = 0
        for tree in self._repository.all_trees():
            for node in tree.all_nodes():
                if node.state in (MessageState.PENDING, MessageState.IN_PROGRESS):
                    node.state = MessageState.ERROR
                    node.error_message = "Lost during server restart"
                    count += 1
        if count:
            logger.info(f"Cleaned up {count} stale nodes during startup")
        return count

    def get_tree_count(self) -> int:
        """Get the number of active message trees."""
        return self._repository.tree_count()

    def set_queue_update_callback(
        self,
        queue_update_callback: Optional[Callable[[MessageTree], Awaitable[None]]],
    ) -> None:
        """Set callback for queue position updates."""
        self._processor.set_queue_update_callback(queue_update_callback)

    def set_node_started_callback(
        self,
        node_started_callback: Optional[Callable[[MessageTree, str], Awaitable[None]]],
    ) -> None:
        """Set callback for when a queued node starts processing."""
        self._processor.set_node_started_callback(node_started_callback)

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree (for external mapping)."""
        self._repository.register_node(node_id, root_id)

    def to_dict(self) -> dict:
        """Serialize all trees."""
        return self._repository.to_dict()

    @classmethod
    def from_dict(
        cls,
        data: dict,
        queue_update_callback: Optional[
            Callable[[MessageTree], Awaitable[None]]
        ] = None,
        node_started_callback: Optional[
            Callable[[MessageTree, str], Awaitable[None]]
        ] = None,
    ) -> "TreeQueueManager":
        """Deserialize from dictionary."""
        manager = cls(
            queue_update_callback=queue_update_callback,
            node_started_callback=node_started_callback,
        )
        manager._repository = TreeRepository.from_dict(data)
        return manager

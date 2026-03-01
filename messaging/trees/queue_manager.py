"""Tree-Based Message Queue Manager - Refactored.

Coordinates data access, async processing, and error handling.
Uses TreeRepository for data, TreeQueueProcessor for async logic.
"""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from ..models import IncomingMessage
from .data import MessageNode, MessageState, MessageTree
from .processor import TreeQueueProcessor
from .repository import TreeRepository

# Backward compatibility: re-export moved classes
__all__ = [
    "MessageNode",
    "MessageState",
    "MessageTree",
    "TreeQueueManager",
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
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None = None,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]]
        | None = None,
        _repository: TreeRepository | None = None,
    ):
        self._repository = _repository or TreeRepository()
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

    def get_tree(self, root_id: str) -> MessageTree | None:
        """Get a tree by its root ID."""
        return self._repository.get_tree(root_id)

    def get_tree_for_node(self, node_id: str) -> MessageTree | None:
        """Get the tree containing a given node."""
        return self._repository.get_tree_for_node(node_id)

    def get_node(self, node_id: str) -> MessageNode | None:
        """Get a node from any tree."""
        return self._repository.get_node(node_id)

    def resolve_parent_node_id(self, msg_id: str) -> str | None:
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

    def get_pending_children(self, node_id: str) -> list[MessageNode]:
        """Get all pending child nodes (recursively) of a given node."""
        return self._repository.get_pending_children(node_id)

    async def mark_node_error(
        self,
        node_id: str,
        error_message: str,
        propagate_to_children: bool = True,
    ) -> list[MessageNode]:
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

    async def cancel_tree(self, root_id: str) -> list[MessageNode]:
        """
        Cancel all queued and in-progress messages in a tree.

        Updates node states to ERROR and returns list of affected nodes
        that were actually active or in the current processing queue.
        """
        tree = self._repository.get_tree(root_id)
        if not tree:
            return []

        cancelled_nodes = []

        cleanup_count = 0
        async with tree.with_lock():
            # 1. Cancel running task
            if tree.cancel_current_task():
                current_id = tree.current_node_id
                if current_id:
                    node = tree.get_node(current_id)
                    if node and node.state not in (
                        MessageState.COMPLETED,
                        MessageState.ERROR,
                    ):
                        tree.set_node_error_sync(node, "Cancelled by user")
                        cancelled_nodes.append(node)

            # 2. Drain queue and mark nodes as cancelled
            queue_nodes = tree.drain_queue_and_mark_cancelled()
            cancelled_nodes.extend(queue_nodes)
            cancelled_ids = {n.node_id for n in cancelled_nodes}

            # 3. Cleanup: Mark ANY other PENDING or IN_PROGRESS nodes as ERROR
            for node in tree.all_nodes():
                if (
                    node.state in (MessageState.PENDING, MessageState.IN_PROGRESS)
                    and node.node_id not in cancelled_ids
                ):
                    tree.set_node_error_sync(node, "Stale task cleaned up")
                    cleanup_count += 1

            tree.reset_processing_state()

        if cancelled_nodes:
            logger.info(
                f"Cancelled {len(cancelled_nodes)} active nodes in tree {root_id}"
            )
        if cleanup_count:
            logger.info(f"Cleaned up {cleanup_count} stale nodes in tree {root_id}")

        return cancelled_nodes

    async def cancel_node(self, node_id: str) -> list[MessageNode]:
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

            tree.set_node_error_sync(node, "Cancelled by user")

            return [node]

    async def cancel_all(self) -> list[MessageNode]:
        """Cancel all messages in all trees."""
        async with self._lock:
            root_ids = list(self._repository.tree_ids())
            all_cancelled: list[MessageNode] = []
            for root_id in root_ids:
                all_cancelled.extend(await self.cancel_tree(root_id))
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
                    tree.set_node_error_sync(node, "Lost during server restart")
                    count += 1
        if count:
            logger.info(f"Cleaned up {count} stale nodes during startup")
        return count

    def get_tree_count(self) -> int:
        """Get the number of active message trees."""
        return self._repository.tree_count()

    def set_queue_update_callback(
        self,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None,
    ) -> None:
        """Set callback for queue position updates."""
        self._processor.set_queue_update_callback(queue_update_callback)

    def set_node_started_callback(
        self,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]] | None,
    ) -> None:
        """Set callback for when a queued node starts processing."""
        self._processor.set_node_started_callback(node_started_callback)

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree (for external mapping)."""
        self._repository.register_node(node_id, root_id)

    async def cancel_branch(self, branch_root_id: str) -> list[MessageNode]:
        """
        Cancel all PENDING/IN_PROGRESS nodes in the subtree (branch_root + descendants).

        Does not call cli_manager.stop_all(). Returns list of cancelled nodes.
        """
        tree = self._repository.get_tree_for_node(branch_root_id)
        if not tree:
            return []

        branch_ids = set(tree.get_descendants(branch_root_id))
        cancelled: list[MessageNode] = []

        async with tree.with_lock():
            for nid in branch_ids:
                node = tree.get_node(nid)
                if not node or node.state in (
                    MessageState.COMPLETED,
                    MessageState.ERROR,
                ):
                    continue

                if tree.is_current_node(nid):
                    self._processor.cancel_current(tree)
                    tree.set_node_error_sync(node, "Cancelled by user")
                    cancelled.append(node)
                else:
                    tree.remove_from_queue(nid)
                    tree.set_node_error_sync(node, "Cancelled by user")
                    cancelled.append(node)

        if cancelled:
            logger.info(f"Cancelled {len(cancelled)} nodes in branch {branch_root_id}")
        return cancelled

    async def remove_branch(
        self, branch_root_id: str
    ) -> tuple[list[MessageNode], str, bool]:
        """
        Remove a branch (subtree) from the tree.

        If branch_root is the tree root, removes the entire tree.

        Returns:
            (removed_nodes, root_id, removed_entire_tree)
        """
        tree = self._repository.get_tree_for_node(branch_root_id)
        if not tree:
            return ([], "", False)

        root_id = tree.root_id

        if branch_root_id == root_id:
            cancelled = await self.cancel_tree(root_id)
            removed_tree = self._repository.remove_tree(root_id)
            if removed_tree:
                return (removed_tree.all_nodes(), root_id, True)
            return (cancelled, root_id, True)

        async with tree.with_lock():
            removed = tree.remove_branch(branch_root_id)

        self._repository.unregister_nodes([n.node_id for n in removed])
        return (removed, root_id, False)

    def get_message_ids_for_chat(self, platform: str, chat_id: str) -> set[str]:
        """Get all message IDs for a given platform/chat."""
        return self._repository.get_message_ids_for_chat(platform, chat_id)

    def to_dict(self) -> dict:
        """Serialize all trees."""
        return self._repository.to_dict()

    @classmethod
    def from_dict(
        cls,
        data: dict,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None = None,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]]
        | None = None,
    ) -> TreeQueueManager:
        """Deserialize from dictionary."""
        return cls(
            queue_update_callback=queue_update_callback,
            node_started_callback=node_started_callback,
            _repository=TreeRepository.from_dict(data),
        )

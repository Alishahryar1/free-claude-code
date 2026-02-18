"""Tree data structures for message queue.

Contains MessageState, MessageNode, and MessageTree classes.
"""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from loguru import logger

from ..models import IncomingMessage


class _SnapshotQueue(asyncio.Queue[str]):
    """
    Queue with snapshot/remove helpers.

    Extends asyncio.Queue to expose read-only snapshot and in-place remove
    without inflating _unfinished_tasks (drain/put would require task_done).
    """

    _queue: deque  # Set by asyncio.Queue.__init__

    def get_snapshot(self) -> list[str]:
        """Return current queue contents in FIFO order (read-only copy)."""
        return list(self._queue)

    def remove_if_present(self, item: str) -> bool:
        """Remove item from queue if present. Returns True if removed."""
        if item not in self._queue:
            return False
        object.__setattr__(self, "_queue", deque(x for x in self._queue if x != item))
        return True


class MessageState(Enum):
    """State of a message node in the tree."""

    PENDING = "pending"  # Queued, waiting to be processed
    IN_PROGRESS = "in_progress"  # Currently being processed by Claude
    COMPLETED = "completed"  # Processing finished successfully
    ERROR = "error"  # Processing failed


@dataclass
class MessageNode:
    """
    A node in the message tree.

    Each node represents a single message and tracks:
    - Its relationship to parent/children
    - Its processing state
    - Claude session information
    """

    node_id: str  # Unique ID (typically message_id)
    incoming: IncomingMessage  # The original message
    status_message_id: str  # Bot's status message ID
    state: MessageState = MessageState.PENDING
    parent_id: str | None = None  # Parent node ID (None for root)
    session_id: str | None = None  # Claude session ID (forked from parent)
    children_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    error_message: str | None = None
    context: Any = None  # Additional context if needed

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "node_id": self.node_id,
            "incoming": {
                "text": self.incoming.text,
                "chat_id": self.incoming.chat_id,
                "user_id": self.incoming.user_id,
                "message_id": self.incoming.message_id,
                "platform": self.incoming.platform,
                "reply_to_message_id": self.incoming.reply_to_message_id,
                "username": self.incoming.username,
            },
            "status_message_id": self.status_message_id,
            "state": self.state.value,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "children_ids": self.children_ids,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MessageNode:
        """Create from dictionary (JSON deserialization)."""
        incoming_data = data["incoming"]
        incoming = IncomingMessage(
            text=incoming_data["text"],
            chat_id=incoming_data["chat_id"],
            user_id=incoming_data["user_id"],
            message_id=incoming_data["message_id"],
            platform=incoming_data["platform"],
            reply_to_message_id=incoming_data.get("reply_to_message_id"),
            username=incoming_data.get("username"),
        )
        return cls(
            node_id=data["node_id"],
            incoming=incoming,
            status_message_id=data["status_message_id"],
            state=MessageState(data["state"]),
            parent_id=data.get("parent_id"),
            session_id=data.get("session_id"),
            children_ids=data.get("children_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            error_message=data.get("error_message"),
        )


class MessageTree:
    """
    A tree of message nodes with queue functionality.

    Provides:
    - O(1) node lookup via hashmap
    - Per-tree message queue
    - Thread-safe operations via asyncio.Lock
    """

    def __init__(self, root_node: MessageNode):
        """
        Initialize tree with a root node.

        Args:
            root_node: The root message node
        """
        self.root_id = root_node.node_id
        self._nodes: dict[str, MessageNode] = {root_node.node_id: root_node}
        self._status_to_node: dict[str, str] = {
            root_node.status_message_id: root_node.node_id
        }
        self._queue: _SnapshotQueue = _SnapshotQueue()
        self._lock = asyncio.Lock()
        self._is_processing = False
        self._current_node_id: str | None = None
        self._current_task: asyncio.Task | None = None

        logger.debug(f"Created MessageTree with root {self.root_id}")

    def set_current_task(self, task: asyncio.Task | None) -> None:
        """Set the current processing task. Caller must hold lock."""
        self._current_task = task

    @property
    def is_processing(self) -> bool:
        """Check if tree is currently processing a message."""
        return self._is_processing

    async def add_node(
        self,
        node_id: str,
        incoming: IncomingMessage,
        status_message_id: str,
        parent_id: str,
    ) -> MessageNode:
        """
        Add a child node to the tree.

        Args:
            node_id: Unique ID for the new node
            incoming: The incoming message
            status_message_id: Bot's status message ID
            parent_id: Parent node ID

        Returns:
            The created MessageNode
        """
        async with self._lock:
            if parent_id not in self._nodes:
                raise ValueError(f"Parent node {parent_id} not found in tree")

            node = MessageNode(
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_message_id,
                parent_id=parent_id,
                state=MessageState.PENDING,
            )

            self._nodes[node_id] = node
            self._status_to_node[status_message_id] = node_id
            self._nodes[parent_id].children_ids.append(node_id)

            logger.debug(f"Added node {node_id} as child of {parent_id}")
            return node

    def get_node(self, node_id: str) -> MessageNode | None:
        """Get a node by ID (O(1) lookup)."""
        return self._nodes.get(node_id)

    def get_root(self) -> MessageNode:
        """Get the root node."""
        return self._nodes[self.root_id]

    def get_children(self, node_id: str) -> list[MessageNode]:
        """Get all child nodes of a given node."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def get_parent(self, node_id: str) -> MessageNode | None:
        """Get the parent node."""
        node = self._nodes.get(node_id)
        if not node or not node.parent_id:
            return None
        return self._nodes.get(node.parent_id)

    def get_parent_session_id(self, node_id: str) -> str | None:
        """
        Get the parent's session ID for forking.

        Returns None for root nodes.
        """
        parent = self.get_parent(node_id)
        return parent.session_id if parent else None

    async def update_state(
        self,
        node_id: str,
        state: MessageState,
        session_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update a node's state."""
        async with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                logger.warning(f"Node {node_id} not found for state update")
                return

            node.state = state
            if session_id:
                node.session_id = session_id
            if error_message:
                node.error_message = error_message
            if state in (MessageState.COMPLETED, MessageState.ERROR):
                node.completed_at = datetime.now(UTC)

            logger.debug(f"Node {node_id} state -> {state.value}")

    async def enqueue(self, node_id: str) -> int:
        """
        Add a node to the processing queue.

        Returns:
            Queue position (1-indexed)
        """
        async with self._lock:
            await self._queue.put(node_id)
            position = self._queue.qsize()
            logger.debug(f"Enqueued node {node_id}, position {position}")
            return position

    async def dequeue(self) -> str | None:
        """
        Get the next node ID from the queue.

        Returns None if queue is empty.
        """
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_queue_snapshot(self) -> list[str]:
        """
        Get a snapshot of the current queue order.

        Returns:
            List of node IDs in FIFO order.
        """
        async with self._lock:
            return self._queue.get_snapshot()

    def get_queue_size(self) -> int:
        """Get number of messages waiting in queue."""
        return self._queue.qsize()

    def remove_from_queue(self, node_id: str) -> bool:
        """
        Remove node_id from the internal queue if present.

        Caller must hold the tree lock (e.g. via with_lock).
        Returns True if node was removed, False if not in queue.
        """
        return self._queue.remove_if_present(node_id)

    @asynccontextmanager
    async def with_lock(self):
        """Async context manager for tree lock. Use when multiple operations need atomicity."""
        async with self._lock:
            yield

    def set_processing_state(self, node_id: str | None, is_processing: bool) -> None:
        """Set processing state. Caller must hold lock for consistency with queue operations."""
        self._is_processing = is_processing
        self._current_node_id = node_id if is_processing else None

    def clear_current_node(self) -> None:
        """Clear the currently processing node ID. Caller must hold lock."""
        self._current_node_id = None

    def is_current_node(self, node_id: str) -> bool:
        """Check if node_id is the currently processing node."""
        return self._current_node_id == node_id

    def put_queue_unlocked(self, node_id: str) -> None:
        """Add node to queue. Caller must hold lock (e.g. via with_lock)."""
        self._queue.put_nowait(node_id)

    def cancel_current_task(self) -> bool:
        """Cancel the currently running task. Returns True if a task was cancelled."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            return True
        return False

    def drain_queue_and_mark_cancelled(
        self, error_message: str = "Cancelled by user"
    ) -> list[MessageNode]:
        """
        Drain the queue, mark each node as ERROR, and return affected nodes.
        Does not acquire lock; caller must ensure no concurrent queue access.
        """
        nodes: list[MessageNode] = []
        while True:
            try:
                node_id = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            node = self._nodes.get(node_id)
            if node:
                node.state = MessageState.ERROR
                node.error_message = error_message
                nodes.append(node)
        return nodes

    def reset_processing_state(self) -> None:
        """Reset processing flags after cancel/cleanup."""
        self._is_processing = False
        self._current_node_id = None

    @property
    def current_node_id(self) -> str | None:
        """Get the ID of the node currently being processed."""
        return self._current_node_id

    def to_dict(self) -> dict:
        """Serialize tree to dictionary."""
        return {
            "root_id": self.root_id,
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> MessageTree:
        """Deserialize tree from dictionary."""
        root_id = data["root_id"]
        nodes_data = data["nodes"]

        # Create root node first
        root_node = MessageNode.from_dict(nodes_data[root_id])
        tree = cls(root_node)

        # Add remaining nodes and build status->node index
        for node_id, node_data in nodes_data.items():
            if node_id != root_id:
                node = MessageNode.from_dict(node_data)
                tree._nodes[node_id] = node
                tree._status_to_node[node.status_message_id] = node_id

        return tree

    def all_nodes(self) -> list[MessageNode]:
        """Get all nodes in the tree."""
        return list(self._nodes.values())

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in this tree."""
        return node_id in self._nodes

    def find_node_by_status_message(self, status_msg_id: str) -> MessageNode | None:
        """Find the node that has this status message ID (O(1) lookup)."""
        node_id = self._status_to_node.get(status_msg_id)
        return self._nodes.get(node_id) if node_id else None

    def get_descendants(self, node_id: str) -> list[str]:
        """
        Get node_id and all descendant IDs (subtree).

        Returns:
            List of node IDs including the given node.
        """
        if node_id not in self._nodes:
            return []
        result: list[str] = []
        stack = [node_id]
        while stack:
            nid = stack.pop()
            result.append(nid)
            node = self._nodes.get(nid)
            if node:
                stack.extend(node.children_ids)
        return result

    def remove_branch(self, branch_root_id: str) -> list[MessageNode]:
        """
        Remove a subtree (branch_root and all descendants) from the tree.

        Updates parent's children_ids. Caller must hold lock for consistency.
        Does not acquire lock internally.

        Returns:
            List of removed nodes.
        """
        if branch_root_id not in self._nodes:
            return []

        parent = self.get_parent(branch_root_id)
        removed = []
        for nid in self.get_descendants(branch_root_id):
            node = self._nodes.get(nid)
            if node:
                removed.append(node)
                del self._nodes[nid]
                del self._status_to_node[node.status_message_id]

        if parent and branch_root_id in parent.children_ids:
            parent.children_ids = [
                c for c in parent.children_ids if c != branch_root_id
            ]

        logger.debug(f"Removed branch {branch_root_id} ({len(removed)} nodes)")
        return removed

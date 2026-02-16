"""Tree data structures for message queue.

Contains MessageState, MessageNode, and MessageTree classes.
"""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any, cast
from dataclasses import dataclass, field

from .models import IncomingMessage
from loguru import logger


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
    parent_id: Optional[str] = None  # Parent node ID (None for root)
    session_id: Optional[str] = None  # Claude session ID (forked from parent)
    children_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
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
    def from_dict(cls, data: dict) -> "MessageNode":
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
        self._nodes: Dict[str, MessageNode] = {root_node.node_id: root_node}
        self._status_to_node: Dict[str, str] = {
            root_node.status_message_id: root_node.node_id
        }
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._is_processing = False
        self._current_node_id: Optional[str] = None
        self._current_task: Optional[asyncio.Task] = None

        logger.debug(f"Created MessageTree with root {self.root_id}")

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

    def get_node(self, node_id: str) -> Optional[MessageNode]:
        """Get a node by ID (O(1) lookup)."""
        return self._nodes.get(node_id)

    def get_root(self) -> MessageNode:
        """Get the root node."""
        return self._nodes[self.root_id]

    def get_children(self, node_id: str) -> List[MessageNode]:
        """Get all child nodes of a given node."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def get_parent(self, node_id: str) -> Optional[MessageNode]:
        """Get the parent node."""
        node = self._nodes.get(node_id)
        if not node or not node.parent_id:
            return None
        return self._nodes.get(node.parent_id)

    def get_parent_session_id(self, node_id: str) -> Optional[str]:
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
        session_id: Optional[str] = None,
        error_message: Optional[str] = None,
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
                node.completed_at = datetime.now(timezone.utc)

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

    async def dequeue(self) -> Optional[str]:
        """
        Get the next node ID from the queue.

        Returns None if queue is empty.
        """
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_queue_snapshot(self) -> List[str]:
        """
        Get a snapshot of the current queue order.

        Returns:
            List of node IDs in FIFO order.
        """
        async with self._lock:
            # Read internal deque directly to avoid mutating queue state.
            # Drain/put approach would inflate _unfinished_tasks without task_done().
            queue_deque = cast(deque, getattr(self._queue, "_queue"))
            return list(queue_deque)

    def get_queue_size(self) -> int:
        """Get number of messages waiting in queue."""
        return self._queue.qsize()

    def remove_from_queue(self, node_id: str) -> bool:
        """
        Remove node_id from the internal queue if present.

        Caller must hold the tree lock (e.g. via with_lock).
        Returns True if node was removed, False if not in queue.

        Note: asyncio.Queue has no built-in remove; we filter via the internal
        deque. O(n) in queue size; acceptable for typical tree queue sizes.
        """
        queue_deque = cast(deque, getattr(self._queue, "_queue"))
        if node_id not in queue_deque:
            return False
        object.__setattr__(
            self._queue, "_queue", deque(x for x in queue_deque if x != node_id)
        )
        return True

    @asynccontextmanager
    async def with_lock(self):
        """Async context manager for tree lock. Use when multiple operations need atomicity."""
        async with self._lock:
            yield

    def set_processing_state(self, node_id: Optional[str], is_processing: bool) -> None:
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
    ) -> List["MessageNode"]:
        """
        Drain the queue, mark each node as ERROR, and return affected nodes.
        Does not acquire lock; caller must ensure no concurrent queue access.
        """
        nodes: List[MessageNode] = []
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
    def current_node_id(self) -> Optional[str]:
        """Get the ID of the node currently being processed."""
        return self._current_node_id

    def to_dict(self) -> dict:
        """Serialize tree to dictionary."""
        return {
            "root_id": self.root_id,
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MessageTree":
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

    def all_nodes(self) -> List[MessageNode]:
        """Get all nodes in the tree."""
        return list(self._nodes.values())

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in this tree."""
        return node_id in self._nodes

    def find_node_by_status_message(self, status_msg_id: str) -> Optional[MessageNode]:
        """Find the node that has this status message ID (O(1) lookup)."""
        node_id = self._status_to_node.get(status_msg_id)
        return self._nodes.get(node_id) if node_id else None

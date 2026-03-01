"""Async queue processor for message trees.

Handles the async processing lifecycle of tree nodes.
"""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from providers.common import get_user_facing_error_message

from .data import MessageNode, MessageState, MessageTree


class TreeQueueProcessor:
    """
    Handles async queue processing for a single tree.

    Separates the async processing logic from the data management.
    """

    def __init__(
        self,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None = None,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]]
        | None = None,
    ):
        self._queue_update_callback = queue_update_callback
        self._node_started_callback = node_started_callback

    def set_queue_update_callback(
        self,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None,
    ) -> None:
        """Update the callback used to refresh queue positions."""
        self._queue_update_callback = queue_update_callback

    def set_node_started_callback(
        self,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]] | None,
    ) -> None:
        """Update the callback used when a queued node starts processing."""
        self._node_started_callback = node_started_callback

    async def _notify_queue_updated(self, tree: MessageTree) -> None:
        """Invoke queue update callback if set."""
        if not self._queue_update_callback:
            return
        try:
            await self._queue_update_callback(tree)
        except Exception as e:
            logger.warning(f"Queue update callback failed: {e}")

    async def _notify_node_started(self, tree: MessageTree, node_id: str) -> None:
        """Invoke node started callback if set."""
        if not self._node_started_callback:
            return
        try:
            await self._node_started_callback(tree, node_id)
        except Exception as e:
            logger.warning(f"Node started callback failed: {e}")

    async def process_node(
        self,
        tree: MessageTree,
        node: MessageNode,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> None:
        """Process a single node and then check the queue."""
        # Skip if already in terminal state (e.g. from error propagation)
        if node.state == MessageState.ERROR:
            logger.info(
                f"Skipping node {node.node_id} as it is already in state {node.state}"
            )
            # Still need to check for next messages
            await self._process_next(tree, processor)
            return

        try:
            await processor(node.node_id, node)
        except asyncio.CancelledError:
            logger.info(f"Task for node {node.node_id} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error processing node {node.node_id}: {e}")
            await tree.update_state(
                node.node_id,
                MessageState.ERROR,
                error_message=get_user_facing_error_message(e),
            )
        finally:
            tree.clear_current_node()
            # Check if there are more messages in the queue
            await self._process_next(tree, processor)

    async def _process_next(
        self,
        tree: MessageTree,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> None:
        """Process the next message in queue, if any."""
        next_node_id = None
        node = None
        async with tree.with_lock():
            next_node_id = await tree.dequeue()

            if not next_node_id:
                tree.set_processing_state(None, False)
                logger.debug(f"Tree {tree.root_id} queue empty, marking as free")
                return

            tree.set_processing_state(next_node_id, True)
            logger.info(f"Processing next queued node {next_node_id}")

            # Process next node (outside lock)
            node = tree.get_node(next_node_id)
            if node:
                tree.set_current_task(
                    asyncio.create_task(self.process_node(tree, node, processor))
                )

        # Notify that this node has started processing and refresh queue positions.
        if next_node_id:
            await self._notify_node_started(tree, next_node_id)
            await self._notify_queue_updated(tree)

    async def enqueue_and_start(
        self,
        tree: MessageTree,
        node_id: str,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> bool:
        """
        Enqueue a node or start processing immediately.

        Args:
            tree: The message tree
            node_id: Node to process
            processor: Async function to process the node

        Returns:
            True if queued, False if processing immediately
        """
        async with tree.with_lock():
            if tree.is_processing:
                tree.put_queue_unlocked(node_id)
                queue_size = tree.get_queue_size()
                logger.info(f"Queued node {node_id}, position {queue_size}")
                return True
            else:
                tree.set_processing_state(node_id, True)

                # Process outside the lock
                node = tree.get_node(node_id)
                if node:
                    tree.set_current_task(
                        asyncio.create_task(self.process_node(tree, node, processor))
                    )
                return False

    def cancel_current(self, tree: MessageTree) -> bool:
        """Cancel the currently running task in a tree."""
        return tree.cancel_current_task()

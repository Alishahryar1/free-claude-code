import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from messaging.models import IncomingMessage
from messaging.trees.data import MessageNode, MessageState, MessageTree
from messaging.trees.processor import TreeQueueProcessor


@pytest.fixture
def tree_processor():
    return TreeQueueProcessor()


@pytest.fixture
def sample_incoming():
    return IncomingMessage(
        text="test message",
        chat_id="chat123",
        user_id="user456",
        message_id="msg789",
        platform="telegram",
    )


@pytest.fixture
def sample_node(sample_incoming):
    return MessageNode(
        node_id="msg789", incoming=sample_incoming, status_message_id="status123"
    )


@pytest.fixture
def sample_tree(sample_node):
    return MessageTree(sample_node)


@pytest.mark.asyncio
async def test_process_node_success(tree_processor, sample_tree, sample_node):
    processor = AsyncMock()

    await tree_processor.process_node(sample_tree, sample_node, processor)

    processor.assert_called_once_with(sample_node.node_id, sample_node)
    assert sample_tree._current_node_id is None


@pytest.mark.asyncio
async def test_process_node_cancelled(tree_processor, sample_tree, sample_node):
    processor = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await tree_processor.process_node(sample_tree, sample_node, processor)

    assert sample_tree._current_node_id is None


@pytest.mark.asyncio
async def test_process_node_exception(tree_processor, sample_tree, sample_node):
    processor = AsyncMock(side_effect=Exception("Test error"))

    # We need to mock update_state to verify it was called
    sample_tree.update_state = AsyncMock()

    await tree_processor.process_node(sample_tree, sample_node, processor)

    sample_tree.update_state.assert_called_once_with(
        sample_node.node_id, MessageState.ERROR, error_message="Test error"
    )
    assert sample_tree._current_node_id is None


@pytest.mark.asyncio
async def test_enqueue_and_start_when_free(tree_processor, sample_tree):
    processor = AsyncMock()
    node_id = "node1"

    # Mock get_node to return a node
    node = MagicMock(spec=MessageNode)
    sample_tree.get_node = MagicMock(return_value=node)

    was_queued = await tree_processor.enqueue_and_start(sample_tree, node_id, processor)

    assert was_queued is False
    assert sample_tree._is_processing is True
    assert sample_tree._current_node_id == node_id
    assert sample_tree._current_task is not None

    # Clean up task
    sample_tree._current_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await sample_tree._current_task


@pytest.mark.asyncio
async def test_enqueue_and_start_when_busy(tree_processor, sample_tree):
    processor = AsyncMock()
    sample_tree._is_processing = True
    node_id = "node1"

    was_queued = await tree_processor.enqueue_and_start(sample_tree, node_id, processor)

    assert was_queued is True
    assert sample_tree._queue.qsize() == 1
    assert sample_tree._queue.get_nowait() == node_id


def test_cancel_current_task(tree_processor, sample_tree):
    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.done.return_value = False
    sample_tree._current_task = mock_task

    cancelled = tree_processor.cancel_current(sample_tree)

    assert cancelled is True
    mock_task.cancel.assert_called_once()


def test_cancel_current_task_already_done(tree_processor, sample_tree):
    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.done.return_value = True
    sample_tree._current_task = mock_task

    cancelled = tree_processor.cancel_current(sample_tree)

    assert cancelled is False
    mock_task.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_process_next_queue_empty(tree_processor, sample_tree):
    processor = AsyncMock()
    sample_tree._is_processing = True

    await tree_processor._process_next(sample_tree, processor)

    assert sample_tree._is_processing is False


@pytest.mark.asyncio
async def test_process_next_with_item(tree_processor, sample_tree):
    processor = AsyncMock()
    await sample_tree._queue.put("next_node")

    node = MagicMock(spec=MessageNode)
    sample_tree.get_node = MagicMock(return_value=node)

    await tree_processor._process_next(sample_tree, processor)

    assert sample_tree._current_node_id == "next_node"
    assert sample_tree._current_task is not None

    # Clean up
    sample_tree._current_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await sample_tree._current_task


@pytest.mark.asyncio
async def test_process_next_triggers_queue_update(sample_tree):
    callback = AsyncMock()
    processor = TreeQueueProcessor(queue_update_callback=callback)

    await sample_tree._queue.put("next_node")
    sample_tree.get_node = MagicMock(return_value=None)

    await processor._process_next(sample_tree, AsyncMock())

    callback.assert_awaited_once_with(sample_tree)


@pytest.mark.asyncio
async def test_process_next_triggers_node_started(sample_tree):
    node_started = AsyncMock()
    processor = TreeQueueProcessor(node_started_callback=node_started)

    await sample_tree._queue.put("next_node")
    sample_tree.get_node = MagicMock(return_value=None)

    await processor._process_next(sample_tree, AsyncMock())

    node_started.assert_awaited_once_with(sample_tree, "next_node")

from unittest.mock import AsyncMock, patch

import pytest

from messaging.handler import ClaudeMessageHandler
from messaging.models import IncomingMessage
from messaging.session import SessionStore
from messaging.trees.queue_manager import TreeQueueManager


@pytest.mark.asyncio
async def test_reply_to_old_status_message_after_restore_routes_to_parent(
    tmp_path, mock_platform, mock_cli_manager
):
    # Build a persisted tree with a root node A and a bot status message id.
    store_path = tmp_path / "sessions.json"
    store = SessionStore(storage_path=str(store_path))

    handler1 = ClaudeMessageHandler(mock_platform, mock_cli_manager, store)
    a_incoming = IncomingMessage(
        text="A",
        chat_id="chat_1",
        user_id="user_1",
        message_id="A",
        platform="telegram",
    )
    tree = await handler1.tree_queue.create_tree(
        "A", a_incoming, status_message_id="status_A"
    )
    handler1.tree_queue.register_node("status_A", tree.root_id)
    store.register_node("status_A", tree.root_id)
    store.save_tree(tree.root_id, tree.to_dict())
    store.flush_pending_save()

    # "Restart": new store instance loads from disk, and we restore TreeQueueManager.
    store2 = SessionStore(storage_path=str(store_path))
    handler2 = ClaudeMessageHandler(mock_platform, mock_cli_manager, store2)
    handler2.tree_queue = TreeQueueManager.from_dict(
        {"trees": store2.get_all_trees(), "node_to_tree": store2.get_node_mapping()},
        queue_update_callback=handler2.update_queue_positions,
        node_started_callback=handler2.mark_node_processing,
    )

    # Prevent background task scheduling; we only want to validate routing/tree mutation.
    mock_platform.queue_send_message = AsyncMock(return_value="status_reply")

    reply = IncomingMessage(
        text="R1",
        chat_id="chat_1",
        user_id="user_1",
        message_id="R1",
        platform="telegram",
        reply_to_message_id="status_A",
    )

    with patch.object(handler2.tree_queue, "enqueue", AsyncMock(return_value=False)):
        await handler2.handle_message(reply)

    restored_tree = handler2.tree_queue.get_tree_for_node("A")
    assert restored_tree is not None
    node_r1 = restored_tree.get_node("R1")
    assert node_r1 is not None
    assert node_r1.parent_id == "A"


@pytest.mark.asyncio
async def test_reply_to_old_status_message_without_mapping_creates_new_conversation(
    tmp_path, mock_platform, mock_cli_manager
):
    store_path = tmp_path / "sessions.json"
    store = SessionStore(storage_path=str(store_path))

    handler1 = ClaudeMessageHandler(mock_platform, mock_cli_manager, store)
    a_incoming = IncomingMessage(
        text="A",
        chat_id="chat_1",
        user_id="user_1",
        message_id="A",
        platform="telegram",
    )
    tree = await handler1.tree_queue.create_tree(
        "A", a_incoming, status_message_id="status_A"
    )
    # Intentionally do NOT register "status_A" mapping.
    store.save_tree(tree.root_id, tree.to_dict())
    store.flush_pending_save()

    store2 = SessionStore(storage_path=str(store_path))
    handler2 = ClaudeMessageHandler(mock_platform, mock_cli_manager, store2)
    handler2.tree_queue = TreeQueueManager.from_dict(
        {"trees": store2.get_all_trees(), "node_to_tree": store2.get_node_mapping()},
        queue_update_callback=handler2.update_queue_positions,
        node_started_callback=handler2.mark_node_processing,
    )
    mock_platform.queue_send_message = AsyncMock(return_value="status_reply")

    reply = IncomingMessage(
        text="R1",
        chat_id="chat_1",
        user_id="user_1",
        message_id="R1",
        platform="telegram",
        reply_to_message_id="status_A",
    )

    with patch.object(handler2.tree_queue, "enqueue", AsyncMock(return_value=False)):
        await handler2.handle_message(reply)

    # Since the mapping is missing, this should be treated as a new conversation.
    new_tree = handler2.tree_queue.get_tree_for_node("R1")
    assert new_tree is not None
    assert new_tree.root_id == "R1"

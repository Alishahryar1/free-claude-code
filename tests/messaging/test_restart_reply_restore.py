import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from free_claude_code.messaging.models import IncomingMessage, MessageScope
from free_claude_code.messaging.trees import MessagingStore, TreeIdentity
from free_claude_code.messaging.trees.node import MessageNode, MessageState
from free_claude_code.messaging.trees.runtime import MessageTree
from free_claude_code.messaging.workflow import MessagingWorkflow

TELEGRAM_CHAT_1 = MessageScope(platform="telegram", chat_id="chat_1")


async def _wait_for_idle(workflow: MessagingWorkflow) -> None:
    await asyncio.wait_for(workflow.tree_queue.wait_idle(), timeout=5)


async def _write_completed_root(store: MessagingStore) -> None:
    tree = MessageTree(
        MessageNode(
            node_id="A",
            scope=TELEGRAM_CHAT_1,
            prompt="A",
            status_message_id="status_A",
            state=MessageState.COMPLETED,
            session_id="sess_A",
        )
    )
    await store.commit_trees((await tree.snapshot(),))


async def _write_interrupted_root(store: MessagingStore) -> None:
    tree = MessageTree(
        MessageNode(
            node_id="A",
            scope=TELEGRAM_CHAT_1,
            prompt="A",
            status_message_id="status_A",
            state=MessageState.IN_PROGRESS,
        )
    )
    await store.commit_trees((await tree.snapshot(),))


def _successful_session(session_id: str):
    session = MagicMock()

    async def events(*_args, **_kwargs):
        yield {"type": "session_info", "session_id": session_id}
        yield {"type": "exit", "code": 0, "stderr": None}

    session.start_task = events
    return session


@pytest.mark.asyncio
async def test_reply_to_old_status_message_after_restore_routes_to_parent(
    messaging_store_factory,
    tmp_path,
    mock_platform,
    mock_cli_manager,
) -> None:
    store_path = tmp_path / "sessions.json"
    await _write_completed_root(
        await messaging_store_factory(storage_path=str(store_path))
    )

    restored_store = await messaging_store_factory(storage_path=str(store_path))
    workflow = MessagingWorkflow(mock_platform, mock_cli_manager, restored_store)
    (await workflow.restore())
    mock_platform.queue_send_message = AsyncMock(return_value="status_reply")
    mock_cli_manager.get_or_create_session.return_value = (
        _successful_session("sess_R1"),
        "pending_R1",
        True,
    )

    await workflow.handle_message(
        IncomingMessage(
            text="R1",
            chat_id="chat_1",
            user_id="user_1",
            message_id="R1",
            platform="telegram",
            reply_to_message_id="status_A",
        )
    )
    await _wait_for_idle(workflow)

    reply = await workflow.tree_queue.get_node(TELEGRAM_CHAT_1, "R1")
    assert reply is not None
    assert reply.parent_id == "A"
    mock_cli_manager.get_or_create_session.assert_called_with(session_id="sess_A")


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_legacy_session_json_restores_through_workflow_and_routes_reply(
    messaging_store_factory,
    wrapped: bool,
    tmp_path,
    mock_platform,
    mock_cli_manager,
) -> None:
    legacy_tree = {
        "root_id": "A",
        "nodes": {
            "A": {
                "node_id": "A",
                "incoming": {
                    "text": "legacy prompt",
                    "chat_id": "chat_1",
                    "user_id": "legacy-user",
                    "message_id": "A",
                    "platform": "telegram",
                },
                "status_message_id": "status_A",
                "state": "completed",
                "parent_id": None,
                "session_id": "sess_A",
                "children_ids": [],
                "created_at": "2025-01-01T00:00:00+00:00",
                "completed_at": "2025-01-01T00:00:01+00:00",
                "error_message": None,
            }
        },
    }
    conversation = {"trees": {"A": legacy_tree}}
    payload = (
        {"conversation": conversation, "message_log": {}}
        if wrapped
        else {
            **conversation,
            "node_to_tree": {"A": "A"},
            "message_log": {},
        }
    )
    store_path = tmp_path / "sessions.json"
    store_path.write_text(json.dumps(payload), encoding="utf-8")
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        (await messaging_store_factory(storage_path=str(store_path))),
    )
    (await workflow.restore())
    mock_platform.queue_send_message = AsyncMock(return_value="status_reply")
    mock_cli_manager.get_or_create_session.return_value = (
        _successful_session("sess_R1"),
        "pending_R1",
        True,
    )

    await workflow.handle_message(
        IncomingMessage(
            text="continue legacy tree",
            chat_id="chat_1",
            user_id="user_1",
            message_id="R1",
            platform="telegram",
            reply_to_message_id="status_A",
        )
    )
    await _wait_for_idle(workflow)

    reply = await workflow.tree_queue.get_node(TELEGRAM_CHAT_1, "R1")
    assert reply is not None and reply.parent_id == "A"
    mock_cli_manager.get_or_create_session.assert_called_with(session_id="sess_A")


@pytest.mark.asyncio
async def test_save_tree_snapshot_restores_status_lookup_without_manual_index(
    messaging_store_factory,
    tmp_path,
    mock_platform,
    mock_cli_manager,
) -> None:
    store_path = tmp_path / "sessions.json"
    await _write_completed_root(
        await messaging_store_factory(storage_path=str(store_path))
    )
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        (await messaging_store_factory(storage_path=str(store_path))),
    )
    (await workflow.restore())

    assert await workflow.tree_queue.resolve_node_id(TELEGRAM_CHAT_1, "status_A") == "A"


@pytest.mark.asyncio
async def test_reply_clear_purges_removed_status_mapping_from_persisted_store(
    messaging_store_factory,
    tmp_path,
    mock_platform,
    mock_cli_manager,
) -> None:
    store_path = tmp_path / "sessions.json"
    store = await messaging_store_factory(storage_path=str(store_path))
    workflow = MessagingWorkflow(mock_platform, mock_cli_manager, store)
    mock_platform.queue_send_message = AsyncMock(
        side_effect=["root_status", "child_status"]
    )
    mock_cli_manager.get_or_create_session.side_effect = [
        (_successful_session("sess_root"), "pending_root", True),
        (_successful_session("sess_child"), "pending_child", True),
    ]

    await workflow.handle_message(
        IncomingMessage(
            text="root",
            chat_id="chat_1",
            user_id="user_1",
            message_id="root",
            platform="telegram",
        )
    )
    await _wait_for_idle(workflow)
    await workflow.handle_message(
        IncomingMessage(
            text="child",
            chat_id="chat_1",
            user_id="user_1",
            message_id="child",
            platform="telegram",
            reply_to_message_id="root",
        )
    )
    await _wait_for_idle(workflow)
    await workflow.handle_message(
        IncomingMessage(
            text="/clear",
            chat_id="chat_1",
            user_id="user_1",
            message_id="clear_command",
            platform="telegram",
            reply_to_message_id="child",
        )
    )

    identity = TreeIdentity(scope=TELEGRAM_CHAT_1, root_id="root")
    persisted = await (
        await messaging_store_factory(storage_path=str(store_path))
    ).load_conversation_snapshot()
    tree = persisted.get_tree(identity)
    assert tree is not None
    assert tree.lookup_ids() == {"root", "root_status"}


@pytest.mark.asyncio
async def test_restore_repairs_interrupted_status_after_delivery_starts(
    messaging_store_factory,
    tmp_path,
    mock_platform,
    mock_cli_manager,
) -> None:
    store_path = tmp_path / "sessions.json"
    await _write_interrupted_root(
        await messaging_store_factory(storage_path=str(store_path))
    )
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        (await messaging_store_factory(storage_path=str(store_path))),
        platform_name="telegram",
    )
    (await workflow.restore())

    await workflow.repair_restored_statuses()
    await workflow.repair_restored_statuses()

    mock_platform.queue_edit_message.assert_awaited_once_with(
        TELEGRAM_CHAT_1.chat_id,
        "status_A",
        workflow.format_status("❌", "Interrupted by server restart"),
        parse_mode="MarkdownV2",
        fire_and_forget=False,
    )


@pytest.mark.asyncio
async def test_workflow_close_stops_cli_and_waits_for_claim_cleanup(
    mock_platform,
    mock_cli_manager,
    mock_session_store,
) -> None:
    workflow = MessagingWorkflow(mock_platform, mock_cli_manager, mock_session_store)
    cleanup_release = asyncio.Event()
    wait_started = asyncio.Event()
    events: list[str] = []

    async def stop_all() -> int:
        events.append("stop")
        return 1

    async def wait_idle() -> None:
        events.append("wait")
        wait_started.set()
        await cleanup_release.wait()

    mock_cli_manager.stop_all = AsyncMock(side_effect=stop_all)
    workflow.tree_queue.wait_idle = AsyncMock(side_effect=wait_idle)

    close_task = asyncio.create_task(workflow.close())
    await wait_started.wait()

    assert events == ["stop", "wait"]

    cleanup_release.set()
    await close_task

    assert events == ["stop", "wait"]

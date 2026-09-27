"""Workflow publication and control effects at real SQLite commit boundaries."""

import asyncio
import sqlite3
from unittest.mock import patch

import pytest
import pytest_asyncio

from free_claude_code.messaging.models import IncomingMessage
from free_claude_code.messaging.platforms.ports import MessagingStartupNotice
from free_claude_code.messaging.trees import MessagingStorageError, TreeQueueManager
from free_claude_code.messaging.workflow import MessagingWorkflow

pytestmark = pytest.mark.asyncio


def message(node):
    return IncomingMessage(
        platform="telegram", chat_id="chat", user_id="user", message_id=node, text=node
    )


async def control(workflow, operation):
    if operation == "stop":
        return await workflow.stop_all_tasks()
    return await workflow.clear_chat("telegram", "chat")


@pytest.mark.parametrize("operation", ["stop", "clear"])
@pytest.mark.parametrize("write_fails", [False, True])
async def test_delayed_prompt_is_invalidated_only_by_committed_control(
    messaging_store_factory, mock_platform, mock_cli_manager, operation, write_fails
):
    storage = await messaging_store_factory()
    workflow = MessagingWorkflow(mock_platform, mock_cli_manager, storage)
    release, status_started, status_release = (asyncio.Event() for _ in range(3))
    processed = []

    async def process(claim):
        processed.append(claim.node.node_id)
        await release.wait()
        await manager.complete_claim(claim, "native")

    manager = TreeQueueManager(process, store=storage)
    workflow._tree_queue = manager
    await manager.admit(message("existing"), "status-existing")

    async def send(*args, **kwargs):
        status_started.set()
        await status_release.wait()
        return "status-new"

    mock_platform.queue_send_message.side_effect = send
    pending = asyncio.create_task(workflow.handle_message(message("new")))
    await status_started.wait()
    before = await storage.load_conversation_snapshot()
    execute = storage.database.execute

    def reject(operation, *, write=True):
        if write:
            raise sqlite3.OperationalError("test disk failure")
        return execute(operation, write=False)

    try:
        if write_fails:
            with (
                patch.object(storage.database, "execute", reject),
                pytest.raises(MessagingStorageError),
            ):
                await control(workflow, operation)
            assert await storage.load_conversation_snapshot() == before
            assert await manager.snapshot() == before
        else:
            await control(workflow, operation)
        status_release.set()
        await pending
        release.set()
        await asyncio.wait_for(manager.wait_idle(), 3)
        assert processed.count("new") == (1 if write_fails else 0)
        if write_fails:
            mock_platform.queue_delete_messages.assert_not_awaited()
        else:
            mock_platform.queue_delete_messages.assert_awaited_once_with(
                "chat", ["status-new"], fire_and_forget=False
            )
    finally:
        status_release.set()
        release.set()
        await pending
        await workflow.close()


async def test_failed_clear_preserves_delayed_startup_notice(
    messaging_store_factory, mock_platform, mock_cli_manager
):
    storage = await messaging_store_factory()
    workflow = MessagingWorkflow(
        mock_platform, mock_cli_manager, storage, platform_name="telegram"
    )
    started, release = asyncio.Event(), asyncio.Event()

    async def send(*args, **kwargs):
        started.set()
        await release.wait()
        return "notice"

    mock_platform.queue_send_message.side_effect = send
    notice = asyncio.create_task(
        workflow.publish_startup_notice(
            MessagingStartupNotice(chat_id="chat", transport_label="test")
        )
    )
    await started.wait()
    execute = storage.database.execute

    def reject(operation, *, write=True):
        if write:
            raise sqlite3.OperationalError("test disk failure")
        return execute(operation, write=False)

    try:
        with (
            patch.object(storage.database, "execute", reject),
            pytest.raises(MessagingStorageError),
        ):
            await workflow.clear_chat("telegram", "chat")
    finally:
        release.set()
        await notice
        await workflow.close()
    assert await storage.get_tracked_message_ids_for_chat("telegram", "chat") == [
        "notice"
    ]
    mock_platform.queue_delete_messages.assert_not_awaited()


@pytest_asyncio.fixture
async def interrupted_workflow(
    messaging_store_factory, mock_platform, mock_cli_manager, monkeypatch
):
    monkeypatch.setattr(
        "free_claude_code.messaging.trees.manager.CANCEL_TASK_DRAIN_TIMEOUT_S", 0.01
    )
    storage = await messaging_store_factory()
    workflow = MessagingWorkflow(mock_platform, mock_cli_manager, storage)
    fail_now, faulted, release = (asyncio.Event() for _ in range(3))
    cancelled, notifications = set(), []

    async def process(claim):
        node = claim.node.node_id
        if node == "root":
            await fail_now.wait()
            try:
                await manager.record_session(claim, "rejected-session")
            except MessagingStorageError:
                faulted.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        cancelled.add(node)
                raise
        else:
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.add(node)
                raise

    manager = TreeQueueManager(
        process, store=storage, unexpected_failure_callback=notifications.append
    )
    workflow._tree_queue = manager
    await manager.admit(message("root"), "status-root")
    await manager.admit(
        message("child"), "status-child", parent_reference_id="status-root"
    )
    await manager.admit(
        message("sibling"), "status-sibling", parent_reference_id="root"
    )
    await manager.admit(message("healthy"), "status-healthy")
    await storage.database.run(
        lambda connection: connection.execute(
            "CREATE TRIGGER reject_session BEFORE UPDATE ON messaging_nodes "
            "WHEN NEW.session_id='rejected-session' "
            "BEGIN SELECT RAISE(ABORT, 'test storage failure'); END"
        )
    )
    fail_now.set()
    await asyncio.wait_for(faulted.wait(), 3)
    await storage.database.run(
        lambda connection: connection.execute("DROP TRIGGER reject_session")
    )
    try:
        yield workflow, storage, release, notifications, cancelled
    finally:
        release.set()
        await workflow.close()


@pytest.mark.parametrize(
    ("operation", "remaining_feedback"),
    [
        ("stop-all", set()),
        ("stop-root", {"child", "sibling"}),
        ("stop-child", {"root", "sibling"}),
        ("clear-child", {"root", "sibling"}),
        ("clear-status", {"sibling"}),
    ],
)
async def test_controls_during_interrupted_cleanup_preserve_ownership_and_feedback(
    interrupted_workflow, mock_cli_manager, operation, remaining_feedback
):
    workflow, storage, release, notifications, cancelled = interrupted_workflow
    manager = workflow.tree_queue
    scope = message("root").scope
    if operation == "stop-all":
        await workflow.stop_all_tasks()
        mock_cli_manager.stop_all.assert_awaited_once()
        assert cancelled == {"root", "healthy"}
    elif operation.startswith("stop-"):
        await workflow.stop_reply(scope, operation.removeprefix("stop-"))
    else:
        result = await workflow.clear_reply(
            scope, "child" if operation == "clear-child" else "status-root"
        )
        assert result is not None
        assert "status-child" in result.delete_message_ids
    # Controls do not reopen admission while the interrupted owner is alive.
    with pytest.raises(MessagingStorageError, match="cleaning up"):
        await manager.admit(
            message("later"), "status-later", parent_reference_id="root"
        )
    await manager.cancel_node(scope, "healthy")
    release.set()
    await asyncio.wait_for(manager.wait_idle(), 3)
    assert {
        target.node_id for result in notifications for target in result.affected
    } == remaining_feedback
    assert await manager.snapshot() == await storage.load_conversation_snapshot()


@pytest.mark.parametrize(
    "operation", ["stop-all", "stop-root", "clear-child", "clear-status", "clear-all"]
)
async def test_failed_control_during_cleanup_preserves_other_workers_and_gate(
    interrupted_workflow, mock_cli_manager, operation
):
    workflow, storage, _release, _notifications, cancelled = interrupted_workflow
    manager = workflow.tree_queue
    before = await storage.load_conversation_snapshot()
    execute = storage.database.execute

    def reject(operation, *, write=True):
        if write:
            raise sqlite3.OperationalError("test disk failure")
        return execute(operation, write=False)

    with (
        patch.object(storage.database, "execute", reject),
        pytest.raises(MessagingStorageError),
    ):
        if operation == "stop-all":
            await workflow.stop_all_tasks()
        elif operation == "stop-root":
            await workflow.stop_reply(message("root").scope, "root")
        elif operation == "clear-all":
            await workflow.clear_chat("telegram", "chat")
        else:
            await workflow.clear_reply(
                message("root").scope,
                "child" if operation == "clear-child" else "status-root",
            )
    assert await storage.load_conversation_snapshot() == before
    assert await manager.snapshot() == before
    assert not cancelled
    mock_cli_manager.stop_all.assert_not_awaited()
    with pytest.raises(MessagingStorageError, match="cleaning up"):
        await manager.admit(
            message("later"), "status-later", parent_reference_id="root"
        )


@pytest.mark.parametrize(
    ("operation", "failure"),
    [("stop", "callback"), ("clear", "callback"), ("stop", "cli")],
)
async def test_committed_barrier_survives_postcommit_failure(
    messaging_store_factory, mock_platform, mock_cli_manager, operation, failure
):
    storage = await messaging_store_factory()
    workflow = MessagingWorkflow(mock_platform, mock_cli_manager, storage)
    status_started, status_release, effect_started, effect_release = (
        asyncio.Event() for _ in range(4)
    )
    processed = []

    async def process(claim):
        processed.append(claim.node.node_id)
        await asyncio.Event().wait()

    async def fail_effect(*_args):
        effect_started.set()
        await effect_release.wait()
        if failure == "callback":
            raise asyncio.CancelledError("delivery cancelled")
        raise RuntimeError("CLI cleanup failed")

    manager = TreeQueueManager(
        process,
        store=storage,
        queue_update_callback=fail_effect if failure == "callback" else None,
    )
    workflow._tree_queue = manager
    root = await manager.admit(message("root"), "status-root")
    assert root.claim is not None
    await manager.admit(message("child"), "status-child", parent_reference_id="root")

    async def send(*args, **kwargs):
        status_started.set()
        await status_release.wait()
        return "status-new"

    mock_platform.queue_send_message.side_effect = send
    if failure == "cli":
        mock_cli_manager.stop_all.side_effect = fail_effect
    pending = asyncio.create_task(workflow.handle_message(message("new")))
    await status_started.wait()
    command = asyncio.create_task(control(workflow, operation))
    try:
        await asyncio.wait_for(effect_started.wait(), 3)
        saved = await storage.load_conversation_snapshot()
        if operation == "clear":
            assert saved.is_empty
        else:
            assert all(
                node["state"] == "error"
                for node in saved.trees[root.claim.identity].nodes.values()
            )
        effect_release.set()
        with pytest.raises(
            asyncio.CancelledError if failure == "callback" else RuntimeError
        ):
            await command
        status_release.set()
        await pending
        assert processed == ["root"]
        mock_platform.queue_delete_messages.assert_awaited_once_with(
            "chat", ["status-new"], fire_and_forget=False
        )
    finally:
        effect_release.set()
        status_release.set()
        await asyncio.gather(pending, command, return_exceptions=True)
        mock_cli_manager.stop_all.side_effect = None
        await workflow.close()


@pytest.mark.parametrize("operation", ["stop", "clear"])
async def test_successful_empty_control_still_invalidates_delayed_admission(
    messaging_store_factory, mock_platform, mock_cli_manager, operation
):
    storage = await messaging_store_factory()
    workflow = MessagingWorkflow(mock_platform, mock_cli_manager, storage)
    started, release = asyncio.Event(), asyncio.Event()

    async def send(*args, **kwargs):
        started.set()
        await release.wait()
        return "status-new"

    mock_platform.queue_send_message.side_effect = send
    prompt = asyncio.create_task(workflow.handle_message(message("new")))
    await started.wait()
    try:
        await control(workflow, operation)
    finally:
        release.set()
        await prompt
        await workflow.close()
    assert (await storage.load_conversation_snapshot()).is_empty
    mock_cli_manager.get_or_create_session.assert_not_awaited()
    mock_platform.queue_delete_messages.assert_awaited_once_with(
        "chat", ["status-new"], fire_and_forget=False
    )

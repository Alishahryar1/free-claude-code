import asyncio
import json
import sqlite3
import threading
from contextlib import closing
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from free_claude_code.messaging.models import IncomingMessage, MessageScope
from free_claude_code.messaging.trees import TreeQueueManager
from free_claude_code.messaging.trees.snapshot import TreeSnapshot
from free_claude_code.messaging.workflow import MessagingWorkflow
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from free_claude_code.runtime.messaging_import import import_legacy
from free_claude_code.runtime.messaging_sqlite import SQLiteMessagingStore
from free_claude_code.runtime.sqlite_database import SQLiteDatabase

pytestmark = pytest.mark.asyncio


def tree(platform="telegram", chat="chat", root="root"):
    return TreeSnapshot(
        scope=MessageScope(platform=platform, chat_id=chat),
        root_id=root,
        nodes={
            root: {
                "node_id": root,
                "status_message_id": f"status-{root}",
                "state": "completed",
                "parent_id": None,
                "parent_reference_id": None,
                "session_id": "native-session",
            }
        },
    )


@pytest_asyncio.fixture
async def storage(tmp_path):
    database = SQLiteDatabase(tmp_path / "fcc.db", tmp_path / "code.lock")
    await database.start()
    try:
        yield SQLiteMessagingStore(database, managed_message_cap=2)
    finally:
        await database.close()


async def test_tree_and_managed_messages_are_durable_and_scope_isolated(storage):
    first, second = tree(), tree("discord")
    await storage.commit_trees((first, second))
    for message in ("one", "two", "two", "three"):
        await storage.record_message_id("telegram", "chat", message, "in", "prompt")
    assert await storage.get_tracked_message_ids_for_chat("telegram", "chat") == [
        "two",
        "three",
    ]
    await storage.commit_trees((), clear_scope=first.scope)
    loaded = await storage.load_conversation_snapshot()
    assert list(loaded.trees) == [second.identity]
    assert loaded.trees[second.identity].nodes["root"]["session_id"] == "native-session"
    assert not await storage.get_tracked_message_ids_for_chat("telegram", "chat")


async def test_reference_collision_rolls_back_whole_write(storage):
    original = tree()
    await storage.commit_trees((original,))
    collision = tree(root="status-root")
    with pytest.raises(sqlite3.IntegrityError):
        await storage.commit_trees((collision,))
    assert (await storage.load_conversation_snapshot()).trees == {
        original.identity: original
    }


@pytest.mark.parametrize("damaged", [False, True])
async def test_import_keeps_valid_data_and_never_replays_after_clear(
    storage, tmp_path, damaged
):
    source = tmp_path / "sessions.json"
    valid = tree()
    source.write_text(
        json.dumps(
            {
                "conversation": {
                    "trees": [valid.to_json(), *([{"broken": True}] if damaged else [])]
                }
            }
        )
    )
    warning = await import_legacy(storage.database, source)
    assert bool(warning) is damaged
    assert source.exists() is damaged
    assert (await storage.load_conversation_snapshot()).trees == {valid.identity: valid}
    await storage.commit_trees((), clear_scope=valid.scope)
    await import_legacy(storage.database, source)
    assert (await storage.load_conversation_snapshot()).is_empty


async def test_unreadable_json_does_not_block_new_conversations(storage, tmp_path):
    source = tmp_path / "sessions.json"
    source.write_text('{"conversation":')
    assert await import_legacy(storage.database, source)
    assert source.read_text() == '{"conversation":'
    await storage.commit_trees((tree(),))
    assert not (await storage.load_conversation_snapshot()).is_empty
    assert await import_legacy(storage.database, source) is None


async def test_pending_import_keeps_restart_repair_information(storage, tmp_path):
    pending = tree()
    pending.nodes["root"]["state"] = "in_progress"
    source = tmp_path / "sessions.json"
    source.write_text(json.dumps(pending_to_legacy(pending)))
    await import_legacy(storage.database, source)
    assert (await storage.load_conversation_snapshot()).trees[pending.identity].nodes[
        "root"
    ]["state"] == "in_progress"


def pending_to_legacy(snapshot):
    return {"trees": {snapshot.root_id: snapshot.to_json()}}


async def test_new_schema_preserves_foreign_keys(storage):
    await storage.commit_trees((tree(),))
    with closing(sqlite3.connect(storage.database.path)) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


async def test_sqlite_prevents_deleting_a_root_without_its_tree(storage):
    original = tree()
    await storage.commit_trees((original,))
    with pytest.raises(sqlite3.IntegrityError):
        await storage.database.run(
            lambda connection: connection.execute("DELETE FROM messaging_nodes")
        )
    assert (await storage.load_conversation_snapshot()).trees == {
        original.identity: original
    }


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("dictionary", [False, True])
async def test_historical_formats_and_message_log_preserve_resume_and_order(
    storage, tmp_path, wrapped, dictionary
):
    original = tree()
    legacy = original.to_json()
    del legacy["scope"]
    legacy["nodes"]["root"]["incoming"] = {"platform": "telegram", "chat_id": "chat"}
    trees = {"old": legacy} if dictionary else [legacy]
    conversation = {"trees": trees}
    payload = {"conversation": conversation} if wrapped else conversation
    payload["message_log"] = {
        "telegram:chat": [
            {"message_id": value, "direction": "out", "kind": "notice"}
            for value in (1, 2, 2, 3)
        ]
    }
    source = tmp_path / "sessions.json"
    source.write_text(json.dumps(payload))
    assert await import_legacy(storage.database, source) is None
    assert not source.exists()
    loaded = (await storage.load_conversation_snapshot()).trees[original.identity]
    assert loaded.nodes["root"]["session_id"] == "native-session"
    await storage.trim()
    assert await storage.get_tracked_message_ids_for_chat("telegram", "chat") == [
        "2",
        "3",
    ]


async def test_cleanup_failure_cannot_resurrect_cleared_history(storage, tmp_path):
    source = tmp_path / "sessions.json"
    original = tree()
    source.write_text(json.dumps(pending_to_legacy(original)))
    with patch.object(type(source), "unlink", side_effect=PermissionError("busy")):
        assert await import_legacy(storage.database, source)
    assert source.exists()
    await storage.commit_trees((), clear_scope=original.scope)
    assert await import_legacy(storage.database, source) is None
    assert not source.exists()
    assert (await storage.load_conversation_snapshot()).is_empty


async def test_invalid_graph_and_invalid_log_entry_do_not_discard_good_records(
    storage, tmp_path
):
    original, invalid = tree(), tree(root="bad")
    invalid.nodes["bad"]["parent_id"] = "missing"
    source = tmp_path / "sessions.json"
    source.write_text(
        json.dumps(
            {
                "trees": [original.to_json(), invalid.to_json()],
                "managed_messages": {
                    "discord:other": [
                        {"message_id": "good", "direction": "in", "kind": "prompt"},
                        {"message_id": "bad"},
                    ]
                },
            }
        )
    )
    assert await import_legacy(storage.database, source)
    assert source.exists()
    assert list((await storage.load_conversation_snapshot()).trees) == [
        original.identity
    ]
    assert await storage.get_tracked_message_ids_for_chat("discord", "other") == [
        "good"
    ]


async def test_failed_import_transaction_keeps_source_and_allows_retry(
    storage, tmp_path
):
    source = tmp_path / "sessions.json"
    source.write_text(json.dumps(pending_to_legacy(tree())))
    with (
        patch(
            "free_claude_code.runtime.messaging_import.write_tree",
            side_effect=sqlite3.OperationalError("disk failure"),
        ),
        pytest.raises(sqlite3.OperationalError),
    ):
        await import_legacy(storage.database, source)
    assert source.exists()
    assert (await storage.load_conversation_snapshot()).is_empty
    assert await import_legacy(storage.database, source) is None
    assert not source.exists()


def incoming(node, *, text="prompt"):
    return IncomingMessage(
        platform="telegram", chat_id="chat", message_id=node, user_id="user", text=text
    )


async def test_failed_admission_never_launches_harness_or_publishes_memory(storage):
    processor = AsyncMock()
    manager = TreeQueueManager(processor, store=storage)
    with (
        patch.object(
            storage, "commit_trees", side_effect=sqlite3.OperationalError("disk full")
        ),
        pytest.raises(sqlite3.OperationalError),
    ):
        await manager.admit(incoming("new"), "status-new")
    assert manager.get_tree_count() == 0
    assert manager.task_count() == 0
    processor.assert_not_awaited()
    assert (await storage.load_conversation_snapshot()).is_empty


async def test_failed_clear_restores_runtime_prompts_queue_and_claim(storage):
    started, release = asyncio.Event(), asyncio.Event()
    prompts = []

    async def process(claim):
        prompts.append(claim.prompt)
        started.set()
        await release.wait()
        await manager.complete_claim(claim, "resumable")

    manager = TreeQueueManager(process, store=storage)
    root = await manager.admit(incoming("root", text="first"), "status-root")
    assert root.claim is not None
    await started.wait()
    await manager.admit(
        incoming("child", text="second"),
        "status-child",
        parent_reference_id="status-root",
    )
    try:
        with (
            patch.object(
                storage,
                "commit_trees",
                side_effect=sqlite3.OperationalError("disk full"),
            ),
            pytest.raises(sqlite3.OperationalError),
        ):
            await manager.clear_scope(root.claim.identity.scope)
        assert manager.task_count() == 1
        assert (
            len(
                (await storage.load_conversation_snapshot())
                .trees[root.claim.identity]
                .nodes
            )
            == 2
        )
    finally:
        release.set()
        await asyncio.wait_for(manager.wait_idle(), 5)
    assert prompts == ["first", "second"]
    child = await manager.get_node(root.claim.identity.scope, "child")
    assert child is not None and child.session_id == "resumable"


async def test_status_subtree_delete_preserves_prompt_sibling_branch(storage):
    original = tree()
    for node, reference in (("status-reply", "status-root"), ("prompt-reply", "root")):
        original.nodes[node] = {
            "node_id": node,
            "status_message_id": f"status-{node}",
            "state": "completed",
            "parent_id": "root",
            "parent_reference_id": reference,
            "session_id": node,
        }
    await storage.commit_trees((original,))
    manager = TreeQueueManager.from_snapshot(
        await storage.load_conversation_snapshot(), AsyncMock(), store=storage
    )
    await manager.remove_message_subtree(original.scope, "status-root")
    persisted = (await storage.load_conversation_snapshot()).trees[original.identity]
    assert set(persisted.nodes) == {"root", "prompt-reply"}
    assert persisted.nodes["root"]["status_message_id"] is None
    assert persisted.nodes["root"]["session_id"] is None
    assert persisted.nodes["prompt-reply"]["parent_reference_id"] == "root"


async def test_failure_after_scope_delete_rolls_back_both_trees_and_message_log(
    storage,
):
    original = tree()
    await storage.commit_trees((original,))
    await storage.record_message_id("telegram", "chat", "tracked", "out", "notice")
    invalid = tree(root="invalid")
    invalid.nodes["invalid"]["state"] = "not-a-state"
    with pytest.raises(ValueError):
        await storage.commit_trees((invalid,), clear_scope=original.scope)
    assert (await storage.load_conversation_snapshot()).trees == {
        original.identity: original
    }
    assert await storage.get_tracked_message_ids_for_chat("telegram", "chat") == [
        "tracked"
    ]


async def test_code_close_does_not_release_shared_database(storage, tmp_path):
    code = SQLiteCodeStore(storage.database)
    await code.start()
    await code.close()
    other = SQLiteDatabase(tmp_path / "fcc.db", tmp_path / "code.lock")
    try:
        with pytest.raises(sqlite3.OperationalError, match="another FCC"):
            await other.start()
        await storage.commit_trees((tree(),))
        assert not (await storage.load_conversation_snapshot()).is_empty
    finally:
        await other.close()


async def test_cancellation_during_commit_preserves_memory_and_database_agreement(
    storage,
):
    entered, release = threading.Event(), threading.Event()
    execute = storage.database.execute

    def blocked(operation, *, write=True):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test commit barrier")
        return execute(operation, write=write)

    manager = TreeQueueManager(AsyncMock(), store=storage)
    with patch.object(storage.database, "execute", blocked):
        admission = asyncio.create_task(manager.admit(incoming("new"), "status-new"))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            admission.cancel()
        finally:
            release.set()
        decision = await admission
    assert decision.accepted
    await asyncio.wait_for(manager.wait_idle(), 5)
    assert (await manager.snapshot()).trees == (
        await storage.load_conversation_snapshot()
    ).trees


async def test_restore_does_not_consume_inactive_platform_status_repairs(storage):
    telegram, discord = tree(), tree("discord")
    telegram.nodes["root"]["state"] = "in_progress"
    discord.nodes["root"]["state"] = "pending"
    await storage.commit_trees((telegram, discord))
    workflow = MessagingWorkflow(
        AsyncMock(), AsyncMock(), storage, platform_name="telegram"
    )
    await workflow.restore()
    await workflow.close()
    loaded = await storage.load_conversation_snapshot()
    assert loaded.trees[telegram.identity].nodes["root"]["state"] == "error"
    assert loaded.trees[discord.identity].nodes["root"]["state"] == "pending"
    next_workflow = MessagingWorkflow(
        AsyncMock(), AsyncMock(), storage, platform_name="discord"
    )
    await next_workflow.restore()
    assert any(
        target.scope.platform == "discord"
        for target in next_workflow.tree_queue.restored_stale_targets
    )

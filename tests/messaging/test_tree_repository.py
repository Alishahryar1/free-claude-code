"""Persistence and restored-manager contracts for messaging trees."""

import asyncio

import pytest

from free_claude_code.messaging.models import IncomingMessage, MessageScope
from free_claude_code.messaging.trees import (
    ConversationSnapshot,
    MessageState,
    NodeClaim,
    TreeIdentity,
    TreeQueueManager,
    TreeSnapshot,
)
from free_claude_code.messaging.trees.node import MessageNode
from free_claude_code.messaging.trees.snapshot import node_to_snapshot

TELEGRAM_CHAT = MessageScope(platform="telegram", chat_id="chat")
ROOT_IDENTITY = TreeIdentity(scope=TELEGRAM_CHAT, root_id="root")


def _incoming(node_id: str, *, reply_to: str | None = None) -> IncomingMessage:
    return IncomingMessage(
        text=f"prompt {node_id}",
        chat_id="chat",
        user_id="user",
        message_id=node_id,
        platform="telegram",
        reply_to_message_id=reply_to,
    )


async def _wait_for_no_tasks(manager: TreeQueueManager) -> None:
    loop = asyncio.get_running_loop()
    for _ in range(20):
        if manager.task_count() == 0:
            return
        checkpoint = asyncio.Event()
        loop.call_soon(checkpoint.set)
        await checkpoint.wait()
    assert manager.task_count() == 0


def _interrupted_conversation() -> ConversationSnapshot:
    root = MessageNode(
        node_id="root",
        scope=TELEGRAM_CHAT,
        prompt="prompt root",
        status_message_id="status-root",
        state=MessageState.COMPLETED,
        session_id="session-root",
        children_ids=["pending", "failed"],
    )
    pending = MessageNode(
        node_id="pending",
        scope=TELEGRAM_CHAT,
        prompt="prompt pending",
        status_message_id="status-pending",
        parent_id="root",
        state=MessageState.PENDING,
        children_ids=["running"],
    )
    running = MessageNode(
        node_id="running",
        scope=TELEGRAM_CHAT,
        prompt="prompt running",
        status_message_id="status-running",
        parent_id="pending",
        state=MessageState.IN_PROGRESS,
    )
    failed = MessageNode(
        node_id="failed",
        scope=TELEGRAM_CHAT,
        prompt="prompt failed",
        status_message_id="status-failed",
        parent_id="root",
        state=MessageState.ERROR,
    )
    snapshot = TreeSnapshot(
        scope=TELEGRAM_CHAT,
        root_id="root",
        nodes={
            node.node_id: node_to_snapshot(node)
            for node in (root, pending, running, failed)
        },
    )
    return ConversationSnapshot(trees={snapshot.identity: snapshot})


@pytest.mark.asyncio
async def test_restore_reconciles_interrupted_nodes_before_manager_exposure() -> None:
    processed: list[str] = []

    async def process(claim: NodeClaim) -> None:
        processed.append(claim.node.node_id)

    manager = TreeQueueManager.from_snapshot(_interrupted_conversation(), process)

    assert len(manager.restored_stale_targets) == 2
    assert manager.restored_snapshot is not None
    assert manager.restored_snapshot == await manager.snapshot()
    assert manager.task_count() == 0
    assert processed == []

    root = await manager.get_node(TELEGRAM_CHAT, "root")
    pending = await manager.get_node(TELEGRAM_CHAT, "pending")
    running = await manager.get_node(TELEGRAM_CHAT, "status-running")
    failed = await manager.get_node(TELEGRAM_CHAT, "failed")
    assert root is not None and root.state is MessageState.COMPLETED
    assert root.session_id == "session-root"
    assert pending is not None and pending.state is MessageState.ERROR
    assert running is not None and running.state is MessageState.ERROR
    assert failed is not None and failed.state is MessageState.ERROR
    assert await manager.resolve_node_id(TELEGRAM_CHAT, "status-pending") == "pending"
    assert {
        (target.scope, target.node_id) for target in manager.restored_stale_targets
    } == {
        (TELEGRAM_CHAT, "pending"),
        (TELEGRAM_CHAT, "running"),
    }

    normalized = await manager.snapshot()
    normalized_tree = normalized.get_tree(ROOT_IDENTITY)
    assert normalized_tree is not None
    assert normalized_tree.nodes["pending"]["state"] == "error"
    assert normalized_tree.nodes["running"]["state"] == "error"
    assert normalized_tree.nodes["failed"]["state"] == "error"
    assert all("error_message" not in node for node in normalized_tree.nodes.values())


@pytest.mark.parametrize("corruption", ["cross_scope", "duplicate_reference"])
def test_restore_rejects_tree_that_violates_scoped_reference_invariants(
    corruption: str,
) -> None:
    snapshot = _interrupted_conversation()
    tree = snapshot.get_tree(ROOT_IDENTITY)
    assert tree is not None
    if corruption == "cross_scope":
        tree.nodes["pending"]["incoming"] = {
            "platform": "telegram",
            "chat_id": "other-chat",
        }
    else:
        tree.nodes["failed"]["status_message_id"] = "status-pending"

    manager = TreeQueueManager.from_snapshot(snapshot, lambda _claim: asyncio.sleep(0))

    assert manager.get_tree_count() == 0
    assert manager.restored_snapshot == ConversationSnapshot()


@pytest.mark.asyncio
async def test_claim_state_and_session_updates_produce_detached_snapshots() -> None:
    release = asyncio.Event()
    started = asyncio.Event()

    async def process(_claim: NodeClaim) -> None:
        started.set()
        await release.wait()

    manager = TreeQueueManager(process)
    decision = await manager.admit(_incoming("root"), "status-root")
    assert decision.claim is not None
    await started.wait()

    session_snapshot = await manager.record_session(decision.claim, "session-root")
    completed_snapshot = await manager.complete_claim(
        decision.claim,
        "session-root",
    )

    assert session_snapshot is not None
    assert session_snapshot.nodes["root"]["state"] == "in_progress"
    assert session_snapshot.nodes["root"]["session_id"] == "session-root"
    assert completed_snapshot is not None
    assert completed_snapshot.nodes["root"]["state"] == "completed"
    assert completed_snapshot.nodes["root"]["session_id"] == "session-root"

    assert decision.snapshot is not None
    decision.snapshot.nodes["root"]["state"] = "mutated-copy"
    view = await manager.get_node(TELEGRAM_CHAT, "status-root")
    assert view is not None and view.state is MessageState.COMPLETED
    assert view.session_id == "session-root"
    assert await manager.get_message_ids_for_chat("telegram", "chat") == {
        "root",
        "status-root",
    }

    release.set()
    await _wait_for_no_tasks(manager)
    assert await manager.snapshot() == ConversationSnapshot(
        trees={ROOT_IDENTITY: completed_snapshot}
    )

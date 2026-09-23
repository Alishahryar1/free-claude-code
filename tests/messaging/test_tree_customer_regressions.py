"""Customer-visible regressions at the tree manager ownership boundary."""

import asyncio

import pytest

from free_claude_code.messaging.models import IncomingMessage, MessageScope
from free_claude_code.messaging.trees import (
    ConversationSnapshot,
    MessageState,
    NodeClaim,
    TreeIdentity,
    TreeQueueManager,
)


def _incoming(chat_id: str) -> IncomingMessage:
    return IncomingMessage(
        text=f"prompt from {chat_id}",
        chat_id=chat_id,
        user_id=f"user-{chat_id}",
        message_id="42",
        platform="telegram",
    )


async def _wait_for_no_tasks(manager: TreeQueueManager) -> None:
    for _ in range(100):
        if manager.task_count() == 0:
            return
        await asyncio.sleep(0)
    raise AssertionError("messaging claims did not finish")


def _snapshot_chat_ids(snapshot: ConversationSnapshot) -> set[str]:
    return {tree.scope.chat_id for tree in snapshot.trees.values()}


@pytest.mark.asyncio
async def test_same_telegram_ids_in_different_chats_remain_independent_after_restore() -> (
    None
):
    async def process(_claim: NodeClaim) -> None:
        return

    manager = TreeQueueManager(process)

    first = await manager.admit(_incoming("chat-a"), "99")
    second = await manager.admit(_incoming("chat-b"), "99")
    await _wait_for_no_tasks(manager)

    assert first.accepted is True
    assert second.accepted is True
    assert manager.get_tree_count() == 2

    snapshot = await manager.snapshot()
    assert len(snapshot.trees) == 2
    assert _snapshot_chat_ids(snapshot) == {"chat-a", "chat-b"}
    chat_a = MessageScope(platform="telegram", chat_id="chat-a")
    chat_b = MessageScope(platform="telegram", chat_id="chat-b")
    assert snapshot.get_tree(TreeIdentity(scope=chat_a, root_id="42")) is not None
    assert snapshot.get_tree(TreeIdentity(scope=chat_b, root_id="42")) is not None

    restored = TreeQueueManager.from_snapshot(snapshot, process)

    assert restored.get_tree_count() == 2
    assert _snapshot_chat_ids(await restored.snapshot()) == {"chat-a", "chat-b"}
    assert await restored.get_message_ids_for_chat("telegram", "chat-a") == {"42", "99"}
    assert await restored.get_message_ids_for_chat("telegram", "chat-b") == {"42", "99"}
    assert await restored.get_node(chat_a, "42") is not None
    assert await restored.get_node(chat_b, "42") is not None


@pytest.mark.asyncio
async def test_successful_completion_overrides_recoverable_failure_and_records_session() -> (
    None
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def process(_claim: NodeClaim) -> None:
        started.set()
        await release.wait()

    manager = TreeQueueManager(process)
    decision = await manager.admit(_incoming("chat"), "99")
    assert decision.claim is not None
    await started.wait()

    try:
        failure = await manager.fail_claim(
            decision.claim,
            propagate=True,
        )
        assert failure.snapshot is not None
        failed_node = next(iter(failure.snapshot.nodes.values()))
        assert failed_node["state"] == MessageState.ERROR.value

        completed = await manager.complete_claim(decision.claim, "session-42")

        assert completed is not None
        completed_node = next(iter(completed.nodes.values()))
        assert completed_node["state"] == MessageState.COMPLETED.value
        assert completed_node["session_id"] == "session-42"

        persisted = await manager.snapshot()
        persisted_node = next(iter(persisted.trees.values())).nodes.values()
        persisted_node = next(iter(persisted_node))
        assert persisted_node["state"] == MessageState.COMPLETED.value
        assert persisted_node["session_id"] == "session-42"
    finally:
        release.set()
        await _wait_for_no_tasks(manager)

"""Deterministic manager concurrency contracts."""

import asyncio

import pytest

from free_claude_code.messaging.models import IncomingMessage, MessageScope
from free_claude_code.messaging.trees import (
    CancellationReason,
    CancellationUiOwner,
    NodeClaim,
    QueueEntry,
    TreeQueueManager,
)

_SCOPE = MessageScope(platform="telegram", chat_id="chat")


def _incoming(node_id: str, *, reply_to: str | None = None) -> IncomingMessage:
    return IncomingMessage(
        text=f"prompt {node_id}",
        chat_id=_SCOPE.chat_id,
        user_id="user",
        message_id=node_id,
        platform=_SCOPE.platform,
        reply_to_message_id=reply_to,
    )


async def _wait_for_no_tasks(manager: TreeQueueManager) -> None:
    loop = asyncio.get_running_loop()
    for _ in range(30):
        if manager.task_count() == 0:
            return
        checkpoint = asyncio.Event()
        loop.call_soon(checkpoint.set)
        await checkpoint.wait()
    assert manager.task_count() == 0


@pytest.mark.asyncio
async def test_one_tree_processes_fifo_with_transition_owned_queue_updates() -> None:
    node_ids = ("root", "a", "b", "c")
    releases = {node_id: asyncio.Event() for node_id in node_ids}
    completions = {node_id: asyncio.Event() for node_id in node_ids}
    started: asyncio.Queue[str] = asyncio.Queue()
    started_callbacks: list[str] = []
    queue_updates: list[tuple[tuple[str, int], ...]] = []
    manager: TreeQueueManager

    async def process(claim: NodeClaim) -> None:
        node_id = claim.node.node_id
        started.put_nowait(node_id)
        await releases[node_id].wait()
        await manager.complete_claim(claim, f"session-{node_id}")
        completions[node_id].set()

    async def capture_started(claim: NodeClaim) -> None:
        started_callbacks.append(claim.node.node_id)

    async def capture_queue(queue: tuple[QueueEntry, ...]) -> None:
        queue_updates.append(
            tuple((entry.node.node_id, entry.position) for entry in queue)
        )

    manager = TreeQueueManager(
        process,
        queue_update_callback=capture_queue,
        node_started_callback=capture_started,
    )
    root = await manager.admit(_incoming("root"), "status-root")
    assert root.claim is not None
    identity = root.claim.identity
    assert await started.get() == "root"

    decisions = [
        await manager.admit(
            _incoming(node_id, reply_to="root"),
            f"status-{node_id}",
            parent_reference_id="root",
        )
        for node_id in node_ids[1:]
    ]
    assert [decision.position for decision in decisions] == [1, 2, 3]

    observed = ["root"]
    for node_id in node_ids:
        releases[node_id].set()
        await completions[node_id].wait()
        if node_id != node_ids[-1]:
            observed.append(await started.get())

    await _wait_for_no_tasks(manager)

    assert observed == list(node_ids)
    assert started_callbacks == ["a", "b", "c"]
    assert queue_updates == [
        (("b", 1), ("c", 2)),
        (("c", 1),),
        (),
    ]
    snapshot = await manager.snapshot()
    assert {
        node_id: snapshot.trees[identity].nodes[node_id]["state"]
        for node_id in node_ids
    } == dict.fromkeys(node_ids, "completed")


@pytest.mark.asyncio
async def test_separate_trees_process_in_parallel() -> None:
    started = {node_id: asyncio.Event() for node_id in ("one", "two")}
    releases = {node_id: asyncio.Event() for node_id in ("one", "two")}
    completed = {node_id: asyncio.Event() for node_id in ("one", "two")}
    all_started = asyncio.Event()
    active = 0
    maximum_active = 0
    manager: TreeQueueManager

    async def process(claim: NodeClaim) -> None:
        nonlocal active, maximum_active
        node_id = claim.node.node_id
        active += 1
        maximum_active = max(maximum_active, active)
        started[node_id].set()
        if all(event.is_set() for event in started.values()):
            all_started.set()
        try:
            await releases[node_id].wait()
            await manager.complete_claim(claim, f"session-{node_id}")
        finally:
            active -= 1
            completed[node_id].set()

    manager = TreeQueueManager(process)
    await asyncio.gather(
        manager.admit(_incoming("one"), "status-one"),
        manager.admit(_incoming("two"), "status-two"),
    )
    await all_started.wait()

    assert maximum_active == 2
    assert active == 2
    assert manager.get_tree_count() == 2

    releases["one"].set()
    releases["two"].set()
    await asyncio.gather(*(event.wait() for event in completed.values()))
    await _wait_for_no_tasks(manager)


@pytest.mark.asyncio
async def test_cancel_all_cancels_active_and_queued_work_across_trees() -> None:
    active_started = {node_id: asyncio.Event() for node_id in ("one", "two")}
    processed: list[str] = []

    async def process(claim: NodeClaim) -> None:
        node_id = claim.node.node_id
        processed.append(node_id)
        if node_id in active_started:
            active_started[node_id].set()
        await asyncio.Event().wait()

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("one"), "status-one")
    await manager.admit(_incoming("two"), "status-two")
    await asyncio.gather(*(event.wait() for event in active_started.values()))
    await manager.admit(
        _incoming("one-child", reply_to="one"),
        "status-one-child",
        parent_reference_id="one",
    )
    await manager.admit(
        _incoming("two-child", reply_to="two"),
        "status-two-child",
        parent_reference_id="two",
    )

    result = await manager.cancel_all(reason=CancellationReason.STOP)

    owners = {effect.node.node_id: effect.ui_owner for effect in result.effects}
    assert owners == {
        "one": CancellationUiOwner.RUNNER,
        "one-child": CancellationUiOwner.WORKFLOW,
        "two": CancellationUiOwner.RUNNER,
        "two-child": CancellationUiOwner.WORKFLOW,
    }
    assert len(result.snapshots) == 2
    assert {
        node["state"]
        for snapshot in result.snapshots
        for node in snapshot.nodes.values()
    } == {"error"}
    assert set(processed) == {"one", "two"}
    assert manager.task_count() == 0


@pytest.mark.asyncio
async def test_branch_removal_atomically_unindexes_subtree_and_preserves_sibling() -> (
    None
):
    root_started = asyncio.Event()
    release_root = asyncio.Event()
    sibling_started = asyncio.Event()
    release_sibling = asyncio.Event()
    unexpected: list[str] = []

    async def process(claim: NodeClaim) -> None:
        node_id = claim.node.node_id
        if node_id == "root":
            root_started.set()
            await release_root.wait()
        elif node_id == "sibling":
            sibling_started.set()
            await release_sibling.wait()
        else:
            unexpected.append(node_id)

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("root"), "status-root")
    await root_started.wait()
    await manager.admit(
        _incoming("branch", reply_to="root"),
        "status-branch",
        parent_reference_id="root",
    )
    await manager.admit(
        _incoming("leaf", reply_to="branch"),
        "status-leaf",
        parent_reference_id="branch",
    )
    await manager.admit(
        _incoming("sibling", reply_to="root"),
        "status-sibling",
        parent_reference_id="root",
    )

    result = await manager.remove_message_subtree(
        _SCOPE,
        "branch",
        reason=CancellationReason.STOP,
    )

    assert result.removed_tree_identity is None
    assert result.delete_message_ids == frozenset(
        {"branch", "status-branch", "leaf", "status-leaf"}
    )
    assert {
        effect.node.node_id: effect.ui_owner for effect in result.cancellation.effects
    } == {
        "branch": CancellationUiOwner.WORKFLOW,
        "leaf": CancellationUiOwner.WORKFLOW,
    }
    assert len(result.cancellation.snapshots) == 1
    assert set(result.cancellation.snapshots[0].nodes) == {"root", "sibling"}
    assert await manager.resolve_node_id(_SCOPE, "branch") is None
    assert await manager.resolve_node_id(_SCOPE, "status-leaf") is None
    assert await manager.resolve_node_id(_SCOPE, "sibling") == "sibling"

    release_root.set()
    await asyncio.wait_for(sibling_started.wait(), timeout=1)
    assert unexpected == []
    release_sibling.set()
    await _wait_for_no_tasks(manager)


@pytest.mark.asyncio
async def test_root_removal_atomically_cancels_and_unindexes_entire_tree() -> None:
    root_started = asyncio.Event()
    processed: list[str] = []

    async def process(claim: NodeClaim) -> None:
        processed.append(claim.node.node_id)
        root_started.set()
        await asyncio.Event().wait()

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("root"), "status-root")
    await root_started.wait()
    await manager.admit(
        _incoming("child", reply_to="root"),
        "status-child",
        parent_reference_id="root",
    )

    result = await manager.remove_message_subtree(
        _SCOPE,
        "root",
        reason=CancellationReason.STOP,
    )

    assert result.removed_tree_identity is not None
    assert result.removed_tree_identity.scope == _SCOPE
    assert result.removed_tree_identity.root_id == "root"
    assert result.delete_message_ids == frozenset(
        {"root", "status-root", "child", "status-child"}
    )
    assert {
        effect.node.node_id: effect.ui_owner for effect in result.cancellation.effects
    } == {
        "root": CancellationUiOwner.RUNNER,
        "child": CancellationUiOwner.WORKFLOW,
    }
    assert result.cancellation.snapshots == ()
    assert manager.get_tree_count() == 0
    assert manager.task_count() == 0
    assert await manager.resolve_node_id(_SCOPE, "root") is None
    assert await manager.resolve_node_id(_SCOPE, "status-child") is None
    assert processed == ["root"]

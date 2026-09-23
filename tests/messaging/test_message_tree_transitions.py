import asyncio
from dataclasses import FrozenInstanceError

import pytest

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.trees.node import MessageNode, MessageState
from free_claude_code.messaging.trees.runtime import MessageTree

_SCOPE = MessageScope(platform="telegram", chat_id="chat")


def _tree() -> MessageTree:
    return MessageTree(
        MessageNode(
            node_id="root",
            scope=_SCOPE,
            prompt="prompt root",
            status_message_id="status-root",
        )
    )


async def _add(
    tree: MessageTree,
    node_id: str,
    status_message_id: str,
    parent_id: str = "root",
    parent_reference_id: str | None = None,
):
    return await tree.add_and_enqueue(
        node_id,
        _SCOPE,
        f"prompt {node_id}",
        status_message_id,
        parent_id,
        parent_reference_id or parent_id,
    )


@pytest.mark.asyncio
async def test_enqueue_finish_and_claim_next_are_atomic_and_fifo() -> None:
    tree = _tree()

    root = await tree.enqueue_or_claim("root")
    child_a = await _add(tree, "child-a", "status-a")
    child_b = await _add(tree, "child-b", "status-b")

    assert root.accepted and root.claim is not None and root.position is None
    assert child_a.claim is None and child_a.position == 1
    assert child_b.claim is None and child_b.position == 2

    completion = await tree.finish_and_claim_next(root.claim.claim_id)

    assert completion.next_claim is not None
    assert completion.next_claim.node.node_id == "child-a"
    assert [(entry.node.node_id, entry.position) for entry in completion.queue] == [
        ("child-b", 1)
    ]


@pytest.mark.asyncio
async def test_duplicate_unknown_and_terminal_admission_are_rejected_without_wedge() -> (
    None
):
    tree = _tree()

    first = await tree.enqueue_or_claim("root")
    duplicate = await tree.enqueue_or_claim("root")
    unknown = await tree.enqueue_or_claim("status-root")
    assert first.claim is not None
    assert duplicate.accepted is False
    assert unknown.accepted is False
    assert duplicate.snapshot is None
    assert unknown.snapshot is None

    await tree.complete_claim(first.claim.claim_id, "session")
    await tree.finish_and_claim_next(first.claim.claim_id)
    terminal = await tree.enqueue_or_claim("root")
    assert terminal.accepted is False
    assert terminal.snapshot is None


@pytest.mark.asyncio
async def test_stale_claim_id_cannot_clear_a_newer_claim() -> None:
    tree = _tree()
    first = await tree.enqueue_or_claim("root")
    assert first.claim is not None
    await tree.finish_and_claim_next(first.claim.claim_id)

    second = await _add(tree, "child", "status-child")
    assert second.claim is not None

    stale = await tree.finish_and_claim_next(first.claim.claim_id)
    assert stale.next_claim is None
    assert await tree.record_session(second.claim.claim_id, "session") is not None
    await tree.finish_and_claim_next(second.claim.claim_id)
    third = await _add(tree, "third", "status-third")
    assert third.claim is not None


@pytest.mark.asyncio
async def test_cancellation_keeps_active_identity_until_matching_finish() -> None:
    tree = _tree()
    root = await tree.enqueue_or_claim("root")
    queued = await _add(tree, "queued", "status-queued")
    assert root.claim is not None and queued.position == 1

    cancelled = await tree.cancel_node("root")

    assert cancelled.active_claim == root.claim
    assert [node.node_id for node in cancelled.nodes] == ["root"]

    completion = await tree.finish_and_claim_next(root.claim.claim_id)
    assert completion.next_claim is not None
    assert completion.next_claim.node.node_id == "queued"


@pytest.mark.asyncio
@pytest.mark.parametrize("propagate", [False, True])
async def test_cancelled_claim_rejects_late_failure_and_descendant_propagation(
    propagate: bool,
) -> None:
    tree = _tree()
    root = await tree.enqueue_or_claim("root")
    queued = await _add(tree, "queued", "status-queued")
    assert root.claim is not None and queued.position == 1

    await tree.cancel_node("root")
    late_failure = await tree.fail_claim(
        root.claim.claim_id,
        propagate=propagate,
    )

    assert late_failure.snapshot is None
    assert late_failure.affected == ()
    assert late_failure.queue_update is None
    queued_view = await tree.node_view("queued")
    assert queued_view is not None
    assert queued_view.state is MessageState.PENDING
    completion = await tree.finish_and_claim_next(root.claim.claim_id)
    assert completion.next_claim is not None
    assert completion.next_claim.node.node_id == "queued"


@pytest.mark.asyncio
async def test_cancelled_queue_member_is_never_claimed() -> None:
    tree = _tree()
    root = await tree.enqueue_or_claim("root")
    await _add(tree, "a", "status-a")
    await _add(tree, "b", "status-b")
    assert root.claim is not None

    cancelled = await tree.cancel_node("a")
    assert [node.node_id for node in cancelled.nodes] == ["a"]

    completion = await tree.finish_and_claim_next(root.claim.claim_id)
    assert completion.next_claim is not None
    assert completion.next_claim.node.node_id == "b"


@pytest.mark.asyncio
async def test_prompt_subtree_removal_returns_every_platform_message() -> None:
    tree = _tree()
    await _add(tree, "child", "status-child")

    removed = await tree.remove_message_subtree("root")

    assert removed.removed_message_ids == frozenset(
        {"root", "status-root", "child", "status-child"}
    )


@pytest.mark.asyncio
async def test_status_subtree_preserves_prompt_siblings_and_invalidates_resume() -> (
    None
):
    tree = _tree()
    root = await tree.enqueue_or_claim("root")
    assert root.claim is not None
    await tree.record_session(root.claim.claim_id, "session-root")
    direct_sibling = await _add(tree, "sibling", "status-sibling")
    status_descendant = await _add(
        tree,
        "descendant",
        "status-descendant",
        parent_reference_id="status-root",
    )
    assert direct_sibling.position == 1
    assert status_descendant.position == 2

    removed = await tree.remove_message_subtree("status-root")

    assert removed.removed_message_ids == frozenset(
        {"status-root", "descendant", "status-descendant"}
    )
    root_view = await tree.node_view("root")
    sibling_view = await tree.node_view("sibling")
    assert root_view is not None
    assert root_view.state is MessageState.ERROR
    assert root_view.session_id is None
    assert sibling_view is not None
    assert sibling_view.state is MessageState.PENDING
    assert await tree.node_view("descendant") is None
    assert await tree.resolve_reply("status-root") is None

    completion = await tree.finish_and_claim_next(root.claim.claim_id)
    assert completion.next_claim is not None
    assert completion.next_claim.node.node_id == "sibling"
    assert completion.next_claim.parent_session_id is None


@pytest.mark.asyncio
async def test_concurrent_duplicate_enqueue_produces_one_claim_only() -> None:
    tree = _tree()
    gate = asyncio.Event()

    async def enqueue():
        await gate.wait()
        return await tree.enqueue_or_claim("root")

    tasks = [asyncio.create_task(enqueue()) for _ in range(20)]
    gate.set()
    results = await asyncio.gather(*tasks)

    assert sum(result.claim is not None for result in results) == 1
    assert sum(result.accepted for result in results) == 1


@pytest.mark.asyncio
async def test_results_are_frozen_and_do_not_expose_mutable_nodes() -> None:
    tree = _tree()
    decision = await tree.enqueue_or_claim("root")
    assert decision.claim is not None

    with pytest.raises(FrozenInstanceError):
        decision.claim.__setattr__("claim_id", "replacement")
    assert not hasattr(decision.claim, "__dict__")
    assert not isinstance(decision.claim.node, MessageNode)


def test_message_tree_has_no_lock_or_partial_mutation_escape_hatches() -> None:
    forbidden = {
        "with_lock",
        "enqueue",
        "dequeue",
        "put_queue_unlocked",
        "remove_from_queue",
        "set_processing_state",
        "clear_current_node",
        "set_current_task",
        "cancel_current_task",
        "set_node_error_sync",
        "drain_queue_and_mark_cancelled",
        "reset_processing_state",
    }

    assert forbidden.isdisjoint(vars(MessageTree))

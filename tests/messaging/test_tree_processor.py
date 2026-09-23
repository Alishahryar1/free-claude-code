"""Manager-level task and cancellation ownership tests."""

import asyncio
import logging

import pytest

from free_claude_code.messaging.models import IncomingMessage, MessageScope
from free_claude_code.messaging.trees import (
    CancellationReason,
    CancellationUiOwner,
    FailureResult,
    MessageState,
    NodeClaim,
    QueueEntry,
    TreeQueueManager,
)
from free_claude_code.messaging.trees import manager as manager_module
from free_claude_code.messaging.trees import processor as processor_module
from free_claude_code.messaging.trees.node import MessageNode
from free_claude_code.messaging.trees.processor import TreeQueueProcessor
from free_claude_code.messaging.trees.runtime import MessageTree

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
    """Yield deterministic ready-queue checkpoints until task cleanup completes."""
    loop = asyncio.get_running_loop()
    for _ in range(20):
        if manager.task_count() == 0:
            return
        checkpoint = asyncio.Event()
        loop.call_soon(checkpoint.set)
        await checkpoint.wait()
    assert manager.task_count() == 0


@pytest.mark.asyncio
async def test_active_cancel_returns_runner_owned_effect_and_terminal_snapshot() -> (
    None
):
    started = asyncio.Event()

    async def process(_claim: NodeClaim) -> None:
        started.set()
        await asyncio.Event().wait()

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("root"), "status-root")
    await started.wait()

    result = await manager.cancel_node(
        _SCOPE,
        "root",
        reason=CancellationReason.STOP,
    )

    assert [(effect.node.node_id, effect.ui_owner) for effect in result.effects] == [
        ("root", CancellationUiOwner.RUNNER)
    ]
    assert len(result.snapshots) == 1
    assert result.snapshots[0].nodes["root"]["state"] == "error"
    view = await manager.get_node(_SCOPE, "root")
    assert view is not None and view.state is MessageState.ERROR
    assert manager.task_count() == 0


@pytest.mark.asyncio
async def test_queued_cancel_returns_workflow_effect_and_exact_queue_update() -> None:
    release_root = asyncio.Event()
    root_started = asyncio.Event()
    child_started = asyncio.Event()
    queue_updates: list[tuple[tuple[str, int], ...]] = []

    async def process(claim: NodeClaim) -> None:
        if claim.node.node_id == "root":
            root_started.set()
            await release_root.wait()
        else:
            child_started.set()

    async def capture_queue(queue: tuple[QueueEntry, ...]) -> None:
        queue_updates.append(
            tuple((entry.node.node_id, entry.position) for entry in queue)
        )

    manager = TreeQueueManager(process, queue_update_callback=capture_queue)
    await manager.admit(_incoming("root"), "status-root")
    await root_started.wait()
    decision = await manager.admit(
        _incoming("child", reply_to="root"),
        "status-child",
        parent_reference_id="root",
    )
    assert decision.position == 1

    result = await manager.cancel_node(
        _SCOPE,
        "child",
        reason=CancellationReason.STOP,
    )

    assert [(effect.node.node_id, effect.ui_owner) for effect in result.effects] == [
        ("child", CancellationUiOwner.WORKFLOW)
    ]
    assert queue_updates == [()]
    assert result.snapshots[0].nodes["child"]["state"] == "error"
    assert child_started.is_set() is False

    release_root.set()
    await _wait_for_no_tasks(manager)
    assert child_started.is_set() is False


@pytest.mark.asyncio
async def test_cancel_cleanup_timeout_is_bounded_and_task_remains_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "CANCEL_TASK_DRAIN_TIMEOUT_S", 0.01)
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def process(_claim: NodeClaim) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_cleanup.wait()
            raise

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("root"), "status-root")
    await started.wait()

    try:
        result = await asyncio.wait_for(
            manager.cancel_node(
                _SCOPE,
                "root",
                reason=CancellationReason.STOP,
            ),
            timeout=0.5,
        )
        await cancellation_seen.wait()
        assert result.effects[0].node.node_id == "root"
        assert manager.task_count() == 1
    finally:
        release_cleanup.set()

    await _wait_for_no_tasks(manager)


@pytest.mark.asyncio
async def test_escaped_processor_failure_persists_effects_through_manager_owner() -> (
    None
):
    started = asyncio.Event()
    release = asyncio.Event()
    failures: list[FailureResult] = []
    queue_updates: list[tuple[QueueEntry, ...]] = []

    async def process(claim: NodeClaim) -> None:
        if claim.node.node_id == "root":
            started.set()
            await release.wait()
            raise RuntimeError("processor boundary failed")

    async def capture_queue(queue: tuple[QueueEntry, ...]) -> None:
        queue_updates.append(queue)

    manager = TreeQueueManager(
        process,
        queue_update_callback=capture_queue,
        unexpected_failure_callback=failures.append,
    )
    await manager.admit(_incoming("root"), "status-root")
    await started.wait()
    await manager.admit(
        _incoming("child", reply_to="root"),
        "status-child",
        parent_reference_id="root",
    )

    release.set()
    await _wait_for_no_tasks(manager)

    assert len(failures) == 1
    failure = failures[0]
    assert failure.snapshot is not None
    assert {target.node_id for target in failure.affected} == {"root", "child"}
    assert failure.snapshot.nodes["root"]["state"] == "error"
    assert failure.snapshot.nodes["child"]["state"] == "error"
    assert queue_updates == [()]


@pytest.mark.asyncio
async def test_wait_idle_spans_successor_publication_and_completion() -> None:
    root_started = asyncio.Event()
    release_root = asyncio.Event()
    child_started = asyncio.Event()
    release_child = asyncio.Event()

    async def process(claim: NodeClaim) -> None:
        if claim.node.node_id == "root":
            root_started.set()
            await release_root.wait()
            return
        child_started.set()
        await release_child.wait()

    manager = TreeQueueManager(process)
    await asyncio.wait_for(manager.wait_idle(), timeout=0.1)
    await manager.admit(_incoming("root"), "status-root")
    await asyncio.wait_for(root_started.wait(), timeout=1)
    await manager.admit(
        _incoming("child", reply_to="root"),
        "status-child",
        parent_reference_id="root",
    )
    idle_task = asyncio.create_task(manager.wait_idle())

    try:
        await asyncio.sleep(0)
        assert not idle_task.done()

        release_root.set()
        await asyncio.wait_for(child_started.wait(), timeout=1)
        assert not idle_task.done()

        release_child.set()
        await asyncio.wait_for(idle_task, timeout=1)
        assert manager.task_count() == 0
    finally:
        release_root.set()
        release_child.set()
        if not idle_task.done():
            idle_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await idle_task


@pytest.mark.asyncio
async def test_wait_idle_spans_pre_run_cancellation_recovery() -> None:
    finish_started = asyncio.Event()
    release_finish = asyncio.Event()

    async def process(_claim: NodeClaim) -> None:
        raise AssertionError("pre-run cancellation must not enter the processor")

    async def fail_claim(_claim: NodeClaim) -> None:
        raise AssertionError("cancellation must not fail the claim")

    async def finish_claim(_tree: MessageTree, _claim: NodeClaim) -> None:
        finish_started.set()
        await release_finish.wait()

    tree = MessageTree(
        MessageNode(
            node_id="root",
            scope=_SCOPE,
            prompt="prompt root",
            status_message_id="status-root",
        )
    )
    decision = await tree.enqueue_or_claim("root")
    assert decision.claim is not None
    processor = TreeQueueProcessor(
        process,
        claim_failure_callback=fail_claim,
        claim_finished_callback=finish_claim,
    )
    processor.launch(tree, decision.claim)
    cancelled = processor.cancel(decision.claim, CancellationReason.STOP)
    assert cancelled is not None
    assert cancelled.runner_started is False
    idle_task = asyncio.create_task(processor.wait_idle())

    try:
        await asyncio.wait_for(finish_started.wait(), timeout=1)
        assert not idle_task.done()

        release_finish.set()
        await asyncio.wait_for(cancelled.task, timeout=1)
        await asyncio.wait_for(idle_task, timeout=1)
        assert processor.task_count() == 0
    finally:
        release_finish.set()
        if not idle_task.done():
            idle_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await idle_task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("log_messaging_error_details", "secret_is_logged"),
    [(False, False), (True, True)],
)
async def test_wait_idle_surfaces_finish_failure_without_leaking_task_owner(
    caplog: pytest.LogCaptureFixture,
    log_messaging_error_details: bool,
    secret_is_logged: bool,
) -> None:
    secret = "unique-finish-callback-secret"
    finish_error = RuntimeError(secret)
    finish_calls = 0

    async def process(_claim: NodeClaim) -> None:
        return

    async def fail_claim(_claim: NodeClaim) -> None:
        raise AssertionError("successful processing must not fail the claim")

    async def finish_claim(_tree: MessageTree, _claim: NodeClaim) -> None:
        nonlocal finish_calls
        finish_calls += 1
        raise finish_error

    tree = MessageTree(
        MessageNode(
            node_id="root",
            scope=_SCOPE,
            prompt="prompt root",
            status_message_id="status-root",
        )
    )
    decision = await tree.enqueue_or_claim("root")
    assert decision.claim is not None
    processor = TreeQueueProcessor(
        process,
        claim_failure_callback=fail_claim,
        claim_finished_callback=finish_claim,
        log_messaging_error_details=log_messaging_error_details,
    )

    with caplog.at_level(logging.ERROR):
        processor.launch(tree, decision.claim)
        with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(processor.wait_idle(), timeout=1)

    assert raised.value is finish_error
    assert finish_calls == 1
    assert processor.task_count() == 0
    await asyncio.wait_for(processor.wait_idle(), timeout=0.1)

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Claim completion callback failed for node root" in messages
    assert "RuntimeError" in messages
    assert (secret in messages) is secret_is_logged


@pytest.mark.asyncio
async def test_failed_launch_rolls_idle_state_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def process(_claim: NodeClaim) -> None:
        return

    async def fail_claim(_claim: NodeClaim) -> None:
        return

    async def finish_claim(_tree: MessageTree, _claim: NodeClaim) -> None:
        return

    tree = MessageTree(
        MessageNode(
            node_id="root",
            scope=_SCOPE,
            prompt="prompt root",
            status_message_id="status-root",
        )
    )
    decision = await tree.enqueue_or_claim("root")
    assert decision.claim is not None
    processor = TreeQueueProcessor(
        process,
        claim_failure_callback=fail_claim,
        claim_finished_callback=finish_claim,
    )

    def fail_create_task(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("launch failed")

    monkeypatch.setattr(
        processor_module.asyncio,
        "create_task",
        fail_create_task,
    )

    with pytest.raises(RuntimeError, match="launch failed"):
        processor.launch(tree, decision.claim)

    assert processor.task_count() == 0
    await asyncio.wait_for(processor.wait_idle(), timeout=0.1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("log_messaging_error_details", "secret_is_logged"),
    [(False, False), (True, True)],
)
async def test_processor_failure_logging_respects_diagnostic_policy(
    caplog: pytest.LogCaptureFixture,
    log_messaging_error_details: bool,
    secret_is_logged: bool,
) -> None:
    secret = "unique-processor-exception-secret"

    async def process(_claim: NodeClaim) -> None:
        raise RuntimeError(secret)

    manager = TreeQueueManager(
        process,
        log_messaging_error_details=log_messaging_error_details,
    )
    with caplog.at_level(logging.ERROR):
        await manager.admit(_incoming("root"), "status-root")
        await asyncio.wait_for(manager.wait_idle(), timeout=1)

    messages = "\n".join(record.getMessage() for record in caplog.records)

    assert "Error processing node root" in messages
    assert "RuntimeError" in messages
    assert (secret in messages) is secret_is_logged

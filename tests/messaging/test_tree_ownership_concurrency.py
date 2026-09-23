import asyncio
import contextlib
import sys

import pytest

from free_claude_code.messaging.models import IncomingMessage, MessageScope
from free_claude_code.messaging.trees import manager as manager_module
from free_claude_code.messaging.trees.manager import TreeQueueManager
from free_claude_code.messaging.trees.transitions import (
    AdmissionRejection,
    CancellationReason,
    CancellationUiOwner,
    NodeClaim,
)

_SCOPE = MessageScope(platform="telegram", chat_id="chat")


def _incoming(node_id: str, *, reply_to: str | None = None) -> IncomingMessage:
    return IncomingMessage(
        text=node_id,
        chat_id=_SCOPE.chat_id,
        user_id="user",
        message_id=node_id,
        platform=_SCOPE.platform,
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


@pytest.mark.asyncio
async def test_cancelled_finisher_cannot_erase_or_overlap_a_new_claim() -> None:
    root_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_cleanup = asyncio.Event()
    child_started = asyncio.Event()
    release_child = asyncio.Event()
    active = 0
    maximum_active = 0

    async def process(claim: NodeClaim) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            if claim.node.node_id == "root":
                root_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()
                    await release_cleanup.wait()
                    raise
            child_started.set()
            await release_child.wait()
        finally:
            active -= 1

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("root"), "status-root")
    await root_started.wait()

    cancellation = asyncio.create_task(
        manager.cancel_node(_SCOPE, "root", reason=CancellationReason.STOP)
    )
    await cancellation_seen.wait()
    child = await manager.admit(
        _incoming("child", reply_to="root"),
        "status-child",
        parent_reference_id="root",
    )

    assert child.position == 1
    assert not child_started.is_set()
    assert maximum_active == 1

    release_cleanup.set()
    await cancellation
    await asyncio.wait_for(child_started.wait(), timeout=1)
    assert maximum_active == 1

    release_child.set()


@pytest.mark.asyncio
async def test_cancelled_runner_exception_cannot_fail_or_skip_queued_claim() -> None:
    root_started = asyncio.Event()
    child_started = asyncio.Event()
    release_child = asyncio.Event()
    unexpected_failures = []

    async def process(claim: NodeClaim) -> None:
        if claim.node.node_id == "root":
            root_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RuntimeError("runner escaped after cancellation") from None
        child_started.set()
        await release_child.wait()

    manager = TreeQueueManager(
        process,
        unexpected_failure_callback=unexpected_failures.append,
    )
    await manager.admit(_incoming("root"), "status-root")
    await root_started.wait()
    await manager.admit(
        _incoming("child", reply_to="root"),
        "status-child",
        parent_reference_id="root",
    )

    try:
        await manager.cancel_node(_SCOPE, "root", reason=CancellationReason.STOP)
        await asyncio.wait_for(child_started.wait(), timeout=1)

        assert len(unexpected_failures) == 1
        assert unexpected_failures[0].affected == ()
        assert unexpected_failures[0].queue_update is None
        assert unexpected_failures[0].snapshot is None
    finally:
        release_child.set()
        await _wait_for_no_tasks(manager)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["global_stop", "global_clear", "branch_clear"],
)
async def test_terminal_operation_serializes_with_successor_task_publication(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    root_started = asyncio.Event()
    release_root = asyncio.Event()
    successor_selected = asyncio.Event()
    release_successor_publication = asyncio.Event()
    child_started = asyncio.Event()
    child_exited = asyncio.Event()
    release_child = asyncio.Event()

    async def process(claim: NodeClaim) -> None:
        if claim.node.node_id == "root":
            root_started.set()
            await release_root.wait()
            return
        child_started.set()
        try:
            await release_child.wait()
        finally:
            child_exited.set()

    manager = TreeQueueManager(process)
    root = await manager.admit(_incoming("root"), "status-root")
    assert root.claim is not None
    await root_started.wait()
    await manager.admit(
        _incoming("child", reply_to="root"),
        "status-child",
        parent_reference_id="root",
    )
    tree = manager._repository.get_tree(root.claim.identity)
    assert tree is not None
    original_finish = tree.finish_and_claim_next

    async def pause_after_successor_selection(claim_id: str):
        completion = await original_finish(claim_id)
        if completion.next_claim is not None:
            successor_selected.set()
            await release_successor_publication.wait()
        return completion

    monkeypatch.setattr(
        tree,
        "finish_and_claim_next",
        pause_after_successor_selection,
    )
    operation_task: asyncio.Task | None = None
    try:
        release_root.set()
        await successor_selected.wait()
        if operation == "global_stop":
            operation_task = asyncio.create_task(
                manager.cancel_all(reason=CancellationReason.STOP)
            )
        elif operation == "global_clear":
            operation_task = asyncio.create_task(
                manager.clear_scope(_SCOPE, reason=CancellationReason.STOP)
            )
        else:
            operation_task = asyncio.create_task(
                manager.remove_message_subtree(
                    _SCOPE,
                    "root",
                    reason=CancellationReason.STOP,
                )
            )
        for _ in range(5):
            await asyncio.sleep(0)
            if operation_task.done():
                break

        assert not operation_task.done()
    finally:
        release_successor_publication.set()
        try:
            if operation_task is not None:
                await operation_task
        finally:
            release_child.set()
            await _wait_for_no_tasks(manager)

    assert manager.get_tree_count() == (1 if operation == "global_stop" else 0)
    assert manager.task_count() == 0
    if child_started.is_set():
        assert child_exited.is_set()


@pytest.mark.asyncio
async def test_duplicate_admission_is_rejected_and_processed_once() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def process(claim: NodeClaim) -> None:
        calls.append(claim.node.node_id)
        started.set()
        await release.wait()

    manager = TreeQueueManager(process)
    first = await manager.admit(_incoming("root"), "status-root")
    duplicate = await manager.admit(_incoming("root"), "status-duplicate")
    await started.wait()

    assert first.accepted is True
    assert duplicate.accepted is False
    assert calls == ["root"]
    assert manager.task_count() == 1

    release.set()


@pytest.mark.asyncio
async def test_node_and_status_references_are_published_together() -> None:
    release = asyncio.Event()

    async def process(_claim: NodeClaim) -> None:
        await release.wait()

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("root"), "status-root")

    assert await manager.resolve_node_id(_SCOPE, "root") == "root"
    assert await manager.resolve_node_id(_SCOPE, "status-root") == "root"

    release.set()


@pytest.mark.asyncio
async def test_callback_failure_does_not_block_the_next_claim() -> None:
    release_root = asyncio.Event()
    child_started = asyncio.Event()
    release_child = asyncio.Event()

    async def process(claim: NodeClaim) -> None:
        if claim.node.node_id == "root":
            await release_root.wait()
        else:
            child_started.set()
            await release_child.wait()

    async def broken_queue_callback(_queue) -> None:
        raise RuntimeError("UI unavailable")

    async def broken_started_callback(_claim) -> None:
        raise RuntimeError("UI unavailable")

    manager = TreeQueueManager(
        process,
        queue_update_callback=broken_queue_callback,
        node_started_callback=broken_started_callback,
    )
    await manager.admit(_incoming("root"), "status-root")
    await manager.admit(
        _incoming("child", reply_to="root"),
        "status-child",
        parent_reference_id="root",
    )

    release_root.set()
    await asyncio.wait_for(child_started.wait(), timeout=1)
    release_child.set()


@pytest.mark.asyncio
async def test_branch_removal_cannot_leave_a_detached_running_descendant() -> None:
    root_started = asyncio.Event()
    release_root = asyncio.Event()

    async def process(claim: NodeClaim) -> None:
        if claim.node.node_id == "root":
            root_started.set()
            await release_root.wait()

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("root"), "status-root")
    await root_started.wait()
    await manager.admit(
        _incoming("branch", reply_to="root"),
        "status-branch",
        parent_reference_id="root",
    )

    removed = await manager.remove_message_subtree(_SCOPE, "branch")
    late = await manager.admit(
        _incoming("late", reply_to="branch"),
        "status-late",
        parent_reference_id="branch",
    )

    assert removed.delete_message_ids == frozenset({"branch", "status-branch"})
    assert await manager.resolve_node_id(_SCOPE, "branch") is None
    assert late.accepted is False
    assert late.rejection is AdmissionRejection.PARENT_REMOVED
    assert await manager.get_node(_SCOPE, "late") is None

    release_root.set()


@pytest.mark.asyncio
async def test_clear_all_drains_terminal_claim_task_before_returning() -> None:
    terminal = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    manager: TreeQueueManager

    async def process(claim: NodeClaim) -> None:
        await manager.complete_claim(claim, "session-root")
        terminal.set()
        try:
            await release_cleanup.wait()
        finally:
            cleanup_finished.set()

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("root"), "status-root")
    await terminal.wait()

    try:
        await manager.clear_scope(_SCOPE, reason=CancellationReason.STOP)

        assert cleanup_finished.is_set()
        assert manager.task_count() == 0
        assert manager.get_tree_count() == 0
    finally:
        release_cleanup.set()
        await _wait_for_no_tasks(manager)


@pytest.mark.asyncio
async def test_clear_all_returns_committed_result_after_caller_cancellation() -> None:
    runner_cancelled = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def process(_claim: NodeClaim) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            runner_cancelled.set()
            try:
                await release_cleanup.wait()
            finally:
                cleanup_finished.set()
            raise

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("root"), "status-root")
    checkpoint = asyncio.Event()
    asyncio.get_running_loop().call_soon(checkpoint.set)
    await checkpoint.wait()

    clear_task = asyncio.create_task(
        manager.clear_scope(_SCOPE, reason=CancellationReason.STOP)
    )
    await runner_cancelled.wait()

    try:
        clear_task.cancel()
        cancellation_checkpoint = asyncio.Event()
        asyncio.get_running_loop().call_soon(cancellation_checkpoint.set)
        await cancellation_checkpoint.wait()

        assert not clear_task.done()
        release_cleanup.set()
        result = await clear_task
        assert {effect.node.node_id for effect in result.effects} == {"root"}
        assert cleanup_finished.is_set()
        assert manager.task_count() == 0
        assert manager.get_tree_count() == 0
    finally:
        release_cleanup.set()
        if not clear_task.done():
            clear_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await clear_task
        await _wait_for_no_tasks(manager)


@pytest.mark.asyncio
async def test_eager_task_factory_cannot_run_claim_before_admission_returns() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def process(_claim: NodeClaim) -> None:
        started.set()
        await release.wait()

    manager = TreeQueueManager(process)
    loop = asyncio.get_running_loop()
    original_factory = loop.get_task_factory()
    try:
        loop.set_task_factory(asyncio.eager_task_factory)
        decision = await manager.admit(_incoming("root"), "status-root")
    finally:
        loop.set_task_factory(original_factory)

    try:
        assert decision.accepted is True
        assert decision.claim is not None
        assert decision.claim.node.scope == _SCOPE
        assert manager.task_count() == 1
        assert not started.is_set()
        await asyncio.sleep(0)
        assert started.is_set()
    finally:
        release.set()
        await _wait_for_no_tasks(manager)


@pytest.mark.asyncio
async def test_claim_launch_uses_portable_event_loop_task_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = asyncio.Event()

    async def process(_claim: NodeClaim) -> None:
        processed.set()

    loop = asyncio.get_running_loop()
    original_create_task = loop.create_task

    def portable_create_task(coro, *, name=None, context=None):
        return original_create_task(coro, name=name, context=context)

    monkeypatch.setattr(loop, "create_task", portable_create_task)
    manager = TreeQueueManager(process)

    decision = await manager.admit(_incoming("root"), "status-root")
    await asyncio.wait_for(manager.wait_idle(), timeout=1)

    assert decision.accepted is True
    assert processed.is_set()


@pytest.mark.skipif(
    sys.platform in {"cygwin", "win32"} or sys.implementation.name == "pypy",
    reason="uvloop is not installed on this Python platform",
)
def test_claim_launch_runs_on_uvloop() -> None:
    uvloop = pytest.importorskip("uvloop")

    async def scenario() -> None:
        processed = asyncio.Event()

        async def process(_claim: NodeClaim) -> None:
            processed.set()

        manager = TreeQueueManager(process)
        decision = await manager.admit(_incoming("root"), "status-root")
        await asyncio.wait_for(manager.wait_idle(), timeout=1)

        assert decision.accepted is True
        assert processed.is_set()

    uvloop.run(scenario())


@pytest.mark.asyncio
async def test_absorbed_pre_run_cancellation_cannot_start_node_processor() -> None:
    root_started = asyncio.Event()
    release_root = asyncio.Event()
    callback_entered = asyncio.Event()
    callback_absorbed_cancellation = asyncio.Event()
    child_processor_started = asyncio.Event()

    async def process(claim: NodeClaim) -> None:
        if claim.node.node_id == "root":
            root_started.set()
            await release_root.wait()
            return
        child_processor_started.set()

    async def announce_started(claim: NodeClaim) -> None:
        if claim.node.node_id != "child":
            return
        callback_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            callback_absorbed_cancellation.set()

    manager = TreeQueueManager(process, node_started_callback=announce_started)
    await manager.admit(_incoming("root"), "status-root")
    await root_started.wait()
    child = await manager.admit(
        _incoming("child", reply_to="root"),
        "status-child",
        parent_reference_id="root",
    )
    assert child.position == 1

    release_root.set()
    await callback_entered.wait()
    result = await manager.cancel_node(
        _SCOPE,
        "child",
        reason=CancellationReason.STOP,
    )

    assert callback_absorbed_cancellation.is_set()
    assert not child_processor_started.is_set()
    assert [(effect.node.node_id, effect.ui_owner) for effect in result.effects] == [
        ("child", CancellationUiOwner.WORKFLOW)
    ]
    node = await manager.get_node(_SCOPE, "child")
    assert node is not None and node.state.value == "error"
    await _wait_for_no_tasks(manager)


@pytest.mark.asyncio
async def test_same_scoped_root_can_be_readmitted_while_detached_claim_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "CANCEL_TASK_DRAIN_TIMEOUT_S", 0.01)
    old_started = asyncio.Event()
    old_cancelled = asyncio.Event()
    release_old = asyncio.Event()
    new_started = asyncio.Event()
    release_new = asyncio.Event()
    claims: list[NodeClaim] = []

    async def process(claim: NodeClaim) -> None:
        claims.append(claim)
        if len(claims) == 1:
            old_started.set()
            while True:
                try:
                    await release_old.wait()
                    return
                except asyncio.CancelledError:
                    old_cancelled.set()
        new_started.set()
        await release_new.wait()

    manager = TreeQueueManager(process)
    old = await manager.admit(_incoming("root"), "status-root")
    await old_started.wait()

    try:
        cleared = await manager.clear_scope(_SCOPE, reason=CancellationReason.STOP)
        await old_cancelled.wait()
        assert {effect.node.node_id for effect in cleared.effects} == {"root"}
        assert manager.get_tree_count() == 0
        assert manager.task_count() == 1

        new = await manager.admit(_incoming("root"), "status-root")
        await new_started.wait()

        assert old.claim is not None
        assert new.claim is not None
        assert new.accepted is True
        assert new.claim.identity == old.claim.identity
        assert new.claim.claim_id != old.claim.claim_id
        assert manager.task_count() == 2
    finally:
        release_old.set()
        release_new.set()
        await _wait_for_no_tasks(manager)


@pytest.mark.asyncio
async def test_active_error_claim_can_be_cancelled_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "CANCEL_TASK_DRAIN_TIMEOUT_S", 0.01)
    started = asyncio.Event()
    first_cancellation = asyncio.Event()
    second_cancellation = asyncio.Event()
    release = asyncio.Event()
    cancellation_count = 0

    async def process(_claim: NodeClaim) -> None:
        nonlocal cancellation_count
        started.set()
        while True:
            try:
                await release.wait()
                return
            except asyncio.CancelledError:
                cancellation_count += 1
                if cancellation_count == 1:
                    first_cancellation.set()
                    continue
                second_cancellation.set()
                raise

    manager = TreeQueueManager(process)
    await manager.admit(_incoming("root"), "status-root")
    await started.wait()

    try:
        first = await manager.cancel_node(
            _SCOPE,
            "root",
            reason=CancellationReason.STOP,
        )
        await first_cancellation.wait()
        assert first.effects[0].ui_owner is CancellationUiOwner.RUNNER
        assert manager.task_count() == 1

        second = await manager.cancel_node(
            _SCOPE,
            "root",
            reason=CancellationReason.STOP,
        )
        await second_cancellation.wait()

        assert [
            (effect.node.node_id, effect.ui_owner) for effect in second.effects
        ] == [("root", CancellationUiOwner.RUNNER)]
        assert cancellation_count == 2
        await _wait_for_no_tasks(manager)
    finally:
        release.set()
        await _wait_for_no_tasks(manager)

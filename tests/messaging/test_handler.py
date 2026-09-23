import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from free_claude_code.messaging.command_context import ReplyClearResult, StopOutcome
from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.platforms.ports import MessagingStartupNotice
from free_claude_code.messaging.session import SessionStore
from free_claude_code.messaging.trees import (
    CancellationReason,
    CancellationResult,
    CancellationUiOwner,
    FailureResult,
    MessageReferenceKind,
    MessageState,
    MessageSubtreeRemovalResult,
    NodeClaim,
    NodeUiTarget,
    QueueEntry,
    ReplyTarget,
    TreeIdentity,
    TreeQueueManager,
    TreeSnapshot,
)
from free_claude_code.messaging.trees.transitions import CancellationEffect
from free_claude_code.messaging.voice import VoiceCancellationResult
from free_claude_code.messaging.workflow import MessagingWorkflow

_SCOPE = MessageScope(platform="telegram", chat_id="chat_1")
_OTHER_SCOPE = MessageScope(platform="telegram", chat_id="other_chat")


def _stop_outcome(
    cancelled_count: int,
    *,
    scopes: frozenset[MessageScope] = frozenset({_SCOPE}),
    fallback_required: bool = False,
) -> StopOutcome:
    return StopOutcome(cancelled_count, scopes, fallback_required)


def _startup_notice(chat_id: str = _SCOPE.chat_id) -> MessagingStartupNotice:
    return MessagingStartupNotice(chat_id=chat_id, transport_label="Bot API")


async def _event_stream(events):
    for event in events:
        await asyncio.sleep(0)
        yield event


def _claim(
    node_id: str = "node_1",
    *,
    prompt: str = "hello",
    parent_session_id: str | None = None,
) -> NodeClaim:
    return NodeClaim(
        identity=TreeIdentity(scope=_SCOPE, root_id="root_1"),
        claim_id="claim_1",
        node=NodeUiTarget(
            scope=_SCOPE,
            node_id=node_id,
            status_message_id="status_1",
        ),
        prompt=prompt,
        parent_session_id=parent_session_id,
    )


def _snapshot(root_id: str = "root_1") -> TreeSnapshot:
    return TreeSnapshot(scope=_SCOPE, root_id=root_id, nodes={})


def _session(events) -> MagicMock:
    session = MagicMock()
    session.start_task.return_value = _event_stream(events)
    return session


async def _wait_for_idle(workflow: MessagingWorkflow) -> None:
    for _ in range(200):
        if workflow.tree_queue.task_count() == 0:
            await asyncio.sleep(0)
            return
        await asyncio.sleep(0.01)
    raise AssertionError("messaging workflow did not become idle")


@pytest.fixture
def handler(mock_platform, mock_cli_manager, mock_session_store):
    default_session = _session([{"type": "exit", "code": 0}])
    mock_cli_manager.get_or_create_session.return_value = (
        default_session,
        "session_1",
        False,
    )
    return MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
        platform_name="telegram",
        voice_cancellation=mock_platform,
    )


@pytest.mark.asyncio
async def test_handle_message_turn_trace_always_includes_full_message_text(
    mock_platform,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
):
    text = "user-message-content-visible-in-trace"
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
    )
    incoming = incoming_message_factory(text=text)
    with (
        patch.object(workflow.turn_intake, "handle_message", new_callable=AsyncMock),
        patch("free_claude_code.messaging.workflow.trace_event") as trace_mock,
    ):
        await workflow.handle_message(incoming)

    assert trace_mock.call_args.kwargs["event"] == "turn.received"
    assert trace_mock.call_args.kwargs["message_text"] == text


@pytest.mark.asyncio
async def test_user_prompt_and_status_are_recorded_for_global_clear(
    handler,
    mock_session_store,
    incoming_message_factory,
) -> None:
    incoming = incoming_message_factory(text="keep me", message_id="user-prompt")

    await handler.handle_message(incoming)
    await _wait_for_idle(handler)

    mock_session_store.record_message_id.assert_has_calls(
        [
            call(
                incoming.platform,
                incoming.chat_id,
                "user-prompt",
                "in",
                "prompt",
            ),
            call(
                incoming.platform,
                incoming.chat_id,
                "msg_123",
                "out",
                "status",
            ),
        ]
    )


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (None, "Launching"),
        (
            ReplyTarget(
                node_id="parent",
                reference_id="parent",
                reference_kind=MessageReferenceKind.PROMPT,
                queue_position=None,
            ),
            "Continuing",
        ),
        (
            ReplyTarget(
                node_id="parent",
                reference_id="status-parent",
                reference_kind=MessageReferenceKind.STATUS,
                queue_position=3,
            ),
            "position 3",
        ),
    ],
)
def test_initial_status_uses_immutable_reply_advice(handler, target, expected):
    assert expected in handler.turn_intake._get_initial_status(target)


@pytest.mark.asyncio
async def test_global_stop_success_uses_existing_status_without_confirmation(
    handler, mock_platform, incoming_message_factory
):
    incoming = incoming_message_factory(text="/stop")
    handler.stop_all_tasks = AsyncMock(return_value=_stop_outcome(5))

    await handler.handle_message(incoming)

    handler.stop_all_tasks.assert_awaited_once()
    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_feedback_scopes",
    (
        frozenset({_OTHER_SCOPE}),
        frozenset({_SCOPE, _OTHER_SCOPE}),
    ),
    ids=("remote-only", "mixed-local-and-remote"),
)
async def test_global_stop_reports_cross_chat_work(
    handler,
    mock_platform,
    incoming_message_factory,
    status_feedback_scopes: frozenset[MessageScope],
) -> None:
    incoming = incoming_message_factory(text="/stop")
    handler.stop_all_tasks = AsyncMock(
        return_value=_stop_outcome(2, scopes=status_feedback_scopes)
    )

    await handler.handle_message(incoming)

    assert (
        "Cancelled 2 pending or active requests"
        in mock_platform.queue_send_message.call_args.args[1]
    )


@pytest.mark.asyncio
async def test_global_stop_noop_reports_nothing_to_stop(
    handler, mock_platform, incoming_message_factory
) -> None:
    incoming = incoming_message_factory(text="/stop")
    handler.stop_all_tasks = AsyncMock(
        return_value=_stop_outcome(0, scopes=frozenset())
    )

    await handler.handle_message(incoming)

    mock_platform.queue_send_message.assert_awaited_once_with(
        incoming.chat_id,
        "⏹ *Stopped\\.* Nothing to stop\\.",
        fire_and_forget=False,
        message_thread_id=None,
    )


@pytest.mark.asyncio
async def test_global_stop_without_status_target_sends_fallback_confirmation(
    handler, mock_platform, incoming_message_factory
) -> None:
    incoming = incoming_message_factory(text="/stop")
    handler.stop_all_tasks = AsyncMock(
        return_value=_stop_outcome(
            1,
            scopes=frozenset(),
            fallback_required=True,
        )
    )

    await handler.handle_message(incoming)

    assert (
        "Cancelled 1 pending or active request"
        in mock_platform.queue_send_message.call_args.args[1]
    )


@pytest.mark.asyncio
async def test_failed_global_stop_never_reports_success(
    handler, mock_platform, incoming_message_factory
) -> None:
    incoming = incoming_message_factory(text="/stop")
    handler.stop_all_tasks = AsyncMock(
        side_effect=RuntimeError("Failed to stop 1 managed Claude session.")
    )

    with pytest.raises(RuntimeError, match="Failed to stop"):
        await handler.handle_message(incoming)

    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_stop_resolves_and_stops_only_target(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    handler.stop_reply = AsyncMock(return_value=_stop_outcome(1))
    handler.stop_all_tasks = AsyncMock(return_value=_stop_outcome(999))
    incoming = incoming_message_factory(
        text="/stop",
        message_id="stop_msg",
        reply_to_message_id="status_root",
    )

    await handler.handle_message(incoming)

    handler.stop_reply.assert_awaited_once_with(incoming.scope, "status_root")
    handler.stop_all_tasks.assert_not_awaited()
    mock_cli_manager.stop_all.assert_not_awaited()
    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_stop_unknown_does_not_stop_all(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    handler.stop_reply = AsyncMock(return_value=_stop_outcome(0, scopes=frozenset()))
    handler.stop_all_tasks = AsyncMock(return_value=_stop_outcome(5))
    incoming = incoming_message_factory(
        text="/stop",
        message_id="stop_msg",
        reply_to_message_id="unknown_msg",
    )

    await handler.handle_message(incoming)

    handler.stop_reply.assert_awaited_once_with(incoming.scope, "unknown_msg")
    handler.stop_all_tasks.assert_not_awaited()
    mock_cli_manager.stop_all.assert_not_awaited()
    assert (
        "Nothing to stop for that message"
        in mock_platform.queue_send_message.call_args.args[1]
    )


@pytest.mark.asyncio
async def test_reply_stop_cancels_voice_when_no_tree_was_admitted(
    handler,
    mock_platform,
    incoming_message_factory,
) -> None:
    mock_platform.cancel_pending_voice.return_value = VoiceCancellationResult(
        scope=_SCOPE,
        voice_message_id="voice",
        status_message_id="voice_status",
        delete_message_ids=frozenset({"voice", "voice_status"}),
    )
    incoming = incoming_message_factory(
        text="/stop",
        message_id="stop_msg",
        reply_to_message_id="voice",
    )

    await handler.handle_message(incoming)
    await asyncio.sleep(0)

    mock_platform.queue_send_message.assert_not_awaited()
    assert mock_platform.queue_edit_message.call_args.args[1] == "voice_status"
    assert "Stopped" in mock_platform.queue_edit_message.call_args.args[2]


@pytest.mark.asyncio
async def test_reply_stop_without_voice_status_sends_fallback_confirmation(
    handler,
    mock_platform,
    incoming_message_factory,
) -> None:
    mock_platform.cancel_pending_voice.return_value = VoiceCancellationResult(
        scope=_SCOPE,
        voice_message_id="voice",
        status_message_id=None,
        delete_message_ids=frozenset({"voice"}),
    )
    incoming = incoming_message_factory(
        text="/stop",
        message_id="stop_msg",
        reply_to_message_id="voice",
    )

    await handler.handle_message(incoming)

    assert "Cancelled 1 request" in mock_platform.queue_send_message.call_args.args[1]
    mock_platform.queue_edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_stop_resolves_tree_after_joining_voice_handoff(
    handler,
    mock_platform,
    incoming_message_factory,
) -> None:
    events: list[str] = []

    async def cancel_voice(*_args) -> VoiceCancellationResult:
        events.append("voice")
        return VoiceCancellationResult(
            scope=_SCOPE,
            voice_message_id="voice",
            status_message_id="voice_status",
            delete_message_ids=frozenset({"voice", "voice_status"}),
        )

    async def resolve_tree(*_args) -> str:
        events.append("tree")
        return "voice"

    mock_platform.cancel_pending_voice.side_effect = cancel_voice
    cancellation = CancellationResult(
        effects=(
            CancellationEffect(
                node=NodeUiTarget(
                    scope=_SCOPE,
                    node_id="voice",
                    status_message_id="voice_status",
                ),
                ui_owner=CancellationUiOwner.WORKFLOW,
            ),
        )
    )
    incoming = incoming_message_factory(
        text="/stop",
        message_id="stop_msg",
        reply_to_message_id="voice",
    )

    with (
        patch.object(
            handler.tree_queue,
            "resolve_node_id",
            side_effect=resolve_tree,
        ),
        patch.object(
            handler.tree_queue,
            "cancel_node",
            new_callable=AsyncMock,
            return_value=cancellation,
        ) as cancel_node,
    ):
        await handler.handle_message(incoming)

    cancel_node.assert_awaited_once_with(
        incoming.scope,
        "voice",
        reason=CancellationReason.STOP,
    )
    assert events == ["voice", "tree"]
    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_stop_port_failure_prevents_tree_transition(
    handler,
    mock_platform,
) -> None:
    mock_platform.cancel_pending_voice.side_effect = RuntimeError("voice port failed")
    with (
        patch.object(
            handler.tree_queue,
            "resolve_node_id",
            new_callable=AsyncMock,
        ) as resolve,
        pytest.raises(RuntimeError, match="voice port failed"),
    ):
        await handler.stop_reply(_SCOPE, "voice")

    resolve.assert_not_awaited()
    mock_platform.queue_edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_stop_renders_voice_before_resolve_failure(
    handler,
    mock_platform,
) -> None:
    mock_platform.cancel_pending_voice.return_value = VoiceCancellationResult(
        scope=_SCOPE,
        voice_message_id="voice",
        status_message_id="voice_status",
        delete_message_ids=frozenset({"voice", "voice_status"}),
    )
    with (
        patch.object(
            handler.tree_queue,
            "resolve_node_id",
            AsyncMock(side_effect=RuntimeError("resolve failed")),
        ),
        pytest.raises(RuntimeError, match="resolve failed"),
    ):
        await handler.stop_reply(_SCOPE, "voice")
    await asyncio.sleep(0)

    mock_platform.queue_edit_message.assert_awaited_once()
    assert mock_platform.queue_edit_message.call_args.args[1] == "voice_status"


@pytest.mark.asyncio
async def test_cancelled_reply_stop_finishes_after_voice_join_and_lock_wait(
    handler,
    mock_platform,
) -> None:
    voice_returned = asyncio.Event()

    async def cancel_voice(*_args) -> VoiceCancellationResult:
        voice_returned.set()
        return VoiceCancellationResult(
            scope=_SCOPE,
            voice_message_id="voice",
            status_message_id="voice_status",
            delete_message_ids=frozenset({"voice", "voice_status"}),
        )

    mock_platform.cancel_pending_voice.side_effect = cancel_voice
    resolve = AsyncMock(return_value=None)
    await handler._state_lock.acquire()
    try:
        with patch.object(handler.tree_queue, "resolve_node_id", resolve):
            stop_task = asyncio.create_task(handler.stop_reply(_SCOPE, "voice"))
            await voice_returned.wait()
            await asyncio.sleep(0)
            stop_task.cancel()
            stop_task.cancel()
            await asyncio.sleep(0)

            assert not stop_task.done()
            handler._state_lock.release()
            with pytest.raises(asyncio.CancelledError):
                await stop_task
    finally:
        if handler._state_lock.locked():
            handler._state_lock.release()
    await asyncio.sleep(0)

    assert stop_task.cancelling() == 2
    resolve.assert_awaited_once_with(_SCOPE, "voice")
    mock_platform.queue_edit_message.assert_awaited_once()
    assert mock_platform.queue_edit_message.call_args.args[1] == "voice_status"


@pytest.mark.asyncio
async def test_cancelled_reply_stop_preserves_later_operation_failure(
    handler,
    mock_platform,
) -> None:
    voice_returned = asyncio.Event()
    failure = RuntimeError("resolve failed after cancellation")

    async def cancel_voice(*_args) -> None:
        voice_returned.set()

    mock_platform.cancel_pending_voice.side_effect = cancel_voice
    resolve = AsyncMock(side_effect=failure)
    await handler._state_lock.acquire()
    try:
        with patch.object(handler.tree_queue, "resolve_node_id", resolve):
            stop_task = asyncio.create_task(handler.stop_reply(_SCOPE, "voice"))
            await voice_returned.wait()
            await asyncio.sleep(0)
            stop_task.cancel()
            await asyncio.sleep(0)

            assert not stop_task.done()
            handler._state_lock.release()
            with pytest.raises(RuntimeError) as raised:
                await stop_task
    finally:
        if handler._state_lock.locked():
            handler._state_lock.release()

    assert raised.value is failure
    assert stop_task.cancelling() == 1
    resolve.assert_awaited_once_with(_SCOPE, "voice")


@pytest.mark.asyncio
async def test_stats_command_reports_cli_and_tree_counts(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    mock_cli_manager.get_stats.return_value = {"active_sessions": 2}

    await handler.handle_message(incoming_message_factory(text="/stats"))

    text = mock_platform.queue_send_message.call_args.args[1]
    assert "Active CLI: 2" in text
    assert "Message Trees: 0" in text
    assert mock_platform.queue_send_message.call_args.kwargs["fire_and_forget"] is False


@pytest.mark.asyncio
async def test_status_echo_is_filtered(
    handler, mock_platform, incoming_message_factory
):
    await handler.handle_message(incoming_message_factory(text="⏳ Thinking..."))

    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_turn_uses_public_admission_and_persists_exact_snapshot(
    handler,
    mock_platform,
    mock_session_store,
    incoming_message_factory,
):
    incoming = incoming_message_factory(text="hello", message_id="node_1")
    mock_platform.queue_send_message.return_value = "status_123"

    await handler.handle_message(incoming)
    await _wait_for_idle(handler)

    assert "Launching" in mock_platform.queue_send_message.call_args.args[1]
    assert mock_session_store.save_tree_snapshot.call_count >= 2
    view = await handler.tree_queue.get_node(incoming.scope, "node_1")
    assert view is not None
    assert view.state is MessageState.COMPLETED


@pytest.mark.asyncio
async def test_duplicate_delivery_removes_its_provisional_status(
    handler,
    mock_platform,
    mock_session_store,
    incoming_message_factory,
):
    incoming = incoming_message_factory(text="hello", message_id="duplicate")
    mock_platform.queue_send_message.side_effect = ["status-first", "status-rejected"]

    await handler.handle_message(incoming)
    await _wait_for_idle(handler)
    await handler.handle_message(incoming)

    mock_platform.queue_delete_messages.assert_awaited_once_with(
        incoming.chat_id,
        ["status-rejected"],
        fire_and_forget=False,
    )
    mock_session_store.forget_tracked_message_ids.assert_called_once_with(
        incoming.platform,
        incoming.chat_id,
        {"status-rejected"},
    )


@pytest.mark.asyncio
async def test_pre_sent_status_is_edited_in_place(
    handler, mock_platform, incoming_message_factory
):
    incoming = incoming_message_factory(
        text="hello",
        message_id="node_1",
        status_message_id="existing_status",
    )

    await handler.handle_message(incoming)
    await _wait_for_idle(handler)

    first_edit = mock_platform.queue_edit_message.call_args_list[0]
    assert first_edit.args[1] == "existing_status"
    assert "Launching" in first_edit.args[2]
    assert first_edit.kwargs["fire_and_forget"] is False
    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_reply_is_rendered_with_atomic_queue_position(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    started = asyncio.Event()

    async def blocking_start(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)
        if False:
            yield {}

    session = MagicMock()
    session.start_task = blocking_start
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    mock_platform.queue_send_message.side_effect = ["status_root", "status_child"]

    root = incoming_message_factory(text="root", message_id="root")
    await handler.handle_message(root)
    await started.wait()
    child = incoming_message_factory(
        text="child",
        message_id="child",
        reply_to_message_id="status_root",
    )
    await handler.handle_message(child)

    assert "position 1" in mock_platform.queue_send_message.call_args.args[1]
    queued_edit = mock_platform.queue_edit_message.call_args_list[-1]
    assert queued_edit.args[1] == "status_child"
    assert "position 1" in queued_edit.args[2]

    await handler.stop_all_tasks()


@pytest.mark.asyncio
async def test_queue_position_callback_consumes_immutable_entries(
    handler, mock_platform
):
    queue = (
        QueueEntry(
            node=NodeUiTarget(
                scope=_SCOPE,
                node_id="child_1",
                status_message_id="status_1",
            ),
            position=1,
        ),
        QueueEntry(
            node=NodeUiTarget(
                scope=_SCOPE,
                node_id="child_2",
                status_message_id="status_2",
            ),
            position=2,
        ),
    )

    await handler.turn_intake.update_queue_positions(queue)
    await asyncio.sleep(0)

    calls = mock_platform.queue_edit_message.call_args_list
    assert [call.args[1] for call in calls] == ["status_1", "status_2"]
    assert "position 1" in calls[0].args[2]
    assert "position 2" in calls[1].args[2]


@pytest.mark.asyncio
async def test_claim_started_callback_renders_processing(handler, mock_platform):
    claim = _claim()

    await handler.turn_intake.mark_node_processing(claim)
    await asyncio.sleep(0)

    args, kwargs = mock_platform.queue_edit_message.call_args
    assert args[0:2] == ("chat_1", "status_1")
    assert "Processing" in args[2]
    assert kwargs["parse_mode"] == "MarkdownV2"


@pytest.mark.asyncio
async def test_stop_all_applies_immutable_ui_ownership_and_snapshots(
    handler, mock_cli_manager, mock_platform, mock_session_store
):
    workflow_owned = CancellationEffect(
        node=NodeUiTarget(
            scope=_SCOPE,
            node_id="queued",
            status_message_id="status_queued",
        ),
        ui_owner=CancellationUiOwner.WORKFLOW,
    )
    runner_owned = CancellationEffect(
        node=NodeUiTarget(
            scope=_SCOPE,
            node_id="active",
            status_message_id="status_active",
        ),
        ui_owner=CancellationUiOwner.RUNNER,
    )
    snapshot = _snapshot()
    result = CancellationResult(
        effects=(workflow_owned, runner_owned),
        snapshots=(snapshot,),
    )
    with patch.object(
        handler.tree_queue,
        "cancel_all",
        AsyncMock(return_value=result),
    ) as cancel_all:
        outcome = await handler.stop_all_tasks()
    await asyncio.sleep(0)

    assert outcome == _stop_outcome(2)
    cancel_all.assert_awaited_once_with(reason=CancellationReason.STOP)
    mock_cli_manager.stop_all.assert_awaited_once()
    assert mock_platform.fire_and_forget.call_count == 1
    assert mock_platform.queue_edit_message.call_args.args[1] == "status_queued"
    mock_session_store.save_tree_snapshot.assert_called_once_with(snapshot)


@pytest.mark.asyncio
async def test_stop_all_joins_voices_before_tree_transaction_and_deduplicates(
    handler,
    mock_cli_manager,
    mock_platform,
) -> None:
    events: list[str] = []
    voice = VoiceCancellationResult(
        scope=_SCOPE,
        voice_message_id="voice",
        status_message_id=None,
        delete_message_ids=frozenset({"voice"}),
    )
    tree_result = CancellationResult(
        effects=(
            CancellationEffect(
                node=NodeUiTarget(
                    scope=_SCOPE,
                    node_id="voice",
                    status_message_id="voice_status",
                ),
                ui_owner=CancellationUiOwner.WORKFLOW,
            ),
        )
    )

    async def cancel_voices() -> tuple[VoiceCancellationResult, ...]:
        assert not handler._state_lock.locked()
        events.append("voices")
        return (voice,)

    async def cancel_trees(*, reason: CancellationReason) -> CancellationResult:
        assert reason is CancellationReason.STOP
        assert handler._state_lock.locked()
        events.append("trees")
        return tree_result

    mock_platform.cancel_all_pending_voices.side_effect = cancel_voices
    with patch.object(handler.tree_queue, "cancel_all", side_effect=cancel_trees):
        outcome = await handler.stop_all_tasks()
    await asyncio.sleep(0)

    assert outcome == _stop_outcome(1)
    assert events == ["voices", "trees"]
    mock_cli_manager.stop_all.assert_awaited_once()
    mock_platform.queue_edit_message.assert_awaited_once()
    assert mock_platform.queue_edit_message.call_args.args[1] == "voice_status"


@pytest.mark.asyncio
async def test_stop_all_voice_join_failure_prevents_false_transition(
    handler,
    mock_cli_manager,
    mock_platform,
) -> None:
    mock_platform.cancel_all_pending_voices.side_effect = RuntimeError(
        "voice handoff cleanup failed"
    )

    with (
        patch.object(
            handler.tree_queue, "cancel_all", new_callable=AsyncMock
        ) as cancel,
        pytest.raises(RuntimeError, match="voice handoff cleanup failed"),
    ):
        await handler.stop_all_tasks()

    cancel.assert_not_awaited()
    mock_cli_manager.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_all_persists_committed_transition_before_cli_shutdown(
    handler,
    mock_cli_manager,
    mock_session_store,
):
    shutdown_started = asyncio.Event()
    release_shutdown = asyncio.Event()
    snapshot = _snapshot()
    result = CancellationResult(snapshots=(snapshot,))

    async def block_shutdown() -> None:
        shutdown_started.set()
        await release_shutdown.wait()

    mock_cli_manager.stop_all.side_effect = block_shutdown
    with patch.object(
        handler.tree_queue,
        "cancel_all",
        AsyncMock(return_value=result),
    ):
        stop_task = asyncio.create_task(handler.stop_all_tasks())
        await shutdown_started.wait()

        mock_session_store.save_tree_snapshot.assert_called_once_with(snapshot)
        stop_task.cancel()
        stop_task.cancel()
        await asyncio.sleep(0)
        assert not stop_task.done()
        release_shutdown.set()
        with pytest.raises(asyncio.CancelledError):
            await stop_task
        assert stop_task.cancelling() == 2


@pytest.mark.asyncio
async def test_terminal_close_waits_past_interactive_drain_timeout(
    monkeypatch,
    mock_platform,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
) -> None:
    monkeypatch.setattr(
        "free_claude_code.messaging.trees.manager.CANCEL_TASK_DRAIN_TIMEOUT_S",
        0.01,
    )
    runner_started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_cleanup = asyncio.Event()
    cli_stop_seen = asyncio.Event()

    async def cancellation_delayed_runner(_claim: NodeClaim) -> None:
        runner_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_cleanup.wait()
            raise

    async def stop_cli() -> None:
        cli_stop_seen.set()

    mock_platform.queue_send_message.return_value = "status_1"
    mock_cli_manager.stop_all.side_effect = stop_cli
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
        platform_name="telegram",
    )
    workflow._tree_queue = TreeQueueManager(cancellation_delayed_runner)
    await workflow.handle_message(
        incoming_message_factory(text="work", message_id="work_1")
    )
    await runner_started.wait()

    close_task = asyncio.create_task(workflow.close())
    try:
        await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
        await asyncio.wait_for(cli_stop_seen.wait(), timeout=1)

        mock_cli_manager.stop_all.assert_awaited_once()
        assert close_task.done() is False
        assert workflow.tree_queue.task_count() == 1
        mock_session_store.flush_pending_save.assert_not_called()

        release_cleanup.set()
        await asyncio.wait_for(close_task, timeout=1)

        assert workflow.tree_queue.task_count() == 0
        mock_session_store.flush_pending_save.assert_called_once()
    finally:
        release_cleanup.set()
        if not close_task.done():
            await asyncio.wait_for(close_task, timeout=1)


@pytest.mark.asyncio
async def test_node_runner_success_uses_claim_and_semantic_completion(
    handler, mock_cli_manager, mock_platform, mock_session_store
):
    claim = _claim(prompt="say hello")
    session = _session(
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "thinking", "thinking": "Let me think"},
                        {"type": "text", "text": "Hello world"},
                    ]
                },
            },
            {"type": "exit", "code": 0},
        ]
    )
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    snapshot = _snapshot()
    with patch.object(
        handler.tree_queue,
        "complete_claim",
        AsyncMock(return_value=snapshot),
    ) as complete_claim:
        await handler.node_runner.process_node(claim)

    complete_claim.assert_awaited_once_with(claim, "session_1")
    mock_session_store.save_tree_snapshot.assert_called_once_with(snapshot)
    rendered = mock_platform.queue_edit_message.call_args_list[-1].args[2]
    assert "✅ *Complete*" in rendered
    assert "Hello world" in rendered
    mock_cli_manager.get_or_create_session.assert_awaited_once_with(session_id=None)
    assert session.start_task.call_args.args == ("say hello",)
    assert session.start_task.call_args.kwargs == {
        "session_id": None,
        "fork_session": False,
    }


@pytest.mark.asyncio
async def test_node_runner_uses_claim_parent_session_for_fork(
    handler, mock_cli_manager
):
    claim = _claim(parent_session_id="parent_session")
    session = _session([{"type": "exit", "code": 0}])
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "child_session",
        False,
    )

    await handler.node_runner.process_node(claim)

    mock_cli_manager.get_or_create_session.assert_awaited_once_with(
        session_id="parent_session"
    )
    assert session.start_task.call_args.kwargs == {
        "session_id": "parent_session",
        "fork_session": True,
    }


@pytest.mark.asyncio
async def test_session_info_records_real_session_through_manager(
    handler, mock_cli_manager, mock_session_store
):
    claim = _claim()
    session = _session(
        [
            {"type": "session_info", "session_id": "real_session"},
            {"type": "exit", "code": 0},
        ]
    )
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "temporary_session",
        True,
    )
    record_snapshot = _snapshot("record")
    complete_snapshot = _snapshot("complete")
    with (
        patch.object(
            handler.tree_queue,
            "record_session",
            AsyncMock(return_value=record_snapshot),
        ) as record_session,
        patch.object(
            handler.tree_queue,
            "complete_claim",
            AsyncMock(return_value=complete_snapshot),
        ) as complete_claim,
    ):
        await handler.node_runner.process_node(claim)

    mock_cli_manager.register_real_session_id.assert_awaited_once_with(
        "temporary_session", "real_session"
    )
    record_session.assert_awaited_once_with(claim, "real_session")
    complete_claim.assert_awaited_once_with(claim, "real_session")
    assert mock_session_store.save_tree_snapshot.call_args_list == [
        ((record_snapshot,), {}),
        ((complete_snapshot,), {}),
    ]


@pytest.mark.asyncio
async def test_session_info_rejects_unregistered_real_session_without_recording_it(
    handler,
    mock_cli_manager,
) -> None:
    from free_claude_code.messaging.node_event_pipeline import (
        handle_session_info_event,
    )

    claim = _claim()
    record_session = AsyncMock()
    mock_cli_manager.register_real_session_id.return_value = False

    with pytest.raises(
        RuntimeError,
        match=r"^Managed Claude session registration failed\.$",
    ) as raised:
        await handle_session_info_event(
            {"type": "session_info", "session_id": "real_session"},
            claim,
            None,
            "temporary_session",
            cli_manager=mock_cli_manager,
            record_session=record_session,
        )

    assert "real_session" not in str(raised.value)
    assert "temporary_session" not in str(raised.value)
    record_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_limit_failure_uses_non_propagating_claim_failure(
    handler, mock_cli_manager, mock_platform
):
    claim = _claim()
    mock_cli_manager.get_or_create_session.side_effect = RuntimeError("session limit")
    result = FailureResult(
        affected=(claim.node,),
        queue_update=None,
        snapshot=_snapshot(),
    )
    with patch.object(
        handler.tree_queue,
        "fail_claim",
        AsyncMock(return_value=result),
    ) as fail_claim:
        await handler.node_runner.process_node(claim)

    fail_claim.assert_awaited_once_with(claim, propagate=False)
    rendered = mock_platform.queue_edit_message.call_args_list[-1].args[2]
    assert "Session limit reached" in rendered


@pytest.mark.asyncio
async def test_non_exit_error_defers_child_failure_until_stream_ends(
    handler, mock_cli_manager, mock_platform, mock_session_store
):
    claim = _claim()
    session = _session([{"type": "error", "error": {"message": "CLI crashed"}}])
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    child = NodeUiTarget(
        scope=_SCOPE,
        node_id="child",
        status_message_id="status_child",
    )
    snapshot = _snapshot()
    result = FailureResult(
        affected=(claim.node, child),
        queue_update=None,
        snapshot=snapshot,
    )
    with patch.object(
        handler.tree_queue,
        "fail_claim",
        AsyncMock(return_value=result),
    ) as fail_claim:
        await handler.node_runner.process_node(claim)
    await asyncio.sleep(0)

    assert fail_claim.await_args_list == [
        call(claim, propagate=False),
        call(claim, propagate=True),
    ]
    assert mock_session_store.save_tree_snapshot.call_args_list == [
        call(snapshot),
        call(snapshot),
    ]
    rendered = "\n".join(
        call.args[2] for call in mock_platform.queue_edit_message.call_args_list
    )
    assert "❌ *Error*" in rendered
    assert "CLI crashed" in rendered
    assert "Parent task failed" in rendered


@pytest.mark.asyncio
async def test_provider_error_exit_does_not_mask_or_complete(
    handler, mock_cli_manager, mock_platform
):
    claim = _claim()
    provider_error = "API Error: Request rejected (429)\nProvider rate limit reached."
    session = _session(
        [
            {"type": "error", "error": {"message": provider_error}},
            {"type": "exit", "code": 1},
        ]
    )
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    failure = FailureResult(
        affected=(claim.node,),
        queue_update=None,
        snapshot=_snapshot(),
    )
    with (
        patch.object(
            handler.tree_queue,
            "fail_claim",
            AsyncMock(return_value=failure),
        ) as fail_claim,
        patch.object(
            handler.tree_queue,
            "complete_claim",
            AsyncMock(),
        ) as complete_claim,
    ):
        await handler.node_runner.process_node(claim)

    assert fail_claim.await_args_list == [
        call(claim, propagate=False),
        call(claim, propagate=True),
    ]
    complete_claim.assert_not_awaited()
    rendered = mock_platform.queue_edit_message.call_args_list[-1].args[2]
    assert "API Error: Request rejected" in rendered
    assert "Process exited with code" not in rendered
    assert "✅ *Complete*" not in rendered


@pytest.mark.asyncio
async def test_success_exit_still_renders_complete_after_non_exit_error(
    handler, mock_cli_manager, mock_platform
):
    claim = _claim()
    session = _session(
        [
            {"type": "error", "error": {"message": "recoverable warning"}},
            {"type": "exit", "code": 0},
        ]
    )
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    failure = FailureResult(
        affected=(claim.node,),
        queue_update=None,
        snapshot=_snapshot(),
    )
    with (
        patch.object(
            handler.tree_queue,
            "fail_claim",
            AsyncMock(return_value=failure),
        ) as fail_claim,
        patch.object(
            handler.tree_queue,
            "complete_claim",
            AsyncMock(return_value=_snapshot()),
        ) as complete_claim,
    ):
        await handler.node_runner.process_node(claim)

    fail_claim.assert_awaited_once_with(
        claim,
        propagate=False,
    )
    complete_claim.assert_awaited_once_with(claim, "session_1")
    assert (
        "✅ *Complete*" in mock_platform.queue_edit_message.call_args_list[-1].args[2]
    )


@pytest.mark.asyncio
async def test_unexpected_runner_exception_uses_detailed_task_failed_ui(
    handler, mock_cli_manager, mock_platform
):
    claim = _claim()

    async def failing_start(*args, **kwargs):
        raise ValueError("runner exploded")
        if False:
            yield {}

    session = MagicMock()
    session.start_task = failing_start
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    failure = FailureResult(
        affected=(claim.node,),
        queue_update=None,
        snapshot=_snapshot(),
    )
    with patch.object(
        handler.tree_queue,
        "fail_claim",
        AsyncMock(return_value=failure),
    ) as fail_claim:
        await handler.node_runner.process_node(claim)

    fail_claim.assert_awaited_once_with(
        claim,
        propagate=True,
    )
    rendered = mock_platform.queue_edit_message.call_args_list[-1].args[2]
    assert "Task Failed" in rendered
    assert "runner exploded" in rendered


@pytest.mark.asyncio
async def test_stop_cancellation_preserves_partial_transcript(
    handler, mock_cli_manager, mock_platform
):
    claim = _claim(prompt="work")
    started = asyncio.Event()

    async def start_task(*args, **kwargs):
        yield {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "partial answer"}]},
        }
        started.set()
        await asyncio.sleep(60)

    session = MagicMock()
    session.start_task = start_task
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    failure = FailureResult(
        affected=(claim.node,),
        queue_update=None,
        snapshot=_snapshot(),
    )
    with patch.object(
        handler.tree_queue,
        "fail_claim",
        AsyncMock(return_value=failure),
    ) as fail_claim:
        task = asyncio.create_task(handler.node_runner.process_node(claim))
        await started.wait()
        task.cancel(CancellationReason.STOP)
        await task

    fail_claim.assert_awaited_once_with(
        claim,
        propagate=False,
    )
    rendered = mock_platform.queue_edit_message.call_args_list[-1].args[2]
    assert "partial answer" in rendered
    assert "⏹ *Stopped\\.*" in rendered
    assert rendered.index("partial answer") < rendered.index("⏹ *Stopped\\.*")


@pytest.mark.asyncio
async def test_global_clear_command_deletes_returned_ids(
    handler, mock_platform, incoming_message_factory
):
    handler.clear_chat = AsyncMock(return_value=frozenset({"100", "101"}))
    incoming = incoming_message_factory(
        text="/clear",
        chat_id="chat_1",
        message_id="150",
    )

    await handler.handle_message(incoming)

    handler.clear_chat.assert_awaited_once_with("telegram", "chat_1")
    mock_platform.queue_delete_messages.assert_awaited_once_with(
        "chat_1",
        ["150", "101", "100"],
        fire_and_forget=False,
    )
    mock_platform.queue_send_message.assert_not_awaited()
    handler.session_store.record_message_id.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    (RuntimeError("clear failed"), asyncio.CancelledError()),
    ids=("failure", "cancellation"),
)
async def test_interrupted_global_clear_records_its_deferred_command_id(
    handler,
    mock_platform,
    mock_session_store,
    incoming_message_factory,
    failure: BaseException,
) -> None:
    handler.clear_chat = AsyncMock(side_effect=failure)
    incoming = incoming_message_factory(
        text="/clear",
        chat_id="chat_1",
        message_id="150",
    )

    with pytest.raises(type(failure)):
        await handler.handle_message(incoming)

    mock_session_store.record_message_id.assert_called_once_with(
        incoming.platform,
        incoming.chat_id,
        incoming.message_id,
        "in",
        "command",
    )
    mock_platform.queue_delete_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_notice_is_published_and_recorded_for_clear(
    handler,
    mock_platform,
    mock_session_store,
) -> None:
    mock_platform.queue_send_message.return_value = "startup_1"

    await handler.publish_startup_notice(_startup_notice())

    mock_platform.queue_send_message.assert_awaited_once_with(
        _SCOPE.chat_id,
        "🚀 *Claude Code Proxy is online\\!* \\(Bot API\\)",
        parse_mode="MarkdownV2",
        fire_and_forget=False,
    )
    mock_session_store.record_message_id.assert_called_once_with(
        _SCOPE.platform,
        _SCOPE.chat_id,
        "startup_1",
        "out",
        "startup",
    )
    mock_platform.queue_delete_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_notice_failure_is_nonfatal_and_records_nothing(
    handler,
    mock_platform,
    mock_session_store,
) -> None:
    mock_platform.queue_send_message.side_effect = RuntimeError("unavailable")

    await handler.publish_startup_notice(_startup_notice())

    mock_session_store.record_message_id.assert_not_called()
    mock_platform.queue_delete_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_notice_without_delivery_receipt_records_nothing(
    handler,
    mock_platform,
    mock_session_store,
) -> None:
    mock_platform.queue_send_message.return_value = None

    await handler.publish_startup_notice(_startup_notice())

    mock_session_store.record_message_id.assert_not_called()
    mock_platform.queue_delete_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_notice_record_failure_is_compensated_outside_state_lock(
    handler,
    mock_platform,
    mock_session_store,
) -> None:
    mock_platform.queue_send_message.return_value = "startup_1"
    mock_session_store.record_message_id.side_effect = OSError("store unavailable")

    async def delete_notice(*args, **kwargs) -> None:
        assert not handler._state_lock.locked()

    mock_platform.queue_delete_messages.side_effect = delete_notice

    await handler.publish_startup_notice(_startup_notice())

    mock_session_store.record_message_id.assert_called_once_with(
        _SCOPE.platform,
        _SCOPE.chat_id,
        "startup_1",
        "out",
        "startup",
    )
    mock_platform.queue_delete_messages.assert_awaited_once_with(
        _SCOPE.chat_id,
        ["startup_1"],
        fire_and_forget=False,
    )
    mock_session_store.forget_tracked_message_ids.assert_called_once_with(
        _SCOPE.platform,
        _SCOPE.chat_id,
        {"startup_1"},
    )


@pytest.mark.asyncio
async def test_failed_startup_notice_compensation_restores_clear_ownership(
    handler,
    mock_platform,
    mock_session_store,
) -> None:
    mock_platform.queue_send_message.return_value = "startup_1"
    mock_session_store.record_message_id.side_effect = (
        OSError("store unavailable"),
        None,
    )
    mock_platform.queue_delete_messages.side_effect = RuntimeError("delete unavailable")

    await handler.publish_startup_notice(_startup_notice())

    assert mock_session_store.record_message_id.call_count == 2
    assert mock_session_store.record_message_id.call_args_list == [
        call(
            _SCOPE.platform,
            _SCOPE.chat_id,
            "startup_1",
            "out",
            "startup",
        ),
        call(
            _SCOPE.platform,
            _SCOPE.chat_id,
            "startup_1",
            "out",
            "startup",
        ),
    ]
    mock_session_store.forget_tracked_message_ids.assert_not_called()


@pytest.mark.asyncio
async def test_startup_notice_publication_propagates_cancellation(
    handler,
    mock_platform,
    mock_session_store,
) -> None:
    send_started = asyncio.Event()
    send_cancelled = asyncio.Event()

    async def send_notice(*args, **kwargs) -> None:
        send_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            send_cancelled.set()

    mock_platform.queue_send_message.side_effect = send_notice
    task = asyncio.create_task(handler.publish_startup_notice(_startup_notice()))
    await send_started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await send_cancelled.wait()
    mock_session_store.record_message_id.assert_not_called()
    mock_platform.queue_delete_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_notice_cancellation_after_receipt_finishes_compensation(
    handler,
    mock_platform,
    mock_session_store,
) -> None:
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    finalizer_started = asyncio.Event()

    async def send_notice(*args, **kwargs) -> str:
        send_started.set()
        await release_send.wait()
        return "startup_1"

    original_finalize = handler._finalize_startup_notice

    async def finalize_notice(*args, **kwargs) -> None:
        finalizer_started.set()
        await original_finalize(*args, **kwargs)

    mock_platform.queue_send_message.side_effect = send_notice
    with patch.object(
        handler,
        "_finalize_startup_notice",
        side_effect=finalize_notice,
    ):
        task = asyncio.create_task(handler.publish_startup_notice(_startup_notice()))
        await send_started.wait()
        await handler._state_lock.acquire()
        try:
            release_send.set()
            await finalizer_started.wait()
        finally:
            handler._state_lock.release()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    mock_session_store.record_message_id.assert_not_called()
    mock_platform.queue_delete_messages.assert_awaited_once_with(
        _SCOPE.chat_id,
        ["startup_1"],
        fire_and_forget=False,
    )
    mock_session_store.forget_tracked_message_ids.assert_called_once_with(
        _SCOPE.platform,
        _SCOPE.chat_id,
        {"startup_1"},
    )


@pytest.mark.asyncio
async def test_concurrent_global_clear_does_not_wait_for_startup_delivery(
    mock_platform,
    mock_cli_manager,
    tmp_path,
) -> None:
    store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        store,
        platform_name="telegram",
        voice_cancellation=mock_platform,
    )
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def send_notice(*args, **kwargs) -> str:
        send_started.set()
        await release_send.wait()
        return "startup_1"

    mock_platform.queue_send_message.side_effect = send_notice
    publish_task = asyncio.create_task(
        workflow.publish_startup_notice(_startup_notice())
    )
    await send_started.wait()
    clear_task = asyncio.create_task(
        workflow.clear_chat(_SCOPE.platform, _SCOPE.chat_id)
    )
    try:
        done, _pending = await asyncio.wait({clear_task}, timeout=1)
        assert clear_task in done
        assert clear_task.result() == frozenset()
    finally:
        release_send.set()
        await asyncio.gather(publish_task, clear_task, return_exceptions=True)

    assert store.get_tracked_message_ids_for_chat(_SCOPE.platform, _SCOPE.chat_id) == []
    mock_platform.queue_delete_messages.assert_awaited_once_with(
        _SCOPE.chat_id,
        ["startup_1"],
        fire_and_forget=False,
    )


@pytest.mark.asyncio
async def test_global_stop_does_not_wait_for_or_invalidate_startup_delivery(
    handler,
    mock_platform,
    mock_session_store,
) -> None:
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def send_notice(*args, **kwargs) -> str:
        send_started.set()
        await release_send.wait()
        return "startup_1"

    mock_platform.queue_send_message.side_effect = send_notice
    publish_task = asyncio.create_task(
        handler.publish_startup_notice(_startup_notice())
    )
    await send_started.wait()
    stop_task = asyncio.create_task(handler.stop_all_tasks())
    try:
        done, _pending = await asyncio.wait({stop_task}, timeout=1)
        assert stop_task in done
        stop_task.result()
    finally:
        release_send.set()
        await asyncio.gather(publish_task, stop_task, return_exceptions=True)

    mock_session_store.record_message_id.assert_called_once_with(
        _SCOPE.platform,
        _SCOPE.chat_id,
        "startup_1",
        "out",
        "startup",
    )
    mock_platform.queue_delete_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_clear_precedes_concurrent_startup_notice_publication(
    mock_platform,
    mock_cli_manager,
    tmp_path,
) -> None:
    store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        store,
        platform_name="telegram",
        voice_cancellation=mock_platform,
    )
    clear_started = asyncio.Event()
    release_clear = asyncio.Event()

    async def clear_trees(
        scope: MessageScope, *, reason: CancellationReason
    ) -> CancellationResult:
        assert reason is CancellationReason.CLEAR
        clear_started.set()
        await release_clear.wait()
        return CancellationResult()

    with patch.object(workflow.tree_queue, "clear_scope", side_effect=clear_trees):
        clear_task = asyncio.create_task(
            workflow.clear_chat(_SCOPE.platform, _SCOPE.chat_id)
        )
        await clear_started.wait()
        publish_task = asyncio.create_task(
            workflow.publish_startup_notice(_startup_notice())
        )
        await asyncio.sleep(0)
        mock_platform.queue_send_message.assert_not_awaited()

        release_clear.set()

        assert await clear_task == frozenset()
        await publish_task

    assert store.get_tracked_message_ids_for_chat(_SCOPE.platform, _SCOPE.chat_id) == [
        "msg_123"
    ]
    mock_platform.queue_delete_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_command_cannot_evict_startup_notice_at_managed_message_cap(
    mock_platform,
    mock_cli_manager,
    incoming_message_factory,
    tmp_path,
) -> None:
    store = SessionStore(
        storage_path=str(tmp_path / "sessions.json"),
        managed_message_cap=1,
    )
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        store,
        platform_name="telegram",
        voice_cancellation=mock_platform,
    )
    store.record_message_id(
        _SCOPE.platform,
        _SCOPE.chat_id,
        "100",
        "out",
        "startup",
    )
    incoming = incoming_message_factory(
        text="/clear",
        chat_id=_SCOPE.chat_id,
        message_id="101",
    )

    await workflow.handle_message(incoming)

    mock_platform.queue_delete_messages.assert_awaited_once_with(
        _SCOPE.chat_id,
        ["101", "100"],
        fire_and_forget=False,
    )


@pytest.mark.asyncio
async def test_clear_chat_is_fully_scoped_to_the_invoking_chat(
    handler, mock_cli_manager, mock_session_store, incoming_message_factory
):
    root_1 = incoming_message_factory(
        text="one",
        chat_id="chat_1",
        message_id="100",
    )
    root_2 = incoming_message_factory(
        text="two",
        chat_id="chat_2",
        message_id="200",
    )
    await handler.tree_queue.admit(root_1, "101")
    await handler.tree_queue.admit(root_2, "201")
    await _wait_for_idle(handler)
    mock_session_store.get_tracked_message_ids_for_chat.return_value = ["42"]
    mock_session_store.reset_mock()
    mock_session_store.get_tracked_message_ids_for_chat.return_value = ["42"]

    message_ids = await handler.clear_chat("telegram", "chat_1")

    assert message_ids == frozenset({"42", "100", "101"})
    assert "200" not in message_ids
    assert handler.get_tree_count() == 1
    assert await handler.tree_queue.get_node(root_2.scope, "200") is not None
    mock_cli_manager.stop_all.assert_not_awaited()
    mock_session_store.clear_scope.assert_called_once_with(_SCOPE)


@pytest.mark.asyncio
async def test_global_clear_cancels_and_deletes_only_current_chat_voices(
    handler,
    mock_platform,
    mock_session_store,
) -> None:
    events: list[str] = []
    current = VoiceCancellationResult(
        scope=_SCOPE,
        voice_message_id="voice",
        status_message_id="voice_status",
        delete_message_ids=frozenset({"voice", "voice_status"}),
    )

    async def cancel_voices(scope: MessageScope) -> tuple[VoiceCancellationResult, ...]:
        assert not handler._state_lock.locked()
        assert scope == _SCOPE
        events.append("voices")
        return (current,)

    async def clear_trees(
        scope: MessageScope, *, reason: CancellationReason
    ) -> CancellationResult:
        assert reason is CancellationReason.CLEAR
        assert handler._state_lock.locked()
        events.append("trees")
        return CancellationResult()

    mock_platform.cancel_pending_voices_in_scope.side_effect = cancel_voices
    mock_session_store.get_tracked_message_ids_for_chat.return_value = ["stored"]
    with patch.object(handler.tree_queue, "clear_scope", side_effect=clear_trees):
        message_ids = await handler.clear_chat("telegram", "chat_1")
    await asyncio.sleep(0)

    assert events == ["voices", "trees"]
    assert message_ids == frozenset({"stored", "voice", "voice_status"})
    assert "other_voice" not in message_ids
    assert "other_status" not in message_ids
    mock_platform.queue_edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_clear_persists_after_tree_detach(
    handler,
    mock_cli_manager,
    mock_session_store,
) -> None:
    events: list[str] = []

    async def clear_trees(
        scope: MessageScope, *, reason: CancellationReason
    ) -> CancellationResult:
        assert scope == _SCOPE
        assert reason is CancellationReason.CLEAR
        events.append("trees.clear_scope")
        return CancellationResult()

    mock_session_store.clear_scope.side_effect = lambda scope: events.append(
        f"store.clear_scope:{scope.chat_id}"
    )

    with patch.object(handler.tree_queue, "clear_scope", side_effect=clear_trees):
        result = await handler.clear_chat("telegram", "chat_1")

    assert result == frozenset()
    assert events == ["trees.clear_scope", "store.clear_scope:chat_1"]
    mock_cli_manager.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_clear_finishes_cleanup_before_final_persistence_error_escapes(
    handler,
    mock_cli_manager,
    mock_session_store,
) -> None:
    events: list[str] = []

    async def clear_trees(
        scope: MessageScope, *, reason: CancellationReason
    ) -> CancellationResult:
        assert scope == _SCOPE
        assert reason is CancellationReason.CLEAR
        events.append("trees.clear_scope")
        return CancellationResult()

    def fail_final_clear(scope: MessageScope) -> None:
        assert scope == _SCOPE
        events.append("store.clear_scope")
        raise OSError("final clear failure")

    mock_session_store.clear_scope.side_effect = fail_final_clear

    with (
        patch.object(handler.tree_queue, "clear_scope", side_effect=clear_trees),
        pytest.raises(OSError, match="final clear failure"),
    ):
        await handler.clear_chat("telegram", "chat_1")

    assert events == ["trees.clear_scope", "store.clear_scope"]
    mock_cli_manager.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_clear_preserves_tree_and_persistence_failures(
    handler,
    mock_cli_manager,
    mock_session_store,
) -> None:
    events: list[str] = []
    persistence_error = OSError("final clear failure")
    tree_error = RuntimeError("tree clear failure")

    async def fail_tree_clear(
        scope: MessageScope, *, reason: CancellationReason
    ) -> CancellationResult:
        assert scope == _SCOPE
        assert reason is CancellationReason.CLEAR
        events.append("trees.clear_scope")
        raise tree_error

    def fail_final_clear(scope: MessageScope) -> None:
        assert scope == _SCOPE
        events.append("store.clear_scope")
        raise persistence_error

    mock_session_store.clear_scope.side_effect = fail_final_clear

    with (
        patch.object(handler.tree_queue, "clear_scope", side_effect=fail_tree_clear),
        pytest.raises(ExceptionGroup) as raised,
    ):
        await handler.clear_chat("telegram", "chat_1")

    assert raised.value.exceptions == (tree_error, persistence_error)
    assert events == ["trees.clear_scope", "store.clear_scope"]
    mock_cli_manager.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_committed_global_clear_attempts_remaining_steps_after_tree_failure(
    handler,
    mock_cli_manager,
    mock_session_store,
) -> None:
    events: list[str] = []
    tree_error = RuntimeError("tree clear failure")

    async def fail_tree_clear(
        scope: MessageScope, *, reason: CancellationReason
    ) -> CancellationResult:
        assert scope == _SCOPE
        assert reason is CancellationReason.CLEAR
        events.append("trees.clear_scope")
        raise tree_error

    mock_session_store.clear_scope.side_effect = lambda scope: events.append(
        f"store.clear_scope:{scope.chat_id}"
    )

    with (
        patch.object(handler.tree_queue, "clear_scope", side_effect=fail_tree_clear),
        pytest.raises(RuntimeError, match="tree clear failure") as raised,
    ):
        await handler.clear_chat("telegram", "chat_1")

    assert raised.value is tree_error
    assert events == [
        "trees.clear_scope",
        "store.clear_scope:chat_1",
    ]
    mock_cli_manager.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_global_clear_finishes_owned_transaction_before_propagating(
    handler,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
):
    root = incoming_message_factory(text="work", message_id="100")
    await handler.tree_queue.admit(root, "101")
    await _wait_for_idle(handler)
    id_read_started = asyncio.Event()
    release_id_read = asyncio.Event()

    async def block_id_read(platform: str, chat_id: str) -> set[str]:
        id_read_started.set()
        await release_id_read.wait()
        return {"100", "101"}

    mock_session_store.reset_mock()
    with patch.object(
        handler.tree_queue,
        "get_message_ids_for_chat",
        new=block_id_read,
    ):
        clear_task = asyncio.create_task(handler.clear_chat("telegram", root.chat_id))
        await id_read_started.wait()
        clear_task.cancel()
        await asyncio.sleep(0)
        assert not clear_task.done()
        release_id_read.set()
        with pytest.raises(asyncio.CancelledError):
            await clear_task

    assert handler._clear_generations[_SCOPE] == 1
    assert handler.get_tree_count() == 0
    mock_session_store.clear_scope.assert_called_once_with(_SCOPE)
    mock_cli_manager.stop_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_clear_with_mention_uses_same_global_command(
    handler, mock_platform, incoming_message_factory
):
    handler.clear_chat = AsyncMock(return_value=frozenset())
    incoming = incoming_message_factory(
        text="/clear@MyBot",
        chat_id="chat_1",
        message_id="10",
    )

    await handler.handle_message(incoming)

    handler.clear_chat.assert_awaited_once_with("telegram", "chat_1")
    mock_platform.queue_delete_messages.assert_awaited_once_with(
        "chat_1",
        ["10"],
        fire_and_forget=False,
    )


@pytest.mark.asyncio
async def test_clear_continues_after_platform_delete_failure(
    handler, mock_platform, incoming_message_factory
):
    handler.clear_chat = AsyncMock(return_value=frozenset({"41", "42"}))
    mock_platform.queue_delete_messages.side_effect = RuntimeError(
        "platform rejected delete"
    )

    await handler.handle_message(
        incoming_message_factory(text="/clear", message_id="150")
    )

    handler.clear_chat.assert_awaited_once()
    mock_platform.queue_delete_messages.assert_awaited_once()


@pytest.mark.asyncio
async def test_reply_clear_status_preserves_prompt_and_persists_remaining_tree(
    handler,
    mock_platform,
    mock_session_store,
    incoming_message_factory,
):
    root = incoming_message_factory(text="root", message_id="100")
    child = incoming_message_factory(
        text="child",
        message_id="102",
        reply_to_message_id="100",
    )
    await handler.tree_queue.admit(root, "101")
    await _wait_for_idle(handler)
    await handler.tree_queue.admit(child, "103", parent_reference_id="100")
    await _wait_for_idle(handler)
    mock_session_store.reset_mock()
    deleted_ids: list[str] = []

    async def capture_delete(chat_id, message_ids, fire_and_forget=True):
        deleted_ids.extend(message_ids)

    mock_platform.queue_delete_messages.side_effect = capture_delete
    await handler.handle_message(
        incoming_message_factory(
            text="/clear",
            message_id="150",
            reply_to_message_id="103",
        )
    )

    assert set(deleted_ids) == {"103", "150"}
    assert "102" not in deleted_ids
    assert "100" not in deleted_ids
    assert "101" not in deleted_ids
    cleared_prompt = await handler.tree_queue.get_node(root.scope, "102")
    assert cleared_prompt is not None
    assert cleared_prompt.state is MessageState.ERROR
    assert cleared_prompt.session_id is None
    assert await handler.tree_queue.get_node(root.scope, "100") is not None
    mock_session_store.save_tree_snapshot.assert_called_once()
    mock_session_store.record_message_id.assert_called_once_with(
        "telegram",
        "chat_1",
        "150",
        "in",
        "command",
    )
    mock_session_store.forget_tracked_message_ids.assert_called_once_with(
        "telegram",
        "chat_1",
        {"103", "150"},
    )


@pytest.mark.asyncio
async def test_reply_clear_unknown_reports_nothing_to_clear(
    handler, mock_platform, mock_session_store, incoming_message_factory
):
    incoming = incoming_message_factory(
        text="/clear",
        message_id="150",
        reply_to_message_id="999",
    )

    await handler.handle_message(incoming)

    assert "Nothing to clear" in mock_platform.queue_send_message.call_args.args[1]
    mock_session_store.record_message_id.assert_any_call(
        incoming.platform,
        incoming.chat_id,
        incoming.message_id,
        "in",
        "command",
    )
    mock_session_store.clear_scope.assert_not_called()


@pytest.mark.asyncio
async def test_reply_clear_root_removes_tree_snapshot(
    handler,
    mock_platform,
    mock_session_store,
    incoming_message_factory,
):
    root = incoming_message_factory(text="root", message_id="100")
    await handler.tree_queue.admit(root, "101")
    await _wait_for_idle(handler)
    mock_session_store.reset_mock()
    deleted_ids: list[str] = []

    async def capture_delete(chat_id, message_ids, fire_and_forget=True):
        deleted_ids.extend(message_ids)

    mock_platform.queue_delete_messages.side_effect = capture_delete
    await handler.handle_message(
        incoming_message_factory(
            text="/clear",
            message_id="150",
            reply_to_message_id="100",
        )
    )

    assert set(deleted_ids) == {"100", "101", "150"}
    mock_session_store.remove_tree_snapshot.assert_called_once_with(
        TreeIdentity(scope=root.scope, root_id="100")
    )
    assert handler.get_tree_count() == 0


@pytest.mark.asyncio
async def test_late_cancelled_runner_cannot_save_or_render_after_chat_clear(
    handler,
    mock_platform,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
):
    started = asyncio.Event()

    async def blocking_start(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)
        if False:
            yield {}

    session = MagicMock()
    session.start_task = blocking_start
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "pending_1",
        True,
    )
    await handler.handle_message(
        incoming_message_factory(text="work", message_id="100")
    )
    await started.wait()
    mock_session_store.reset_mock()
    mock_platform.queue_edit_message.reset_mock()

    await handler.clear_chat("telegram", "chat_1")

    mock_session_store.save_tree_snapshot.assert_not_called()
    mock_session_store.clear_scope.assert_called_once_with(_SCOPE)
    mock_platform.queue_edit_message.assert_not_awaited()
    assert handler.get_tree_count() == 0


@pytest.mark.asyncio
async def test_global_clear_removes_snapshot_saved_during_detach_window(
    tmp_path,
    mock_platform,
    mock_cli_manager,
    incoming_message_factory,
):
    runner_started = asyncio.Event()
    release_runner = asyncio.Event()
    id_read_started = asyncio.Event()
    release_id_read = asyncio.Event()

    async def finish_after_release(*args, **kwargs):
        runner_started.set()
        await release_runner.wait()
        yield {"type": "exit", "code": 0}

    session = MagicMock()
    session.start_task = finish_after_release
    mock_cli_manager.get_or_create_session.return_value = (
        session,
        "session_1",
        False,
    )
    mock_platform.queue_send_message.return_value = "status-new"
    store_path = tmp_path / "sessions.json"
    store = SessionStore(storage_path=str(store_path))
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        store,
        platform_name="telegram",
    )
    incoming = incoming_message_factory(text="work", message_id="new")
    await workflow.handle_message(incoming)
    await runner_started.wait()

    get_ids = workflow.tree_queue.get_message_ids_for_chat

    async def block_id_read(platform: str, chat_id: str) -> set[str]:
        id_read_started.set()
        await release_id_read.wait()
        return await get_ids(platform, chat_id)

    try:
        with patch.object(
            workflow.tree_queue,
            "get_message_ids_for_chat",
            new=block_id_read,
        ):
            clear_task = asyncio.create_task(
                workflow.clear_chat("telegram", incoming.chat_id)
            )
            await id_read_started.wait()
            release_runner.set()
            await _wait_for_idle(workflow)

            assert not store.load_conversation_snapshot().is_empty
            store.flush_pending_save()
            assert (
                not SessionStore(storage_path=str(store_path))
                .load_conversation_snapshot()
                .is_empty
            )
            release_id_read.set()
            await clear_task

        assert store.load_conversation_snapshot().is_empty
        assert (
            SessionStore(storage_path=str(store_path))
            .load_conversation_snapshot()
            .is_empty
        )
    finally:
        release_runner.set()
        release_id_read.set()
        await workflow.close()


@pytest.mark.asyncio
async def test_global_clear_invalidates_inflight_prompt_without_waiting_for_status(
    handler,
    mock_platform,
    incoming_message_factory,
):
    status_send_started = asyncio.Event()
    release_status_send = asyncio.Event()

    async def block_status_send(*args, **kwargs):
        status_send_started.set()
        await release_status_send.wait()
        return "status-new"

    mock_platform.queue_send_message.side_effect = block_status_send
    prompt_task = asyncio.create_task(
        handler.handle_message(
            incoming_message_factory(text="new prompt", message_id="new")
        )
    )
    await status_send_started.wait()
    clear_task = asyncio.create_task(handler.clear_chat("telegram", "chat_1"))

    try:
        await asyncio.wait_for(clear_task, timeout=1)
        release_status_send.set()
        await prompt_task
        assert handler.get_tree_count() == 0
        mock_platform.queue_delete_messages.assert_awaited_once_with(
            "chat_1",
            ["status-new"],
            fire_and_forget=False,
        )
    finally:
        release_status_send.set()
        if not prompt_task.done():
            prompt_task.cancel()
        if not clear_task.done():
            clear_task.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_message_id", "expected_deleted_ids"),
    [
        ("101", {"100", "101", "150"}),
        (None, {"100", "150"}),
    ],
)
async def test_reply_clear_pending_voice_cancels_and_reports(
    handler,
    mock_platform,
    incoming_message_factory,
    status_message_id,
    expected_deleted_ids,
) -> None:
    delete_message_ids = {"100"}
    if status_message_id is not None:
        delete_message_ids.add(status_message_id)
    handler.clear_reply = AsyncMock(
        return_value=ReplyClearResult(
            delete_message_ids=frozenset(delete_message_ids),
            tree_matched=False,
        )
    )
    deleted_ids: list[str] = []

    async def capture_delete(chat_id, message_ids, fire_and_forget=True):
        deleted_ids.extend(message_ids)

    mock_platform.queue_delete_messages.side_effect = capture_delete
    incoming = incoming_message_factory(
        text="/clear",
        message_id="150",
        reply_to_message_id="100",
    )

    await handler.handle_message(incoming)

    handler.clear_reply.assert_awaited_once_with(incoming.scope, "100")
    assert set(deleted_ids) == expected_deleted_ids
    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_clear_deletes_owned_result_ids(
    handler,
    mock_platform,
    incoming_message_factory,
) -> None:
    handler.clear_reply = AsyncMock(
        return_value=ReplyClearResult(
            delete_message_ids=frozenset({"tree_status"}),
            tree_matched=True,
        )
    )
    deleted_ids: list[str] = []

    async def capture_delete(_chat_id, message_ids, fire_and_forget=True):
        deleted_ids.extend(message_ids)

    mock_platform.queue_delete_messages.side_effect = capture_delete
    incoming = incoming_message_factory(
        text="/clear",
        message_id="clear",
        reply_to_message_id="voice",
    )

    await handler.handle_message(incoming)

    handler.clear_reply.assert_awaited_once_with(incoming.scope, "voice")
    assert set(deleted_ids) == {
        "tree_status",
        "clear",
    }
    assert "voice" not in deleted_ids
    mock_platform.queue_send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_clear_joins_voice_then_unions_authoritative_branch_ids(
    handler,
    mock_platform,
) -> None:
    events: list[str] = []

    async def cancel_voice(*_args) -> VoiceCancellationResult:
        events.append("voice")
        return VoiceCancellationResult(
            scope=_SCOPE,
            voice_message_id="voice",
            status_message_id="voice_status",
            delete_message_ids=frozenset({"voice", "voice_status"}),
        )

    branch = MessageSubtreeRemovalResult(
        cancellation=CancellationResult(),
        removed_tree_identity=None,
        delete_message_ids=frozenset({"tree_status"}),
        tree_matched=True,
    )
    mock_platform.cancel_pending_voice.side_effect = cancel_voice
    with patch.object(
        handler.tree_queue,
        "remove_message_subtree",
        new_callable=AsyncMock,
        return_value=branch,
    ) as remove_message_subtree:
        result = await handler.clear_reply(_SCOPE, "voice")
    await asyncio.sleep(0)

    assert result == ReplyClearResult(
        delete_message_ids=frozenset({"voice", "voice_status", "tree_status"}),
        tree_matched=True,
    )
    assert events == ["voice"]
    remove_message_subtree.assert_awaited_once_with(
        _SCOPE,
        "voice",
        reason=CancellationReason.CLEAR,
    )
    mock_platform.queue_edit_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_reply_clear_finishes_voice_only_owner_after_lock_wait(
    handler,
    mock_platform,
) -> None:
    voice_returned = asyncio.Event()

    async def cancel_voice(*_args) -> VoiceCancellationResult:
        voice_returned.set()
        return VoiceCancellationResult(
            scope=_SCOPE,
            voice_message_id="voice",
            status_message_id="voice_status",
            delete_message_ids=frozenset({"voice", "voice_status"}),
        )

    mock_platform.cancel_pending_voice.side_effect = cancel_voice
    remove_subtree = AsyncMock(
        return_value=MessageSubtreeRemovalResult(
            cancellation=CancellationResult(),
            removed_tree_identity=None,
            delete_message_ids=frozenset(),
            tree_matched=False,
        )
    )
    await handler._state_lock.acquire()
    try:
        with patch.object(
            handler.tree_queue,
            "remove_message_subtree",
            remove_subtree,
        ):
            clear_task = asyncio.create_task(handler.clear_reply(_SCOPE, "voice"))
            await voice_returned.wait()
            await asyncio.sleep(0)
            clear_task.cancel()
            await asyncio.sleep(0)

            assert not clear_task.done()
            handler._state_lock.release()
            with pytest.raises(asyncio.CancelledError):
                await clear_task
    finally:
        if handler._state_lock.locked():
            handler._state_lock.release()
    await asyncio.sleep(0)

    remove_subtree.assert_awaited_once_with(
        _SCOPE,
        "voice",
        reason=CancellationReason.CLEAR,
    )
    mock_platform.queue_edit_message.assert_not_awaited()

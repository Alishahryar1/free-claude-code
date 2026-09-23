import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.messaging.command_context import StopOutcome
from free_claude_code.messaging.models import IncomingMessage, MessageScope
from free_claude_code.messaging.node_event_pipeline import process_parsed_cli_event
from free_claude_code.messaging.rendering.telegram_markdown import (
    escape_md_v2,
    render_markdown_to_mdv2,
)
from free_claude_code.messaging.trees import (
    CancellationReason,
    CancellationResult,
    CancellationUiOwner,
    FailureResult,
    MessageReferenceKind,
    NodeClaim,
    NodeUiTarget,
    QueueDecision,
    QueueEntry,
    ReplyTarget,
    TreeIdentity,
    TreeSnapshot,
)
from free_claude_code.messaging.trees.transitions import CancellationEffect
from free_claude_code.messaging.workflow import MessagingWorkflow

_SCOPE = MessageScope(platform="telegram", chat_id="c")


def _claim() -> NodeClaim:
    return NodeClaim(
        identity=TreeIdentity(scope=_SCOPE, root_id="root"),
        claim_id="claim-1",
        node=NodeUiTarget(
            scope=_SCOPE,
            node_id="n1",
            status_message_id="s1",
        ),
        prompt="hi",
        parent_session_id=None,
    )


def _snapshot(marker: str) -> TreeSnapshot:
    return TreeSnapshot(
        scope=_SCOPE,
        root_id="root",
        nodes={"root": {"marker": marker}},
    )


def _decision(*, position: int | None = None) -> QueueDecision:
    return QueueDecision(
        claim=_claim() if position is None else None,
        position=position,
        snapshot=_snapshot("admitted"),
    )


def test_render_markdown_to_mdv2_empty_returns_empty():
    assert render_markdown_to_mdv2("") == ""


def test_render_markdown_to_mdv2_covers_common_structures():
    md = (
        "# Heading\n\n"
        "Text with *em* and **strong** and ~~strike~~ and `code`.\n\n"
        "- item1\n"
        "- item2\n\n"
        "3. third\n\n"
        "> quote\n\n"
        "[link](http://example.com/a\\)b)\n\n"
        "![alt](http://example.com/img.png)\n\n"
        "```python\nprint('x')\n```\n"
    )
    out = render_markdown_to_mdv2(md)
    assert "*Heading*" in out
    assert "_em_" in out
    assert "*strong*" in out
    assert "~strike~" in out
    assert "`code`" in out
    assert "\\- item1" in out
    assert "3\\." in out
    assert "> quote" in out
    assert "[link]" in out
    assert escape_md_v2("alt (http://example.com/img.png)") in out
    assert "```" in out


def test_render_markdown_to_mdv2_escapes_image_fallback_as_plain_text():
    out = render_markdown_to_mdv2("![alt](http://x.com/a_b.png)")

    assert out == escape_md_v2("alt (http://x.com/a_b.png)")
    assert "(" not in out.replace("\\(", "")
    assert ")" not in out.replace("\\)", "")
    assert "." not in out.replace("\\.", "")
    assert "_" not in out.replace("\\_", "")


def test_render_markdown_to_mdv2_escapes_image_without_alt_text():
    out = render_markdown_to_mdv2("![](http://x.com/a-b.png)")

    assert out == escape_md_v2("http://x.com/a-b.png")
    assert "." not in out.replace("\\.", "")
    assert "-" not in out.replace("\\-", "")


def test_render_markdown_to_mdv2_escapes_image_embedded_in_text():
    out = render_markdown_to_mdv2("see ![d](https://e.com/d-1.png) here")

    assert escape_md_v2("d (https://e.com/d-1.png)") in out


def test_render_markdown_to_mdv2_renders_table_as_code_block():
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nAfter.\n"
    out = render_markdown_to_mdv2(md)
    assert "```" in out
    assert "| a" in out
    assert "| b" in out
    assert "| ---" in out
    assert "After" in out


def test_render_markdown_to_mdv2_table_without_blank_line_still_renders():
    md = "Here's a table:\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    out = render_markdown_to_mdv2(md)
    assert "Here's a table" in out
    assert "```" in out
    assert "| a" in out
    assert "| ---" in out


def test_render_markdown_to_mdv2_table_escapes_backticks_and_backslashes_in_cells():
    md = "| a | b |\n|---|---|\n| \\\\ | `` ` `` |\n"
    out = render_markdown_to_mdv2(md)
    assert "```" in out
    # In Telegram code blocks we escape backslashes and backticks.
    assert "\\\\" in out  # rendered cell backslash becomes double-backslash
    assert "\\`" in out  # rendered cell backtick is escaped


def test_render_markdown_to_mdv2_table_inside_list_keeps_bullet_prefix():
    md = "-\n  | a | b |\n  |---|---|\n  | 1 | 2 |\n"
    out = render_markdown_to_mdv2(md)
    assert "```" in out
    assert out.lstrip().startswith("\\-")
    assert out.find("\\-") < out.find("```")


def test_get_initial_status_branches():
    platform = MagicMock()
    cli_manager = MagicMock()
    session_store = MagicMock()
    handler = MessagingWorkflow(platform, cli_manager, session_store)

    s1 = handler.turn_intake._get_initial_status(
        ReplyTarget(
            node_id="p",
            reference_id="p",
            reference_kind=MessageReferenceKind.PROMPT,
            queue_position=3,
        )
    )
    assert "Queued" in s1
    assert "position 3" in s1 or "position 3" in s1.replace("\\", "")

    s2 = handler.turn_intake._get_initial_status(
        ReplyTarget(
            node_id="p",
            reference_id="status-p",
            reference_kind=MessageReferenceKind.STATUS,
            queue_position=None,
        )
    )
    assert "Continuing" in s2

    s3 = handler.turn_intake._get_initial_status(None)
    assert "Launching" in s3


@pytest.mark.asyncio
async def test_update_queue_positions_renders_immutable_queue_entries():
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock()
    platform.fire_and_forget = MagicMock(
        side_effect=lambda c: getattr(c, "close", lambda: None)()
    )

    cli_manager = MagicMock()
    session_store = MagicMock()
    handler = MessagingWorkflow(platform, cli_manager, session_store)

    await handler.turn_intake.update_queue_positions(())
    platform.fire_and_forget.assert_not_called()

    await handler.turn_intake.update_queue_positions(
        (
            QueueEntry(
                node=NodeUiTarget(
                    scope=_SCOPE,
                    node_id="n1",
                    status_message_id="s",
                ),
                position=2,
            ),
        )
    )
    assert platform.fire_and_forget.call_count == 1
    assert "position 2" in platform.queue_edit_message.call_args.args[2]


@pytest.mark.asyncio
async def test_node_runner_process_node_session_limit_marks_error_and_updates_ui():
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock()
    platform.fire_and_forget = MagicMock(
        side_effect=lambda c: getattr(c, "close", lambda: None)()
    )

    cli_manager = MagicMock()
    cli_manager.get_or_create_session = AsyncMock(side_effect=RuntimeError("limit"))
    cli_manager.get_stats.return_value = {"active_sessions": 0}

    session_store = MagicMock()
    handler = MessagingWorkflow(platform, cli_manager, session_store)

    claim = _claim()
    snapshot = _snapshot("error")
    fail_claim = AsyncMock(
        return_value=FailureResult(affected=(), queue_update=None, snapshot=snapshot)
    )
    with patch.object(
        handler.tree_queue,
        "fail_claim",
        fail_claim,
    ):
        await handler.node_runner.process_node(claim)
    assert platform.queue_edit_message.await_count >= 1
    fail_claim.assert_awaited_once_with(
        claim,
        propagate=False,
    )
    session_store.save_tree_snapshot.assert_called_once_with(snapshot)


@pytest.mark.asyncio
async def test_node_runner_cancellation_marks_error_and_saves_tree():
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock()
    platform.fire_and_forget = MagicMock(
        side_effect=lambda c: getattr(c, "close", lambda: None)()
    )

    async def _cancelled_start_task(*args, **kwargs):
        raise asyncio.CancelledError
        yield

    mock_session = MagicMock()
    mock_session.start_task = _cancelled_start_task
    cli_manager = MagicMock()
    cli_manager.get_or_create_session = AsyncMock(
        return_value=(mock_session, "s1", False)
    )
    cli_manager.remove_session = AsyncMock()
    cli_manager.get_stats.return_value = {"active_sessions": 0}

    session_store = MagicMock()
    handler = MessagingWorkflow(platform, cli_manager, session_store)

    claim = _claim()
    snapshot = _snapshot("cancelled")
    fail_claim = AsyncMock(
        return_value=FailureResult(affected=(), queue_update=None, snapshot=snapshot)
    )
    with patch.object(
        handler.tree_queue,
        "fail_claim",
        fail_claim,
    ):
        await handler.node_runner.process_node(claim)

    fail_claim.assert_awaited_once_with(
        claim,
        propagate=False,
    )
    session_store.save_tree_snapshot.assert_called_once_with(snapshot)


@pytest.mark.asyncio
async def test_stop_all_tasks_saves_tree_for_cancelled_nodes():
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock()
    platform.fire_and_forget = MagicMock(
        side_effect=lambda c: getattr(c, "close", lambda: None)()
    )

    cli_manager = MagicMock()
    cli_manager.stop_all = AsyncMock()
    cli_manager.get_stats.return_value = {"active_sessions": 0}

    session_store = MagicMock()
    handler = MessagingWorkflow(platform, cli_manager, session_store)

    snapshot = _snapshot("ok")
    result = CancellationResult(
        effects=(
            CancellationEffect(
                node=_claim().node,
                ui_owner=CancellationUiOwner.RUNNER,
            ),
        ),
        snapshots=(snapshot,),
    )
    cancel_all = AsyncMock(return_value=result)
    with patch.object(
        handler.tree_queue,
        "cancel_all",
        cancel_all,
    ):
        outcome = await handler.stop_all_tasks()
    assert outcome == StopOutcome(
        cancelled_count=1,
        status_feedback_scopes=frozenset({_SCOPE}),
        fallback_required=False,
    )
    cancel_all.assert_awaited_once_with(reason=CancellationReason.STOP)
    cli_manager.stop_all.assert_awaited_once()
    session_store.save_tree_snapshot.assert_called_once_with(snapshot)


@pytest.mark.asyncio
async def test_handle_message_unresolved_reply_is_admitted_as_new():
    platform = MagicMock()
    platform.queue_send_message = AsyncMock(return_value="status_1")
    platform.queue_edit_message = AsyncMock()

    cli_manager = MagicMock()
    cli_manager.get_stats.return_value = {"active_sessions": 0}

    session_store = MagicMock()
    handler = MessagingWorkflow(platform, cli_manager, session_store)

    resolve_reply = AsyncMock(return_value=None)
    admit = AsyncMock(return_value=_decision())

    incoming = IncomingMessage(
        text="reply",
        chat_id="c",
        user_id="u",
        message_id="m1",
        platform="telegram",
        reply_to_message_id="some_reply",
    )

    with (
        patch.object(handler.tree_queue, "resolve_reply", resolve_reply),
        patch.object(handler.tree_queue, "admit", admit),
    ):
        await handler.handle_message(incoming)

    resolve_reply.assert_awaited_once_with(incoming.scope, "some_reply")
    admit.assert_awaited_once_with(
        incoming,
        "status_1",
        parent_reference_id=None,
    )


@pytest.mark.asyncio
async def test_update_ui_handles_transcript_render_exception():
    """When transcript.render raises, update_ui catches and does not crash."""
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock()
    platform.fire_and_forget = MagicMock(
        side_effect=lambda c: getattr(c, "close", lambda: None)()
    )

    cli_manager = MagicMock()
    session_store = MagicMock()

    async def _mock_start_task(*args, **kwargs):
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        }
        yield {"type": "complete", "status": "success"}

    mock_session = MagicMock()
    mock_session.start_task = _mock_start_task
    cli_manager.get_or_create_session = AsyncMock(
        return_value=(mock_session, "s1", False)
    )
    cli_manager.remove_session = AsyncMock()
    cli_manager.get_stats.return_value = {"active_sessions": 0}

    handler = MessagingWorkflow(platform, cli_manager, session_store)
    claim = _claim()
    snapshot = _snapshot("complete")

    with (
        patch.object(
            handler.node_runner, "_create_transcript_and_render_ctx"
        ) as mock_create,
        patch.object(
            handler.tree_queue,
            "complete_claim",
            AsyncMock(return_value=snapshot),
        ),
    ):
        transcript = MagicMock()
        transcript.render = MagicMock(side_effect=ValueError("render failed"))
        render_ctx = MagicMock()
        mock_create.return_value = (transcript, render_ctx)

        await handler.node_runner.process_node(claim)

    assert transcript.render.call_count >= 1


@pytest.mark.asyncio
async def test_handle_message_incoming_text_none_safe():
    """handle_message does not crash when incoming.text is None (e.g. malformed adapter)."""
    platform = MagicMock()
    platform.queue_send_message = AsyncMock(return_value="status_1")
    platform.queue_edit_message = AsyncMock()

    cli_manager = MagicMock()
    cli_manager.get_stats.return_value = {"active_sessions": 0}

    session_store = MagicMock()
    handler = MessagingWorkflow(platform, cli_manager, session_store)
    admit = AsyncMock(return_value=_decision())

    incoming = MagicMock()
    incoming.text = None
    incoming.chat_id = "c"
    incoming.user_id = "u"
    incoming.message_id = "m1"
    incoming.platform = "telegram"
    incoming.reply_to_message_id = None
    incoming.status_message_id = None
    incoming.message_thread_id = None
    incoming.is_reply = MagicMock(return_value=False)

    with patch.object(handler.tree_queue, "admit", admit):
        await handler.handle_message(incoming)
    admit.assert_awaited_once_with(
        incoming,
        "status_1",
        parent_reference_id=None,
    )


@pytest.mark.asyncio
async def test_process_parsed_event_malformed_content_continues():
    """Malformed/unknown parsed event does not crash process_parsed_cli_event."""
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock()

    cli_manager = MagicMock()
    session_store = MagicMock()
    handler = MessagingWorkflow(platform, cli_manager, session_store)

    transcript = MagicMock()
    update_ui = AsyncMock()
    complete_claim = AsyncMock()
    fail_claim = AsyncMock()

    last_status, had = await process_parsed_cli_event(
        parsed={"type": "unknown_type"},
        transcript=transcript,
        update_ui=update_ui,
        last_status=None,
        had_transcript_events=False,
        claim=_claim(),
        captured_session_id=None,
        format_status=handler.format_status,
        complete_claim=complete_claim,
        fail_claim=fail_claim,
    )
    assert last_status is None
    assert had is False
    complete_claim.assert_not_awaited()
    fail_claim.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_parsed_event_failed_complete_does_not_mark_success():
    """Failed terminal events are not rendered as successful completion."""
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock()

    cli_manager = MagicMock()
    session_store = MagicMock()
    handler = MessagingWorkflow(platform, cli_manager, session_store)

    transcript = MagicMock()
    update_ui = AsyncMock()
    complete_claim = AsyncMock()
    fail_claim = AsyncMock()

    last_status, had = await process_parsed_cli_event(
        parsed={"type": "complete", "status": "failed"},
        transcript=transcript,
        update_ui=update_ui,
        last_status="❌ Error",
        had_transcript_events=True,
        claim=_claim(),
        captured_session_id="session_1",
        format_status=handler.format_status,
        complete_claim=complete_claim,
        fail_claim=fail_claim,
    )

    assert last_status == "❌ Error"
    assert had is True
    update_ui.assert_not_awaited()
    complete_claim.assert_not_awaited()
    fail_claim.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_update_ui_edit_failure_does_not_crash():
    """When queue_edit_message raises during streaming, node_runner.process_node continues and completes."""
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock(
        side_effect=RuntimeError("Telegram API error")
    )
    platform.fire_and_forget = MagicMock(
        side_effect=lambda c: getattr(c, "close", lambda: None)()
    )

    async def _mock_start_task(*args, **kwargs):
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": " world"},
        }
        yield {"type": "complete", "status": "success"}

    mock_session = MagicMock()
    mock_session.start_task = _mock_start_task
    cli_manager = MagicMock()
    cli_manager.get_or_create_session = AsyncMock(
        return_value=(mock_session, "s1", False)
    )
    cli_manager.remove_session = AsyncMock()
    cli_manager.get_stats.return_value = {"active_sessions": 0}

    session_store = MagicMock()
    handler = MessagingWorkflow(platform, cli_manager, session_store)
    snapshot = _snapshot("complete")
    with patch.object(
        handler.tree_queue,
        "complete_claim",
        AsyncMock(return_value=snapshot),
    ):
        await handler.node_runner.process_node(_claim())

    cli_manager.remove_session.assert_awaited_once()

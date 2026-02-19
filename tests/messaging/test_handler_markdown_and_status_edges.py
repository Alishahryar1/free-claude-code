from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from messaging.handler import ClaudeMessageHandler
from messaging.models import IncomingMessage
from messaging.telegram_markdown import render_markdown_to_mdv2
from messaging.tree_data import MessageNode, MessageState


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
    assert "alt (http://example.com/img.png)" in out
    assert "```" in out


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
    handler = ClaudeMessageHandler(platform, cli_manager, session_store)

    with (
        patch.object(
            handler.tree_queue, "is_node_tree_busy", MagicMock(return_value=True)
        ),
        patch.object(handler.tree_queue, "get_queue_size", MagicMock(return_value=2)),
    ):
        s1 = handler._get_initial_status(tree=object(), parent_node_id="p")
    assert "Queued" in s1
    assert "position 3" in s1 or "position 3" in s1.replace("\\", "")

    with patch.object(
        handler.tree_queue, "is_node_tree_busy", MagicMock(return_value=False)
    ):
        s2 = handler._get_initial_status(tree=object(), parent_node_id="p")
    assert "Continuing" in s2

    s3 = handler._get_initial_status(tree=None, parent_node_id=None)
    assert "Launching" in s3


@pytest.mark.asyncio
async def test_update_queue_positions_handles_snapshot_error_and_skips_non_pending():
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock()
    platform.fire_and_forget = MagicMock(
        side_effect=lambda c: getattr(c, "close", lambda: None)()
    )

    cli_manager = MagicMock()
    session_store = MagicMock()
    handler = ClaudeMessageHandler(platform, cli_manager, session_store)

    # Snapshot error is swallowed.
    tree = MagicMock()
    tree.get_queue_snapshot = AsyncMock(side_effect=RuntimeError("boom"))
    await handler._update_queue_positions(tree)
    platform.fire_and_forget.assert_not_called()

    # Normal path: only PENDING nodes get an update.
    node_pending = MagicMock()
    node_pending.state = MessageState.PENDING
    node_pending.incoming.chat_id = "c"
    node_pending.status_message_id = "s"

    node_done = MagicMock()
    node_done.state = MessageState.COMPLETED

    tree.get_queue_snapshot = AsyncMock(return_value=["n1", "n2"])
    tree.get_node = MagicMock(side_effect=[node_pending, node_done])

    await handler._update_queue_positions(tree)
    assert platform.fire_and_forget.call_count == 1


@pytest.mark.asyncio
async def test_process_node_session_limit_marks_error_and_updates_ui():
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock()
    platform.fire_and_forget = MagicMock(
        side_effect=lambda c: getattr(c, "close", lambda: None)()
    )

    cli_manager = MagicMock()
    cli_manager.get_or_create_session = AsyncMock(side_effect=RuntimeError("limit"))
    cli_manager.get_stats.return_value = {"active_sessions": 0}

    session_store = MagicMock()
    handler = ClaudeMessageHandler(platform, cli_manager, session_store)

    fake_tree = MagicMock()
    fake_tree.update_state = AsyncMock()
    with patch.object(
        handler.tree_queue, "get_tree_for_node", MagicMock(return_value=fake_tree)
    ):
        incoming = IncomingMessage(
            text="hi",
            chat_id="c",
            user_id="u",
            message_id="n1",
            platform="telegram",
        )
        node = MessageNode(node_id="n1", incoming=incoming, status_message_id="s1")

        await handler._process_node("n1", node)
    assert platform.queue_edit_message.await_count >= 1
    fake_tree.update_state.assert_awaited()


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
    handler = ClaudeMessageHandler(platform, cli_manager, session_store)

    incoming = IncomingMessage(
        text="hi",
        chat_id="c",
        user_id="u",
        message_id="n1",
        platform="telegram",
    )
    node = MessageNode(node_id="n1", incoming=incoming, status_message_id="s1")

    tree = MagicMock()
    tree.root_id = "root"
    tree.to_dict = MagicMock(return_value={"root": "ok"})
    with (
        patch.object(handler.tree_queue, "cancel_all", AsyncMock(return_value=[node])),
        patch.object(
            handler.tree_queue, "get_tree_for_node", MagicMock(return_value=tree)
        ),
    ):
        count = await handler.stop_all_tasks()
    assert count == 1
    cli_manager.stop_all.assert_awaited_once()
    session_store.save_tree.assert_called_once_with("root", {"root": "ok"})


@pytest.mark.asyncio
async def test_handle_message_reply_with_tree_but_no_parent_treated_as_new():
    platform = MagicMock()
    platform.queue_send_message = AsyncMock(return_value="status_1")
    platform.queue_edit_message = AsyncMock()

    cli_manager = MagicMock()
    cli_manager.get_stats.return_value = {"active_sessions": 0}

    session_store = MagicMock()
    handler = ClaudeMessageHandler(platform, cli_manager, session_store)

    # Force "tree exists but parent can't be resolved" branch.
    handler.tree_queue = MagicMock()
    handler.tree_queue.get_tree_for_node.return_value = object()
    handler.tree_queue.resolve_parent_node_id.return_value = None
    handler.tree_queue.create_tree = AsyncMock(
        return_value=MagicMock(root_id="root", to_dict=MagicMock(return_value={"t": 1}))
    )
    handler.tree_queue.register_node = MagicMock()
    handler.tree_queue.enqueue = AsyncMock(return_value=False)

    incoming = IncomingMessage(
        text="reply",
        chat_id="c",
        user_id="u",
        message_id="m1",
        platform="telegram",
        reply_to_message_id="some_reply",
    )

    await handler.handle_message(incoming)
    handler.tree_queue.create_tree.assert_awaited_once()


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

    handler = ClaudeMessageHandler(platform, cli_manager, session_store)
    handler.tree_queue = MagicMock()
    handler.tree_queue.get_tree_for_node.return_value = None

    incoming = IncomingMessage(
        text="hi",
        chat_id="c",
        user_id="u",
        message_id="n1",
        platform="telegram",
    )
    node = MessageNode(node_id="n1", incoming=incoming, status_message_id="s1")

    with patch.object(handler, "_create_transcript_and_render_ctx") as mock_create:
        transcript = MagicMock()
        transcript.render = MagicMock(side_effect=ValueError("render failed"))
        render_ctx = MagicMock()
        mock_create.return_value = (transcript, render_ctx)

        await handler._process_node("n1", node)

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
    handler = ClaudeMessageHandler(platform, cli_manager, session_store)
    handler.tree_queue = MagicMock()
    handler.tree_queue.get_tree_for_node.return_value = None
    handler.tree_queue.resolve_parent_node_id.return_value = None
    handler.tree_queue.create_tree = AsyncMock(
        return_value=MagicMock(root_id="root", to_dict=MagicMock(return_value={"t": 1}))
    )
    handler.tree_queue.register_node = MagicMock()
    handler.tree_queue.enqueue = AsyncMock(return_value=True)

    incoming = MagicMock()
    incoming.text = None
    incoming.chat_id = "c"
    incoming.user_id = "u"
    incoming.message_id = "m1"
    incoming.platform = "telegram"
    incoming.reply_to_message_id = None
    incoming.is_reply = MagicMock(return_value=False)

    await handler.handle_message(incoming)
    handler.tree_queue.create_tree.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_parsed_event_malformed_content_continues():
    """Malformed/unknown parsed event does not crash _process_parsed_event."""
    platform = MagicMock()
    platform.queue_edit_message = AsyncMock()

    cli_manager = MagicMock()
    session_store = MagicMock()
    handler = ClaudeMessageHandler(platform, cli_manager, session_store)

    transcript = MagicMock()
    update_ui = AsyncMock()

    last_status, had = await handler._process_parsed_event(
        parsed={"type": "unknown_type"},
        transcript=transcript,
        update_ui=update_ui,
        last_status=None,
        had_transcript_events=False,
        tree=None,
        node_id="n1",
        captured_session_id=None,
    )
    assert last_status is None
    assert had is False


@pytest.mark.asyncio
async def test_handler_update_ui_edit_failure_does_not_crash():
    """When queue_edit_message raises during streaming, _process_node continues and completes."""
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
    handler = ClaudeMessageHandler(platform, cli_manager, session_store)
    handler.tree_queue = MagicMock()
    handler.tree_queue.get_tree_for_node.return_value = None

    incoming = IncomingMessage(
        text="hi",
        chat_id="c",
        user_id="u",
        message_id="n1",
        platform="telegram",
    )
    node = MessageNode(node_id="n1", incoming=incoming, status_message_id="s1")

    await handler._process_node("n1", node)

    cli_manager.remove_session.assert_awaited_once()

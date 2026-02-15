"""
Claude Message Handler

Platform-agnostic Claude interaction logic.
Handles the core workflow of processing user messages via Claude CLI.
Uses tree-based queuing for message ordering.
"""

import time
import asyncio
import logging
import os
from typing import Optional, Tuple

from .base import MessagingPlatform, SessionManagerInterface
from .models import IncomingMessage
from .session import SessionStore
from .tree_queue import TreeQueueManager, MessageNode, MessageState, MessageTree
from .event_parser import parse_cli_event
from .transcript import TranscriptBuffer, RenderCtx
from .telegram_markdown import (
    escape_md_v2,
    escape_md_v2_code,
    mdv2_bold,
    mdv2_code_inline,
    format_status,
    render_markdown_to_mdv2,
)

logger = logging.getLogger(__name__)


class ClaudeMessageHandler:
    """
    Platform-agnostic handler for Claude interactions.

    Uses a tree-based message queue where:
    - New messages create a tree root
    - Replies become children of the message being replied to
    - Each node has state: PENDING, IN_PROGRESS, COMPLETED, ERROR
    - Per-tree queue ensures ordered processing
    """

    def __init__(
        self,
        platform: MessagingPlatform,
        cli_manager: SessionManagerInterface,
        session_store: SessionStore,
    ):
        self.platform = platform
        self.cli_manager = cli_manager
        self.session_store = session_store
        self.tree_queue = TreeQueueManager(
            queue_update_callback=self._update_queue_positions,
            node_started_callback=self._mark_node_processing,
        )

    async def handle_message(self, incoming: IncomingMessage) -> None:
        """
        Main entry point for handling an incoming message.

        Determines if this is a new conversation or reply,
        creates/extends the message tree, and queues for processing.
        """
        # Check for commands
        parts = (incoming.text or "").strip().split()
        cmd = parts[0] if parts else ""
        cmd_base = cmd.split("@", 1)[0] if cmd else ""

        # Record incoming message ID for best-effort UI clearing (/clear), even if
        # we later ignore this message (status/command/etc).
        try:
            if incoming.message_id is not None:
                kind = "command" if cmd_base.startswith("/") else "content"
                self.session_store.record_message_id(
                    incoming.platform,
                    incoming.chat_id,
                    str(incoming.message_id),
                    direction="in",
                    kind=kind,
                )
        except Exception as e:
            logger.debug(f"Failed to record incoming message_id: {e}")

        if cmd_base == "/clear":
            await self._handle_clear_command(incoming)
            return

        if cmd_base == "/stop":
            await self._handle_stop_command(incoming)
            return

        if cmd_base == "/stats":
            await self._handle_stats_command(incoming)
            return

        # Filter out status messages (our own messages)
        if any(
            incoming.text.startswith(p)
            for p in ["⏳", "💭", "🔧", "✅", "❌", "🚀", "🤖", "📋", "📊", "🔄"]
        ):
            return

        # Check if this is a reply to an existing node in a tree
        parent_node_id = None
        tree = None

        if incoming.is_reply() and incoming.reply_to_message_id:
            # Look up if the replied-to message is in any tree (could be a node or status message)
            reply_id = incoming.reply_to_message_id
            tree = self.tree_queue.get_tree_for_node(reply_id)
            if tree:
                # Resolve to actual node ID (handles status message replies)
                parent_node_id = self.tree_queue.resolve_parent_node_id(reply_id)
                if parent_node_id:
                    logger.info(f"Found tree for reply, parent node: {parent_node_id}")
                else:
                    logger.warning(
                        f"Reply to {incoming.reply_to_message_id} found tree but no valid parent node"
                    )
                    tree = None  # Treat as new conversation

        # Generate node ID
        node_id = incoming.message_id

        # Send initial status message
        status_text = self._get_initial_status(tree, parent_node_id)
        status_msg_id = await self.platform.queue_send_message(
            incoming.chat_id,
            status_text,
            reply_to=incoming.message_id,
            fire_and_forget=False,
        )
        try:
            if status_msg_id:
                self.session_store.record_message_id(
                    incoming.platform,
                    incoming.chat_id,
                    str(status_msg_id),
                    direction="out",
                    kind="status",
                )
        except Exception as e:
            logger.debug(f"Failed to record status message_id: {e}")

        # Create or extend tree
        if parent_node_id and tree and status_msg_id:
            # Reply to existing node - add as child
            tree, node = await self.tree_queue.add_to_tree(
                parent_node_id=parent_node_id,
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_msg_id,
            )
            # Register status message as a node too for reply chains
            self.tree_queue.register_node(status_msg_id, tree.root_id)
            self.session_store.register_node(status_msg_id, tree.root_id)
            self.session_store.register_node(node_id, tree.root_id)
        elif status_msg_id:
            # New conversation - create new tree
            tree = await self.tree_queue.create_tree(
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_msg_id,
            )
            # Register status message
            self.tree_queue.register_node(status_msg_id, tree.root_id)
            self.session_store.register_node(node_id, tree.root_id)
            self.session_store.register_node(status_msg_id, tree.root_id)

        # Persist tree
        if tree:
            self.session_store.save_tree(tree.root_id, tree.to_dict())

        # Enqueue for processing
        was_queued = await self.tree_queue.enqueue(
            node_id=node_id,
            processor=self._process_node,
        )

        if was_queued and status_msg_id:
            # Update status to show queue position
            queue_size = self.tree_queue.get_queue_size(node_id)
            await self.platform.queue_edit_message(
                incoming.chat_id,
                status_msg_id,
                format_status("📋", "Queued", f"(position {queue_size}) - waiting..."),
                parse_mode="MarkdownV2",
            )

    async def _update_queue_positions(self, tree: MessageTree) -> None:
        """Refresh queued status messages after a dequeue."""
        try:
            queued_ids = await tree.get_queue_snapshot()
        except Exception as e:
            logger.warning(f"Failed to read queue snapshot: {e}")
            return

        if not queued_ids:
            return

        position = 0
        for node_id in queued_ids:
            node = tree.get_node(node_id)
            if not node or node.state != MessageState.PENDING:
                continue
            position += 1
            self.platform.fire_and_forget(
                self.platform.queue_edit_message(
                    node.incoming.chat_id,
                    node.status_message_id,
                    format_status(
                        "📋", "Queued", f"(position {position}) - waiting..."
                    ),
                    parse_mode="MarkdownV2",
                )
            )

    async def _mark_node_processing(self, tree: MessageTree, node_id: str) -> None:
        """Update the dequeued node's status to processing immediately."""
        node = tree.get_node(node_id)
        if not node or node.state == MessageState.ERROR:
            return
        self.platform.fire_and_forget(
            self.platform.queue_edit_message(
                node.incoming.chat_id,
                node.status_message_id,
                format_status("🔄", "Processing..."),
                parse_mode="MarkdownV2",
            )
        )

    def _create_transcript_and_render_ctx(
        self,
    ) -> Tuple[TranscriptBuffer, RenderCtx]:
        """Create transcript buffer and render context for node processing."""
        transcript = TranscriptBuffer(show_tool_results=False)
        render_ctx = RenderCtx(
            bold=mdv2_bold,
            code_inline=mdv2_code_inline,
            escape_code=escape_md_v2_code,
            escape_text=escape_md_v2,
            render_markdown=render_markdown_to_mdv2,
        )
        return transcript, render_ctx

    async def _handle_session_info_event(
        self,
        event_data: dict,
        tree: Optional[MessageTree],
        node_id: str,
        captured_session_id: Optional[str],
        temp_session_id: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Handle session_info event; return updated (captured_session_id, temp_session_id)."""
        if event_data.get("type") != "session_info":
            return captured_session_id, temp_session_id

        real_session_id = event_data.get("session_id")
        if not real_session_id or not temp_session_id:
            return captured_session_id, temp_session_id

        await self.cli_manager.register_real_session_id(
            temp_session_id, real_session_id
        )
        if tree and real_session_id:
            await tree.update_state(
                node_id,
                MessageState.IN_PROGRESS,
                session_id=real_session_id,
            )
            self.session_store.save_tree(tree.root_id, tree.to_dict())

        return real_session_id, None

    async def _process_parsed_event(
        self,
        parsed: dict,
        transcript: TranscriptBuffer,
        update_ui,
        last_status: Optional[str],
        had_transcript_events: bool,
        tree: Optional[MessageTree],
        node_id: str,
        captured_session_id: Optional[str],
    ) -> Tuple[Optional[str], bool]:
        """Process a single parsed CLI event. Returns (last_status, had_transcript_events)."""
        ptype = parsed.get("type")
        transcript_types = (
            "thinking_start",
            "thinking_delta",
            "thinking_chunk",
            "thinking_stop",
            "text_start",
            "text_delta",
            "text_chunk",
            "text_stop",
            "tool_use_start",
            "tool_use_delta",
            "tool_use_stop",
            "tool_use",
            "tool_result",
            "block_stop",
            "error",
        )

        if ptype in transcript_types:
            transcript.apply(parsed)
            had_transcript_events = True

        if ptype in ("thinking_start", "thinking_delta", "thinking_chunk"):
            await update_ui(format_status("🧠", "Claude is thinking..."))
            last_status = format_status("🧠", "Claude is thinking...")
        elif ptype in ("text_start", "text_delta", "text_chunk"):
            await update_ui(format_status("🧠", "Claude is working..."))
            last_status = format_status("🧠", "Claude is working...")
        elif ptype in ("tool_use_start", "tool_use_delta", "tool_use"):
            if parsed.get("name") == "Task":
                await update_ui(format_status("🤖", "Subagent working..."))
                last_status = format_status("🤖", "Subagent working...")
            else:
                await update_ui(format_status("⏳", "Executing tools..."))
                last_status = format_status("⏳", "Executing tools...")
        elif ptype == "tool_result":
            await update_ui(format_status("⏳", "Executing tools..."))
            last_status = format_status("⏳", "Executing tools...")
        elif ptype == "block_stop":
            await update_ui(last_status, force=True)
        elif ptype == "complete":
            if not had_transcript_events:
                transcript.apply({"type": "text_chunk", "text": "Done."})
            logger.info("HANDLER: Task complete, updating UI")
            await update_ui(format_status("✅", "Complete"), force=True)
            if tree and captured_session_id:
                await tree.update_state(
                    node_id,
                    MessageState.COMPLETED,
                    session_id=captured_session_id,
                )
                self.session_store.save_tree(tree.root_id, tree.to_dict())
        elif ptype == "error":
            error_msg = parsed.get("message", "Unknown error")
            logger.error(f"HANDLER: Error event received: {error_msg[:200]}")
            logger.info("HANDLER: Updating UI with error status")
            await update_ui(format_status("❌", "Error"), force=True)
            if tree:
                await self._propagate_error_to_children(
                    node_id, error_msg, "Parent task failed"
                )

        return last_status, had_transcript_events

    async def _process_node(
        self,
        node_id: str,
        node: MessageNode,
    ) -> None:
        """Core task processor - handles a single Claude CLI interaction."""
        incoming = node.incoming
        status_msg_id = node.status_message_id
        chat_id = incoming.chat_id

        tree = self.tree_queue.get_tree_for_node(node_id)
        if tree:
            await tree.update_state(node_id, MessageState.IN_PROGRESS)

        transcript, render_ctx = self._create_transcript_and_render_ctx()

        last_ui_update = 0.0
        last_displayed_text = None
        had_transcript_events = False
        captured_session_id = None
        temp_session_id = None
        last_status: Optional[str] = None

        parent_session_id = None
        if tree and node.parent_id:
            parent_session_id = tree.get_parent_session_id(node_id)
            if parent_session_id:
                logger.info(f"Will fork from parent session: {parent_session_id}")

        async def update_ui(status: Optional[str] = None, force: bool = False) -> None:
            nonlocal last_ui_update, last_displayed_text, last_status
            now = time.time()
            if not force and now - last_ui_update < 1.0:
                return

            last_ui_update = now
            if status is not None:
                last_status = status
            display = transcript.render(render_ctx, limit_chars=3900, status=status)
            if display and display != last_displayed_text:
                logger.debug(
                    "TELEGRAM_EDIT: node_id=%s chat_id=%s msg_id=%s force=%s status=%r chars=%d",
                    node_id,
                    chat_id,
                    status_msg_id,
                    bool(force),
                    status,
                    len(display),
                )
                if os.getenv("DEBUG_TELEGRAM_EDITS") == "1":
                    logger.debug("TELEGRAM_EDIT_TEXT:\n%s", display)
                else:
                    head = display[:500]
                    tail = display[-500:] if len(display) > 500 else ""
                    logger.debug("TELEGRAM_EDIT_PREVIEW_HEAD:\n%s", head)
                    if tail:
                        logger.debug("TELEGRAM_EDIT_PREVIEW_TAIL:\n%s", tail)
                last_displayed_text = display
                await self.platform.queue_edit_message(
                    chat_id, status_msg_id, display, parse_mode="MarkdownV2"
                )

        try:
            try:
                (
                    cli_session,
                    session_or_temp_id,
                    is_new,
                ) = await self.cli_manager.get_or_create_session(session_id=None)
                if is_new:
                    temp_session_id = session_or_temp_id
                else:
                    captured_session_id = session_or_temp_id
            except RuntimeError as e:
                transcript.apply({"type": "error", "message": str(e)})
                await update_ui(
                    format_status("⏳", "Session limit reached"), force=True
                )
                if tree:
                    await tree.update_state(
                        node_id, MessageState.ERROR, error_message=str(e)
                    )
                return

            logger.info(f"HANDLER: Starting CLI task processing for node {node_id}")
            event_count = 0
            async for event_data in cli_session.start_task(
                incoming.text,
                session_id=parent_session_id,
                fork_session=bool(parent_session_id),
            ):
                if not isinstance(event_data, dict):
                    logger.warning(
                        f"HANDLER: Non-dict event received: {type(event_data)}"
                    )
                    continue
                event_count += 1
                if event_count % 10 == 0:
                    logger.debug(f"HANDLER: Processed {event_count} events so far")

                (
                    captured_session_id,
                    temp_session_id,
                ) = await self._handle_session_info_event(
                    event_data, tree, node_id, captured_session_id, temp_session_id
                )
                if event_data.get("type") == "session_info":
                    continue

                parsed_list = parse_cli_event(event_data)
                logger.debug(f"HANDLER: Parsed {len(parsed_list)} events from CLI")

                for parsed in parsed_list:
                    (
                        last_status,
                        had_transcript_events,
                    ) = await self._process_parsed_event(
                        parsed,
                        transcript,
                        update_ui,
                        last_status,
                        had_transcript_events,
                        tree,
                        node_id,
                        captured_session_id,
                    )

        except asyncio.CancelledError:
            logger.warning(f"HANDLER: Task cancelled for node {node_id}")
            cancel_reason = None
            if isinstance(node.context, dict):
                cancel_reason = node.context.get("cancel_reason")

            if cancel_reason == "stop":
                await update_ui(format_status("⏹", "Stopped."), force=True)
            else:
                transcript.apply({"type": "error", "message": "Task was cancelled"})
                await update_ui(format_status("❌", "Cancelled"), force=True)

            # Do not propagate cancellation to children; a reply-scoped "/stop"
            # should only stop the targeted task.
            if tree:
                await tree.update_state(
                    node_id, MessageState.ERROR, error_message="Cancelled by user"
                )
        except Exception as e:
            logger.error(
                f"HANDLER: Task failed with exception: {type(e).__name__}: {e}"
            )
            error_msg = str(e)[:200]
            transcript.apply({"type": "error", "message": error_msg})
            await update_ui(format_status("💥", "Task Failed"), force=True)
            if tree:
                await self._propagate_error_to_children(
                    node_id, error_msg, "Parent task failed"
                )
        finally:
            logger.info(f"HANDLER: _process_node completed for node {node_id}")
            # Free the session-manager slot. Session IDs are persisted in the tree and
            # can be resumed later by ID; we don't need to keep a CLISession instance
            # around after this node completes.
            try:
                if captured_session_id:
                    await self.cli_manager.remove_session(captured_session_id)
                elif temp_session_id:
                    await self.cli_manager.remove_session(temp_session_id)
            except Exception as e:
                logger.debug(f"Failed to remove session for node {node_id}: {e}")

    async def _propagate_error_to_children(
        self,
        node_id: str,
        error_msg: str,
        child_status_text: str,
    ) -> None:
        """Mark node as error and propagate to pending children with UI updates."""
        affected = await self.tree_queue.mark_node_error(
            node_id, error_msg, propagate_to_children=True
        )
        # Update status messages for all affected children (skip first = current node)
        for child in affected[1:]:
            self.platform.fire_and_forget(
                self.platform.queue_edit_message(
                    child.incoming.chat_id,
                    child.status_message_id,
                    format_status("❌", "Cancelled:", child_status_text),
                    parse_mode="MarkdownV2",
                )
            )

    def _get_initial_status(
        self,
        tree: Optional[object],
        parent_node_id: Optional[str],
    ) -> str:
        """Get initial status message text."""
        if tree and parent_node_id:
            # Reply to existing tree
            if self.tree_queue.is_node_tree_busy(parent_node_id):
                queue_size = self.tree_queue.get_queue_size(parent_node_id) + 1
                return format_status(
                    "📋", "Queued", f"(position {queue_size}) - waiting..."
                )
            return format_status("🔄", "Continuing conversation...")

        # New conversation
        stats = self.cli_manager.get_stats()
        if stats["active_sessions"] >= stats["max_sessions"]:
            return format_status(
                "⏳",
                "Waiting for slot...",
                f"({stats['active_sessions']}/{stats['max_sessions']})",
            )
        return format_status("⏳", "Launching new Claude CLI instance...")

    async def stop_all_tasks(self) -> int:
        """
        Stop all pending and in-progress tasks.

        Order of operations:
        1. Cancel tree queue tasks (uses internal locking)
        2. Stop CLI sessions
        3. Update UI for all affected nodes
        """
        # 1. Cancel tree queue tasks using the public async method
        logger.info("Cancelling tree queue tasks...")
        cancelled_nodes = await self.tree_queue.cancel_all()
        logger.info(f"Cancelled {len(cancelled_nodes)} nodes")

        # 2. Stop CLI sessions - this kills subprocesses and ensures everything is dead
        logger.info("Stopping all CLI sessions...")
        await self.cli_manager.stop_all()

        # 3. Update UI and persist state for all cancelled nodes
        for node in cancelled_nodes:
            self.platform.fire_and_forget(
                self.platform.queue_edit_message(
                    node.incoming.chat_id,
                    node.status_message_id,
                    format_status("⏹", "Stopped."),
                    parse_mode="MarkdownV2",
                )
            )

            # Persist tree state
            tree = self.tree_queue.get_tree_for_node(node.node_id)
            if tree:
                self.session_store.save_tree(tree.root_id, tree.to_dict())

        return len(cancelled_nodes)

    async def stop_task(self, node_id: str) -> int:
        """
        Stop a single queued or in-progress task node.

        Used when the user replies "/stop" to a specific status/user message.
        """
        tree = self.tree_queue.get_tree_for_node(node_id)
        if tree:
            node = tree.get_node(node_id)
            if node and node.state not in (MessageState.COMPLETED, MessageState.ERROR):
                # Used by _process_node cancellation path to render "Stopped."
                node.context = {"cancel_reason": "stop"}

        cancelled_nodes = await self.tree_queue.cancel_node(node_id)

        for node in cancelled_nodes:
            self.platform.fire_and_forget(
                self.platform.queue_edit_message(
                    node.incoming.chat_id,
                    node.status_message_id,
                    format_status("⏹", "Stopped."),
                    parse_mode="MarkdownV2",
                )
            )

            tree = self.tree_queue.get_tree_for_node(node.node_id)
            if tree:
                self.session_store.save_tree(tree.root_id, tree.to_dict())

        return len(cancelled_nodes)

    async def _handle_stop_command(self, incoming: IncomingMessage) -> None:
        """Handle /stop command from messaging platform."""
        # Reply-scoped stop: reply "/stop" to stop only that task.
        if incoming.is_reply() and incoming.reply_to_message_id:
            reply_id = incoming.reply_to_message_id
            tree = self.tree_queue.get_tree_for_node(reply_id)
            node_id = self.tree_queue.resolve_parent_node_id(reply_id) if tree else None

            if not node_id:
                msg_id = await self.platform.queue_send_message(
                    incoming.chat_id,
                    format_status("⏹", "Stopped.", "Nothing to stop for that message."),
                    fire_and_forget=False,
                )
                try:
                    if msg_id:
                        self.session_store.record_message_id(
                            incoming.platform,
                            incoming.chat_id,
                            str(msg_id),
                            direction="out",
                            kind="command",
                        )
                except Exception:
                    pass
                return

            count = await self.stop_task(node_id)
            noun = "request" if count == 1 else "requests"
            msg_id = await self.platform.queue_send_message(
                incoming.chat_id,
                format_status("⏹", "Stopped.", f"Cancelled {count} {noun}."),
                fire_and_forget=False,
            )
            try:
                if msg_id:
                    self.session_store.record_message_id(
                        incoming.platform,
                        incoming.chat_id,
                        str(msg_id),
                        direction="out",
                        kind="command",
                    )
            except Exception:
                pass
            return

        # Global stop: legacy behavior (stop everything)
        count = await self.stop_all_tasks()
        msg_id = await self.platform.queue_send_message(
            incoming.chat_id,
            format_status(
                "⏹", "Stopped.", f"Cancelled {count} pending or active requests."
            ),
            fire_and_forget=False,
        )
        try:
            if msg_id:
                self.session_store.record_message_id(
                    incoming.platform,
                    incoming.chat_id,
                    str(msg_id),
                    direction="out",
                    kind="command",
                )
        except Exception:
            pass

    async def _handle_stats_command(self, incoming: IncomingMessage) -> None:
        """Handle /stats command."""
        stats = self.cli_manager.get_stats()
        tree_count = self.tree_queue.get_tree_count()
        msg_id = await self.platform.queue_send_message(
            incoming.chat_id,
            "📊 "
            + mdv2_bold("Stats")
            + "\n"
            + escape_md_v2(f"• Active CLI: {stats['active_sessions']}")
            + "\n"
            + escape_md_v2(f"• Max CLI: {stats['max_sessions']}")
            + "\n"
            + escape_md_v2(f"• Message Trees: {tree_count}"),
            fire_and_forget=False,
        )
        try:
            if msg_id:
                self.session_store.record_message_id(
                    incoming.platform,
                    incoming.chat_id,
                    str(msg_id),
                    direction="out",
                    kind="command",
                )
        except Exception:
            pass

    async def _handle_clear_command(self, incoming: IncomingMessage) -> None:
        """
        Handle /clear global command.

        Order:
        1. Stop all pending/in-progress tasks.
        2. Best-effort delete tracked chat messages for this chat.
        3. Clear sessions.json (entire store) and reset in-memory queue state.
        """
        # 1) Stop tasks first (ensures no more work is running).
        await self.stop_all_tasks()

        # 2) Clear chat: best-effort delete messages we can identify.
        msg_ids: set[str] = set()

        # Add any recorded message IDs for this chat (commands, command replies, etc).
        try:
            for mid in self.session_store.get_message_ids_for_chat(
                incoming.platform, incoming.chat_id
            ):
                if mid is not None:
                    msg_ids.add(str(mid))
        except Exception as e:
            logger.debug(f"Failed to read message log for /clear: {e}")

        try:
            data = self.tree_queue.to_dict()
            trees = data.get("trees", {})
            for tree_data in trees.values():
                nodes = tree_data.get("nodes", {})
                for node_data in nodes.values():
                    inc = node_data.get("incoming", {}) or {}
                    if str(inc.get("platform")) != str(incoming.platform):
                        continue
                    if str(inc.get("chat_id")) != str(incoming.chat_id):
                        continue

                    mid = inc.get("message_id")
                    if mid is not None:
                        msg_ids.add(str(mid))

                    sid = node_data.get("status_message_id")
                    if sid is not None:
                        msg_ids.add(str(sid))
        except Exception as e:
            logger.warning(f"Failed to gather messages for /clear: {e}")

        # Also delete the command message itself.
        if incoming.message_id is not None:
            msg_ids.add(str(incoming.message_id))

        def _as_int(s: str) -> int | None:
            try:
                return int(str(s))
            except Exception:
                return None

        numeric: list[tuple[int, str]] = []
        non_numeric: list[str] = []
        for mid in msg_ids:
            n = _as_int(mid)
            if n is None:
                non_numeric.append(mid)
            else:
                numeric.append((n, mid))

        numeric.sort(reverse=True)
        ordered = [mid for _, mid in numeric] + non_numeric

        # If platform supports batch deletes, prefer it.
        batch_fn = getattr(self.platform, "queue_delete_messages", None)
        if callable(batch_fn):
            try:
                # Telegram supports up to 100 per request.
                CHUNK = 100
                for i in range(0, len(ordered), CHUNK):
                    chunk = ordered[i : i + CHUNK]
                    await batch_fn(incoming.chat_id, chunk, fire_and_forget=False)
            except Exception as e:
                logger.debug(f"/clear batch delete failed: {type(e).__name__}: {e}")
        else:
            for mid in ordered:
                try:
                    await self.platform.queue_delete_message(
                        incoming.chat_id,
                        mid,
                        fire_and_forget=False,
                    )
                except Exception as e:
                    # Deleting is best-effort; platform adapters also treat common cases as no-op.
                    logger.debug(
                        f"/clear delete failed for msg {mid}: {type(e).__name__}: {e}"
                    )

        # 3) Clear persistent state and reset in-memory queue/tree state.
        try:
            self.session_store.clear_all()
        except Exception as e:
            logger.warning(f"Failed to clear session store: {e}")

        self.tree_queue = TreeQueueManager(
            queue_update_callback=self._update_queue_positions,
            node_started_callback=self._mark_node_processing,
        )

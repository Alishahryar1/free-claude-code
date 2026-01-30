"""
Claude Message Handler

Platform-agnostic Claude interaction logic.
Handles the core workflow of processing user messages via Claude CLI.
Uses tree-based queuing for message ordering.
"""

import time
import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from .base import MessagingPlatform
from .models import IncomingMessage
from .session import SessionStore
from .tree_queue import TreeQueueManager, MessageNode, MessageState
from cli import CLISessionManager, CLIParser

if TYPE_CHECKING:
    pass

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
        cli_manager: CLISessionManager,
        session_store: SessionStore,
    ):
        self.platform = platform
        self.cli_manager = cli_manager
        self.session_store = session_store
        self.tree_queue = TreeQueueManager()

    async def handle_message(self, incoming: IncomingMessage) -> None:
        """
        Main entry point for handling an incoming message.

        Determines if this is a new conversation or reply,
        creates/extends the message tree, and queues for processing.
        """
        # Check for commands
        if incoming.text == "/stop":
            await self._handle_stop_command(incoming)
            return

        if incoming.text == "/stats":
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

        if incoming.is_reply():
            # Look up if the replied-to message is in any tree (could be a node or status message)
            tree = self.tree_queue.get_tree_for_node(incoming.reply_to_message_id)
            if tree:
                # Resolve to actual node ID (handles status message replies)
                parent_node_id = self.tree_queue.resolve_parent_node_id(
                    incoming.reply_to_message_id
                )
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
        status_msg_id = await self.platform.send_message(
            incoming.chat_id,
            status_text,
            reply_to=incoming.message_id if incoming.message_id != incoming.message_id else None,
        )

        # Create or extend tree
        if parent_node_id and tree:
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
        else:
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
        self.session_store.save_tree(tree.root_id, tree.to_dict())

        # Enqueue for processing
        was_queued = await self.tree_queue.enqueue(
            node_id=node_id,
            processor=self._process_node,
        )

        if was_queued:
            # Update status to show queue position
            queue_size = self.tree_queue.get_queue_size(node_id)
            await self.platform.edit_message(
                incoming.chat_id,
                status_msg_id,
                f"📋 **Queued** (position {queue_size}) - waiting...",
                parse_mode="markdown",
            )

    async def _process_node(
        self,
        node_id: str,
        node: MessageNode,
    ) -> None:
        """Core task processor - handles a single Claude CLI interaction."""
        incoming = node.incoming
        status_msg_id = node.status_message_id
        chat_id = incoming.chat_id

        # Update node state to IN_PROGRESS
        tree = self.tree_queue.get_tree_for_node(node_id)
        if tree:
            await tree.update_state(node_id, MessageState.IN_PROGRESS)

        # Components for structured display
        components = {
            "thinking": [],
            "tools": [],
            "subagents": [],
            "content": [],
            "errors": [],
        }

        last_ui_update = 0.0
        captured_session_id = None
        temp_session_id = None

        # Get parent session ID for forking (if child node)
        parent_session_id = None
        if tree and node.parent_id:
            parent_session_id = tree.get_parent_session_id(node_id)
            if parent_session_id:
                logger.info(f"Will fork from parent session: {parent_session_id}")

        async def update_ui(status: Optional[str] = None, force: bool = False) -> None:
            nonlocal last_ui_update
            now = time.time()

            # Small 1s debounce for UI sanity (not a hard rate limit)
            if not force and now - last_ui_update < 1.0:
                return

            # Retry logic for flood wait errors
            max_retries = 3 if not force else 5
            for attempt in range(max_retries):
                try:
                    display = self._build_message(components, status)
                    if display:
                        await self.platform.edit_message(
                            chat_id, status_msg_id, display, parse_mode="markdown"
                        )
                        last_ui_update = now
                        return  # Success, exit retry loop
                except Exception as e:
                    error_msg = str(e).lower()
                    is_flood_error = "flood" in error_msg or "wait" in error_msg
                    
                    if is_flood_error and attempt < max_retries - 1:
                        # Wait before retry on flood error
                        wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                        logger.warning(f"Flood wait on edit (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Log and give up after retries or if not a flood error
                        if is_flood_error:
                            last_ui_update = now + 10  # Skip for 10s on persistent flood error
                        logger.error(f"UI update failed after {attempt + 1} attempts: {e}")
                        return

        try:
            # Get or create CLI session
            try:
                (
                    cli_session,
                    session_or_temp_id,
                    is_new,
                ) = await self.cli_manager.get_or_create_session(
                    session_id=parent_session_id  # Fork from parent if available
                )
                if is_new:
                    temp_session_id = session_or_temp_id
                else:
                    captured_session_id = session_or_temp_id
            except RuntimeError as e:
                components["errors"].append(str(e))
                await update_ui("⏳ **Session limit reached**", force=True)
                if tree:
                    await tree.update_state(
                        node_id, MessageState.ERROR, error_message=str(e)
                    )
                return

            # Process CLI events
            logger.info(f"HANDLER: Starting CLI task processing for node {node_id}")
            event_count = 0
            async for event_data in cli_session.start_task(
                incoming.text, session_id=captured_session_id
            ):
                if not isinstance(event_data, dict):
                    logger.warning(f"HANDLER: Non-dict event received: {type(event_data)}")
                    continue
                event_count += 1
                if event_count % 10 == 0:
                    logger.debug(f"HANDLER: Processed {event_count} events so far")

                # Handle session_info event
                if event_data.get("type") == "session_info":
                    real_session_id = event_data.get("session_id")
                    if real_session_id and temp_session_id:
                        await self.cli_manager.register_real_session_id(
                            temp_session_id, real_session_id
                        )
                        captured_session_id = real_session_id
                        temp_session_id = None
                    continue

                parsed_list = CLIParser.parse_event(event_data)
                logger.debug(f"HANDLER: Parsed {len(parsed_list)} events from CLI")

                for parsed in parsed_list:
                    if parsed["type"] == "thinking":
                        components["thinking"].append(parsed["text"])
                        await update_ui("🧠 **Claude is thinking...**")

                    elif parsed["type"] == "content":
                        if parsed.get("text"):
                            components["content"].append(parsed["text"])
                            await update_ui("🧠 **Claude is working...**")

                    elif parsed["type"] == "tool_start":
                        names = [t.get("name") for t in parsed.get("tools", [])]
                        components["tools"].extend(names)
                        await update_ui("⏳ **Executing tools...**")

                    elif parsed["type"] == "subagent_start":
                        tasks = parsed.get("tasks", [])
                        components["subagents"].extend(tasks)
                        await update_ui("🤖 **Subagent working...**")

                    elif parsed["type"] == "complete":
                        if not any(components.values()):
                            components["content"].append("Done.")
                        logger.info(f"HANDLER: Task complete, updating UI")
                        # Always force final complete status to bypass flood wait
                        await update_ui("✅ **Complete**", force=True)

                        # Update node state and session
                        if tree and captured_session_id:
                            await tree.update_state(
                                node_id,
                                MessageState.COMPLETED,
                                session_id=captured_session_id,
                            )
                            self.session_store.save_tree(tree.root_id, tree.to_dict())

                    elif parsed["type"] == "error":
                        error_msg = parsed.get("message", "Unknown error")
                        logger.error(f"HANDLER: Error event received: {error_msg[:200]}")
                        components["errors"].append(error_msg)
                        logger.info(f"HANDLER: Updating UI with error status")
                        # Always force error status to bypass flood wait
                        await update_ui("❌ **Error**", force=True)
                        if tree:
                            # Mark this node and propagate to pending children
                            affected = await self.tree_queue.mark_node_error(
                                node_id, error_msg, propagate_to_children=True
                            )
                            # Update status messages for all affected children
                            for child in affected[1:]:  # Skip first (current node)
                                try:
                                    await self.platform.edit_message(
                                        child.incoming.chat_id,
                                        child.status_message_id,
                                        f"❌ **Cancelled:** Parent task failed",
                                        parse_mode="markdown",
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to update child status: {e}")

        except asyncio.CancelledError:
            logger.warning(f"HANDLER: Task cancelled for node {node_id}")
            components["errors"].append("Task was cancelled")
            # Always force cancelled status to bypass flood wait
            await update_ui("❌ **Cancelled**", force=True)
            if tree:
                # Mark this node and propagate to pending children
                affected = await self.tree_queue.mark_node_error(
                    node_id, "Cancelled by user", propagate_to_children=True
                )
                # Update status messages for all affected children
                for child in affected[1:]:
                    try:
                        await self.platform.edit_message(
                            child.incoming.chat_id,
                            child.status_message_id,
                            f"❌ **Cancelled:** Parent task was stopped",
                            parse_mode="markdown",
                        )
                    except Exception as e:
                        logger.error(f"Failed to update child status: {e}")
        except Exception as e:
            logger.error(f"HANDLER: Task failed with exception: {type(e).__name__}: {e}")
            error_msg = str(e)[:200]
            components["errors"].append(error_msg)
            # Always force error status to bypass flood wait
            await update_ui("💥 **Task Failed**", force=True)
            if tree:
                # Mark this node and propagate to pending children
                affected = await self.tree_queue.mark_node_error(
                    node_id, error_msg, propagate_to_children=True
                )
                # Update status messages for all affected children
                for child in affected[1:]:
                    try:
                        await self.platform.edit_message(
                            child.incoming.chat_id,
                            child.status_message_id,
                            f"❌ **Cancelled:** Parent task failed",
                            parse_mode="markdown",
                        )
                    except Exception as e:
                        logger.error(f"Failed to update child status: {e}")
        finally:
            logger.info(f"HANDLER: _process_node completed for node {node_id}, errors={len(components['errors'])}")

    def _build_message(
        self,
        components: dict,
        status: Optional[str] = None,
    ) -> str:
        """
        Build unified message with specific order:
        1. Thinking
        2. Tools
        3. Subagents
        4. Content
        5. Errors
        6. Status (Bottom)
        """
        lines = []

        # 1. Thinking
        if components["thinking"]:
            full_thinking = "".join(components["thinking"])
            display = full_thinking
            if len(display) > 800:
                display = display[:795] + "..."
            lines.append(f"💭 **Thinking:**\n```\n{display}\n```")

        # 2. Tools
        if components["tools"]:
            unique_tools = []
            seen = set()
            for t in components["tools"]:
                if t not in seen:
                    unique_tools.append(t)
                    seen.add(t)
            lines.append(f"🛠 **Tools:** `{', '.join(unique_tools)}`")

        # 3. Subagents
        if components["subagents"]:
            for task in components["subagents"]:
                lines.append(f"🤖 **Subagent:** `{task}`")

        # 4. Content
        if components["content"]:
            full_content = "".join(components["content"])
            lines.append(full_content)

        # 5. Errors
        if components["errors"]:
            for err in components["errors"]:
                lines.append(f"⚠️ **Error:** `{err}`")

        # 6. Status (Bottom)
        if status:
            lines.append("")
            lines.append(status)

        result = "\n".join(lines)

        # Truncate if too long (Telegram limit ~4096)
        if len(result) > 3800:
            result = "..." + result[-3795:]

        return result

    def _get_initial_status(
        self,
        tree: Optional[object],
        parent_node_id: Optional[str],
    ) -> str:
        """Get initial status message text."""
        if tree:
            # Reply to existing tree
            if self.tree_queue.is_node_tree_busy(parent_node_id):
                queue_size = self.tree_queue.get_queue_size(parent_node_id) + 1
                return f"📋 **Queued** (position {queue_size}) - waiting..."
            return "🔄 **Continuing conversation...**"

        # New conversation
        stats = self.cli_manager.get_stats()
        if stats["active_sessions"] >= stats["max_sessions"]:
            return f"⏳ **Waiting for slot...** ({stats['active_sessions']}/{stats['max_sessions']})"
        return "⏳ **Launching new Claude CLI instance...**"

    async def stop_all_tasks(self) -> int:
        """
        Stop all pending and in-progress tasks.

        Order of operations:
        1. Stop CLI sessions first (kills subprocesses, unblocks I/O)
        2. Cancel async tasks in tree queue
        3. Update UI for all affected nodes
        """
        # 1. Stop CLI sessions FIRST - this kills subprocess and unblocks I/O
        logger.info("Stopping all CLI sessions...")
        await self.cli_manager.stop_all()

        # 2. Cancel tree queue tasks (now they should unblock immediately)
        logger.info("Cancelling tree queue tasks...")
        cancelled_nodes = await self.tree_queue.cancel_all()
        logger.info(f"Cancelled {len(cancelled_nodes)} nodes")

        # 3. Update UI for all cancelled nodes
        for node in cancelled_nodes:
            try:
                await self.platform.edit_message(
                    node.incoming.chat_id,
                    node.status_message_id,
                    "⏹ **Stopped.**",
                    parse_mode="markdown",
                )
            except Exception as e:
                logger.error(f"Failed to update status for cancelled node: {e}")

        return len(cancelled_nodes)

    async def _handle_stop_command(self, incoming: IncomingMessage) -> None:
        """Handle /stop command from messaging platform."""
        count = await self.stop_all_tasks()
        await self.platform.send_message(
            incoming.chat_id,
            f"⏹ **Stopped.** Cancelled {count} pending or active requests.",
        )

    async def _handle_stats_command(self, incoming: IncomingMessage) -> None:
        """Handle /stats command."""
        stats = self.cli_manager.get_stats()
        tree_count = len(self.tree_queue._trees)
        await self.platform.send_message(
            incoming.chat_id,
            f"📊 **Stats**\n• Active CLI: {stats['active_sessions']}\n• Max CLI: {stats['max_sessions']}\n• Message Trees: {tree_count}",
        )

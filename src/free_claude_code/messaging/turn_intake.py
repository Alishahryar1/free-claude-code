"""Inbound messaging turn intake and queue admission."""

from collections.abc import Awaitable, Callable

from loguru import logger

from free_claude_code.core.trace import trace_event

from .cli_event_constants import STATUS_MESSAGE_PREFIXES
from .command_context import MessagingCommandContext
from .command_dispatcher import (
    dispatch_command,
    parse_command_base,
)
from .models import AdmissionToken, IncomingMessage, MessageScope
from .platforms.ports import OutboundMessenger
from .session import SessionStore
from .trees import (
    AdmissionRejection,
    NodeClaim,
    QueueDecision,
    QueueEntry,
    ReplyTarget,
)


class MessagingTurnIntake:
    """Owns inbound turn classification and queue admission."""

    def __init__(
        self,
        *,
        platform_name: str,
        outbound: OutboundMessenger,
        session_store: SessionStore,
        command_context: MessagingCommandContext,
        resolve_reply: Callable[[MessageScope, str], Awaitable[ReplyTarget | None]],
        admit_turn: Callable[
            [IncomingMessage, str, str | None, AdmissionToken],
            Awaitable[QueueDecision | None],
        ],
        format_status: Callable[[str, str, str | None], str],
        get_parse_mode: Callable[[], str | None],
        record_outgoing_message: Callable[[str, str, str | None, str], bool],
    ) -> None:
        self.platform_name = platform_name
        self.outbound = outbound
        self.session_store = session_store
        self._command_context = command_context
        self._resolve_reply = resolve_reply
        self._admit_turn = admit_turn
        self._format_status = format_status
        self._get_parse_mode = get_parse_mode
        self._record_outgoing_message = record_outgoing_message

    async def handle_message(
        self,
        incoming: IncomingMessage,
        *,
        admission_token: AdmissionToken,
    ) -> None:
        """
        Handle an inbound platform message and queue it if it is a user prompt.
        """
        cmd_base = parse_command_base(incoming.text)

        if await dispatch_command(self._command_context, incoming, cmd_base):
            return

        text = incoming.text or ""
        if any(text.startswith(p) for p in STATUS_MESSAGE_PREFIXES):
            return

        reply_target: ReplyTarget | None = None

        if incoming.is_reply() and incoming.reply_to_message_id:
            reply_id = incoming.reply_to_message_id
            reply_target = await self._resolve_reply(incoming.scope, reply_id)
            if reply_target is not None:
                logger.info(
                    "Found tree for reply, parent node: {}", reply_target.node_id
                )

        node_id = incoming.message_id
        status_text = self._get_initial_status(reply_target)
        if incoming.status_message_id:
            status_msg_id = incoming.status_message_id
            await self.outbound.queue_edit_message(
                incoming.chat_id,
                status_msg_id,
                status_text,
                parse_mode=self._get_parse_mode(),
                fire_and_forget=False,
            )
        else:
            status_msg_id = await self.outbound.queue_send_message(
                incoming.chat_id,
                status_text,
                reply_to=incoming.message_id,
                fire_and_forget=False,
                message_thread_id=incoming.message_thread_id,
            )
        self._record_outgoing_message(
            incoming.platform, incoming.chat_id, status_msg_id, "status"
        )
        if status_msg_id is None:
            return

        decision = await self._admit_turn(
            incoming,
            status_msg_id,
            reply_target.reference_id if reply_target is not None else None,
            admission_token,
        )
        if decision is None:
            logger.info(
                "Discarded messaging admission invalidated by a stop/clear boundary for node {}",
                node_id,
            )
            await self._discard_rejected_messages(
                incoming,
                status_msg_id,
                include_prompt=False,
            )
            return
        if not decision.accepted:
            include_prompt = decision.rejection is AdmissionRejection.PARENT_REMOVED
            logger.debug(
                "Rejected messaging admission for node {}: {}",
                node_id,
                decision.rejection.value
                if decision.rejection is not None
                else "unknown",
            )
            await self._discard_rejected_messages(
                incoming,
                status_msg_id,
                include_prompt=include_prompt,
            )
            return

        if decision.position is not None and status_msg_id:
            trace_event(
                stage="routing",
                event="turn.queued",
                source=self.platform_name,
                chat_id=incoming.chat_id,
                platform_message_id=node_id,
                status_message_id=status_msg_id,
                queue_size=decision.position,
            )
            await self.outbound.queue_edit_message(
                incoming.chat_id,
                status_msg_id,
                self._format_status(
                    "📋",
                    "Queued",
                    f"(position {decision.position}) - waiting...",
                ),
                parse_mode=self._get_parse_mode(),
            )

    async def _discard_rejected_messages(
        self,
        incoming: IncomingMessage,
        status_message_id: str,
        *,
        include_prompt: bool,
    ) -> None:
        """Remove messages created by an admission that cannot commit."""
        message_ids = {status_message_id}
        if include_prompt:
            message_ids.add(str(incoming.message_id))
        try:
            await self.outbound.queue_delete_messages(
                incoming.chat_id,
                list(message_ids),
                fire_and_forget=False,
            )
        except Exception as exc:
            logger.debug(
                "Failed to remove rejected status message: {}",
                type(exc).__name__,
            )
        try:
            self.session_store.forget_tracked_message_ids(
                incoming.platform,
                incoming.chat_id,
                message_ids,
            )
        except Exception as exc:
            logger.debug(
                "Failed to forget rejected status message: {}",
                type(exc).__name__,
            )

    async def update_queue_positions(self, queue: tuple[QueueEntry, ...]) -> None:
        """Refresh queued status messages after a dequeue."""
        for entry in queue:
            self.outbound.fire_and_forget(
                self.outbound.queue_edit_message(
                    entry.node.scope.chat_id,
                    entry.node.status_message_id,
                    self._format_status(
                        "📋",
                        "Queued",
                        f"(position {entry.position}) - waiting...",
                    ),
                    parse_mode=self._get_parse_mode(),
                )
            )

    async def mark_node_processing(self, claim: NodeClaim) -> None:
        """Update the dequeued node's status to processing immediately."""
        self.outbound.fire_and_forget(
            self.outbound.queue_edit_message(
                claim.node.scope.chat_id,
                claim.node.status_message_id,
                self._format_status("🔄", "Processing...", None),
                parse_mode=self._get_parse_mode(),
            )
        )

    def _get_initial_status(
        self,
        reply_target: ReplyTarget | None,
    ) -> str:
        """Get initial status message text."""
        if reply_target is not None:
            if reply_target.queue_position is not None:
                return self._format_status(
                    "📋",
                    "Queued",
                    f"(position {reply_target.queue_position}) - waiting...",
                )
            return self._format_status("🔄", "Continuing conversation...", None)

        return self._format_status("⏳", "Launching new Claude CLI instance...", None)


__all__ = ["MessagingTurnIntake"]

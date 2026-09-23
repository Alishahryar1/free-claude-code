"""Typed dependency surface for messaging slash commands."""

from dataclasses import dataclass
from typing import Protocol

from .managed_protocols import ManagedClaudeSessionManagerProtocol
from .models import MessageScope
from .platforms.ports import OutboundMessenger
from .transcript import RenderCtx


@dataclass(frozen=True, slots=True)
class ReplyClearResult:
    """Customer-facing result of clearing one literal reply subtree."""

    delete_message_ids: frozenset[str]
    tree_matched: bool


@dataclass(frozen=True, slots=True)
class StopOutcome:
    """Customer-facing stop result after terminal status ownership is assigned."""

    cancelled_count: int
    status_feedback_scopes: frozenset[MessageScope]
    fallback_required: bool

    def requires_confirmation(self, scope: MessageScope) -> bool:
        """Return whether this scope lacks complete terminal status feedback."""
        return (
            self.cancelled_count == 0
            or self.fallback_required
            or self.status_feedback_scopes != frozenset({scope})
        )


class MessagingCommandContext(Protocol):
    """Operations commands need from the messaging workflow."""

    outbound: OutboundMessenger
    cli_manager: ManagedClaudeSessionManagerProtocol

    def format_status(self, emoji: str, label: str, suffix: str | None = None) -> str:
        """Format a platform-specific status line."""
        ...

    def get_render_ctx(self) -> RenderCtx:
        """Return the render context for command output."""
        ...

    async def stop_all_tasks(self) -> StopOutcome:
        """Stop every pending or active messaging task."""
        ...

    async def stop_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> StopOutcome:
        """Stop the exact voice/tree owner of a replied-to message."""
        ...

    def get_tree_count(self) -> int:
        """Return the number of conversation trees."""
        ...

    async def clear_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> ReplyClearResult | None:
        """Clear the literal subtree rooted at a replied-to message."""
        ...

    async def clear_chat(self, platform: str, chat_id: str) -> frozenset[str]:
        """Clear one chat and return every tracked platform message ID."""
        ...

    def forget_tracked_message_ids(
        self,
        platform: str,
        chat_id: str,
        message_ids: set[str],
    ) -> None:
        """Forget platform message IDs removed from the managed conversation."""
        ...

    def record_outgoing_message(
        self,
        platform: str,
        chat_id: str,
        msg_id: str | None,
        kind: str,
    ) -> bool:
        """Record an outgoing platform message ID and report registry ownership."""
        ...


__all__ = ["MessagingCommandContext", "ReplyClearResult", "StopOutcome"]

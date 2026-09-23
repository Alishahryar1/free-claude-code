"""Platform-agnostic message models."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class MessageScope:
    """Platform chat namespace in which message IDs are unique."""

    platform: str
    chat_id: str


@dataclass(frozen=True, slots=True)
class AdmissionToken:
    """Generations that must still match when a turn commits."""

    stop_generation: int
    clear_generation: int


@dataclass
class IncomingMessage:
    """
    Platform-agnostic incoming message.

    Adapters convert platform-specific events to this format.
    """

    text: str
    chat_id: str
    user_id: str
    message_id: str
    platform: str  # "telegram", "discord", "slack", etc.

    # Optional fields
    reply_to_message_id: str | None = None
    # Forum topic ID (Telegram); required when replying in forum supergroups
    message_thread_id: str | None = None
    username: str | None = None
    # Pre-sent status message ID (e.g. "Transcribing voice note..."); handler edits in place
    status_message_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Platform-specific raw event stays at ingress and is never persisted by trees.
    raw_event: Any = None

    def is_reply(self) -> bool:
        """Check if this message is a reply to another message."""
        return self.reply_to_message_id is not None

    @property
    def scope(self) -> MessageScope:
        """Return the namespace that owns this message's platform IDs."""
        return MessageScope(platform=str(self.platform), chat_id=str(self.chat_id))

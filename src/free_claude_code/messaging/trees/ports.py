"""Durable messaging operations implemented by the runtime storage adapter."""

from typing import Protocol

from ..models import MessageScope
from .identity import TreeIdentity
from .snapshot import ConversationSnapshot, TreeSnapshot


class MessagingStorageError(RuntimeError):
    """A messaging transition could not be durably stored."""


class MessagingStore(Protocol):
    async def load_conversation_snapshot(self) -> ConversationSnapshot: ...

    async def commit_trees(
        self,
        snapshots: tuple[TreeSnapshot, ...],
        *,
        removed: tuple[TreeIdentity, ...] = (),
        clear_scope: MessageScope | None = None,
    ) -> None: ...

    async def record_message_id(
        self, platform: str, chat_id: str, message_id: str, direction: str, kind: str
    ) -> None: ...

    async def get_tracked_message_ids_for_chat(
        self, platform: str, chat_id: str
    ) -> list[str]: ...

    async def forget_tracked_message_ids(
        self, platform: str, chat_id: str, message_ids: set[str]
    ) -> None: ...

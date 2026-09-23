"""Persistent messaging conversation state store."""

import threading
from copy import deepcopy

from loguru import logger

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.trees import (
    ConversationSnapshot,
    TreeIdentity,
    TreeSnapshot,
)

from .managed_message_log import ManagedMessageLog
from .persistence import DebouncedJsonPersistence


class SessionStore:
    """
    Persistent storage for conversation snapshots and managed platform messages.

    The store reads both the old raw ``trees``/``node_to_tree`` shape and the
    current typed ``conversation`` snapshot shape. Runtime callers deal in typed
    snapshots only.
    """

    def __init__(
        self,
        storage_path: str = "sessions.json",
        *,
        managed_message_cap: int | None = None,
    ) -> None:
        self.storage_path = storage_path
        self._lock = threading.RLock()
        self._conversation = ConversationSnapshot()
        self._managed_messages = ManagedMessageLog(cap=managed_message_cap)
        self._dirty = False
        self._persistence = DebouncedJsonPersistence(
            storage_path,
            snapshot=self._snapshot_for_persistence,
            on_dirty=self._set_dirty,
        )
        self._load()

    @property
    def dirty(self) -> bool:
        return self._dirty

    def _set_dirty(self, dirty: bool) -> None:
        with self._lock:
            self._dirty = dirty

    def _load(self) -> None:
        try:
            data = self._persistence.load_json()
        except Exception as e:
            logger.error("Failed to load sessions: {}", e)
            return

        conversation_data = data.get("conversation") if isinstance(data, dict) else None
        if not isinstance(conversation_data, dict):
            conversation_data = data

        with self._lock:
            self._conversation = ConversationSnapshot.from_json(conversation_data)
            raw_messages = {}
            if isinstance(data, dict):
                raw_messages = data.get("managed_messages", data.get("message_log", {}))
            self._managed_messages = ManagedMessageLog.from_json(
                raw_messages,
                cap=self._managed_messages.cap,
            )
            message_count = sum(
                len(items) for items in self._managed_messages.to_json().values()
            )
            logger.info(
                "Loaded {} trees and {} managed message IDs from {}",
                len(self._conversation.trees),
                message_count,
                self.storage_path,
            )

    def _snapshot_for_persistence(self) -> dict:
        with self._lock:
            return {
                "conversation": self._conversation.to_json(),
                "managed_messages": self._managed_messages.to_json(),
            }

    def load_conversation_snapshot(self) -> ConversationSnapshot:
        with self._lock:
            return deepcopy(self._conversation)

    def save_conversation_snapshot(self, snapshot: ConversationSnapshot) -> None:
        with self._lock:
            self._conversation = deepcopy(snapshot)
            self._persistence.schedule_save()

    def save_tree_snapshot(self, snapshot: TreeSnapshot) -> None:
        with self._lock:
            self._conversation = self._conversation.with_tree(deepcopy(snapshot))
            self._persistence.schedule_save()
            logger.debug("Saved tree {}", snapshot.root_id)

    def remove_tree_snapshot(self, identity: TreeIdentity) -> None:
        with self._lock:
            self._conversation = self._conversation.without_tree(identity)
            self._persistence.schedule_save()

    def flush_pending_save(self) -> None:
        self._persistence.flush()

    def record_message_id(
        self,
        platform: str,
        chat_id: str,
        message_id: str,
        direction: str,
        kind: str,
    ) -> None:
        if message_id is None:
            return
        with self._lock:
            recorded = self._managed_messages.record(
                platform=str(platform),
                chat_id=str(chat_id),
                message_id=str(message_id),
                direction=str(direction),
                kind=str(kind),
            )
            if recorded:
                self._persistence.schedule_save()

    def get_tracked_message_ids_for_chat(
        self, platform: str, chat_id: str
    ) -> list[str]:
        with self._lock:
            return self._managed_messages.ids_for_chat(str(platform), str(chat_id))

    def forget_tracked_message_ids(
        self, platform: str, chat_id: str, message_ids: set[str]
    ) -> None:
        with self._lock:
            removed = self._managed_messages.remove_ids(
                str(platform),
                str(chat_id),
                {str(message_id) for message_id in message_ids},
            )
            if removed:
                self._persistence.schedule_save()

    def clear_scope(self, scope: MessageScope) -> None:
        """Authoritatively clear one platform chat while preserving others."""
        with self._lock:
            self._conversation = self._conversation.without_scope(scope)
            self._managed_messages.clear_chat(scope.platform, scope.chat_id)
            self._write_current_state()

    def _write_current_state(self) -> None:
        self._set_dirty(True)
        self._persistence.write_data(self._snapshot_for_persistence())

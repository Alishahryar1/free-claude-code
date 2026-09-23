"""Persist platform messages belonging to FCC-managed conversations."""

from datetime import UTC, datetime
from typing import Any


class ManagedMessageLog:
    """Track managed inbound and outbound messages in insertion order."""

    def __init__(self, *, cap: int | None = None) -> None:
        self._items: dict[str, list[dict[str, Any]]] = {}
        self._ids: dict[str, set[str]] = {}
        self._cap = cap

    @property
    def cap(self) -> int | None:
        return self._cap

    @classmethod
    def from_json(cls, raw_log: Any, *, cap: int | None = None) -> ManagedMessageLog:
        """Load current and legacy message-log entries."""
        log = cls(cap=cap)
        if not isinstance(raw_log, dict):
            return log
        for chat_key, items in raw_log.items():
            if not isinstance(chat_key, str) or not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                message_id = item.get("message_id")
                direction = str(item.get("direction") or "")
                kind = str(item.get("kind") or "")
                if message_id is None or direction not in {"in", "out"} or not kind:
                    continue
                log._append(
                    chat_key,
                    str(message_id),
                    ts=str(item.get("ts") or ""),
                    direction=direction,
                    kind=kind,
                )
        return log

    def to_json(self) -> dict[str, list[dict[str, Any]]]:
        return {chat_key: list(items) for chat_key, items in self._items.items()}

    def record(
        self,
        *,
        platform: str,
        chat_id: str,
        message_id: str,
        direction: str,
        kind: str,
    ) -> bool:
        """Record one managed platform message."""
        if direction not in {"in", "out"}:
            raise ValueError("Managed message direction must be 'in' or 'out'")
        if not kind:
            raise ValueError("Managed message kind cannot be empty")
        return self._append(
            make_chat_key(platform, chat_id),
            str(message_id),
            ts=datetime.now(UTC).isoformat(),
            direction=direction,
            kind=str(kind),
        )

    def ids_for_chat(self, platform: str, chat_id: str) -> list[str]:
        chat_key = make_chat_key(platform, chat_id)
        return [str(item["message_id"]) for item in self._items.get(chat_key, [])]

    def remove_ids(self, platform: str, chat_id: str, message_ids: set[str]) -> bool:
        chat_key = make_chat_key(platform, chat_id)
        if not message_ids or chat_key not in self._items:
            return False

        before_count = len(self._items[chat_key])
        retained = [
            item
            for item in self._items[chat_key]
            if str(item["message_id"]) not in message_ids
        ]
        if retained:
            self._items[chat_key] = retained
            self._ids[chat_key] = {str(item["message_id"]) for item in retained}
        else:
            self._items.pop(chat_key)
            self._ids.pop(chat_key, None)
        return len(retained) != before_count

    def clear_chat(self, platform: str, chat_id: str) -> bool:
        chat_key = make_chat_key(platform, chat_id)
        removed = self._items.pop(chat_key, None) is not None
        self._ids.pop(chat_key, None)
        return removed

    def _append(
        self,
        chat_key: str,
        message_id: str,
        *,
        ts: str,
        direction: str,
        kind: str,
    ) -> bool:
        seen = self._ids.setdefault(chat_key, set())
        if message_id in seen:
            return False
        self._items.setdefault(chat_key, []).append(
            {
                "message_id": message_id,
                "ts": ts,
                "direction": direction,
                "kind": kind,
            }
        )
        seen.add(message_id)
        self._trim(chat_key)
        return True

    def _trim(self, chat_key: str) -> None:
        if self._cap is None or self._cap <= 0:
            return
        items = self._items.get(chat_key, [])
        if len(items) <= self._cap:
            return
        retained = items[-self._cap :]
        self._items[chat_key] = retained
        self._ids[chat_key] = {str(item["message_id"]) for item in retained}


def make_chat_key(platform: str, chat_id: str) -> str:
    return f"{platform}:{chat_id}"

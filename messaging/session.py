"""
Session Store for Messaging Platforms

Provides persistent storage for mapping platform messages to Claude CLI session IDs
and message trees for conversation continuation.
"""

import json
import os
import threading
from datetime import UTC, datetime
from typing import Any

from loguru import logger


class SessionStore:
    """
    Persistent storage for message ↔ Claude session mappings and message trees.

    Uses a JSON file for storage with thread-safe operations.
    Platform-agnostic: works with any messaging platform.
    """

    def __init__(self, storage_path: str = "sessions.json"):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self._trees: dict[str, dict] = {}  # root_id -> tree data
        self._node_to_tree: dict[str, str] = {}  # node_id -> root_id
        # Per-chat message ID log used to support best-effort UI clearing (/clear).
        # Key: "{platform}:{chat_id}" -> list of records
        self._message_log: dict[str, list[dict[str, Any]]] = {}
        self._message_log_ids: dict[str, set[str]] = {}
        self._dirty = False
        self._save_timer: threading.Timer | None = None
        self._save_debounce_secs = 0.5
        cap_raw = os.getenv("MAX_MESSAGE_LOG_ENTRIES_PER_CHAT", "").strip()
        try:
            self._message_log_cap: int | None = int(cap_raw) if cap_raw else None
        except ValueError:
            self._message_log_cap = None
        self._load()

    def _make_chat_key(self, platform: str, chat_id: str) -> str:
        return f"{platform}:{chat_id}"

    def _load(self) -> None:
        """Load sessions and trees from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, encoding="utf-8") as f:
                data = json.load(f)

            # Load trees
            self._trees = data.get("trees", {})
            self._node_to_tree = data.get("node_to_tree", {})

            # Load message log (optional/backward compatible)
            raw_log = data.get("message_log", {}) or {}
            if isinstance(raw_log, dict):
                self._message_log = {}
                self._message_log_ids = {}
                for chat_key, items in raw_log.items():
                    if not isinstance(chat_key, str) or not isinstance(items, list):
                        continue
                    cleaned: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        mid = it.get("message_id")
                        if mid is None:
                            continue
                        mid_s = str(mid)
                        if mid_s in seen:
                            continue
                        seen.add(mid_s)
                        cleaned.append(
                            {
                                "message_id": mid_s,
                                "ts": str(it.get("ts") or ""),
                                "direction": str(it.get("direction") or ""),
                                "kind": str(it.get("kind") or ""),
                            }
                        )
                    self._message_log[chat_key] = cleaned
                    self._message_log_ids[chat_key] = seen

            logger.info(
                f"Loaded {len(self._trees)} trees and "
                f"{sum(len(v) for v in self._message_log.values())} msg_ids from {self.storage_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")

    def _snapshot(self) -> dict:
        """Snapshot current state for serialization. Caller must hold self._lock."""
        return {
            "trees": dict(self._trees),
            "node_to_tree": dict(self._node_to_tree),
            "message_log": {k: list(v) for k, v in self._message_log.items()},
        }

    def _write_data(self, data: dict) -> None:
        """Write data dict to disk. Must be called WITHOUT holding self._lock."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _schedule_save(self) -> None:
        """Schedule a debounced save. Caller must hold self._lock."""
        self._dirty = True
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        self._save_timer = threading.Timer(
            self._save_debounce_secs, self._save_from_timer
        )
        self._save_timer.daemon = True
        self._save_timer.start()

    def _save_from_timer(self) -> None:
        """Timer callback: save if dirty. Runs in timer thread."""
        with self._lock:
            if not self._dirty:
                self._save_timer = None
                return
            snapshot = self._snapshot()
            self._dirty = False
            self._save_timer = None
        try:
            self._write_data(snapshot)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
            with self._lock:
                self._dirty = True

    def _flush_save(self) -> dict:
        """Cancel pending timer and snapshot current state. Caller must hold self._lock.
        Returns snapshot dict; caller must call _write_data(snapshot) outside the lock."""
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        self._dirty = False
        return self._snapshot()

    def flush_pending_save(self) -> None:
        """Flush any pending debounced save. Call on shutdown to avoid losing data."""
        with self._lock:
            snapshot = self._flush_save()
        try:
            self._write_data(snapshot)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
            with self._lock:
                self._dirty = True

    def record_message_id(
        self,
        platform: str,
        chat_id: str,
        message_id: str,
        direction: str,
        kind: str,
    ) -> None:
        """Record a message_id for later best-effort deletion (/clear)."""
        if message_id is None:
            return

        chat_key = self._make_chat_key(str(platform), str(chat_id))
        mid = str(message_id)

        with self._lock:
            seen = self._message_log_ids.setdefault(chat_key, set())
            if mid in seen:
                return

            rec = {
                "message_id": mid,
                "ts": datetime.now(UTC).isoformat(),
                "direction": str(direction),
                "kind": str(kind),
            }
            self._message_log.setdefault(chat_key, []).append(rec)
            seen.add(mid)

            # Optional cap to prevent unbounded growth if configured.
            if self._message_log_cap is not None and self._message_log_cap > 0:
                items = self._message_log.get(chat_key, [])
                if len(items) > self._message_log_cap:
                    self._message_log[chat_key] = items[-self._message_log_cap :]
                    self._message_log_ids[chat_key] = {
                        str(x.get("message_id")) for x in self._message_log[chat_key]
                    }

            self._schedule_save()

    def get_message_ids_for_chat(self, platform: str, chat_id: str) -> list[str]:
        """Get all recorded message IDs for a chat (in insertion order)."""
        chat_key = self._make_chat_key(str(platform), str(chat_id))
        with self._lock:
            items = self._message_log.get(chat_key, [])
            return [
                str(x.get("message_id"))
                for x in items
                if x.get("message_id") is not None
            ]

    def clear_all(self) -> None:
        """Clear all stored sessions/trees/mappings and persist an empty store."""
        with self._lock:
            self._trees.clear()
            self._node_to_tree.clear()
            self._message_log.clear()
            self._message_log_ids.clear()
            snapshot = self._flush_save()
        try:
            self._write_data(snapshot)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
            with self._lock:
                self._dirty = True

    # ==================== Tree Methods ====================

    def save_tree(self, root_id: str, tree_data: dict) -> None:
        """
        Save a message tree.

        Args:
            root_id: Root node ID of the tree
            tree_data: Serialized tree data from tree.to_dict()
        """
        with self._lock:
            self._trees[root_id] = tree_data

            # Update node-to-tree mapping
            for node_id in tree_data.get("nodes", {}):
                self._node_to_tree[node_id] = root_id

            self._schedule_save()
            logger.debug(f"Saved tree {root_id}")

    def get_tree(self, root_id: str) -> dict | None:
        """Get a tree by its root ID."""
        with self._lock:
            return self._trees.get(root_id)

    def get_tree_root_for_node(self, node_id: str) -> str | None:
        """Get the root ID of the tree containing a node."""
        with self._lock:
            return self._node_to_tree.get(node_id)

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree root."""
        with self._lock:
            self._node_to_tree[node_id] = root_id
            self._schedule_save()

    def remove_node_mappings(self, node_ids: list[str]) -> None:
        """Remove node IDs from the node-to-tree mapping."""
        with self._lock:
            for nid in node_ids:
                self._node_to_tree.pop(nid, None)
            self._schedule_save()

    def remove_tree(self, root_id: str) -> None:
        """Remove a tree and all its node mappings from the store."""
        with self._lock:
            tree_data = self._trees.pop(root_id, None)
            if tree_data:
                for node_id in tree_data.get("nodes", {}):
                    self._node_to_tree.pop(node_id, None)
                self._schedule_save()

    def get_all_trees(self) -> dict[str, dict]:
        """Get all stored trees (public accessor)."""
        with self._lock:
            return dict(self._trees)

    def get_node_mapping(self) -> dict[str, str]:
        """Get the node-to-tree mapping (public accessor)."""
        with self._lock:
            return dict(self._node_to_tree)

    def sync_from_tree_data(
        self, trees: dict[str, dict], node_to_tree: dict[str, str]
    ) -> None:
        """Sync internal tree state from external data and persist."""
        with self._lock:
            self._trees = trees
            self._node_to_tree = node_to_tree
            self._schedule_save()

    def cleanup_old_trees(self, max_age_days: int = 30) -> int:
        """Remove trees older than max_age_days."""
        with self._lock:
            cutoff = datetime.now(UTC)
            removed = 0
            to_remove = []

            for root_id, tree_data in self._trees.items():
                try:
                    nodes = tree_data.get("nodes", {})
                    root_node = nodes.get(root_id, {})
                    created_str = root_node.get("created_at")
                    if created_str:
                        created = datetime.fromisoformat(created_str)
                        age_days = (cutoff - created).days
                        if age_days > max_age_days:
                            to_remove.append(root_id)
                except Exception:
                    pass

            for root_id in to_remove:
                tree_data = self._trees.pop(root_id)
                # Remove node mappings
                for node_id in tree_data.get("nodes", {}):
                    self._node_to_tree.pop(node_id, None)
                removed += 1

            if removed:
                self._schedule_save()
                logger.info(f"Cleaned up {removed} old trees")

            return removed

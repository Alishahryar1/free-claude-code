"""
Session Store for Messaging Platforms

Provides persistent storage for mapping platform messages to Claude CLI session IDs
and message trees for conversation continuation.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
import threading
from loguru import logger


@dataclass
class SessionRecord:
    """A single session record."""

    session_id: str
    chat_id: str
    initial_msg_id: str
    last_msg_id: str
    platform: str
    created_at: str
    updated_at: str


class SessionStore:
    """
    Persistent storage for message ↔ Claude session mappings and message trees.

    Uses a JSON file for storage with thread-safe operations.
    Platform-agnostic: works with any messaging platform.
    """

    def __init__(self, storage_path: str = "sessions.json"):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionRecord] = {}
        self._msg_to_session: Dict[
            str, str
        ] = {}  # "platform:chat_id:msg_id" -> session_id
        self._trees: Dict[str, dict] = {}  # root_id -> tree data
        self._node_to_tree: Dict[str, str] = {}  # node_id -> root_id
        # Per-chat message ID log used to support best-effort UI clearing (/clear).
        # Key: "{platform}:{chat_id}" -> list of records
        self._message_log: Dict[str, List[Dict[str, Any]]] = {}
        self._message_log_ids: Dict[str, set[str]] = {}
        self._dirty = False
        self._save_timer: Optional[threading.Timer] = None
        self._save_debounce_secs = 0.5
        self._load()

    def _make_key(self, platform: str, chat_id: str, msg_id: str) -> str:
        """Create a unique key from platform, chat_id and msg_id."""
        return f"{platform}:{chat_id}:{msg_id}"

    def _make_chat_key(self, platform: str, chat_id: str) -> str:
        return f"{platform}:{chat_id}"

    def _load(self) -> None:
        """Load sessions and trees from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load sessions (legacy support)
            for sid, record_data in data.get("sessions", {}).items():
                if "platform" not in record_data:
                    record_data["platform"] = "telegram"
                for field in ["chat_id", "initial_msg_id", "last_msg_id"]:
                    if isinstance(record_data.get(field), int):
                        record_data[field] = str(record_data[field])

                record = SessionRecord(**record_data)
                self._sessions[sid] = record
                self._msg_to_session[
                    self._make_key(
                        record.platform, record.chat_id, record.initial_msg_id
                    )
                ] = sid
                self._msg_to_session[
                    self._make_key(record.platform, record.chat_id, record.last_msg_id)
                ] = sid

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
                    cleaned: List[Dict[str, Any]] = []
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
                f"Loaded {len(self._sessions)} sessions, {len(self._trees)} trees, "
                f"and {sum(len(v) for v in self._message_log.values())} msg_ids from {self.storage_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")

    def _save(self) -> None:
        """Persist sessions and trees to disk. Caller must hold self._lock."""
        try:
            data = {
                "sessions": {
                    sid: asdict(record) for sid, record in self._sessions.items()
                },
                "trees": self._trees,
                "node_to_tree": self._node_to_tree,
                "message_log": self._message_log,
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")

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
            self._save()
            self._dirty = False
            self._save_timer = None

    def _flush_save(self) -> None:
        """Immediate save, cancel any pending debounced save. Caller must hold self._lock."""
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        self._dirty = False
        self._save()

    def flush_pending_save(self) -> None:
        """Flush any pending debounced save. Call on shutdown to avoid losing data."""
        with self._lock:
            self._flush_save()

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
                "ts": datetime.now(timezone.utc).isoformat(),
                "direction": str(direction),
                "kind": str(kind),
            }
            self._message_log.setdefault(chat_key, []).append(rec)
            seen.add(mid)

            # Optional cap to prevent unbounded growth if configured.
            # Default is unlimited as requested.
            try:
                cap_raw = os.getenv("MAX_MESSAGE_LOG_ENTRIES_PER_CHAT", "").strip()
                if cap_raw:
                    cap = int(cap_raw)
                    if cap > 0:
                        items = self._message_log.get(chat_key, [])
                        if len(items) > cap:
                            # Drop oldest entries and rebuild seen set.
                            self._message_log[chat_key] = items[-cap:]
                            self._message_log_ids[chat_key] = {
                                str(x.get("message_id"))
                                for x in self._message_log[chat_key]
                            }
            except Exception:
                pass

            self._schedule_save()

    def get_message_ids_for_chat(self, platform: str, chat_id: str) -> List[str]:
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
            self._sessions.clear()
            self._msg_to_session.clear()
            self._trees.clear()
            self._node_to_tree.clear()
            self._message_log.clear()
            self._message_log_ids.clear()
            self._flush_save()

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
            for node_id in tree_data.get("nodes", {}).keys():
                self._node_to_tree[node_id] = root_id

            self._schedule_save()
            logger.debug(f"Saved tree {root_id}")

    def get_tree(self, root_id: str) -> Optional[dict]:
        """Get a tree by its root ID."""
        with self._lock:
            return self._trees.get(root_id)

    def get_tree_root_for_node(self, node_id: str) -> Optional[str]:
        """Get the root ID of the tree containing a node."""
        with self._lock:
            return self._node_to_tree.get(node_id)

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree root."""
        with self._lock:
            self._node_to_tree[node_id] = root_id
            self._schedule_save()

    def get_all_trees(self) -> Dict[str, dict]:
        """Get all stored trees (public accessor)."""
        with self._lock:
            return dict(self._trees)

    def get_node_mapping(self) -> Dict[str, str]:
        """Get the node-to-tree mapping (public accessor)."""
        with self._lock:
            return dict(self._node_to_tree)

    def sync_from_tree_data(
        self, trees: Dict[str, dict], node_to_tree: Dict[str, str]
    ) -> None:
        """Sync internal tree state from external data and persist."""
        with self._lock:
            self._trees = trees
            self._node_to_tree = node_to_tree
            self._schedule_save()

    def cleanup_old_trees(self, max_age_days: int = 30) -> int:
        """Remove trees older than max_age_days."""
        with self._lock:
            cutoff = datetime.now(timezone.utc)
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
                for node_id in tree_data.get("nodes", {}).keys():
                    self._node_to_tree.pop(node_id, None)
                removed += 1

            if removed:
                self._schedule_save()
                logger.info(f"Cleaned up {removed} old trees")

            return removed

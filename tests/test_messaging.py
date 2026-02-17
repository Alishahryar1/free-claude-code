"""Tests for messaging/ module."""

import pytest
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

# --- Existing Tests ---


class TestMessagingModels:
    """Test messaging models."""

    def test_incoming_message_creation(self):
        """Test IncomingMessage dataclass."""
        from messaging.models import IncomingMessage

        msg = IncomingMessage(
            text="Hello",
            chat_id="123",
            user_id="456",
            message_id="789",
            platform="telegram",
        )
        assert msg.text == "Hello"
        assert msg.chat_id == "123"
        assert msg.platform == "telegram"
        assert msg.is_reply() is False

    def test_incoming_message_with_reply(self):
        """Test IncomingMessage as a reply."""
        from messaging.models import IncomingMessage

        msg = IncomingMessage(
            text="Reply text",
            chat_id="123",
            user_id="456",
            message_id="789",
            platform="discord",
            reply_to_message_id="100",
        )
        assert msg.is_reply() is True
        assert msg.reply_to_message_id == "100"


class TestMessagingBase:
    """Test MessagingPlatform ABC."""

    def test_platform_is_abstract(self):
        """Verify MessagingPlatform cannot be instantiated."""
        from messaging.base import MessagingPlatform

        with pytest.raises(TypeError):
            MessagingPlatform()


class TestSessionStore:
    """Test SessionStore."""

    def test_session_store_init(self, tmp_path):
        """Test SessionStore initialization."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        assert store._trees == {}

    # --- Tree Tests ---

    def test_save_and_get_tree(self, tmp_path):
        """Test saving and retrieving trees."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        tree_data = {
            "root": "r1",
            "nodes": {"r1": {"content": "root"}, "n1": {"content": "child"}},
        }
        store.save_tree("r1", tree_data)

        loaded = store.get_tree("r1")
        assert loaded == tree_data

        # Verify node mapping
        assert store.get_tree_root_for_node("r1") == "r1"
        assert store.get_tree_root_for_node("n1") == "r1"

    def test_register_node(self, tmp_path):
        """Test manual node registration."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        store.register_node("n_manual", "r_manual")
        assert store.get_tree_root_for_node("n_manual") == "r_manual"

    def test_cleanup_old_trees(self, tmp_path):
        """Test cleaning up expired trees."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        old_date = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()

        # Old tree
        store.save_tree(
            "old_root", {"nodes": {"old_root": {"created_at": old_date}, "child": {}}}
        )

        # New tree
        store.save_tree(
            "new_root",
            {
                "nodes": {
                    "new_root": {"created_at": datetime.now(timezone.utc).isoformat()}
                }
            },
        )

        removed = store.cleanup_old_trees(30)
        assert removed == 1

        assert store.get_tree("old_root") is None
        assert (
            store.get_tree_root_for_node("child") is None
        )  # Node mapping should be gone
        assert store.get_tree("new_root") is not None

    # --- Persistence & Edge Cases ---

    def test_load_existing_file_with_trees(self, tmp_path):
        """Test loading file with trees (legacy sessions ignored)."""
        from messaging.session import SessionStore

        data = {
            "sessions": {},
            "trees": {"r1": {"root_id": "r1", "nodes": {"r1": {}}}},
            "node_to_tree": {"r1": "r1"},
            "message_log": {},
        }

        p = tmp_path / "sessions.json"
        with open(p, "w") as f:
            json.dump(data, f)

        store = SessionStore(storage_path=str(p))
        assert store.get_tree("r1") is not None

    def test_load_corrupt_file(self, tmp_path):
        """Test loading corrupt/invalid json file."""
        p = tmp_path / "sessions.json"
        with open(p, "w") as f:
            f.write("{invalid json")

        from messaging.session import SessionStore

        # Should log error and start empty, avoiding crash
        store = SessionStore(storage_path=str(p))
        assert store._trees == {}

    def test_save_error_handling(self, tmp_path):
        """Test error during save."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        store.save_tree("r1", {"root_id": "r1", "nodes": {"r1": {}}})

        # Mock open to raise exception
        with patch("builtins.open", side_effect=IOError("Disk full")):
            store.save_tree("r2", {"root_id": "r2", "nodes": {"r2": {}}})

        # Should log error but not crash. Tree should be in memory.
        assert "r2" in store._trees


class TestTreeQueueManager:
    """Test TreeQueueManager."""

    def test_tree_queue_manager_init(self):
        """Test TreeQueueManager initialization."""
        from messaging.tree_queue import TreeQueueManager

        mgr = TreeQueueManager()
        assert mgr.get_tree_count() == 0

    def test_tree_not_busy_initially(self):
        """Test tree is not busy when no messages."""
        from messaging.tree_queue import TreeQueueManager

        mgr = TreeQueueManager()
        assert mgr.is_tree_busy("nonexistent") is False

    def test_get_queue_size_empty(self):
        """Test queue size is 0 for non-existent node."""
        from messaging.tree_queue import TreeQueueManager

        mgr = TreeQueueManager()
        assert mgr.get_queue_size("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_create_tree_and_enqueue(self):
        """Test creating a tree and enqueueing."""
        from messaging.tree_queue import TreeQueueManager
        from messaging.models import IncomingMessage

        mgr = TreeQueueManager()
        processed = []

        async def processor(node_id, node):
            processed.append(node_id)

        incoming = IncomingMessage(
            text="test", chat_id="1", user_id="1", message_id="1", platform="test"
        )

        await mgr.create_tree("1", incoming, "status_1")
        was_queued = await mgr.enqueue("1", processor)

        # First message should process immediately, not queue
        assert was_queued is False

    def test_cancel_tree_empty(self):
        """Test cancelling non-existent tree."""
        from messaging.tree_queue import TreeQueueManager

        mgr = TreeQueueManager()
        cancelled = mgr.cancel_tree("nonexistent")
        assert cancelled == []

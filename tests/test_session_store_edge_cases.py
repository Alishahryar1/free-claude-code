"""Edge case tests for messaging/session.py SessionStore."""

import json
import os
import pytest
from unittest.mock import patch

from messaging.session import SessionStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a SessionStore using a temp file."""
    path = str(tmp_path / "sessions.json")
    return SessionStore(storage_path=path)


class TestSessionStoreLoadEdgeCases:
    """Tests for loading corrupted/malformed data."""

    def test_load_corrupted_json(self, tmp_path):
        """Corrupted JSON file is handled gracefully (logs error, starts empty)."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write("{invalid json")

        store = SessionStore(storage_path=path)
        assert len(store._sessions) == 0
        assert len(store._trees) == 0

    def test_load_truncated_json(self, tmp_path):
        """Truncated JSON file is handled gracefully."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write('{"sessions": {"s1": {"session_id": "s1"')

        store = SessionStore(storage_path=path)
        assert len(store._sessions) == 0

    def test_load_empty_file(self, tmp_path):
        """Empty file is handled gracefully."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write("")

        store = SessionStore(storage_path=path)
        assert len(store._sessions) == 0

    def test_load_nonexistent_file(self, tmp_path):
        """Non-existent file starts with empty state."""
        path = str(tmp_path / "nonexistent.json")
        store = SessionStore(storage_path=path)
        assert len(store._sessions) == 0

    def test_load_legacy_int_fields(self, tmp_path):
        """Legacy format with int chat_id/msg_id is converted to string."""
        path = str(tmp_path / "sessions.json")
        data = {
            "sessions": {
                "s1": {
                    "session_id": "s1",
                    "chat_id": 12345,
                    "initial_msg_id": 100,
                    "last_msg_id": 200,
                    "platform": "telegram",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "updated_at": "2025-01-01T00:00:00+00:00",
                }
            },
            "trees": {},
            "node_to_tree": {},
        }
        with open(path, "w") as f:
            json.dump(data, f)

        store = SessionStore(storage_path=path)
        record = store._sessions["s1"]
        assert record.chat_id == "12345"
        assert record.initial_msg_id == "100"
        assert record.last_msg_id == "200"


class TestSessionStoreSaveEdgeCases:
    """Tests for save failure handling."""

    def test_save_io_error_handled(self, tmp_store):
        """Write failure in _save() is logged but doesn't raise."""
        tmp_store.save_session("s1", "c1", "m1")
        # Make the path read-only dir to trigger write error
        with patch("builtins.open", side_effect=IOError("disk full")):
            # _save() catches all exceptions and logs them
            tmp_store._save()
            # Should not raise


class TestSessionStoreOperationEdgeCases:
    """Tests for edge cases in session operations."""

    def test_rename_session_not_found(self, tmp_store):
        """rename_session with non-existent old_id returns False."""
        result = tmp_store.rename_session("nonexistent", "new_id")
        assert result is False

    def test_update_last_message_not_found(self, tmp_store):
        """update_last_message with unknown session_id logs warning."""
        # Should not raise
        tmp_store.update_last_message("nonexistent", "msg_1")

    def test_get_session_by_msg_not_found(self, tmp_store):
        """Looking up non-existent message returns None."""
        result = tmp_store.get_session_by_msg("c1", "m999")
        assert result is None

    def test_get_session_record_not_found(self, tmp_store):
        """Getting non-existent session record returns None."""
        result = tmp_store.get_session_record("nonexistent")
        assert result is None


class TestSessionStoreCleanupEdgeCases:
    """Tests for cleanup with malformed data."""

    def test_cleanup_sessions_malformed_timestamp(self, tmp_store):
        """Malformed created_at in cleanup doesn't crash."""
        tmp_store.save_session("s1", "c1", "m1")
        # Corrupt the timestamp
        tmp_store._sessions["s1"].created_at = "not-a-date"
        # Should not crash - the except block silently skips bad records
        removed = tmp_store.cleanup_old_sessions(max_age_days=0)
        assert removed == 0  # Skipped due to parse error

    def test_cleanup_trees_malformed_timestamp(self, tmp_store):
        """Malformed created_at in cleanup_old_trees doesn't crash."""
        tmp_store._trees["root1"] = {"nodes": {"root1": {"created_at": "bad-date"}}}
        removed = tmp_store.cleanup_old_trees(max_age_days=0)
        assert removed == 0  # Skipped due to parse error

    def test_cleanup_trees_missing_created_at(self, tmp_store):
        """Tree node without created_at is not cleaned up."""
        tmp_store._trees["root1"] = {"nodes": {"root1": {}}}
        removed = tmp_store.cleanup_old_trees(max_age_days=0)
        assert removed == 0

    def test_update_tree_node_nonexistent_tree(self, tmp_store):
        """Updating a node in a nonexistent tree logs warning."""
        tmp_store.update_tree_node("nonexistent", "node1", {"data": "test"})
        # Should not crash, tree not created
        assert "nonexistent" not in tmp_store._trees

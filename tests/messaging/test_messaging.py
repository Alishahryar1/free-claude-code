"""Tests for messaging/ module."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Existing Tests ---


class TestMessagingModels:
    """Test messaging models."""

    def test_incoming_message_creation(self):
        """Test IncomingMessage dataclass."""
        from free_claude_code.messaging.models import IncomingMessage

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
        from free_claude_code.messaging.models import IncomingMessage

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


class TestMessagingPorts:
    """Test explicit messaging platform component ports."""

    def test_components_bundle_runtime_and_outbound(self):
        """Verify the factory handoff shape is explicit."""
        from free_claude_code.messaging.platforms.ports import (
            MessagingPlatformComponents,
        )

        runtime = MagicMock()
        runtime.name = "telegram"
        runtime.start = AsyncMock()
        runtime.quiesce = AsyncMock()
        runtime.close = AsyncMock()
        runtime.on_message = MagicMock()
        outbound = MagicMock()
        outbound.queue_send_message = AsyncMock()
        outbound.queue_edit_message = AsyncMock()
        outbound.queue_delete_messages = AsyncMock()
        outbound.fire_and_forget = MagicMock()
        components = MessagingPlatformComponents(
            name="telegram",
            runtime=runtime,
            outbound=outbound,
            voice_cancellation=None,
        )
        assert components.runtime is runtime
        assert components.outbound is outbound


class TestSessionStore:
    """Test SessionStore."""

    def test_session_store_init(self, tmp_path):
        """Test SessionStore initialization."""
        from free_claude_code.messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        assert store.load_conversation_snapshot().is_empty

    # --- Tree Tests ---

    def test_save_and_get_tree(self, tmp_path):
        """Test saving and retrieving trees."""
        from free_claude_code.messaging.models import MessageScope
        from free_claude_code.messaging.session import SessionStore
        from free_claude_code.messaging.trees import TreeIdentity, TreeSnapshot

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        scope = MessageScope(platform="telegram", chat_id="chat")

        tree_data = {
            "scope": {"platform": scope.platform, "chat_id": scope.chat_id},
            "root_id": "r1",
            "nodes": {
                "r1": {"node_id": "r1", "status_message_id": "s1"},
                "n1": {"node_id": "n1", "status_message_id": "s2"},
            },
        }
        snapshot = TreeSnapshot.from_json(tree_data)
        assert snapshot is not None
        store.save_tree_snapshot(snapshot)

        identity = TreeIdentity(scope=scope, root_id="r1")
        loaded = store.load_conversation_snapshot().get_tree(identity)
        assert loaded is not None
        assert loaded == snapshot
        assert loaded.lookup_ids() == {"r1", "s1", "n1", "s2"}

    # --- Persistence & Edge Cases ---

    def test_load_existing_file_with_trees(self, tmp_path):
        """Test loading file with trees (legacy sessions ignored)."""
        from free_claude_code.messaging.models import MessageScope
        from free_claude_code.messaging.session import SessionStore
        from free_claude_code.messaging.trees import TreeIdentity

        data = {
            "sessions": {},
            "trees": {
                "r1": {
                    "root_id": "r1",
                    "nodes": {
                        "r1": {
                            "node_id": "r1",
                            "incoming": {
                                "platform": "telegram",
                                "chat_id": "chat",
                            },
                        }
                    },
                }
            },
            "node_to_tree": {"r1": "r1"},
            "message_log": {},
        }

        p = tmp_path / "sessions.json"
        with open(p, "w") as f:
            json.dump(data, f)

        store = SessionStore(storage_path=str(p))
        identity = TreeIdentity(
            scope=MessageScope(platform="telegram", chat_id="chat"),
            root_id="r1",
        )
        assert store.load_conversation_snapshot().get_tree(identity) is not None

    def test_load_corrupt_file(self, tmp_path):
        """Test loading corrupt/invalid json file."""
        p = tmp_path / "sessions.json"
        with open(p, "w") as f:
            f.write("{invalid json")

        from free_claude_code.messaging.session import SessionStore

        # Should log error and start empty, avoiding crash
        store = SessionStore(storage_path=str(p))
        assert store.load_conversation_snapshot().is_empty

    def test_save_error_handling(self, tmp_path):
        """Test error during save."""
        from free_claude_code.messaging.models import MessageScope
        from free_claude_code.messaging.session import SessionStore
        from free_claude_code.messaging.trees import TreeIdentity, TreeSnapshot

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        scope = MessageScope(platform="telegram", chat_id="chat")
        snapshot = TreeSnapshot(scope=scope, root_id="r1", nodes={"r1": {}})
        store.save_tree_snapshot(snapshot)

        with (
            patch(
                "free_claude_code.messaging.session.persistence.os.replace",
                side_effect=OSError("Disk full"),
            ),
            pytest.raises(OSError, match="Disk full"),
        ):
            store.flush_pending_save()

        assert store.dirty is True
        identity = TreeIdentity(scope=scope, root_id="r1")
        assert store.load_conversation_snapshot().get_tree(identity) is not None


class TestTreeQueueManager:
    """Test TreeQueueManager."""

    def test_tree_queue_manager_init(self):
        from free_claude_code.messaging.trees import TreeQueueManager

        async def process(_claim):
            return None

        mgr = TreeQueueManager(process)
        assert mgr.get_tree_count() == 0

    @pytest.mark.asyncio
    async def test_admit_creates_tree_and_claim(self):
        from free_claude_code.messaging.models import IncomingMessage
        from free_claude_code.messaging.trees import TreeQueueManager

        processed = asyncio.Event()

        async def processor(claim):
            assert claim.node.node_id == "1"
            processed.set()

        incoming = IncomingMessage(
            text="test",
            chat_id="1",
            user_id="1",
            message_id="1",
            platform="test",
        )

        mgr = TreeQueueManager(processor)
        decision = await mgr.admit(incoming, "status_1")

        assert decision.accepted is True
        assert decision.claim is not None
        await processed.wait()

    @pytest.mark.asyncio
    async def test_cancel_unknown_node_is_empty(self):
        from free_claude_code.messaging.models import MessageScope
        from free_claude_code.messaging.trees import TreeQueueManager

        async def process(_claim):
            return None

        mgr = TreeQueueManager(process)
        scope = MessageScope(platform="test", chat_id="1")
        cancelled = await mgr.cancel_node(scope, "nonexistent")
        assert cancelled.effects == ()

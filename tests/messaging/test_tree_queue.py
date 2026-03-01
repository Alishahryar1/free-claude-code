"""Tests for tree-based message queue system."""

import asyncio

import pytest

from messaging.models import IncomingMessage
from messaging.trees.queue_manager import (
    MessageNode,
    MessageState,
    MessageTree,
    TreeQueueManager,
)


class TestMessageState:
    """Test MessageState enum."""

    def test_state_values(self):
        """Test state enum values."""
        assert MessageState.PENDING.value == "pending"
        assert MessageState.IN_PROGRESS.value == "in_progress"
        assert MessageState.COMPLETED.value == "completed"
        assert MessageState.ERROR.value == "error"


class TestMessageNode:
    """Test MessageNode dataclass."""

    def test_node_creation(self):
        """Test creating a message node."""
        incoming = IncomingMessage(
            text="Hello",
            chat_id="123",
            user_id="456",
            message_id="789",
            platform="telegram",
        )
        node = MessageNode(
            node_id="789",
            incoming=incoming,
            status_message_id="status_1",
        )

        assert node.node_id == "789"
        assert node.state == MessageState.PENDING
        assert node.parent_id is None
        assert node.children_ids == []
        assert node.session_id is None

    def test_node_to_dict(self):
        """Test serializing a node."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="2",
            message_id="3",
            platform="test",
        )
        node = MessageNode(
            node_id="3",
            incoming=incoming,
            status_message_id="s1",
            state=MessageState.COMPLETED,
            session_id="sess_123",
        )

        data = node.to_dict()
        assert data["node_id"] == "3"
        assert data["state"] == "completed"
        assert data["session_id"] == "sess_123"

    def test_node_from_dict(self):
        """Test deserializing a node."""
        data = {
            "node_id": "n1",
            "incoming": {
                "text": "Hello",
                "chat_id": "c1",
                "user_id": "u1",
                "message_id": "m1",
                "platform": "test",
            },
            "status_message_id": "s1",
            "state": "in_progress",
            "parent_id": "parent_1",
            "session_id": None,
            "children_ids": ["child_1"],
            "created_at": "2025-01-01T00:00:00",
        }

        node = MessageNode.from_dict(data)
        assert node.node_id == "n1"
        assert node.state == MessageState.IN_PROGRESS
        assert node.parent_id == "parent_1"
        assert "child_1" in node.children_ids


class TestMessageTree:
    """Test MessageTree class."""

    def test_tree_creation(self):
        """Test creating a tree with root node."""
        incoming = IncomingMessage(
            text="Root",
            chat_id="1",
            user_id="1",
            message_id="root_msg",
            platform="test",
        )
        root = MessageNode(
            node_id="root_msg",
            incoming=incoming,
            status_message_id="status_1",
        )

        tree = MessageTree(root)
        assert tree.root_id == "root_msg"
        assert tree.get_node("root_msg") is not None
        assert tree.is_processing is False

    @pytest.mark.asyncio
    async def test_add_child_node(self):
        """Test adding a child node to the tree."""
        # Create root
        root_incoming = IncomingMessage(
            text="Root",
            chat_id="1",
            user_id="1",
            message_id="root",
            platform="test",
        )
        root = MessageNode(
            node_id="root",
            incoming=root_incoming,
            status_message_id="s1",
        )
        tree = MessageTree(root)

        # Add child
        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        child = await tree.add_node(
            node_id="child",
            incoming=child_incoming,
            status_message_id="s2",
            parent_id="root",
        )

        assert child.node_id == "child"
        assert child.parent_id == "root"
        assert "child" in tree.get_root().children_ids
        parent = tree.get_parent("child")
        assert parent is not None
        assert parent.node_id == "root"

    @pytest.mark.asyncio
    async def test_update_state(self):
        """Test updating node state."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        root = MessageNode(node_id="m1", incoming=incoming, status_message_id="s1")
        tree = MessageTree(root)

        await tree.update_state("m1", MessageState.IN_PROGRESS)
        node = tree.get_node("m1")
        assert node is not None
        assert node.state == MessageState.IN_PROGRESS

        await tree.update_state("m1", MessageState.COMPLETED, session_id="sess_abc")
        node = tree.get_node("m1")
        assert node is not None
        assert node.state == MessageState.COMPLETED
        assert node.session_id == "sess_abc"
        assert node.completed_at is not None

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self):
        """Test queue operations."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        root = MessageNode(node_id="m1", incoming=incoming, status_message_id="s1")
        tree = MessageTree(root)

        # Enqueue
        pos = await tree.enqueue("m1")
        assert pos == 1
        assert tree.get_queue_size() == 1

        # Dequeue
        node_id = await tree.dequeue()
        assert node_id == "m1"
        assert tree.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_queue_snapshot(self):
        """Test queue snapshot order."""
        incoming = IncomingMessage(
            text="Root",
            chat_id="1",
            user_id="1",
            message_id="root",
            platform="test",
        )
        root = MessageNode(node_id="root", incoming=incoming, status_message_id="s1")
        tree = MessageTree(root)

        child_incoming_1 = IncomingMessage(
            text="Child 1",
            chat_id="1",
            user_id="1",
            message_id="child_1",
            platform="test",
            reply_to_message_id="root",
        )
        child_incoming_2 = IncomingMessage(
            text="Child 2",
            chat_id="1",
            user_id="1",
            message_id="child_2",
            platform="test",
            reply_to_message_id="root",
        )

        await tree.add_node(
            node_id="child_1",
            incoming=child_incoming_1,
            status_message_id="s2",
            parent_id="root",
        )
        await tree.add_node(
            node_id="child_2",
            incoming=child_incoming_2,
            status_message_id="s3",
            parent_id="root",
        )

        await tree.enqueue("child_1")
        await tree.enqueue("child_2")

        snapshot = await tree.get_queue_snapshot()
        assert snapshot == ["child_1", "child_2"]

    def test_tree_serialization(self):
        """Test tree to_dict and from_dict."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        root = MessageNode(
            node_id="m1",
            incoming=incoming,
            status_message_id="s1",
            state=MessageState.COMPLETED,
            session_id="sess_1",
        )
        tree = MessageTree(root)

        data = tree.to_dict()
        restored = MessageTree.from_dict(data)

        assert restored.root_id == "m1"
        node = restored.get_node("m1")
        assert node is not None
        assert node.session_id == "sess_1"

    @pytest.mark.asyncio
    async def test_get_descendants(self):
        """Test get_descendants returns node and all descendants."""
        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        root = MessageNode(
            node_id="root", incoming=root_incoming, status_message_id="s1"
        )
        tree = MessageTree(root)

        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        await tree.add_node("child", child_incoming, "s2", "root")

        grandchild_incoming = IncomingMessage(
            text="Grand",
            chat_id="1",
            user_id="1",
            message_id="grand",
            platform="test",
            reply_to_message_id="child",
        )
        await tree.add_node("grand", grandchild_incoming, "s3", "child")

        assert tree.get_descendants("root") == ["root", "child", "grand"]
        assert tree.get_descendants("child") == ["child", "grand"]
        assert tree.get_descendants("grand") == ["grand"]
        assert tree.get_descendants("nonexistent") == []

    @pytest.mark.asyncio
    async def test_remove_branch(self):
        """Test remove_branch removes subtree and updates parent."""
        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        root = MessageNode(
            node_id="root", incoming=root_incoming, status_message_id="s1"
        )
        tree = MessageTree(root)

        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        await tree.add_node("child", child_incoming, "s2", "root")

        grandchild_incoming = IncomingMessage(
            text="Grand",
            chat_id="1",
            user_id="1",
            message_id="grand",
            platform="test",
            reply_to_message_id="child",
        )
        await tree.add_node("grand", grandchild_incoming, "s3", "child")

        async with tree.with_lock():
            removed = tree.remove_branch("child")

        assert len(removed) == 2
        assert {n.node_id for n in removed} == {"child", "grand"}
        assert tree.get_node("child") is None
        assert tree.get_node("grand") is None
        assert tree.get_node("root") is not None
        assert "child" not in tree.get_root().children_ids


class TestTreeQueueManager:
    """Test TreeQueueManager class."""

    @pytest.mark.asyncio
    async def test_create_tree(self):
        """Test creating a new tree."""
        manager = TreeQueueManager()

        incoming = IncomingMessage(
            text="New message",
            chat_id="1",
            user_id="1",
            message_id="msg_1",
            platform="test",
        )

        tree = await manager.create_tree(
            node_id="msg_1",
            incoming=incoming,
            status_message_id="status_1",
        )

        assert tree is not None
        assert tree.root_id == "msg_1"
        assert manager.get_tree("msg_1") is tree

    @pytest.mark.asyncio
    async def test_add_reply_to_tree(self):
        """Test adding a reply to existing tree."""
        manager = TreeQueueManager()

        # Create root
        root_incoming = IncomingMessage(
            text="Root",
            chat_id="1",
            user_id="1",
            message_id="root",
            platform="test",
        )
        await manager.create_tree("root", root_incoming, "s1")

        # Add reply
        reply_incoming = IncomingMessage(
            text="Reply",
            chat_id="1",
            user_id="1",
            message_id="reply",
            platform="test",
            reply_to_message_id="root",
        )
        tree, node = await manager.add_to_tree(
            parent_node_id="root",
            node_id="reply",
            incoming=reply_incoming,
            status_message_id="s2",
        )

        assert node.parent_id == "root"
        assert manager.get_tree_for_node("reply") is tree

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self):
        """Test enqueueing and processing."""
        manager = TreeQueueManager()
        processed = []

        async def processor(node_id, node):
            processed.append(node_id)
            await asyncio.sleep(0.01)  # Simulate work

        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        await manager.create_tree("m1", incoming, "s1")

        was_queued = await manager.enqueue("m1", processor)
        assert was_queued is False  # First message processes immediately

        # Wait for processing
        await asyncio.sleep(0.1)
        assert "m1" in processed

    @pytest.mark.asyncio
    async def test_queue_when_busy(self):
        """Test that messages queue when tree is busy."""
        manager = TreeQueueManager()
        processing_started = asyncio.Event()
        processing_complete = asyncio.Event()

        async def slow_processor(node_id, node):
            processing_started.set()
            await processing_complete.wait()

        # Create tree with root
        root_incoming = IncomingMessage(
            text="Root",
            chat_id="1",
            user_id="1",
            message_id="root",
            platform="test",
        )
        await manager.create_tree("root", root_incoming, "s1")

        # Start processing root
        was_queued = await manager.enqueue("root", slow_processor)
        assert was_queued is False

        # Wait for processing to start
        await processing_started.wait()

        # Add a child
        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        await manager.add_to_tree("root", "child", child_incoming, "s2")

        # Try to enqueue child - should be queued since tree is busy
        was_queued = await manager.enqueue("child", slow_processor)
        assert was_queued is True
        assert manager.get_queue_size("child") == 1

        # Cleanup
        processing_complete.set()

    @pytest.mark.asyncio
    async def test_cancel_tree(self):
        """Test cancelling a tree."""
        manager = TreeQueueManager()
        processing_complete = asyncio.Event()

        async def slow_processor(node_id, node):
            await processing_complete.wait()

        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        await manager.create_tree("m1", incoming, "s1")
        await manager.enqueue("m1", slow_processor)

        # Cancel
        cancelled = await manager.cancel_tree("m1")
        assert len(cancelled) == 1

        processing_complete.set()

    @pytest.mark.asyncio
    async def test_cancel_branch(self):
        """Test cancel_branch cancels only nodes in subtree."""
        manager = TreeQueueManager()

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        await manager.create_tree("root", root_incoming, "s1")

        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        tree, _ = await manager.add_to_tree("root", "child", child_incoming, "s2")

        sibling_incoming = IncomingMessage(
            text="Sibling",
            chat_id="1",
            user_id="1",
            message_id="sibling",
            platform="test",
            reply_to_message_id="root",
        )
        await manager.add_to_tree("root", "sibling", sibling_incoming, "s3")

        cancelled = await manager.cancel_branch("child")
        assert len(cancelled) == 1
        assert cancelled[0].node_id == "child"

        child_node = tree.get_node("child")
        assert child_node is not None
        assert child_node.state == MessageState.ERROR

        sibling_node = tree.get_node("sibling")
        assert sibling_node is not None
        assert sibling_node.state == MessageState.PENDING

    @pytest.mark.asyncio
    async def test_remove_branch_non_root(self):
        """Test remove_branch removes only the subtree when branch is not root."""
        manager = TreeQueueManager()

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        await manager.create_tree("root", root_incoming, "s1")

        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        tree, _ = await manager.add_to_tree("root", "child", child_incoming, "s2")

        removed, root_id, removed_entire = await manager.remove_branch("child")

        assert len(removed) == 1
        assert removed[0].node_id == "child"
        assert root_id == "root"
        assert removed_entire is False
        assert manager.get_tree_for_node("child") is None
        assert manager.get_tree("root") is not None
        assert tree.get_node("child") is None
        assert "child" not in tree.get_root().children_ids

    @pytest.mark.asyncio
    async def test_remove_branch_root_removes_tree(self):
        """Test remove_branch when branch is root removes entire tree."""
        manager = TreeQueueManager()

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        await manager.create_tree("root", root_incoming, "s1")

        removed, root_id, removed_entire = await manager.remove_branch("root")

        assert len(removed) == 1
        assert root_id == "root"
        assert removed_entire is True
        assert manager.get_tree("root") is None
        assert manager.get_tree_for_node("root") is None


class TestSessionStoreTrees:
    """Test SessionStore tree methods."""

    def test_save_and_get_tree(self, tmp_path):
        """Test saving and retrieving a tree."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        tree_data = {
            "root_id": "root_1",
            "nodes": {
                "root_1": {
                    "node_id": "root_1",
                    "state": "completed",
                    "session_id": "sess_abc",
                }
            },
        }

        store.save_tree("root_1", tree_data)

        retrieved = store.get_tree("root_1")
        assert retrieved is not None
        assert retrieved["root_id"] == "root_1"

    def test_get_tree_by_root_id(self, tmp_path):
        """Test getting tree by root ID and node mapping."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        tree_data = {
            "root_id": "root",
            "nodes": {
                "root": {"node_id": "root"},
                "child": {"node_id": "child"},
            },
        }

        store.save_tree("root", tree_data)

        retrieved = store.get_tree("root")
        assert retrieved is not None
        assert retrieved["root_id"] == "root"
        assert store.get_node_mapping()["child"] == "root"

    def test_register_node(self, tmp_path):
        """Test registering a node to a tree."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        store.register_node("new_node", "root_tree")

        assert store.get_node_mapping()["new_node"] == "root_tree"

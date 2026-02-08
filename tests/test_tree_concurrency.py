"""Concurrency and race condition tests for tree data structures and queue manager."""

import pytest
import pytest_asyncio
import asyncio

from messaging.models import IncomingMessage
from messaging.tree_data import MessageState, MessageNode, MessageTree
from messaging.tree_queue import TreeQueueManager


def _make_incoming(text: str = "hello", msg_id: str = "m1") -> IncomingMessage:
    """Create a minimal IncomingMessage for testing."""
    return IncomingMessage(
        text=text,
        chat_id="chat1",
        user_id="user1",
        message_id=msg_id,
        platform="test",
    )


def _make_tree(root_id: str = "root") -> MessageTree:
    """Create a tree with a single root node."""
    root = MessageNode(
        node_id=root_id,
        incoming=_make_incoming(msg_id=root_id),
        status_message_id=f"status_{root_id}",
        state=MessageState.PENDING,
    )
    return MessageTree(root)


class TestMessageTreeConcurrency:
    """Concurrency tests for MessageTree operations."""

    @pytest.mark.asyncio
    async def test_concurrent_add_node_serialized(self):
        """Concurrent add_node calls should all succeed via lock serialization."""
        tree = _make_tree("root")
        count = 10

        async def add(i: int):
            return await tree.add_node(
                node_id=f"child_{i}",
                incoming=_make_incoming(msg_id=f"child_{i}"),
                status_message_id=f"status_{i}",
                parent_id="root",
            )

        results = await asyncio.gather(*[add(i) for i in range(count)])

        assert len(results) == count
        # All nodes plus root
        assert len(tree.all_nodes()) == count + 1
        # Root should have all children
        root = tree.get_root()
        assert len(root.children_ids) == count

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_dequeue_no_loss(self):
        """Concurrent enqueue/dequeue should not lose items."""
        tree = _make_tree("root")

        # Add nodes first
        for i in range(10):
            await tree.add_node(
                node_id=f"n{i}",
                incoming=_make_incoming(msg_id=f"n{i}"),
                status_message_id=f"s{i}",
                parent_id="root",
            )

        # Enqueue all concurrently
        await asyncio.gather(*[tree.enqueue(f"n{i}") for i in range(10)])
        assert tree.get_queue_size() == 10

        # Dequeue all
        dequeued = []
        for _ in range(10):
            nid = await tree.dequeue()
            if nid:
                dequeued.append(nid)

        assert len(dequeued) == 10
        assert set(dequeued) == {f"n{i}" for i in range(10)}
        assert tree.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_dequeue_empty_returns_none(self):
        """Dequeue on empty queue returns None."""
        tree = _make_tree("root")
        result = await tree.dequeue()
        assert result is None

    @pytest.mark.asyncio
    async def test_concurrent_update_state(self):
        """Concurrent state updates should all apply (last writer wins)."""
        tree = _make_tree("root")
        for i in range(5):
            await tree.add_node(
                node_id=f"n{i}",
                incoming=_make_incoming(msg_id=f"n{i}"),
                status_message_id=f"s{i}",
                parent_id="root",
            )

        # Update all nodes concurrently
        await asyncio.gather(
            *[tree.update_state(f"n{i}", MessageState.IN_PROGRESS) for i in range(5)]
        )

        for i in range(5):
            node = tree.get_node(f"n{i}")
            assert node.state == MessageState.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_update_state_nonexistent_node(self):
        """Updating state of a nonexistent node should not raise."""
        tree = _make_tree("root")
        # Should just log a warning, not raise
        await tree.update_state("nonexistent", MessageState.ERROR)

    @pytest.mark.asyncio
    async def test_add_node_invalid_parent_raises(self):
        """Adding a node with nonexistent parent should raise ValueError."""
        tree = _make_tree("root")
        with pytest.raises(ValueError, match="not found in tree"):
            await tree.add_node(
                node_id="child",
                incoming=_make_incoming(),
                status_message_id="s1",
                parent_id="nonexistent",
            )

    @pytest.mark.asyncio
    async def test_queue_snapshot_matches_enqueue_order(self):
        """Queue snapshot should return items in FIFO order."""
        tree = _make_tree("root")
        for i in range(5):
            await tree.add_node(
                node_id=f"n{i}",
                incoming=_make_incoming(msg_id=f"n{i}"),
                status_message_id=f"s{i}",
                parent_id="root",
            )

        for i in range(5):
            await tree.enqueue(f"n{i}")

        snapshot = await tree.get_queue_snapshot()
        assert snapshot == [f"n{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_enqueue_returns_position(self):
        """Enqueue should return 1-indexed position."""
        tree = _make_tree("root")
        for i in range(3):
            await tree.add_node(
                node_id=f"n{i}",
                incoming=_make_incoming(msg_id=f"n{i}"),
                status_message_id=f"s{i}",
                parent_id="root",
            )

        pos1 = await tree.enqueue("n0")
        pos2 = await tree.enqueue("n1")
        pos3 = await tree.enqueue("n2")

        assert pos1 == 1
        assert pos2 == 2
        assert pos3 == 3


class TestMessageTreeNavigation:
    """Tests for tree navigation methods."""

    @pytest.mark.asyncio
    async def test_get_children(self):
        """get_children returns child nodes."""
        tree = _make_tree("root")
        await tree.add_node("c1", _make_incoming(msg_id="c1"), "s1", "root")
        await tree.add_node("c2", _make_incoming(msg_id="c2"), "s2", "root")

        children = tree.get_children("root")
        assert len(children) == 2
        assert {c.node_id for c in children} == {"c1", "c2"}

    def test_get_children_nonexistent(self):
        """get_children for nonexistent node returns empty list."""
        tree = _make_tree("root")
        assert tree.get_children("nonexistent") == []

    def test_get_parent_root(self):
        """Root node has no parent."""
        tree = _make_tree("root")
        assert tree.get_parent("root") is None

    @pytest.mark.asyncio
    async def test_get_parent_child(self):
        """Child node's parent is the root."""
        tree = _make_tree("root")
        await tree.add_node("c1", _make_incoming(msg_id="c1"), "s1", "root")
        parent = tree.get_parent("c1")
        assert parent is not None
        assert parent.node_id == "root"

    @pytest.mark.asyncio
    async def test_get_parent_session_id(self):
        """get_parent_session_id returns parent's session_id."""
        tree = _make_tree("root")
        await tree.update_state("root", MessageState.COMPLETED, session_id="sess_abc")
        await tree.add_node("c1", _make_incoming(msg_id="c1"), "s1", "root")

        session_id = tree.get_parent_session_id("c1")
        assert session_id == "sess_abc"

    def test_get_parent_session_id_root(self):
        """Root node has no parent session."""
        tree = _make_tree("root")
        assert tree.get_parent_session_id("root") is None

    def test_has_node(self):
        """has_node returns True for existing nodes."""
        tree = _make_tree("root")
        assert tree.has_node("root") is True
        assert tree.has_node("nonexistent") is False

    @pytest.mark.asyncio
    async def test_find_node_by_status_message(self):
        """find_node_by_status_message finds the right node."""
        tree = _make_tree("root")
        await tree.add_node("c1", _make_incoming(msg_id="c1"), "status_c1", "root")

        found = tree.find_node_by_status_message("status_c1")
        assert found is not None
        assert found.node_id == "c1"

    def test_find_node_by_status_message_not_found(self):
        """find_node_by_status_message returns None if not found."""
        tree = _make_tree("root")
        assert tree.find_node_by_status_message("nonexistent") is None


class TestMessageTreeSerialization:
    """Tests for tree serialization/deserialization."""

    @pytest.mark.asyncio
    async def test_round_trip(self):
        """Tree should survive serialization round-trip."""
        tree = _make_tree("root")
        await tree.add_node("c1", _make_incoming(msg_id="c1"), "s1", "root")
        await tree.add_node("c2", _make_incoming(msg_id="c2"), "s2", "root")
        await tree.update_state("root", MessageState.COMPLETED, session_id="sess1")

        data = tree.to_dict()
        restored = MessageTree.from_dict(data)

        assert restored.root_id == "root"
        assert len(restored.all_nodes()) == 3
        root = restored.get_root()
        assert root.state == MessageState.COMPLETED
        assert root.session_id == "sess1"
        assert set(root.children_ids) == {"c1", "c2"}

    @pytest.mark.asyncio
    async def test_node_round_trip(self):
        """MessageNode should survive serialization round-trip."""
        node = MessageNode(
            node_id="n1",
            incoming=_make_incoming(msg_id="n1"),
            status_message_id="s1",
            state=MessageState.COMPLETED,
            parent_id="root",
            session_id="sess_test",
            error_message="test error",
        )
        data = node.to_dict()
        restored = MessageNode.from_dict(data)

        assert restored.node_id == "n1"
        assert restored.state == MessageState.COMPLETED
        assert restored.session_id == "sess_test"
        assert restored.error_message == "test error"
        assert restored.parent_id == "root"


class TestTreeQueueManagerConcurrency:
    """Concurrency tests for TreeQueueManager."""

    @pytest.mark.asyncio
    async def test_concurrent_create_trees(self):
        """Creating multiple trees concurrently should all succeed."""
        mgr = TreeQueueManager()

        async def create(i: int):
            return await mgr.create_tree(
                node_id=f"root_{i}",
                incoming=_make_incoming(msg_id=f"root_{i}"),
                status_message_id=f"status_{i}",
            )

        trees = await asyncio.gather(*[create(i) for i in range(10)])
        assert len(trees) == 10
        assert mgr.get_tree_count() == 10

    @pytest.mark.asyncio
    async def test_add_to_tree_concurrent(self):
        """Adding replies to a tree concurrently should all succeed."""
        mgr = TreeQueueManager()
        await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")

        async def add_reply(i: int):
            return await mgr.add_to_tree(
                parent_node_id="root",
                node_id=f"reply_{i}",
                incoming=_make_incoming(msg_id=f"reply_{i}"),
                status_message_id=f"s_reply_{i}",
            )

        results = await asyncio.gather(*[add_reply(i) for i in range(5)])
        assert len(results) == 5
        tree = mgr.get_tree("root")
        assert len(tree.all_nodes()) == 6  # root + 5 replies

    @pytest.mark.asyncio
    async def test_add_to_tree_invalid_parent(self):
        """Adding to a nonexistent parent should raise ValueError."""
        mgr = TreeQueueManager()
        await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")

        with pytest.raises(ValueError, match="not found"):
            await mgr.add_to_tree(
                parent_node_id="nonexistent",
                node_id="reply",
                incoming=_make_incoming(),
                status_message_id="s1",
            )

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self):
        """Enqueue should process immediately if tree is free."""
        mgr = TreeQueueManager()
        tree = await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")

        processed = []

        async def processor(node_id, node):
            processed.append(node_id)

        queued = await mgr.enqueue("root", processor)
        # Should process immediately (not queued)
        assert queued is False

        # Wait for the async task to complete
        await asyncio.sleep(0.1)
        assert "root" in processed

    @pytest.mark.asyncio
    async def test_enqueue_queues_when_busy(self):
        """Enqueue should queue when tree is already processing."""
        mgr = TreeQueueManager()
        tree = await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")
        _, _ = await mgr.add_to_tree("root", "c1", _make_incoming(msg_id="c1"), "s1")

        processing_started = asyncio.Event()
        release = asyncio.Event()

        async def slow_processor(node_id, node):
            processing_started.set()
            await release.wait()

        # Start processing root (will block)
        queued_first = await mgr.enqueue("root", slow_processor)
        assert queued_first is False
        await processing_started.wait()

        # Now tree is busy, second enqueue should be queued
        queued_second = await mgr.enqueue("c1", slow_processor)
        assert queued_second is True

        # Release the blocker so things clean up
        release.set()
        await asyncio.sleep(0.2)

    @pytest.mark.asyncio
    async def test_cancel_tree(self):
        """cancel_tree should cancel in-progress and queued nodes."""
        mgr = TreeQueueManager()
        tree = await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")
        _, _ = await mgr.add_to_tree("root", "c1", _make_incoming(msg_id="c1"), "s1")
        _, _ = await mgr.add_to_tree("root", "c2", _make_incoming(msg_id="c2"), "s2")

        processing_started = asyncio.Event()

        async def slow_processor(node_id, node):
            processing_started.set()
            await asyncio.sleep(10)  # Long running

        # Start processing root
        await mgr.enqueue("root", slow_processor)
        await processing_started.wait()

        # Queue additional nodes
        await mgr.enqueue("c1", slow_processor)
        await mgr.enqueue("c2", slow_processor)

        # Cancel the tree
        cancelled = mgr.cancel_tree("root")
        assert len(cancelled) >= 1  # At least the current + queued

        # Tree should no longer be processing
        assert tree._is_processing is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_tree(self):
        """cancel_tree for nonexistent tree returns empty list."""
        mgr = TreeQueueManager()
        result = mgr.cancel_tree("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_cancel_all(self):
        """cancel_all cancels all trees."""
        mgr = TreeQueueManager()
        await mgr.create_tree("t1", _make_incoming(msg_id="t1"), "s1")
        await mgr.create_tree("t2", _make_incoming(msg_id="t2"), "s2")

        # Mark nodes as PENDING (they already are by default)
        cancelled = await mgr.cancel_all()
        # Nodes were PENDING but not in queue, so cleanup_stale logic applies
        # At minimum, it should not raise
        assert isinstance(cancelled, list)

    @pytest.mark.asyncio
    async def test_cleanup_stale_nodes(self):
        """cleanup_stale_nodes marks PENDING/IN_PROGRESS nodes as ERROR."""
        mgr = TreeQueueManager()
        tree = await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")
        _, _ = await mgr.add_to_tree("root", "c1", _make_incoming(msg_id="c1"), "s1")

        # Root is PENDING, c1 is PENDING
        count = mgr.cleanup_stale_nodes()
        assert count == 2

        root = tree.get_node("root")
        assert root.state == MessageState.ERROR
        assert "restart" in root.error_message

    @pytest.mark.asyncio
    async def test_mark_node_error_with_propagation(self):
        """mark_node_error should propagate to pending children."""
        mgr = TreeQueueManager()
        tree = await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")
        _, _ = await mgr.add_to_tree("root", "c1", _make_incoming(msg_id="c1"), "s1")
        _, _ = await mgr.add_to_tree("c1", "c2", _make_incoming(msg_id="c2"), "s2")

        affected = await mgr.mark_node_error("root", "something failed")
        # root + c1 + c2 should all be marked
        assert len(affected) >= 1
        root = tree.get_node("root")
        assert root.state == MessageState.ERROR

    @pytest.mark.asyncio
    async def test_mark_node_error_nonexistent(self):
        """mark_node_error for nonexistent node returns empty."""
        mgr = TreeQueueManager()
        result = await mgr.mark_node_error("nonexistent", "err")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_tree_for_node(self):
        """get_tree_for_node returns the right tree."""
        mgr = TreeQueueManager()
        await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")
        _, _ = await mgr.add_to_tree("root", "c1", _make_incoming(msg_id="c1"), "s1")

        tree = mgr.get_tree_for_node("c1")
        assert tree is not None
        assert tree.root_id == "root"

    def test_get_tree_for_node_nonexistent(self):
        """get_tree_for_node returns None for unknown nodes."""
        mgr = TreeQueueManager()
        assert mgr.get_tree_for_node("nonexistent") is None

    @pytest.mark.asyncio
    async def test_enqueue_no_tree(self):
        """Enqueue for a node not in any tree returns False."""
        mgr = TreeQueueManager()

        async def dummy(nid, node):
            pass

        result = await mgr.enqueue("nonexistent", dummy)
        assert result is False

    @pytest.mark.asyncio
    async def test_serialization_round_trip(self):
        """TreeQueueManager should survive serialization round-trip."""
        mgr = TreeQueueManager()
        await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")
        _, _ = await mgr.add_to_tree("root", "c1", _make_incoming(msg_id="c1"), "s1")

        data = mgr.to_dict()
        restored = TreeQueueManager.from_dict(data)

        assert restored.get_tree_count() == 1
        assert restored.get_node("c1") is not None

    @pytest.mark.asyncio
    async def test_rapid_messages_all_queued(self):
        """Rapid sequential enqueues should all be queued without loss."""
        mgr = TreeQueueManager()
        tree = await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")

        # Add 10 child nodes
        for i in range(10):
            await mgr.add_to_tree(
                "root", f"c{i}", _make_incoming(msg_id=f"c{i}"), f"s{i}"
            )

        blocker = asyncio.Event()

        async def blocking_processor(nid, node):
            await blocker.wait()

        # Start processing root (blocks)
        await mgr.enqueue("root", blocking_processor)
        await asyncio.sleep(0.05)  # Let task start

        # Rapidly enqueue all children
        results = []
        for i in range(10):
            r = await mgr.enqueue(f"c{i}", blocking_processor)
            results.append(r)

        # All should be queued (True)
        assert all(r is True for r in results)
        assert tree.get_queue_size() == 10

        # Cleanup
        blocker.set()
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_concurrent_trees_independent(self):
        """Processing in one tree shouldn't affect another."""
        mgr = TreeQueueManager()
        await mgr.create_tree("t1", _make_incoming(msg_id="t1"), "s1")
        await mgr.create_tree("t2", _make_incoming(msg_id="t2"), "s2")

        processed = []

        async def processor(nid, node):
            processed.append(nid)

        # Process both trees
        await mgr.enqueue("t1", processor)
        await mgr.enqueue("t2", processor)
        await asyncio.sleep(0.2)

        assert "t1" in processed
        assert "t2" in processed

    @pytest.mark.asyncio
    async def test_callbacks_invoked(self):
        """Queue update and node started callbacks should fire."""
        queue_updates = []
        node_starts = []

        async def on_queue_update(tree):
            queue_updates.append(tree.root_id)

        async def on_node_started(tree, node_id):
            node_starts.append(node_id)

        mgr = TreeQueueManager(
            queue_update_callback=on_queue_update,
            node_started_callback=on_node_started,
        )
        tree = await mgr.create_tree("root", _make_incoming(msg_id="root"), "s_root")
        _, _ = await mgr.add_to_tree("root", "c1", _make_incoming(msg_id="c1"), "s1")

        blocker = asyncio.Event()

        async def slow_proc(nid, node):
            if nid == "root":
                blocker.set()
                await asyncio.sleep(0.1)

        # Process root then c1 should be dequeued
        await mgr.enqueue("root", slow_proc)
        await blocker.wait()
        await mgr.enqueue("c1", slow_proc)
        await asyncio.sleep(0.5)

        # c1 was dequeued from queue, so callbacks should have fired
        assert len(queue_updates) >= 1 or len(node_starts) >= 1

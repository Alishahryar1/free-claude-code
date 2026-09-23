"""Focused tests for the tree package's private graph and queue values."""

import pytest

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.trees.graph import MessageTreeGraph
from free_claude_code.messaging.trees.node import MessageNode, MessageState
from free_claude_code.messaging.trees.queue import MessageNodeQueue
from free_claude_code.messaging.trees.snapshot import (
    TreeSnapshot,
    node_from_snapshot,
    node_to_snapshot,
)

_SCOPE = MessageScope(platform="telegram", chat_id="chat")


def _root() -> MessageNode:
    return MessageNode(
        node_id="root",
        scope=_SCOPE,
        prompt="prompt root",
        status_message_id="status-root",
    )


def _add(
    graph: MessageTreeGraph,
    node_id: str,
    status_message_id: str,
    parent_id: str,
) -> MessageNode:
    return graph.add_node(
        node_id=node_id,
        scope=_SCOPE,
        prompt=f"prompt {node_id}",
        status_message_id=status_message_id,
        parent_id=parent_id,
        parent_reference_id=parent_id,
    )


def test_node_snapshot_round_trip_preserves_execution_state_only() -> None:
    node = _root()
    node.children_ids.extend(["child-a", "child-b"])
    node.update_state(MessageState.ERROR)
    node.update_state(MessageState.COMPLETED, session_id="session-root")

    snapshot = node_to_snapshot(node)
    restored = node_from_snapshot(snapshot, _SCOPE)

    assert restored.node_id == "root"
    assert restored.scope == _SCOPE
    assert restored.prompt == ""
    assert restored.status_message_id == "status-root"
    assert restored.state is MessageState.COMPLETED
    assert restored.session_id == "session-root"
    assert restored.parent_reference_id is None
    assert restored.children_ids == []
    assert "children_ids" not in snapshot
    assert "created_at" not in snapshot
    assert "completed_at" not in snapshot
    assert "error_message" not in snapshot


def test_graph_snapshot_round_trip_preserves_links_and_status_lookup() -> None:
    graph = MessageTreeGraph(_root())
    child = _add(graph, "child", "status-child", "root")
    _add(graph, "grandchild", "status-grandchild", "child")
    child.update_state(MessageState.COMPLETED, session_id="session-child")

    restored = MessageTreeGraph.from_snapshot(graph.snapshot())
    parent = restored.get_parent("grandchild")
    status_child = restored.find_node_by_status_message("status-child")

    assert restored.root_id == "root"
    assert parent is not None
    assert parent.node_id == "child"
    assert restored.get_parent_session_id("grandchild") == "session-child"
    assert status_child is not None
    assert status_child.node_id == "child"
    assert restored.get_descendants("root") == ["root", "child", "grandchild"]


def test_graph_restore_normalizes_numeric_ids_to_string_references() -> None:
    graph = MessageTreeGraph(_root())
    _add(graph, "child", "status-child", "root")
    snapshot = graph.snapshot()
    snapshot.nodes["child"]["node_id"] = 2
    snapshot.nodes["child"]["status_message_id"] = 123
    snapshot.nodes["child"]["parent_id"] = "root"
    snapshot.nodes["2"] = snapshot.nodes.pop("child")

    restored = MessageTreeGraph.from_snapshot(snapshot)

    child = restored.find_node_by_status_message("123")
    assert child is not None and child.node_id == "2"


def test_graph_restore_normalizes_legacy_parent_to_prompt_reference() -> None:
    graph = MessageTreeGraph(_root())
    _add(graph, "child", "status-child", "root")
    snapshot = graph.snapshot()
    snapshot.nodes["child"].pop("parent_reference_id")

    restored = MessageTreeGraph.from_snapshot(snapshot)

    child = restored.get_node("child")
    assert child is not None
    assert child.parent_reference_id == "root"
    assert restored.snapshot().nodes["child"]["parent_reference_id"] == "root"


def test_cleared_status_round_trip_preserves_prompt_anchor() -> None:
    graph = MessageTreeGraph(_root())
    graph.clear_status("root")

    restored = MessageTreeGraph.from_snapshot(graph.snapshot())

    root = restored.get_root()
    assert root.status_message_id is None
    assert root.state is MessageState.ERROR
    assert restored.resolve_reference("root") is not None
    assert restored.resolve_reference("status-root") is None


def test_snapshot_rejects_runnable_node_without_status() -> None:
    snapshot = MessageTreeGraph(_root()).snapshot()
    snapshot.nodes["root"]["status_message_id"] = None

    with pytest.raises(ValueError, match="requires a status"):
        MessageTreeGraph.from_snapshot(snapshot)


def test_graph_rejects_duplicate_node_and_status_identity() -> None:
    graph = MessageTreeGraph(_root())
    _add(graph, "child", "status-child", "root")

    with pytest.raises(ValueError, match="already exists"):
        _add(graph, "child", "status-other", "root")
    with pytest.raises(ValueError, match="already exists"):
        _add(graph, "other", "status-child", "root")
    with pytest.raises(ValueError, match="must be distinct"):
        _add(graph, "same", "same", "root")


def test_graph_reference_subtree_can_remove_exact_nodes_and_status_lookups() -> None:
    graph = MessageTreeGraph(_root())
    _add(graph, "branch", "status-branch", "root")
    _add(graph, "leaf", "status-leaf", "branch")
    _add(graph, "sibling", "status-sibling", "root")

    references = graph.get_reference_descendants("branch")
    graph.remove_nodes(
        {reference for reference in references if not reference.startswith("status-")}
    )

    assert graph.get_node("branch") is None
    assert graph.get_node("leaf") is None
    assert graph.find_node_by_status_message("status-branch") is None
    assert graph.find_node_by_status_message("status-leaf") is None
    assert graph.get_descendants("root") == ["root", "sibling"]


def test_tree_snapshot_rejects_invalid_wire_shapes() -> None:
    assert TreeSnapshot.from_json(None) is None
    assert TreeSnapshot.from_json({"root_id": "root", "nodes": []}) is None
    assert TreeSnapshot.from_json({"nodes": {}}) is None


def test_node_queue_is_unique_fifo_and_supports_atomic_removal() -> None:
    queue = MessageNodeQueue()

    assert queue.put("a") is True
    assert queue.put("b") is True
    assert queue.put("a") is False
    assert queue.items() == ("a", "b")
    assert queue.remove("a") is True
    assert queue.remove("a") is False
    assert queue.items() == ("b",)
    assert queue.pop() == "b"
    assert queue.pop() is None

    assert queue.put("c") is True
    assert queue.put("d") is True
    assert queue.drain() == ("c", "d")
    assert queue.items() == ()

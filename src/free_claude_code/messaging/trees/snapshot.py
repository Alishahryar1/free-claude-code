"""Serializable messaging conversation snapshots."""

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ..models import MessageScope
from .identity import TreeIdentity
from .node import MessageNode, MessageState


@dataclass(frozen=True)
class TreeSnapshot:
    """Detached persisted representation of one conversation tree."""

    scope: MessageScope
    root_id: str
    nodes: dict[str, dict[str, Any]]

    @property
    def identity(self) -> TreeIdentity:
        return TreeIdentity(scope=self.scope, root_id=self.root_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "scope": {
                "platform": self.scope.platform,
                "chat_id": self.scope.chat_id,
            },
            "root_id": self.root_id,
            "nodes": dict(self.nodes),
        }

    @classmethod
    def from_json(cls, data: Any) -> TreeSnapshot | None:
        if not isinstance(data, dict):
            return None
        root_id = data.get("root_id")
        nodes = data.get("nodes")
        if root_id is None or not isinstance(nodes, dict):
            return None
        normalized_root_id = str(root_id)
        scope_data = data.get("scope")
        scope: MessageScope | None = None
        if isinstance(scope_data, dict):
            platform = scope_data.get("platform")
            chat_id = scope_data.get("chat_id")
            if platform is not None and chat_id is not None:
                scope = MessageScope(platform=str(platform), chat_id=str(chat_id))
        if scope is None:
            scope = _legacy_scope(normalized_root_id, nodes)
        if scope is None:
            logger.warning(
                "Skipping messaging tree snapshot without recoverable scope: "
                "root_id={}",
                normalized_root_id,
            )
            return None
        return cls(scope=scope, root_id=normalized_root_id, nodes=dict(nodes))

    def lookup_ids(self) -> set[str]:
        lookup_ids: set[str] = set()
        for node_key, node_data in self.nodes.items():
            lookup_ids.add(str(node_key))
            if not isinstance(node_data, dict):
                continue
            node_id = node_data.get("node_id")
            if node_id is not None:
                lookup_ids.add(str(node_id))
            status_message_id = node_data.get("status_message_id")
            if status_message_id is not None:
                lookup_ids.add(str(status_message_id))
        return lookup_ids


@dataclass(frozen=True)
class ConversationSnapshot:
    """Detached persisted conversation trees keyed by customer identity."""

    trees: dict[TreeIdentity, TreeSnapshot] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.trees

    def to_json(self) -> dict[str, Any]:
        return {"trees": [tree.to_json() for tree in self.trees.values()]}

    @classmethod
    def from_json(cls, data: Any) -> ConversationSnapshot:
        if not isinstance(data, dict):
            return cls()
        raw_trees = data.get("trees", [])
        if isinstance(raw_trees, dict):
            candidates = raw_trees.values()
        elif isinstance(raw_trees, list):
            candidates = raw_trees
        else:
            return cls()

        trees: dict[TreeIdentity, TreeSnapshot] = {}
        for raw_tree in candidates:
            snapshot = TreeSnapshot.from_json(raw_tree)
            if snapshot is None:
                continue
            trees[snapshot.identity] = snapshot
        return cls(trees=trees)

    def get_tree(self, identity: TreeIdentity) -> TreeSnapshot | None:
        return self.trees.get(identity)

    def with_tree(self, tree_snapshot: TreeSnapshot) -> ConversationSnapshot:
        trees = dict(self.trees)
        trees[tree_snapshot.identity] = tree_snapshot
        return ConversationSnapshot(trees=trees)

    def without_tree(self, identity: TreeIdentity) -> ConversationSnapshot:
        trees = dict(self.trees)
        trees.pop(identity, None)
        return ConversationSnapshot(trees=trees)

    def without_scope(self, scope: MessageScope) -> ConversationSnapshot:
        """Remove every tree belonging to one platform chat."""
        return ConversationSnapshot(
            trees={
                identity: tree
                for identity, tree in self.trees.items()
                if identity.scope != scope
            }
        )


def _legacy_scope(
    root_id: str,
    nodes: dict[str, Any],
) -> MessageScope | None:
    """Derive scope from pre-scope snapshots without retaining a runtime shim."""
    root = nodes.get(root_id)
    if not isinstance(root, dict):
        root = next(
            (
                node
                for node in nodes.values()
                if isinstance(node, dict) and str(node.get("node_id")) == root_id
            ),
            None,
        )
    if not isinstance(root, dict):
        return None
    return node_scope_from_snapshot(root)


def node_scope_from_snapshot(data: dict[str, Any]) -> MessageScope | None:
    """Read the redundant scope carried by legacy node payloads, if present."""
    incoming = data.get("incoming")
    if not isinstance(incoming, dict):
        return None
    platform = incoming.get("platform")
    chat_id = incoming.get("chat_id")
    if platform is None or chat_id is None:
        return None
    return MessageScope(platform=str(platform), chat_id=str(chat_id))


def node_to_snapshot(node: MessageNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "status_message_id": node.status_message_id,
        "state": node.state.value,
        "parent_id": node.parent_id,
        "parent_reference_id": node.parent_reference_id,
        "session_id": node.session_id,
    }


def node_from_snapshot(data: dict[str, Any], scope: MessageScope) -> MessageNode:
    state = MessageState(data["state"])
    status_message_id = _optional_id(
        data.get("status_message_id"),
        "status_message_id",
    )
    if state in (MessageState.PENDING, MessageState.IN_PROGRESS) and (
        status_message_id is None
    ):
        raise ValueError("Runnable tree snapshot node requires a status message")
    parent_id = _optional_id(data.get("parent_id"), "parent_id")
    parent_reference_id = _optional_id(
        data.get("parent_reference_id"),
        "parent_reference_id",
    )
    if parent_reference_id is None:
        parent_reference_id = parent_id
    return MessageNode(
        node_id=_required_id(data.get("node_id"), "node_id"),
        scope=scope,
        prompt="",
        status_message_id=status_message_id,
        state=state,
        parent_id=parent_id,
        parent_reference_id=parent_reference_id,
        session_id=_optional_id(data.get("session_id"), "session_id"),
    )


def _required_id(value: Any, field_name: str) -> str:
    normalized = _optional_id(value, field_name)
    if normalized is None:
        raise ValueError(f"Tree snapshot {field_name} is required")
    return normalized


def _optional_id(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise ValueError(f"Tree snapshot {field_name} must be a string or integer")
    normalized = str(value)
    if not normalized:
        raise ValueError(f"Tree snapshot {field_name} cannot be empty")
    return normalized

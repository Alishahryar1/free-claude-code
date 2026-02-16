"""Repository for message tree data access.

Provides data access layer for managing trees and node mappings.
"""

from typing import Dict, Optional, List

from loguru import logger

from .tree_data import MessageTree, MessageNode, MessageState


class TreeRepository:
    """
    Repository for message tree data access.

    Manages the storage and lookup of trees and node-to-tree mappings.
    """

    def __init__(self):
        self._trees: Dict[str, MessageTree] = {}  # root_id -> tree
        self._node_to_tree: Dict[str, str] = {}  # node_id -> root_id

    def get_tree(self, root_id: str) -> Optional[MessageTree]:
        """Get a tree by its root ID."""
        return self._trees.get(root_id)

    def get_tree_for_node(self, node_id: str) -> Optional[MessageTree]:
        """Get the tree containing a given node."""
        root_id = self._node_to_tree.get(node_id)
        if not root_id:
            return None
        return self._trees.get(root_id)

    def get_node(self, node_id: str) -> Optional[MessageNode]:
        """Get a node from any tree."""
        tree = self.get_tree_for_node(node_id)
        return tree.get_node(node_id) if tree else None

    def add_tree(self, root_id: str, tree: MessageTree) -> None:
        """Add a new tree to the repository."""
        self._trees[root_id] = tree
        self._node_to_tree[root_id] = root_id
        logger.debug("TREE_REPO: add_tree root_id=%s", root_id)

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree."""
        self._node_to_tree[node_id] = root_id
        logger.debug("TREE_REPO: register_node node_id=%s root_id=%s", node_id, root_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node is registered in any tree."""
        return node_id in self._node_to_tree

    def tree_count(self) -> int:
        """Get the number of trees in the repository."""
        return len(self._trees)

    def is_tree_busy(self, root_id: str) -> bool:
        """Check if a tree is currently processing."""
        tree = self._trees.get(root_id)
        return tree.is_processing if tree else False

    def is_node_tree_busy(self, node_id: str) -> bool:
        """Check if the tree containing a node is busy."""
        tree = self.get_tree_for_node(node_id)
        return tree.is_processing if tree else False

    def get_queue_size(self, node_id: str) -> int:
        """Get queue size for the tree containing a node."""
        tree = self.get_tree_for_node(node_id)
        return tree.get_queue_size() if tree else 0

    def resolve_parent_node_id(self, msg_id: str) -> Optional[str]:
        """
        Resolve a message ID to the actual parent node ID.

        Handles the case where msg_id is a status message ID
        (which maps to the tree but isn't an actual node).

        Returns:
            The node_id to use as parent, or None if not found
        """
        tree = self.get_tree_for_node(msg_id)
        if not tree:
            return None

        # Check if msg_id is an actual node
        if tree.has_node(msg_id):
            return msg_id

        # Otherwise, it might be a status message - find the owning node
        node = tree.find_node_by_status_message(msg_id)
        if node:
            return node.node_id

        return None

    def get_pending_children(self, node_id: str) -> List[MessageNode]:
        """
        Get all pending child nodes (recursively) of a given node.

        Used for error propagation - when a node fails, its pending
        children should also be marked as failed.
        """
        tree = self.get_tree_for_node(node_id)
        if not tree:
            return []

        pending = []
        node = tree.get_node(node_id)
        if not node:
            return []

        for child_id in node.children_ids:
            child = tree.get_node(child_id)
            if child and child.state == MessageState.PENDING:
                pending.append(child)
                # Recursively get children of pending children
                pending.extend(self.get_pending_children(child_id))

        return pending

    def all_trees(self) -> List[MessageTree]:
        """Get all trees in the repository."""
        return list(self._trees.values())

    def tree_ids(self) -> List[str]:
        """Get all tree root IDs."""
        return list(self._trees.keys())

    def to_dict(self) -> dict:
        """Serialize all trees."""
        return {
            "trees": {rid: tree.to_dict() for rid, tree in self._trees.items()},
            "node_to_tree": self._node_to_tree.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TreeRepository":
        """Deserialize from dictionary."""
        from .tree_data import MessageTree

        repo = cls()
        for root_id, tree_data in data.get("trees", {}).items():
            repo._trees[root_id] = MessageTree.from_dict(tree_data)
        repo._node_to_tree = data.get("node_to_tree", {})
        return repo

"""Repository for message tree data access.

Provides data access layer for managing trees and node mappings.
"""

from loguru import logger

from .data import MessageNode, MessageState, MessageTree


class TreeRepository:
    """
    Repository for message tree data access.

    Manages the storage and lookup of trees and node-to-tree mappings.
    """

    def __init__(self):
        self._trees: dict[str, MessageTree] = {}  # root_id -> tree
        self._node_to_tree: dict[str, str] = {}  # node_id -> root_id

    def get_tree(self, root_id: str) -> MessageTree | None:
        """Get a tree by its root ID."""
        return self._trees.get(root_id)

    def get_tree_for_node(self, node_id: str) -> MessageTree | None:
        """Get the tree containing a given node."""
        root_id = self._node_to_tree.get(node_id)
        if not root_id:
            return None
        return self._trees.get(root_id)

    def get_node(self, node_id: str) -> MessageNode | None:
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

    def resolve_parent_node_id(self, msg_id: str) -> str | None:
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

    def get_pending_children(self, node_id: str) -> list[MessageNode]:
        """
        Get all pending child nodes (recursively) of a given node.

        Used for error propagation - when a node fails, its pending
        children should also be marked as failed.
        """
        tree = self.get_tree_for_node(node_id)
        if not tree:
            return []

        pending: list[MessageNode] = []
        stack = [node_id]

        while stack:
            current_id = stack.pop()
            node = tree.get_node(current_id)
            if not node:
                continue
            for child_id in node.children_ids:
                child = tree.get_node(child_id)
                if child and child.state == MessageState.PENDING:
                    pending.append(child)
                    stack.append(child_id)

        return pending

    def all_trees(self) -> list[MessageTree]:
        """Get all trees in the repository."""
        return list(self._trees.values())

    def tree_ids(self) -> list[str]:
        """Get all tree root IDs."""
        return list(self._trees.keys())

    def unregister_nodes(self, node_ids: list[str]) -> None:
        """Remove node IDs from the node-to-tree mapping."""
        for nid in node_ids:
            self._node_to_tree.pop(nid, None)

    def remove_tree(self, root_id: str) -> MessageTree | None:
        """
        Remove a tree and all its node mappings from the repository.

        Returns:
            The removed tree, or None if not found.
        """
        tree = self._trees.pop(root_id, None)
        if not tree:
            return None
        for node in tree.all_nodes():
            self._node_to_tree.pop(node.node_id, None)
        logger.debug("TREE_REPO: remove_tree root_id=%s", root_id)
        return tree

    def get_message_ids_for_chat(self, platform: str, chat_id: str) -> set[str]:
        """Get all message IDs (incoming + status) for a given platform/chat."""
        msg_ids: set[str] = set()
        for tree in self._trees.values():
            for node in tree.all_nodes():
                if str(node.incoming.platform) == str(platform) and str(
                    node.incoming.chat_id
                ) == str(chat_id):
                    if node.incoming.message_id is not None:
                        msg_ids.add(str(node.incoming.message_id))
                    if node.status_message_id:
                        msg_ids.add(str(node.status_message_id))
        return msg_ids

    def to_dict(self) -> dict:
        """Serialize all trees."""
        return {
            "trees": {rid: tree.to_dict() for rid, tree in self._trees.items()},
            "node_to_tree": self._node_to_tree.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TreeRepository:
        """Deserialize from dictionary."""
        from .data import MessageTree

        repo = cls()
        for root_id, tree_data in data.get("trees", {}).items():
            repo._trees[root_id] = MessageTree.from_dict(tree_data)
        repo._node_to_tree = data.get("node_to_tree", {})
        return repo

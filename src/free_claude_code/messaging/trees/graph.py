"""In-memory graph for one messaging conversation tree."""

from loguru import logger

from ..models import MessageScope
from .identity import TreeIdentity
from .node import MessageNode, MessageReferenceKind, MessageState
from .snapshot import (
    TreeSnapshot,
    node_from_snapshot,
    node_scope_from_snapshot,
    node_to_snapshot,
)


class MessageTreeGraph:
    """Own parent/child links, node lookup, and status-message lookup."""

    def __init__(self, root_node: MessageNode) -> None:
        if root_node.status_message_id == root_node.node_id:
            raise ValueError("Prompt and status message IDs must be distinct")
        self.root_id = root_node.node_id
        self.identity = TreeIdentity(
            scope=root_node.scope,
            root_id=root_node.node_id,
        )
        self._nodes: dict[str, MessageNode] = {root_node.node_id: root_node}
        self._status_to_node: dict[str, str] = {}
        if root_node.status_message_id is not None:
            self._status_to_node[root_node.status_message_id] = root_node.node_id

    def add_node(
        self,
        *,
        node_id: str,
        scope: MessageScope,
        prompt: str,
        status_message_id: str,
        parent_id: str,
        parent_reference_id: str,
    ) -> MessageNode:
        if scope != self.identity.scope:
            raise ValueError("A reply cannot cross platform chat boundaries")
        if parent_id not in self._nodes:
            raise ValueError(f"Parent node {parent_id} not found in tree")
        parent_reference = self.resolve_reference(parent_reference_id)
        if parent_reference is None or parent_reference[0].node_id != parent_id:
            raise ValueError("Reply reference does not belong to its logical parent")
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} already exists in tree")
        if status_message_id == node_id:
            raise ValueError("Prompt and status message IDs must be distinct")
        if node_id in self._status_to_node:
            raise ValueError(f"Message reference {node_id} already exists in tree")
        if status_message_id in self._status_to_node:
            raise ValueError(
                f"Status message {status_message_id} already exists in tree"
            )
        if status_message_id in self._nodes:
            raise ValueError(
                f"Message reference {status_message_id} already exists in tree"
            )

        node = MessageNode(
            node_id=node_id,
            scope=scope,
            prompt=prompt,
            status_message_id=status_message_id,
            parent_id=parent_id,
            parent_reference_id=parent_reference_id,
            state=MessageState.PENDING,
        )
        self._nodes[node_id] = node
        self._status_to_node[status_message_id] = node_id
        self._nodes[parent_id].children_ids.append(node_id)
        logger.debug("Added node {} as child of {}", node_id, parent_id)
        return node

    def get_node(self, node_id: str) -> MessageNode | None:
        return self._nodes.get(node_id)

    def get_root(self) -> MessageNode:
        return self._nodes[self.root_id]

    def get_parent(self, node_id: str) -> MessageNode | None:
        node = self._nodes.get(node_id)
        if not node or not node.parent_id:
            return None
        return self._nodes.get(node.parent_id)

    def get_parent_session_id(self, node_id: str) -> str | None:
        parent = self.get_parent(node_id)
        return parent.session_id if parent else None

    def find_node_by_status_message(self, status_msg_id: str) -> MessageNode | None:
        node_id = self._status_to_node.get(status_msg_id)
        return self._nodes.get(node_id) if node_id else None

    def resolve_reference(
        self, reference_id: str
    ) -> tuple[MessageNode, MessageReferenceKind] | None:
        """Resolve an exact prompt or FCC status reference."""
        node = self._nodes.get(reference_id)
        if node is not None:
            return node, MessageReferenceKind.PROMPT
        node = self.find_node_by_status_message(reference_id)
        if node is not None:
            return node, MessageReferenceKind.STATUS
        return None

    def all_nodes(self) -> list[MessageNode]:
        return list(self._nodes.values())

    def get_descendants(self, node_id: str) -> list[str]:
        if node_id not in self._nodes:
            return []
        result: list[str] = []
        stack = [node_id]
        while stack:
            current_id = stack.pop()
            result.append(current_id)
            node = self._nodes.get(current_id)
            if node:
                stack.extend(node.children_ids)
        return result

    def get_reference_descendants(self, reference_id: str) -> list[str]:
        """Return the literal platform reply subtree rooted at a reference."""
        if self.resolve_reference(reference_id) is None:
            return []

        children: dict[str, list[str]] = {}
        for node in self._nodes.values():
            if node.status_message_id is not None:
                children.setdefault(node.node_id, []).append(node.status_message_id)
            if node.parent_reference_id is not None:
                children.setdefault(node.parent_reference_id, []).append(node.node_id)

        result: list[str] = []
        stack = [reference_id]
        while stack:
            current_id = stack.pop()
            result.append(current_id)
            stack.extend(children.get(current_id, ()))
        return result

    def remove_nodes(self, node_ids: set[str]) -> None:
        """Remove an exact set of nodes after reference-subtree calculation."""
        for node_id in node_ids:
            node = self._nodes.get(node_id)
            if node is None:
                continue
            if node.parent_id is not None:
                parent = self._nodes.get(node.parent_id)
                if parent is not None:
                    parent.children_ids = [
                        child_id
                        for child_id in parent.children_ids
                        if child_id != node_id
                    ]
            if node.status_message_id is not None:
                self._status_to_node.pop(node.status_message_id, None)
            self._nodes.pop(node_id, None)

    def clear_status(self, node_id: str) -> None:
        """Remove one status reference while preserving its prompt node."""
        node = self._nodes.get(node_id)
        if node is None or node.status_message_id is None:
            return
        self._status_to_node.pop(node.status_message_id, None)
        node.clear_status()

    def all_reference_ids(self) -> set[str]:
        """Return every prompt and live FCC status reference in the tree."""
        references = set(self._nodes)
        references.update(self._status_to_node)
        return references

    def snapshot(self) -> TreeSnapshot:
        return TreeSnapshot(
            scope=self.identity.scope,
            root_id=self.root_id,
            nodes={
                node_id: node_to_snapshot(node) for node_id, node in self._nodes.items()
            },
        )

    @classmethod
    def from_snapshot(cls, snapshot: TreeSnapshot) -> MessageTreeGraph:
        root_data = snapshot.nodes[snapshot.root_id]
        if not isinstance(root_data, dict):
            raise ValueError("Tree snapshot contains an invalid root node")
        if node_scope_from_snapshot(root_data) not in (None, snapshot.scope):
            raise ValueError("Tree snapshot contains a cross-scope node")
        root_node = node_from_snapshot(root_data, snapshot.scope)
        if root_node.node_id != snapshot.root_id:
            raise ValueError("Tree snapshot root key does not match its node ID")
        graph = cls(root_node)
        reference_owner = {root_node.node_id: root_node.node_id}
        if root_node.status_message_id is not None:
            reference_owner[root_node.status_message_id] = root_node.node_id
        for snapshot_node_id, node_data in snapshot.nodes.items():
            if snapshot_node_id == snapshot.root_id:
                continue
            if not isinstance(node_data, dict):
                raise ValueError("Tree snapshot contains an invalid node")
            if node_scope_from_snapshot(node_data) not in (None, snapshot.scope):
                raise ValueError("Tree snapshot contains a cross-scope node")
            node = node_from_snapshot(node_data, snapshot.scope)
            if str(snapshot_node_id) != node.node_id:
                raise ValueError("Tree snapshot node key does not match its node ID")
            if node.status_message_id == node.node_id:
                raise ValueError("Prompt and status message IDs must be distinct")
            if node.node_id in graph._nodes:
                raise ValueError(f"Duplicate node {node.node_id} in tree snapshot")
            references = {node.node_id}
            if node.status_message_id is not None:
                references.add(node.status_message_id)
            for reference in references:
                owner = reference_owner.get(reference)
                if owner is not None and owner != node.node_id:
                    raise ValueError(
                        f"Duplicate message reference {reference} in tree snapshot"
                    )
                reference_owner[reference] = node.node_id
            graph._nodes[node.node_id] = node
            if node.status_message_id is not None:
                graph._status_to_node[node.status_message_id] = node.node_id

        if root_node.parent_id is not None or root_node.parent_reference_id is not None:
            raise ValueError("Tree snapshot root cannot have a parent")
        for node in graph._nodes.values():
            if node.node_id == graph.root_id:
                continue
            if node.parent_id is None or node.parent_id not in graph._nodes:
                raise ValueError(f"Node {node.node_id} has no valid parent")
            if node.parent_reference_id is None:
                raise ValueError(f"Node {node.node_id} has no exact parent reference")
            parent_reference = graph.resolve_reference(node.parent_reference_id)
            if (
                parent_reference is None
                or parent_reference[0].node_id != node.parent_id
            ):
                raise ValueError(
                    f"Node {node.node_id} has an invalid exact parent reference"
                )
            graph._nodes[node.parent_id].children_ids.append(node.node_id)
        if set(graph.get_descendants(graph.root_id)) != set(graph._nodes):
            raise ValueError("Tree snapshot contains a disconnected branch")
        return graph

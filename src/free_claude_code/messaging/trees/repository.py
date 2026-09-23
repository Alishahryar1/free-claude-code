"""Manager-owned index of messaging tree aggregates."""

from loguru import logger

from ..models import MessageScope
from .identity import TreeIdentity
from .runtime import MessageTree
from .snapshot import ConversationSnapshot
from .transitions import NodeUiTarget


class TreeRepository:
    """Store aggregates and map node/status references to their root."""

    def __init__(self) -> None:
        self._trees: dict[TreeIdentity, MessageTree] = {}
        self._reference_to_tree: dict[tuple[MessageScope, str], TreeIdentity] = {}

    def get_tree(self, identity: TreeIdentity) -> MessageTree | None:
        return self._trees.get(identity)

    def get_tree_for_reference(
        self,
        scope: MessageScope,
        reference_id: str,
    ) -> MessageTree | None:
        identity = self._reference_to_tree.get((scope, reference_id))
        return self._trees.get(identity) if identity is not None else None

    def has_reference(self, scope: MessageScope, reference_id: str) -> bool:
        return (scope, reference_id) in self._reference_to_tree

    def add_tree(
        self,
        tree: MessageTree,
        *,
        node_id: str,
        status_message_id: str,
    ) -> None:
        identity = tree.identity
        if identity in self._trees:
            raise ValueError(f"Tree {identity} already exists")
        self.register_node(
            identity=identity,
            node_id=node_id,
            status_message_id=status_message_id,
        )
        self._trees[identity] = tree
        logger.debug("TREE_REPO: add_tree identity={}", identity)

    def register_node(
        self,
        *,
        identity: TreeIdentity,
        node_id: str,
        status_message_id: str,
    ) -> None:
        if self.has_reference(identity.scope, node_id) or self.has_reference(
            identity.scope, status_message_id
        ):
            raise ValueError("Node or status message is already registered")
        self._reference_to_tree[(identity.scope, node_id)] = identity
        self._reference_to_tree[(identity.scope, status_message_id)] = identity

    def unregister_references(
        self,
        identity: TreeIdentity,
        reference_ids: set[str],
    ) -> None:
        for reference_id in reference_ids:
            key = (identity.scope, reference_id)
            if self._reference_to_tree.get(key) == identity:
                self._reference_to_tree.pop(key)

    def remove_tree(
        self,
        identity: TreeIdentity,
    ) -> MessageTree | None:
        tree = self._trees.pop(identity, None)
        if tree is None:
            return None
        self._reference_to_tree = {
            key: owner
            for key, owner in self._reference_to_tree.items()
            if owner != identity
        }
        logger.debug("TREE_REPO: remove_tree identity={}", identity)
        return tree

    def trees(self) -> tuple[MessageTree, ...]:
        return tuple(self._trees.values())

    def tree_count(self) -> int:
        return len(self._trees)

    @classmethod
    def from_snapshot(
        cls, snapshot: ConversationSnapshot
    ) -> tuple[
        TreeRepository,
        ConversationSnapshot,
        tuple[NodeUiTarget, ...],
    ]:
        repo = cls()
        normalized = ConversationSnapshot()
        stale_targets: list[NodeUiTarget] = []
        for tree_snapshot in snapshot.trees.values():
            try:
                tree = MessageTree.from_snapshot(tree_snapshot)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid messaging tree snapshot: {}",
                    type(exc).__name__,
                )
                continue
            references = tree_snapshot.lookup_ids()
            if tree.identity in repo._trees or any(
                repo.has_reference(tree.identity.scope, reference)
                for reference in references
            ):
                logger.warning("Skipping duplicate messaging tree {}", tree.identity)
                continue
            repo._trees[tree.identity] = tree
            for reference in references:
                repo._reference_to_tree[(tree.identity.scope, reference)] = (
                    tree.identity
                )
            stale_targets.extend(tree.restored_stale_targets)
            normalized_tree = tree.restored_snapshot
            if normalized_tree is not None:
                normalized = normalized.with_tree(normalized_tree)
        return repo, normalized, tuple(stale_targets)


__all__ = ["TreeRepository"]

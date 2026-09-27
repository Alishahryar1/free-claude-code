"""Rollback state for a manager-owned durable messaging transition."""

from typing import TYPE_CHECKING

from ..models import MessageScope
from .identity import TreeIdentity
from .repository import TreeRepository
from .runtime import MessageTree, TreeCheckpoint

if TYPE_CHECKING:
    from .ports import MessagingStore


class TreeTransaction:
    def __init__(self, repository: TreeRepository, store: MessagingStore) -> None:
        self.repository = repository
        self.store = store
        self.before: dict[TreeIdentity, tuple[MessageTree, TreeCheckpoint] | None] = {}
        self.clear_scope: MessageScope | None = None

    def watch(self, tree: MessageTree | None) -> None:
        if tree is not None and tree.identity not in self.before:
            self.before[tree.identity] = (tree, tree.checkpoint())

    def inserted(self, identity: TreeIdentity) -> None:
        self.before[identity] = None

    async def commit(self) -> None:
        changed, removed = [], []
        for identity, previous in self.before.items():
            current = self.repository.get_tree(identity)
            if current is None:
                removed.append(identity)
            else:
                snapshot = await current.snapshot()
                if previous is None or snapshot != previous[1].graph.snapshot():
                    changed.append(snapshot)
        if changed or removed or self.clear_scope is not None:
            await self.store.commit_trees(
                tuple(changed), removed=tuple(removed), clear_scope=self.clear_scope
            )

    async def rollback(self) -> None:
        for identity, previous in self.before.items():
            if previous is None:
                await self.repository.restore_tree(identity, None)
            else:
                tree, checkpoint = previous
                tree.rollback(checkpoint)
                await self.repository.restore_tree(identity, tree)

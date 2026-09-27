"""Transactional messaging records in the application-owned FCC database."""

import sqlite3
from datetime import UTC, datetime

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.trees import (
    ConversationSnapshot,
    TreeIdentity,
    TreeSnapshot,
    normalize_tree_snapshot,
)

from .sqlite_database import SQLiteDatabase

_FIELDS = (
    "node_id",
    "status_message_id",
    "state",
    "parent_id",
    "parent_reference_id",
    "session_id",
)


def write_tree(connection: sqlite3.Connection, snapshot: TreeSnapshot) -> None:
    """Update only changed rows, inserting new parents before their children."""
    normalized = normalize_tree_snapshot(snapshot)
    scope = (snapshot.scope.platform, snapshot.scope.chat_id)
    identity = (*scope, snapshot.root_id)
    connection.execute(
        "INSERT INTO messaging_trees VALUES (?,?,?) ON CONFLICT DO NOTHING", identity
    )
    previous = {
        row["node_id"]: dict(row)
        for row in connection.execute(
            "SELECT * FROM messaging_nodes WHERE platform=? AND chat_id=? AND root_id=?",
            identity,
        )
    }
    for node_id in previous.keys() - normalized.nodes.keys():
        connection.execute(
            "DELETE FROM messaging_nodes WHERE platform=? AND chat_id=? AND node_id=?",
            (*scope, node_id),
        )
    for node_id in normalized.nodes:
        node = normalized.nodes[node_id]
        before = previous.get(node_id)
        if before is not None and all(
            before[field] == node.get(field) for field in _FIELDS
        ):
            continue
        connection.execute(
            "INSERT INTO messaging_nodes (platform,chat_id,root_id,node_id,status_message_id,state,parent_id,parent_reference_id,session_id) "
            "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(platform,chat_id,node_id) DO UPDATE SET "
            "root_id=excluded.root_id,status_message_id=excluded.status_message_id,state=excluded.state,"
            "parent_id=excluded.parent_id,parent_reference_id=excluded.parent_reference_id,session_id=excluded.session_id",
            (*identity, *(node.get(field) for field in _FIELDS)),
        )


class SQLiteMessagingStore:
    def __init__(
        self, database: SQLiteDatabase, *, managed_message_cap: int | None = None
    ) -> None:
        self.database = database
        self._cap = managed_message_cap

    async def load_conversation_snapshot(self) -> ConversationSnapshot:
        def load(connection: sqlite3.Connection) -> ConversationSnapshot:
            trees: dict[TreeIdentity, TreeSnapshot] = {}
            for row in connection.execute(
                "SELECT * FROM messaging_nodes ORDER BY rowid"
            ):
                scope = MessageScope(platform=row["platform"], chat_id=row["chat_id"])
                identity = TreeIdentity(scope=scope, root_id=row["root_id"])
                snapshot = trees.setdefault(
                    identity, TreeSnapshot(scope, identity.root_id, {})
                )
                snapshot.nodes[row["node_id"]] = {
                    field: row[field] for field in _FIELDS
                }
            return ConversationSnapshot(trees)

        return await self.database.run(load, write=False)

    async def commit_trees(
        self,
        snapshots: tuple[TreeSnapshot, ...],
        *,
        removed: tuple[TreeIdentity, ...] = (),
        clear_scope: MessageScope | None = None,
    ) -> None:
        def commit(connection: sqlite3.Connection) -> None:
            if clear_scope is not None:
                scope = (clear_scope.platform, clear_scope.chat_id)
                connection.execute(
                    "DELETE FROM messaging_trees WHERE platform=? AND chat_id=?", scope
                )
                connection.execute(
                    "DELETE FROM messaging_managed_messages WHERE platform=? AND chat_id=?",
                    scope,
                )
            for identity in removed:
                connection.execute(
                    "DELETE FROM messaging_trees WHERE platform=? AND chat_id=? AND root_id=?",
                    (identity.scope.platform, identity.scope.chat_id, identity.root_id),
                )
            for snapshot in snapshots:
                write_tree(connection, snapshot)

        await self.database.run(commit)

    def _trim(
        self, connection: sqlite3.Connection, platform: str, chat_id: str
    ) -> None:
        if self._cap is not None and self._cap > 0:
            connection.execute(
                "DELETE FROM messaging_managed_messages WHERE sequence IN ("
                "SELECT sequence FROM messaging_managed_messages WHERE platform=? AND chat_id=? "
                "ORDER BY sequence DESC LIMIT -1 OFFSET ?)",
                (platform, chat_id, self._cap),
            )

    async def trim(self) -> None:
        def trim(connection: sqlite3.Connection) -> None:
            for row in connection.execute(
                "SELECT DISTINCT platform,chat_id FROM messaging_managed_messages"
            ).fetchall():
                self._trim(connection, row[0], row[1])

        await self.database.run(trim)

    async def record_message_id(
        self, platform: str, chat_id: str, message_id: str, direction: str, kind: str
    ) -> None:
        def record(connection: sqlite3.Connection) -> None:
            connection.execute(
                "INSERT INTO messaging_managed_messages(platform,chat_id,message_id,ts,direction,kind) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(platform,chat_id,message_id) DO NOTHING",
                (
                    str(platform),
                    str(chat_id),
                    str(message_id),
                    datetime.now(UTC).isoformat(),
                    direction,
                    kind,
                ),
            )
            self._trim(connection, platform, chat_id)

        await self.database.run(record)

    async def get_tracked_message_ids_for_chat(
        self, platform: str, chat_id: str
    ) -> list[str]:
        return await self.database.run(
            lambda connection: [
                row[0]
                for row in connection.execute(
                    "SELECT message_id FROM messaging_managed_messages WHERE platform=? AND chat_id=? ORDER BY sequence",
                    (platform, chat_id),
                )
            ],
            write=False,
        )

    async def forget_tracked_message_ids(
        self, platform: str, chat_id: str, message_ids: set[str]
    ) -> None:
        def forget(connection: sqlite3.Connection) -> None:
            connection.executemany(
                "DELETE FROM messaging_managed_messages WHERE platform=? AND chat_id=? AND message_id=?",
                [(platform, chat_id, item) for item in message_ids],
            )

        await self.database.run(forget)

"""One-time, best-effort import of FCC's retired messaging JSON file."""

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from free_claude_code.messaging.trees import (
    TreeIdentity,
    TreeSnapshot,
    normalize_tree_snapshot,
)

from .messaging_sqlite import write_tree
from .sqlite_database import SQLiteDatabase


@dataclass
class LegacyData:
    outcome: str = "complete"
    trees: dict[TreeIdentity, TreeSnapshot] = field(default_factory=dict)
    messages: list[tuple[str, str, str, str, str, str]] = field(default_factory=list)
    skipped: int = 0

    def reject(self) -> None:
        self.outcome = "partial"
        self.skipped += 1


def read_legacy(path: Path) -> LegacyData:
    result = LegacyData()
    try:
        if path.is_symlink():
            return LegacyData(outcome="unreadable", skipped=1)
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return LegacyData(outcome="absent")
    except OSError, UnicodeError, ValueError:
        return LegacyData(outcome="unreadable", skipped=1)
    if not isinstance(raw, dict):
        return LegacyData(outcome="unreadable", skipped=1)
    conversation = raw.get("conversation", raw)
    if not isinstance(conversation, dict):
        result.reject()
        conversation = raw
    candidates = conversation.get("trees", [])
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    if not isinstance(candidates, list):
        result.reject()
        candidates = []
    # Historical JSON loading used the last duplicate tree identity.
    decoded: dict[TreeIdentity, TreeSnapshot] = {}
    for candidate in candidates:
        try:
            snapshot = TreeSnapshot.from_json(candidate)
            if snapshot is None:
                raise ValueError("Missing tree scope")
            if snapshot.identity in decoded:
                result.reject()
            decoded[snapshot.identity] = snapshot
        except KeyError, TypeError, ValueError:
            result.reject()
    for snapshot in decoded.values():
        try:
            normalized = normalize_tree_snapshot(snapshot)
            result.trees[normalized.identity] = normalized
        except KeyError, TypeError, ValueError:
            result.reject()
    messages = raw.get("managed_messages", raw.get("message_log", {}))
    if not isinstance(messages, dict):
        result.reject()
        messages = {}
    for chat_key, items in messages.items():
        if (
            not isinstance(chat_key, str)
            or ":" not in chat_key
            or not isinstance(items, list)
        ):
            result.reject()
            continue
        platform, chat_id = chat_key.split(":", 1)
        for item in items:
            record = _message(platform, chat_id, item)
            if record is None:
                result.reject()
            else:
                result.messages.append(record)
    return result


def _message(
    platform: str, chat_id: str, item: Any
) -> tuple[str, str, str, str, str, str] | None:
    if not isinstance(item, dict) or item.get("message_id") is None:
        return None
    direction, kind = str(item.get("direction") or ""), str(item.get("kind") or "")
    if direction not in {"in", "out"} or not kind:
        return None
    return (
        platform,
        chat_id,
        str(item["message_id"]),
        str(item.get("ts") or ""),
        direction,
        kind,
    )


async def import_legacy(database: SQLiteDatabase, path: Path) -> str | None:
    return await database.work(lambda: _import_legacy(database, path))


def _try_import(connection: sqlite3.Connection, write: Callable[[], object]) -> bool:
    """Reject one damaged unit without publishing part of a tree or its references."""
    connection.execute("SAVEPOINT legacy_record")
    try:
        write()
    except sqlite3.IntegrityError, UnicodeEncodeError:
        connection.execute("ROLLBACK TO legacy_record")
        connection.execute("RELEASE legacy_record")
        return False
    connection.execute("RELEASE legacy_record")
    return True


def _import_legacy(database: SQLiteDatabase, path: Path) -> str | None:
    receipt = database.execute(
        lambda connection: connection.execute(
            "SELECT * FROM messaging_legacy_import"
        ).fetchone(),
        write=False,
    )
    warning = None
    if receipt is None:
        data = read_legacy(path)

        def commit(connection: sqlite3.Connection) -> bool:
            references: set[tuple[str, str, str]] = set()
            trees = 0
            for snapshot in data.trees.values():
                scoped = {
                    (snapshot.scope.platform, snapshot.scope.chat_id, reference)
                    for reference in snapshot.lookup_ids()
                }
                if scoped & references or not _try_import(
                    connection, partial(write_tree, connection, snapshot)
                ):
                    data.reject()
                    continue
                references.update(scoped)
                trees += 1
            seen: set[tuple[str, str, str]] = set()
            for record in data.messages:
                key = record[:3]
                if key in seen:
                    continue
                if not _try_import(
                    connection,
                    partial(
                        connection.execute,
                        "INSERT INTO messaging_managed_messages(platform,chat_id,message_id,ts,direction,kind) VALUES (?,?,?,?,?,?)",
                        record,
                    ),
                ):
                    data.reject()
                    continue
                seen.add(key)
            complete = data.outcome in {"complete", "absent"}
            connection.execute(
                "INSERT INTO messaging_legacy_import VALUES ('sessions.json',?,?,?,?,?)",
                (
                    data.outcome,
                    trees,
                    len(seen),
                    data.skipped,
                    int(complete),
                ),
            )
            return complete

        complete = database.execute(commit)
        if not complete:
            return f"Some previous messaging history could not be restored. Messaging can still be used. The original file was kept at {path}."
    elif not receipt["cleanup_pending"]:
        # Incomplete-import notices are one-time. The receipt still prevents replay.
        return None
    try:
        if path.is_symlink():
            raise OSError("Legacy source is redirected")
        path.unlink(missing_ok=True)
        for temporary in path.parent.glob(".sessions.*.tmp.json"):
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink()
    except OSError:
        warning = f"Messaging history was migrated. Old files at {path.parent} could not be removed yet. FCC will retry cleanup at startup."
    else:
        database.execute(
            lambda connection: connection.execute(
                "UPDATE messaging_legacy_import SET cleanup_pending=0"
            ).close()
        )
    return warning

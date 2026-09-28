import sqlite3
from contextlib import closing

import pytest

from free_claude_code.runtime import sqlite_database
from tests.runtime.test_sqlite_database import historical_database, snapshot


def test_harness_upgrade_preserves_history_and_accepts_claude(tmp_path):
    path = tmp_path / "fcc.db"
    historical_database(path, 3)
    before = snapshot(path)
    sqlite_database.initialize_database(path)
    assert snapshot(path) == before
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("UPDATE code_sessions SET harness='claude', mode='acceptEdits'")
        db.execute("UPDATE code_runs SET mode='plan'")
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE code_sessions SET harness='unknown'")


@pytest.mark.parametrize(
    "table", ["code_prompts", "code_items", "code_runs", "code_sessions"]
)
def test_harness_upgrade_rolls_back_after_each_drop(tmp_path, monkeypatch, table):
    path = tmp_path / "fcc.db"
    historical_database(path, 3)
    migrations = sqlite_database.MIGRATIONS
    monkeypatch.setattr(sqlite_database, "MIGRATIONS", migrations[:5])
    sqlite_database.initialize_database(path)
    before = snapshot(path)
    with closing(sqlite3.connect(path)) as db:
        schema = db.execute(
            "SELECT name,sql FROM sqlite_schema ORDER BY name"
        ).fetchall()

    original = migrations[5][1]

    def fail_after_drop(connection):
        dropped = False

        def trace(sql):
            nonlocal dropped
            if sql == f"DROP TABLE {table}":
                dropped = True

        def progress():
            return int(dropped)

        connection.set_trace_callback(trace)
        connection.set_progress_handler(progress, 1)
        try:
            original(connection)
        finally:
            connection.set_progress_handler(None, 0)
            connection.set_trace_callback(None)

    monkeypatch.setattr(
        sqlite_database, "MIGRATIONS", (*migrations[:5], (6, fail_after_drop))
    )
    with pytest.raises(sqlite3.DatabaseError):
        sqlite_database.initialize_database(path)
    assert snapshot(path) == before
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (5,)
        assert (
            db.execute("SELECT name,sql FROM sqlite_schema ORDER BY name").fetchall()
            == schema
        )

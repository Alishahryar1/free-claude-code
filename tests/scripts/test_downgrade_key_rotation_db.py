"""Rollback removes cooldown state while retaining unrelated SQLite records."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from scripts import downgrade_key_rotation_db as downgrade


def create_database(path: Path, version: int, *, cooldown_view: bool = False) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE conversations (message TEXT NOT NULL)")
        connection.execute("INSERT INTO conversations VALUES ('retain this message')")
        if cooldown_view:
            connection.execute(
                "CREATE VIEW provider_key_cooldowns AS SELECT * FROM conversations"
            )
        else:
            connection.execute(
                "CREATE TABLE provider_key_cooldowns (credential_id TEXT PRIMARY KEY)"
            )
            connection.execute(
                "INSERT INTO provider_key_cooldowns VALUES ('hashed-id')"
            )
        connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()


def test_downgrade_preserves_conversations_and_backs_up_original(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fcc.db"
    create_database(path, 6)
    assert downgrade.main(["downgrade", str(path)]) == 0
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (5,)
        assert connection.execute("SELECT * FROM conversations").fetchall() == [
            ("retain this message",)
        ]
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'provider_key_cooldowns'"
            ).fetchall()
            == []
        )
    backups = list(tmp_path.glob("fcc.db.bak-*"))
    assert len(backups) == 1
    with closing(sqlite3.connect(backups[0])) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (6,)
        assert connection.execute(
            "SELECT * FROM provider_key_cooldowns"
        ).fetchall() == [("hashed-id",)]


@pytest.mark.parametrize("version", [0, 5, 7])
def test_other_schema_versions_are_unchanged(tmp_path: Path, version: int) -> None:
    path = tmp_path / "fcc.db"
    create_database(path, version)
    before = path.read_bytes()
    assert downgrade.main(["downgrade", str(path)]) == 1
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("fcc.db.bak-*"))


def test_missing_database_is_not_created(tmp_path: Path) -> None:
    path = tmp_path / "missing.db"
    assert downgrade.main(["downgrade", str(path)]) == 2
    assert not path.exists()


def test_existing_backup_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fcc.db"
    create_database(path, 6)
    backup = tmp_path / "fcc.db.bak-fixed"
    backup.write_bytes(b"preserve existing backup")

    def fixed_timestamp(_format: str) -> str:
        return "fixed"

    monkeypatch.setattr(downgrade.time, "strftime", fixed_timestamp)
    before = path.read_bytes()
    assert downgrade.main(["downgrade", str(path)]) == 1
    assert backup.read_bytes() == b"preserve existing backup"
    assert path.read_bytes() == before


def test_sql_error_rolls_back_changes_and_keeps_backup(tmp_path: Path) -> None:
    path = tmp_path / "fcc.db"
    create_database(path, 6, cooldown_view=True)
    assert downgrade.main(["downgrade", str(path)]) == 1
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (6,)
        assert connection.execute("SELECT * FROM conversations").fetchall() == [
            ("retain this message",)
        ]
    assert len(list(tmp_path.glob("fcc.db.bak-*"))) == 1

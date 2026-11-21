"""Exercise the live smoke's SQL assertion without launching a harness."""

import sqlite3
from contextlib import closing, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from free_claude_code.config.paths import FCC_DATABASE_FILENAME
from smoke.product import test_codex_modes_product_live as smoke


@pytest.mark.parametrize("database_exists", [False, True])
def test_child_review_smoke_reads_canonical_database_without_creating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, database_exists: bool
):
    home = tmp_path / "home/.fcc"
    home.mkdir(parents=True)
    (home / "code").mkdir()
    database_path = home / FCC_DATABASE_FILENAME
    if database_exists:
        with closing(sqlite3.connect(database_path)) as connection, connection:
            connection.execute("CREATE TABLE code_sessions(id, native_thread_id)")
            connection.execute("INSERT INTO code_sessions VALUES ('session','parent')")
            connection.execute("CREATE TABLE code_items(session_id,kind,raw)")
            connection.execute(
                "INSERT INTO code_items VALUES ('session','subagent_auto_review',?)",
                ('{"threadId":"child"}',),
            )
    (tmp_path / "codex-home").mkdir()
    (tmp_path / "child-marker.txt").write_text("smoke")

    def replace_environment(*_: object) -> tuple[dict[str, str], set[str]]:
        return ({}, set())

    monkeypatch.setattr(smoke, "_environment", replace_environment)

    def replace_canned_provider(*_: object):
        return nullcontext(("http://provider.invalid", {"reviews": 1}))

    monkeypatch.setattr(
        smoke,
        "_canned_provider",
        replace_canned_provider,
    )

    def replace_SmokeServerDriver(*args: object, **kwargs: object):
        return SimpleNamespace(
            run=lambda: nullcontext(SimpleNamespace(base_url="http://server.invalid"))
        )

    monkeypatch.setattr(
        smoke,
        "SmokeServerDriver",
        replace_SmokeServerDriver,
    )

    async def completed_review(*_):
        return "session"

    monkeypatch.setattr(smoke, "_exercise_child_review", completed_review)
    config = MagicMock(spec=smoke.SmokeConfig, timeout_s=1)
    if database_exists:
        smoke.test_codex_child_review_local_e2e(config, tmp_path)
    else:
        with pytest.raises(sqlite3.OperationalError, match="unable to open"):
            smoke.test_codex_child_review_local_e2e(config, tmp_path)
    assert not (home / "code/code.db").exists()
    assert database_path.exists() == database_exists

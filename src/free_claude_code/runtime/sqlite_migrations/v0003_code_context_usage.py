"""Version 3: optional model context usage for Code sessions."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    connection.execute(
        "ALTER TABLE code_sessions ADD COLUMN context_used_tokens INTEGER "
        "CHECK(context_used_tokens IS NULL OR context_used_tokens >= 0)"
    )

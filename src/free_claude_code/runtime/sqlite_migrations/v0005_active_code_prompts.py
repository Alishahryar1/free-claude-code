"""Index the live prompt working set independently of saved history."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE INDEX code_prompts_active ON code_prompts(session_id, id) "
        "WHERE status IN ('pending', 'answering')"
    )

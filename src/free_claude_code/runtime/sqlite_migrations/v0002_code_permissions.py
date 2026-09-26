"""Version 2: persisted permission mode and original native defaults."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    for table in ("code_sessions", "code_runs"):
        connection.execute(
            f"ALTER TABLE {table} ADD COLUMN mode TEXT NOT NULL DEFAULT 'config' "
            "CHECK(mode IN ('config','ask','auto_review','full_access'))"
        )
    connection.execute(
        "ALTER TABLE code_sessions ADD COLUMN native_permission_defaults TEXT "
        "CHECK(native_permission_defaults IS NULL OR "
        "(json_valid(native_permission_defaults) AND json_type(native_permission_defaults) = 'object'))"
    )

"""Persist credential cooldown fingerprints without storing credentials."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE provider_key_cooldowns (
            credential_id TEXT PRIMARY KEY NOT NULL,
            provider_id TEXT NOT NULL,
            key_number INTEGER NOT NULL CHECK (key_number > 0),
            cooldown_until REAL NOT NULL,
            last_error TEXT
        )
    """)
    connection.execute(
        "CREATE INDEX provider_key_cooldowns_expiry "
        "ON provider_key_cooldowns(cooldown_until)"
    )

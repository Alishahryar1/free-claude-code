"""Roll an FCC SQLite database back from schema v6 (key rotation) to v5.

Stop FCC first. The script copies the database to a timestamped backup, then
drops only the ``provider_key_cooldowns`` table and sets ``user_version`` to 5.
No conversation, messaging or code-session data is touched.

Usage: python scripts/downgrade_key_rotation_db.py /path/to/fcc.db
"""

import shutil
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"Database not found: {path}")
        return 2
    with closing(sqlite3.connect(path)) as probe:
        version: object = probe.execute("PRAGMA user_version").fetchone()[0]
    if version != 6:
        print(f"Nothing to do: schema version is {version}, expected 6.")
        return 1
    backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    if backup.exists():
        print(f"Backup already exists: {backup}. Nothing changed; retry later.")
        return 1
    shutil.copy2(path, backup)
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, backup.with_name(backup.name + suffix))
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TABLE IF EXISTS provider_key_cooldowns")
        connection.execute("PRAGMA user_version = 5")
        connection.execute("COMMIT")
    except sqlite3.Error as error:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        print(f"Downgrade failed, database unchanged: {error}")
        return 1
    finally:
        connection.close()
    print(f"Downgraded to schema 5. Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

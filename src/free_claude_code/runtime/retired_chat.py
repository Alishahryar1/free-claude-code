"""Remove saved data belonging to the retired Admin Chat Sessions feature."""

from loguru import logger

from free_claude_code.config import paths
from free_claude_code.core.interprocess_lock import InterprocessFileLock


def remove_retired_chat_history() -> None:
    """Delete only owned history files, deferring failures to the next startup."""
    try:
        directory = paths.config_dir_path().resolve() / "chat"
        try:
            resolved = directory.resolve(strict=True)
        except FileNotFoundError:
            return
        if resolved != directory:
            logger.warning("Chat history cleanup deferred: redirected directory")
            return
        files = tuple(
            directory / name
            for name in ("chat.db", "chat.db-wal", "chat.db-shm", "chat.db-journal")
        )
        for path in files:
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            break
        else:
            return

        lock = InterprocessFileLock(directory / "chat.lock")
        try:
            if not lock.acquire():
                logger.warning("Chat history cleanup deferred: history is in use")
                return
            for path in files:
                path.unlink(missing_ok=True)
        finally:
            # Keep the lock file: unlinking it could create two independent locks.
            lock.release()
    except OSError as exc:
        logger.warning(
            "Chat history cleanup deferred until next startup: exc_type={}",
            type(exc).__name__,
        )

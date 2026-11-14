"""Private smoke runtime state with the developer's original credential ownership."""

import os
import time
from collections.abc import Generator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from free_claude_code.config import paths
from free_claude_code.config.env_migrations import atomic_write_managed_config
from free_claude_code.config.loader import ManagedConfigStore

_CONFIG_DIR = "FCC_SMOKE_CONFIG_DIR"
_SOURCE_DIR = "FCC_SMOKE_SOURCE_CONFIG_DIR"


@contextmanager
def server_environment(
    env: Mapping[str, str], *, caller_home: bool
) -> Generator[dict[str, str]]:
    child_env = dict(env)
    child_env.pop(_CONFIG_DIR, None)
    child_env.pop(_SOURCE_DIR, None)
    if caller_home:
        yield child_env
        return

    snapshot = ManagedConfigStore().read()
    directory = TemporaryDirectory(prefix="fcc-smoke-")
    try:
        private = Path(directory.name)
        atomic_write_managed_config(snapshot.managed, path=private / ".env")
        child_env[_CONFIG_DIR] = str(private)
        child_env[_SOURCE_DIR] = str(snapshot.path.parent)
        yield child_env
    finally:
        # Windows can release child database/lock handles just after taskkill
        # returns. Retry only that sharing violation, never hide cleanup failure.
        deadline = time.monotonic() + 5
        while True:
            try:
                directory.cleanup()
                break
            except PermissionError as error:
                if (
                    getattr(error, "winerror", None) != 32
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(0.05)


def serve() -> None:
    """Set FCC paths before importing the real server entrypoint."""
    directory = os.environ.pop(_CONFIG_DIR, None)
    with ExitStack() as stack:
        if directory is not None:
            source = Path(os.environ.pop(_SOURCE_DIR))
            with patch.object(paths, "config_dir_path", lambda: source):
                auth = paths.openai_auth_path()
                auth_lock = paths.openai_auth_lock_path()
                copilot = paths.github_copilot_auth_path()
            # Keep rotating OAuth credentials and their refresh lock together.
            stack.enter_context(
                patch.multiple(
                    paths,
                    config_dir_path=lambda: Path(directory),
                    openai_auth_path=lambda: auth,
                    openai_auth_lock_path=lambda: auth_lock,
                    github_copilot_auth_path=lambda: copilot,
                )
            )

        from free_claude_code.cli.entrypoints import serve as run_server

        run_server()

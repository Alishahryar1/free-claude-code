"""Atomic text-file replacement for local client configuration."""

import os
import stat
import tempfile
from pathlib import Path


def ensure_private_permissions(path: Path) -> None:
    """Restrict POSIX access; Windows files inherit their profile directory ACL."""
    if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        path.chmod(0o600)


def atomic_write_text(path: Path, content: str, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if private and os.name != "nt":
            ensure_private_permissions(temporary)
        elif path.exists():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            # Windows cannot delete a temporary file after copying a read-only mode.
            temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
            temporary.unlink()

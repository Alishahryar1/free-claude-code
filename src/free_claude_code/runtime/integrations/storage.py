"""Atomic client writes and minimal, private undo records."""

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from free_claude_code.application.integrations import (
    IntegrationAction,
    IntegrationError,
    IntegrationId,
)

from .documents import MISSING


def content_hash(data: bytes | None) -> str:
    return hashlib.sha256(
        b"missing" if data is None else b"present\0" + data
    ).hexdigest()


def check_path(path: Path) -> None:
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise IntegrationError(
                "This configuration uses a symlink or junction. Use manual setup for this location."
            )


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    data: bytes | None
    identity: tuple[int, int, int, int] | None

    @classmethod
    def read(cls, path: Path) -> FileSnapshot:
        check_path(path)
        try:
            before = path.stat()
            if not stat.S_ISREG(before.st_mode):
                raise IntegrationError("A configuration path is not a regular file.")
            data = path.read_bytes()
            after = path.stat()
            identity = (after.st_ino, after.st_mtime_ns, after.st_size, after.st_mode)
            if identity != (
                before.st_ino,
                before.st_mtime_ns,
                before.st_size,
                before.st_mode,
            ):
                raise IntegrationError(
                    "A configuration file changed while being read. Refresh and try again.",
                    status_code=409,
                )
            return cls(path, data, identity)
        except FileNotFoundError:
            return cls(path, None, None)
        except OSError:
            raise IntegrationError(
                "Could not read a configuration file. Check the displayed path and its permissions."
            ) from None

    @property
    def digest(self) -> str:
        return content_hash(self.data)

    def require_unchanged(self) -> None:
        if FileSnapshot.read(self.path) != self:
            raise IntegrationError(
                "Settings changed since this preview. Refresh and confirm the new changes.",
                status_code=409,
            )

    def require_writable(self) -> None:
        if self.identity is not None and not self.identity[3] & 0o222:
            raise IntegrationError(
                "The configuration file is read-only. Change its permissions before continuing."
            )


class SavedValue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    present: bool
    value: JsonValue = None

    @classmethod
    def capture(cls, value: object) -> SavedValue:
        return cls.model_validate(
            {
                "present": value is not MISSING,
                "value": None if value is MISSING else value,
            }
        )

    def unpack(self) -> object:
        return self.value if self.present else MISSING


class FieldHistory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prior: SavedValue | None
    last: SavedValue


class Ownership(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    item: IntegrationId
    path: str
    fields: dict[str, FieldHistory]
    created_containers: list[list[str]] = []


class PendingWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    phase: Literal["pending"] = "pending"
    item: IntegrationId
    action: IntegrationAction
    path: str
    before: str
    after: str
    previous: Ownership | None
    following: Ownership | None


def atomic_write(
    path: Path, data: bytes, *, mode: int = 0o600, expected: FileSnapshot | None = None
) -> None:
    check_path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".fcc-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(stat.S_IMODE(mode))
        if expected is not None:
            expected.require_unchanged()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_record(path: Path, record: Ownership | PendingWrite | None) -> None:
    if record is None:
        path.unlink(missing_ok=True)
    else:
        atomic_write(path, record.model_dump_json().encode())

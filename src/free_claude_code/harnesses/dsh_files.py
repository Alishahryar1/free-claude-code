"""Round-trip YAML and cooperative locks for DSH-owned configuration files."""

import errno
import io
import os
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError

from free_claude_code.harnesses.config_file import atomic_write_text


class DshConfigError(ValueError):
    """A native configuration cannot be changed without losing user intent."""


def regular_path(path: Path, *, root: Path | None = None) -> None:
    """Reject redirected files and managed directory ancestors without following them."""
    candidates = [path]
    if root is not None:
        if not path.is_relative_to(root):
            raise DshConfigError(
                "DSH configuration must remain inside its selected home."
            )
        candidates.extend(
            parent
            for parent in path.parents
            if parent != root and parent.is_relative_to(root)
        )
    for item in candidates:
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(
            info, "st_file_attributes", 0
        ) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise DshConfigError(
                "DSH integration files and directories must be regular paths, not links. Keep the managed files in the selected DSH home."
            )
        expected = stat.S_ISREG if item == path else stat.S_ISDIR
        if not expected(info.st_mode):
            raise DshConfigError(
                "DSH integration needs regular configuration files and directories."
            )


def yaml_parser() -> YAML:
    parser = YAML(typ="rt", pure=True)
    parser.preserve_quotes = True
    parser.allow_duplicate_keys = False
    return parser


def read_yaml(path: Path, *, sequence: bool = False) -> CommentedMap | CommentedSeq:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        source = ""
    try:
        document = yaml_parser().load(source)
    except YAMLError, UnicodeError:
        # YAML parser messages can quote a line containing a credential.
        raise DshConfigError(
            f"Invalid YAML in {path.name}. Correct it and retry."
        ) from None
    if document is None:
        document = CommentedSeq() if sequence else CommentedMap()
    expected = CommentedSeq if sequence else CommentedMap
    if not isinstance(document, expected):
        raise DshConfigError(f"Unexpected document structure in {path.name}.")
    return document


def mapping(value: object) -> CommentedMap:
    if not isinstance(value, CommentedMap) or value.tag.value not in (
        None,
        "tag:yaml.org,2002:map",
    ):
        raise DshConfigError(
            "DSH settings need a plain YAML mapping at the FCC integration fields."
        )
    if value.anchor.value:
        raise DshConfigError(
            "DSH settings use an anchor at an FCC integration field. Separate that field before configuring FCC."
        )
    return value


def write_text(path: Path, content: str, *, private: bool = False) -> bool:
    try:
        if path.read_text(encoding="utf-8") == content:
            return False
    except FileNotFoundError:
        pass
    deadline = time.monotonic() + 2
    while True:
        try:
            atomic_write_text(path, content, private=private)
            return True
        except OSError as exc:
            if (
                os.name != "nt"
                or exc.errno not in (errno.EACCES, errno.EBUSY, errno.EPERM)
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(0.05)


def yaml_text(document: CommentedMap | CommentedSeq) -> str:
    # ruamel's flow emitter cannot round-trip a commented empty root sequence.
    # Both styles spell an empty sequence as []; block style retains its comment.
    if isinstance(document, CommentedSeq) and not document:
        document.fa.set_block_style()
    stream = io.StringIO()
    yaml_parser().dump(document, stream)
    return stream.getvalue()


def write_yaml(
    path: Path, document: CommentedMap | CommentedSeq, *, private: bool = False
) -> bool:
    return write_text(path, yaml_text(document), private=private)


@contextmanager
def file_lock(path: Path, *, wait: float) -> Iterator[None]:
    """Honor DSH's exclusive PID-file lock; leave stale takeover to DSH."""
    deadline = time.monotonic() + wait
    delay = 0.02
    while True:
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
                0o600,
            )
            break
        except FileExistsError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "DSH is updating its configuration. Finish the native edit and retry."
                ) from None
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.2)
    record = f"{os.getpid()}\n".encode()
    identity = os.fstat(descriptor)
    try:
        os.write(descriptor, record)
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            current = path.stat()
            if (current.st_dev, current.st_ino) == (
                identity.st_dev,
                identity.st_ino,
            ) and path.read_bytes() == record:
                path.unlink()
        except FileNotFoundError:
            pass

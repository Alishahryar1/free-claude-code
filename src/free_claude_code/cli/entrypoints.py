"""Lightweight entry points for installed Free Claude Code commands."""

import os
import socket
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path

from free_claude_code.core.version import package_version

PROJECT_PORT_BASE = 18100
PROJECT_PORT_RANGE = 2000


def serve(argv: Sequence[str] | None = None) -> None:
    """Start the FastAPI server (registered as ``fcc-server``)."""
    if _print_version_if_requested(argv):
        return
    _apply_project_arg(argv, program="fcc-server")

    # Keep the server composition root off metadata-only command paths.
    from free_claude_code.cli.commands import serve as run_server

    args = sys.argv[1:] if argv is None else argv
    run_server(console_logging="--auto-started" not in args)


def _print_version_if_requested(argv: Sequence[str] | None) -> bool:
    args = sys.argv[1:] if argv is None else argv
    if "--version" not in args:
        return False
    print(f"free-claude-code {package_version()}")
    return True


def _apply_project_arg(
    argv: Sequence[str] | None, *, program: str = "fcc-server"
) -> None:
    """Give this instance a global per-project config slot for ``--project DIR``.

    The slot lives under ``~/.fcc/projects/<dashed-path>/`` (Claude Code's own
    naming pattern), keeping config, the ``fcc.db`` database, logs, and locks
    OUT of the project directory itself — nothing FCC-related lands where a
    user might commit or expose it. A missing slot and its managed ``.env``
    are scaffolded on first run; every home path follows the slot afterwards
    via :func:`free_claude_code.config.paths.config_dir_path`.

    The slot also receives a stable port, allocated once from a hash of the
    project path and persisted in its ``.env``, so a project keeps the same
    port on every launch without the user ever choosing one; an explicit
    ``PORT=`` in the slot always wins.

    An explicit ``FCC_HOME`` already present in the environment wins over the
    flag, so launchers and tests keep full control. The flag pair is stripped
    from ``sys.argv`` when ``argv`` is ``None`` so downstream argument
    scanners never see it.

    Every ``fcc-`` client launcher (``fcc-claude``, ``fcc-codex``, ...) routes
    through the same hook, so ``fcc-claude --project DIR`` resolves the exact
    slot the server would have used.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--project" not in args:
        return
    try:
        i = args.index("--project")
        raw = args[i + 1]
    except ValueError, IndexError:
        print(f"usage: {program} --project <directory>")
        raise SystemExit(2) from None
    project = Path(raw).expanduser().resolve()
    from free_claude_code.config import paths

    slot = paths.project_slot_path(project)
    slot.mkdir(parents=True, exist_ok=True)
    env_file = slot / ".env"
    if not env_file.exists():
        env_file.write_text(
            f"# Free Claude Code configuration for:\n"
            f"#   {project}\n"
            f"# Managed settings edited at /admin live in this slot; nothing\n"
            f"# FCC-related is written into the project directory.\n"
            f"FCC_CONFIG_SCHEMA=2\n",
            encoding="utf-8",
        )
    _allocate_project_port(slot, slot.name)
    os.environ.setdefault("FCC_HOME", str(slot))
    print(f"project slot: {slot}")
    keep = [a for j, a in enumerate(args) if j not in (i, i + 1)]
    if argv is None:
        sys.argv[1:] = keep


def _slot_env_port(env_file: Path) -> int | None:
    """Return an explicit PORT from a managed .env file, if any."""

    if not env_file.is_file():
        return None
    try:
        text = env_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("PORT="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _reserved_project_ports(exclude: Path) -> set[int]:
    """Ports already claimed by the default home or any other project slot."""

    from free_claude_code.config import paths

    reserved: set[int] = set()
    default_port = _slot_env_port(paths._default_config_dir_path() / ".env")
    if default_port:
        reserved.add(default_port)
    projects_root = paths._default_config_dir_path() / "projects"
    if projects_root.is_dir():
        for slot in projects_root.iterdir():
            if not slot.is_dir() or slot == exclude:
                continue
            port = _slot_env_port(slot / ".env")
            if port:
                reserved.add(port)
    return reserved


def _port_is_listening(port: int) -> bool:
    """Report whether something already accepts connections on this port."""

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def _allocate_project_port(slot: Path, name: str) -> int:
    """Give this project slot a stable port, allocated once and never regenerated.

    An explicit ``PORT=`` in the slot's managed ``.env`` always wins. Otherwise
    the port is derived deterministically from the slot name, so the same
    project keeps the same port across launches, reinstalls, and machines.
    When the derived port is already claimed by another slot or an active
    listener, the next free port in range is used instead; the chosen value is
    persisted into the slot ``.env`` so it never changes afterwards.
    """

    env_file = slot / ".env"
    explicit = _slot_env_port(env_file)
    if explicit is not None:
        return explicit
    reserved = _reserved_project_ports(slot)
    offset = zlib.crc32(name.encode("utf-8")) % PROJECT_PORT_RANGE
    port = PROJECT_PORT_BASE + offset
    for _ in range(PROJECT_PORT_RANGE):
        if port not in reserved and not _port_is_listening(port):
            break
        port = PROJECT_PORT_BASE + (port - PROJECT_PORT_BASE + 1) % PROJECT_PORT_RANGE
    else:  # pragma: no cover - the whole range is claimed
        print("No free project port is available; set PORT=<port> in the slot .env.")
        raise SystemExit(2)
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n# Stable port for this project's FCC instance, allocated once;\n"
            "# edit here or in /admin to choose a different one.\n"
            f"PORT={port}\n"
        )
    print(f"project port: {port}")
    return port


def _strip_project_pair(seq: Sequence[str]) -> list[str]:
    """Drop a ``--project DIR`` pair so downstream scanners never see it."""

    args = list(seq)
    if "--project" not in args:
        return args
    i = args.index("--project")
    return [a for j, a in enumerate(args) if j not in (i, i + 1)]


def doctor(argv: Sequence[str] | None = None) -> None:
    """Print and copy a local diagnostic report (registered as ``fcc-doctor``)."""
    if _print_version_if_requested(argv):
        return
    from free_claude_code.cli.doctor import main

    main(argv)

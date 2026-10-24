"""Lightweight entry points for installed Free Claude Code commands."""

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from free_claude_code.core.version import package_version


def serve(argv: Sequence[str] | None = None) -> None:
    """Start the FastAPI server (registered as ``fcc-server``)."""
    if _print_version_if_requested(argv):
        return
    _apply_project_arg(argv)

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


def _apply_project_arg(argv: Sequence[str] | None) -> None:
    """Give this instance a global per-project config slot for ``--project DIR``.

    The slot lives under ``~/.fcc/projects/<dashed-path>/`` (Claude Code's own
    naming pattern), keeping config, the ``fcc.db`` database, logs, and locks
    OUT of the project directory itself — nothing FCC-related lands where a
    user might commit or expose it. A missing slot and its managed ``.env``
    are scaffolded on first run; every home path follows the slot afterwards
    via :func:`free_claude_code.config.paths.config_dir_path`.

    An explicit ``FCC_HOME`` already present in the environment wins over the
    flag, so launchers and tests keep full control. The flag pair is stripped
    from ``sys.argv`` so downstream argument scanners never see it.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--project" not in args:
        return
    try:
        i = args.index("--project")
        raw = args[i + 1]
    except ValueError, IndexError:
        print("usage: fcc-server --project <directory>")
        raise SystemExit(2) from None
    project = Path(raw).expanduser().resolve()
    from free_claude_code.config.paths import project_slot_path

    slot = project_slot_path(project)
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
    os.environ.setdefault("FCC_HOME", str(slot))
    print(f"project slot: {slot}")
    keep = [a for j, a in enumerate(args) if j not in (i, i + 1)]
    if argv is None:
        sys.argv[1:] = keep


def doctor(argv: Sequence[str] | None = None) -> None:
    """Print and copy a local diagnostic report (registered as ``fcc-doctor``)."""
    if _print_version_if_requested(argv):
        return
    from free_claude_code.cli.doctor import main

    main(argv)

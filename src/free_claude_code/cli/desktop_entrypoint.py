"""Lightweight entrypoint for the optional FCC desktop shell."""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from free_claude_code.cli.desktop_assets import export_app_icon
from free_claude_code.core.windows_dpi import enable_dpi_awareness


def launch(argv: Sequence[str] | None = None) -> None:
    """Export installer assets or launch the supported native tray adapter."""

    args = tuple(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "--export-icon":
        export_app_icon(Path(args[1]))
        return
    if args == ("--install-macos-app",) and sys.platform == "darwin":
        from free_claude_code.cli.macos_app import install

        try:
            install()
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"Could not install the macOS app: {error}", file=sys.stderr)
            raise SystemExit(1) from error
        return
    auto_started = args == ("--auto-started",)
    if args and not auto_started:
        print("Usage: fcc-desktop [--export-icon PATH]", file=sys.stderr)
        raise SystemExit(2)
    if sys.platform not in {"darwin", "win32"}:
        print("FCC Desktop is supported on Windows and macOS.", file=sys.stderr)
        raise SystemExit(1)

    enable_dpi_awareness()

    from free_claude_code.cli.desktop_tray import launch as launch_tray

    launch_tray(auto_started=auto_started)

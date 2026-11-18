"""Dispatch the signed Windows bundle's launchers through the wheel entry points."""

import sys
from importlib.metadata import distribution
from pathlib import Path


def main() -> int | None:
    launcher = Path(sys.argv.pop(1))
    sys.argv[0] = str(launcher.with_suffix(""))
    entries = distribution("free-claude-code").entry_points.select(name=launcher.stem)
    for entry in entries:
        if entry.group in {"console_scripts", "gui_scripts"}:
            return entry.load()()
    raise SystemExit(f"Unknown FCC command: {launcher.stem}")


if __name__ == "__main__":
    sys.exit(main())

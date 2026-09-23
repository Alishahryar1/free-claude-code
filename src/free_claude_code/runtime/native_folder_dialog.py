"""Short-lived OS dialog helper; stdout carries only its JSON result."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from free_claude_code.core.windows_dpi import enable_dpi_awareness

_TITLE = "Choose a Code Session folder"
_MAC_SCRIPT = """
on run argv
    try
        if item 1 of argv is "" then
            set chosen to choose folder with prompt "Choose a Code Session folder"
        else
            set chosen to choose folder with prompt "Choose a Code Session folder" default location (POSIX file (item 1 of argv))
        end if
        return POSIX path of chosen
    on error messageText number errorNumber
        if errorNumber is -128 then return ""
        error messageText number errorNumber
    end try
end run
"""
_DISPLAY_ERRORS = (
    "cannot open display",
    "failed to open display",
    "no qt platform plugin could be initialized",
    "could not connect to display",
)


def _initial_directory(value: str) -> str | None:
    if value:
        try:
            path = Path(value).expanduser().resolve(strict=True)
            if path.is_dir():
                return str(path)
        except OSError, ValueError:
            pass
    return None


def _windows(initial: str | None) -> str | None:
    enable_dpi_awareness()

    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        return (
            filedialog.askdirectory(
                parent=root, title=_TITLE, initialdir=initial or "", mustexist=True
            )
            or None
        )
    finally:
        root.destroy()


def _macos(initial: str | None) -> str | None:
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", _MAC_SCRIPT, initial or ""],
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8").removesuffix("\n") or None


def _linux(initial: str | None) -> str | None:
    env = os.environ.copy()
    if executable := shutil.which("zenity"):
        command = [executable, "--file-selection", "--directory", f"--title={_TITLE}"]
        if initial:
            command.append(f"--filename={initial.rstrip('/')}/")
        env = {
            key: value for key, value in env.items() if not key.startswith("ZENITY_")
        }
    elif executable := shutil.which("kdialog"):
        command = [executable, "--title", _TITLE, "--getexistingdirectory"]
        if initial:
            command.append(initial)
    else:
        raise RuntimeError("No desktop folder picker is available")
    result = subprocess.run(command, capture_output=True, env=env, check=False)
    path = result.stdout.decode("utf-8").removesuffix("\n")
    if result.returncode == 0 and path:
        return path
    diagnostic = result.stderr.decode("utf-8", errors="replace")
    if any(message in diagnostic.lower() for message in _DISPLAY_ERRORS):
        raise RuntimeError(diagnostic)
    if result.returncode == 1:
        return None
    raise RuntimeError(diagnostic or "Folder picker returned no directory")


def main() -> int:
    try:
        initial = _initial_directory(sys.argv[1] if len(sys.argv) > 1 else "")
        if sys.platform == "win32":
            path = _windows(initial)
        elif sys.platform == "darwin":
            path = _macos(initial)
        elif sys.platform.startswith("linux"):
            path = _linux(initial)
        else:
            raise RuntimeError("No native folder picker for this platform")
        print(json.dumps({"path": path}, ensure_ascii=True))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(json.dumps({"error": "unavailable"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())

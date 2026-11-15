"""Repair an installed Claude without loading FCC's managed-session runtime."""

import argparse
import io
import shutil
import subprocess
import sys

from free_claude_code.harnesses.claude import (
    CLAUDE_BINARY_NAME,
    CLAUDE_COMPATIBILITY_CHECK,
    CLAUDE_MINIMUM_VERSION,
    CLAUDE_MINIMUM_VERSION_TEXT,
    claude_version,
)


def _version(binary: str) -> tuple[int, int, int]:
    result = subprocess.run(
        [binary, *CLAUDE_COMPATIBILITY_CHECK.args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=CLAUDE_COMPATIBILITY_CHECK.timeout_seconds,
    )
    version = claude_version(result.stdout)
    if result.returncode != 0 or version is None:
        raise ValueError(f"Could not read a stable Claude Code version from {binary}.")
    return version


def main(argv: list[str] | None = None) -> int:
    # Redirected Windows consoles may not encode the selected installation path.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="_fcc-update-claude")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        binary = shutil.which(CLAUDE_BINARY_NAME)
        if binary is None:
            print("Claude Code is not installed; skipping Claude maintenance.")
            return 0
        if args.dry_run:
            print(
                f"Would check {binary} and run its update command only if older than "
                f"{CLAUDE_MINIMUM_VERSION_TEXT}."
            )
            return 0
        if _version(binary) >= CLAUDE_MINIMUM_VERSION:
            print(
                f"Claude Code satisfies >={CLAUDE_MINIMUM_VERSION_TEXT}; leaving it unchanged."
            )
            return 0
        print(
            f"Updating Claude Code at {binary} to meet >={CLAUDE_MINIMUM_VERSION_TEXT}.",
            flush=True,
        )
        result = subprocess.run([binary, "update"], check=False)
        if result.returncode != 0:
            raise ValueError(
                f"Claude Code update failed with exit code {result.returncode}."
            )
        binary = shutil.which(CLAUDE_BINARY_NAME)
        if binary is None:
            raise ValueError(
                "Claude Code is no longer available on PATH after updating."
            )
        version = _version(binary)
        if version < CLAUDE_MINIMUM_VERSION:
            found = ".".join(map(str, version))
            raise ValueError(
                f"Claude Code at {binary} is still {found} after updating."
            )
        print(f"Verified Claude Code {'.'.join(map(str, version))}.")
        return 0
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        print(f"{exc} {CLAUDE_COMPATIBILITY_CHECK.failure_message}", file=sys.stderr)
        return 1

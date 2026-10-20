"""Standalone version preflight for the native update launchers."""

import argparse
import re
import sys
from importlib.metadata import PackageNotFoundError, version

import httpx
from packaging.version import Version

PYPI_URL = "https://pypi.org/pypi/free-claude-code/json"
UPDATE_AVAILABLE = 10


def main(argv: list[str] | None = None) -> int:
    """Return 10 to authorize installation, 0 to skip, or 1 on check failure."""
    parser = argparse.ArgumentParser(prog="_fcc-update-check")
    parser.add_argument("--print-version", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        installed = Version(version("free-claude-code"))
        if str(installed) == "0+unknown":
            raise ValueError("installed FCC version is unknown")
        with httpx.Client(timeout=10.0) as client:
            response = client.get(PYPI_URL)
            response.raise_for_status()
            metadata = response.json()
        published = metadata["info"]["version"]
        if (
            not isinstance(published, str)
            or re.fullmatch(
                r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", published
            )
            is None
        ):
            raise ValueError("PyPI did not return a canonical FCC release version")
        latest = Version(published)
        files = metadata["urls"]
        if not isinstance(files, list) or not any(
            isinstance(file, dict) and file.get("yanked") is False for file in files
        ):
            raise ValueError("PyPI did not return an available stable FCC release")
    except (
        PackageNotFoundError,
        httpx.HTTPError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"Could not check for FCC updates: {exc}. Please retry.", file=sys.stderr)
        return 1

    output = sys.stderr if arguments.print_version else sys.stdout
    if installed == latest:
        print(f"FCC {installed} is already up to date.", file=output)
        return 0
    if installed > latest:
        print(
            f"FCC {installed} is newer than the published version {latest}. No update needed.",
            file=output,
        )
        return 0
    print(
        f"FCC update available: {installed} -> {latest}. Running the installer.",
        file=output,
    )
    if arguments.print_version:
        print(published)
    return UPDATE_AVAILABLE

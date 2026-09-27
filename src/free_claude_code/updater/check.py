"""Standalone version preflight for the native update launchers."""

import sys
from importlib.metadata import PackageNotFoundError, version

import httpx
from packaging.version import Version

PYPI_URL = "https://pypi.org/pypi/free-claude-code/json"
UPDATE_AVAILABLE = 10


def main() -> int:
    """Return 10 to authorize installation, 0 to skip, or 1 on check failure."""
    try:
        installed = Version(version("free-claude-code"))
        if str(installed) == "0+unknown":
            raise ValueError("installed FCC version is unknown")
        with httpx.Client(timeout=10.0) as client:
            response = client.get(PYPI_URL)
            response.raise_for_status()
            metadata = response.json()
        latest = Version(metadata["info"]["version"])
        files = metadata["urls"]
        if (
            latest.is_prerelease
            or latest.is_devrelease
            or not isinstance(files, list)
            or not any(
                isinstance(file, dict) and file.get("yanked") is False for file in files
            )
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

    if installed == latest:
        print(f"FCC {installed} is already up to date.")
        return 0
    if installed > latest:
        print(
            f"FCC {installed} is newer than the published version {latest}. No update needed."
        )
        return 0
    print(f"FCC update available: {installed} -> {latest}. Running the installer.")
    return UPDATE_AVAILABLE

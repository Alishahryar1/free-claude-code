"""Canonical installed Free Claude Code package version."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

_DISTRIBUTION_NAME = "free-claude-code"
_UNKNOWN_VERSION = "0+unknown"


def package_version() -> str:
    """Return installed metadata, or an explicit source-only fallback."""
    try:
        return distribution_version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return _UNKNOWN_VERSION

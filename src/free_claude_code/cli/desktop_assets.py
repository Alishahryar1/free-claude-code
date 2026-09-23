"""Packaged visual assets owned by the FCC desktop shell."""

from importlib.resources import files
from pathlib import Path

_ICON_FILES = {
    ".icns": "app-icon.icns",
    ".ico": "app-icon.ico",
    ".png": "app-icon.png",
}


def app_icon_bytes(suffix: str) -> bytes:
    """Read the packaged app icon matching a native file suffix."""

    normalized_suffix = suffix.lower()
    try:
        filename = _ICON_FILES[normalized_suffix]
    except KeyError as exc:
        supported = ", ".join(sorted(_ICON_FILES))
        raise ValueError(
            f"Unsupported app icon format {suffix!r}; expected one of: {supported}"
        ) from exc
    return files("free_claude_code").joinpath("assets", filename).read_bytes()


def export_app_icon(destination: Path) -> None:
    """Copy the packaged native icon to an installer-owned destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(app_icon_bytes(destination.suffix))

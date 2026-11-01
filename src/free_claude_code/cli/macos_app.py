"""Construct the macOS application around the installed managed Python runtime."""

import os
import platform
import plistlib
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from importlib.metadata import metadata
from importlib.resources import files
from pathlib import Path

from packaging.specifiers import SpecifierSet

from free_claude_code.cli.desktop_assets import export_app_icon

BUNDLE_ID = "io.github.alishahryar1.free-claude-code"
OWNER_FILE = ".free-claude-code-owner"


def runtime_settings() -> tuple[dict[str, str], str]:
    """Keep the venv executable while locating the standalone runtime's library."""
    if (
        platform.python_implementation() != "CPython"
        or sysconfig.get_config_var("Py_GIL_DISABLED")
        or sysconfig.get_config_var("Py_DEBUG")
    ):
        raise ValueError(
            "The desktop app requires a GIL-enabled CPython release build."
        )
    version = platform.python_version()
    requirement = metadata("free-claude-code")["Requires-Python"]
    if not requirement or version not in SpecifierSet(requirement):
        raise ValueError(
            "The installed Python version does not match Free Claude Code."
        )
    architecture = platform.machine()
    if architecture not in {"arm64", "x86_64"}:
        raise ValueError(f"Unsupported macOS Python architecture: {architecture}")
    # Do not resolve this symlink: CPython needs its venv path for pyvenv.cfg.
    executable = Path(sys.executable).absolute()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("The installed Python executable is missing or inaccessible.")
    library_name = sysconfig.get_config_var("LDLIBRARY")
    if not isinstance(library_name, str) or Path(library_name).name != library_name:
        raise ValueError("The managed Python shared library is unavailable.")
    library = Path(sys.base_prefix) / "lib" / library_name
    if not library.is_file():
        raise ValueError(f"The managed Python shared library is missing: {library}")
    return {
        "PythonExecutable": str(executable),
        "PythonLibrary": str(library.absolute()),
        "PythonVersion": version,
    }, architecture


def is_owned(app: Path) -> bool:
    if app.is_symlink() or not app.is_dir():
        return False
    # Existing shell bundles used Contents directly. Signed bundles use Resources.
    for directory in (app / "Contents/Resources", app / "Contents"):
        marker = directory / OWNER_FILE
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == BUNDLE_ID:
            return True
    return False


def install() -> Path:
    """Sign a complete sibling bundle before replacing an FCC-owned application."""
    if sys.platform != "darwin":
        raise ValueError("Application installation is supported only on macOS.")
    home = Path.home()
    app = home / "Applications/Free Claude Code.app"
    if (app.exists() or app.is_symlink()) and not is_owned(app):
        raise ValueError(
            f"An app not managed by Free Claude Code already exists at {app}."
        )
    settings, architecture = runtime_settings()
    template = (
        files("free_claude_code")
        .joinpath("assets", "macos", f"fcc-desktop-{architecture}")
        .read_bytes()
    )
    app.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=".fcc-app-", dir=app.parent))
    backup = root / "previous.app"
    published = False
    try:
        staged = root / app.name
        contents = staged / "Contents"
        resources = contents / "Resources"
        resources.mkdir(parents=True)
        (contents / "MacOS").mkdir()
        executable = contents / "MacOS/fcc-desktop"
        executable.write_bytes(template)
        executable.chmod(0o755)
        export_app_icon(resources / "AppIcon.icns")
        (resources / OWNER_FILE).write_text(BUNDLE_ID + "\n", encoding="utf-8")
        with (resources / "runtime.plist").open("wb") as stream:
            plistlib.dump(settings, stream)
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump(
                {
                    "CFBundleDisplayName": "Free Claude Code",
                    "CFBundleExecutable": "fcc-desktop",
                    "CFBundleIdentifier": BUNDLE_ID,
                    "CFBundleIconFile": "AppIcon",
                    "CFBundleName": "Free Claude Code",
                    "CFBundlePackageType": "APPL",
                    "LSMultipleInstancesProhibited": True,
                    "LSUIElement": True,
                },
                stream,
            )
        subprocess.run(
            ["/usr/bin/codesign", "--force", "--sign", "-", str(staged)], check=True
        )
        subprocess.run(
            ["/usr/bin/codesign", "--verify", "--strict", str(staged)], check=True
        )
        if app.exists():
            app.rename(backup)
        try:
            staged.rename(app)
        except OSError:
            if backup.exists():
                backup.rename(app)
            raise
        published = True
    finally:
        if published or not backup.exists():
            shutil.rmtree(root)
        else:
            print(
                f"The previous application is preserved at {backup}.", file=sys.stderr
            )
    desktop = home / "Desktop/Free Claude Code.app"
    desktop.parent.mkdir(parents=True, exist_ok=True)
    if desktop.is_symlink() and desktop.readlink() == app:
        return app
    if desktop.exists() or desktop.is_symlink():
        print(f"A non-FCC item already exists at {desktop}; leaving it unchanged.")
    else:
        desktop.symlink_to(app, target_is_directory=True)
    return app

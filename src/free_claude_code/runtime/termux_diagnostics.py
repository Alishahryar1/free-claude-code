"""Termux-specific checks that do not expose FCC configuration secrets."""

import importlib
import os
import platform
import shutil
import socket
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.paths import config_dir_path
from free_claude_code.config.settings import Settings
from free_claude_code.platforms.termux import (
    TermuxEnvironment,
    detect_termux_environment,
)

_REQUIRED_IMPORTS = (
    "fastapi",
    "uvicorn",
    "httpx",
    "pydantic",
    "pydantic_core",
    "sqlalchemy",
    "aiohttp",
    "cryptography",
)
_OPTIONAL_IMPORTS = ("tiktoken",)
_LAUNCHERS = (
    "claude",
    "codex",
    "pi",
    "opencode",
    "cline",
    "hermes",
    "dsh",
    "grok",
    "muse",
    "agy",
    "aider",
)


def _check(
    name: str, status: str, detail: str, fix: str | None = None
) -> dict[str, str]:
    result = {"name": name, "status": status, "detail": detail}
    if fix:
        result["suggested_fix"] = fix
    return result


def _localhost_round_trip() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(1.0)
        with socket.create_connection(listener.getsockname(), timeout=1.0) as client:
            connection, _address = listener.accept()
            connection.close()
            client.close()
    return True


def _config_is_writable() -> bool:
    config_dir = config_dir_path()
    probe_dir = config_dir if config_dir.is_dir() else config_dir.parent
    descriptor, probe_name = tempfile.mkstemp(prefix=".fcc-doctor-", dir=probe_dir)
    os.close(descriptor)
    Path(probe_name).unlink()
    return True


def _termux_paths_available(environment: TermuxEnvironment) -> bool:
    if environment.prefix is None or environment.home is None:
        return False
    prefix_bin = environment.prefix / "bin"
    return (
        prefix_bin.is_dir()
        and environment.home.is_dir()
        and os.access(prefix_bin, os.W_OK)
        and os.access(environment.home, os.W_OK)
    )


def _sqlite_round_trip() -> bool:
    with closing(sqlite3.connect(":memory:")) as database:
        return database.execute("SELECT 1").fetchone() == (1,)


def _config_settings() -> tuple[Settings | None, str]:
    store = ManagedConfigStore()
    if not store.path.is_file():
        return None, "missing"
    try:
        return store.read().settings, "valid"
    except ValueError:
        return None, "invalid"
    except OSError:
        return None, "unavailable"


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))
    return True


def collect_termux_diagnostics() -> dict[str, Any]:
    """Collect PASS/WARNING/FAIL checks without printing credentials or paths."""
    environment = detect_termux_environment()
    checks: list[dict[str, str]] = []

    if environment.termux:
        checks.append(_check("Termux", "PASS", "Termux environment detected."))
    else:
        checks.append(
            _check(
                "Termux",
                "FAIL",
                "This command is not running in a native Termux environment.",
                "Run fcc-doctor --termux inside the Termux app.",
            )
        )

    checks.append(
        _check(
            "Android runtime",
            "PASS" if environment.android else "FAIL",
            "Android runtime detected."
            if environment.android
            else "Android runtime was not detected.",
            None
            if environment.android
            else "Install and run FCC in Termux on Android.",
        )
    )
    checks.append(
        _check(
            "Bionic libc",
            "PASS" if environment.bionic else "FAIL",
            "Android/Bionic runtime detected."
            if environment.bionic
            else "Bionic libc was not detected.",
            None if environment.bionic else "Use Termux's native Python package.",
        )
    )
    checks.append(
        _check(
            "Architecture",
            "PASS" if environment.arm64 else "FAIL",
            f"Detected {platform.machine()}."
            if environment.arm64
            else f"FCC Termux support requires AArch64; detected {platform.machine()}.",
            None if environment.arm64 else "Use a Termux installation on ARM64.",
        )
    )
    try:
        paths_available = environment.termux and _termux_paths_available(environment)
    except OSError:
        paths_available = False
    checks.append(
        _check(
            "Termux paths",
            "PASS" if paths_available else "FAIL",
            "Termux home and prefix paths are available and writable."
            if paths_available
            else "Termux home or prefix bin directory is unavailable or not writable.",
            None
            if paths_available
            else "Check TERMUX_VERSION, PREFIX, HOME, and Termux directory permissions.",
        )
    )

    version_ok = (3, 14) <= sys.version_info[:2] < (3, 15)
    checks.append(
        _check(
            "Python",
            "PASS" if version_ok else "FAIL",
            f"Python {platform.python_version()} is "
            + ("supported." if version_ok else "outside the supported 3.14 range."),
            None if version_ok else "Install Termux's current Python package (3.14.x).",
        )
    )
    python_exists = bool(sys.executable) and Path(sys.executable).is_file()
    checks.append(
        _check(
            "Python executable",
            "PASS" if python_exists else "FAIL",
            "The active Python executable is available."
            if python_exists
            else "The active Python executable could not be found.",
            None if python_exists else "Reinstall Python with `pkg install python`.",
        )
    )

    uv_path = shutil.which("uv")
    checks.append(
        _check(
            "uv",
            "PASS" if uv_path else "WARNING",
            "uv is available." if uv_path else "uv was not found on PATH.",
            None if uv_path else "Install it with `pkg install uv`.",
        )
    )

    missing_required: list[str] = []
    for module_name in _REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
        except ImportError, OSError:
            missing_required.append(module_name)
    checks.append(
        _check(
            "Core dependencies",
            "FAIL" if missing_required else "PASS",
            "Required runtime dependencies import successfully."
            if not missing_required
            else "One or more required runtime dependencies could not be imported.",
            None
            if not missing_required
            else "Rerun scripts/install-termux.sh to repair the FCC environment.",
        )
    )
    try:
        sqlite_available = _sqlite_round_trip()
    except sqlite3.Error:
        sqlite_available = False
    checks.append(
        _check(
            "SQLite",
            "PASS" if sqlite_available else "FAIL",
            "SQLite memory database round-trip succeeded."
            if sqlite_available
            else "Python SQLite could not execute a basic query.",
            None
            if sqlite_available
            else "Reinstall Termux Python with SQLite support and rerun the installer.",
        )
    )

    missing_optional: list[str] = []
    for module_name in _OPTIONAL_IMPORTS:
        try:
            importlib.import_module(module_name)
        except ImportError, OSError:
            missing_optional.append(module_name)
    checks.append(
        _check(
            "Optional native extensions",
            "WARNING" if missing_optional else "PASS",
            "Optional tokenization extensions are available."
            if not missing_optional
            else "Optional tokenization is unavailable; approximate estimates will be used.",
        )
    )
    updater_path = shutil.which("fcc-update")
    checks.append(
        _check(
            "FCC updater",
            "PASS" if updater_path else "WARNING",
            "fcc-update is available."
            if updater_path
            else "fcc-update was not found on PATH.",
            None
            if updater_path
            else "Run scripts/install-termux.sh to install the Termux update command.",
        )
    )

    try:
        _config_is_writable()
    except OSError:
        checks.append(
            _check(
                "Configuration storage",
                "FAIL",
                "The FCC configuration location is not writable.",
                "Check Termux storage permissions and the ownership of ~/.fcc.",
            )
        )
    else:
        checks.append(
            _check(
                "Configuration storage",
                "PASS",
                "The FCC configuration location is writable.",
            )
        )

    _settings, config_state = _config_settings()
    config_status = {
        "valid": ("PASS", "FCC configuration is valid.", None),
        "missing": (
            "WARNING",
            "No FCC configuration exists yet.",
            "Start fcc-server once to create the default configuration.",
        ),
        "invalid": (
            "FAIL",
            "FCC configuration could not be validated.",
            "Back up and correct ~/.fcc/.env; do not share its contents.",
        ),
        "unavailable": (
            "FAIL",
            "FCC configuration could not be read.",
            "Check permissions on ~/.fcc/.env.",
        ),
    }[config_state]
    checks.append(_check("Configuration", *config_status))

    try:
        _localhost_round_trip()
    except OSError:
        checks.append(
            _check(
                "Localhost networking",
                "FAIL",
                "A loopback TCP connection could not be completed.",
                "Check Android VPN/firewall settings and retry in Termux.",
            )
        )
    else:
        checks.append(_check("Localhost networking", "PASS", "Loopback TCP works."))

    port = _settings.port if _settings is not None else Settings().port
    try:
        _port_is_available(port)
    except OSError:
        checks.append(
            _check(
                "FCC port",
                "WARNING",
                f"Port {port} is already occupied; this may be a running FCC server.",
                "Stop the other listener or set a free PORT in ~/.fcc/.env.",
            )
        )
    else:
        checks.append(
            _check("FCC port", "PASS", f"Port {port} is available on localhost.")
        )

    available_launchers = [name for name in _LAUNCHERS if shutil.which(name)]
    checks.append(
        _check(
            "Coding-agent launchers",
            "PASS" if available_launchers else "WARNING",
            "Available: " + ", ".join(available_launchers)
            if available_launchers
            else "No supported external coding-agent executable is on PATH.",
            None
            if available_launchers
            else "Install a Termux-compatible coding agent; FCC does not install agents here.",
        )
    )
    browser_opener = shutil.which("termux-open-url")
    checks.append(
        _check(
            "Admin UI browser handoff",
            "PASS" if browser_opener else "WARNING",
            "Termux browser handoff is available."
            if browser_opener
            else "Automatic browser launching may be unavailable.",
            None
            if browser_opener
            else "Open the localhost URL printed by fcc-server manually in Android's browser.",
        )
    )

    statuses = {check["status"] for check in checks}
    summary = (
        "FAIL" if "FAIL" in statuses else "WARNING" if "WARNING" in statuses else "PASS"
    )
    return {
        "platform": "Termux",
        "summary": summary,
        "checks": checks,
    }

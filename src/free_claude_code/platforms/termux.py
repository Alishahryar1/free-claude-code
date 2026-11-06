"""Native Android Termux environment detection."""

import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TermuxEnvironment:
    """Termux and Android runtime facts used by platform-specific features."""

    termux: bool
    android: bool
    bionic: bool
    arm64: bool
    prefix: Path | None
    home: Path | None
    android_api_level: int | None


def _android_api_level() -> int | None:
    get_api_level = getattr(sys, "getandroidapilevel", None)
    if not callable(get_api_level):
        return None
    try:
        value = int(get_api_level())
    except OSError, TypeError, ValueError:
        return None
    return value if value > 0 else None


def is_android_runtime() -> bool:
    """Detect Android without relying on the Linux ``sys.platform`` value."""
    release = platform.release().casefold()
    return (
        sys.platform == "android"
        or platform.system().casefold() == "android"
        or "android" in release
        or _android_api_level() is not None
    )


def is_bionic_runtime() -> bool:
    """Identify Bionic using Android runtime evidence or libc metadata."""
    libc_name, _libc_version = platform.libc_ver()
    return is_android_runtime() or "bionic" in libc_name.casefold()


def is_termux_runtime(environ: Mapping[str, str] | None = None) -> bool:
    """Check Termux's exported runtime identity and native Android runtime."""
    env = os.environ if environ is None else environ
    prefix = env.get("PREFIX")
    home = env.get("HOME")
    return bool(
        env.get("TERMUX_VERSION")
        and prefix
        and Path(prefix).is_absolute()
        and home
        and Path(home).is_absolute()
        and is_android_runtime()
    )


def detect_termux_environment(
    environ: Mapping[str, str] | None = None,
) -> TermuxEnvironment:
    """Return a consistent snapshot of Termux, Android, libc, and architecture."""
    env = os.environ if environ is None else environ
    termux = is_termux_runtime(env)
    prefix = Path(env["PREFIX"]) if termux else None
    home = Path(env["HOME"]) if termux else None
    return TermuxEnvironment(
        termux=termux,
        android=is_android_runtime(),
        bionic=is_bionic_runtime(),
        arm64=platform.machine().casefold() in {"aarch64", "arm64"},
        prefix=prefix,
        home=home,
        android_api_level=_android_api_level(),
    )

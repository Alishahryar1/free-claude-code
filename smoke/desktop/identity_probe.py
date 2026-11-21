"""Observe real AppKit process identity and status-item geometry."""

import json
import locale
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol

from PIL import Image
from pystray import Icon, Menu, MenuItem

from free_claude_code.core.json_types import JsonObject


def write_report(output: Path, report: JsonObject):
    pending = output.with_suffix(".pending")
    pending.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pending.replace(output)


class _Bundle(Protocol):
    def bundlePath(self) -> str: ...
    def bundleIdentifier(self) -> str | None: ...


class _BundleFactory(Protocol):
    def mainBundle(self) -> _Bundle: ...


class _Application(Protocol):
    def bundleIdentifier(self) -> str | None: ...


class _RunningApplication(Protocol):
    def currentApplication(self) -> _Application: ...


class _Timer(Protocol):
    def invalidate(self) -> None: ...


class _TimerFactory(Protocol):
    def scheduledTimerWithTimeInterval_repeats_block_(
        self, interval: float, repeat: bool, block: Callable[[object], None], /
    ) -> _Timer: ...


def schedule_probe(observe: Callable[[object], None]) -> _Timer:
    factory: _TimerFactory | None = getattr(
        import_module("Foundation"), "NSTimer", None
    )
    if factory is None:
        raise RuntimeError("Foundation.NSTimer is unavailable")
    return factory.scheduledTimerWithTimeInterval_repeats_block_(0.1, True, observe)


def process_report() -> JsonObject:
    bundle_factory: _BundleFactory | None = getattr(
        import_module("Foundation"), "NSBundle", None
    )
    if bundle_factory is None:
        raise RuntimeError("Foundation.NSBundle is unavailable")
    running: _RunningApplication | None = getattr(
        import_module("AppKit"), "NSRunningApplication", None
    )
    if running is None:
        raise RuntimeError("AppKit.NSRunningApplication is unavailable")
    bundle = bundle_factory.mainBundle()
    extensions = [
        "PIL._imaging",
        "pydantic_core._pydantic_core",
        "grpc._cython.cygrpc",
        "tiktoken._tiktoken",
    ]
    for extension in extensions:
        import_module(extension)
    from free_claude_code.cli.launchers.common import server_command

    return {
        "pid": os.getpid(),
        "process_executable": subprocess.check_output(
            ["/bin/ps", "-p", str(os.getpid()), "-o", "comm="], text=True
        ).strip(),
        "executable": sys.executable,
        "text_encoding": locale.getpreferredencoding(False),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "bundle_path": bundle.bundlePath(),
        "bundle_id": bundle.bundleIdentifier(),
        "application_bundle_id": running.currentApplication().bundleIdentifier(),
        "main_thread": threading.current_thread() is threading.main_thread(),
        "extensions": extensions,
        "server_command": server_command(),
        "macos_version": subprocess.check_output(
            ["/usr/bin/sw_vers", "-productVersion"], text=True
        ).strip(),
    }


def tray_report(icon: Icon) -> JsonObject:
    intersects: Callable[[object, object], bool] | None = getattr(
        import_module("Foundation"), "NSIntersectsRect", None
    )
    if intersects is None:
        raise RuntimeError("Foundation.NSIntersectsRect is unavailable")
    item = icon._status_item
    window = item.button().window()
    screen = window.screen() if window is not None else None
    frame = window.frame() if window is not None else None
    return {
        "visible": bool(item.isVisible()),
        "button_hidden": bool(item.button().isHidden()),
        "screen": bool(screen),
        "frame": [frame.origin.x, frame.origin.y, frame.size.width, frame.size.height]
        if frame is not None
        else None,
        "intersects_screen": (
            frame is not None
            and screen is not None
            and intersects(frame, screen.frame())
        ),
    }


def main() -> None:
    output = Path(os.environ["FCC_MACOS_PROBE_OUTPUT"])
    report = process_report()
    write_report(output, report)

    def quit_probe(item: Icon) -> None:
        item.stop()

    icon = Icon(
        "fcc-launcher-probe",
        Image.new("RGB", (32, 32), "red"),
        "FCC launcher probe",
        Menu(MenuItem("Quit", quit_probe)),
    )
    # Give both launch paths the same AppKit layout observation interval.
    ready_at = time.monotonic() + 2

    def observe(_timer: object) -> None:
        if time.monotonic() < ready_at:
            return
        report.update(tray_report(icon))
        write_report(output, report)
        if output.with_suffix(".captured").exists():
            icon.stop()

    timer = schedule_probe(observe)
    try:
        icon.run()
    finally:
        timer.invalidate()
        report["loop_finished"] = True
        write_report(output, report)


if __name__ == "__main__":
    main()

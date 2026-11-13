"""Observe real AppKit process identity and status-item geometry."""

import json
import locale
import os
import subprocess
import sys
import threading
import time
from importlib import import_module
from pathlib import Path

from PIL import Image
from pystray import Icon, Menu, MenuItem


def write_report(output, report):
    pending = output.with_suffix(".pending")
    pending.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pending.replace(output)


def process_report():
    Foundation = import_module("Foundation")
    AppKit = import_module("AppKit")
    bundle = Foundation.NSBundle.mainBundle()
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
        "bundle_path": str(bundle.bundlePath()),
        "bundle_id": bundle.bundleIdentifier(),
        "application_bundle_id": AppKit.NSRunningApplication.currentApplication().bundleIdentifier(),
        "main_thread": threading.current_thread() is threading.main_thread(),
        "extensions": extensions,
        "server_command": server_command(),
        "macos_version": subprocess.check_output(
            ["/usr/bin/sw_vers", "-productVersion"], text=True
        ).strip(),
    }


def tray_report(icon):
    Foundation = import_module("Foundation")
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
        "intersects_screen": bool(
            frame is not None
            and screen is not None
            and Foundation.NSIntersectsRect(frame, screen.frame())
        ),
    }


def main():
    Foundation = import_module("Foundation")
    output = Path(os.environ["FCC_MACOS_PROBE_OUTPUT"])
    report = process_report()
    write_report(output, report)
    icon = Icon(
        "fcc-launcher-probe",
        Image.new("RGB", (32, 32), "red"),
        "FCC launcher probe",
        Menu(MenuItem("Quit", lambda item: item.stop())),
    )
    # Give both launch paths the same AppKit layout observation interval.
    ready_at = time.monotonic() + 2

    def observe(_timer):
        if time.monotonic() < ready_at:
            return
        report.update(tray_report(icon))
        write_report(output, report)
        if output.with_suffix(".captured").exists():
            icon.stop()

    timer = Foundation.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        0.1, True, observe
    )
    try:
        icon.run()
    finally:
        timer.invalidate()
        report["loop_finished"] = True
        write_report(output, report)


if __name__ == "__main__":
    main()

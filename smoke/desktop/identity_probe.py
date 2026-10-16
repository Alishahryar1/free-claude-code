"""Observe a real pystray loop without starting FCC or accessing user settings."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import AppKit
import Foundation
from PIL import Image
from pystray import Icon, Menu, MenuItem


def main():
    if sys.argv[1:2] == ["--export-icon"]:
        Image.new("RGBA", (512, 512), "red").save(sys.argv[2])
        return
    output = Path(os.environ["FCC_MACOS_PROBE_OUTPUT"])
    bundle = Foundation.NSBundle.mainBundle()
    report = {
        "pid": os.getpid(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "bundle_path": str(bundle.bundlePath()),
        "bundle_id": bundle.bundleIdentifier(),
        "application_bundle_id": AppKit.NSRunningApplication.currentApplication().bundleIdentifier(),
        "main_thread": threading.current_thread() is threading.main_thread(),
    }
    output.write_text(json.dumps(report), encoding="utf-8")
    icon = Icon(
        "fcc-launcher-probe",
        Image.new("RGB", (32, 32), "red"),
        "FCC launcher probe",
        Menu(MenuItem("Quit", lambda item: item.stop())),
    )

    def observe(_timer):
        try:
            item = icon._status_item
            window = item.button().window()
            screen = window.screen() if window is not None else None
            frame = window.frame() if window is not None else None
            report.update(
                visible=bool(item.isVisible()),
                button_hidden=bool(item.button().isHidden()),
                screen=bool(screen),
                frame=(
                    [
                        frame.origin.x,
                        frame.origin.y,
                        frame.size.width,
                        frame.size.height,
                    ]
                    if frame is not None
                    else None
                ),
                intersects_screen=bool(
                    frame is not None
                    and screen is not None
                    and Foundation.NSIntersectsRect(frame, screen.frame())
                ),
            )
            capture = subprocess.run(
                ["/usr/sbin/screencapture", "-x", str(output.with_suffix(".png"))],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            report["screenshot_exit"] = capture.returncode
            report["screenshot_error"] = capture.stderr
        finally:
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            icon.stop()

    timer = Foundation.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
        2, False, observe
    )
    try:
        icon.run()
    finally:
        timer.invalidate()


if __name__ == "__main__":
    main()

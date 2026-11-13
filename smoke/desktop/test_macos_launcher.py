"""Compare the historical shell app with the installed native launcher."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from smoke.desktop.support import capture, launched, wait_exit, wait_report

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS")


@pytest.fixture(scope="module")
def observed_launches(installed_desktop):
    installed = installed_desktop
    probe = Path(__file__).with_name("identity_probe.py")
    output = installed["output"] / "terminal.json"
    output.unlink(missing_ok=True)
    output.with_suffix(".captured").unlink(missing_ok=True)
    env = installed["launch_env"] | {"FCC_MACOS_PROBE_OUTPUT": str(output)}
    with subprocess.Popen([str(installed["python"]), str(probe)], env=env) as process:
        try:
            terminal = wait_report(output, lambda r: "intersects_screen" in r, process)
            capture(output)
            output.with_suffix(".captured").touch()
            assert process.wait(timeout=10) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    # Preserve the pre-fix exec wrapper as the regression control.
    legacy = installed["home"] / "Legacy FCC.app"
    contents = legacy / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Info.plist").write_bytes(
        (installed["app"] / "Contents/Info.plist").read_bytes()
    )
    (contents / "Resources").mkdir()
    (contents / "Resources/AppIcon.icns").write_bytes(
        (installed["app"] / "Contents/Resources/AppIcon.icns").read_bytes()
    )
    executable = contents / "MacOS/fcc-desktop"
    executable.write_text(
        f"#!/bin/sh\nexec {shlex.quote(str(installed['python']))} {shlex.quote(str(probe))}\n"
    )
    executable.chmod(0o755)
    reports = []
    for name, app, extra in (
        ("legacy", legacy, {}),
        ("native", installed["app"], {"FCC_NATIVE_CASE": "identity"}),
    ):
        output = installed["output"] / f"{name}.json"
        with launched(app, output, installed["launch_env"] | extra) as process:
            report = wait_report(output, lambda r: "intersects_screen" in r, process)
            capture(output)
            output.with_suffix(".captured").touch()
            assert process.wait(timeout=10) == 0
            wait_exit(report["pid"])
            print(json.dumps(report, indent=2))
            reports.append(report)
    return terminal, *reports


def test_finder_launch_keeps_fcc_bundle_identity(installed_desktop, observed_launches):
    _terminal, legacy, actual = observed_launches
    assert legacy["bundle_id"] != "io.github.alishahryar1.free-claude-code"
    assert actual["bundle_id"] == "io.github.alishahryar1.free-claude-code"
    assert Path(actual["bundle_path"]).resolve() == installed_desktop["app"].resolve()
    assert (
        Path(actual["process_executable"]).resolve()
        == (installed_desktop["app"] / "Contents/MacOS/fcc-desktop").resolve()
    )
    assert actual["main_thread"]
    assert actual["text_encoding"].lower().replace("-", "") == "utf8"
    assert actual["executable"] == str(installed_desktop["python"])
    assert actual["prefix"] == str(installed_desktop["venv"])
    assert actual["server_command"] == [
        str(installed_desktop["venv"] / "bin/fcc-desktop"),
        "--auto-started",
    ]


def test_finder_status_item_is_placed_on_screen(observed_launches):
    terminal, _legacy, actual = observed_launches
    assert terminal["intersects_screen"], terminal
    assert actual["visible"] and not actual["button_hidden"], actual
    assert actual["intersects_screen"], actual
    assert actual["frame"][2] > 0 and actual["frame"][3] > 0

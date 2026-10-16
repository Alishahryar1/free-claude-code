"""The installed native app keeps FCC's server, menu, and subprocess contracts."""

import json
import plistlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from smoke.desktop.support import capture, launched, wait_report

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS")


def wait_status(port, *, previous=None):
    deadline = time.monotonic() + 30
    with httpx.Client(trust_env=False, timeout=1) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(f"http://127.0.0.1:{port}/admin/api/status")
                response.raise_for_status()
                status = response.json()
                if status["status"] == "running" and status["instance_id"] != previous:
                    return status
            except httpx.HTTPError, ValueError:
                pass
            time.sleep(0.1)
    raise AssertionError("FCC did not serve a new running instance")


def command(output, action):
    pending = output.with_suffix(".next-command")
    pending.write_text(action)
    pending.replace(output.with_suffix(".command"))


def test_installed_native_app_lifecycle(installed_desktop):
    installed = installed_desktop
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = installed["launch_env"] | {"PORT": str(port), "FCC_NATIVE_CASE": "lifecycle"}
    output = installed["output"] / "lifecycle.json"
    with launched(installed["app"], output, env) as process:
        report = wait_report(
            output,
            lambda r: (
                r.get("intersects_screen") and r.get("server_status") == "running"
            ),
            process,
        )
        capture(output)
        initial = wait_status(port)
        # Exercise the real process discovery in both shell scripts without uninstalling.
        for script, body in (
            ("install.sh", "assert_no_fcc_processes_running"),
            ("uninstall.sh", "fcc_process_ids fcc-desktop"),
        ):
            source = (
                (ROOT / "scripts" / script)
                .read_text()
                .split('\nparse_args "$@"\n', 1)[0]
            )
            scenario = installed["root"] / f"guard-{script}"
            scenario.write_text(source + "\n" + body + "\n")
            guard = subprocess.run(
                ["/bin/sh", str(scenario)],
                env=installed["env"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            assert str(report["pid"]) in guard.stdout + guard.stderr
            assert (guard.returncode != 0) == (script == "install.sh")
        duplicate = installed["output"] / "duplicate.json"
        with launched(
            installed["app"], duplicate, env, args=("--auto-started",)
        ) as second:
            assert second.wait(timeout=30) == 0
        duplicate_report = json.loads(duplicate.read_text())
        assert duplicate_report["pid"] != report["pid"]
        assert duplicate_report["loop_finished"]
        assert wait_status(port)["instance_id"] == initial["instance_id"]

        command(output, "subprocess")
        report = wait_report(output, lambda r: "subprocess" in r, process)
        children = report["subprocess"]
        assert children["folder"] == str(installed["home"])
        assert children["child"]["executable"] == str(installed["python"])
        assert children["child"]["prefix"] == str(installed["venv"])
        assert Path(children["child"]["package"]).is_relative_to(installed["venv"])

        command(output, "Restart Server")
        restarted = wait_status(port, previous=initial["instance_id"])
        assert restarted["instance_id"] != initial["instance_id"]
        assert process.poll() is None
        command(output, "Quit")
        assert process.wait(timeout=30) == 0
        report = json.loads(output.read_text())
        assert report["actions"] == ["Restart Server", "Quit"]
        assert report["loop_finished"]
    with socket.socket() as freed:
        freed.bind(("127.0.0.1", port))
    # Locks must be available to a fresh interpreter after Quit.
    subprocess.run(
        [
            str(installed["python"]),
            "-c",
            "from pathlib import Path; from free_claude_code.core.interprocess_lock import InterprocessFileLock; "
            "locks=[InterprocessFileLock(Path.home()/'.fcc'/name) for name in ('desktop.lock','server.owner.lock')]; "
            "assert all(lock.acquire() for lock in locks); [lock.release() for lock in locks]",
        ],
        env=installed["env"],
        check=True,
        timeout=30,
    )

    updated = installed["root"] / "updated venv"
    installed["create_venv"](updated)
    subprocess.run(
        [str(updated / "bin/fcc-desktop"), "--install-macos-app"],
        env=installed["env"],
        check=True,
        timeout=30,
    )
    with (installed["app"] / "Contents/Resources/runtime.plist").open("rb") as stream:
        assert plistlib.load(stream)["PythonExecutable"] == str(updated / "bin/python")
    output = installed["output"] / "updated.json"
    with launched(installed["app"], output, env) as process:
        report = wait_report(
            output,
            lambda r: (
                r.get("intersects_screen") and r.get("server_status") == "running"
            ),
            process,
        )
        assert report["executable"] == str(updated / "bin/python")
        wait_status(port)
        command(output, "Quit")
        assert process.wait(timeout=30) == 0

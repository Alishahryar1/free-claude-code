"""Launch and observe only the disposable applications owned by a native test."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_exit(pid):
    deadline = time.monotonic() + 10
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not pid_alive(pid), f"Test process {pid} did not exit"


def wait_report(output, predicate, process, timeout=30):
    deadline = time.monotonic() + timeout
    report = {}
    while time.monotonic() < deadline:
        if output.exists():
            report = json.loads(output.read_text())
            assert "error" not in report, report.get("error")
            if predicate(report):
                return report
        # Launch Services can stop tracking the old shell wrapper after exec.
        assert process.poll() is None or (
            report.get("pid") and pid_alive(report["pid"])
        ), f"Launcher exited before observation: {report}"
        time.sleep(0.1)
    raise AssertionError(f"Native observation timed out: {report}")


def capture(output):
    result = subprocess.run(
        ["/usr/sbin/screencapture", "-x", str(output.with_suffix(".png"))],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    output.with_suffix(".capture.json").write_text(
        json.dumps(
            {
                "returncode": result.returncode,
                "stderr": result.stderr,
            }
        )
    )


def process_paths():
    rows = subprocess.check_output(["/bin/ps", "-axo", "pid=,comm="], text=True)
    result = {}
    for row in rows.splitlines():
        pid, _, executable = row.strip().partition(" ")
        result[int(pid)] = executable.strip()
    return result


def bundle_pids(app):
    return {
        pid
        for pid, executable in process_paths().items()
        if executable == str(app / "Contents/MacOS/fcc-desktop")
    }


@contextmanager
def launched(app, output, env, *, args=(), wait=True):
    existing = set(process_paths())
    for path in (
        output,
        output.with_suffix(".captured"),
        output.with_suffix(".command"),
    ):
        path.unlink(missing_ok=True)
    command = ["/usr/bin/open", "-n"]
    if wait:
        command.append("-W")
    for key, value in (env | {"FCC_MACOS_PROBE_OUTPUT": str(output)}).items():
        command.extend(["--env", f"{key}={value}"])
    command.extend(["--stderr", str(output.with_suffix(".stderr.log")), str(app)])
    if args:
        command.extend(["--args", *args])
    with subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    ) as process:
        try:
            yield process
        finally:
            if wait or process.poll() is None:
                owned = bundle_pids(app)
                if output.exists():
                    report = json.loads(output.read_text())
                    pid = report.get("pid")
                    executable = process_paths().get(pid)
                    if (
                        executable == report.get("process_executable")
                        and executable
                        and Path(executable).is_relative_to(Path(env["HOME"]).parent)
                    ):
                        owned.add(pid)
                for pid in owned - existing:
                    with suppress(ProcessLookupError):
                        if sys.platform == "darwin":
                            os.kill(pid, signal.SIGKILL)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

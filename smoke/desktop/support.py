"""Launch and observe only the disposable applications owned by a native test."""

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress


def wait_report(output, predicate, process, timeout=30):
    deadline = time.monotonic() + timeout
    report = {}
    while time.monotonic() < deadline:
        if output.exists():
            report = json.loads(output.read_text())
            assert "error" not in report, report.get("error")
            if predicate(report):
                return report
        assert process.poll() is None, f"Launcher exited before observation: {report}"
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


def bundle_pids(app):
    rows = subprocess.check_output(["/bin/ps", "-axo", "pid=,comm="], text=True)
    result = set()
    for row in rows.splitlines():
        pid, _, executable = row.strip().partition(" ")
        if executable.strip() == str(app / "Contents/MacOS/fcc-desktop"):
            result.add(int(pid))
    return result


@contextmanager
def launched(app, output, env, *, args=(), wait=True):
    existing = bundle_pids(app)
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
            if process.poll() is None:
                for pid in bundle_pids(app) - existing:
                    with suppress(ProcessLookupError):
                        if sys.platform == "darwin":
                            os.kill(pid, signal.SIGKILL)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

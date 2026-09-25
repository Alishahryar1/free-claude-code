import io
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from e2e.thread_diagnostics import dump_threads_after


@pytest.mark.parametrize("workers", ["0", "2"])
def test_diagnostic_reaches_stderr_with_pytest_capture(tmp_path, workers):
    plugin = Path(__file__).resolve().parents[1] / "e2e/thread_diagnostics.py"
    (tmp_path / "thread_diagnostics.py").write_text(
        plugin.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(0.2)\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "xdist.plugin",
            "-p",
            "thread_diagnostics",
            "-n",
            workers,
            "-o",
            "thread_dump_timeout=0.03",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTEST_ADDOPTS": "",
            "PYTEST_PLUGINS": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_slow.py::test_slow" in result.stderr
    assert "Thread " in result.stderr


class DiagnosticOutput(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushed = threading.Event()

    def flush(self):
        self.flushed.set()


def test_slow_test_reports_stacks_without_interrupting_execution():
    output = DiagnosticOutput()
    with dump_threads_after(0.001, output, "slow-test"):
        assert output.flushed.wait(5)
        assert "slow-test" in output.getvalue()
        assert "test_slow_test_reports_stacks_without_interrupting_execution" in (
            output.getvalue()
        )
    assert not any(t.name == "fcc-test-diagnostics" for t in threading.enumerate())


@pytest.mark.parametrize("fails", [False, True])
def test_completion_cancels_and_joins_pending_diagnostic(fails):
    output = DiagnosticOutput()
    try:
        with dump_threads_after(60, output, "completed-test"):
            if fails:
                raise RuntimeError("original failure")
    except RuntimeError as error:
        assert fails and str(error) == "original failure"
    assert output.getvalue() == ""
    assert not any(t.name == "fcc-test-diagnostics" for t in threading.enumerate())

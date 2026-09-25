import io
import threading

import pytest

from e2e.thread_diagnostics import dump_threads_after


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

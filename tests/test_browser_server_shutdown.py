import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path
from textwrap import dedent

import pytest

from e2e import server_shutdown
from e2e.code_support import CodeControl


def test_completed_server_needs_no_diagnostics(tmp_path, request, capfd):
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    server_shutdown.join_server(thread, CodeControl(tmp_path), request)
    assert capfd.readouterr().err == ""


def test_slow_server_finishes_after_diagnostic_threshold(
    tmp_path, request, monkeypatch, capfd
):
    control = CodeControl(tmp_path)
    ready = threading.Event()
    gate = None

    async def serve():
        nonlocal gate
        control.loop = asyncio.get_running_loop()
        gate = asyncio.Event()
        job = asyncio.create_task(gate.wait(), name="pending-code-job")
        control.service._jobs.add(job)
        ready.set()
        await job

    snapshot = server_shutdown._task_snapshot

    def release_after_snapshot(control):
        result = snapshot(control)
        assert gate is not None
        gate.set()
        return result

    monkeypatch.setattr(server_shutdown, "_task_snapshot", release_after_snapshot)
    thread = threading.Thread(target=lambda: asyncio.run(serve()))
    thread.start()
    try:
        assert ready.wait(5)
        server_shutdown.join_server(
            thread, control, request, diagnostic_after=0.01, timeout=5
        )
        assert not thread.is_alive()
        output = capfd.readouterr().err
        assert request.node.nodeid in output
        assert "pending-code-job" in output
        assert "Code jobs: 1" in output
        assert "serve" in output
    finally:
        if thread.is_alive():
            assert control.loop is not None and gate is not None
            control.loop.call_soon_threadsafe(gate.set)
        thread.join(5)


@pytest.mark.parametrize("workers", ["0", "1"])
@pytest.mark.parametrize("extra_failure", [False, True])
def test_shutdown_timeout_stops_later_tests(tmp_path, workers, extra_failure):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_shutdown.py").write_text(
        dedent(f"""
        import asyncio
        import threading
        from pathlib import Path
        import pytest
        from e2e.code_support import CodeControl
        from e2e.server_shutdown import join_server

        @pytest.fixture
        def owner(request, tmp_path):
            def fail_too():
                raise ValueError("additional teardown failure")
            if {extra_failure!r}:
                request.addfinalizer(fail_too)
            control = CodeControl(tmp_path)
            release = threading.Event()
            ready = threading.Event()
            async def serve():
                control.loop = asyncio.get_running_loop()
                ready.set()
                # Deliberately stall the loop so diagnostics cannot answer.
                release.wait()
            thread = threading.Thread(target=lambda: asyncio.run(serve()))
            thread.start()
            assert ready.wait(5)
            yield
            try:
                join_server(thread, control, request,
                            diagnostic_after=0.01, timeout=0.05)
            finally:
                release.set()
                thread.join(5)

        def test_a(owner):
            pass

        def test_z():
            Path("later-ran.txt").write_text("ran")
        """),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "xdist.plugin", "-n", workers],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(root), str(root / "src"))),
            "PYTEST_ADDOPTS": "",
            "PYTEST_PLUGINS": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "Admin browser-test server did not stop within" in output
    assert "Server loop did not provide a task snapshot" in output
    if extra_failure:
        assert "additional teardown failure" in output
    assert not (tmp_path / "later-ran.txt").exists(), output

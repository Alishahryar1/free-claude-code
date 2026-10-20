"""Bounded cleanup for the browser fixture's server thread."""

import asyncio
import faulthandler
import inspect
import sys
import threading
import time
from concurrent.futures import Future

import pytest

from e2e.code_support import CodeControl


def _task_snapshot(control: CodeControl) -> str:
    jobs = control.service._jobs
    lines = [f"Code jobs: {sum(not job.done() for job in jobs)}"]
    for task in asyncio.all_tasks():
        lines.append(f"Task {task.get_name()} (Code job: {task in jobs}):")
        current = task.get_coro()
        while inspect.iscoroutine(current) or inspect.isgenerator(current):
            frame = (
                current.cr_frame if inspect.iscoroutine(current) else current.gi_frame
            )
            if frame is not None:
                lines.append(
                    f"  {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"
                )
            current = (
                current.cr_await
                if inspect.iscoroutine(current)
                else current.gi_yieldfrom
            )
    return "\n".join(lines)


def join_server(
    thread: threading.Thread,
    control: CodeControl,
    request: pytest.FixtureRequest,
    *,
    diagnostic_after: float = 5,
    timeout: float = 30,
) -> None:
    # Native interruption can itself take five seconds before durable cleanup.
    # Diagnose at that point, but give the entire fixture shutdown its own budget.
    started = time.monotonic()
    deadline = started + timeout
    thread.join(min(diagnostic_after, timeout))
    if not thread.is_alive():
        return

    print(
        f"Server shutdown still pending after {time.monotonic() - started:.2f}s: "
        f"{request.node.nodeid}",
        file=sys.stderr,
    )
    faulthandler.dump_traceback(file=sys.stderr)
    snapshot: Future[str] = Future()

    def capture() -> None:
        try:
            snapshot.set_result(_task_snapshot(control))
        except Exception as error:
            snapshot.set_result(f"Server task snapshot failed: {type(error).__name__}")

    if control.loop is None:
        snapshot.set_result("Server loop unavailable for task snapshot")
    else:
        try:
            control.loop.call_soon_threadsafe(capture)
        except RuntimeError:
            snapshot.set_result("Server loop closed before task snapshot")

    thread.join(max(0, deadline - time.monotonic()))
    print(
        snapshot.result()
        if snapshot.done()
        else "Server loop did not provide a task snapshot",
        file=sys.stderr,
    )
    if thread.is_alive():
        reason = (
            f"Admin browser-test server did not stop within {timeout:g}s: "
            f"{request.node.nodeid}"
        )
        request.session.shouldstop = reason
        faulthandler.dump_traceback(file=sys.stderr)
        pytest.fail(reason)

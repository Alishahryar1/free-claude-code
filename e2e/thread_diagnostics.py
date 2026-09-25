"""Slow-test stack snapshots without the native faulthandler watchdog."""

import os
import sys
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

import pytest

_output = pytest.StashKey[TextIO]()


@contextmanager
def dump_threads_after(
    timeout: float, output: TextIO, test_name: str
) -> Iterator[None]:
    def dump() -> None:
        print(f"Test still running after {timeout:g}s: {test_name}", file=output)
        for thread_id, frame in sys._current_frames().items():
            print(f"\nThread {thread_id}:", file=output)
            traceback.print_stack(frame, file=output)
        output.flush()

    timer = threading.Timer(timeout, dump)
    timer.name = "fcc-test-diagnostics"
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join()


def pytest_addoption(parser) -> None:
    parser.addini("thread_dump_timeout", "Print slow-test thread stacks", default="0")


def pytest_configure(config: pytest.Config) -> None:
    if float(config.getini("thread_dump_timeout")) > 0:
        # Windows xdist workers preserve stderr here and redirect descriptor 2 to NUL.
        # Keep a stable output descriptor across pytest's per-test capture changes.
        config.stash[_output] = os.fdopen(
            os.dup(sys.stderr.fileno()),
            "w",
            encoding="utf-8",
            errors="backslashreplace",
        )


def pytest_unconfigure(config: pytest.Config) -> None:
    if _output in config.stash:
        config.stash[_output].close()
        del config.stash[_output]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item):
    timeout = float(item.config.getini("thread_dump_timeout"))
    if timeout <= 0:
        return (yield)
    with dump_threads_after(timeout, item.config.stash[_output], item.nodeid):
        return (yield)

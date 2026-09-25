"""Temporary Windows CI observations. Remove after diagnosing Pi's stall."""

import ctypes
import json
import os
import subprocess
from ctypes import wintypes
from itertools import count
from pathlib import Path
from time import monotonic, time

_sequence = count()
_compared = False


def _cpu_seconds(pid: int) -> dict:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (
        ctypes.POINTER(wintypes.FILETIME),
    ) * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.GetSystemTimes.argtypes = (ctypes.POINTER(wintypes.FILETIME),) * 3
    kernel.GetSystemTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return {"cpu_error": ctypes.get_last_error()}
    try:
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(system),
            ctypes.byref(user),
        ):
            return {"cpu_error": ctypes.get_last_error()}
        sample = {
            "system_cpu_seconds": ((system.dwHighDateTime << 32) | system.dwLowDateTime)
            / 10_000_000,
            "user_cpu_seconds": ((user.dwHighDateTime << 32) | user.dwLowDateTime)
            / 10_000_000,
        }
        idle, host_system, host_user = (wintypes.FILETIME() for _ in range(3))
        if kernel.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(host_system), ctypes.byref(host_user)
        ):
            for name, stamp in (
                ("idle", idle),
                ("system", host_system),
                ("user", host_user),
            ):
                sample[f"host_{name}_seconds"] = (
                    (stamp.dwHighDateTime << 32) | stamp.dwLowDateTime
                ) / 10_000_000
        return sample
    finally:
        kernel.CloseHandle(handle)


def _observe(command: list[str], phase: str, label: str):
    directory = Path("test-results/node-diagnostics")
    directory.mkdir(parents=True, exist_ok=True)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "serial")
    path = directory / f"{phase}-{worker}-{next(_sequence)}-{label}.jsonl"
    started = monotonic()
    with path.open("w", encoding="utf-8", buffering=1) as log:

        def record(event: str, **fields) -> None:
            log.write(
                json.dumps({"event": event, "elapsed": monotonic() - started, **fields})
                + "\n"
            )
            log.flush()

        record(
            "launch",
            at=time(),
            executable=command[0],
            cwd=str(Path.cwd()),
            test=os.environ.get("PYTEST_CURRENT_TEST"),
            cpu_count=os.cpu_count(),
        )
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
        ) as process:
            record("created", pid=process.pid)
            communicating = monotonic()
            deadline = communicating + 60
            timed_out = False
            try:
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        record("sample", **_cpu_seconds(process.pid))
                        if monotonic() >= deadline:
                            timed_out = True
                            process.kill()
                            stdout, stderr = process.communicate()
                            break
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            elapsed = monotonic() - started
            waited = monotonic() - communicating
            record(
                "finished",
                returncode=process.returncode,
                observation_timeout=timed_out,
                communicate_seconds=waited,
                exceeded_original_timeout=waited > 10,
                stdout=stdout,
                stderr=stderr,
            )
    print(f"Node diagnostic {phase}/{label}: {elapsed:.3f}s, log={path}", flush=True)
    return subprocess.CompletedProcess(
        command, process.returncode, stdout, stderr
    ), waited


def run_node_diagnostics(command: list[str], phase: str):
    global _compared
    result, elapsed = _observe(command, phase, "original")
    if not _compared:
        _compared = True
        # Controls deliberately follow the first original command, preserving its
        # cold position in the existing suite. Their order is visible in filenames.
        node = command[0]
        _observe(
            [node, "--eval", 'console.error("commonjs-entered", process.version);'],
            phase,
            "commonjs",
        )
        _observe(
            [
                node,
                "--experimental-strip-types",
                "--input-type=module",
                "--eval",
                'import { writeSync } from "node:fs"; '
                'writeSync(2, "esm-entered " + process.version);',
            ],
            phase,
            "esm",
        )
        _observe(command, phase, "original-repeat")
    if elapsed > 10:
        raise subprocess.TimeoutExpired(
            command, 10, output=result.stdout, stderr=result.stderr
        )
    return result

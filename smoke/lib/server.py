"""Subprocess lifecycle helpers for local smoke servers."""

import os
import socket
import subprocess
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import httpx

from .child_process import cmd_fcc_server
from .config import SmokeConfig, redacted


@dataclass(slots=True)
class RunningServer:
    base_url: str
    port: int
    log_path: Path
    process: subprocess.Popen[bytes]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def start_server(
    config: SmokeConfig,
    *,
    env_overrides: dict[str, str] | None = None,
    env_unset: Iterable[str] = (),
    command: list[str] | None = None,
    name: str = "server",
) -> Iterator[RunningServer]:
    port = find_free_port()
    config.results_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.results_dir / f"{name}-{config.worker_id}-{port}.log"

    env = os.environ.copy()
    for key in env_unset:
        env.pop(key, None)
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "LOG_FILE": str(log_path),
            "FCC_OPEN_BROWSER": "0",
            "MESSAGING_PLATFORM": "none",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if env_overrides:
        env.update(env_overrides)

    cmd = command or cmd_fcc_server()

    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=config.root,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        running = RunningServer(
            base_url=f"http://127.0.0.1:{port}",
            port=port,
            log_path=log_path,
            process=process,
        )
        try:
            _wait_for_health(running, timeout_s=config.timeout_s)
            yield running
        finally:
            _stop_process(process)


def _wait_for_health(server: RunningServer, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            break
        try:
            response = httpx.get(f"{server.base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)

    log_excerpt = ""
    with suppress(OSError):
        log_excerpt = server.log_path.read_text(encoding="utf-8", errors="replace")[
            -2000:
        ]
    raise AssertionError(
        "Smoke server did not become healthy. "
        f"last_error={last_error!r} log={redacted(log_excerpt)!r}"
    )


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

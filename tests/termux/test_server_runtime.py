import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = "from free_claude_code.cli.entrypoints import serve; serve()"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _server_environment(home: Path, port: int) -> dict[str, str]:
    environment = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(ROOT / "src"),
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "OPEN_ADMIN_BROWSER": "false",
    }
    for name in (
        "PREFIX",
        "TERMUX_VERSION",
        "LD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "TMPDIR",
        "LANG",
    ):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _start_server(home: Path, port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", SERVER_SCRIPT],
        cwd=ROOT,
        env=_server_environment(home, port),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_health(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 30
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output, _ = process.communicate()
            raise AssertionError(f"FCC server exited before becoming ready:\n{output}")
        try:
            response = httpx.get(url, timeout=0.5, trust_env=False)
        except httpx.HTTPError:
            time.sleep(0.1)
            continue
        assert response.status_code == 200
        return
    raise AssertionError("FCC server did not become ready on localhost.")


def _stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        output, _ = process.communicate(timeout=15)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        output, _ = process.communicate(timeout=5)
        raise AssertionError(
            f"FCC server did not shut down gracefully:\n{output}"
        ) from exc
    assert process.returncode == -signal.SIGTERM, output
    assert "Application shutdown complete." in output


@pytest.mark.skipif(os.name == "nt", reason="Termux runs on POSIX Android.")
def test_server_localhost_http_shutdown_and_config_persistence(tmp_path):
    port = _free_port()
    home = tmp_path / "home"
    home.mkdir()
    config_file = home / ".fcc" / ".env"

    process = _start_server(home, port)
    try:
        _wait_for_health(process, port)
    finally:
        _stop_server(process)

    assert config_file.is_file()
    persisted_config = config_file.read_bytes()

    process = _start_server(home, port)
    try:
        _wait_for_health(process, port)
    finally:
        _stop_server(process)

    assert config_file.read_bytes() == persisted_config

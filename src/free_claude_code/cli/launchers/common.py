"""Shared process helpers for installed client CLI launchers."""

import errno
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from free_claude_code.cli.local_http import open_local_request
from free_claude_code.cli.process_registry import (
    kill_pid_tree_best_effort,
    register_pid,
    unregister_pid,
)
from free_claude_code.config import paths
from free_claude_code.config.paths import (
    server_startup_lock_path,
    server_startup_log_path,
)
from free_claude_code.core.interprocess_lock import InterprocessFileLock

PROXY_PREFLIGHT_PATH = "/health"
PROXY_PREFLIGHT_TIMEOUT_SECONDS = 1.5
PROXY_PREFLIGHT_BUDGET_SECONDS = 30.0

SERVER_STARTUP_TIMEOUT_SECONDS = 30.0
SERVER_STARTUP_LOCK_TIMEOUT_SECONDS = 30.0
SERVER_STARTUP_POLL_SECONDS = 0.25


class ProxyState(StrEnum):
    """What one health probe learned about the local proxy."""

    HEALTHY = "healthy"
    # An HTTP server answered, so a proxy exists even if it is unhealthy.
    HTTP_ERROR = "http_error"
    # A listener accepted TCP but never answered: the app is still loading.
    STARTING = "starting"
    # Nothing accepted the connection, so no server is running.
    UNREACHABLE = "unreachable"
    NETWORK_ERROR = "network_error"


@dataclass(frozen=True, slots=True)
class ProxyProbe:
    state: ProxyState
    error: str | None = None


def _probe_proxy(proxy_root_url: str, *, deadline: float) -> ProxyProbe:
    url = f"{proxy_root_url.rstrip('/')}{PROXY_PREFLIGHT_PATH}"
    request = Request(url, method="GET")
    while True:
        try:
            with open_local_request(
                request,
                timeout=min(
                    PROXY_PREFLIGHT_TIMEOUT_SECONDS,
                    max(0.001, deadline - time.monotonic()),
                ),
            ) as response:
                status_code = response.status
        except HTTPError as exc:
            return ProxyProbe(ProxyState.HTTP_ERROR, f"returned HTTP {exc.code}")
        except (URLError, OSError) as exc:
            reason = exc.reason if isinstance(exc, URLError) else exc
            # A reserved listener can accept TCP while the HTTP app is loading.
            # Refusals and other failures still return the normal launch hint.
            if isinstance(reason, TimeoutError):
                address = urlsplit(url)
                try:
                    with socket.create_connection(
                        (address.hostname or "127.0.0.1", address.port or 80),
                        timeout=PROXY_PREFLIGHT_TIMEOUT_SECONDS,
                    ):
                        pass
                except ConnectionRefusedError, TimeoutError:
                    return ProxyProbe(ProxyState.UNREACHABLE, str(reason))
                except OSError as exc:
                    return ProxyProbe(ProxyState.NETWORK_ERROR, str(exc))
                if time.monotonic() < deadline:
                    continue
                return ProxyProbe(ProxyState.STARTING, str(reason))
            if isinstance(reason, ConnectionRefusedError) or getattr(
                reason, "errno", None
            ) in {errno.ECONNREFUSED, 10061}:
                return ProxyProbe(ProxyState.UNREACHABLE, str(reason))
            return ProxyProbe(ProxyState.NETWORK_ERROR, str(reason))
        if not 200 <= status_code < 300:
            return ProxyProbe(ProxyState.HTTP_ERROR, f"returned HTTP {status_code}")
        return ProxyProbe(ProxyState.HEALTHY)


def preflight_proxy(proxy_root_url: str) -> str | None:
    """Return an error message when the local proxy health check is unreachable."""

    return _probe_proxy(
        proxy_root_url, deadline=time.monotonic() + PROXY_PREFLIGHT_BUDGET_SECONDS
    ).error


def server_command() -> list[str]:
    """Run the installed platform owner with its normal process identity."""

    name = "fcc-desktop" if sys.platform in {"win32", "darwin"} else "fcc-server"
    suffix = ".exe" if sys.platform == "win32" else ""
    sibling = Path(sys.executable).parent / f"{name}{suffix}"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return [str(sibling)]
    raise FileNotFoundError(
        f"Installed {name} command is missing or not executable: {sibling}"
    )


def _spawn_server(env: Mapping[str, str]) -> subprocess.Popen[bytes]:
    """Start the platform owner independently and preserve early diagnostics."""

    command = server_command()
    child_env = dict(env)
    if sys.platform in {"win32", "darwin"}:
        child_env["FCC_DESKTOP_STARTED_BY_LAUNCHER"] = "1"
    log_path = server_startup_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as output:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=child_env,
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
                if sys.platform == "win32"
                else 0
            ),
            start_new_session=sys.platform != "win32",
        )


def _desktop_owner_running() -> bool:
    """A manually launched Desktop may own startup outside our launcher lock."""

    if sys.platform not in {"win32", "darwin"}:
        return False
    lock = InterprocessFileLock(paths.config_dir_path() / "desktop.lock")
    try:
        acquired = lock.acquire()
    except OSError:
        return False
    if acquired:
        lock.release()
    return not acquired


def _desktop_port_owner_running(proxy_root_url: str) -> bool:
    """Check whether Desktop owns the requested port, including startup."""

    if sys.platform not in {"win32", "darwin"}:
        return False
    port = urlsplit(proxy_root_url).port
    if port is None:
        return False
    lock = InterprocessFileLock(paths.desktop_port_lock_path(port))
    try:
        acquired = lock.acquire()
    except OSError:
        return False
    if acquired:
        lock.release()
    return not acquired


def _stop_owned_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=2)
        except OSError, subprocess.TimeoutExpired:
            pass
    except OSError:
        pass


def _unreachable_message(proxy_root_url: str, probe: ProxyProbe, detail: str) -> str:
    name = "fcc-desktop" if sys.platform in {"win32", "darwin"} else "fcc-server"
    return (
        f"Free Claude Code proxy is not reachable at {proxy_root_url}: {probe.error}\n"
        f"{detail}\nStartup log: {server_startup_log_path()}\n"
        f"Start it manually with: {name}"
    )


def _wait_for_server(
    proxy_root_url: str,
    process: subprocess.Popen[bytes] | None,
    *,
    deadline: float,
) -> ProxyProbe | str:
    """Poll until the spawned server answers, exits, or the budget runs out."""

    while True:
        probe = _probe_proxy(proxy_root_url, deadline=deadline)
        if probe.state in {
            ProxyState.HEALTHY,
            ProxyState.HTTP_ERROR,
            ProxyState.NETWORK_ERROR,
        }:
            return probe
        exit_code = process.poll() if process is not None else None
        port_owner_running = _desktop_port_owner_running(proxy_root_url)
        if exit_code is not None and port_owner_running:
            process = None
        elif exit_code is not None:
            if _desktop_owner_running():
                return "Existing Desktop is running on a different port."
            name = (
                "fcc-desktop" if sys.platform in {"win32", "darwin"} else "fcc-server"
            )
            return f"{name} exited with code {exit_code} before it was ready."
        if process is None and not port_owner_running:
            return "Existing Desktop stopped before it was ready."
        if time.monotonic() >= deadline:
            return (
                "Free Claude Code did not become ready within "
                f"{SERVER_STARTUP_TIMEOUT_SECONDS:g} seconds."
            )
        time.sleep(SERVER_STARTUP_POLL_SECONDS)


def ensure_proxy_available(
    proxy_root_url: str, *, env: Mapping[str, str]
) -> str | None:
    """Return an error message when the proxy is down and cannot be started here.

    A missing server is started once, under an interprocess lock, so concurrent
    launchers share one platform owner instead of racing to bind the same port.
    """

    probe = _probe_proxy(
        proxy_root_url, deadline=time.monotonic() + PROXY_PREFLIGHT_BUDGET_SECONDS
    )
    if probe.state is ProxyState.HEALTHY:
        return None
    if probe.state is not ProxyState.UNREACHABLE:
        return _unreachable_message(
            proxy_root_url, probe, "A server is answering on that address."
        )
    lock = InterprocessFileLock(server_startup_lock_path())
    try:
        acquired = lock.acquire(wait=True, timeout=SERVER_STARTUP_LOCK_TIMEOUT_SECONDS)
    except OSError as exc:
        return _unreachable_message(
            proxy_root_url, probe, f"Could not acquire the startup lock: {exc}"
        )
    try:
        # The startup budget starts now: time spent waiting for another
        # launcher's attempt must not shorten this launcher's own attempt.
        deadline = time.monotonic() + SERVER_STARTUP_TIMEOUT_SECONDS
        # Another launcher may have started the server while this one waited.
        probe = _probe_proxy(proxy_root_url, deadline=deadline)
        if probe.state is ProxyState.HEALTHY:
            return None
        if probe.state is not ProxyState.UNREACHABLE:
            return _unreachable_message(
                proxy_root_url, probe, "A server is answering on that address."
            )
        if not acquired:
            return _unreachable_message(
                proxy_root_url,
                probe,
                "Another launcher is still starting the server; retry shortly.",
            )
        if _desktop_port_owner_running(proxy_root_url):
            outcome = _wait_for_server(proxy_root_url, None, deadline=deadline)
            if isinstance(outcome, ProxyProbe) and outcome.state is ProxyState.HEALTHY:
                return None
            return _unreachable_message(
                proxy_root_url,
                probe,
                outcome
                if isinstance(outcome, str)
                else "Existing Desktop is not ready.",
            )
        if _desktop_owner_running():
            return _unreachable_message(
                proxy_root_url,
                probe,
                "Existing Desktop is running on a different port. "
                "Stop it or use its configured PORT.",
            )
        try:
            process = _spawn_server(env)
        except OSError as exc:
            return _unreachable_message(
                proxy_root_url, probe, f"Could not start FCC: {exc}"
            )
        print(
            f"Starting Free Claude Code at {proxy_root_url} "
            "(it keeps running after this client exits).",
            file=sys.stderr,
        )
        try:
            outcome = _wait_for_server(proxy_root_url, process, deadline=deadline)
        except BaseException:
            _stop_owned_process(process)
            raise
        if isinstance(outcome, str):
            _stop_owned_process(process)
            return _unreachable_message(proxy_root_url, probe, outcome)
        if outcome.state is ProxyState.HEALTHY:
            return None
        _stop_owned_process(process)
        return _unreachable_message(
            proxy_root_url, outcome, "FCC started but is unhealthy."
        )
    finally:
        lock.release()


def resolve_client_binary(
    *,
    binary_name: str,
    display_name: str,
    install_hint: str,
) -> str:
    """Resolve an installed client binary or exit with a user-facing hint."""

    client_command = shutil.which(binary_name)
    if client_command is None:
        print(
            f"Could not find {display_name} command: {binary_name}",
            file=sys.stderr,
        )
        print(install_hint, file=sys.stderr)
        raise SystemExit(127)
    return client_command


def run_client_process(
    *,
    command: list[str],
    env: Mapping[str, str],
    binary_name: str,
    display_name: str,
    install_hint: str,
) -> None:
    """Run a client CLI command and mirror its exit code."""

    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(command, env=dict(env))
        if process.pid:
            register_pid(process.pid)
        return_code = process.wait()
    except FileNotFoundError:
        print(
            f"Could not find {display_name} command: {binary_name}",
            file=sys.stderr,
        )
        print(install_hint, file=sys.stderr)
        raise SystemExit(127) from None
    except KeyboardInterrupt:
        if process is not None and process.pid:
            kill_pid_tree_best_effort(process.pid)
            process.wait()
        raise
    finally:
        if process is not None and process.pid:
            unregister_pid(process.pid)

    raise SystemExit(return_code)

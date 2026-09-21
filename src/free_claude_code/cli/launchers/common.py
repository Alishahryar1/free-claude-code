"""Shared process helpers for installed client CLI launchers."""

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

from free_claude_code.cli.local_http import open_local_request
from free_claude_code.cli.process_registry import (
    kill_pid_tree_best_effort,
    register_pid,
    unregister_pid,
)
from free_claude_code.config.paths import server_startup_lock_path
from free_claude_code.core.interprocess_lock import InterprocessFileLock

PROXY_PREFLIGHT_PATH = "/health"
PROXY_PREFLIGHT_TIMEOUT_SECONDS = 1.5
PROXY_PREFLIGHT_BUDGET_SECONDS = 30.0

AUTO_START_ENV = "FCC_AUTO_START_SERVER"
SERVER_STARTUP_TIMEOUT_SECONDS = 30.0
SERVER_STARTUP_LOCK_TIMEOUT_SECONDS = 30.0
SERVER_STARTUP_POLL_SECONDS = 0.25
SERVER_COMMAND_NAME = "fcc-server"
MANUAL_START_HINT = f"Start it in another terminal with: {SERVER_COMMAND_NAME}"

_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})


class ProxyState(StrEnum):
    """What one health probe learned about the local proxy."""

    HEALTHY = "healthy"
    # An HTTP server answered, so a proxy exists even if it is unhealthy.
    HTTP_ERROR = "http_error"
    # A listener accepted TCP but never answered: the app is still loading.
    STARTING = "starting"
    # Nothing accepted the connection, so no server is running.
    UNREACHABLE = "unreachable"


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
                if time.monotonic() < deadline:
                    continue
                return ProxyProbe(ProxyState.STARTING, str(reason))
            return ProxyProbe(ProxyState.UNREACHABLE, str(reason))
        if not 200 <= status_code < 300:
            return ProxyProbe(ProxyState.HTTP_ERROR, f"returned HTTP {status_code}")
        return ProxyProbe(ProxyState.HEALTHY)


def preflight_proxy(proxy_root_url: str) -> str | None:
    """Return an error message when the local proxy health check is unreachable."""

    return _probe_proxy(
        proxy_root_url, deadline=time.monotonic() + PROXY_PREFLIGHT_BUDGET_SECONDS
    ).error


def auto_start_enabled(env: Mapping[str, str]) -> bool:
    """Launchers start a missing server unless the environment opts out."""

    return env.get(AUTO_START_ENV, "").strip().lower() not in _DISABLED_VALUES


def server_command() -> list[str]:
    """Prefer the installed fcc-server beside this interpreter, then the package."""

    suffix = ".exe" if sys.platform == "win32" else ""
    sibling = Path(sys.executable).parent / f"{SERVER_COMMAND_NAME}{suffix}"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return [str(sibling)]
    return [
        sys.executable,
        "-c",
        "from free_claude_code.cli.entrypoints import serve; serve()",
    ]


def _spawn_server(env: Mapping[str, str]) -> subprocess.Popen[bytes]:
    """Start fcc-server detached so it outlives this launcher and its client."""

    if sys.platform == "win32":
        return subprocess.Popen(
            server_command(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=dict(env),
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0),
        )
    return subprocess.Popen(
        server_command(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=dict(env),
        start_new_session=True,
    )


def _unreachable_message(proxy_root_url: str, probe: ProxyProbe, detail: str) -> str:
    return (
        f"Free Claude Code proxy is not reachable at {proxy_root_url}: {probe.error}\n"
        f"{detail}\n{MANUAL_START_HINT}"
    )


def _wait_for_server(
    proxy_root_url: str,
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
) -> ProxyProbe | str:
    """Poll until the spawned server answers, exits, or the budget runs out."""

    while True:
        probe = _probe_proxy(proxy_root_url, deadline=deadline)
        if probe.state in {ProxyState.HEALTHY, ProxyState.HTTP_ERROR}:
            return probe
        exit_code = process.poll()
        if exit_code is not None:
            return (
                f"{SERVER_COMMAND_NAME} exited with code {exit_code} before it was "
                f"ready. Run {SERVER_COMMAND_NAME} in a terminal to see its output."
            )
        if time.monotonic() >= deadline:
            return (
                f"{SERVER_COMMAND_NAME} did not become ready within "
                f"{SERVER_STARTUP_TIMEOUT_SECONDS:g} seconds. It may still be "
                "starting; retry shortly."
            )
        time.sleep(SERVER_STARTUP_POLL_SECONDS)


def ensure_proxy_available(
    proxy_root_url: str, *, env: Mapping[str, str]
) -> str | None:
    """Return an error message when the proxy is down and cannot be started here.

    A missing server is started once, under an interprocess lock, so concurrent
    launchers share one fcc-server instead of racing to bind the same port.
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
    if not auto_start_enabled(env):
        return _unreachable_message(
            proxy_root_url, probe, f"Automatic start is disabled by {AUTO_START_ENV}."
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
        try:
            process = _spawn_server(env)
        except OSError as exc:
            return _unreachable_message(
                proxy_root_url, probe, f"Could not start {SERVER_COMMAND_NAME}: {exc}"
            )
        print(
            f"Starting Free Claude Code server at {proxy_root_url} "
            "(it keeps running after this client exits).",
            file=sys.stderr,
        )
        outcome = _wait_for_server(proxy_root_url, process, deadline=deadline)
        if isinstance(outcome, str):
            return _unreachable_message(proxy_root_url, probe, outcome)
        if outcome.state is ProxyState.HEALTHY:
            return None
        return _unreachable_message(
            proxy_root_url, outcome, f"{SERVER_COMMAND_NAME} started but is unhealthy."
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

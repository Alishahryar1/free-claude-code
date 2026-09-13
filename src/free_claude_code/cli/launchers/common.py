"""Shared process helpers for installed client CLI launchers."""

import shutil
import subprocess
import sys
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request

from free_claude_code.cli.local_http import open_local_request
from free_claude_code.cli.process_registry import (
    kill_pid_tree_best_effort,
    register_pid,
    unregister_pid,
)

PROXY_PREFLIGHT_PATH = "/health"
PROXY_PREFLIGHT_TIMEOUT_SECONDS = 1.5


def preflight_proxy(proxy_root_url: str) -> str | None:
    """Return an error message when the local proxy health check is unreachable."""

    url = f"{proxy_root_url.rstrip('/')}{PROXY_PREFLIGHT_PATH}"
    request = Request(url, method="GET")
    try:
        with open_local_request(
            request, timeout=PROXY_PREFLIGHT_TIMEOUT_SECONDS
        ) as response:
            status_code = response.status
    except HTTPError as exc:
        return f"returned HTTP {exc.code}"
    except URLError as exc:
        return str(exc.reason)
    except OSError as exc:
        return str(exc)

    if not 200 <= status_code < 300:
        return f"returned HTTP {status_code}"
    return None


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

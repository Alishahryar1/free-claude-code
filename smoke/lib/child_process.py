"""Child-process commands for smoke (avoid nested ``uv run`` on Windows).

Nested ``uv run`` can try to refresh console scripts while they are locked
(``fcc-server.exe`` in use), causing flaky smoke. The smoke runner is
already executed under the project environment (``uv run pytest``), so children
should use the same interpreter.
"""

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


def python_exe() -> str:
    return sys.executable


def cmd_python_c(script: str) -> list[str]:
    return [python_exe(), "-c", script]


def cmd_fcc_version() -> list[str]:
    return [
        python_exe(),
        "-c",
        (
            "import sys; "
            "sys.argv = ['fcc-server', '--version']; "
            "from free_claude_code.cli.entrypoints import serve; serve()"
        ),
    ]


def cmd_fcc_server() -> list[str]:
    return [
        python_exe(),
        "-c",
        "from free_claude_code.cli.entrypoints import serve; serve()",
    ]


def run_captured_text(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a smoke child process with deterministic captured text decoding."""
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )

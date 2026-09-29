"""Server logging contracts run outside pytest's process-wide logging state."""

import asyncio
import json
import logging
import logging.config
import subprocess
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import uvicorn
from loguru import logger

from free_claude_code.cli.commands import ServerSupervisor
from free_claude_code.cli.uvicorn_server import RuntimeServer
from free_claude_code.config.logging_config import configure_logging
from free_claude_code.config.settings import Settings
from free_claude_code.runtime.application import ApplicationRuntime


def _exercise_server_logging(
    path: Path, console: bool, level: str, without_streams: bool
) -> None:
    baseline = deepcopy(uvicorn.config.LOGGING_CONFIG)
    # An earlier server may already have installed Uvicorn's console handlers.
    logging.config.dictConfig(baseline)
    settings = Settings(host="127.0.0.1", port=8082, log_level=level)
    instances = []

    def build_app(settings, restart_callback):
        instances.append(f"instance-{len(instances) + 1}")
        configure_logging(path, level=settings.log_level)
        runtime = MagicMock(
            spec=ApplicationRuntime,
            instance_id=instances[-1],
            settings=settings,
            _draining=False,
            _http_ready=asyncio.Event(),
            is_closed=True,
            begin_shutdown=lambda: None,
            close=AsyncMock(return_value=True),
        )
        runtime.http_started.side_effect = lambda: ApplicationRuntime.http_started(
            runtime
        )
        return SimpleNamespace(runtime=runtime)

    def run(server, sockets):
        server._on_started()
        logging.getLogger("uvicorn.access").info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:1234",
            "POST",
            "/test-request",
            "1.1",
            200,
        )
        try:
            raise ValueError("controlled server failure")
        except ValueError:
            logging.getLogger("uvicorn.error").exception("ASGI request failed")
        logging.getLogger("test.dependency").warning("dependency warning")

    with (
        patch("free_claude_code.runtime.bootstrap.build_asgi_app", build_app),
        patch.object(RuntimeServer, "run", run),
        patch.multiple(sys, stdout=None, stderr=None)
        if without_streams
        else nullcontext(),
    ):
        supervisor = ServerSupervisor(console_logging=console)
        for _ in range(2):
            supervisor._run_bound(
                settings, [], open_admin_browser=False, restart_generation=0
            )
    logger.warning("outside server run")
    logger.complete()
    assert baseline == uvicorn.config.LOGGING_CONFIG


@pytest.mark.parametrize(
    ("console", "without_streams"),
    [(True, False), (False, False), (False, True)],
    ids=["cli", "desktop", "desktop-no-console"],
)
@pytest.mark.parametrize("level", ["INFO", "WARNING"])
def test_server_logs_reach_file_once_per_start(
    tmp_path, console, level, without_streams
):
    path = tmp_path / "server.log"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from tests.cli.test_server_logging import _exercise_server_logging; "
            "_exercise_server_logging(Path(sys.argv[1]), sys.argv[2] == 'True', "
            "sys.argv[3], sys.argv[4] == 'True')",
            str(path),
            str(console),
            level,
            str(without_streams),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    messages = [row["record"]["message"] for row in rows]
    info_count = 2 if level == "INFO" else 0
    assert sum(message.startswith("Admin UI:") for message in messages) == info_count
    assert sum("/test-request" in message for message in messages) == info_count
    assert messages.count("ASGI request failed") == 2
    assert messages.count("dependency warning") == 2
    failures = [
        row for row in rows if row["record"]["message"] == "ASGI request failed"
    ]
    for row in failures:
        assert row["record"]["exception"]["type"] == "ValueError"
        assert "controlled server failure" in row["text"]
        assert "Traceback" in row["text"]
    server_records = [
        row["record"]
        for row in rows
        if row["record"]["message"] != "outside server run"
    ]
    assert {record["extra"]["instance_id"] for record in server_records} == {
        "instance-1",
        "instance-2",
    }
    assert len({record["process"]["id"] for record in server_records}) == 1
    assert "instance_id" not in rows[-1]["record"]["extra"]
    assert result.stdout.count("/test-request") == (2 if console else 0)
    assert result.stderr.count("Admin UI:") == (2 if console else 0)
    assert result.stderr.count("ASGI request failed") == (2 if console else 0)

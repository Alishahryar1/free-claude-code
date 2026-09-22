"""Automatic server launches retain runtime logs across config restarts."""

import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("automatic", [False, True])
def test_server_entrypoint_routes_logs_across_restarts(tmp_path, automatic):
    log_path = tmp_path / "server.log"
    script = textwrap.dedent("""
        import logging
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from loguru import logger
        from free_claude_code.cli import commands, entrypoints, uvicorn_server
        from free_claude_code.config.logging_config import configure_logging
        from free_claude_code.config.settings import Settings
        from free_claude_code.runtime import bootstrap

        settings = Settings.model_construct(
            host="127.0.0.1", port=0, open_admin_browser=False,
        )
        generations = []
        async def closed():
            return True

        def build(settings, restart_callback):
            configure_logging(sys.argv[1])
            generations.append(restart_callback)
            return SimpleNamespace(runtime=SimpleNamespace(
                is_closed=True, begin_shutdown=lambda: None, close=closed,
                http_started=lambda: None,
            ))

        def run(self, **kwargs):
            generation = len(generations)
            logging.getLogger("uvicorn.access").info(
                '%s - "%s %s HTTP/%s" %d', "client", "GET",
                f"/generation-{generation}", "1.1", 200,
            )
            logging.getLogger("uvicorn.error").error(f"error-generation-{generation}")
            logger.info(f"application-generation-{generation}")
            if generation == 1:
                generations[-1]()

        with (
            patch.object(commands, "load_server_settings", return_value=settings),
            patch.object(commands, "kill_all_best_effort"),
            patch.object(bootstrap, "build_asgi_app", side_effect=build),
            patch.object(uvicorn_server.RuntimeServer, "run", run),
        ):
            entrypoints.serve(["--auto-started"] if sys.argv[2] == "True" else [])
        logger.complete()
        assert len(generations) == 2
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(log_path), str(automatic)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    saved = log_path.read_text(encoding="utf-8")
    for generation in (1, 2):
        assert f"application-generation-{generation}" in saved
        if automatic:
            assert f"/generation-{generation}" in saved
            assert f"error-generation-{generation}" in saved
        else:
            assert f"/generation-{generation}" in result.stdout
            assert f"error-generation-{generation}" in result.stderr
    if automatic:
        assert result.stdout == result.stderr == ""

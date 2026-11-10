"""Tests for the root server.py ASGI entrypoint."""

import runpy
from unittest.mock import patch


def test_server_module_exports_app_and_create_app() -> None:
    import server

    assert server.app is not None
    assert callable(server.create_app)
    new_app = server.create_app()
    assert new_app is not None


def test_server_main_invokes_uvicorn_run() -> None:
    import uvicorn

    with patch.object(uvicorn, "run") as mock_run:
        runpy.run_module("server", run_name="__main__")
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert "host" in call_kwargs
        assert "port" in call_kwargs
        assert call_kwargs["log_level"] == "debug"

def test_server_module_exports_app_and_create_app():
    import server

    assert server.app is not None
    assert callable(server.create_app)


def test_server_main_invokes_uvicorn_run(monkeypatch):
    import runpy
    from types import SimpleNamespace
    from unittest.mock import patch

    import config.settings as settings_mod
    import uvicorn as uvicorn_mod

    # Patch settings used by server.__main__ block.
    old_get_settings = settings_mod.get_settings
    mock_settings = SimpleNamespace(host="127.0.0.1", port=9999)

    try:
        with (
            patch.object(settings_mod, "get_settings", lambda: mock_settings),
            patch.object(uvicorn_mod, "run") as mock_run,
        ):
            runpy.run_module("server", run_name="__main__")
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 9999
            assert call_kwargs["log_level"] == "debug"
    finally:
        settings_mod.get_settings = old_get_settings

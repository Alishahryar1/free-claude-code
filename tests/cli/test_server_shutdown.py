"""Exercise supervised shutdown with real HTTP event-feed connections."""

import asyncio
import socket
import threading
from pathlib import Path

import httpx
import pytest

from free_claude_code.cli import commands
from free_claude_code.config.settings import Settings
from free_claude_code.runtime.application import RestartCallback
from free_claude_code.runtime.asgi import RuntimeASGIApp
from free_claude_code.runtime.bootstrap import build_asgi_app


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_supervisor_drains_admin_event_feed_without_forced_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    restart: bool,
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    settings = Settings().model_copy(
        update={
            "host": "127.0.0.1",
            "port": port,
            "messaging_platform": "none",
            "voice_note_enabled": False,
            "open_admin_browser": False,
            "log_file": str(tmp_path / "server.log"),
        }
    )
    apps: list[RuntimeASGIApp] = []

    def create_app(settings: Settings, *, restart_callback: RestartCallback):
        app = build_asgi_app(settings, restart_callback=restart_callback)
        # Only provider discovery is external; retain the real Code, HTTP,
        # runtime cleanup, and supervisor lifecycle under investigation.
        monkeypatch.setattr(
            app.runtime.provider_manager, "start_model_list_refresh", lambda: None
        )
        apps.append(app)
        return app

    monkeypatch.setattr(commands, "load_server_settings", lambda: settings)
    monkeypatch.setattr(
        "free_claude_code.runtime.bootstrap.configure_logging",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("free_claude_code.runtime.bootstrap.build_asgi_app", create_app)
    monkeypatch.setattr(commands, "kill_all_best_effort", lambda: None)
    monkeypatch.setattr(commands, "SERVER_GRACEFUL_SHUTDOWN_SECONDS", 0.2)
    supervisor = commands.ServerSupervisor(console_logging=False)
    thread = threading.Thread(target=supervisor.run, daemon=True)
    thread.start()
    stream_error: Exception | None = None
    try:
        async with asyncio.timeout(5):
            while supervisor.status != commands.ServerStatus.RUNNING:
                assert thread.is_alive()
                await asyncio.sleep(0.01)
        async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
            status_url = f"http://127.0.0.1:{port}/admin/api/status"
            old_instance = (await client.get(status_url)).json()["instance_id"]
            events_url = f"http://127.0.0.1:{port}/admin/api/code/events"
            async with (
                client.stream("GET", events_url) as first,
                client.stream("GET", events_url) as second,
            ):
                streams = [first.aiter_lines(), second.aiter_lines()]
                assert first.status_code == second.status_code == 200
                for lines in streams:
                    async for line in lines:
                        if line.startswith("event: feed.ready"):
                            break
                if restart:
                    assert supervisor.request_restart()
                else:
                    supervisor.request_stop()
                for lines in streams:
                    try:
                        async for _line in lines:
                            pass
                    except httpx.HTTPError as exc:
                        stream_error = exc
            if restart:
                async with asyncio.timeout(5):
                    while (
                        len(apps) < 2
                        or supervisor.status != commands.ServerStatus.RUNNING
                    ):
                        assert thread.is_alive()
                        await asyncio.sleep(0.01)
                new_status = (await client.get(status_url)).json()
                assert new_status["status"] == "running"
                assert new_status["instance_id"] != old_instance
    finally:
        supervisor.request_stop()
        await asyncio.to_thread(thread.join, 5)
    assert not thread.is_alive()
    assert stream_error is None, f"Event feed was cut off: {stream_error}"
    assert "timeout graceful shutdown exceeded" not in caplog.text
    assert all(app.runtime.is_closed for app in apps)
    assert len(apps) == (2 if restart else 1)

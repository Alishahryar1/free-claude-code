import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from free_claude_code.config.settings import Settings
from smoke.lib import child_process
from smoke.lib import e2e as smoke_e2e
from smoke.lib import http as smoke_http
from smoke.lib import server as smoke_server
from smoke.lib.child_process import (
    cmd_fcc_server,
    cmd_python_c,
    run_captured_text,
)
from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import ConversationDriver
from smoke.lib.http import collect_message_stream
from smoke.lib.server import RunningServer


def test_fcc_server_command_uses_cli_entrypoint() -> None:
    assert cmd_fcc_server() == [
        child_process.python_exe(),
        "-c",
        "from free_claude_code.cli.entrypoints import serve; serve()",
    ]


def test_start_server_disables_cli_admin_browser(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            captured["command"] = command
            captured.update(kwargs)

        def poll(self) -> int | None:
            return None

    config = SmokeConfig(
        root=tmp_path,
        results_dir=tmp_path / "results",
        live=True,
        interactive=False,
        targets=frozenset(),
        provider_matrix=frozenset(),
        timeout_s=1.0,
        prompt="",
        claude_bin="claude",
        worker_id="test",
        settings=Settings(),
    )

    monkeypatch.setattr(smoke_server, "find_free_port", lambda: 4567)
    monkeypatch.setattr(smoke_server.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        smoke_server, "_wait_for_health", lambda _server, *, timeout_s: None
    )
    monkeypatch.setattr(smoke_server, "_stop_process", lambda _process: None)

    with smoke_server.start_server(config):
        pass

    env_obj = captured["env"]
    assert isinstance(env_obj, dict)
    env = {str(key): value for key, value in env_obj.items()}
    assert env["FCC_OPEN_BROWSER"] == "0"
    assert env["HOST"] == "127.0.0.1"
    assert env["PORT"] == "4567"


def test_collect_message_stream_explicitly_requests_sse(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_stream(_method: str, _url: str, **kwargs: object):
        captured.update(kwargs)
        return _FakeStreamResponse()

    monkeypatch.setattr(smoke_http.httpx, "stream", fake_stream)
    payload = {"model": "m", "messages": [], "stream": False}

    events = collect_message_stream(
        _running_server(tmp_path),
        payload,
        _smoke_config(tmp_path),
        headers={"authorization": "Bearer test"},
    )

    assert captured["json"] == {**payload, "stream": True}
    assert payload["stream"] is False
    assert events[-1].event == "message_stop"


def test_conversation_driver_explicitly_requests_sse(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_stream(_method: str, _url: str, **kwargs: object):
        captured.update(kwargs)
        return _FakeStreamResponse()

    monkeypatch.setattr(smoke_e2e.httpx, "stream", fake_stream)
    payload = {"model": "m", "messages": [], "stream": False}

    turn = ConversationDriver(
        _running_server(tmp_path),
        _smoke_config(tmp_path),
    ).stream(payload, headers={"authorization": "Bearer test"})

    assert captured["json"] == {**payload, "stream": True}
    assert payload["stream"] is False
    assert turn.request["stream"] is True
    assert turn.events[-1].event == "message_stop"


def test_run_captured_text_uses_utf8_replacement(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls["command"] = command
        calls.update(kwargs)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(child_process.subprocess, "run", fake_run)

    result = run_captured_text(
        ("cmd", "arg"),
        cwd=tmp_path,
        env={"FCC_TEST": "1"},
        timeout=1.0,
    )

    assert result.stdout == "ok"
    assert calls["command"] == ["cmd", "arg"]
    assert calls["cwd"] == tmp_path
    assert calls["env"] == {"FCC_TEST": "1"}
    assert calls["capture_output"] is True
    assert calls["text"] is True
    assert calls["encoding"] == "utf-8"
    assert calls["errors"] == "replace"
    assert calls["timeout"] == 1.0
    assert calls["check"] is False


def test_run_captured_text_replaces_invalid_utf8_bytes(tmp_path: Path) -> None:
    result = run_captured_text(
        cmd_python_c(
            "import sys; "
            "sys.stdout.buffer.write(bytes([0x8f])); "
            "sys.stderr.buffer.write(bytes([0x8f]))"
        ),
        cwd=tmp_path,
        timeout=10.0,
    )

    assert result.returncode == 0
    assert result.stdout == "\ufffd"
    assert result.stderr == "\ufffd"


class _FakeStreamResponse:
    status_code = 200

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return

    def iter_lines(self):
        return iter(
            (
                "event: message_start",
                'data: {"type":"message_start","message":{}}',
                "",
                "event: content_block_start",
                'data: {"type":"content_block_start","index":0,'
                '"content_block":{"type":"text","text":""}}',
                "",
                "event: content_block_delta",
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"ok"}}',
                "",
                "event: content_block_stop",
                'data: {"type":"content_block_stop","index":0}',
                "",
                "event: message_stop",
                'data: {"type":"message_stop"}',
                "",
            )
        )


def _smoke_config(tmp_path: Path) -> SmokeConfig:
    return SmokeConfig(
        root=tmp_path,
        results_dir=tmp_path / "results",
        live=False,
        interactive=False,
        targets=frozenset(),
        provider_matrix=frozenset(),
        timeout_s=1.0,
        prompt="",
        claude_bin="claude",
        worker_id="test",
        settings=Settings(),
    )


def _running_server(tmp_path: Path) -> RunningServer:
    return RunningServer(
        base_url="http://127.0.0.1:1234",
        port=1234,
        log_path=tmp_path / "server.log",
        process=MagicMock(spec=subprocess.Popen),
    )

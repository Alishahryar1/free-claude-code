"""Installed Antigravity uses the real FCC launcher against isolated HTTP fixtures."""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import httpx2
import pytest
import uvicorn
from PIL import Image

from free_claude_code.cli.process_registry import kill_pid_tree_best_effort
from free_claude_code.config import env_migrations, paths
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses import antigravity_integration
from tests.api.support import create_test_app, provider_manager_for_app
from tests.providers.test_custom_provider import definition, event, provider
from tests.providers.test_gemini_transport import chat_response

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("clients")]

_MARKER = "FCC_ANTIGRAVITY_OK"


def _reply(wire: str, calls: list[Path], reasoning: bool):
    if wire == "openai_chat":
        deltas: list[JsonObject] = (
            [{"reasoning_content": "FCC_NATIVE_THOUGHT"}] if reasoning else []
        )
        deltas.append(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": f"call-{index}",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"Path": str(path)}),
                        },
                    }
                    for index, path in enumerate(calls)
                ]
            }
            if calls
            else {"content": _MARKER}
        )
        return chat_response(deltas, finish="tool_calls" if calls else "stop")
    if wire == "openai_responses":
        output: list[JsonObject] = (
            [
                {
                    "type": "reasoning",
                    "id": "rs_native",
                    "encrypted_content": "FCC_OPAQUE",
                }
            ]
            if reasoning
            else []
        )
        output += (
            [
                {
                    "type": "function_call",
                    "id": f"fc_{index}",
                    "call_id": f"call-{index}",
                    "name": "read_file",
                    "arguments": json.dumps({"Path": str(path)}),
                    "status": "completed",
                }
                for index, path in enumerate(calls)
            ]
            if calls
            else [
                {
                    "type": "message",
                    "id": "msg_native",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": _MARKER, "annotations": []}
                    ],
                }
            ]
        )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=event(
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_native",
                        "object": "response",
                        "status": "completed",
                        "output": output,
                    },
                },
            ),
        )
    blocks: list[JsonObject] = (
        [
            {
                "type": "thinking",
                "thinking": "FCC_NATIVE_THOUGHT",
                "signature": "FCC_SIGNED",
            }
        ]
        if reasoning
        else []
    )
    blocks += (
        [
            {
                "type": "tool_use",
                "id": f"call-{index}",
                "name": "read_file",
                "input": {"Path": str(path)},
            }
            for index, path in enumerate(calls)
        ]
        if calls
        else [{"type": "text", "text": _MARKER}]
    )
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_native",
                "type": "message",
                "role": "assistant",
                "model": "backend",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }
    ]
    for index, block in enumerate(blocks):
        events.extend(
            [
                {"type": "content_block_start", "index": index, "content_block": block},
                {"type": "content_block_stop", "index": index},
            ]
        )
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use" if calls else "end_turn"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(event(item["type"], item) for item in events),
    )


@pytest.mark.parametrize(
    "wire", ["openai_chat", "openai_responses", "anthropic_messages"]
)
@pytest.mark.parametrize("mode", ["parallel", "image", "resume"])
def test_native_launcher_connect_tools_resume_disconnect(
    wire, mode, tmp_path, monkeypatch
):
    native = shutil.which("agy")
    if native is None:
        pytest.skip("Antigravity CLI is not installed")
    launcher = Path(sys.executable).with_name(
        "fcc-agy.exe" if os.name == "nt" else "fcc-agy"
    )
    assert launcher.is_file(), "Install the current FCC project before native smoke"
    home, workspace = tmp_path / "home", tmp_path / "workspace"
    workspace.mkdir()
    first, second, picture = (
        workspace / "first.txt",
        workspace / "second.txt",
        workspace / "fixture.png",
    )
    first.write_text("FCC_FIRST_CONTENT")
    second.write_text("FCC_SECOND_CONTENT")
    Image.new("RGB", (1, 1), "red").save(picture)
    native_settings = home / ".gemini/antigravity-cli/settings.json"
    native_settings.parent.mkdir(parents=True)
    original = {"modelProvider": "", "fccSmokeUnrelated": {"keep": True}}
    native_settings.write_text(json.dumps(original))
    monkeypatch.setattr(paths, "config_dir_path", lambda: home / ".fcc")
    monkeypatch.setattr(antigravity_integration, "config_path", lambda: native_settings)
    monkeypatch.setattr(env_migrations, "legacy_env_paths", lambda: ())
    monkeypatch.setattr(env_migrations, "verified_checkout_env_path", lambda: None)
    sent, ingress = [], []
    issued = False

    def upstream(request):
        nonlocal issued
        body = json.loads(request.content)
        sent.append(body)
        initial = bool(body.get("tools")) and not issued
        if initial:
            issued = True
        calls = (
            (
                [first, second]
                if mode == "parallel"
                else [picture]
                if mode == "image"
                else []
            )
            if initial
            else []
        )
        return _reply(wire, calls, initial)

    config = definition(api_format=wire, api_keys=["synthetic-upstream"])
    instance = provider(config, openai_handler=upstream, messages_handler=upstream)
    selected = f"{config.provider_id}/agy-backend:variant"
    token = "synthetic-agy-proxy"
    env_migrations.atomic_write_managed_config(
        {"ANTHROPIC_AUTH_TOKEN": token}, path=home / ".fcc/.env"
    )
    app = create_test_app(
        Settings(
            MODEL=selected,
            custom_providers=(config,),
            PROXY_AUTH_ENABLED=True,
            proxy_auth_token=token,
        ),
        providers={config.provider_id: instance},
    )

    @app.middleware("http")
    async def capture(request, call_next):
        ingress.append(request.url.path)
        return await call_next(request)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))

    async def serve():
        try:
            await server.serve(sockets=[sock])
        finally:
            await instance.cleanup()

    thread = threading.Thread(target=lambda: asyncio.run(serve()), daemon=True)
    thread.start()
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "PROCESSOR_ARCHITECTURE",
            "NUMBER_OF_PROCESSORS",
            "OS",
            "PROGRAMFILES",
            "PROGRAMFILES(X86)",
        }
    }
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        LOCALAPPDATA=str(home / "AppData/Local"),
        APPDATA=str(home / "AppData/Roaming"),
        TEMP=str(home / "tmp"),
        TMP=str(home / "tmp"),
        HOST="127.0.0.1",
        PORT=str(sock.getsockname()[1]),
        ANTHROPIC_AUTH_TOKEN=token,
        FCC_OPEN_BROWSER="0",
        PATH=str(Path(native).parent) + os.pathsep + env.get("PATH", ""),
    )
    for key in ("LOCALAPPDATA", "APPDATA", "TEMP"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    def run(phase, *args):
        process = subprocess.Popen(
            [str(launcher), *args],
            cwd=workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out, err = process.communicate(timeout=35)
        except subprocess.TimeoutExpired:
            kill_pid_tree_best_effort(process.pid)
            out, err = process.communicate(timeout=5)
            pytest.fail(
                f"Antigravity {phase} timed out: {err.decode(errors='replace')}"
            )
        (tmp_path / f"{phase}.stdout.txt").write_bytes(out)
        (tmp_path / f"{phase}.stderr.txt").write_bytes(err)
        assert process.returncode == 0, err.decode(errors="replace") or out.decode(
            errors="replace"
        )
        assert token not in (out + err).decode(errors="replace")
        return out.decode()

    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started
        admin = f"http://127.0.0.1:{sock.getsockname()[1]}/admin/api/integrations/antigravity"
        with httpx.Client(trust_env=False) as client:
            response = client.post(admin + "/connect")
            assert response.status_code == 200, response.text
            assert response.json()["connected"] is True
        assert "FCC:" in run("models", "models")
        assert json.loads(native_settings.read_text())["modelProvider"] == "gemini"
        assert token not in native_settings.read_text()
        args = [
            "--print",
            "Read the requested files, then reply " + _MARKER,
            "--print-timeout",
            "20s",
            "--output-format",
            "json",
        ]
        first_output = json.loads(run("initial", *args))
        assert first_output["status"] == "SUCCESS"
        assert _MARKER in json.dumps(first_output)
        if mode == "resume":
            resumed = json.loads(
                run("resumed", "--conversation", first_output["conversation_id"], *args)
            )
            assert resumed["status"] == "SUCCESS" and _MARKER in json.dumps(resumed)
        main = [body for body in sent if body.get("tools")]
        assert len(main) >= 2
        followup = json.dumps(main[1])
        if mode == "parallel":
            assert "FCC_FIRST_CONTENT" in followup and "FCC_SECOND_CONTENT" in followup
        if mode == "image":
            assert "image/png" in followup and "iVBOR" in followup
        assert {
            "openai_chat": "FCC_NATIVE_THOUGHT",
            "openai_responses": "FCC_OPAQUE",
            "anthropic_messages": "FCC_SIGNED",
        }[wire] in followup
        assert all(body["model"] == "agy-backend:variant" for body in sent)
        assert f"/v1beta/models/{selected}:streamGenerateContent" in ingress
        assert provider_manager_for_app(app)._current.active_leases == 0
        with httpx.Client(trust_env=False) as client:
            response = client.post(admin + "/disconnect")
            assert response.status_code == 200, response.text
            assert response.json()["connected"] is False
        assert json.loads(native_settings.read_text()) == original
        assert not (home / ".fcc/antigravity-integration.json").exists()
        assert list((home / ".gemini/antigravity-cli").rglob("*.db")), (
            "Native history must survive disconnect"
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive()

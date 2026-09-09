"""Installed Codex modes through FCC; local cases never use provider credentials."""

import asyncio
import json
import os
import shlex
import shutil
import sys
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from free_claude_code.config.env_migrations import (
    atomic_write_managed_config,
    settings_env_keys,
)
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from smoke.lib.child_process import run_captured_text
from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import SmokeServerDriver

pytestmark = [
    pytest.mark.live,
    pytest.mark.clients,
    pytest.mark.smoke_target("clients"),
]


def _environment(tmp_path: Path, model: str) -> tuple[dict[str, str], set[str]]:
    binary = shutil.which("codex")
    if not binary:
        pytest.skip("missing_env: Codex is not installed")
    version = run_captured_text([binary, "--version"], timeout=10, check=True)
    print(version.stdout.strip())
    atomic_write_managed_config(
        {"ANTHROPIC_AUTH_TOKEN": "codex-mode-smoke", "PROXY_AUTH_ENABLED": "true"},
        path=tmp_path / "home" / ".fcc" / ".env",
    )
    native_home = tmp_path / "codex-home"
    native_home.mkdir()
    (native_home / "config.toml").write_text(
        'approval_policy = "on-request"\nsandbox_mode = "read-only"\n'
        "[analytics]\nenabled = false\n",
        encoding="utf-8",
    )
    env = {
        "HOME": str(tmp_path / "home"),
        "USERPROFILE": str(tmp_path / "home"),
        "CODEX_HOME": str(native_home),
        "FCC_OPEN_BROWSER": "0",
        "MODEL": model,
        "MODEL_FABLE": model,
        "MODEL_OPUS": model,
        "MODEL_SONNET": model,
        "MODEL_HAIKU": model,
        "MODEL_FALLBACKS": "",
        "MESSAGING_PLATFORM": "none",
        "ANTHROPIC_AUTH_TOKEN": "codex-mode-smoke",
        "PATH": os.pathsep.join(
            [
                str(Path(binary).parent),
                str(Path(sys.executable).parent),
                os.environ.get("PATH", ""),
            ]
        ),
    }
    # A fresh home excludes connected accounts and project configuration.
    unset = set(settings_env_keys())
    unset.update({"FCC_ENV_FILE", "OPENAI_API_KEY", "CODEX_API_KEY"})
    return env, unset


@contextmanager
def _canned_provider() -> Iterator[tuple[str, dict[str, Any]]]:
    state: dict[str, Any] = {"requests": [], "command": None, "reviews": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            body = json.dumps(
                {"data": [{"id": "codex-mode-smoke", "object": "model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(request)
            try:
                assert self.path == "/v1/chat/completions"
                assert request["model"] == "codex-mode-smoke"
                schema = (
                    request.get("response_format", {})
                    .get("json_schema", {})
                    .get("schema", {})
                )
                if "outcome" in schema.get("properties", {}):
                    state["reviews"] += 1
                    delta = {
                        "content": json.dumps(
                            {
                                "outcome": "allow",
                                "risk_level": "low",
                                "user_authorization": "high",
                                "rationale": "Disposable local smoke action",
                            }
                        )
                    }
                    reason = "stop"
                elif state["command"]:
                    names = [
                        tool["function"]["name"]
                        for tool in request["tools"]
                        if tool["type"] == "function"
                    ]
                    assert "exec_command" in names, names
                    arguments = {
                        "cmd": state["command"],
                        "max_output_tokens": 1000,
                    }
                    if state["mode"] != "full_access":
                        arguments.update(
                            sandbox_permissions="require_escalated",
                            justification="Create the disposable smoke marker",
                        )
                    state["command"] = None
                    delta = {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_" + uuid.uuid4().hex,
                                "type": "function",
                                "function": {
                                    "name": "exec_command",
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ]
                    }
                    reason = "tool_calls"
                else:
                    delta = {"content": "FCC_MODE_SMOKE_DONE"}
                    reason = "stop"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for payload, finish in ((delta, None), ({}, reason)):
                    chunk = {
                        "id": "chatcmpl-" + uuid.uuid4().hex,
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "codex-mode-smoke",
                        "choices": [
                            {"index": 0, "delta": payload, "finish_reason": finish}
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as exc:
                state["error"] = repr(exc)
                self.send_error(500, "Local smoke fixture failed")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def _events(response: httpx.Response, queue: asyncio.Queue) -> None:
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            queue.put_nowait(json.loads(line[6:]))


async def _exercise(
    base_url: str,
    workspace: Path,
    modes: tuple[str, ...],
    state: dict[str, Any] | None,
    timeout_s: float,
    session_id: str | None = None,
) -> str:
    workspace.mkdir(exist_ok=True)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        bootstrap = (await client.get("/admin/api/code/bootstrap")).json()
        assert bootstrap["available"], bootstrap
        path = f"/admin/api/code/sessions/{session_id or uuid.uuid4()}"
        prepared = (
            await client.get(path)
            if session_id
            else await client.post(
                "/admin/api/code/sessions",
                json={"session_id": path.rsplit("/", 1)[1], "cwd": str(workspace)},
            )
        )
        prepared.raise_for_status()
        session = prepared.json()["session"] if session_id else prepared.json()
        queue = asyncio.Queue()
        async with client.stream(
            "GET", "/admin/api/code/events", timeout=None
        ) as response:
            response.raise_for_status()
            reading = asyncio.create_task(_events(response, queue))
            try:
                await asyncio.wait_for(queue.get(), timeout_s)
                for mode in modes:
                    changed = await client.patch(
                        path,
                        json={"expected_revision": session["revision"], "mode": mode},
                    )
                    changed.raise_for_status()
                    session = changed.json()
                    marker = workspace.parent / f"marker-{uuid.uuid4().hex}.txt"
                    command = (
                        f"Set-Content -LiteralPath '{str(marker).replace(chr(39), chr(39) * 2)}' -Value smoke"
                        if os.name == "nt"
                        else f"printf smoke > {shlex.quote(str(marker))}"
                    )
                    if state is not None:
                        state["command"] = command
                        state["mode"] = mode
                    escalation = (
                        " with sandbox_permissions=require_escalated"
                        if mode != "full_access"
                        else ""
                    )
                    prompt = f"Run this exact harmless command once using exec_command{escalation}, then report its result: {command}"
                    posted = await client.post(
                        path + "/turns",
                        json={
                            "operation_id": str(uuid.uuid4()),
                            "expected_revision": session["revision"],
                            "expected_epoch": bootstrap["epoch"],
                            "text": prompt,
                        },
                    )
                    posted.raise_for_status()
                    run_id = posted.json()["id"]
                    approvals = 0
                    async with asyncio.timeout(timeout_s):
                        while True:
                            event = await queue.get()
                            if event.get("session_id") != session["id"]:
                                continue
                            pending = event.get("prompt")
                            if pending and pending["status"] == "pending":
                                assert mode in {"config", "ask"}, pending
                                approvals += 1
                                choices = pending["form"]["choices"]
                                choice = next(
                                    choice["id"]
                                    for choice in choices
                                    if choice["label"] in {"Allow once", "Allow"}
                                )
                                answer = await client.post(
                                    path + f"/prompts/{pending['id']}/responses",
                                    json={
                                        "response_id": str(uuid.uuid4()),
                                        "answer": {"choice": choice},
                                    },
                                )
                                answer.raise_for_status()
                            run = event.get("run")
                            if (
                                run
                                and run["id"] == run_id
                                and run["status"]
                                in {"completed", "failed", "interrupted"}
                            ):
                                assert run["status"] == "completed", run
                                break
                    detail = (await client.get(path)).json()
                    session = detail["session"]
                    assert marker.exists(), detail
                    assert approvals == (1 if mode in {"config", "ask"} else 0), detail
                    reviews = [
                        item
                        for item in detail["items"]
                        if item["run_id"] == run_id and item["kind"] == "auto_review"
                    ]
                    if mode == "auto_review":
                        assert reviews and all(
                            item["complete"]
                            and item["title"] == "Auto-review: Approved"
                            for item in reviews
                        ), detail
                    else:
                        assert not reviews, detail
                    print(
                        f"{mode}: command completed; manual approvals={approvals}; reviews={len(reviews)}"
                    )
            finally:
                reading.cancel()
                await asyncio.gather(reading, return_exceptions=True)
        return session["id"]


def test_codex_modes_local_e2e(smoke_config: SmokeConfig, tmp_path: Path) -> None:
    env, unset = _environment(tmp_path, "lmstudio/codex-mode-smoke")
    with _canned_provider() as (url, state):
        env["LM_STUDIO_BASE_URL"] = url
        driver = SmokeServerDriver(
            smoke_config, name="codex-modes-local", env_overrides=env, env_unset=unset
        )
        try:
            with driver.run() as server:
                session_id = asyncio.run(
                    _exercise(
                        server.base_url,
                        tmp_path / "workspace",
                        ("ask", "auto_review", "full_access"),
                        state,
                        smoke_config.timeout_s,
                    )
                )
            # A new FCC process must restore the original native settings when
            # resuming the conversation that last ran with Full access.
            with driver.run() as server:
                asyncio.run(
                    _exercise(
                        server.base_url,
                        tmp_path / "workspace",
                        ("config",),
                        state,
                        smoke_config.timeout_s,
                        session_id,
                    )
                )
            assert "error" not in state, state.get("error")
            assert state["reviews"] >= 1
        finally:
            (tmp_path / "local-requests.json").write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )


def test_codex_modes_free_provider_e2e(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    model = os.getenv("FCC_SMOKE_CODEX_FREE_MODEL")
    if not model:
        pytest.skip(
            "missing_env: select FCC_SMOKE_CODEX_FREE_MODEL to run free live inference"
        )
    provider, name = model.split("/", 1)
    assert provider == "open_router", (
        "This live scenario verifies OpenRouter's published zero pricing."
    )
    catalog = httpx.get("https://openrouter.ai/api/v1/models", timeout=20)
    catalog.raise_for_status()
    entry = next(entry for entry in catalog.json()["data"] if entry["id"] == name)
    assert all(float(entry["pricing"][key]) == 0 for key in ("prompt", "completion")), (
        "Selected model is not free"
    )
    descriptor = PROVIDER_CATALOG[provider]
    assert descriptor.credential_attr and descriptor.credential_env
    key = getattr(smoke_config.settings, descriptor.credential_attr)
    if not key:
        pytest.skip("missing_env: OpenRouter key is unavailable")
    env, unset = _environment(tmp_path, model)
    env[descriptor.credential_env] = key
    with SmokeServerDriver(
        smoke_config, name="codex-modes-free", env_overrides=env, env_unset=unset
    ).run() as server:
        asyncio.run(
            _exercise(
                server.base_url,
                tmp_path / "workspace",
                ("auto_review",),
                None,
                max(smoke_config.timeout_s, 120),
            )
        )

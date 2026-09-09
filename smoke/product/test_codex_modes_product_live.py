"""Installed Codex modes through FCC; local cases never use provider credentials."""

import asyncio
import json
import os
import shlex
import shutil
import sqlite3
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


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ]
    }


def _marker_command(marker: Path) -> str:
    return (
        f"Set-Content -LiteralPath '{str(marker).replace(chr(39), chr(39) * 2)}' -Value smoke"
        if os.name == "nt"
        else f"printf smoke > {shlex.quote(str(marker))}"
    )


@contextmanager
def _canned_provider(
    review_release: threading.Event | None = None,
    child_finished: threading.Event | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
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
                # The child starts without the parent's history. Route the canned
                # actor by that boundary, independently of task-message encoding.
                child_request = child_finished is not None and not any(
                    "FCC_PARENT_REVIEW_WORK" in json.dumps(message.get("content", ""))
                    for message in request["messages"]
                    if message["role"] == "user"
                )
                if "outcome" in schema.get("properties", {}):
                    state["reviews"] += 1
                    if review_release is not None:
                        assert review_release.wait(30), (
                            "Local child reviewer was not released"
                        )
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
                elif state.get("spawn"):
                    functions = {
                        tool["function"]["name"]: tool["function"]
                        for tool in request["tools"]
                        if tool["type"] == "function"
                    }
                    spawn_name = next(
                        (
                            name
                            for name in functions
                            if name.rsplit("__", 1)[-1] == "spawn_agent"
                        ),
                        None,
                    )
                    assert spawn_name is not None, list(functions)
                    properties = functions[spawn_name]["parameters"]["properties"]
                    arguments = {
                        "message": "FCC_CHILD_REVIEW_WORK: Execute the disposable local smoke command, then finish."
                    }
                    if "task_name" in properties:
                        arguments["task_name"] = "smoke_child"
                    if "fork_turns" in properties:
                        arguments["fork_turns"] = "none"
                    elif "fork_context" in properties:
                        arguments["fork_context"] = False
                    state["spawn"] = False
                    delta = _tool_call(spawn_name, arguments)
                    reason = "tool_calls"
                elif state["command"] and (child_finished is None or child_request):
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
                    delta = _tool_call("exec_command", arguments)
                    reason = "tool_calls"
                else:
                    if child_request and child_finished is not None:
                        child_finished.set()
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
                self.send_error(400, "Local smoke fixture failed")

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
                    command = _marker_command(marker)
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


async def _exercise_child_review(
    base_url: str,
    workspace: Path,
    release: threading.Event,
    finished: threading.Event,
    timeout_s: float,
) -> str:
    workspace.mkdir()
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        bootstrap = (await client.get("/admin/api/code/bootstrap")).json()
        assert bootstrap["available"], bootstrap
        session_id = str(uuid.uuid4())
        path = f"/admin/api/code/sessions/{session_id}"
        created = await client.post(
            "/admin/api/code/sessions",
            json={"session_id": session_id, "cwd": str(workspace)},
        )
        created.raise_for_status()
        changed = await client.patch(
            path,
            json={
                "expected_revision": created.json()["revision"],
                "mode": "auto_review",
            },
        )
        changed.raise_for_status()
        session = changed.json()
        queue = asyncio.Queue()

        async def send(text: str) -> str:
            response = await client.post(
                path + "/turns",
                json={
                    "operation_id": str(uuid.uuid4()),
                    "expected_revision": session["revision"],
                    "expected_epoch": bootstrap["epoch"],
                    "text": text,
                },
            )
            response.raise_for_status()
            return response.json()["id"]

        async def event() -> dict[str, Any]:
            while True:
                value = await queue.get()
                if value.get("session_id") == session_id:
                    assert value.get("prompt", {}).get("status") != "pending", value
                    run = value.get("run") or {}
                    assert run.get("status") not in {"failed", "interrupted"}, value
                    return value

        async with client.stream(
            "GET", "/admin/api/code/events", timeout=None
        ) as response:
            response.raise_for_status()
            reading = asyncio.create_task(_events(response, queue))
            try:
                async with asyncio.timeout(timeout_s):
                    await queue.get()
                    first = await send(
                        "FCC_PARENT_REVIEW_WORK: Delegate a harmless local command to one child agent."
                    )
                    review = None
                    parent_done = False
                    while review is None or not parent_done:
                        value = await event()
                        item = value.get("item", {})
                        if item.get("kind") == "subagent_auto_review":
                            review = item
                        parent_done |= (
                            value.get("run", {}).get("id") == first
                            and value["run"]["status"] == "completed"
                        )
                    detail = (await client.get(path)).json()
                    assert (
                        not review["complete"]
                        and review["id"] in detail["active_review_ids"]
                    ), detail
                    session = detail["session"]
                    second = await send("Reply while the child review is pending.")
                    while True:
                        value = await event()
                        if (
                            value.get("run", {}).get("id") == second
                            and value["run"]["status"] == "completed"
                        ):
                            break
                    release.set()
                    while True:
                        value = await event()
                        item = value.get("item", {})
                        if item.get("id") == review["id"] and item["complete"]:
                            assert item["title"] == "Sub-agent Auto-review: Approved", (
                                item
                            )
                            assert (
                                item["run_id"] == first
                                and item["sequence"] == review["sequence"]
                            ), item
                            assert (
                                value["run"]["id"] == second
                                and value["run"]["status"] == "completed"
                            ), value
                            break
                    assert await asyncio.to_thread(finished.wait, timeout_s), (
                        "Child did not finish its command"
                    )
                    detail = (await client.get(path)).json()
                    assert detail["active_review_ids"] == [], detail
                    print(
                        "child Auto-review: parent finished; next message completed; original review approved"
                    )
            finally:
                release.set()
                reading.cancel()
                await asyncio.gather(reading, return_exceptions=True)
        return session_id


def test_codex_child_review_local_e2e(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    env, unset = _environment(tmp_path, "lmstudio/codex-mode-smoke")
    with (tmp_path / "codex-home" / "config.toml").open(
        "a", encoding="utf-8"
    ) as config:
        config.write("[features]\nmulti_agent = true\nmulti_agent_v2 = true\n")
    release, finished = threading.Event(), threading.Event()
    marker = tmp_path / "child-marker.txt"
    with _canned_provider(release, finished) as (url, state):
        env["LM_STUDIO_BASE_URL"] = url
        state.update(spawn=True, mode="auto_review", command=_marker_command(marker))
        try:
            with SmokeServerDriver(
                smoke_config,
                name="codex-child-review-local",
                env_overrides=env,
                env_unset=unset,
            ).run() as server:
                session_id = asyncio.run(
                    _exercise_child_review(
                        server.base_url,
                        tmp_path / "workspace",
                        release,
                        finished,
                        smoke_config.timeout_s,
                    )
                )
                assert marker.read_text().strip() == "smoke"
                with sqlite3.connect(
                    tmp_path / "home" / ".fcc" / "code" / "code.db"
                ) as database:
                    root = database.execute(
                        "SELECT native_thread_id FROM code_sessions WHERE id = ?",
                        (session_id,),
                    ).fetchone()[0]
                    rows = database.execute(
                        "SELECT raw FROM code_items WHERE session_id = ? AND kind = 'subagent_auto_review'",
                        (session_id,),
                    ).fetchall()
                    assert rows and all(
                        json.loads(row[0])["threadId"] != root for row in rows
                    )
            assert "error" not in state, state.get("error")
            assert state["reviews"] == 1
        finally:
            release.set()
            (tmp_path / "child-requests.json").write_text(
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

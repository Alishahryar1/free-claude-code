"""Credential-free subprocess coverage for provider generation replacement."""

import json
import threading
import time
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from smoke.lib.config import SmokeConfig
from smoke.lib.server import RunningServer, start_server

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("api")]


class FakeOpenAIUpstream:
    """Minimal OpenAI-chat upstream with an optionally held response stream."""

    def __init__(self, label: str, *, hold_stream: bool) -> None:
        self.label = label
        self.hold_stream = hold_stream
        self.chat_started = threading.Event()
        self.release_stream = threading.Event()
        self.chat_requests: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Fake upstream is not running")
        port = int(self._server.server_address[1])
        return f"http://127.0.0.1:{port}/v1"

    def __enter__(self) -> FakeOpenAIUpstream:
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path.rstrip("/").endswith("/models"):
                    upstream._send_models(self)
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                if self.path.rstrip("/").endswith("/chat/completions"):
                    upstream._send_chat(self)
                    return
                self.send_error(404)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"fake-openai-{self.label}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release_stream.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _send_models(self, handler: BaseHTTPRequestHandler) -> None:
        payload = json.dumps(
            {
                "object": "list",
                "data": [
                    {
                        "id": f"model-{self.label}",
                        "object": "model",
                        "created": 0,
                        "owned_by": "smoke",
                    }
                ],
            }
        ).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _send_chat(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("content-length", "0"))
        body = handler.rfile.read(length)
        request = json.loads(body) if body else {}
        self.chat_requests.append(request)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        try:
            self._write_chunk(
                handler,
                content=f"provider-{self.label}-start ",
                finish_reason=None,
            )
            if self.hold_stream:
                time.sleep(0.85)
                self._write_chunk(
                    handler,
                    content=f"provider-{self.label}-held ",
                    finish_reason=None,
                )
            self.chat_started.set()
            if self.hold_stream and not self.release_stream.wait(timeout=30):
                return
            self._write_chunk(
                handler,
                content=f"provider-{self.label}-finish",
                finish_reason=None,
            )
            self._write_chunk(handler, content=None, finish_reason="stop")
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
        except OSError:
            return

    def _write_chunk(
        self,
        handler: BaseHTTPRequestHandler,
        *,
        content: str | None,
        finish_reason: str | None,
    ) -> None:
        delta: dict[str, str] = {}
        if content is not None:
            delta = {"role": "assistant", "content": content}
        payload = {
            "id": f"chatcmpl-{self.label}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": f"model-{self.label}",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        handler.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        handler.wfile.flush()


def _message_payload(*, stream: bool) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "generation smoke"}],
        "stream": stream,
    }


def _write_initial_managed_config(home: Path, upstream: FakeOpenAIUpstream) -> None:
    config_path = home / ".fcc" / ".env"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "MODEL=lmstudio/model-a",
                f"LM_STUDIO_BASE_URL={upstream.base_url}",
                "ENABLE_MODEL_THINKING=false",
                "LOG_LEVEL=DEBUG",
                "MESSAGING_PLATFORM=none",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _wait_for_generation_close(server: RunningServer, generation_id: int) -> None:
    deadline = time.monotonic() + 10
    generation_field = f'"generation_id": {generation_id}'
    while time.monotonic() < deadline:
        text = server.log_path.read_text(encoding="utf-8", errors="replace")
        if "provider_generation.closed" in text and generation_field in text:
            return
        time.sleep(0.1)
    raise AssertionError(
        f"generation {generation_id} close trace was not written:\n"
        + server.log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
    )


def test_provider_hot_swap_preserves_inflight_stream_e2e(
    smoke_config: SmokeConfig,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    downstream_started = threading.Event()
    old_body: list[str] = []
    old_errors: list[BaseException] = []
    credential_env_keys = {
        descriptor.credential_env
        for descriptor in PROVIDER_CATALOG.values()
        if descriptor.credential_env is not None
    }

    with ExitStack() as stack:
        upstream_a = stack.enter_context(FakeOpenAIUpstream("a", hold_stream=True))
        upstream_b = stack.enter_context(FakeOpenAIUpstream("b", hold_stream=False))
        _write_initial_managed_config(home, upstream_a)
        server = stack.enter_context(
            start_server(
                smoke_config,
                name="runtime-ownership",
                env_overrides={
                    "HOME": str(home),
                    "USERPROFILE": str(home),
                },
                env_unset=credential_env_keys
                | {
                    "ANTHROPIC_AUTH_TOKEN",
                    "ENABLE_MODEL_THINKING",
                    "FCC_ENV_FILE",
                    "LM_STUDIO_BASE_URL",
                    "LOG_LEVEL",
                    "MODEL",
                    "MODEL_FABLE",
                    "MODEL_HAIKU",
                    "MODEL_OPUS",
                    "MODEL_SONNET",
                    "REASONING_FABLE",
                    "REASONING_HAIKU",
                    "REASONING_OPUS",
                    "REASONING_POLICY",
                    "REASONING_SONNET",
                },
            )
        )

        managed_config = (home / ".fcc" / ".env").read_text(encoding="utf-8")
        assert "REASONING_POLICY=off" in managed_config
        assert "ENABLE_MODEL_THINKING" not in managed_config

        def consume_old_stream() -> None:
            try:
                with httpx.stream(
                    "POST",
                    f"{server.base_url}/v1/messages",
                    json=_message_payload(stream=True),
                    timeout=30,
                ) as response:
                    response.raise_for_status()
                    downstream_started.set()
                    old_body.append("".join(response.iter_text()))
            except BaseException as exc:
                old_errors.append(exc)
                downstream_started.set()

        old_thread = threading.Thread(
            target=consume_old_stream,
            name="old-generation-consumer",
        )
        old_thread.start()
        assert downstream_started.wait(timeout=10)
        assert upstream_a.chat_started.wait(timeout=10)
        assert old_errors == []

        apply_response = httpx.post(
            f"{server.base_url}/admin/api/config/apply",
            json={
                "values": {
                    "MODEL": "lmstudio/model-b",
                    "LM_STUDIO_BASE_URL": upstream_b.base_url,
                }
            },
            timeout=smoke_config.timeout_s,
        )
        assert apply_response.status_code == 200, apply_response.text
        apply_body = apply_response.json()
        assert apply_body["applied"] is True
        assert apply_body["pending_fields"] == []
        assert apply_body["restart"]["required"] is False

        new_response = httpx.post(
            f"{server.base_url}/v1/messages",
            json=_message_payload(stream=False),
            timeout=smoke_config.timeout_s,
        )
        assert new_response.status_code == 200, new_response.text
        new_text = "".join(
            block.get("text", "") for block in new_response.json()["content"]
        )
        assert new_text == "provider-b-start provider-b-finish"
        assert not old_body

        upstream_a.release_stream.set()
        old_thread.join(timeout=10)
        assert not old_thread.is_alive()
        assert old_errors == []
        assert "provider-a-start " in old_body[0]
        assert "provider-a-held " in old_body[0]
        assert "provider-a-finish" in old_body[0]
        assert len(upstream_a.chat_requests) == 1
        assert len(upstream_b.chat_requests) == 1
        _wait_for_generation_close(server, generation_id=1)

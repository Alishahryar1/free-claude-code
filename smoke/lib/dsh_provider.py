"""A deterministic local model for native DSH protocol and title smoke tests."""

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal

from free_claude_code.core.json_types import JsonObject


@dataclass
class DshProvider:
    model: str
    marker: str
    failure: Literal["none", "main", "title"] = "none"
    read_path: str | None = None
    tool_call: tuple[str, JsonObject] | None = None
    protocol: Literal["chat", "responses"] = "chat"
    hold_release: threading.Event | None = None
    hold_started: threading.Event = field(default_factory=threading.Event)
    requests: list[JsonObject] = field(default_factory=list)
    title_started: threading.Event = field(default_factory=threading.Event)
    title_finished: threading.Event = field(default_factory=threading.Event)

    def purpose(self, body: JsonObject) -> str:
        messages = body.get("messages", body.get("input"))
        if str(body.get("instructions", "")).startswith(
            "Create a concise title for an AI coding-assistant session"
        ):
            return "title"
        if isinstance(messages, list) and messages:
            if "You are now acting as a compaction engine" in str(messages[-1]):
                return "compaction"
            first = messages[0]
            if isinstance(
                first, dict
            ) and "Create a concise title for an AI coding-assistant session" in str(
                first.get("content", "")
            ):
                assert body.get("max_tokens", body.get("max_output_tokens")) == 64
                assert not body.get("tools")
                return "title"
            if any(
                isinstance(message, dict)
                and message.get("role") == "user"
                and "FCC_CHILD_TASK" in str(message.get("content", ""))
                for message in messages
            ):
                return "subagent"
        return "main"


@contextmanager
def dsh_provider(scenario: DshProvider) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self.json_response(
                    200, {"data": [{"id": scenario.model, "object": "model"}]}
                )
            elif self.path == "/api/v0/models":
                self.json_response(200, {"data": []})
            elif self.path.startswith("/api/v1/models?"):
                self.json_response(
                    200,
                    {
                        "success": True,
                        "output": {
                            "total": 1,
                            "page_no": 1,
                            "models": [
                                {
                                    "model": scenario.model,
                                    "features": ["function-calling"],
                                    "capabilities": ["TG", "VU"],
                                    "inference_metadata": {
                                        "request_modality": ["TEXT", "IMAGE"]
                                    },
                                    "model_info": {
                                        "context_window": 200000,
                                        "max_output_tokens": 4096,
                                    },
                                }
                            ],
                        },
                    },
                )
            else:
                self.json_response(404, {"error": "not found"})

        def do_POST(self) -> None:
            body = json.loads(
                self.rfile.read(int(self.headers.get("content-length", "0")))
            )
            purpose = scenario.purpose(body)
            scenario.requests.append(
                {"path": self.path, "purpose": purpose, "body": body}
            )
            try:
                if purpose == "title":
                    scenario.title_started.set()
                elif purpose == "main" and not scenario.title_started.wait(10):
                    self.json_response(
                        500,
                        {"error": {"message": "Native title request did not arrive"}},
                    )
                    return
                if scenario.failure == purpose:
                    self.json_response(
                        400,
                        {
                            "error": {
                                "type": "invalid_request_error",
                                "message": "DSH fixture rejected this request",
                            }
                        },
                    )
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("connection", "close")
                self.end_headers()
                if scenario.protocol == "responses":
                    self.responses_stream(
                        "Local fixture task" if purpose == "title" else scenario.marker
                    )
                elif purpose == "title":
                    self.chunk({"role": "assistant", "content": "Local fixture task"})
                    self.chunk({}, "stop")
                elif purpose == "subagent":
                    self.chunk({"role": "assistant", "content": "FCC_CHILD_DONE"})
                    self.chunk({}, "stop")
                elif purpose == "compaction":
                    self.chunk(
                        {
                            "role": "assistant",
                            "content": "## Current Work\nThe fixture task succeeded. Continue the local smoke check.",
                        }
                    )
                    self.chunk({}, "stop")
                elif (scenario.read_path or scenario.tool_call) and not any(
                    message.get("role") == "tool" for message in body["messages"]
                ):
                    self.chunk(
                        {
                            "role": "assistant",
                            "reasoning_content": "Reading the local fixture.",
                        }
                    )
                    self.chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_fixture_read",
                                    "type": "function",
                                    "function": {
                                        "name": scenario.tool_call[0]
                                        if scenario.tool_call
                                        else "read",
                                        "arguments": json.dumps(
                                            scenario.tool_call[1]
                                            if scenario.tool_call
                                            else {"file_path": scenario.read_path}
                                        ),
                                    },
                                }
                            ]
                        }
                    )
                    self.chunk({}, "tool_calls")
                else:
                    self.chunk({"role": "assistant", "content": scenario.marker})
                    if scenario.hold_release is not None:
                        scenario.hold_started.set()
                        scenario.hold_release.wait(30)
                    self.chunk({}, "stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError, ConnectionResetError, ConnectionAbortedError:
                # DSH can cancel title work while closing a one-shot runtime.
                pass
            finally:
                if purpose == "title":
                    scenario.title_finished.set()
                self.close_connection = True

        def responses_stream(self, text: str) -> None:
            message = {
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
            response = {
                "id": "resp_fixture",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": scenario.model,
                "output": [message],
                "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            }
            events = [
                {
                    "type": "response.created",
                    "response": {**response, "status": "in_progress", "output": []},
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**message, "status": "in_progress", "content": []},
                },
                {
                    "type": "response.content_part.added",
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                },
                {
                    "type": "response.output_text.done",
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                },
                {
                    "type": "response.content_part.done",
                    "item_id": "msg_fixture",
                    "output_index": 0,
                    "content_index": 0,
                    "part": message["content"][0],
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": message,
                },
                {"type": "response.completed", "response": response},
            ]
            for sequence, event in enumerate(events):
                self.wfile.write(
                    f"event: {event['type']}\ndata: {json.dumps(event | {'sequence_number': sequence})}\n\n".encode()
                )
                self.wfile.flush()

        def chunk(self, delta: JsonObject, finish: str | None = None) -> None:
            payload = {
                "id": "chatcmpl-dsh-fixture",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": scenario.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        def json_response(self, status: int, value: object) -> None:
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

"""Reusable product E2E smoke drivers."""

import asyncio
import json
import os
import subprocess
import time
import uuid
import wave
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from free_claude_code.config.provider_catalog import SUPPORTED_PROVIDER_IDS
from free_claude_code.core.anthropic.stream_contracts import (
    SSEEvent,
    assert_anthropic_stream_contract,
    event_index,
    has_tool_use,
    parse_sse_lines,
    text_content,
)
from free_claude_code.harnesses.claude import build_claude_proxy_env
from free_claude_code.messaging.models import IncomingMessage, MessageScope
from free_claude_code.messaging.session import SessionStore
from free_claude_code.messaging.voice import VoiceCancellationResult
from free_claude_code.messaging.workflow import MessagingWorkflow
from smoke.lib.child_process import run_captured_text
from smoke.lib.config import ProviderModel, SmokeConfig, auth_headers
from smoke.lib.http import conversation_headers
from smoke.lib.server import RunningServer, start_server
from smoke.lib.skips import fail_missing_env, skip_if_upstream_unavailable_events


@dataclass(slots=True)
class ConversationTurn:
    request: dict[str, Any]
    events: list[SSEEvent]

    @property
    def assistant_content(self) -> list[dict[str, Any]]:
        return assistant_content_from_events(self.events)

    @property
    def text(self) -> str:
        return text_content(self.events)


class SmokeServerDriver:
    """Start a local proxy server for a product scenario."""

    def __init__(
        self,
        config: SmokeConfig,
        *,
        name: str,
        env_overrides: dict[str, str] | None = None,
        env_unset: Iterable[str] = (),
        command: list[str] | None = None,
    ) -> None:
        self.config = config
        self.name = name
        self.env_overrides = env_overrides
        self.env_unset = tuple(env_unset)
        self.command = command

    @contextmanager
    def run(self) -> Iterator[RunningServer]:
        with start_server(
            self.config,
            env_overrides=self.env_overrides,
            env_unset=self.env_unset,
            command=self.command,
            name=self.name,
        ) as server:
            yield server


class ConversationDriver:
    """Drive multi-turn Anthropic-compatible conversations through the server."""

    def __init__(self, server: RunningServer, config: SmokeConfig) -> None:
        self.server = server
        self.config = config
        self.messages: list[dict[str, Any]] = []
        self.turns: list[ConversationTurn] = []
        self._session_id = str(uuid.uuid4())

    def ask(
        self,
        text: str,
        *,
        model: str = "fcc-smoke-default",
        max_tokens: int = 256,
        extra: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        append_assistant: bool = True,
    ) -> ConversationTurn:
        self.messages.append({"role": "user", "content": text})
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": list(self.messages),
        }
        if extra:
            payload.update(extra)
        turn = self.stream(payload, headers=headers)
        if append_assistant:
            self.messages.append(
                {"role": "assistant", "content": turn.assistant_content or turn.text}
            )
        return turn

    def stream(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
    ) -> ConversationTurn:
        request_headers = conversation_headers(headers, session_id=self._session_id)
        stream_payload = {**payload, "stream": True}
        with httpx.stream(
            "POST",
            f"{self.server.base_url}/v1/messages",
            headers=request_headers,
            json=stream_payload,
            timeout=self.config.timeout_s,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                raise AssertionError(
                    f"stream request failed: HTTP {response.status_code} {body[:1000]}"
                )
            events = parse_sse_lines(response.iter_lines())
        assert_anthropic_stream_contract(events)
        turn = ConversationTurn(stream_payload, events)
        self.turns.append(turn)
        return turn

    def stream_expect_http_error(
        self,
        payload: dict[str, Any],
        *,
        expected_status: int,
    ) -> dict[str, Any]:
        response = httpx.post(
            f"{self.server.base_url}/v1/messages",
            headers=conversation_headers(session_id=self._session_id),
            json=payload,
            timeout=self.config.timeout_s,
        )
        assert response.status_code == expected_status, response.text
        return response.json()


class ProviderMatrixDriver:
    """Resolve provider models and enforce matrix semantics for product smoke."""

    ALL_PROVIDERS: tuple[str, ...] = SUPPORTED_PROVIDER_IDS

    def __init__(self, config: SmokeConfig) -> None:
        self.config = config

    def configured_models(self) -> list[ProviderModel]:
        return self.config.provider_models()

    def provider_smoke_models(self) -> list[ProviderModel]:
        selected = self.config.provider_matrix
        missing_selected = [
            provider
            for provider in selected
            if provider in self.ALL_PROVIDERS
            and not self.config.has_provider_configuration(provider)
        ]
        if missing_selected:
            fail_missing_env(
                "selected providers are not configured: "
                + ", ".join(sorted(missing_selected))
            )

        models = self.config.provider_smoke_models()
        if not models and os.getenv("FCC_ALLOW_NO_PROVIDER_SMOKE") != "1":
            fail_missing_env(
                "no configured provider smoke models; set FCC_ALLOW_NO_PROVIDER_SMOKE=1 "
                "only for no-provider smoke collection"
            )
        return models

    def first_model(self) -> ProviderModel:
        models = self.provider_smoke_models()
        if not models:
            pytest.skip("missing_env: no configured provider model")
        return models[0]


class ClientProtocolDriver:
    """Build recorded/representative client protocol requests."""

    @staticmethod
    def vscode_headers() -> dict[str, str]:
        headers = auth_headers()
        headers.update(
            {
                "anthropic-beta": "messages-2023-12-15",
                "user-agent": "Claude-Code-VSCode product smoke",
            }
        )
        return headers

    @staticmethod
    def jetbrains_headers() -> dict[str, str]:
        headers = auth_headers()
        headers["user-agent"] = "JetBrains-ACP product smoke"
        return headers

    @staticmethod
    def adaptive_thinking_payload() -> dict[str, Any]:
        return {
            "model": "claude-opus-4-7",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "unsigned thought"},
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "Hello."},
                    ],
                },
                {"role": "user", "content": "Reply with exactly FCC_SMOKE_CLIENT"},
            ],
            "thinking": {"type": "adaptive", "budget_tokens": 1024},
        }

    @staticmethod
    def tool_result_payload() -> dict[str, Any]:
        return {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "Use echo_smoke once."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_client_smoke",
                            "name": "echo_smoke",
                            "input": {"value": "FCC_SMOKE_CLIENT"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_client_smoke",
                            "content": "FCC_SMOKE_CLIENT",
                        }
                    ],
                },
            ],
            "tools": [echo_tool_schema()],
            "thinking": {"type": "adaptive"},
        }

    @staticmethod
    def run_claude_prompt(
        *,
        claude_bin: str,
        server: RunningServer,
        config: SmokeConfig,
        cwd: Path,
        prompt: str,
        model: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = build_claude_proxy_env(
            proxy_root_url=server.base_url,
            auth_token=config.settings.proxy_auth_token,
            base_env=os.environ,
        )
        command = [
            claude_bin,
            "--bare",
            "--no-session-persistence",
            "--tools",
            "",
            "--system-prompt",
            "Reply with exactly the requested smoke token and no other text.",
        ]
        if model is not None:
            command.extend(["--model", model])
        command.extend(["-p", prompt])
        return run_captured_text(
            command,
            cwd=cwd,
            env=env,
            timeout=config.timeout_s,
            check=False,
        )


class FakePlatform:
    """In-memory platform that exercises the real message handler."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.handler: Callable[[IncomingMessage], Awaitable[None]] | None = None
        self.sent: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []
        self._counter = 0
        self._tasks: list[asyncio.Future[Any]] = []
        self._pending_voice: dict[
            tuple[MessageScope, str], VoiceCancellationResult
        ] = {}

    async def start(self) -> None:
        return None

    async def quiesce(self) -> None:
        return None

    async def close(self) -> None:
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    @property
    def is_connected(self) -> bool:
        return True

    def on_message(self, handler: Callable[[IncomingMessage], Awaitable[None]]) -> None:
        self.handler = handler

    def continue_message_sequence_after(self, previous: FakePlatform) -> None:
        """Model platform-owned message IDs surviving an FCC restart."""
        self._counter = previous._counter

    async def emit(self, incoming: IncomingMessage) -> None:
        assert self.handler is not None
        await self.handler(incoming)

    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        self._tasks.append(asyncio.ensure_future(task))

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: str | None = None,
    ) -> str:
        self._counter += 1
        message_id = f"{self.name}_msg_{self._counter}"
        self.sent.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "reply_to": reply_to,
                "parse_mode": parse_mode,
                "message_thread_id": message_thread_id,
            }
        )
        return message_id

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
    ) -> None:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": parse_mode,
            }
        )

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        self.deletes.append({"chat_id": chat_id, "message_id": message_id})

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None:
        message_id = await self.send_message(
            chat_id,
            text,
            reply_to=reply_to,
            parse_mode=parse_mode,
            message_thread_id=message_thread_id,
        )
        return None if fire_and_forget else message_id

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
    ) -> None:
        await self.edit_message(chat_id, message_id, text, parse_mode=parse_mode)

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        fire_and_forget: bool = True,
    ) -> None:
        for message_id in message_ids:
            await self.delete_message(chat_id, message_id)

    def seed_pending_voice(
        self, chat_id: str, voice_message_id: str, status_message_id: str
    ) -> None:
        scope = MessageScope(platform=self.name, chat_id=chat_id)
        result = VoiceCancellationResult(
            scope=scope,
            voice_message_id=voice_message_id,
            status_message_id=status_message_id,
            delete_message_ids=frozenset({voice_message_id, status_message_id}),
        )
        self._pending_voice[(scope, voice_message_id)] = result
        self._pending_voice[(scope, status_message_id)] = result

    async def cancel_pending_voice(
        self, scope: MessageScope, reply_id: str
    ) -> VoiceCancellationResult | None:
        result = self._pending_voice.get((scope, reply_id))
        if result is None:
            return None
        self._pending_voice.pop((scope, result.voice_message_id), None)
        if result.status_message_id is not None:
            self._pending_voice.pop((scope, result.status_message_id), None)
        delete_message_ids = {result.voice_message_id, result.status_message_id}
        if reply_id == result.status_message_id:
            delete_message_ids = {result.status_message_id}
        return VoiceCancellationResult(
            scope=result.scope,
            voice_message_id=result.voice_message_id,
            status_message_id=result.status_message_id,
            delete_message_ids=frozenset(
                message_id
                for message_id in delete_message_ids
                if message_id is not None
            ),
        )

    async def cancel_all_pending_voices(
        self,
    ) -> tuple[VoiceCancellationResult, ...]:
        results = tuple(
            {
                (result.scope, result.voice_message_id): result
                for result in self._pending_voice.values()
            }.values()
        )
        self._pending_voice.clear()
        return results

    async def cancel_pending_voices_in_scope(
        self,
        scope: MessageScope,
    ) -> tuple[VoiceCancellationResult, ...]:
        results = tuple(
            {
                result.voice_message_id: result
                for (entry_scope, _reference_id), result in self._pending_voice.items()
                if entry_scope == scope
            }.values()
        )
        for result in results:
            self._pending_voice.pop((scope, result.voice_message_id), None)
            if result.status_message_id is not None:
                self._pending_voice.pop((scope, result.status_message_id), None)
        return results

    @property
    def pending_voice_count(self) -> int:
        return len(
            {
                (result.scope, result.voice_message_id)
                for result in self._pending_voice.values()
            }
        )


class FakeCLISession:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []
        self.is_busy = False

    async def start_task(
        self, prompt: str, session_id: str | None = None, fork_session: bool = False
    ) -> AsyncGenerator[dict[str, Any]]:
        self.calls.append(
            {"prompt": prompt, "session_id": session_id, "fork_session": fork_session}
        )
        self.is_busy = True
        try:
            for event in self.events:
                await asyncio.sleep(0)
                yield event
        finally:
            self.is_busy = False


class FakeCLIManager:
    def __init__(self, event_batches: list[list[dict[str, Any]]] | None = None) -> None:
        self.event_batches = event_batches or [default_cli_events("fake_session_1")]
        self.sessions: list[FakeCLISession] = []
        self.registered: list[tuple[str, str]] = []
        self.removed: list[str] = []
        self.stopped = False

    async def get_or_create_session(
        self, session_id: str | None = None
    ) -> tuple[FakeCLISession, str, bool]:
        index = len(self.sessions)
        events = self.event_batches[min(index, len(self.event_batches) - 1)]
        session = FakeCLISession(events)
        self.sessions.append(session)
        return session, session_id or f"pending_{index}", session_id is None

    async def register_real_session_id(
        self, temp_id: str, real_session_id: str
    ) -> bool:
        self.registered.append((temp_id, real_session_id))
        return True

    async def stop_all(self) -> None:
        self.stopped = True

    async def remove_session(self, session_id: str) -> bool:
        self.removed.append(session_id)
        return True

    def get_stats(self) -> dict[str, int]:
        return {"active_sessions": len(self.sessions), "pending_sessions": 0}


@dataclass(slots=True)
class FakePlatformDriver:
    platform_name: str
    tmp_path: Path
    event_batches: list[list[dict[str, Any]]] | None = None
    platform: FakePlatform = field(init=False)
    cli_manager: FakeCLIManager = field(init=False)
    session_store: SessionStore = field(init=False)
    workflow: MessagingWorkflow = field(init=False)

    def __post_init__(self) -> None:
        self.platform = FakePlatform(self.platform_name)
        self.cli_manager = FakeCLIManager(self.event_batches)
        self.session_store = SessionStore(
            storage_path=str(self.tmp_path / f"{self.platform_name}-sessions.json")
        )
        self.workflow = MessagingWorkflow(
            self.platform,
            self.cli_manager,
            self.session_store,
            platform_name=self.platform_name,
            voice_cancellation=self.platform,
        )
        self.platform.on_message(self.workflow.handle_message)

    async def send(
        self,
        text: str,
        *,
        chat_id: str = "chat_1",
        message_id: str | None = None,
        reply_to: str | None = None,
    ) -> IncomingMessage:
        incoming = await self.emit(
            text,
            chat_id=chat_id,
            message_id=message_id,
            reply_to=reply_to,
        )
        await self.wait_for_idle()
        return incoming

    async def emit(
        self,
        text: str,
        *,
        chat_id: str = "chat_1",
        message_id: str | None = None,
        reply_to: str | None = None,
    ) -> IncomingMessage:
        """Deliver a message without waiting for background claims to finish."""
        incoming = IncomingMessage(
            text=text,
            chat_id=chat_id,
            user_id="user_1",
            message_id=message_id or f"in_{uuid.uuid4().hex[:8]}",
            platform=self.platform_name,
            reply_to_message_id=reply_to,
        )
        await self.platform.emit(incoming)
        return incoming

    async def wait_for_idle(self, *, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pending = [task for task in self.platform._tasks if not task.done()]
            if not pending and await self._all_tree_nodes_terminal():
                self.session_store.flush_pending_save()
                return
            await asyncio.sleep(0.02)
        raise AssertionError("fake platform did not become idle")

    async def _all_tree_nodes_terminal(self) -> bool:
        snapshot = await self.workflow.tree_queue.snapshot()
        for tree in snapshot.trees.values():
            for node in tree.nodes.values():
                if node.get("state") in {"pending", "in_progress"}:
                    return False
        return True


class VoiceFixtureDriver:
    @staticmethod
    def write_tone_wav(path: Path) -> None:
        import math

        sample_rate = 16000
        duration_s = 0.25
        amplitude = 8000
        frames = bytearray()
        for i in range(int(sample_rate * duration_s)):
            sample = int(amplitude * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames.extend(sample.to_bytes(2, byteorder="little", signed=True))
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(bytes(frames))


def echo_tool_schema() -> dict[str, Any]:
    return {
        "name": "echo_smoke",
        "description": "Echo a test value.",
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }


def assistant_content_from_events(events: list[SSEEvent]) -> list[dict[str, Any]]:
    blocks: dict[int, dict[str, Any]] = {}
    block_order: list[int] = []
    for event in events:
        if event.event == "content_block_start":
            index = event_index(event)
            block = event.data.get("content_block", {})
            if isinstance(block, dict):
                blocks[index] = dict(block)
                block_order.append(index)
            continue
        if event.event == "content_block_delta":
            index = event_index(event)
            block = blocks.get(index)
            delta = event.data.get("delta", {})
            if not isinstance(block, dict) or not isinstance(delta, dict):
                continue
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = str(block.get("text", "")) + str(delta.get("text", ""))
            elif delta_type == "thinking_delta":
                block["thinking"] = str(block.get("thinking", "")) + str(
                    delta.get("thinking", "")
                )
            elif delta_type == "input_json_delta":
                block["_partial_json"] = str(block.get("_partial_json", "")) + str(
                    delta.get("partial_json", "")
                )

    content: list[dict[str, Any]] = []
    for index in block_order:
        block = blocks[index]
        if block.get("type") == "tool_use":
            partial = str(block.pop("_partial_json", ""))
            if partial:
                try:
                    block["input"] = json.loads(partial)
                except json.JSONDecodeError:
                    block["input"] = {}
        content.append(block)
    return content


def tool_use_blocks(events: list[SSEEvent]) -> list[dict[str, Any]]:
    return [
        block
        for block in assistant_content_from_events(events)
        if block.get("type") == "tool_use"
    ]


def default_cli_events(session_id: str) -> list[dict[str, Any]]:
    return [
        {"type": "session_info", "session_id": session_id},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "Inspect the request."},
                    {
                        "type": "tool_use",
                        "id": "toolu_fake",
                        "name": "Read",
                        "input": {"file_path": "README.md"},
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_fake",
                        "content": "Free Claude Code",
                    },
                    {"type": "text", "text": "Fake platform answer."},
                ]
            },
        },
        {"type": "exit", "code": 0, "stderr": None},
    ]


def assert_product_stream(events: list[SSEEvent]) -> None:
    assert_anthropic_stream_contract(events)
    assert text_content(events).strip() or has_tool_use(events), (
        "product stream emitted neither text nor tool_use"
    )


def assert_native_thinking_stream(events: list[SSEEvent], *, context: str) -> None:
    skip_if_upstream_unavailable_events(events)
    assert_anthropic_stream_contract(events)
    assert any(
        event.event == "content_block_delta"
        and isinstance(delta := event.data.get("delta"), dict)
        and delta.get("type") == "thinking_delta"
        for event in events
    ), f"{context} completed without native thinking output"

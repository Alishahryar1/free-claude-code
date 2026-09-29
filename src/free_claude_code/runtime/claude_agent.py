"""Installed Claude Code execution through the public Python Agent SDK."""

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from free_claude_code.application.code_sessions.models import (
    CodeCatalog,
    CodeConflictError,
    CodeMode,
    CodeModel,
    CodeModeOption,
    CodeUnavailableError,
    CodeValidationError,
    HarnessCapabilities,
    HarnessEvent,
    HarnessId,
    NativeHistoryMissing,
    NativeThread,
)
from free_claude_code.application.code_sessions.ports import EventSink, HarnessSelection
from free_claude_code.application.model_catalog import read_model_catalog
from free_claude_code.application.ports import RequestRuntimePort
from free_claude_code.cli.process_registry import register_pid, unregister_pid
from free_claude_code.config.model_refs import split_provider_model_ref
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.core.async_tasks import run_sync_owned
from free_claude_code.core.gateway_model_ids import (
    gateway_model_id,
    no_thinking_gateway_model_id,
)
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses.claude import build_claude_proxy_env

CLAUDE_MODES = tuple(
    CodeModeOption(id=mode, name=name)
    for mode, name in (
        ("config", "Use config"),
        ("default", "Manual"),
        ("acceptEdits", "Accept edits"),
        ("plan", "Plan"),
        ("dontAsk", "Don't ask"),
        ("bypassPermissions", "Bypass permissions"),
    )
)
_EFFORTS = ("low", "medium", "high", "xhigh", "max")
_INSTALL = "Install Claude Code with: npm install -g @anthropic-ai/claude-code"


class ClaudeHarnessFactory:
    id: HarnessId = "claude"
    name = "Claude Code"
    prepare_on_open = True
    modes: tuple[CodeModeOption, ...] = CLAUDE_MODES

    def __init__(
        self,
        runtime: RequestRuntimePort,
        *,
        binary: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._runtime, self._binary, self._env = runtime, binary, env

    def availability(self) -> tuple[bool, str | None]:
        return (
            (True, None)
            if self._binary or shutil.which("claude")
            else (False, "Claude Code is not installed. " + _INSTALL)
        )

    def catalog(self) -> CodeCatalog:
        settings = self._runtime.current_settings()
        catalog = read_model_catalog(self._runtime, settings)
        models = []
        for model in catalog.models:
            provider, name = split_provider_model_ref(model.provider_model_ref)
            models.append(
                CodeModel(
                    id=model.provider_model_ref,
                    display_name=model.display_name,
                    provider_id=provider,
                    model_name=name,
                    context_window_tokens=model.context_window_tokens,
                )
            )
        return CodeCatalog(settings.model, tuple(models))

    async def prepare(
        self, model: str, reasoning_effort: str | None, mode: CodeMode
    ) -> _ClaudeSelection:
        binary = self._binary or shutil.which("claude")
        if binary is None:
            raise CodeUnavailableError("Claude Code is not installed. " + _INSTALL)
        snapshot = await self._runtime.wait_for_catalog()
        settings = snapshot.settings
        entry = next(
            (
                entry
                for entry in read_model_catalog(snapshot, settings).models
                if entry.provider_model_ref == model
            ),
            None,
        )
        if entry is None:
            raise CodeValidationError(
                "This model is unavailable. Choose another model."
            )
        if mode not in {option.id for option in CLAUDE_MODES} | {"auto"}:
            raise CodeValidationError(
                "This permission mode is unavailable for Claude Code."
            )
        if reasoning_effort is not None and reasoning_effort not in _EFFORTS:
            raise CodeValidationError("This effort is unavailable for Claude Code.")
        env = build_claude_proxy_env(
            proxy_root_url=local_proxy_root_url(settings),
            auth_token=settings.proxy_auth_token,
            base_env=self._env if self._env is not None else os.environ,
        )
        wire_model = (
            no_thinking_gateway_model_id(model)
            if entry.supports_reasoning is False
            else gateway_model_id(model)
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "env": env,
                    "binary": binary,
                    "model": wire_model,
                    "effort": reasoning_effort,
                    "mode": mode,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return _ClaudeSelection(
            binary, env, wire_model, model, reasoning_effort, mode, fingerprint
        )

    async def open_history(self, cwd: str, sink: EventSink) -> ClaudeHistory:
        return ClaudeHistory(cwd)


def _check_version(binary: str) -> None:
    result = subprocess.run(
        [binary, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
    if (
        result.returncode
        or match is None
        or tuple(map(int, match.groups())) < (2, 1, 283)
    ):
        raise CodeUnavailableError(
            "Update Claude Code to 2.1.283 or later for browser Code sessions."
        )


@dataclass(frozen=True)
class _ClaudeSelection:
    binary: str
    env: Mapping[str, str] = field(repr=False)
    wire_model: str
    model: str
    reasoning_effort: str | None
    mode: CodeMode
    configuration_key: str

    async def open(self, cwd: str, sink: EventSink) -> ClaudeConnection:
        await run_sync_owned(lambda: _check_version(self.binary))
        return ClaudeConnection(cwd, self, sink)


class ClaudeHistory:
    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self._submitted: frozenset[str] = frozenset()

    async def read_thread(self, thread_id: str) -> NativeThread:
        from claude_agent_sdk import get_session_info, get_session_messages

        from .claude_protocol import ClaudeProtocol

        def read():
            if get_session_info(thread_id, directory=self.cwd) is None:
                raise NativeHistoryMissing(
                    "Claude Code's native session is missing. Saved FCC history was retained."
                )
            return ClaudeProtocol.history(
                thread_id,
                get_session_messages(thread_id, directory=self.cwd),
                self._submitted,
            )

        return await run_sync_owned(read)

    async def delete_thread(self, thread_id: str) -> None:
        from claude_agent_sdk import delete_session

        def delete():
            try:
                delete_session(thread_id, directory=self.cwd)
            except FileNotFoundError:
                raise NativeHistoryMissing(
                    "Claude Code's native session was already removed."
                ) from None

        await run_sync_owned(delete)

    async def close(self) -> None:
        pass


class ClaudeConnection(ClaudeHistory):
    def __init__(self, cwd: str, selection: _ClaudeSelection, sink: EventSink) -> None:
        from claude_agent_sdk import ClaudeSDKClient

        from .claude_protocol import ClaudePrompt, ClaudeProtocol

        super().__init__(cwd)
        self.selection, self.sink = selection, sink
        self.generation = str(uuid.uuid4())
        self.thread_id: str | None = None
        self._client: ClaudeSDKClient | None = None
        self._protocol: ClaudeProtocol | None = None
        self._reader: asyncio.Task[None] | None = None
        self._closing: asyncio.Task[None] | None = None
        self._events = asyncio.Lock()
        self._prompts: dict[str, tuple[ClaudePrompt, asyncio.Future[JsonObject]]] = {}
        self._native_models: list[JsonObject] = []
        self._pid: int | None = None

    def supports(self, selection: HarnessSelection) -> bool:
        return (
            self._client is not None
            and self._closing is None
            and (self._reader is None or not self._reader.done())
            and selection.configuration_key == self.selection.configuration_key
        )

    def capabilities(self, selection: HarnessSelection) -> HarnessCapabilities:
        native = next(
            (
                model
                for model in self._native_models
                if model.get("value") == self.selection.wire_model
            ),
            {},
        )
        efforts = native.get("supportedEffortLevels")
        modes = CLAUDE_MODES
        if native.get("supportsAutoMode") is True:
            modes += (CodeModeOption(id="auto", name="Auto"),)
        return HarnessCapabilities(
            generation=self.generation,
            model=selection.model,
            configuration_key=selection.configuration_key,
            modes=modes,
            reasoning_efforts=tuple(
                value
                for value in efforts
                if isinstance(value, str) and value in _EFFORTS
            )
            if isinstance(efforts, list) and native.get("supportsEffort") is True
            else (),
        )

    async def _connect(self, thread_id: str, *, resume: bool) -> None:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        from claude_agent_sdk.types import EffortLevel, PermissionMode

        from .claude_protocol import ClaudeProtocol

        selected = self.selection
        self.thread_id = thread_id
        self._protocol = ClaudeProtocol(self.generation, thread_id)
        options = ClaudeAgentOptions(
            cli_path=selected.binary,
            cwd=self.cwd,
            env=dict(selected.env),
            model=selected.wire_model,
            effort=cast(EffortLevel | None, selected.reasoning_effort),
            permission_mode=None
            if selected.mode == "config"
            else cast(PermissionMode, selected.mode),
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["user", "project", "local"],
            include_partial_messages=True,
            can_use_tool=self._can_use_tool,
            extra_args={"replay-user-messages": None},
            session_id=None if resume else thread_id,
            resume=thread_id if resume else None,
        )
        self._client = ClaudeSDKClient(options=options)
        try:
            await self._client.connect()
            info = await self._client.get_server_info() or {}
            if type(info.get("pid")) is int:
                self._pid = info["pid"]
                register_pid(self._pid)
            models = info.get("models")
            if isinstance(models, list):
                self._native_models = [
                    model for model in models if isinstance(model, dict)
                ]
            capabilities = self.capabilities(selected)
            if (
                selected.reasoning_effort is not None
                and selected.reasoning_effort not in capabilities.reasoning_efforts
            ):
                raise CodeValidationError(
                    "Claude did not report support for the selected effort. Choose Native/default."
                )
            if selected.mode not in {option.id for option in capabilities.modes}:
                raise CodeValidationError(
                    "Claude did not report support for the selected permission mode."
                )
            self._reader = asyncio.create_task(self._read(), name="fcc-claude-events")
        except BaseException:
            await self.close()
            raise

    async def create_thread(self) -> NativeThread:
        identity = str(uuid.uuid4())
        await self._connect(identity, resume=False)
        return NativeThread(identity)

    async def resume_thread(
        self, thread_id: str, *, submitted_run_ids: frozenset[str] = frozenset()
    ) -> NativeThread:
        self._submitted = submitted_run_ids
        history = await self.read_thread(thread_id)
        await self._connect(thread_id, resume=True)
        return history

    async def start_turn(
        self,
        text: str,
        selection: HarnessSelection,
        client_id: str,
        permission_defaults: JsonObject | None,
    ) -> str:
        if (
            not self.supports(selection)
            or self._protocol is None
            or self._client is None
            or self.thread_id is None
        ):
            raise CodeUnavailableError(
                "Claude's session is not ready. Input was not sent."
            )
        async with self._events:
            self._protocol.begin(client_id)

        async def prompt():
            yield {
                "type": "user",
                "uuid": client_id,
                "session_id": self.thread_id,
                "origin": {"kind": "human"},
                "message": {"role": "user", "content": text},
            }

        await self._client.query(prompt(), session_id=self.thread_id)
        return client_id

    async def interrupt(self, turn_id: str) -> None:
        if (
            self._client is not None
            and self._protocol is not None
            and self._protocol.pending == turn_id
        ):
            self._protocol.interrupted = True
            await self._client.interrupt()

    async def _read(self) -> None:
        assert self._client is not None and self._protocol is not None
        message = "Claude Code's process ended."
        try:
            async for native in self._client.receive_messages():
                async with self._events:
                    for event in self._protocol.feed(native):
                        await self.sink(event)
                    if self._protocol.unattributed_result:
                        message = "Claude's result could not be matched to its submitted input. Input was not resent."
                        break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = str(exc) or type(exc).__name__
        finally:
            if self._closing is None:
                # Release SDK resources before publishing a detached connection.
                try:
                    await self._client.disconnect()
                except Exception as exc:
                    message = f"{message} Cleanup failed: {exc}"
                finally:
                    if self._pid is not None:
                        unregister_pid(self._pid)
                await self.sink(
                    HarnessEvent(
                        self.generation, self.thread_id, "closed", message=message
                    )
                )

    async def _can_use_tool(self, name, inputs, context):
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        from .claude_protocol import ClaudePrompt

        assert self._protocol is not None
        identity = context.tool_use_id
        async with self._events:
            turn = self._protocol.turn_for_tool(identity or "")
            if turn is None or not identity or self._closing is not None:
                return PermissionResultDeny(
                    message="The tool request could not be matched to a live FCC turn."
                )
            prompt = ClaudePrompt(identity, turn, name, inputs)
            request = prompt.request()
            future: asyncio.Future[JsonObject] = (
                asyncio.get_running_loop().create_future()
            )
            self._prompts[identity] = prompt, future
            try:
                await self.sink(
                    self._protocol.event("prompt", turn_id=turn, prompt=request)
                )
            except BaseException:
                self._prompts.pop(identity, None)
                future.cancel()
                raise
        try:
            response = await future
            if response["allow"]:
                return PermissionResultAllow(
                    updated_input=dict(cast(JsonObject, response["updated_input"]))
                )
            return PermissionResultDeny(message="The user declined this tool request.")
        finally:
            self._prompts.pop(identity, None)
            async with self._events:
                await self.sink(self._protocol.event("resolved", request_id=identity))

    def prepare_answer(self, request_id: str | int, answer: JsonObject) -> JsonObject:
        pending = self._prompts.get(str(request_id))
        if pending is None or pending[1].done():
            raise CodeConflictError("This Claude prompt is no longer active.")
        return pending[0].answer(answer)

    async def respond(self, request_id: str | int, response: JsonObject) -> None:
        pending = self._prompts.get(str(request_id))
        if pending is None or pending[1].done():
            raise CodeConflictError("This Claude prompt is no longer active.")
        pending[1].set_result(response)

    async def close(self) -> None:
        if self._closing is None:
            self._closing = asyncio.create_task(self._close(), name="fcc-claude-close")
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._closing)
                break
            except asyncio.CancelledError:
                if self._closing.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _close(self) -> None:
        for _, future in tuple(self._prompts.values()):
            future.cancel()
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
        try:
            if self._client is not None:
                await self._client.disconnect()
        finally:
            if self._pid is not None:
                unregister_pid(self._pid)
            self._prompts.clear()

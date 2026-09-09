"""Owned Codex app-server processes over bidirectional stdio JSONL."""

import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import uuid
from collections.abc import Mapping
from contextlib import ExitStack, suppress
from dataclasses import dataclass, replace

from free_claude_code.application.code_sessions.models import (
    CodeCatalog,
    CodeConflictError,
    CodeMode,
    CodeModel,
    CodeUnavailableError,
    CodeValidationError,
    HarnessEvent,
    NativeHistoryMissing,
    NativeThread,
)
from free_claude_code.application.code_sessions.ports import EventSink, HarnessSelection
from free_claude_code.application.ports import RequestRuntimePort
from free_claude_code.cli.launchers.codex import SPEC, prepare_codex_launch
from free_claude_code.cli.launchers.codex_model_catalog import build_codex_model_catalog
from free_claude_code.cli.launchers.resources import LaunchResources
from free_claude_code.cli.launchers.runner import LaunchContext
from free_claude_code.cli.process_registry import register_pid, unregister_pid
from free_claude_code.config.model_refs import split_provider_model_ref
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.version import package_version

from .codex_catalog import current_codex_models
from .codex_protocol import (
    CodexProtocol,
    NativePrompt,
    array_value,
    object_value,
    string_value,
)


class CodexAppServer:
    def __init__(
        self,
        command: list[str],
        env: Mapping[str, str],
        cwd: str,
        sink: EventSink,
        *,
        model_slugs: Mapping[str, str] | None = None,
        fingerprints: Mapping[str, str] | None = None,
        reasoning: Mapping[str, bool] | None = None,
        resources: ExitStack | None = None,
    ) -> None:
        self.generation = str(uuid.uuid4())
        self.thread_id: str | None = None
        self._owned_threads: set[str] = set()
        self._command, self._env, self._cwd, self._sink = command, dict(env), cwd, sink
        self._model_slugs = dict(model_slugs or {})
        self._fingerprints = dict(fingerprints or {})
        self._reasoning = dict(reasoning or {})
        self._resources = resources or ExitStack()
        self._protocol = CodexProtocol(self.generation)
        self._pending: dict[str, tuple[str, asyncio.Future[JsonObject]]] = {}
        self._requests: dict[str | int, NativePrompt] = {}
        self._queue: asyncio.Queue[HarnessEvent | None] = asyncio.Queue()
        self._writer_lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None
        self._dispatcher: asyncio.Task[None] | None = None
        self._closing: asyncio.Task[None] | None = None
        self._reaping: asyncio.Task[None] | None = None
        self._alive = False
        self._stderr_tail = bytearray()

    @property
    def process(self) -> asyncio.subprocess.Process:
        if self._process is None:
            raise CodeUnavailableError("Codex has not started.")
        return self._process

    async def start(self) -> None:
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                cwd=self._cwd,
                env=self._env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            register_pid(self._process.pid)
            self._alive = True
            self._reader = asyncio.create_task(self._read(), name="fcc-codex-jsonl")
            self._stderr = asyncio.create_task(
                self._read_stderr(), name="fcc-codex-stderr"
            )
            self._dispatcher = asyncio.create_task(
                self._dispatch(), name="fcc-codex-events"
            )
            await self.rpc(
                "initialize",
                {
                    "clientInfo": {
                        "name": "free_claude_code",
                        "title": "Free Claude Code",
                        "version": package_version(),
                    },
                    "capabilities": {
                        "experimentalApi": True,
                        "mcpServerOpenaiFormElicitation": False,
                    },
                },
            )
            await self._write({"method": "initialized", "params": {}})
        except BaseException:
            await self.close()
            raise

    def supports(self, selection: HarnessSelection) -> bool:
        return (
            self._alive
            and self._closing is None
            and self._fingerprints.get(selection.model) == selection.configuration_key
        )

    async def create_thread(self) -> NativeThread:
        response = await self.rpc(
            "thread/start", {"cwd": self._cwd, "modelProvider": "fcc"}
        )
        native = self._thread_with_permissions(response)
        self.thread_id = native.id
        return native

    async def resume_thread(self, thread_id: str) -> NativeThread:
        response = await self.rpc(
            "thread/resume",
            {"threadId": thread_id, "cwd": self._cwd, "modelProvider": "fcc"},
        )
        native = self._thread_with_permissions(response)
        self.thread_id = native.id
        return native

    async def read_thread(self, thread_id: str) -> NativeThread:
        return self._protocol.history(
            await self.rpc("thread/read", {"threadId": thread_id, "includeTurns": True})
        )

    def _thread_with_permissions(self, response: JsonObject) -> NativeThread:
        defaults = {
            key: response.get(key)
            for key in (
                "approvalPolicy",
                "approvalsReviewer",
                "activePermissionProfile",
                "sandbox",
            )
        }
        _permission_settings("config", defaults)
        return replace(self._protocol.history(response), permission_defaults=defaults)

    async def start_turn(
        self,
        text: str,
        selection: HarnessSelection,
        client_id: str,
        permission_defaults: JsonObject,
    ) -> str:
        model = self._model_slugs.get(selection.model)
        if not self.thread_id or model is None:
            raise CodeUnavailableError(
                "Codex is not prepared for the selected FCC model."
            )
        params: JsonObject = {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": text}],
            "clientUserMessageId": client_id,
            "model": model,
            **_permission_settings(selection.mode, permission_defaults),
        }
        if self._reasoning.get(selection.model, True):
            off = selection.reasoning_effort == "off"
            params["summary"] = "none" if off else "auto"
            params["effort"] = "none" if off else selection.reasoning_effort
        result = await self.rpc("turn/start", params)
        turn_id = string_value(object_value(result.get("turn")).get("id"))
        if not turn_id:
            raise CodeUnavailableError(
                "Codex did not return a turn ID; input was not resent."
            )
        return turn_id

    async def interrupt(self, turn_id: str) -> None:
        await self.rpc(
            "turn/interrupt", {"threadId": self.thread_id, "turnId": turn_id}
        )

    def prepare_answer(self, request_id: str | int, answer: JsonObject) -> JsonObject:
        prompt = self._requests.get(request_id)
        if prompt is None:
            raise CodeConflictError("This native prompt is no longer active.")
        return prompt.answer(answer)

    async def respond(self, request_id: str | int, response: JsonObject) -> None:
        async with self._writer_lock:
            if not self._alive or request_id not in self._requests:
                raise CodeConflictError("This native prompt is no longer active.")
            self._write_frame({"id": request_id, "result": response})
            self._requests.pop(request_id, None)
            assert self.process.stdin is not None
            await self.process.stdin.drain()

    async def delete_thread(self, thread_id: str) -> None:
        await self.rpc("thread/delete", {"threadId": thread_id})

    async def rpc(self, method: str, params: JsonObject) -> JsonObject:
        if not self._alive:
            raise CodeUnavailableError("The Codex connection is closed.")
        request_id = f"fcc-{uuid.uuid4()}"
        future: asyncio.Future[JsonObject] = asyncio.get_running_loop().create_future()
        # A cancelled caller does not erase the pending native response/identity.
        future.add_done_callback(
            lambda value: value.exception() if not value.cancelled() else None
        )
        self._pending[request_id] = method, future
        try:
            await self._write({"id": request_id, "method": method, "params": params})
            async with asyncio.timeout(30):
                return await asyncio.shield(future)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise CodeUnavailableError(
                "Codex did not acknowledge the request. It was not resent."
            ) from exc

    async def _write(self, message: JsonObject) -> None:
        async with self._writer_lock:
            self._write_frame(message)
            assert self.process.stdin is not None
            await self.process.stdin.drain()

    def _write_frame(self, message: JsonObject) -> None:
        if not self._alive or self.process.stdin is None:
            raise CodeUnavailableError("The Codex connection is closed.")
        self.process.stdin.write(
            (json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n").encode(
                "utf-8"
            )
        )

    async def _read(self) -> None:
        assert self.process.stdout is not None
        buffer = bytearray()
        message = "The Codex process ended."
        discarded_output: asyncio.Task[None] | None = None
        try:
            while chunk := await self.process.stdout.read(65536):
                buffer.extend(chunk)
                while (position := buffer.find(b"\n")) >= 0:
                    frame = bytes(buffer[:position])
                    del buffer[: position + 1]
                    if frame.strip():
                        await self._packet(
                            object_value(json.loads(frame.decode("utf-8")))
                        )
            if buffer.strip():
                raise ValueError("Incomplete native JSONL frame")
        except Exception as exc:
            message = f"The Codex connection failed ({type(exc).__name__})."
            discarded_output = asyncio.create_task(self._discard_stdout())
        finally:
            self._alive = False
            for _, future in self._pending.values():
                if not future.done():
                    future.set_exception(CodeUnavailableError(message))
            self._pending.clear()
            self._requests.clear()
            # A failed pipe does not prove the native agent has stopped working.
            # Keep the session reserved until its owned process is actually gone.
            await self._reap_owned_process()
            if discarded_output is not None:
                await discarded_output
            self._queue.put_nowait(
                HarnessEvent(self.generation, self.thread_id, "closed", message=message)
            )
            self._queue.put_nowait(None)

    async def _discard_stdout(self) -> None:
        # Keep draining a failed protocol stream until EOF, so a full pipe
        # cannot block native shutdown or retain its subprocess transport.
        assert self.process.stdout is not None
        while await self.process.stdout.read(65536):
            pass

    async def _packet(self, packet: JsonObject) -> None:
        method = string_value(packet.get("method"))
        if not method:
            request_id = packet.get("id")
            pending = (
                self._pending.pop(request_id, None)
                if isinstance(request_id, str)
                else None
            )
            if pending is None:
                return
            operation, future = pending
            if future.done():
                return
            if "error" in packet:
                error = object_value(packet["error"])
                message = (
                    self._redact(string_value(error.get("message")))
                    or "Codex rejected the request."
                )
                if operation in {
                    "thread/read",
                    "thread/resume",
                    "thread/delete",
                } and any(
                    phrase in message.lower()
                    for phrase in (
                        "thread not found",
                        "thread does not exist",
                        "no rollout found",
                        "not found for thread",
                    )
                ):
                    future.set_exception(NativeHistoryMissing(message))
                elif error.get("code") == -32601:
                    future.set_exception(
                        CodeUnavailableError(
                            "This Codex version lacks the required app-server method. Update Codex."
                        )
                    )
                else:
                    future.set_exception(CodeConflictError(message))
                return
            result = object_value(packet.get("result"))
            if operation in {"thread/start", "thread/resume"}:
                identity = string_value(object_value(result.get("thread")).get("id"))
                if identity:
                    self.thread_id = identity
                    self._owned_threads.add(identity)
            future.set_result(result)
            return
        params = object_value(packet.get("params"))
        if method == "thread/started":
            thread = object_value(params.get("thread"))
            source = object_value(object_value(thread.get("source")).get("subAgent"))
            parent = object_value(source.get("thread_spawn")).get("parent_thread_id")
            child = string_value(thread.get("id"))
            if child and isinstance(parent, str) and parent in self._owned_threads:
                self._owned_threads.add(child)
            return
        request_id = packet.get("id")
        if isinstance(request_id, str | int) and not isinstance(request_id, bool):
            try:
                thread_id = string_value(params.get("threadId")) or self.thread_id
                if thread_id and thread_id not in self._owned_threads:
                    raise CodeValidationError(
                        "Codex requested input for a conversation outside this Code session."
                    )
                prompt = NativePrompt(method, request_id, params)
            except CodeValidationError as exc:
                await self._write(
                    {"id": request_id, "error": {"code": -32601, "message": str(exc)}}
                )
                self._queue.put_nowait(
                    HarnessEvent(
                        self.generation, self.thread_id, "error", message=str(exc)
                    )
                )
                return
            self._requests[request_id] = prompt
            request = prompt.request()
            if thread_id != self.thread_id:
                request = replace(
                    request,
                    turn_id=None,
                    form={
                        **request.form,
                        "detail": "A Codex subagent needs your input.\n"
                        + string_value(request.form.get("detail")),
                    },
                )
            self._queue.put_nowait(
                HarnessEvent(
                    self.generation,
                    self.thread_id,
                    "prompt",
                    turn_id=request.turn_id,
                    prompt=request,
                )
            )
            return
        event = self._protocol.notification(method, params)
        if event is not None:
            if event.kind == "resolved" and event.request_id is not None:
                self._requests.pop(event.request_id, None)
                if event.thread_id in self._owned_threads:
                    event = replace(event, thread_id=self.thread_id)
            if event.kind == "turn_completed":
                for key, prompt in tuple(self._requests.items()):
                    if (
                        prompt.params.get("turnId") == event.turn_id
                        and prompt.params.get("threadId") == event.thread_id
                    ):
                        self._requests.pop(key, None)
                        if (
                            event.thread_id != self.thread_id
                            and event.thread_id in self._owned_threads
                        ):
                            self._queue.put_nowait(
                                HarnessEvent(
                                    self.generation,
                                    self.thread_id,
                                    "resolved",
                                    request_id=key,
                                )
                            )
            self._queue.put_nowait(event)

    async def _dispatch(self) -> None:
        while (event := await self._queue.get()) is not None:
            await self._sink(event)

    async def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        while chunk := await self.process.stderr.read(8192):
            self._stderr_tail.extend(chunk)
            del self._stderr_tail[:-16384]

    def _redact(self, text: str) -> str:
        for key, value in self._env.items():
            if value and any(
                part in key.upper()
                for part in ("TOKEN", "API_KEY", "PASSWORD", "SECRET")
            ):
                text = text.replace(value, "[redacted]")
        return text

    async def close(self) -> None:
        if asyncio.current_task() is self._dispatcher:
            await self._reap_owned_process()
            return
        if self._closing is None:
            self._closing = asyncio.create_task(
                self._close(),
                name="fcc-codex-close",
            )
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        await self._reap_owned_process()
        if self._reader is not None:
            await self._reader
        if self._dispatcher is not None:
            await self._dispatcher

    async def _reap_owned_process(self) -> None:
        if self._reaping is None:
            self._reaping = asyncio.create_task(self._reap(), name="fcc-codex-reap")
        await asyncio.shield(self._reaping)

    async def _reap(self) -> None:
        process = self._process
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                async with asyncio.timeout(5):
                    await process.wait()
            except TimeoutError:
                await self._kill_tree(process)
            if self._stderr is not None:
                await self._stderr
            unregister_pid(process.pid)
        self._alive = False
        self._resources.close()

    @staticmethod
    async def _kill_tree(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            await killer.wait()
        else:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()


@dataclass(frozen=True, slots=True)
class _CodexSelection:
    context: LaunchContext
    model: str
    configuration_key: str
    fingerprints: dict[str, str]
    reasoning_effort: str | None
    mode: CodeMode

    async def open(self, cwd: str, sink: EventSink) -> CodexAppServer:
        resources = ExitStack()
        try:
            prepared = prepare_codex_launch(
                self.context,
                [
                    "-c",
                    "features.default_mode_request_user_input=true",
                    "-c",
                    "tools.experimental_request_user_input.enabled=true",
                    "app-server",
                ],
                LaunchResources(resources),
            )
            connection = CodexAppServer(
                prepared.command,
                prepared.env,
                cwd,
                sink,
                model_slugs={
                    model.provider_model_ref: model.wire_slug
                    for model in self.context.models
                },
                fingerprints=self.fingerprints,
                reasoning={
                    model.provider_model_ref: model.supports_reasoning is not False
                    for model in self.context.models
                },
                resources=resources,
            )
            await connection.start()
            return connection
        except BaseException:
            resources.close()
            raise


class CodexHarnessFactory:
    def __init__(
        self,
        runtime: RequestRuntimePort,
        *,
        binary: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._runtime, self._binary, self._env = runtime, binary, env

    def availability(self) -> tuple[bool, str | None]:
        if self._binary or shutil.which("codex"):
            return True, None
        return False, "Codex is not installed. " + SPEC.install_hint

    def catalog(self) -> CodeCatalog:
        settings = self._runtime.current_settings()
        models = current_codex_models(self._runtime, settings)
        entries = array_value(build_codex_model_catalog(models).get("models"))
        return CodeCatalog(
            settings.model,
            tuple(
                _model_option(model.provider_model_ref, object_value(entry))
                for model, entry in zip(models, entries, strict=True)
            ),
        )

    def prepare(
        self, model: str, reasoning_effort: str | None, mode: CodeMode
    ) -> _CodexSelection:
        binary = self._binary or shutil.which("codex")
        if binary is None:
            raise CodeUnavailableError("Codex is not installed. " + SPEC.install_hint)
        settings = self._runtime.current_settings()
        models = current_codex_models(self._runtime, settings)
        selected = next(
            (
                candidate
                for candidate in models
                if candidate.provider_model_ref == model
            ),
            None,
        )
        if selected is None:
            raise CodeValidationError(
                "This model is unavailable. Choose another model."
            )
        catalog = build_codex_model_catalog(models)
        entries = {
            string_value(entry.get("slug")): entry
            for value in array_value(catalog.get("models"))
            if (entry := object_value(value))
        }
        option = _model_option(model, entries[selected.wire_slug])
        if (
            reasoning_effort is not None
            and reasoning_effort not in option.reasoning_efforts
        ):
            raise CodeValidationError(
                "This effort is unavailable. Choose another effort."
            )
        effort = reasoning_effort or option.default_reasoning_effort
        fingerprints = {
            model.provider_model_ref: _fingerprint(entries[model.wire_slug])
            for model in models
        }
        context = LaunchContext(
            binary,
            settings.model_copy(update={"model": model}),
            local_proxy_root_url(settings),
            settings.proxy_auth_token,
            dict(self._env if self._env is not None else os.environ),
            models,
        )
        return _CodexSelection(
            context, model, fingerprints[model], fingerprints, effort, mode
        )

    async def open_history(self, cwd: str, sink: EventSink) -> CodexAppServer:
        binary = self._binary or shutil.which("codex")
        if binary is None:
            raise CodeUnavailableError(
                "Codex is needed to remove its native conversation. "
                + SPEC.install_hint
            )
        connection = CodexAppServer(
            [binary, "app-server"],
            dict(self._env if self._env is not None else os.environ),
            cwd,
            sink,
        )
        await connection.start()
        return connection


def _model_option(model: str, entry: JsonObject) -> CodeModel:
    provider_id, model_name = split_provider_model_ref(model)
    efforts = tuple(
        string_value(object_value(level).get("effort"))
        for level in array_value(entry.get("supported_reasoning_levels"))
    )
    return CodeModel(
        id=model,
        display_name=string_value(entry.get("display_name")) or model,
        provider_id=provider_id,
        model_name=model_name,
        reasoning_efforts=tuple(
            "off" if effort == "none" else effort for effort in efforts
        )
        or ("off",),
        default_reasoning_effort=string_value(entry.get("default_reasoning_level"))
        or "off",
    )


def _permission_settings(mode: CodeMode, defaults: JsonObject) -> JsonObject:
    if mode != "config":
        return {
            "approvalPolicy": "never" if mode == "full_access" else "on-request",
            "approvalsReviewer": "auto_review" if mode == "auto_review" else "user",
            "permissions": ":danger-full-access"
            if mode == "full_access"
            else ":workspace",
        }
    policy = defaults.get("approvalPolicy")
    reviewer = string_value(defaults.get("approvalsReviewer"))
    profile = string_value(
        object_value(defaults.get("activePermissionProfile")).get("id")
    )
    sandbox = object_value(defaults.get("sandbox"))
    if (
        not isinstance(policy, str | dict)
        or not policy
        or not reviewer
        or (not profile and not sandbox.get("type"))
    ):
        raise CodeUnavailableError(
            "Codex did not return complete native permission settings. Your input was not sent."
        )
    return {
        "approvalPolicy": policy,
        "approvalsReviewer": reviewer,
        **({"permissions": profile} if profile else {"sandboxPolicy": sandbox}),
    }


def _fingerprint(entry: JsonObject) -> str:
    behavior = {
        key: value
        for key, value in entry.items()
        if key not in {"display_name", "description", "priority", "visibility"}
    }
    return hashlib.sha256(
        json.dumps(behavior, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()

"""Controllable native harness used by application and browser tests."""

import asyncio
import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock

from free_claude_code.application.code_sessions.models import (
    CodeCatalog,
    CodeConflictError,
    CodeMode,
    CodeModel,
    CodeUnavailableError,
    CodeValidationError,
    HarnessEvent,
    ItemUpdate,
    NativeHistoryMissing,
    NativeThread,
    NativeTurn,
    PromptRequest,
    RunStatus,
)
from free_claude_code.application.code_sessions.ports import EventSink, HarnessSelection
from free_claude_code.config.model_refs import split_provider_model_ref
from free_claude_code.core.json_types import JsonObject
from free_claude_code.runtime.codex_app_server import CodexAppServer
from free_claude_code.runtime.codex_protocol import CodexProtocol


class CodexPackets:
    """Feed native packets through the real adapter without starting a process."""

    def __init__(self, connection):
        self.connection = connection
        self.native = CodexAppServer([], {}, "", connection.sink)
        self.native.generation = connection.generation
        self.native._protocol = CodexProtocol(connection.generation)
        self.parents: dict[str, str] = {}
        self.native.rpc = AsyncMock(side_effect=self.metadata)

    async def metadata(self, method: str, params: JsonObject) -> JsonObject:
        assert method == "thread/read" and params["includeTurns"] is False
        identity = params["threadId"]
        assert isinstance(identity, str)
        parent = self.parents.get(identity)
        return {
            "thread": {
                "id": identity,
                "source": {"subAgent": {"thread_spawn": {"parent_thread_id": parent}}}
                if parent
                else "vscode",
            }
        }

    async def send(self, method, params):
        if self.native.thread_id is None:
            future = asyncio.get_running_loop().create_future()
            self.native._pending["register"] = ("thread/resume", future)
            await self.native._packet(
                {
                    "id": "register",
                    "result": {"thread": {"id": self.connection.thread_id}},
                }
            )
        await self.native._packet({"method": method, "params": params})
        self.native._queue.put_nowait(None)
        await self.native._dispatch()

    async def spawn(self, child="child", parent=None):
        self.parents[child] = parent or self.connection.thread_id
        await self.send(
            "thread/started",
            {
                "thread": {
                    "id": child,
                    "source": {
                        "subAgent": {
                            "thread_spawn": {
                                "parent_thread_id": parent or self.connection.thread_id,
                                "depth": 1 if parent is None else 2,
                            }
                        }
                    },
                }
            },
        )

    async def review(
        self, child="child", review_id="review", status="inProgress", turn="child-turn"
    ):
        payload = {
            "threadId": child,
            "turnId": turn,
            "reviewId": review_id,
            "targetItemId": "command",
            "startedAtMs": 1000,
            "action": {
                "type": "command",
                "command": "echo harmless",
                "cwd": "/tmp",
                "source": "shell",
            },
            "review": {"status": status, "rationale": f"Native {status}"},
        }
        if status != "inProgress":
            payload.update(completedAtMs=2000, decisionSource="agent")
        await self.send(
            "item/autoApprovalReview/started"
            if status == "inProgress"
            else "item/autoApprovalReview/completed",
            payload,
        )

    async def warning(self, child="child", message="Native warning"):
        await self.send("guardianWarning", {"threadId": child, "message": message})


class FakeHarness:
    def __init__(self):
        self.connections: list[FakeConnection] = []
        self.model = "provider/model"
        self.permission_defaults: JsonObject = {
            "approvalPolicy": "on-request",
            "approvalsReviewer": "user",
            "sandbox": {"type": "workspaceWrite"},
            "activePermissionProfile": {"id": ":workspace"},
        }
        self.configurations = {self.model: "capabilities-1"}
        self.efforts = ("off", "low", "medium", "high", "xhigh", "max")
        self.default_effort = "medium"
        self.creation_gate = asyncio.Event()
        self.start_gate = asyncio.Event()
        self.creation_gate.set()
        self.start_gate.set()
        self.creating = asyncio.Event()
        self.started = asyncio.Event()
        self.submitted = asyncio.Event()
        self.interrupted = asyncio.Event()
        self.input_changed = asyncio.Event()
        self.answer_gate = asyncio.Event()
        self.answer_gate.set()
        self.answering = asyncio.Event()
        self.answered = asyncio.Event()
        self.interrupt_gate = asyncio.Event()
        self.interrupt_gate.set()
        self.turn_count = 0
        self.histories: dict[str, list[ItemUpdate]] = {}
        self.delete_error: Exception | None = None
        self.delete_gate = asyncio.Event()
        self.delete_gate.set()
        self.deleting = asyncio.Event()
        self.delete_before_error = False

    def availability(self):
        return True, None

    def catalog(self):
        return CodeCatalog(
            self.model,
            tuple(
                CodeModel(
                    id=model,
                    display_name=model,
                    provider_id=split_provider_model_ref(model)[0],
                    model_name=split_provider_model_ref(model)[1],
                    reasoning_efforts=self.efforts,
                    default_reasoning_effort=self.default_effort,
                )
                for model in self.configurations
            ),
        )

    def prepare(self, model, reasoning_effort, mode):
        if model not in self.configurations:
            raise CodeValidationError("This model is unavailable.")
        if reasoning_effort is not None and reasoning_effort not in self.efforts:
            raise CodeValidationError("This reasoning effort is unavailable.")
        return FakeSelection(
            self,
            model,
            self.configurations[model],
            dict(self.configurations),
            reasoning_effort or self.default_effort,
            mode,
        )

    async def open_history(self, cwd: str, sink: EventSink):
        return await self.prepare(self.model, None, "config").open(cwd, sink)

    async def wait_inputs(self, count: int):
        while sum(len(connection.inputs) for connection in self.connections) < count:
            self.input_changed.clear()
            await self.input_changed.wait()


@dataclass
class FakeSelection:
    harness: FakeHarness
    model: str
    configuration_key: str
    catalog: dict[str, str]
    reasoning_effort: str | None
    mode: CodeMode

    async def open(self, cwd: str, sink: EventSink):
        connection = FakeConnection(self.harness, sink, self.catalog)
        self.harness.connections.append(connection)
        return connection


class FakeConnection:
    def __init__(self, harness: FakeHarness, sink: EventSink, catalog: dict[str, str]):
        self.harness = harness
        self.sink = sink
        self.catalog = catalog
        self.generation = str(uuid.uuid4())
        self.thread_id: str | None = None
        self.inputs: list[tuple[str, str, str]] = []
        self.efforts: list[str | None] = []
        self.modes: list[CodeMode] = []
        self.defaults: list[JsonObject] = []
        self.interrupts: list[str] = []
        self.deleted: list[str] = []
        self.resumed: list[str] = []
        self.answers: list[tuple[str | int, JsonObject]] = []
        self.requests: dict[str | int, PromptRequest] = {}
        self.closed = False

    def supports(self, selection: HarnessSelection):
        return (
            not self.closed
            and self.catalog.get(selection.model) == selection.configuration_key
        )

    async def create_thread(self):
        self.harness.creating.set()
        await self.harness.creation_gate.wait()
        if self.closed:
            raise CodeUnavailableError("Native process closed during creation.")
        self.thread_id = f"native-{len(self.harness.histories) + 1}"
        self.harness.histories[self.thread_id] = []
        return NativeThread(
            self.thread_id, permission_defaults=self.harness.permission_defaults
        )

    async def resume_thread(self, thread_id: str):
        self.resumed.append(thread_id)
        self.thread_id = thread_id
        return await self.read_thread(thread_id)

    async def read_thread(self, thread_id: str):
        if thread_id not in self.harness.histories:
            raise NativeHistoryMissing("Native conversation not found.")
        groups: dict[str, list[ItemUpdate]] = {}
        for item in self.harness.histories[thread_id]:
            groups.setdefault(item.turn_id, []).append(item)
        return NativeThread(
            thread_id,
            tuple(
                NativeTurn(turn_id, tuple(items)) for turn_id, items in groups.items()
            ),
            self.harness.permission_defaults,
        )

    async def start_turn(
        self,
        text: str,
        selection: HarnessSelection,
        client_id: str,
        permission_defaults: JsonObject,
    ):
        self.inputs.append((client_id, text, selection.model))
        self.efforts.append(selection.reasoning_effort)
        self.modes.append(selection.mode)
        self.defaults.append(permission_defaults)
        self.harness.input_changed.set()
        self.harness.submitted.set()
        self.harness.turn_count += 1
        turn_id = f"turn-{self.harness.turn_count}"
        await self.harness.start_gate.wait()
        if self.closed:
            raise CodeUnavailableError("Native process closed during submission.")
        await self.sink(
            HarnessEvent(
                self.generation, self.thread_id, "turn_started", turn_id=turn_id
            )
        )
        self.harness.started.set()
        return turn_id

    async def interrupt(self, turn_id: str):
        self.interrupts.append(turn_id)
        self.harness.interrupted.set()
        await self.harness.interrupt_gate.wait()
        if self.closed:
            raise CodeUnavailableError(
                "Native process closed before interrupt acknowledgement."
            )
        await self.finish(turn_id, "interrupted")

    async def finish(self, turn_id: str, status: RunStatus = "completed", message=None):
        await self.sink(
            HarnessEvent(
                self.generation,
                self.thread_id,
                "turn_completed",
                turn_id=turn_id,
                status=status,
                message=message,
            )
        )

    async def text(
        self, turn_id: str, item_id: str, text: str, *, complete=False, kind="text"
    ):
        item = ItemUpdate(
            turn_id,
            item_id,
            kind,
            text=text,
            complete=complete,
            raw={"id": item_id, "text": text},
        )
        if self.thread_id:
            self.harness.histories[self.thread_id].append(item)
        await self.sink(
            HarnessEvent(
                self.generation, self.thread_id, "item", turn_id=turn_id, item=item
            )
        )

    async def prompt(self, request_id: str | int, turn_id: str | None = "turn-1"):
        prompt = PromptRequest(
            request_id,
            "approval",
            {
                "title": "Run command?",
                "choices": [
                    {"id": "accept", "label": "Allow"},
                    {"id": "decline", "label": "Decline"},
                ],
            },
            {"requestId": request_id},
            turn_id,
        )
        self.requests[request_id] = prompt
        await self.sink(
            HarnessEvent(
                self.generation,
                self.thread_id,
                "prompt",
                turn_id=turn_id,
                prompt=prompt,
            )
        )

    def prepare_answer(self, request_id: str | int, answer: JsonObject):
        if request_id not in self.requests:
            raise CodeConflictError("Prompt resolved.")
        return dict(answer)

    async def respond(self, request_id: str | int, response: JsonObject):
        self.harness.answering.set()
        await self.harness.answer_gate.wait()
        try:
            if request_id not in self.requests:
                raise CodeConflictError("Prompt resolved.")
            self.answers.append((request_id, response))
            await self.resolve(request_id)
        finally:
            self.harness.answered.set()

    async def resolve(self, request_id: str | int):
        self.requests.pop(request_id, None)
        await self.sink(
            HarnessEvent(
                self.generation, self.thread_id, "resolved", request_id=request_id
            )
        )

    async def delete_thread(self, thread_id: str):
        self.deleted.append(thread_id)
        self.harness.deleting.set()
        await self.harness.delete_gate.wait()
        if self.harness.delete_before_error:
            self.harness.histories.pop(thread_id, None)
        if self.harness.delete_error:
            raise self.harness.delete_error
        self.harness.histories.pop(thread_id, None)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.harness.creation_gate.set()
        self.harness.start_gate.set()
        self.harness.answer_gate.set()
        self.harness.interrupt_gate.set()
        self.harness.delete_gate.set()
        await self.sink(HarnessEvent(self.generation, self.thread_id, "closed"))

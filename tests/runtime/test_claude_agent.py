import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest
from claude_agent_sdk import ResultMessage, UserMessage
from claude_agent_sdk.types import ToolPermissionContext

from free_claude_code.application.code_sessions.models import (
    CodeConflictError,
    CodeValidationError,
)
from free_claude_code.runtime.claude_agent import ClaudeConnection, _ClaudeSelection


class SDKClient:
    instances: ClassVar[list] = []

    def __init__(self, *, options):
        self.options = options
        self.messages = asyncio.Queue()
        self.queries = []
        self.disconnected = False
        self.__class__.instances.append(self)

    async def connect(self):
        pass

    async def get_server_info(self):
        return {
            "models": [
                {
                    "value": self.options.model,
                    "supportsEffort": True,
                    "supportedEffortLevels": ["low", "high"],
                    "supportsAutoMode": False,
                }
            ]
        }

    async def query(self, prompt, session_id):
        self.queries.extend([message async for message in prompt])
        message = self.queries[-1]
        await self.messages.put(
            UserMessage(
                message["message"]["content"],
                uuid=message["uuid"],
                origin=message["origin"],
            )
        )

    async def receive_messages(self):
        while True:
            yield await self.messages.get()

    async def interrupt(self):
        await self.messages.put(
            ResultMessage(
                "success",
                1,
                1,
                False,
                1,
                self.options.session_id,
                origin={"kind": "human"},
            )
        )

    async def disconnect(self):
        self.disconnected = True


@pytest.fixture
def sdk(monkeypatch):
    SDKClient.instances = []
    monkeypatch.setattr("claude_agent_sdk.ClaudeSDKClient", SDKClient)
    return SDKClient


@pytest.mark.asyncio
async def test_reader_reports_closure_even_when_sdk_cleanup_fails(
    sdk, tmp_path, monkeypatch
):
    events = []
    closed = asyncio.Event()

    async def sink(event):
        events.append(event)
        if event.kind == "closed":
            closed.set()

    connection = ClaudeConnection(str(tmp_path), selection(), sink)
    await connection.create_thread()
    client = sdk.instances[0]
    await connection.start_turn("hi", selection(), "run", None)

    async def failed_disconnect():
        raise RuntimeError("Cleanup failed")

    monkeypatch.setattr(client, "disconnect", failed_disconnect)
    try:
        await client.messages.put(ResultMessage("success", 1, 1, False, 1, "s"))
        async with asyncio.timeout(1):
            await closed.wait()
        assert "Cleanup failed" in events[-1].message
    finally:
        monkeypatch.undo()
        await connection.close()


def selection(mode="config", effort=None):
    return _ClaudeSelection(
        "installed-claude",
        {},
        "anthropic/provider/model",
        "provider/model",
        effort,
        mode,
        "fingerprint",
    )


@pytest.mark.asyncio
async def test_native_preparation_sends_nothing_and_stop_settles_turn(sdk, tmp_path):
    events = []
    started, finished = asyncio.Event(), asyncio.Event()

    async def sink(event):
        events.append(event)
        if event.kind == "turn_started":
            started.set()
        if event.kind == "turn_completed":
            finished.set()

    connection = ClaudeConnection(str(tmp_path), selection(), sink)
    try:
        native = await connection.create_thread()
        client = sdk.instances[0]
        assert client.options.cli_path == "installed-claude"
        assert client.options.permission_mode is None
        assert client.options.system_prompt == {
            "type": "preset",
            "preset": "claude_code",
        }
        assert not client.queries
        assert native.permission_defaults is None
        await connection.start_turn("hello", selection(), "run", None)
        await asyncio.wait_for(started.wait(), 2)
        assert client.queries[0]["uuid"] == "run"
        await connection.interrupt("run")
        await asyncio.wait_for(finished.wait(), 2)
        assert events[-1].status == "interrupted"
    finally:
        await connection.close()
    assert client.disconnected


@pytest.mark.asyncio
async def test_unsupported_effort_closes_without_query(sdk, tmp_path):
    async def sink(event):
        pass

    connection = ClaudeConnection(str(tmp_path), selection(effort="max"), sink)
    with pytest.raises(CodeValidationError):
        await connection.create_thread()
    assert sdk.instances[0].disconnected
    assert not sdk.instances[0].queries


@pytest.mark.asyncio
async def test_permission_answer_and_cancellation_leave_no_waiters(sdk, tmp_path):
    prompts = asyncio.Queue()
    started = asyncio.Event()

    async def sink(event):
        if event.kind == "turn_started":
            started.set()
        if event.kind == "prompt":
            await prompts.put(event.prompt)

    connection = ClaudeConnection(str(tmp_path), selection(), sink)
    await connection.create_thread()
    try:
        await connection.start_turn("work", selection(), "run", None)
        await asyncio.wait_for(started.wait(), 2)
        task = asyncio.create_task(
            connection._can_use_tool(
                "Bash",
                {"command": "echo hello"},
                ToolPermissionContext(tool_use_id="tool"),
            )
        )
        prompt = await asyncio.wait_for(prompts.get(), 2)
        response = connection.prepare_answer(prompt.request_id, {"choice": "allow"})
        await connection.respond(prompt.request_id, response)
        result = await task
        assert result.behavior == "allow"
        with pytest.raises(CodeConflictError):
            connection.prepare_answer(prompt.request_id, {"choice": "allow"})
        task = asyncio.create_task(
            connection._can_use_tool(
                "Bash",
                {"command": "echo hello"},
                ToolPermissionContext(tool_use_id="cancel"),
            )
        )
        await asyncio.wait_for(prompts.get(), 2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert not connection._prompts
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_history_uses_exact_submitted_ids_and_public_delete(
    sdk, monkeypatch, tmp_path
):
    from claude_agent_sdk.types import SessionMessage

    deleted = []
    monkeypatch.setattr(
        "claude_agent_sdk.get_session_info", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        "claude_agent_sdk.get_session_messages",
        lambda *args, **kwargs: [
            SessionMessage(
                "user", "summary", "native", {"content": "compacted summary"}
            ),
            SessionMessage(
                "assistant",
                "old",
                "native",
                {
                    "id": "old-msg",
                    "content": [{"type": "text", "text": "unattributed"}],
                },
            ),
            SessionMessage("user", "run", "native", {"content": "actual input"}),
            SessionMessage(
                "assistant",
                "answer",
                "native",
                {"id": "msg", "content": [{"type": "text", "text": "actual answer"}]},
            ),
        ],
    )
    monkeypatch.setattr(
        "claude_agent_sdk.delete_session",
        lambda identity, **kwargs: deleted.append(identity),
    )

    async def sink(event):
        pass

    connection = ClaudeConnection(str(tmp_path), selection(), sink)
    try:
        history = await connection.resume_thread(
            "native", submitted_run_ids=frozenset({"run"})
        )
        assert [turn.id for turn in history.turns] == ["run"]
        assert [item.text for item in history.turns[0].items] == [
            "actual input",
            "actual answer",
        ]
        assert sdk.instances[0].options.resume == "native"
    finally:
        await connection.close()
    await connection.delete_thread("native")
    assert deleted == ["native"]


@pytest.mark.asyncio
async def test_cancelled_close_waits_for_owned_sdk_cleanup(sdk, tmp_path):
    async def sink(event):
        pass

    connection = ClaudeConnection(str(tmp_path), selection(), sink)
    await connection.create_thread()
    entered, release = asyncio.Event(), asyncio.Event()
    client = sdk.instances[0]

    async def disconnect():
        entered.set()
        await release.wait()
        client.disconnected = True

    client.disconnect = disconnect
    closing = asyncio.create_task(connection.close())
    await entered.wait()
    try:
        closing.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.gather(closing, return_exceptions=True)
        await connection.close()
    assert client.disconnected

import asyncio
import json

import pytest

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.platforms.ports import MessagingStartupNotice
from free_claude_code.messaging.trees import MessageState, TreeIdentity
from smoke.lib.e2e import FakeCLISession, FakePlatformDriver, default_cli_events

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("messaging")]


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_messaging_fake_full_flow_e2e(platform_name: str, tmp_path) -> None:
    driver = FakePlatformDriver(platform_name, tmp_path)

    incoming = await driver.send("Please inspect README.", message_id="root_1")

    node = await driver.workflow.tree_queue.get_node(
        incoming.scope,
        incoming.message_id,
    )
    assert node is not None
    assert node.state == MessageState.COMPLETED
    assert driver.platform.sent
    assert driver.platform.edits
    edit_text = "\n".join(edit["text"] for edit in driver.platform.edits)
    assert "Fake platform answer" in edit_text
    assert "Read" in edit_text


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_messaging_subagent_control_e2e(platform_name: str, tmp_path) -> None:
    task_events = [
        {"type": "session_info", "session_id": "sess_task"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "Need a focused worker."},
                    {
                        "type": "tool_use",
                        "id": "toolu_task",
                        "name": "Task",
                        "input": {"description": "inspect", "prompt": "inspect"},
                    },
                    {"type": "text", "text": "Subagent result rendered."},
                ]
            },
        },
        {"type": "exit", "code": 0, "stderr": None},
    ]
    driver = FakePlatformDriver(platform_name, tmp_path, event_batches=[task_events])

    await driver.send("Delegate this safely.", message_id="root_task")

    edit_text = "\n".join(edit["text"] for edit in driver.platform.edits)
    assert "Subagent" in edit_text
    assert "Tool calls" in edit_text


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_messaging_commands_stop_clear_stats_e2e(
    platform_name: str, tmp_path
) -> None:
    driver = FakePlatformDriver(platform_name, tmp_path)
    root = await driver.send("start work", message_id="root_1")
    root_status_id = driver.platform.sent[-1]["message_id"]

    await driver.send("/stats", message_id="stats_1")
    await driver.send("/stop", message_id="stop_1", reply_to=root.message_id)
    await driver.send("/clear", message_id="clear_1", reply_to=root.message_id)
    global_prompt = await driver.send("clear globally", message_id="global_prompt")
    global_status_id = driver.platform.sent[-1]["message_id"]
    await driver.send("/clear", message_id="clear_all")

    sent_text = "\n".join(sent["text"] for sent in driver.platform.sent)
    deleted = {entry["message_id"] for entry in driver.platform.deletes}
    assert "Stats" in sent_text
    assert "Nothing to stop for that message" in sent_text
    assert {
        root.message_id,
        root_status_id,
        global_prompt.message_id,
        global_status_id,
        "clear_1",
        "clear_all",
    } <= deleted
    assert driver.session_store.load_conversation_snapshot().trees == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_reply_clear_uses_literal_platform_subtree_e2e(
    platform_name: str,
    tmp_path,
) -> None:
    driver = FakePlatformDriver(platform_name, tmp_path)
    root = await driver.send("root", message_id="root")
    root_status = driver.platform.sent[-1]["message_id"]
    sibling = await driver.send(
        "prompt sibling",
        message_id="sibling",
        reply_to=root.message_id,
    )
    sibling_status = driver.platform.sent[-1]["message_id"]
    descendant = await driver.send(
        "status descendant",
        message_id="descendant",
        reply_to=root_status,
    )
    descendant_status = driver.platform.sent[-1]["message_id"]

    await driver.send(
        "/clear",
        message_id="clear-status",
        reply_to=root_status,
    )

    deleted = {entry["message_id"] for entry in driver.platform.deletes}
    assert {
        root_status,
        descendant.message_id,
        descendant_status,
        "clear-status",
    } <= deleted
    assert {root.message_id, sibling.message_id, sibling_status}.isdisjoint(deleted)
    root_view = await driver.workflow.tree_queue.get_node(root.scope, root.message_id)
    sibling_view = await driver.workflow.tree_queue.get_node(
        sibling.scope, sibling.message_id
    )
    assert root_view is not None and root_view.state is MessageState.ERROR
    assert root_view.session_id is None
    assert sibling_view is not None and sibling_view.state is MessageState.COMPLETED
    assert (
        await driver.workflow.tree_queue.get_node(
            descendant.scope, descendant.message_id
        )
        is None
    )

    await driver.send(
        "fresh continuation",
        message_id="fresh",
        reply_to=root.message_id,
    )
    fresh_call = next(
        call
        for session in driver.cli_manager.sessions
        for call in session.calls
        if call["prompt"] == "fresh continuation"
    )
    assert fresh_call["session_id"] is None


@pytest.mark.asyncio
async def test_messaging_startup_notice_is_clearable_e2e(tmp_path) -> None:
    first_driver = FakePlatformDriver("telegram", tmp_path)
    scope = MessageScope(platform="telegram", chat_id="chat_1")

    await first_driver.workflow.publish_startup_notice(
        MessagingStartupNotice(
            chat_id=scope.chat_id,
            transport_label="Bot API",
        )
    )
    first_startup_id = first_driver.platform.sent[-1]["message_id"]
    first_driver.session_store.flush_pending_save()

    driver = FakePlatformDriver("telegram", tmp_path)
    await driver.platform.send_message("untracked", "advance fake message ID")
    await driver.workflow.publish_startup_notice(
        MessagingStartupNotice(
            chat_id=scope.chat_id,
            transport_label="Bot API",
        )
    )
    second_startup = driver.platform.sent[-1]
    second_startup_id = second_startup["message_id"]
    assert second_startup["text"] == (
        "🚀 *Claude Code Proxy is online\\!* \\(Bot API\\)"
    )
    assert second_startup["parse_mode"] == "MarkdownV2"
    assert driver.session_store.get_tracked_message_ids_for_chat(
        scope.platform, scope.chat_id
    ) == [first_startup_id, second_startup_id]

    await driver.send("/clear", message_id="clear_startup")

    deleted = {entry["message_id"] for entry in driver.platform.deletes}
    assert {first_startup_id, second_startup_id, "clear_startup"} <= deleted
    assert (
        driver.session_store.get_tracked_message_ids_for_chat(
            scope.platform, scope.chat_id
        )
        == []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_messaging_active_stop_uses_status_only_e2e(
    platform_name: str,
    tmp_path,
    monkeypatch,
) -> None:
    driver = FakePlatformDriver(platform_name, tmp_path)
    started = asyncio.Event()

    class GatedActiveSession(FakeCLISession):
        async def start_task(
            self,
            prompt: str,
            session_id: str | None = None,
            fork_session: bool = False,
        ):
            self.calls.append(
                {
                    "prompt": prompt,
                    "session_id": session_id,
                    "fork_session": fork_session,
                }
            )
            self.is_busy = True
            try:
                yield {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "partial answer"}]
                    },
                }
                started.set()
                await asyncio.Event().wait()
            finally:
                self.is_busy = False

    async def controlled_session(session_id: str | None = None):
        session = GatedActiveSession([])
        driver.cli_manager.sessions.append(session)
        return session, session_id or "pending_0", session_id is None

    monkeypatch.setattr(
        driver.cli_manager,
        "get_or_create_session",
        controlled_session,
    )
    root = await driver.emit("active work", message_id="root_active")
    await started.wait()
    status_id = driver.platform.sent[-1]["message_id"]
    sent_before_stop = len(driver.platform.sent)

    await driver.emit(
        "/stop",
        message_id="stop_active",
        reply_to=root.message_id,
    )
    await driver.wait_for_idle()

    assert len(driver.platform.sent) == sent_before_stop
    stopped_edits = [
        edit
        for edit in driver.platform.edits
        if edit["message_id"] == status_id and "Stopped" in edit["text"]
    ]
    assert stopped_edits
    assert "partial answer" in stopped_edits[-1]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_tree_threading_e2e(platform_name: str, tmp_path) -> None:
    batches = [default_cli_events("sess_root"), default_cli_events("sess_branch")]
    driver = FakePlatformDriver(platform_name, tmp_path, event_batches=batches)

    root = await driver.send("root prompt", message_id="root_1")
    branch = await driver.send(
        "branch prompt", message_id="branch_1", reply_to=root.message_id
    )

    branch_node = await driver.workflow.tree_queue.get_node(
        branch.scope,
        branch.message_id,
    )
    assert branch_node is not None
    assert branch_node.parent_id == root.message_id
    assert driver.cli_manager.sessions[1].calls[0]["session_id"] == "sess_root"
    assert driver.cli_manager.sessions[1].calls[0]["fork_session"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_messaging_queued_scoped_cancel_e2e(
    platform_name: str,
    tmp_path,
    monkeypatch,
) -> None:
    driver = FakePlatformDriver(platform_name, tmp_path)
    root_started = asyncio.Event()
    release_root = asyncio.Event()

    class GatedRootSession(FakeCLISession):
        async def start_task(
            self,
            prompt: str,
            session_id: str | None = None,
            fork_session: bool = False,
        ):
            self.calls.append(
                {
                    "prompt": prompt,
                    "session_id": session_id,
                    "fork_session": fork_session,
                }
            )
            self.is_busy = True
            root_started.set()
            try:
                await release_root.wait()
                for event in self.events:
                    await asyncio.sleep(0)
                    yield event
            finally:
                self.is_busy = False

    async def controlled_session(session_id: str | None = None):
        index = len(driver.cli_manager.sessions)
        events = default_cli_events(f"sess_{index}")
        session = GatedRootSession(events) if index == 0 else FakeCLISession(events)
        driver.cli_manager.sessions.append(session)
        return session, session_id or f"pending_{index}", session_id is None

    monkeypatch.setattr(
        driver.cli_manager,
        "get_or_create_session",
        controlled_session,
    )

    root = await driver.emit("root", message_id="root")
    await root_started.wait()
    cancelled = await driver.emit(
        "cancel me",
        message_id="cancelled",
        reply_to=root.message_id,
    )
    cancelled_status_id = driver.platform.sent[-1]["message_id"]
    survivor = await driver.emit(
        "run me",
        message_id="survivor",
        reply_to=root.message_id,
    )
    sent_before_stop = len(driver.platform.sent)
    await driver.emit(
        "/stop",
        message_id="stop-cancelled",
        reply_to=cancelled.message_id,
    )
    assert len(driver.platform.sent) == sent_before_stop

    release_root.set()
    await driver.wait_for_idle()

    prompts = [
        call["prompt"]
        for session in driver.cli_manager.sessions
        for call in session.calls
    ]
    assert prompts == ["root", "run me"]
    cancelled_view = await driver.workflow.tree_queue.get_node(
        cancelled.scope,
        cancelled.message_id,
    )
    survivor_view = await driver.workflow.tree_queue.get_node(
        survivor.scope,
        survivor.message_id,
    )
    assert cancelled_view is not None
    assert cancelled_view.state is MessageState.ERROR
    assert survivor_view is not None
    assert survivor_view.state is MessageState.COMPLETED

    rendered = "\n".join(
        entry["text"] for entry in driver.platform.sent + driver.platform.edits
    )
    assert "position 1" in rendered
    assert "position 2" in rendered
    assert "Stopped" in rendered
    assert any(
        edit["message_id"] == cancelled_status_id and "Stopped" in edit["text"]
        for edit in driver.platform.edits
    )
    driver.session_store.flush_pending_save()
    persisted = driver.session_store.load_conversation_snapshot()
    root_identity = TreeIdentity(scope=root.scope, root_id=root.message_id)
    assert persisted.trees[root_identity].nodes["survivor"]["state"] == "completed"


@pytest.mark.asyncio
async def test_restart_restore_and_session_persistence_e2e(tmp_path) -> None:
    first = FakePlatformDriver("telegram", tmp_path)
    root = await first.send("persist me", message_id="root_1")
    first.session_store.flush_pending_save()

    session_file = tmp_path / "telegram-sessions.json"
    payload = json.loads(session_file.read_text(encoding="utf-8"))
    assert payload["conversation"]["trees"]
    assert payload["managed_messages"]

    restored = FakePlatformDriver("telegram", tmp_path)
    restored.platform.continue_message_sequence_after(first.platform)
    restored.workflow.restore()
    saved = restored.session_store.load_conversation_snapshot()
    assert saved.trees
    identity = TreeIdentity(scope=root.scope, root_id=root.message_id)
    assert saved.get_tree(identity) is not None

    reply = await restored.send(
        "continue from disk",
        message_id="reply_1",
        reply_to=root.message_id,
    )
    reply_view = await restored.workflow.tree_queue.get_node(
        reply.scope,
        reply.message_id,
    )
    assert reply_view is not None
    assert reply_view.parent_id == root.message_id
    call = restored.cli_manager.sessions[0].calls[0]
    assert call["session_id"] == "fake_session_1"
    assert call["fork_session"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_same_message_ids_are_isolated_by_chat_e2e(
    platform_name: str,
    tmp_path,
) -> None:
    driver = FakePlatformDriver(platform_name, tmp_path)
    first = await driver.send("first chat", chat_id="chat_a", message_id="42")
    second = await driver.send("second chat", chat_id="chat_b", message_id="42")

    snapshot = await driver.workflow.tree_queue.snapshot()
    assert set(snapshot.trees) == {
        TreeIdentity(scope=first.scope, root_id="42"),
        TreeIdentity(scope=second.scope, root_id="42"),
    }

    reply = await driver.send(
        "first chat reply",
        chat_id="chat_a",
        message_id="43",
        reply_to="42",
    )
    reply_view = await driver.workflow.tree_queue.get_node(reply.scope, "43")
    assert reply_view is not None
    assert reply_view.parent_id == "42"


@pytest.mark.asyncio
@pytest.mark.parametrize("platform_name", ["discord", "telegram"])
async def test_voice_platform_fake_e2e(platform_name: str, tmp_path) -> None:
    driver = FakePlatformDriver(platform_name, tmp_path)
    driver.platform.seed_pending_voice("chat_1", "voice_msg_1", "voice_status_1")

    await driver.send("/clear", message_id="clear_voice", reply_to="voice_msg_1")

    driver.platform.seed_pending_voice("chat_1", "voice_msg_2", "voice_status_2")
    sent_before_stop = len(driver.platform.sent)
    await driver.send("/stop", message_id="stop_voice")

    deleted = {entry["message_id"] for entry in driver.platform.deletes}
    assert {"voice_msg_1", "voice_status_1", "clear_voice"} <= deleted
    assert driver.platform.pending_voice_count == 0
    sent_text = "\n".join(sent["text"] for sent in driver.platform.sent)
    assert "Voice note cancelled" not in sent_text
    assert len(driver.platform.sent) == sent_before_stop
    assert any(
        edit["message_id"] == "voice_status_2" and "Stopped" in edit["text"]
        for edit in driver.platform.edits
    )

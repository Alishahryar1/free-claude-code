import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.platforms.voice_flow import (
    VOICE_DISABLED_MESSAGE,
    VOICE_TRANSCRIPTION_ERROR_MESSAGE,
    VoiceNoteFlow,
    VoiceNoteRequest,
    audio_suffix_from_metadata,
    is_audio_metadata,
)
from free_claude_code.messaging.trees.runtime import MessageTree
from free_claude_code.messaging.voice import Transcriber
from free_claude_code.messaging.workflow import MessagingWorkflow

VOICE_SCOPE = MessageScope(platform="telegram", chat_id="chat")


class FatalVoiceError(BaseException):
    """Fatal sentinel used to verify ownership finalization."""


class MockTranscriber:
    def __init__(self, result: str = "hello from voice") -> None:
        self.run = AsyncMock(return_value=result)
        self.close_run = AsyncMock()
        self.paths: list[Path] = []

    async def transcribe(self, file_path: Path) -> str:
        self.paths.append(file_path)
        return await self.run(file_path)

    async def close(self) -> None:
        await self.close_run()


def _flow(*, enabled: bool = True) -> tuple[VoiceNoteFlow, MockTranscriber]:
    transcriber = MockTranscriber()
    configured: Transcriber | None = transcriber if enabled else None
    return (
        VoiceNoteFlow(
            transcriber=configured,
            log_raw_messaging_content=False,
            log_api_error_tracebacks=False,
        ),
        transcriber,
    )


def _request(
    *,
    download_to=None,
    reply_text=None,
    message_id: str = "voice",
) -> VoiceNoteRequest:
    async def default_download_to(path: Path) -> None:
        path.write_bytes(b"voice")

    return VoiceNoteRequest(
        platform="telegram",
        chat_id="chat",
        user_id="user",
        message_id=message_id,
        raw_event={"raw": True},
        content_type="audio/ogg",
        temp_suffix=".ogg",
        status_text="transcribing",
        status_parse_mode="MarkdownV2",
        message_thread_id="thread",
        reply_to_message_id="reply",
        download_to=download_to or default_download_to,
        reply_text=reply_text or AsyncMock(),
    )


@pytest.mark.asyncio
async def test_voice_flow_success_builds_incoming_message() -> None:
    flow, transcriber = _flow()
    handler = AsyncMock()
    queue_send = AsyncMock(return_value="status")
    queue_delete = AsyncMock()
    downloaded_paths: list[Path] = []

    async def download_to(path: Path) -> None:
        downloaded_paths.append(path)
        path.write_bytes(b"voice")

    handled = await flow.handle(
        _request(download_to=download_to),
        message_handler=handler,
        queue_send_message=queue_send,
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    queue_send.assert_awaited_once_with(
        "chat",
        "transcribing",
        reply_to="voice",
        parse_mode="MarkdownV2",
        fire_and_forget=False,
        message_thread_id="thread",
    )
    queue_delete.assert_not_awaited()
    handler.assert_awaited_once()
    incoming = handler.call_args.args[0]
    assert incoming.text == "hello from voice"
    assert incoming.chat_id == "chat"
    assert incoming.message_id == "voice"
    assert incoming.reply_to_message_id == "reply"
    assert incoming.message_thread_id == "thread"
    assert incoming.status_message_id == "status"
    transcriber.run.assert_awaited_once()
    assert transcriber.paths == downloaded_paths
    assert downloaded_paths and not downloaded_paths[0].exists()


@pytest.mark.asyncio
async def test_voice_flow_disabled_replies_without_transcribing() -> None:
    flow, transcriber = _flow(enabled=False)
    reply_text = AsyncMock()

    handled = await flow.handle(
        _request(reply_text=reply_text),
        message_handler=AsyncMock(),
        queue_send_message=AsyncMock(),
        queue_delete_messages=AsyncMock(),
    )

    assert handled is True
    reply_text.assert_awaited_once_with(VOICE_DISABLED_MESSAGE)
    transcriber.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_flow_missing_status_id_stops_before_transcription() -> None:
    flow, transcriber = _flow()
    reply_text = AsyncMock()
    handler = AsyncMock()
    queue_delete = AsyncMock()

    handled = await flow.handle(
        _request(reply_text=reply_text),
        message_handler=handler,
        queue_send_message=AsyncMock(return_value=None),
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    transcriber.run.assert_not_awaited()
    handler.assert_not_awaited()
    queue_delete.assert_not_awaited()
    reply_text.assert_awaited_once_with(VOICE_TRANSCRIPTION_ERROR_MESSAGE)
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None


@pytest.mark.asyncio
async def test_voice_flow_cancelled_transcription_preserves_owned_status() -> None:
    flow, transcriber = _flow()

    async def canceling_transcribe(_path: Path) -> str:
        await flow.cancel_pending_voice(VOICE_SCOPE, "voice")
        return "ignored"

    transcriber.run.side_effect = canceling_transcribe
    handler = AsyncMock()
    queue_send = AsyncMock(return_value="status")
    queue_delete = AsyncMock()

    handled = await flow.handle(
        _request(),
        message_handler=handler,
        queue_send_message=queue_send,
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    handler.assert_not_awaited()
    queue_delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_flow_cancel_during_status_delivery_prevents_transcription() -> (
    None
):
    flow, transcriber = _flow()
    status_send_started = asyncio.Event()
    release_status_send = asyncio.Event()
    handler = AsyncMock()
    queue_delete = AsyncMock()

    async def send_status(*_args, **_kwargs) -> str:
        status_send_started.set()
        await release_status_send.wait()
        return "status"

    handle_task = asyncio.create_task(
        flow.handle(
            _request(),
            message_handler=handler,
            queue_send_message=send_status,
            queue_delete_messages=queue_delete,
        )
    )

    try:
        await asyncio.wait_for(status_send_started.wait(), timeout=1)
        cancelled = await flow.cancel_pending_voice(VOICE_SCOPE, "voice")

        assert cancelled is not None
        assert cancelled.voice_message_id == "voice"
        assert cancelled.status_message_id is None

        release_status_send.set()
        assert await asyncio.wait_for(handle_task, timeout=1) is True
    finally:
        release_status_send.set()
        if not handle_task.done():
            handle_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await handle_task

    transcriber.run.assert_not_awaited()
    handler.assert_not_awaited()
    queue_delete.assert_awaited_once_with("chat", ["status"])


@pytest.mark.asyncio
async def test_voice_flow_cancel_during_status_delivery_suppresses_late_failure() -> (
    None
):
    flow, transcriber = _flow()
    status_send_started = asyncio.Event()
    release_status_send = asyncio.Event()
    reply_text = AsyncMock()
    handler = AsyncMock()

    async def fail_status_send(*_args, **_kwargs) -> str:
        status_send_started.set()
        await release_status_send.wait()
        raise RuntimeError("late status failure")

    handle_task = asyncio.create_task(
        flow.handle(
            _request(reply_text=reply_text),
            message_handler=handler,
            queue_send_message=fail_status_send,
            queue_delete_messages=AsyncMock(),
        )
    )

    try:
        await asyncio.wait_for(status_send_started.wait(), timeout=1)
        assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is not None

        release_status_send.set()
        assert await asyncio.wait_for(handle_task, timeout=1) is True
    finally:
        release_status_send.set()
        if not handle_task.done():
            handle_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await handle_task

    transcriber.run.assert_not_awaited()
    handler.assert_not_awaited()
    reply_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_flow_cancel_during_transcription_suppresses_late_failure() -> None:
    flow, transcriber = _flow()
    transcription_started = asyncio.Event()
    release_transcription = asyncio.Event()
    reply_text = AsyncMock()
    handler = AsyncMock()
    queue_delete = AsyncMock()

    async def fail_transcription(_path: Path) -> str:
        transcription_started.set()
        await release_transcription.wait()
        raise RuntimeError("late transcription failure")

    transcriber.run.side_effect = fail_transcription
    handle_task = asyncio.create_task(
        flow.handle(
            _request(reply_text=reply_text),
            message_handler=handler,
            queue_send_message=AsyncMock(return_value="status"),
            queue_delete_messages=queue_delete,
        )
    )

    try:
        await asyncio.wait_for(transcription_started.wait(), timeout=1)
        cancelled = await flow.cancel_pending_voice(VOICE_SCOPE, "voice")
        assert cancelled is not None
        assert cancelled.status_message_id == "status"

        release_transcription.set()
        assert await asyncio.wait_for(handle_task, timeout=1) is True
    finally:
        release_transcription.set()
        if not handle_task.done():
            handle_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await handle_task

    handler.assert_not_awaited()
    reply_text.assert_not_awaited()
    queue_delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_stop_status_survives_late_transcription_success(
    mock_platform,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
) -> None:
    flow, transcriber = _flow()
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
        platform_name="telegram",
        voice_cancellation=flow,
    )
    transcription_started = asyncio.Event()
    release_transcription = asyncio.Event()

    async def delayed_transcription(_path: Path) -> str:
        transcription_started.set()
        await release_transcription.wait()
        return "late voice prompt"

    transcriber.run.side_effect = delayed_transcription
    mock_platform.queue_send_message.return_value = "voice_status"
    voice_task = asyncio.create_task(
        flow.handle(
            _request(),
            message_handler=workflow.handle_message,
            queue_send_message=mock_platform.queue_send_message,
            queue_delete_messages=mock_platform.queue_delete_messages,
        )
    )
    await asyncio.wait_for(transcription_started.wait(), timeout=1)

    try:
        await workflow.handle_message(
            incoming_message_factory(
                text="/stop",
                chat_id="chat",
                message_id="stop_command",
                reply_to_message_id="voice",
            )
        )
    finally:
        release_transcription.set()
    assert await asyncio.wait_for(voice_task, timeout=1) is True
    await asyncio.sleep(0)

    stopped = workflow.format_status("⏹", "Stopped.")
    assert any(
        call.args[:3] == ("chat", "voice_status", stopped)
        for call in mock_platform.queue_edit_message.await_args_list
    )
    mock_platform.queue_delete_messages.assert_not_awaited()
    assert await workflow.tree_queue.resolve_node_id(VOICE_SCOPE, "voice") is None


@pytest.mark.asyncio
async def test_voice_flow_cancel_during_handoff_stops_and_drains_handler() -> None:
    flow, _transcriber = _flow()
    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()
    release_handler = asyncio.Event()
    queue_delete = AsyncMock()

    async def handler(_incoming) -> None:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            handler_cancelled.set()
            await release_handler.wait()
            raise

    handle_task = asyncio.create_task(
        flow.handle(
            _request(),
            message_handler=handler,
            queue_send_message=AsyncMock(return_value="status"),
            queue_delete_messages=queue_delete,
        )
    )

    try:
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        cancel_task = asyncio.create_task(
            flow.cancel_pending_voice(VOICE_SCOPE, "voice")
        )
        await asyncio.wait_for(handler_cancelled.wait(), timeout=1)

        assert not cancel_task.done()
        release_handler.set()
        cancelled = await asyncio.wait_for(cancel_task, timeout=1)
        assert cancelled is not None
        assert cancelled.status_message_id == "status"
        assert await asyncio.wait_for(handle_task, timeout=1) is True
    finally:
        release_handler.set()
        if not handle_task.done():
            handle_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await handle_task

    queue_delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_flow_cancel_during_handoff_suppresses_late_handler_error() -> None:
    flow, _transcriber = _flow()
    handler_started = asyncio.Event()
    reply_text = AsyncMock()

    async def handler(_incoming) -> None:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError("late handler failure") from None

    handle_task = asyncio.create_task(
        flow.handle(
            _request(reply_text=reply_text),
            message_handler=handler,
            queue_send_message=AsyncMock(return_value="status"),
            queue_delete_messages=AsyncMock(),
        )
    )

    await asyncio.wait_for(handler_started.wait(), timeout=1)
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "status") is not None
    assert await asyncio.wait_for(handle_task, timeout=1) is True
    reply_text.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/clear"])
async def test_reply_command_cancels_voice_at_tree_admission_commit(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    mock_platform,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
) -> None:
    flow, _transcriber = _flow()
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
        platform_name="telegram",
        voice_cancellation=flow,
    )
    admission_mutated = asyncio.Event()
    release_admission = asyncio.Event()
    original_enqueue_or_claim = MessageTree.enqueue_or_claim

    async def mutate_then_block(tree: MessageTree, node_id: str):
        decision = await original_enqueue_or_claim(tree, node_id)
        if node_id == "voice":
            admission_mutated.set()
            await release_admission.wait()
        return decision

    monkeypatch.setattr(MessageTree, "enqueue_or_claim", mutate_then_block)
    mock_platform.queue_send_message.return_value = "voice_status"
    voice_task = asyncio.create_task(
        flow.handle(
            _request(),
            message_handler=workflow.handle_message,
            queue_send_message=mock_platform.queue_send_message,
            queue_delete_messages=mock_platform.queue_delete_messages,
        )
    )
    await asyncio.wait_for(admission_mutated.wait(), timeout=1)

    command_task = asyncio.create_task(
        workflow.handle_message(
            incoming_message_factory(
                text=command,
                chat_id="chat",
                message_id="command",
                reply_to_message_id="voice",
            )
        )
    )
    try:
        await asyncio.sleep(0)
        assert not command_task.done()
    finally:
        release_admission.set()
        await asyncio.wait_for(command_task, timeout=1)
        voice_result = await asyncio.wait_for(voice_task, timeout=1)
    assert voice_result is True
    await asyncio.sleep(0)

    mock_session_store.save_tree_snapshot.assert_called()
    if command == "/clear":
        assert workflow.get_tree_count() == 0
        assert await workflow.tree_queue.resolve_node_id(VOICE_SCOPE, "voice") is None
    else:
        assert workflow.get_tree_count() == 1
        assert (
            await workflow.tree_queue.resolve_node_id(VOICE_SCOPE, "voice") == "voice"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/clear"])
async def test_global_command_rejects_transcription_that_finishes_late(
    command: str,
    mock_platform,
    mock_cli_manager,
    mock_session_store,
    incoming_message_factory,
) -> None:
    flow, transcriber = _flow()
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
        platform_name="telegram",
        voice_cancellation=flow,
    )
    transcription_started = asyncio.Event()
    release_transcription = asyncio.Event()

    async def delayed_transcription(_path: Path) -> str:
        transcription_started.set()
        await release_transcription.wait()
        return "late voice prompt"

    transcriber.run.side_effect = delayed_transcription
    mock_platform.queue_send_message.return_value = "voice_status"
    voice_task = asyncio.create_task(
        flow.handle(
            _request(),
            message_handler=workflow.handle_message,
            queue_send_message=mock_platform.queue_send_message,
            queue_delete_messages=mock_platform.queue_delete_messages,
        )
    )
    await asyncio.wait_for(transcription_started.wait(), timeout=1)

    await asyncio.wait_for(
        workflow.handle_message(
            incoming_message_factory(
                text=command,
                message_id="command",
                chat_id=VOICE_SCOPE.chat_id,
            )
        ),
        timeout=1,
    )
    release_transcription.set()
    assert await asyncio.wait_for(voice_task, timeout=1) is True

    assert workflow.get_tree_count() == 0
    assert await workflow.tree_queue.resolve_node_id(VOICE_SCOPE, "voice") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/clear"])
async def test_transcribed_global_command_does_not_cancel_its_own_handoff(
    command: str,
    mock_platform,
    mock_cli_manager,
    mock_session_store,
) -> None:
    flow, transcriber = _flow()
    transcriber.run.return_value = command
    workflow = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
        platform_name="telegram",
        voice_cancellation=flow,
    )
    mock_platform.queue_send_message.return_value = "voice_status"

    assert (
        await asyncio.wait_for(
            flow.handle(
                _request(),
                message_handler=workflow.handle_message,
                queue_send_message=mock_platform.queue_send_message,
                queue_delete_messages=mock_platform.queue_delete_messages,
            ),
            timeout=1,
        )
        is True
    )

    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None


@pytest.mark.asyncio
async def test_voice_flow_caller_cancellation_drains_handoff_child() -> None:
    flow, _transcriber = _flow()
    handler_started = asyncio.Event()
    handler_cancelled = asyncio.Event()
    release_handler = asyncio.Event()
    queue_delete = AsyncMock()

    async def handler(_incoming) -> None:
        handler_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            handler_cancelled.set()
            await release_handler.wait()
            raise

    handle_task = asyncio.create_task(
        flow.handle(
            _request(),
            message_handler=handler,
            queue_send_message=AsyncMock(return_value="status"),
            queue_delete_messages=queue_delete,
        )
    )
    await asyncio.wait_for(handler_started.wait(), timeout=1)
    handle_task.cancel()
    await asyncio.wait_for(handler_cancelled.wait(), timeout=1)

    assert not handle_task.done()
    release_handler.set()
    with pytest.raises(asyncio.CancelledError):
        await handle_task
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None
    queue_delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_flow_task_cancellation_waits_then_cleans_pending_state() -> None:
    flow, transcriber = _flow()
    started = asyncio.Event()
    cancellation_received = asyncio.Event()
    release = asyncio.Event()
    stopped = asyncio.Event()

    async def cancellation_safe_transcribe(_path: Path) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
            return "unreachable"
        except asyncio.CancelledError:
            cancellation_received.set()
            await release.wait()
            stopped.set()
            raise

    transcriber.run.side_effect = cancellation_safe_transcribe
    handler = AsyncMock()
    queue_delete = AsyncMock()
    handle_task = asyncio.create_task(
        flow.handle(
            _request(),
            message_handler=handler,
            queue_send_message=AsyncMock(return_value="status"),
            queue_delete_messages=queue_delete,
        )
    )

    await started.wait()
    handle_task.cancel()
    await cancellation_received.wait()

    assert not handle_task.done()
    queue_delete.assert_not_awaited()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await handle_task

    assert stopped.is_set()
    handler.assert_not_awaited()
    queue_delete.assert_awaited_once_with("chat", ["status"])
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None


@pytest.mark.asyncio
async def test_voice_flow_repeated_cancellation_cannot_interrupt_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow, transcriber = _flow()
    transcription_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def blocked_transcription(_path: Path) -> str:
        transcription_started.set()
        await asyncio.Event().wait()
        return "unreachable"

    original_discard = flow._pending_voice.discard

    async def delayed_discard(claim) -> bool:
        cleanup_started.set()
        await release_cleanup.wait()
        return await original_discard(claim)

    transcriber.run.side_effect = blocked_transcription
    monkeypatch.setattr(flow._pending_voice, "discard", delayed_discard)
    queue_delete = AsyncMock()
    handle_task = asyncio.create_task(
        flow.handle(
            _request(),
            message_handler=AsyncMock(),
            queue_send_message=AsyncMock(return_value="status"),
            queue_delete_messages=queue_delete,
        )
    )

    await asyncio.wait_for(transcription_started.wait(), timeout=1)
    handle_task.cancel("first cancellation")
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    handle_task.cancel("second cancellation")
    await asyncio.sleep(0)

    assert not handle_task.done()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(handle_task, timeout=1)

    queue_delete.assert_awaited_once_with("chat", ["status"])
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "status") is None


@pytest.mark.asyncio
async def test_voice_flow_fatal_status_failure_releases_claim() -> None:
    flow, transcriber = _flow()
    queue_delete = AsyncMock()

    async def fatal_status(*_args, **_kwargs) -> str:
        raise FatalVoiceError

    with pytest.raises(FatalVoiceError):
        await flow.handle(
            _request(),
            message_handler=AsyncMock(),
            queue_send_message=fatal_status,
            queue_delete_messages=queue_delete,
        )

    transcriber.run.assert_not_awaited()
    queue_delete.assert_not_awaited()
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None


@pytest.mark.asyncio
async def test_voice_flow_fatal_download_failure_releases_status_ownership() -> None:
    flow, transcriber = _flow()
    queue_delete = AsyncMock()

    async def fatal_download(_path: Path) -> None:
        raise FatalVoiceError

    with pytest.raises(FatalVoiceError):
        await flow.handle(
            _request(download_to=fatal_download),
            message_handler=AsyncMock(),
            queue_send_message=AsyncMock(return_value="status"),
            queue_delete_messages=queue_delete,
        )

    transcriber.run.assert_not_awaited()
    queue_delete.assert_awaited_once_with("chat", ["status"])
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "status") is None


@pytest.mark.asyncio
async def test_voice_flow_fatal_transcription_releases_status_ownership() -> None:
    flow, transcriber = _flow()
    transcriber.run.side_effect = FatalVoiceError()
    handler = AsyncMock()
    queue_delete = AsyncMock()

    with pytest.raises(FatalVoiceError):
        await flow.handle(
            _request(),
            message_handler=handler,
            queue_send_message=AsyncMock(return_value="status"),
            queue_delete_messages=queue_delete,
        )

    handler.assert_not_awaited()
    queue_delete.assert_awaited_once_with("chat", ["status"])
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "status") is None


@pytest.mark.asyncio
async def test_voice_flow_download_failure_cleans_pending_state() -> None:
    flow, transcriber = _flow()
    reply_text = AsyncMock()
    queue_delete = AsyncMock()

    async def failing_download(_path: Path) -> None:
        raise RuntimeError("download failed")

    handled = await flow.handle(
        _request(download_to=failing_download, reply_text=reply_text),
        message_handler=AsyncMock(),
        queue_send_message=AsyncMock(return_value="status"),
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    transcriber.run.assert_not_awaited()
    queue_delete.assert_awaited_once_with("chat", ["status"])
    reply_text.assert_awaited_once_with(VOICE_TRANSCRIPTION_ERROR_MESSAGE)
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None


@pytest.mark.asyncio
async def test_voice_flow_transcription_failure_cleans_pending_state() -> None:
    flow, transcriber = _flow()
    transcriber.run.side_effect = RuntimeError("transcription failed")
    reply_text = AsyncMock()
    queue_delete = AsyncMock()

    handled = await flow.handle(
        _request(reply_text=reply_text),
        message_handler=AsyncMock(),
        queue_send_message=AsyncMock(return_value="status"),
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    queue_delete.assert_awaited_once_with("chat", ["status"])
    reply_text.assert_awaited_once_with(VOICE_TRANSCRIPTION_ERROR_MESSAGE)
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None


@pytest.mark.asyncio
async def test_voice_flow_handler_failure_cleans_pending_without_deleting_status() -> (
    None
):
    flow, _transcriber = _flow()
    reply_text = AsyncMock()
    queue_delete = AsyncMock()

    async def failing_handler(_incoming) -> None:
        raise RuntimeError("handler failed")

    handled = await flow.handle(
        _request(reply_text=reply_text),
        message_handler=failing_handler,
        queue_send_message=AsyncMock(return_value="status"),
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    queue_delete.assert_not_awaited()
    reply_text.assert_awaited_once_with(VOICE_TRANSCRIPTION_ERROR_MESSAGE)
    assert await flow.cancel_pending_voice(VOICE_SCOPE, "voice") is None


@pytest.mark.asyncio
async def test_voice_flow_rejects_oversized_audio_before_transcription(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "free_claude_code.messaging.platforms.voice_flow.MAX_AUDIO_SIZE_BYTES",
        3,
    )
    flow, transcriber = _flow()
    reply_text = AsyncMock()
    queue_delete = AsyncMock()

    async def download(path: Path) -> None:
        path.write_bytes(b"four")

    handled = await flow.handle(
        _request(download_to=download, reply_text=reply_text),
        message_handler=AsyncMock(),
        queue_send_message=AsyncMock(return_value="status"),
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    transcriber.run.assert_not_awaited()
    queue_delete.assert_awaited_once_with("chat", ["status"])
    assert reply_text.await_args is not None
    assert "too large" in reply_text.await_args.args[0]


def test_audio_metadata_helpers() -> None:
    assert is_audio_metadata("voice.ogg", "application/octet-stream") is True
    assert is_audio_metadata("file.txt", "audio/ogg") is True
    assert is_audio_metadata("file.txt", "text/plain") is False
    assert (
        audio_suffix_from_metadata(filename="voice.ogg", content_type="audio/mp4")
        == ".mp4"
    )
    assert (
        audio_suffix_from_metadata(filename="clip.m4a", content_type="audio/mp4")
        == ".m4a"
    )
    assert (
        audio_suffix_from_metadata(filename="clip.m4a", content_type="audio/mpeg")
        == ".mp3"
    )
    assert audio_suffix_from_metadata(content_type="audio/mpeg") == ".mp3"
    assert audio_suffix_from_metadata(filename="clip.m4a") == ".m4a"
    assert audio_suffix_from_metadata(content_type="audio/mp4") == ".mp4"
    assert audio_suffix_from_metadata(content_type="audio/wav") == ".wav"

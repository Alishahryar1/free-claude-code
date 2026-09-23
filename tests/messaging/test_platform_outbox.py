import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.messaging.platforms.outbox import PlatformOutbox


def _noop_outbox(*, limiter=None, delete_many=None) -> PlatformOutbox:
    async def send(
        chat_id: str,
        text: str,
        reply_to: str | None,
        parse_mode: str | None,
        message_thread_id: str | None,
    ) -> str:
        return f"{chat_id}:{text}:{reply_to}:{parse_mode}:{message_thread_id}"

    async def edit(
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None,
    ) -> None:
        return None

    async def default_delete_many(chat_id: str, message_ids: list[str]) -> None:
        return None

    return PlatformOutbox(
        limiter=limiter or MagicMock(),
        send=send,
        edit=edit,
        delete_many=delete_many or default_delete_many,
    )


@pytest.mark.asyncio
async def test_queue_send_awaits_required_limiter() -> None:
    limiter = MagicMock()

    async def enqueue(operation, dedup_key=None):
        return await operation()

    limiter.enqueue = AsyncMock(side_effect=enqueue)
    outbox = _noop_outbox(limiter=limiter)

    result = await outbox.queue_send_message(
        "chat",
        "hello",
        reply_to="reply",
        parse_mode="MarkdownV2",
        fire_and_forget=False,
        message_thread_id="thread",
    )

    assert result == "chat:hello:reply:MarkdownV2:thread"
    limiter.enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_queue_edit_awaits_limiter_with_dedup_key() -> None:
    limiter = MagicMock()
    limiter.enqueue = AsyncMock()
    outbox = _noop_outbox(limiter=limiter)

    await outbox.queue_edit_message(
        "chat",
        "message",
        "updated",
        parse_mode="MarkdownV2",
        fire_and_forget=False,
    )

    limiter.enqueue.assert_awaited_once()
    operation = limiter.enqueue.call_args.args[0]
    assert limiter.enqueue.call_args.kwargs["dedup_key"] == "edit:chat:message"
    await operation()


@pytest.mark.asyncio
async def test_queue_delete_many_skips_empty_batches() -> None:
    limiter = MagicMock()
    outbox = _noop_outbox(limiter=limiter)

    await outbox.queue_delete_messages("chat", [], fire_and_forget=True)

    limiter.fire_and_forget.assert_not_called()


@pytest.mark.asyncio
async def test_queue_delete_many_dedups_by_batch() -> None:
    limiter = MagicMock()
    outbox = _noop_outbox(limiter=limiter)

    await outbox.queue_delete_messages("chat", ["1", "2"], fire_and_forget=True)

    limiter.fire_and_forget.assert_called_once()
    assert (
        limiter.fire_and_forget.call_args.kwargs["dedup_key"]
        == "del_bulk:chat:11f0530a8259fffb"
    )


@pytest.mark.asyncio
async def test_queue_delete_many_snapshots_ids_before_queueing() -> None:
    limiter = MagicMock()
    deleted: list[list[str]] = []

    async def delete_many(_chat_id: str, message_ids: list[str]) -> None:
        deleted.append(message_ids)

    outbox = _noop_outbox(limiter=limiter, delete_many=delete_many)
    message_ids = ["1", "2"]

    await outbox.queue_delete_messages("chat", message_ids, fire_and_forget=True)
    message_ids.append("3")
    operation = limiter.fire_and_forget.call_args.args[0]
    await operation()

    assert deleted == [["1", "2"]]


@pytest.mark.asyncio
async def test_close_cancels_and_settles_owned_background_work() -> None:
    outbox = _noop_outbox()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def pending() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    outbox.fire_and_forget(pending())
    await started.wait()

    await outbox.close()

    assert cancelled.is_set()
    assert outbox._background_tasks == set()


@pytest.mark.asyncio
async def test_completed_background_failure_is_observed_and_released() -> None:
    outbox = _noop_outbox()

    async def fail() -> None:
        raise RuntimeError("background failed")

    with patch("free_claude_code.messaging.platforms.outbox.logger.error") as error_log:
        outbox.fire_and_forget(fail())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert outbox._background_tasks == set()
    error_log.assert_called_once_with(
        "Outbound background task failed: exc_type={}",
        "RuntimeError",
    )


@pytest.mark.asyncio
async def test_close_rejects_all_later_work() -> None:
    limiter = MagicMock()
    outbox = _noop_outbox(limiter=limiter)
    await outbox.close()

    with pytest.raises(RuntimeError, match="outbox is closed"):
        await outbox.queue_send_message("chat", "message")

    ran = False

    async def late_task() -> None:
        nonlocal ran
        ran = True

    with pytest.raises(RuntimeError, match="outbox is closed"):
        outbox.fire_and_forget(late_task())
    await asyncio.sleep(0)

    assert ran is False
    assert outbox._background_tasks == set()
    limiter.fire_and_forget.assert_not_called()

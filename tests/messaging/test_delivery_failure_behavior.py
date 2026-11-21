"""What a failed Telegram call, and the queue behind it, do next.

Raised with python-telegram-bot's own exception classes: ``BadRequest`` is a
``NetworkError`` subclass, so tests with a bare ``TelegramError`` miss how the
library actually reports a rejected request.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from telegram.error import (
    BadRequest,
    Forbidden,
    NetworkError,
    RetryAfter,
    TelegramError,
    TimedOut,
)

from free_claude_code.messaging.delivery_failures import PlatformRateLimited
from free_claude_code.messaging.limiter import MessagingRateLimiter
from free_claude_code.messaging.platforms import telegram_io
from tests.messaging.test_telegram_edge_cases import _telegram_runtime

OK = object()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch):
    sleep = AsyncMock()
    monkeypatch.setattr(telegram_io.asyncio, "sleep", sleep)
    return sleep


async def _run(
    outcomes: Sequence[object], *, parse_mode: str | None = None, suppress: bool = True
):
    """Call _with_retry against scripted outcomes; return the result and calls."""
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        messenger = _telegram_runtime(bot_token="t").outbound
    calls: list[str | None] = []
    script = iter(outcomes)

    async def call(parse_mode: str | None = parse_mode):
        calls.append(parse_mode)
        outcome = next(script)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    try:
        result = await messenger._with_retry(
            call, parse_mode=parse_mode, suppress_known_message_errors=suppress
        )
    except Exception as error:
        return error, calls
    return result, calls


def _delays(sleep: AsyncMock) -> list[float]:
    return [call.args[0] for call in sleep.call_args_list]


@pytest.mark.asyncio
async def test_network_errors_back_off_then_succeed(sleeps: AsyncMock):
    result, calls = await _run([TimedOut(), NetworkError("reset"), OK])

    assert result is OK
    assert len(calls) == 3
    assert _delays(sleeps) == [1, 2]


@pytest.mark.asyncio
async def test_network_errors_raise_after_three_attempts(sleeps: AsyncMock):
    error = NetworkError("reset")
    result, calls = await _run([error, error, error])

    assert result is error
    assert len(calls) == 3
    assert _delays(sleeps) == [1, 2]


@pytest.mark.asyncio
async def test_unparseable_markdown_is_resent_as_plain_text(sleeps: AsyncMock):
    error = BadRequest("Can't parse entities: unsupported start tag")
    result, calls = await _run([error, OK], parse_mode="MarkdownV2")

    assert result is OK
    assert calls == ["MarkdownV2", None]
    assert sleeps.call_count == 0


@pytest.mark.asyncio
async def test_edit_of_a_missing_message_is_ignored(sleeps: AsyncMock):
    result, calls = await _run([BadRequest("Message to edit not found")])

    assert result is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_other_bad_requests_raise_at_once(sleeps: AsyncMock):
    error = BadRequest("Chat not found")
    result, calls = await _run([error])

    assert result is error
    assert len(calls) == 1
    assert sleeps.call_count == 0


@pytest.mark.asyncio
async def test_unmodified_message_is_ignored(sleeps: AsyncMock):
    result, calls = await _run(
        [BadRequest("Message is not modified: specified new message content")]
    )

    assert result is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_forbidden_delete_is_ignored(sleeps: AsyncMock):
    result, calls = await _run([Forbidden("Not enough rights to delete message")])

    assert result is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_known_errors_raise_when_suppression_is_off(sleeps: AsyncMock):
    error = Forbidden("Message can't be deleted for everyone")
    result, calls = await _run([error], suppress=False)

    assert result is error
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_rate_limit_waits_then_retries(sleeps: AsyncMock):
    result, calls = await _run([RetryAfter(5), OK])

    assert result is OK
    assert len(calls) == 2
    assert _delays(sleeps) == [5]


@pytest.mark.asyncio
async def test_rate_limits_share_the_three_call_budget(sleeps: AsyncMock):
    result, calls = await _run([RetryAfter(5), RetryAfter(7), OK])

    assert result is OK
    assert len(calls) == 3
    assert _delays(sleeps) == [5, 7]


@pytest.mark.asyncio
async def test_third_rate_limit_hands_its_wait_to_the_queue(sleeps: AsyncMock):
    third = RetryAfter(9)
    result, calls = await _run([RetryAfter(5), TimedOut(), third])

    assert isinstance(result, PlatformRateLimited)
    assert result.retry_after == 9
    assert result.__cause__ is third
    assert len(calls) == 3
    assert _delays(sleeps) == [5, 2]


@pytest.mark.asyncio
async def test_failed_plain_resend_is_still_retried(sleeps: AsyncMock):
    from telegram.error import TelegramError

    result, calls = await _run(
        [TelegramError("Can't parse entities: bad"), TimedOut(), OK],
        parse_mode="MarkdownV2",
    )

    assert result is OK
    assert calls == ["MarkdownV2", None, None]
    assert _delays(sleeps) == [1]


@pytest.mark.asyncio
async def test_unmodified_message_is_ignored_in_any_case(sleeps: AsyncMock):
    result, calls = await _run([NetworkError("message is not modified")])

    assert result is None
    assert len(calls) == 1


async def _queue_pause(error: Exception) -> float:
    """Fail one queued job with ``error``; return the pause it left behind."""
    limiter = MessagingRateLimiter(rate_limit=100, rate_window=1.0)
    limiter.start()
    try:

        async def fail():
            raise error

        with pytest.raises(type(error)):
            await limiter.enqueue(fail)
        await asyncio.sleep(0)
        loop = asyncio.get_running_loop()
        return max(0.0, limiter._paused_until - loop.time())
    finally:
        await limiter.shutdown(timeout=1)


@pytest.mark.asyncio
async def test_platform_rate_limit_pauses_the_queue_for_its_wait():
    pause = await _queue_pause(PlatformRateLimited(5))

    assert 4 < pause <= 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ValueError("Timed out waiting for the lock"),
        ValueError("Flood wait"),
        RetryAfter(5),
        ValueError("bad payload"),
    ],
    ids=["wait-wording", "flood-wording", "raw-retry-after", "unrelated"],
)
async def test_other_errors_do_not_pause_the_queue(error: Exception):
    assert await _queue_pause(error) == 0


@dataclass
class _DeleteBot:
    delete_messages: AsyncMock
    delete_message: AsyncMock


def _bulk_messenger(bulk_outcomes: Sequence[TelegramError]):
    """A messenger whose bot fails bulk deletes as scripted and logs single ones."""
    bulk = iter(bulk_outcomes)

    async def delete_messages(*, chat_id: object, message_ids: object):
        outcome = next(bulk)
        if isinstance(outcome, BaseException):
            raise outcome

    bot = _DeleteBot(
        delete_messages=AsyncMock(side_effect=delete_messages),
        delete_message=AsyncMock(),
    )
    messenger = telegram_io.TelegramMessenger(
        get_application=lambda: SimpleNamespace(bot=bot),
        limiter=MessagingRateLimiter(rate_limit=100, rate_window=1.0),
    )
    return messenger, bot


@pytest.mark.asyncio
async def test_rate_limited_bulk_delete_hands_its_wait_to_the_queue(sleeps: AsyncMock):
    messenger, bot = _bulk_messenger([RetryAfter(5), RetryAfter(6), RetryAfter(7)])

    with pytest.raises(PlatformRateLimited) as raised:
        await messenger.delete_messages("1", ["10", "11"])

    assert raised.value.retry_after == 7
    assert bot.delete_message.await_count == 0
    assert _delays(sleeps) == [5, 6]


@pytest.mark.asyncio
async def test_rejected_bulk_delete_falls_back_to_single_deletes(sleeps: AsyncMock):
    messenger, bot = _bulk_messenger([BadRequest("Method not available")])

    await messenger.delete_messages("1", ["10", "11"])

    assert [
        call.kwargs["message_id"] for call in bot.delete_message.await_args_list
    ] == [
        10,
        11,
    ]

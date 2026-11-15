"""What a failed Telegram call, and the queue behind it, do next.

Raised with python-telegram-bot's own exception classes: ``BadRequest`` is a
``NetworkError`` subclass, so tests with a bare ``TelegramError`` miss how the
library actually reports a rejected request.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from telegram.error import BadRequest, Forbidden, NetworkError, RetryAfter, TimedOut

from free_claude_code.messaging.limiter import MessagingRateLimiter
from free_claude_code.messaging.platforms import telegram_io
from tests.messaging.test_telegram_edge_cases import _telegram_runtime

OK = object()


@pytest.fixture
def sleeps(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(telegram_io.asyncio, "sleep", sleep)
    return sleep


async def _run(outcomes, *, parse_mode=None, suppress=True):
    """Call _with_retry against scripted outcomes; return the result and calls."""
    with patch(
        "free_claude_code.messaging.platforms.telegram.TELEGRAM_AVAILABLE", True
    ):
        messenger = _telegram_runtime(bot_token="t").outbound
    calls: list[str | None] = []
    script = iter(outcomes)

    async def call(parse_mode=parse_mode):
        calls.append(parse_mode)
        outcome = next(script)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    kwargs = {"parse_mode": parse_mode} if parse_mode is not None else {}
    try:
        result = await messenger._with_retry(
            call, suppress_known_message_errors=suppress, **kwargs
        )
    except Exception as error:
        return error, calls
    return result, calls


def _delays(sleep: AsyncMock) -> list[float]:
    return [call.args[0] for call in sleep.call_args_list]


@pytest.mark.asyncio
async def test_network_errors_back_off_then_succeed(sleeps):
    result, calls = await _run([TimedOut(), NetworkError("reset"), OK])

    assert result is OK
    assert len(calls) == 3
    assert _delays(sleeps) == [1, 2]


@pytest.mark.asyncio
async def test_network_errors_raise_after_three_attempts(sleeps):
    error = NetworkError("reset")
    result, calls = await _run([error, error, error])

    assert result is error
    assert len(calls) == 3
    assert _delays(sleeps) == [1, 2]


@pytest.mark.asyncio
async def test_unparseable_markdown_is_retried_as_a_network_error(sleeps):
    error = BadRequest("Can't parse entities: unsupported start tag")
    result, calls = await _run([error, error, error], parse_mode="MarkdownV2")

    assert result is error
    assert calls == ["MarkdownV2"] * 3
    assert _delays(sleeps) == [1, 2]


@pytest.mark.asyncio
async def test_edit_of_a_missing_message_is_retried_then_raised(sleeps):
    error = BadRequest("Message to edit not found")
    result, calls = await _run([error, error, error])

    assert result is error
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_other_bad_requests_are_retried_then_raised(sleeps):
    error = BadRequest("Chat not found")
    result, calls = await _run([error, error, error])

    assert result is error
    assert len(calls) == 3
    assert _delays(sleeps) == [1, 2]


@pytest.mark.asyncio
async def test_unmodified_message_is_ignored(sleeps):
    result, calls = await _run(
        [BadRequest("Message is not modified: specified new message content")]
    )

    assert result is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_forbidden_delete_is_ignored(sleeps):
    result, calls = await _run([Forbidden("Not enough rights to delete message")])

    assert result is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_known_errors_raise_when_suppression_is_off(sleeps):
    error = Forbidden("Message can't be deleted for everyone")
    result, calls = await _run([error], suppress=False)

    assert result is error
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_rate_limit_waits_then_retries(sleeps):
    result, calls = await _run([RetryAfter(5), OK])

    assert result is OK
    assert len(calls) == 2
    assert _delays(sleeps) == [5]


@pytest.mark.asyncio
async def test_second_rate_limit_escapes_the_retry_policy(sleeps):
    second = RetryAfter(7)
    result, calls = await _run([RetryAfter(5), second])

    assert result is second
    assert len(calls) == 2
    assert _delays(sleeps) == [5]


@pytest.mark.asyncio
async def test_failed_plain_retry_escapes_the_retry_policy(sleeps):
    from telegram.error import TelegramError

    timeout = TimedOut()
    result, calls = await _run(
        [TelegramError("Can't parse entities: bad"), timeout],
        parse_mode="MarkdownV2",
    )

    assert result is timeout
    assert calls == ["MarkdownV2", None]
    assert sleeps.call_count == 0


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
async def test_telegram_rate_limit_pauses_the_queue_for_thirty_seconds():
    pause = await _queue_pause(RetryAfter(5))

    assert 29 < pause <= 30


@pytest.mark.asyncio
async def test_any_error_mentioning_wait_pauses_the_queue():
    pause = await _queue_pause(ValueError("Timed out waiting for the lock"))

    assert 29 < pause <= 30


@pytest.mark.asyncio
async def test_non_numeric_wait_seconds_leave_the_queue_unpaused():
    class FloodWait(Exception):
        seconds = "soon"

    pause = await _queue_pause(FloodWait("Flood wait"))

    assert pause == 0


@pytest.mark.asyncio
async def test_unrelated_errors_do_not_pause_the_queue():
    assert await _queue_pause(ValueError("bad payload")) == 0

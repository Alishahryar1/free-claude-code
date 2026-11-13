"""The next-event deadline excludes downstream work and spans discarded pings."""

import asyncio

import pytest

from free_claude_code.providers.stream_read import read_stream_event

pytestmark = pytest.mark.asyncio


def is_keepalive(event):
    return event.get("type") == "ping"


async def test_heartbeats_do_not_renew_deadline():
    reads = 0

    async def source():
        nonlocal reads
        while True:
            await asyncio.sleep(0.005)
            reads += 1
            yield {"type": "ping"}

    iterator = source()
    try:
        with pytest.raises(TimeoutError):
            await read_stream_event(iterator, timeout=0.04, is_keepalive=is_keepalive)
        assert reads > 1
    finally:
        await iterator.aclose()


@pytest.mark.parametrize("timeout", [0, -1, 0.01])
async def test_ready_heartbeat_flood_cannot_evade_timeout(timeout):
    async def source():
        while True:
            yield {"type": "ping"}

    iterator = source()
    try:
        with pytest.raises(TimeoutError):
            await read_stream_event(
                iterator, timeout=timeout, is_keepalive=is_keepalive
            )
    finally:
        await iterator.aclose()


async def test_each_real_event_gets_new_allowance_and_consumer_time_is_excluded():
    events = [{"type": "started"}, {"usage": {"tokens": 0}}, {"opaque": "state"}]

    async def source():
        for event in events:
            yield {"type": "ping"}
            await asyncio.sleep(0.005)
            yield event

    iterator = source()
    try:
        for event in events:
            assert (
                await read_stream_event(
                    iterator, timeout=0.03, is_keepalive=is_keepalive
                )
                is event
            )
            await asyncio.sleep(0.05)
        with pytest.raises(StopAsyncIteration):
            await read_stream_event(iterator, timeout=0.03, is_keepalive=is_keepalive)
    finally:
        await iterator.aclose()


@pytest.mark.parametrize(
    "event",
    [{}, {"choices": []}, {"usage": None}, {"type": "error"}, {"type": "completed"}],
)
async def test_non_heartbeat_events_are_not_reinterpreted(event):
    async def source():
        yield event

    iterator = source()
    try:
        assert (
            await read_stream_event(iterator, timeout=1, is_keepalive=is_keepalive)
            is event
        )
    finally:
        await iterator.aclose()

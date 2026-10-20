import asyncio

import pytest

from free_claude_code.application.session_events import (
    EventOverflowError,
    EventPublisher,
)


@pytest.mark.asyncio
async def test_subscribers_receive_the_same_ordered_events_independently() -> None:
    publisher = EventPublisher(queue_size=4)
    first = publisher.subscribe()
    second = publisher.subscribe()
    first_events = first.__aiter__()
    second_events = second.__aiter__()

    publisher.publish("session.created", {"session_id": "one"})
    publisher.publish("session.updated", {"session_id": "one", "revision": 2})

    assert [await anext(first_events), await anext(first_events)] == [
        await anext(second_events),
        await anext(second_events),
    ]

    await first.aclose()
    publisher.publish("session.deleted", {"session_id": "one"})

    with pytest.raises(StopAsyncIteration):
        await anext(first_events)
    assert (await anext(second_events)).event == "session.deleted"

    await second.aclose()


@pytest.mark.asyncio
async def test_slow_subscriber_overflow_does_not_block_a_healthy_subscriber() -> None:
    publisher = EventPublisher(queue_size=2)
    slow = publisher.subscribe()
    healthy = publisher.subscribe()
    slow_events = slow.__aiter__()
    healthy_events = healthy.__aiter__()

    for revision in range(1, 4):
        published = publisher.publish(
            "session.updated",
            {"session_id": "one", "revision": revision},
        )
        assert await asyncio.wait_for(anext(healthy_events), timeout=1) == published

    with pytest.raises(EventOverflowError) as raised:
        await asyncio.wait_for(anext(slow_events), timeout=1)
    assert raised.value.cursor == 3

    publisher.publish("session.deleted", {"session_id": "one"})
    assert (await anext(healthy_events)).event == "session.deleted"

    await slow.aclose()
    await healthy.aclose()


@pytest.mark.asyncio
async def test_publisher_close_releases_every_waiting_subscription() -> None:
    publisher = EventPublisher()
    first = publisher.subscribe().__aiter__()
    second = publisher.subscribe().__aiter__()

    publisher.close()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(first), timeout=1)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(second), timeout=1)

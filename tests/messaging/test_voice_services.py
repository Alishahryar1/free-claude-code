import asyncio

import pytest

from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.voice import (
    PendingVoiceClaim,
    PendingVoiceRegistry,
    VoiceCancellationResult,
    VoiceHandoffOutcome,
)

TELEGRAM_CHAT = MessageScope(platform="telegram", chat_id="chat")
DISCORD_CHAT = MessageScope(platform="discord", chat_id="chat")


class AsyncNoop:
    async def __call__(self) -> None:
        return None


class FatalVoiceFailure(BaseException):
    pass


async def _reserve_bound(
    registry: PendingVoiceRegistry,
    *,
    scope: MessageScope = TELEGRAM_CHAT,
    voice_id: str = "voice-1",
    status_id: str = "status-1",
) -> PendingVoiceClaim:
    claim = await registry.reserve(scope, voice_id)
    assert claim is not None
    assert await registry.bind_status(claim, status_id) is True
    return claim


@pytest.mark.asyncio
async def test_cancel_before_status_binding_rejects_late_flow() -> None:
    registry = PendingVoiceRegistry()
    claim = await registry.reserve(TELEGRAM_CHAT, "voice-1")
    callback_called = False

    async def callback() -> None:
        nonlocal callback_called
        callback_called = True

    assert claim is not None
    cancellation = await registry.cancel(TELEGRAM_CHAT, "voice-1")
    assert cancellation is not None
    assert cancellation == VoiceCancellationResult(
        scope=TELEGRAM_CHAT,
        voice_message_id="voice-1",
        status_message_id=None,
        delete_message_ids=frozenset({"voice-1"}),
    )
    assert cancellation.delete_message_ids == frozenset({"voice-1"})
    assert await registry.bind_status(claim, "status-1") is False
    assert await registry.handoff(claim, callback) is VoiceHandoffOutcome.REJECTED
    assert callback_called is False


@pytest.mark.asyncio
async def test_handoff_remains_addressable_until_callback_completes() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    started = asyncio.Event()
    release = asyncio.Event()

    async def callback() -> None:
        started.set()
        await release.wait()

    handoff_task = asyncio.create_task(registry.handoff(claim, callback))
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await registry.reserve(TELEGRAM_CHAT, "voice-1") is None
    release.set()
    assert await asyncio.wait_for(handoff_task, timeout=1) is (
        VoiceHandoffOutcome.COMPLETED
    )
    assert await registry.cancel(TELEGRAM_CHAT, "status-1") is None


@pytest.mark.asyncio
async def test_cancel_removes_then_drains_handoff_without_holding_lock() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    started = asyncio.Event()
    cancellation_received = asyncio.Event()
    release = asyncio.Event()

    async def callback() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_received.set()
            await release.wait()
            raise

    handoff_task = asyncio.create_task(registry.handoff(claim, callback))
    await asyncio.wait_for(started.wait(), timeout=1)
    cancel_task = asyncio.create_task(registry.cancel(TELEGRAM_CHAT, "status-1"))
    await asyncio.wait_for(cancellation_received.wait(), timeout=1)

    assert not cancel_task.done()
    replacement = await asyncio.wait_for(
        registry.reserve(TELEGRAM_CHAT, "voice-1"), timeout=1
    )
    assert replacement is not None
    assert await registry.bind_status(replacement, "status-new") is True

    release.set()
    cancellation = await asyncio.wait_for(cancel_task, timeout=1)
    assert cancellation is not None
    assert cancellation == VoiceCancellationResult(
        scope=TELEGRAM_CHAT,
        voice_message_id="voice-1",
        status_message_id="status-1",
        delete_message_ids=frozenset({"status-1"}),
    )
    assert cancellation.delete_message_ids == frozenset({"status-1"})
    assert await asyncio.wait_for(handoff_task, timeout=1) is (
        VoiceHandoffOutcome.CANCELLED
    )
    assert await registry.cancel(TELEGRAM_CHAT, "status-new") is not None


@pytest.mark.asyncio
async def test_voice_reference_authorizes_voice_and_status_deletion() -> None:
    registry = PendingVoiceRegistry()
    await _reserve_bound(registry)

    cancellation = await registry.cancel(TELEGRAM_CHAT, "voice-1")

    assert cancellation is not None
    assert cancellation.delete_message_ids == frozenset({"voice-1", "status-1"})


@pytest.mark.asyncio
async def test_cancel_scope_preserves_other_platform_chats() -> None:
    registry = PendingVoiceRegistry()
    await _reserve_bound(registry)
    await _reserve_bound(
        registry,
        scope=DISCORD_CHAT,
        voice_id="voice-2",
        status_id="status-2",
    )

    cancellations = await registry.cancel_scope(TELEGRAM_CHAT)

    assert len(cancellations) == 1
    assert cancellations[0].delete_message_ids == frozenset({"voice-1", "status-1"})
    assert await registry.cancel(DISCORD_CHAT, "status-2") is not None


@pytest.mark.asyncio
async def test_handoff_propagates_callback_error_when_completion_wins() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)

    async def callback() -> None:
        raise RuntimeError("handler failed")

    with pytest.raises(RuntimeError, match="handler failed"):
        await registry.handoff(claim, callback)

    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is None


@pytest.mark.asyncio
async def test_handoff_suppresses_late_callback_error_when_cancellation_wins() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    started = asyncio.Event()

    async def callback() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError("late handler failure") from None

    handoff_task = asyncio.create_task(registry.handoff(claim, callback))
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is not None
    assert await asyncio.wait_for(handoff_task, timeout=1) is (
        VoiceHandoffOutcome.CANCELLED
    )


@pytest.mark.asyncio
async def test_handoff_caller_cancellation_drains_child_and_propagates() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    started = asyncio.Event()
    cancellation_received = asyncio.Event()
    release = asyncio.Event()

    async def callback() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_received.set()
            await release.wait()
            raise

    handoff_task = asyncio.create_task(registry.handoff(claim, callback))
    await asyncio.wait_for(started.wait(), timeout=1)
    handoff_task.cancel()
    await asyncio.wait_for(cancellation_received.wait(), timeout=1)
    assert handoff_task.cancelling() == 1
    assert await registry.reserve(TELEGRAM_CHAT, "voice-1") is None
    assert await registry.reserve(TELEGRAM_CHAT, "status-1") is None

    handoff_task.cancel()
    await asyncio.sleep(0)
    assert handoff_task.cancelling() == 2
    handoff_task.cancel()
    await asyncio.sleep(0)
    assert handoff_task.cancelling() == 3
    assert not handoff_task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await handoff_task
    assert handoff_task.cancelling() == 3
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is None


@pytest.mark.asyncio
async def test_handoff_drains_child_when_cancellation_is_already_pending() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    started = asyncio.Event()
    finished = asyncio.Event()
    release = asyncio.Event()

    async def callback() -> None:
        started.set()
        try:
            await release.wait()
        finally:
            finished.set()

    async def cancelled_caller() -> VoiceHandoffOutcome:
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        return await registry.handoff(claim, callback)

    handoff_task = asyncio.create_task(cancelled_caller())
    try:
        with pytest.raises(asyncio.CancelledError):
            await handoff_task
        assert started.is_set()
        assert finished.is_set()
        assert handoff_task.cancelling() == 1
        assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is None
    finally:
        release.set()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_external_cancel_owns_aliases_during_caller_cancelled_drain() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    started = asyncio.Event()
    caller_cancellation_received = asyncio.Event()
    external_cancellation_received = asyncio.Event()
    release = asyncio.Event()

    async def callback() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            caller_cancellation_received.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                external_cancellation_received.set()
                await release.wait()
                raise

    handoff_task = asyncio.create_task(registry.handoff(claim, callback))
    await asyncio.wait_for(started.wait(), timeout=1)

    handoff_task.cancel()
    await asyncio.wait_for(caller_cancellation_received.wait(), timeout=1)
    cancel_task = asyncio.create_task(registry.cancel(TELEGRAM_CHAT, "status-1"))
    await asyncio.wait_for(external_cancellation_received.wait(), timeout=1)

    replacement = await registry.reserve(TELEGRAM_CHAT, "voice-1")
    assert replacement is not None
    assert await registry.bind_status(replacement, "status-new") is True
    release.set()

    cancellation = await asyncio.wait_for(cancel_task, timeout=1)
    assert cancellation == VoiceCancellationResult(
        scope=TELEGRAM_CHAT,
        voice_message_id="voice-1",
        status_message_id="status-1",
        delete_message_ids=frozenset({"status-1"}),
    )
    with pytest.raises(asyncio.CancelledError):
        await handoff_task
    assert await registry.cancel(TELEGRAM_CHAT, "status-new") is not None


@pytest.mark.asyncio
async def test_handoff_finalization_survives_repeated_caller_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    finalization_started = asyncio.Event()
    release_finalization = asyncio.Event()
    complete_if_owned = registry._complete_if_owned

    async def gated_completion(entry) -> bool:
        finalization_started.set()
        await release_finalization.wait()
        return await complete_if_owned(entry)

    monkeypatch.setattr(registry, "_complete_if_owned", gated_completion)
    handoff_task = asyncio.create_task(registry.handoff(claim, AsyncNoop()))
    await asyncio.wait_for(finalization_started.wait(), timeout=1)

    handoff_task.cancel()
    await asyncio.sleep(0)
    handoff_task.cancel()
    await asyncio.sleep(0)
    assert not handoff_task.done()

    release_finalization.set()
    with pytest.raises(asyncio.CancelledError):
        await handoff_task
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is None


@pytest.mark.asyncio
async def test_cancel_drain_survives_repeated_caller_cancellation() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    started = asyncio.Event()
    child_cancelled = asyncio.Event()
    release = asyncio.Event()

    async def callback() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            await release.wait()
            raise

    handoff_task = asyncio.create_task(registry.handoff(claim, callback))
    await asyncio.wait_for(started.wait(), timeout=1)
    cancel_task = asyncio.create_task(registry.cancel(TELEGRAM_CHAT, "voice-1"))
    await asyncio.wait_for(child_cancelled.wait(), timeout=1)

    cancel_task.cancel()
    await asyncio.sleep(0)
    cancel_task.cancel()
    await asyncio.sleep(0)
    assert not cancel_task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await cancel_task
    assert await asyncio.wait_for(handoff_task, timeout=1) is (
        VoiceHandoffOutcome.CANCELLED
    )
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is None


@pytest.mark.asyncio
async def test_timeout_and_independent_cancellation_remain_distinct() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    started = asyncio.Event()
    child_cancelled = asyncio.Event()
    release = asyncio.Event()

    async def callback() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            await release.wait()
            raise

    handoff_task = asyncio.create_task(registry.handoff(claim, callback))
    await asyncio.wait_for(started.wait(), timeout=1)

    async def timed_cancel() -> VoiceCancellationResult | None:
        async with asyncio.timeout(0.01):
            return await registry.cancel(TELEGRAM_CHAT, "voice-1")

    cancel_task = asyncio.create_task(timed_cancel())
    await asyncio.wait_for(child_cancelled.wait(), timeout=1)
    await asyncio.sleep(0.05)
    assert cancel_task.cancelling() == 1
    cancel_task.cancel()
    await asyncio.sleep(0)
    assert cancel_task.cancelling() == 2
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await cancel_task
    assert await asyncio.wait_for(handoff_task, timeout=1) is (
        VoiceHandoffOutcome.CANCELLED
    )


@pytest.mark.asyncio
async def test_reentrant_callback_cancellation_does_not_join_a_cycle() -> None:
    registry = PendingVoiceRegistry()
    first = await _reserve_bound(
        registry,
        voice_id="voice-1",
        status_id="status-1",
    )
    second = await _reserve_bound(
        registry,
        voice_id="voice-2",
        status_id="status-2",
    )
    second_started = asyncio.Event()
    first_cancellation: VoiceCancellationResult | None = None
    second_cancellation: VoiceCancellationResult | None = None

    async def first_callback() -> None:
        nonlocal second_cancellation
        await second_started.wait()
        second_cancellation = await registry.cancel(TELEGRAM_CHAT, "voice-2")

    async def second_callback() -> None:
        nonlocal first_cancellation
        second_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            first_cancellation = await registry.cancel(TELEGRAM_CHAT, "voice-1")
            raise

    first_handoff = asyncio.create_task(registry.handoff(first, first_callback))
    second_handoff = asyncio.create_task(registry.handoff(second, second_callback))

    assert await asyncio.wait_for(first_handoff, timeout=1) is (
        VoiceHandoffOutcome.COMPLETED
    )
    assert await asyncio.wait_for(second_handoff, timeout=1) is (
        VoiceHandoffOutcome.CANCELLED
    )
    assert first_cancellation is None
    assert second_cancellation is not None
    assert second_cancellation.voice_message_id == "voice-2"


@pytest.mark.asyncio
async def test_nested_callback_task_cannot_cancel_its_own_claim() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    cancellation: VoiceCancellationResult | None = None

    async def nested_cancel() -> None:
        nonlocal cancellation
        cancellation = await registry.cancel(TELEGRAM_CHAT, "status-1")

    async def callback() -> None:
        await asyncio.create_task(nested_cancel())

    assert await asyncio.wait_for(registry.handoff(claim, callback), timeout=1) is (
        VoiceHandoffOutcome.COMPLETED
    )
    assert cancellation is None
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is None


@pytest.mark.asyncio
async def test_nested_callback_task_is_excluded_from_bulk_cancellation() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    cancellations: tuple[VoiceCancellationResult, ...] | None = None

    async def nested_cancel_all() -> None:
        nonlocal cancellations
        cancellations = await registry.cancel_all()

    async def callback() -> None:
        await asyncio.create_task(nested_cancel_all())

    assert await asyncio.wait_for(registry.handoff(claim, callback), timeout=1) is (
        VoiceHandoffOutcome.COMPLETED
    )
    assert cancellations == ()
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is None


@pytest.mark.asyncio
async def test_fatal_callback_failure_releases_aliases_and_propagates() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)

    async def callback() -> None:
        raise FatalVoiceFailure

    with pytest.raises(FatalVoiceFailure):
        await registry.handoff(claim, callback)

    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is None
    assert await registry.cancel(TELEGRAM_CHAT, "status-1") is None


@pytest.mark.asyncio
async def test_fatal_failure_after_caller_cancellation_releases_aliases() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    started = asyncio.Event()
    cancellation_received = asyncio.Event()
    release = asyncio.Event()

    async def callback() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_received.set()
            await release.wait()
            raise FatalVoiceFailure from None

    handoff_task = asyncio.create_task(registry.handoff(claim, callback))
    await asyncio.wait_for(started.wait(), timeout=1)
    handoff_task.cancel()
    await asyncio.wait_for(cancellation_received.wait(), timeout=1)

    assert await registry.reserve(TELEGRAM_CHAT, "voice-1") is None
    release.set()
    with pytest.raises(FatalVoiceFailure):
        await handoff_task

    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is None
    assert await registry.cancel(TELEGRAM_CHAT, "status-1") is None


@pytest.mark.asyncio
async def test_fatal_loser_failure_is_retrieved_and_propagated() -> None:
    registry = PendingVoiceRegistry()
    claim = await _reserve_bound(registry)
    started = asyncio.Event()

    async def callback() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise FatalVoiceFailure from None

    handoff_task = asyncio.create_task(registry.handoff(claim, callback))
    await asyncio.wait_for(started.wait(), timeout=1)

    with pytest.raises(FatalVoiceFailure):
        await registry.cancel(TELEGRAM_CHAT, "voice-1")
    with pytest.raises(FatalVoiceFailure):
        await handoff_task

    assert await registry.cancel(TELEGRAM_CHAT, "status-1") is None


@pytest.mark.asyncio
async def test_cancel_all_deduplicates_aliases_and_excludes_current_child() -> None:
    registry = PendingVoiceRegistry()
    first = await _reserve_bound(registry, voice_id="voice-1", status_id="status-1")
    second = await _reserve_bound(
        registry,
        scope=DISCORD_CHAT,
        voice_id="voice-2",
        status_id="status-2",
    )
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    cancellations: tuple[VoiceCancellationResult, ...] | None = None

    async def first_callback() -> None:
        first_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    async def second_callback() -> None:
        nonlocal cancellations
        await first_started.wait()
        cancellations = await registry.cancel_all()

    first_handoff = asyncio.create_task(registry.handoff(first, first_callback))
    second_handoff = asyncio.create_task(registry.handoff(second, second_callback))

    assert await asyncio.wait_for(first_handoff, timeout=1) is (
        VoiceHandoffOutcome.CANCELLED
    )
    assert await asyncio.wait_for(second_handoff, timeout=1) is (
        VoiceHandoffOutcome.COMPLETED
    )
    assert first_cancelled.is_set()
    assert cancellations is not None
    assert len(cancellations) == 1
    assert {result.scope for result in cancellations} == {TELEGRAM_CHAT}
    assert {result.delete_message_ids for result in cancellations} == {
        frozenset({"voice-1", "status-1"}),
    }


@pytest.mark.asyncio
async def test_stale_claim_cannot_mutate_reused_voice_id() -> None:
    registry = PendingVoiceRegistry()
    stale_claim = await registry.reserve(TELEGRAM_CHAT, "voice-1")

    assert stale_claim is not None
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is not None

    current_claim = await _reserve_bound(
        registry, voice_id="voice-1", status_id="status-current"
    )

    assert current_claim != stale_claim
    assert await registry.bind_status(stale_claim, "status-stale") is False
    assert (
        await registry.handoff(stale_claim, AsyncNoop()) is VoiceHandoffOutcome.REJECTED
    )
    assert await registry.discard(stale_claim) is False
    assert await registry.cancel(TELEGRAM_CHAT, "status-current") is not None


@pytest.mark.asyncio
async def test_pending_voice_registry_rejects_duplicate_and_unbound_handoff() -> None:
    registry = PendingVoiceRegistry()
    claim = await registry.reserve(TELEGRAM_CHAT, "voice-1")

    assert claim is not None
    assert await registry.reserve(TELEGRAM_CHAT, "voice-1") is None
    assert await registry.handoff(claim, AsyncNoop()) is VoiceHandoffOutcome.REJECTED
    assert await registry.cancel(TELEGRAM_CHAT, "voice-1") is not None

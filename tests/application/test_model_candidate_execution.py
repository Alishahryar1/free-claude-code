"""Independent candidate execution with request-local delivery evidence."""

from unittest.mock import AsyncMock, Mock

import pytest

from free_claude_code.application.model_fallback import stream_model_candidates
from free_claude_code.application.routing import (
    ProviderModelCandidate,
    ProviderModelTarget,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
    current_stream_delivery,
)
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.stream_recovery import (
    CompletedCall,
    ContinuationSeed,
    DeliveredPrefix,
    SourceRecoverySnapshot,
    StreamFailureContext,
)


def _candidates(count=1):
    return tuple(
        ProviderModelCandidate(
            ProviderModelTarget("provider", f"m{i}", f"provider/m{i}")
        )
        for i in range(count)
    )


@pytest.mark.asyncio
async def test_delivery_and_opening_are_deferred_and_packets_preserved():
    construction_state = StreamDeliveryState()
    consumption_state = StreamDeliveryState()
    packets = (
        StreamOutput(("text",)),
        StreamOutput((), (CompletedCall("messages", (0,), "call-1"),)),
    )
    seen = []
    closed = []

    async def body():
        try:
            for packet in packets:
                yield packet
        finally:
            closed.append(current_stream_delivery())

    async def open_candidate(index, candidate, continuation):
        seen.append(current_stream_delivery())
        return body()

    get_delivery = Mock(side_effect=current_stream_delivery)
    with bind_stream_delivery(construction_state):
        stream = stream_model_candidates(
            _candidates(),
            open_candidate,
            get_delivery=get_delivery,
            request_id="late-delivery",
            wire_api="messages",
        )
    get_delivery.assert_not_called()
    assert seen == []
    with bind_stream_delivery(consumption_state):
        output = [packet async for packet in stream]
    get_delivery.assert_called_once_with()
    assert seen == closed == [consumption_state]
    assert all(
        actual is expected for actual, expected in zip(output, packets, strict=True)
    )


@pytest.mark.asyncio
async def test_missing_delivery_never_opens_a_candidate():
    open_candidate = AsyncMock()
    stream = stream_model_candidates(
        _candidates(),
        open_candidate,
        get_delivery=current_stream_delivery,
        request_id="missing-delivery",
        wire_api="messages",
    )
    with (
        bind_stream_delivery(None),
        pytest.raises(RuntimeError, match="delivery context"),
    ):
        await anext(stream)
    open_candidate.assert_not_awaited()


def _prefix(text):
    return DeliveredPrefix(
        text=text,
        thinking="",
        has_calls=False,
        eligible=True,
        can_handoff=False,
        unsafe_reason=None,
        continuation=ContinuationSeed("messages", ({"type": "text", "text": text},)),
    )


@pytest.mark.asyncio
async def test_interleaved_sequences_use_their_own_latest_delivered_prefix():
    left_state, right_state = StreamDeliveryState(), StreamDeliveryState()
    left_prefix, right_prefix = _prefix("left"), _prefix("right")
    left_state.bind_prefix(lambda: left_prefix)
    right_state.bind_prefix(lambda: right_prefix)
    opened = []
    closed = []

    def opener(label):
        async def body(index):
            try:
                yield StreamOutput((f"{label}-{index}",))
                if index < 2:
                    raise ExecutionFailure(
                        kind=FailureKind.UPSTREAM,
                        status_code=502,
                        message="stream interrupted",
                        retryable=True,
                        stream_context=StreamFailureContext(SourceRecoverySnapshot()),
                    )
            finally:
                closed.append((label, index))

        async def open_candidate(index, candidate, continuation):
            opened.append((label, index, continuation))
            return body(index)

        return open_candidate

    left = stream_model_candidates(
        _candidates(3),
        opener("left"),
        get_delivery=lambda: left_state,
        request_id="left",
        wire_api="messages",
    )
    right = stream_model_candidates(
        _candidates(3),
        opener("right"),
        get_delivery=lambda: right_state,
        request_id="right",
        wire_api="messages",
    )
    try:
        assert (await anext(left)).frames == ("left-0",)
        assert (await anext(right)).frames == ("right-0",)
        left_state.release_content()
        right_state.release_content()
        assert (await anext(left)).frames == ("left-1",)
        left_prefix = _prefix("left extended")
        assert (await anext(left)).frames == ("left-2",)
        assert right_state.attempt_revision == 0
        assert right_state.transition is None
        assert (await anext(right)).frames == ("right-1",)
        right_prefix = _prefix("right extended")
        assert (await anext(right)).frames == ("right-2",)
        assert [row[2].readable_text("text") for row in opened if row[1]] == [
            "left",
            "left extended",
            "right",
            "right extended",
        ]
        assert left_state.attempt_revision == right_state.attempt_revision == 2
        assert left_state.transition == right_state.transition == "continue"
        assert [packet async for packet in left] == []
        assert [packet async for packet in right] == []
    finally:
        await left.aclose()
        await right.aclose()
    assert sorted(closed) == [
        (label, index) for label in ("left", "right") for index in range(3)
    ]

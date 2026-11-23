"""What the model fallback sequence does after each kind of candidate failure."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx2
import pytest
from loguru import logger

if TYPE_CHECKING:
    from loguru import Message

from free_claude_code.application.model_fallback import stream_model_candidates
from free_claude_code.application.routing import (
    ProviderModelCandidate,
    ProviderModelTarget,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.stream_delivery import (
    StreamDeliveryState,
    bind_stream_delivery,
)
from free_claude_code.core.stream_output import StreamOutput
from free_claude_code.core.stream_recovery import (
    ContinuationSeed,
    DeliveredPrefix,
    SourceRecoverySnapshot,
    StreamFailureContext,
)
from tests.providers.test_lmstudio import (
    make_request,
    metadata_response,
    provider_with_http,
)

NO_CONTEXT = None
CLEAN = SourceRecoverySnapshot()


def _failure(
    context: SourceRecoverySnapshot | None,
    *,
    status: int = 502,
    retryable: bool = True,
) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.UPSTREAM,
        status_code=status,
        message=f"failed with {status}",
        retryable=retryable,
        stream_context=StreamFailureContext(context) if context is not None else None,
    )


@dataclass(frozen=True)
class Candidate:
    """One scripted candidate: frames it yields, then how it ends."""

    frames: tuple[str, ...] = ()
    release: bool = False
    failure: ExecutionFailure | None = None
    at_open: bool = False


def ok(*frames: str) -> Candidate:
    return Candidate(frames=frames or ("ok",), release=True)


def open_fails(failure: ExecutionFailure) -> Candidate:
    return Candidate(failure=failure, at_open=True)


def fails_after(
    *frames: str, failure: ExecutionFailure, release: bool = True
) -> Candidate:
    return Candidate(frames=frames, release=release, failure=failure)


@dataclass
class Outcome:
    opened: list[tuple[int, str | None]] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)
    raised: int | None = None
    transition: str | None = None
    fallbacks: list[tuple[str, str | None]] = field(default_factory=list)
    selected: list[int] = field(default_factory=list)


async def _run(
    *script: Candidate, prefix_calls: bool = False, reads: list[str] | None = None
) -> Outcome:
    state = StreamDeliveryState()
    delivered: list[str] = []

    def prefix() -> DeliveredPrefix:
        text = "".join(delivered)
        if reads is not None:
            reads.append(text)
        return DeliveredPrefix(
            text=text,
            thinking="",
            has_calls=prefix_calls,
            eligible=True,
            can_handoff=True,
            unsafe_reason=None,
            continuation=ContinuationSeed(
                "messages", ({"type": "text", "text": text},)
            ),
        )

    state.bind_prefix(prefix)
    outcome = Outcome()

    async def body(candidate: Candidate) -> AsyncIterator[StreamOutput]:
        for frame in candidate.frames:
            yield StreamOutput((frame,))
            if candidate.release:
                delivered.append(frame)
                state.release_content()
        if candidate.failure is not None:
            raise candidate.failure

    async def open_candidate(
        index: int,
        _candidate: ProviderModelCandidate,
        continuation: ContinuationSeed | None,
    ) -> AsyncIterator[StreamOutput]:
        outcome.opened.append(
            (index, continuation.readable_text("text") if continuation else None)
        )
        candidate = script[index]
        if candidate.at_open:
            assert candidate.failure is not None
            raise candidate.failure
        return body(candidate)

    candidates = tuple(
        ProviderModelCandidate(ProviderModelTarget("p", f"m{i}", f"p/m{i}"))
        for i in range(len(script))
    )

    def sink(message: Message) -> None:
        payload = message.record["extra"].get("trace_payload")
        if payload is None:
            return
        if payload["event"] == "free_claude_code.model_fallback.started":
            outcome.fallbacks.append(
                (payload["recovery_action"], payload["recovery_reason"])
            )
        elif payload["event"] == "free_claude_code.model_fallback.selected":
            index: int = payload["candidate_index"]
            outcome.selected.append(index)

    sink_id = logger.add(sink, level="DEBUG")
    try:
        async for chunk in stream_model_candidates(
            candidates,
            open_candidate,
            get_delivery=lambda: state,
            request_id="decisions",
            wire_api="messages",
        ):
            outcome.frames.extend(chunk.frames)
    except ExecutionFailure as failure:
        outcome.raised = failure.status_code
    finally:
        logger.remove(sink_id)
    outcome.transition = state.transition
    return outcome


@pytest.mark.asyncio
async def test_first_candidate_success_never_falls_back() -> None:
    outcome = await _run(ok("a"), ok("b"))

    assert outcome == Outcome(opened=[(0, None)], frames=["a"])


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", [True, False])
async def test_failure_without_context_before_any_delivery_restarts(
    retryable: bool,
) -> None:
    outcome = await _run(
        open_fails(_failure(NO_CONTEXT, status=400, retryable=retryable)), ok("b")
    )

    assert outcome == Outcome(
        opened=[(0, None), (1, None)],
        frames=["b"],
        fallbacks=[("restart", "no_candidate_output")],
        selected=[2],
    )


@pytest.mark.asyncio
async def test_hidden_output_restarts_from_scratch() -> None:
    outcome = await _run(
        fails_after("a", failure=_failure(CLEAN), release=False), ok("b")
    )

    assert outcome == Outcome(
        opened=[(0, None), (1, None)],
        frames=["a", "b"],
        fallbacks=[("restart", "no_delivered_content")],
        selected=[2],
    )


@pytest.mark.asyncio
async def test_released_output_continues_from_the_delivered_prefix() -> None:
    outcome = await _run(fails_after("a", failure=_failure(CLEAN)), ok("b"))

    assert outcome == Outcome(
        opened=[(0, None), (1, "a")],
        frames=["a", "b"],
        transition="continue",
        fallbacks=[("continue", "portable_prefix")],
        selected=[2],
    )


@pytest.mark.asyncio
async def test_each_continuation_carries_the_latest_prefix() -> None:
    outcome = await _run(
        fails_after("a", failure=_failure(CLEAN)),
        fails_after("b", failure=_failure(CLEAN)),
        ok("c"),
    )

    assert outcome.opened == [(0, None), (1, "a"), (2, "ab")]
    assert outcome.frames == ["a", "b", "c"]
    assert outcome.fallbacks == [("continue", "portable_prefix")] * 2


@pytest.mark.asyncio
async def test_released_tool_calls_hand_off_instead_of_falling_back() -> None:
    outcome = await _run(
        fails_after("a", failure=_failure(CLEAN)), ok("b"), prefix_calls=True
    )

    assert outcome == Outcome(opened=[(0, None)], frames=["a"], transition="handoff")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "release"),
    [
        pytest.param(
            SourceRecoverySnapshot(normal_stop_seen=True), True, id="normal_stop"
        ),
        pytest.param(
            SourceRecoverySnapshot(unsafe_reason="native_content"),
            False,
            id="unsafe_before_release",
        ),
        pytest.param(
            SourceRecoverySnapshot(structured_output=True), True, id="structured"
        ),
    ],
)
async def test_unsafe_transitions_raise_without_falling_back(
    source: SourceRecoverySnapshot, release: bool
) -> None:
    outcome = await _run(
        fails_after("a", failure=_failure(source), release=release), ok("b")
    )

    assert outcome.opened == [(0, None)]
    assert outcome.raised == 502
    assert outcome.fallbacks == []


@pytest.mark.asyncio
async def test_failure_without_context_after_own_delivery_raises() -> None:
    outcome = await _run(fails_after("a", failure=_failure(NO_CONTEXT)), ok("b"))

    assert outcome == Outcome(opened=[(0, None)], frames=["a"], raised=502)


@pytest.mark.asyncio
async def test_last_candidate_failure_raises() -> None:
    outcome = await _run(
        fails_after("a", failure=_failure(CLEAN)),
        open_fails(_failure(NO_CONTEXT, status=503)),
    )

    assert outcome.opened == [(0, None), (1, "a")]
    assert outcome.raised == 503
    assert outcome.fallbacks == [("continue", "portable_prefix")]


@pytest.mark.asyncio
async def test_silent_continuation_failure_without_context_keeps_the_seed() -> None:
    # Candidate 1 continues from "a", then fails before producing anything and
    # without stream context. The "a" seed is still valid for candidate 2.
    outcome = await _run(
        fails_after("a", failure=_failure(CLEAN)),
        open_fails(_failure(NO_CONTEXT, status=429)),
        ok("c"),
    )

    assert outcome == Outcome(
        opened=[(0, None), (1, "a"), (2, "a")],
        frames=["a", "c"],
        transition="continue",
        fallbacks=[
            ("continue", "portable_prefix"),
            ("continue", "no_candidate_output"),
        ],
        selected=[3],
    )


@pytest.mark.asyncio
async def test_continuation_failure_without_context_after_own_output_raises() -> None:
    outcome = await _run(
        fails_after("a", failure=_failure(CLEAN)),
        fails_after("b", failure=_failure(NO_CONTEXT, status=429)),
        ok("c"),
    )

    assert outcome.opened == [(0, None), (1, "a")]
    assert outcome.frames == ["a", "b"]
    assert outcome.raised == 429


@pytest.mark.asyncio
async def test_lmstudio_budget_rejection_of_a_continuation_keeps_the_seed() -> None:
    # LM Studio checks the prepared body against its loaded context before
    # sending. A continuation body carries the delivered prefix, so it can fail
    # that check locally, without stream context, before producing anything.
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            return metadata_response(context=1000)
        raise AssertionError("LM Studio must not send an over-budget body")

    delivered = "word " * 1000
    state = StreamDeliveryState()
    state.bind_prefix(
        lambda: DeliveredPrefix(
            text=delivered,
            thinking="",
            has_calls=False,
            eligible=True,
            can_handoff=True,
            unsafe_reason=None,
            continuation=ContinuationSeed(
                "messages", ({"type": "text", "text": delivered},)
            ),
        )
    )

    async def first() -> AsyncIterator[StreamOutput]:
        yield StreamOutput(("first",))
        state.release_content()
        raise _failure(CLEAN)

    async def last() -> AsyncIterator[StreamOutput]:
        yield StreamOutput(("last",))

    seeds: list[str | None] = []
    async with provider_with_http(handler) as lmstudio:

        async def open_candidate(
            index: int, _candidate, continuation: ContinuationSeed | None
        ):
            seeds.append(continuation.readable_text("text") if continuation else None)
            if index == 1:
                return lmstudio.stream_messages(
                    make_request(), continuation=continuation
                )
            return first() if index == 0 else last()

        candidates = tuple(
            ProviderModelCandidate(ProviderModelTarget("p", f"m{i}", f"p/m{i}"))
            for i in range(3)
        )
        with bind_stream_delivery(state):
            frames = [
                frame
                async for chunk in stream_model_candidates(
                    candidates,
                    open_candidate,
                    get_delivery=lambda: state,
                    request_id="lmstudio-budget",
                    wire_api="messages",
                )
                for frame in chunk.frames
            ]

    assert seeds == [None, delivered, delivered]
    assert frames == ["first", "last"]


@pytest.mark.asyncio
async def test_prefix_is_read_only_for_failures_with_stream_evidence() -> None:
    reads: list[str] = []
    outcome = await _run(
        fails_after("a", failure=_failure(CLEAN)),
        open_fails(_failure(NO_CONTEXT, status=429)),
        ok("c"),
        reads=reads,
    )

    assert outcome.frames == ["a", "c"]
    # Read once, for the first failure; the second carries no stream evidence.
    assert reads == ["a"]

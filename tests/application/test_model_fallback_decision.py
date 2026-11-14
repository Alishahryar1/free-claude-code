"""The pure step that follows one failed model candidate."""

import pytest

from free_claude_code.application.model_fallback.decision import (
    CandidateFailure,
    Fail,
    Fallback,
    Handoff,
    next_step,
)
from free_claude_code.core.stream_recovery import (
    ContinuationSeed,
    DeliveredPrefix,
    SourceRecoverySnapshot,
    StreamFailureContext,
)

SEED = ContinuationSeed("messages", ({"type": "text", "text": "a"},))


def _prefix(*, has_calls: bool = False) -> DeliveredPrefix:
    return DeliveredPrefix(
        text="a",
        thinking="",
        has_calls=has_calls,
        eligible=True,
        can_handoff=True,
        unsafe_reason=None,
        continuation=SEED,
    )


def _failure(
    source: SourceRecoverySnapshot | None, *, produced_output: bool = False
) -> CandidateFailure:
    return CandidateFailure(
        context=StreamFailureContext(source) if source is not None else None,
        retryable=True,
        produced_output=produced_output,
    )


CLEAN = SourceRecoverySnapshot()


@pytest.mark.parametrize(
    ("failure", "released", "prefix", "seed", "has_next", "expected"),
    [
        pytest.param(
            _failure(None),
            False,
            None,
            None,
            True,
            Fallback(None, "restart", "no_candidate_output"),
            id="no_context_before_delivery",
        ),
        pytest.param(
            _failure(None, produced_output=True),
            False,
            None,
            None,
            True,
            Fallback(None, "restart", "no_candidate_output"),
            id="no_context_hidden_output",
        ),
        pytest.param(
            _failure(None),
            True,
            None,
            SEED,
            True,
            Fallback(SEED, "continue", "no_candidate_output"),
            id="no_context_silent_continuation",
        ),
        pytest.param(
            _failure(None, produced_output=True),
            True,
            None,
            SEED,
            True,
            Fail(),
            id="no_context_after_own_output",
        ),
        pytest.param(
            _failure(None),
            True,
            None,
            None,
            True,
            Fail(),
            id="no_context_released_without_seed",
        ),
        pytest.param(
            _failure(CLEAN),
            False,
            None,
            None,
            True,
            Fallback(None, "restart", "no_delivered_content"),
            id="restart",
        ),
        pytest.param(
            _failure(CLEAN),
            True,
            _prefix(),
            None,
            True,
            Fallback(SEED, "continue", "portable_prefix"),
            id="continue",
        ),
        pytest.param(
            _failure(CLEAN),
            True,
            _prefix(has_calls=True),
            None,
            False,
            Handoff(),
            id="handoff_even_on_last_candidate",
        ),
        pytest.param(
            _failure(SourceRecoverySnapshot(normal_stop_seen=True)),
            True,
            _prefix(),
            None,
            True,
            Fail(),
            id="stop",
        ),
        pytest.param(
            _failure(CLEAN),
            True,
            _prefix(),
            None,
            False,
            Fail(),
            id="no_candidates_left",
        ),
        pytest.param(
            _failure(None),
            False,
            None,
            None,
            False,
            Fail(),
            id="no_candidates_left_without_context",
        ),
    ],
)
def test_next_step(failure, released, prefix, seed, has_next, expected):
    assert (
        next_step(
            failure,
            content_released=released,
            read_prefix=lambda: prefix,
            seed=seed,
            has_next=has_next,
        )
        == expected
    )


def test_prefix_is_read_only_with_stream_evidence():
    def unread() -> DeliveredPrefix:
        raise AssertionError("prefix read without stream evidence")

    step = next_step(
        _failure(None),
        content_released=True,
        read_prefix=unread,
        seed=SEED,
        has_next=True,
    )

    assert step == Fallback(SEED, "continue", "no_candidate_output")

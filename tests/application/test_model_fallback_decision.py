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
    RecoveryAction,
    SourceRecoverySnapshot,
    StreamFailureContext,
)

RESTART = RecoveryAction.RESTART
CONTINUE = RecoveryAction.CONTINUE
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
            Fallback(None, RESTART, "no_candidate_output"),
            id="no_context_before_delivery",
        ),
        pytest.param(
            _failure(None, produced_output=True),
            False,
            None,
            None,
            True,
            Fallback(None, RESTART, "no_candidate_output"),
            id="no_context_hidden_output",
        ),
        pytest.param(
            _failure(None),
            True,
            None,
            SEED,
            True,
            Fallback(SEED, CONTINUE, "no_candidate_output"),
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
            Fallback(None, RESTART, "no_delivered_content"),
            id="restart",
        ),
        pytest.param(
            _failure(CLEAN),
            True,
            _prefix(),
            None,
            True,
            Fallback(SEED, CONTINUE, "portable_prefix"),
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
def test_next_step(
    failure: CandidateFailure,
    released: bool,
    prefix: DeliveredPrefix | None,
    seed: ContinuationSeed | None,
    has_next: bool,
    expected: Fail | Fallback | Handoff,
):
    assert (
        next_step(
            failure,
            content_released=released,
            prefix=prefix,
            seed=seed,
            has_next=has_next,
        )
        == expected
    )


def test_prefix_is_ignored_without_stream_evidence():
    step = next_step(
        _failure(None),
        content_released=True,
        prefix=_prefix(has_calls=True),
        seed=SEED,
        has_next=True,
    )

    assert step == Fallback(SEED, CONTINUE, "no_candidate_output")

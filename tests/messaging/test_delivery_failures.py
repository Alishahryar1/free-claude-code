"""Delivery failure rules decide from values alone: no platform, loop or clock."""

import pytest

from free_claude_code.messaging.delivery_failures import (
    MAX_ATTEMPTS,
    CallState,
    DeliveryFailure,
    FailureKind,
    Ignore,
    PlatformRateLimited,
    Raise,
    RateLimited,
    Retry,
    RetryPlain,
    decide,
    queue_pause,
)
from free_claude_code.messaging.delivery_failures.decision import Decision

K = FailureKind


def _state(
    attempts: int = 1, *, formatted: bool = False, suppress_known: bool = True
) -> CallState:
    return CallState(
        attempts=attempts, formatted=formatted, suppress_known=suppress_known
    )


@pytest.mark.parametrize(
    ("failure", "state", "expected"),
    [
        pytest.param(DeliveryFailure(K.NETWORK), _state(1), Retry(1.0), id="net-1"),
        pytest.param(DeliveryFailure(K.NETWORK), _state(2), Retry(2.0), id="net-2"),
        pytest.param(DeliveryFailure(K.NETWORK), _state(3), Raise(), id="net-out"),
        pytest.param(
            DeliveryFailure(K.RATE_LIMITED, 5.0), _state(1), Retry(5.0), id="rate-1"
        ),
        pytest.param(
            DeliveryFailure(K.RATE_LIMITED, 9.0),
            _state(MAX_ATTEMPTS),
            RateLimited(9.0),
            id="rate-out",
        ),
        pytest.param(DeliveryFailure(K.UNMODIFIED), _state(), Ignore(), id="same"),
        pytest.param(
            DeliveryFailure(K.UNMODIFIED),
            _state(suppress_known=False),
            Raise(),
            id="same-raised",
        ),
        pytest.param(DeliveryFailure(K.GONE), _state(3), Ignore(), id="gone"),
        pytest.param(
            DeliveryFailure(K.GONE),
            _state(suppress_known=False),
            Raise(),
            id="gone-raised",
        ),
        pytest.param(
            DeliveryFailure(K.UNPARSEABLE),
            _state(3, formatted=True),
            RetryPlain(),
            id="plain",
        ),
        pytest.param(
            DeliveryFailure(K.UNPARSEABLE), _state(), Raise(), id="already-plain"
        ),
        pytest.param(DeliveryFailure(K.REJECTED), _state(), Raise(), id="rejected"),
    ],
)
def test_decide(failure: DeliveryFailure, state: CallState, expected: Decision) -> None:
    assert decide(failure, state) == expected


def test_every_kind_is_decided() -> None:
    for kind in FailureKind:
        decide(DeliveryFailure(kind), _state())


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (PlatformRateLimited(5), 5),
        (ValueError("Flood wait: retry after 2 seconds"), None),
        (RuntimeError("Timed out waiting"), None),
    ],
)
def test_queue_pause(error: Exception, expected: int | None) -> None:
    assert queue_pause(error) == expected

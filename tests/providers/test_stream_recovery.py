"""Undelivered retry policy uses actual consumer state and source termination."""

import pytest

from free_claude_code.core.stream_delivery import StreamDeliveryState
from free_claude_code.providers.stream_recovery import can_retry_undelivered_stream


@pytest.mark.parametrize("released", [False, True])
@pytest.mark.parametrize("normal_stop", [False, True])
@pytest.mark.parametrize("retryable", [False, True])
@pytest.mark.parametrize("attempts", [0, 1])
def test_restart_requires_undelivered_retryable_nonterminal_response(
    released, normal_stop, retryable, attempts
):
    state = StreamDeliveryState()
    if released:
        state.release_content()
    assert can_retry_undelivered_stream(
        state,
        retryable=retryable,
        attempts_remaining=attempts,
        normal_stop_seen=normal_stop,
    ) == (not released and not normal_stop and retryable and attempts > 0)

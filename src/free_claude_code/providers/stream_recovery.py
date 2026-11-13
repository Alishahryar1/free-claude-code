"""Retry policy based on delivery evidence, independent of presentation buffering."""

from free_claude_code.core.stream_delivery import StreamDeliveryState

from .failure_policy import RetryableProviderProtocolError


class TruncatedProviderStreamError(RetryableProviderProtocolError):
    """An upstream stream ended without its required terminal marker."""


def can_retry_undelivered_stream(
    delivery: StreamDeliveryState | None,
    *,
    retryable: bool,
    attempts_remaining: int,
    normal_stop_seen: bool,
) -> bool:
    """A clean restart is safe only before content and a normal model stop."""
    return (
        delivery is not None
        and not delivery.content_released
        and retryable
        and attempts_remaining > 0
        and not normal_stop_seen
    )

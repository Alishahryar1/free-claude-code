"""Exceptions that carry provider recovery state between attempts."""


class ProviderRecoveryExhausted(RuntimeError):
    """The fixed cause and retry deadline selected when recovery stops execution."""

    def __init__(
        self, error: Exception, *, retry_not_before: float | None = None
    ) -> None:
        super().__init__("Provider recovery retry budget was exhausted.")
        if isinstance(error, ProviderRecoveryExhausted):
            self._last_error = error.last_error
            self._retry_not_before = error.retry_not_before
        else:
            self._last_error = error
            self._retry_not_before = retry_not_before

    @property
    def last_error(self) -> Exception:
        return self._last_error

    @property
    def retry_not_before(self) -> float | None:
        return self._retry_not_before


class RetryableProviderProtocolError(RuntimeError):
    """A malformed upstream protocol result eligible for provider retry."""


def underlying_provider_error(exc: Exception) -> Exception:
    """Return the raw failure retained by an exhausted recovery wrapper."""
    while isinstance(exc, ProviderRecoveryExhausted):
        exc = exc.last_error
    return exc

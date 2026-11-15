"""Shared HTTP lifecycle helpers for upstream provider clients."""

import asyncio
import inspect
from typing import Any, TypeVar

from loguru import logger

from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderAttempt,
    ProviderFailureDecision,
)
from free_claude_code.providers.canonical_failures import ProviderFailureOverride

_ResourceT = TypeVar("_ResourceT")


async def maybe_await_aclose(response: Any) -> None:
    """Call ``aclose`` on httpx-like responses; ignore sync test doubles."""
    close = getattr(response, "aclose", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def close_provider_stream(
    stream: Any,
    *,
    active_error: BaseException | None,
    provider_name: str,
    request_id: str | None,
) -> None:
    """Close one stream without letting cleanup change its established outcome."""
    try:
        await maybe_await_aclose(stream)
    except (Exception, asyncio.CancelledError) as close_error:
        if isinstance(close_error, asyncio.CancelledError):
            current_task = asyncio.current_task()
            if (
                isinstance(active_error, asyncio.CancelledError)
                or current_task is None
                or current_task.cancelling()
            ):
                raise
        active_error_type = (
            type(active_error).__name__ if active_error is not None else None
        )
        trace_event(
            stage="provider",
            event="provider.stream.close_failed",
            source="provider",
            provider=provider_name,
            request_id=request_id,
            close_exc_type=type(close_error).__name__,
            preserved_exc_type=active_error_type,
        )
        logger.warning(
            "{}_STREAM_CLOSE_FAILED request_id={} close_exc_type={} "
            "preserved_exc_type={}",
            provider_name,
            request_id,
            type(close_error).__name__,
            active_error_type,
        )


class ProviderAttemptScope:
    """Own one admitted provider attempt and its optional transport resource."""

    def __init__(
        self,
        attempt: ProviderAttempt,
        *,
        provider_name: str,
        request_id: str | None,
    ) -> None:
        self.attempt = attempt
        self._resource: object | None = None
        self._provider_name = provider_name
        self._request_id = request_id
        self._closed = False

    def retain(self, resource: _ResourceT) -> _ResourceT:
        """Retain and return the sole transport resource owned by this scope."""
        if self._closed:
            raise RuntimeError("provider attempt scope is already closed")
        if self._resource is not None:
            raise RuntimeError("provider attempt scope already owns a resource")
        self._resource = resource
        return resource

    def adopt(self, wrapper: _ResourceT) -> _ResourceT:
        """Hand the retained resource to a wrapper that now owns and closes it."""
        if self._closed:
            raise RuntimeError("provider attempt scope is already closed")
        if self._resource is None:
            raise RuntimeError("provider attempt scope owns no resource to wrap")
        self._resource = wrapper
        return wrapper

    async def fail(
        self,
        error: Exception,
        *,
        provider_failure_override: ProviderFailureOverride | None = None,
    ) -> ProviderFailureDecision:
        """Close a failed attempt before letting its caller retry."""
        decision = await self.attempt.fail(
            error, provider_failure_override=provider_failure_override
        )
        if not decision.retry_allowed:
            return decision
        await self.aclose(active_error=decision.error)
        return self.attempt.failure_after_cleanup(decision)

    async def aclose(self, *, active_error: BaseException | None) -> None:
        """Close transport state without masking the established operation outcome."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._resource is not None:
                await close_provider_stream(
                    self._resource,
                    active_error=active_error,
                    provider_name=self._provider_name,
                    request_id=self._request_id,
                )
        finally:
            await self.attempt.aclose()

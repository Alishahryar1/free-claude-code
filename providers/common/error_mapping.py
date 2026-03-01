"""Error mapping for OpenAI-compatible providers (NIM, OpenRouter, LM Studio)."""

import httpx
import openai

from providers.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    ProviderError,
    RateLimitError,
)
from providers.rate_limit import GlobalRateLimiter


def get_user_facing_error_message(
    e: Exception,
    *,
    read_timeout_s: float | None = None,
) -> str:
    """Return a readable, non-empty error message for users."""
    message = str(e).strip()
    if message:
        return message

    if isinstance(e, httpx.ReadTimeout):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Provider request timed out."
    if isinstance(e, httpx.ConnectTimeout):
        return "Could not connect to provider."
    if isinstance(e, TimeoutError):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Request timed out."

    if isinstance(e, (RateLimitError, openai.RateLimitError)):
        return "Provider rate limit reached. Please retry shortly."
    if isinstance(e, (AuthenticationError, openai.AuthenticationError)):
        return "Provider authentication failed. Check API key."
    if isinstance(e, (InvalidRequestError, openai.BadRequestError)):
        return "Invalid request sent to provider."
    if isinstance(e, OverloadedError):
        return "Provider is currently overloaded. Please retry."
    if isinstance(e, APIError):
        if e.status_code in (502, 503, 504):
            return "Provider is temporarily unavailable. Please retry."
        return "Provider API request failed."
    if isinstance(e, ProviderError):
        return "Provider request failed."

    return "Provider request failed unexpectedly."


def append_request_id(message: str, request_id: str | None) -> str:
    """Append request_id suffix when available."""
    base = message.strip() or "Provider request failed unexpectedly."
    if request_id:
        return f"{base} (request_id={request_id})"
    return base


def map_error(e: Exception) -> Exception:
    """Map OpenAI exception to specific ProviderError."""
    message = get_user_facing_error_message(e)
    if isinstance(e, openai.AuthenticationError):
        return AuthenticationError(message, raw_error=str(e))
    if isinstance(e, openai.RateLimitError):
        # Trigger global rate limit block
        GlobalRateLimiter.get_instance().set_blocked(60)  # Default 60s cooldown
        return RateLimitError(message, raw_error=str(e))
    if isinstance(e, openai.BadRequestError):
        return InvalidRequestError(message, raw_error=str(e))
    if isinstance(e, openai.InternalServerError):
        raw_message = str(e)
        if "overloaded" in raw_message.lower() or "capacity" in raw_message.lower():
            return OverloadedError(message, raw_error=raw_message)
        return APIError(message, status_code=500, raw_error=str(e))
    if isinstance(e, openai.APIError):
        return APIError(
            message, status_code=getattr(e, "status_code", 500), raw_error=str(e)
        )

    return e

"""Error mapping for OpenRouter provider.

OpenRouter uses OpenAI-compatible API, so we reuse the same error mapping.
"""

import openai

from providers.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    OverloadedError,
    APIError,
)
from providers.rate_limit import GlobalRateLimiter


def map_error(e: Exception) -> Exception:
    """Map OpenAI exception to specific ProviderError."""
    if isinstance(e, openai.AuthenticationError):
        return AuthenticationError(str(e), raw_error=str(e))
    if isinstance(e, openai.RateLimitError):
        GlobalRateLimiter.get_instance().set_blocked(60)
        return RateLimitError(str(e), raw_error=str(e))
    if isinstance(e, openai.BadRequestError):
        return InvalidRequestError(str(e), raw_error=str(e))
    if isinstance(e, openai.InternalServerError):
        message = str(e)
        if "overloaded" in message.lower() or "capacity" in message.lower():
            return OverloadedError(message, raw_error=str(e))
        return APIError(message, status_code=500, raw_error=str(e))
    if isinstance(e, openai.APIError):
        return APIError(
            str(e), status_code=getattr(e, "status_code", 500), raw_error=str(e)
        )

    return e

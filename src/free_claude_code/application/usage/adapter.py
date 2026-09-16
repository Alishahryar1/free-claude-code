"""Provider usage capability adapters for header extraction and quota inspection."""

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG

from .models import RateLimitSnapshot


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        val_str = str(value).strip()
        if not val_str:
            return None
        # Handle cases where value might be float string like "100.0"
        return int(float(val_str))
    except (ValueError, TypeError):
        return None


def _clean_header_dict(headers: Mapping[str, str]) -> dict[str, str]:
    """Return lowercase headers containing only rate limit and quota keys (no auth/cookies)."""
    clean: dict[str, str] = {}
    for k, v in headers.items():
        lower_k = k.lower()
        if any(token in lower_k for token in ("ratelimit", "rate-limit", "retry-after", "quota")):
            clean[lower_k] = str(v)
    return clean


class ProviderUsageAdapter(ABC):
    """Normalized provider capability adapter for telemetry and quota extraction."""

    def __init__(self, provider_id: str) -> None:
        self._provider_id = provider_id

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def is_local(self) -> bool:
        desc = PROVIDER_CATALOG.get(self._provider_id)
        return desc.local if desc is not None else False

    @abstractmethod
    def extract_rate_limits(
        self,
        headers: Mapping[str, str],
        *,
        updated_at: float | None = None,
    ) -> RateLimitSnapshot:
        """Extract sanitized quota and rate limit status from upstream HTTP response headers."""
        ...


class GroqUsageAdapter(ProviderUsageAdapter):
    """Groq rate limit header parser."""

    def __init__(self) -> None:
        super().__init__("groq")

    def extract_rate_limits(
        self,
        headers: Mapping[str, str],
        *,
        updated_at: float | None = None,
    ) -> RateLimitSnapshot:
        ts = updated_at or time.time()
        norm = {k.lower(): str(v) for k, v in headers.items()}
        req_limit = _int_or_none(norm.get("x-ratelimit-limit-requests"))
        req_rem = _int_or_none(norm.get("x-ratelimit-remaining-requests"))
        req_reset = norm.get("x-ratelimit-reset-requests")

        tok_limit = _int_or_none(norm.get("x-ratelimit-limit-tokens"))
        tok_rem = _int_or_none(norm.get("x-ratelimit-remaining-tokens"))
        tok_reset = norm.get("x-ratelimit-reset-tokens")

        retry_after = _int_or_none(norm.get("retry-after"))
        quota_available = (req_limit is not None or tok_limit is not None or req_rem is not None)

        return RateLimitSnapshot(
            provider_id=self.provider_id,
            updated_at=ts,
            quota_available=quota_available,
            status_label="Connected" if quota_available else "Provider quota unavailable",
            requests_limit=req_limit,
            requests_remaining=req_rem,
            requests_reset=req_reset,
            tokens_limit=tok_limit,
            tokens_remaining=tok_rem,
            tokens_reset=tok_reset,
            retry_after=retry_after,
            raw_headers=_clean_header_dict(headers),
        )


class AnthropicUsageAdapter(ProviderUsageAdapter):
    """Anthropic / native Messages protocol rate limit header parser."""

    def __init__(self, provider_id: str = "anthropic") -> None:
        super().__init__(provider_id)

    def extract_rate_limits(
        self,
        headers: Mapping[str, str],
        *,
        updated_at: float | None = None,
    ) -> RateLimitSnapshot:
        ts = updated_at or time.time()
        norm = {k.lower(): str(v) for k, v in headers.items()}
        req_limit = _int_or_none(norm.get("anthropic-ratelimit-requests-limit"))
        req_rem = _int_or_none(norm.get("anthropic-ratelimit-requests-remaining"))
        req_reset = norm.get("anthropic-ratelimit-requests-reset")

        tok_limit = _int_or_none(norm.get("anthropic-ratelimit-tokens-limit"))
        tok_rem = _int_or_none(norm.get("anthropic-ratelimit-tokens-remaining"))
        tok_reset = norm.get("anthropic-ratelimit-tokens-reset")

        retry_after = _int_or_none(norm.get("retry-after"))
        quota_available = (req_limit is not None or tok_limit is not None or req_rem is not None)

        return RateLimitSnapshot(
            provider_id=self.provider_id,
            updated_at=ts,
            quota_available=quota_available,
            status_label="Connected" if quota_available else "Provider quota unavailable",
            requests_limit=req_limit,
            requests_remaining=req_rem,
            requests_reset=req_reset,
            tokens_limit=tok_limit,
            tokens_remaining=tok_rem,
            tokens_reset=tok_reset,
            retry_after=retry_after,
            raw_headers=_clean_header_dict(headers),
        )


class OllamaUsageAdapter(ProviderUsageAdapter):
    """Ollama local engine adapter: local / no cloud token quota."""

    def __init__(self, provider_id: str = "ollama") -> None:
        super().__init__(provider_id)

    @property
    def is_local(self) -> bool:
        return True

    def extract_rate_limits(
        self,
        headers: Mapping[str, str],
        *,
        updated_at: float | None = None,
    ) -> RateLimitSnapshot:
        ts = updated_at or time.time()
        return RateLimitSnapshot(
            provider_id=self.provider_id,
            updated_at=ts,
            quota_available=False,
            status_label="Local / none",
            requests_limit=None,
            requests_remaining=None,
            requests_reset=None,
            tokens_limit=None,
            tokens_remaining=None,
            tokens_reset=None,
            retry_after=None,
            raw_headers={},
        )


class NvidiaNimUsageAdapter(ProviderUsageAdapter):
    """NVIDIA NIM provider adapter."""

    def __init__(self) -> None:
        super().__init__("nvidia_nim")

    def extract_rate_limits(
        self,
        headers: Mapping[str, str],
        *,
        updated_at: float | None = None,
    ) -> RateLimitSnapshot:
        ts = updated_at or time.time()
        norm = {k.lower(): str(v) for k, v in headers.items()}
        # Check standard x-ratelimit headers if present
        req_limit = _int_or_none(norm.get("x-ratelimit-limit-requests"))
        req_rem = _int_or_none(norm.get("x-ratelimit-remaining-requests"))
        req_reset = norm.get("x-ratelimit-reset-requests")

        tok_limit = _int_or_none(norm.get("x-ratelimit-limit-tokens"))
        tok_rem = _int_or_none(norm.get("x-ratelimit-remaining-tokens"))
        tok_reset = norm.get("x-ratelimit-reset-tokens")

        retry_after = _int_or_none(norm.get("retry-after"))
        quota_available = (req_limit is not None or tok_limit is not None or req_rem is not None)

        return RateLimitSnapshot(
            provider_id=self.provider_id,
            updated_at=ts,
            quota_available=quota_available,
            status_label="Connected" if quota_available else "Not exposed by provider",
            requests_limit=req_limit,
            requests_remaining=req_rem,
            requests_reset=req_reset,
            tokens_limit=tok_limit,
            tokens_remaining=tok_rem,
            tokens_reset=tok_reset,
            retry_after=retry_after,
            raw_headers=_clean_header_dict(headers),
        )


class StandardOpenAIUsageAdapter(ProviderUsageAdapter):
    """Universal OpenAI-compatible rate limit header extractor."""

    def extract_rate_limits(
        self,
        headers: Mapping[str, str],
        *,
        updated_at: float | None = None,
    ) -> RateLimitSnapshot:
        ts = updated_at or time.time()
        norm = {k.lower(): str(v) for k, v in headers.items()}

        req_limit = (
            _int_or_none(norm.get("x-ratelimit-limit-requests"))
            or _int_or_none(norm.get("ratelimit-limit-requests"))
        )
        req_rem = (
            _int_or_none(norm.get("x-ratelimit-remaining-requests"))
            or _int_or_none(norm.get("ratelimit-remaining-requests"))
        )
        req_reset = (
            norm.get("x-ratelimit-reset-requests")
            or norm.get("ratelimit-reset-requests")
        )

        tok_limit = (
            _int_or_none(norm.get("x-ratelimit-limit-tokens"))
            or _int_or_none(norm.get("ratelimit-limit-tokens"))
        )
        tok_rem = (
            _int_or_none(norm.get("x-ratelimit-remaining-tokens"))
            or _int_or_none(norm.get("ratelimit-remaining-tokens"))
        )
        tok_reset = (
            norm.get("x-ratelimit-reset-tokens")
            or norm.get("ratelimit-reset-tokens")
        )

        retry_after = _int_or_none(norm.get("retry-after"))
        quota_available = (req_limit is not None or tok_limit is not None or req_rem is not None)

        label = (
            "Local / none"
            if self.is_local
            else ("Connected" if quota_available else "Provider quota unavailable")
        )

        return RateLimitSnapshot(
            provider_id=self.provider_id,
            updated_at=ts,
            quota_available=quota_available,
            status_label=label,
            requests_limit=req_limit,
            requests_remaining=req_rem,
            requests_reset=req_reset,
            tokens_limit=tok_limit,
            tokens_remaining=tok_rem,
            tokens_reset=tok_reset,
            retry_after=retry_after,
            raw_headers=_clean_header_dict(headers),
        )


_ADAPTER_INSTANCES: dict[str, ProviderUsageAdapter] = {
    "groq": GroqUsageAdapter(),
    "nvidia_nim": NvidiaNimUsageAdapter(),
    "ollama": OllamaUsageAdapter("ollama"),
    "lmstudio": OllamaUsageAdapter("lmstudio"),
    "llamacpp": OllamaUsageAdapter("llamacpp"),
    "github_copilot": AnthropicUsageAdapter("github_copilot"),
}


def get_usage_adapter(provider_id: str) -> ProviderUsageAdapter:
    """Resolve or construct the usage adapter for any known or future provider."""
    adapter = _ADAPTER_INSTANCES.get(provider_id)
    if adapter is not None:
        return adapter
    return StandardOpenAIUsageAdapter(provider_id)

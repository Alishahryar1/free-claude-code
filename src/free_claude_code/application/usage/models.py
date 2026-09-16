"""Normalized domain models for provider usage and observability telemetry."""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Telemetry record for one physical or candidate execution attempt."""

    request_id: str
    timestamp: float
    wire_api: str  # "messages" | "responses"
    gateway_model: str  # Original model requested by client
    provider_id: str  # Provider ID (e.g. "groq", "nvidia_nim", "ollama")
    provider_model: str  # Upstream model name
    provider_model_ref: str  # Normalized provider:model ref
    is_primary: bool
    fallback_index: int = 0
    fallback_from_ref: str | None = None
    fallback_reason: str | None = None
    status: str = "success"  # "success" | "failed"
    http_status: int | None = None
    error_category: str | None = None
    duration_ms: float = 0.0
    ttft_ms: float | None = None  # Time to first token
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    is_estimated: bool = False
    tokens_per_second: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass(frozen=True, slots=True)
class RateLimitSnapshot:
    """Latest quota and rate limit status captured from upstream response headers."""

    provider_id: str
    updated_at: float
    quota_available: bool
    status_label: (
        str  # e.g. "Quota active", "Provider quota unavailable", "Local / none"
    )
    requests_limit: int | None = None
    requests_remaining: int | None = None
    requests_reset: str | None = None
    tokens_limit: int | None = None
    tokens_remaining: int | None = None
    tokens_reset: str | None = None
    retry_after: int | None = None
    raw_headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """Aggregated usage metrics across a selected time horizon."""

    time_range: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_tokens: int = 0
    fallback_count: int = 0
    avg_duration_ms: float = 0.0
    avg_ttft_ms: float | None = None
    estimated_cost: float | None = None
    cost_label: str = "Unknown"


@dataclass(frozen=True, slots=True)
class ModelUsageStats:
    """Usage breakdown aggregated for one model."""

    model_ref: str
    provider_id: str
    provider_model: str
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    avg_duration_ms: float
    tokens_per_second: float | None
    error_count: int
    error_rate: float


@dataclass(frozen=True, slots=True)
class ProviderUsageStats:
    """Usage breakdown aggregated for one provider."""

    provider_id: str
    display_name: str
    is_local: bool
    requests: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    success_count: int
    failure_count: int
    error_rate: float
    quota: RateLimitSnapshot | None
    avg_speed_tok_s: float | None


@dataclass(frozen=True, slots=True)
class FallbackDestinationStats:
    """Statistics for one fallback target model."""

    target_ref: str
    triggered_count: int
    succeeded_count: int
    avg_duration_ms: float


@dataclass(frozen=True, slots=True)
class FallbackAnalytics:
    """Detailed routing and fallback behavior metrics."""

    primary_requests: int
    primary_successes: int
    primary_fallbacks: int
    fallback_success_rate: float
    destinations: tuple[FallbackDestinationStats, ...] = ()
    reasons: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class TimeSeriesPoint:
    """One discrete time interval bucket for graphing."""

    bucket_start: float
    label: str
    requests: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    errors: int
    fallbacks: int
    avg_duration_ms: float

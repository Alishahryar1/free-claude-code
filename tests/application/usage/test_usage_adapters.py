"""Unit tests for normalized provider usage adapters."""

from free_claude_code.application.usage.adapter import (
    AnthropicUsageAdapter,
    GroqUsageAdapter,
    NvidiaNimUsageAdapter,
    OllamaUsageAdapter,
    StandardOpenAIUsageAdapter,
    get_usage_adapter,
)


def test_groq_usage_adapter_parses_ratelimit_headers():
    adapter = GroqUsageAdapter()
    headers = {
        "x-ratelimit-limit-requests": "14400",
        "x-ratelimit-remaining-requests": "14399",
        "x-ratelimit-reset-requests": "6s",
        "x-ratelimit-limit-tokens": "18000",
        "x-ratelimit-remaining-tokens": "17950",
        "x-ratelimit-reset-tokens": "166ms",
        "retry-after": "2",
    }
    snapshot = adapter.extract_rate_limits(headers)
    assert snapshot.provider_id == "groq"
    assert snapshot.quota_available is True
    assert snapshot.requests_limit == 14400
    assert snapshot.requests_remaining == 14399
    assert snapshot.requests_reset == "6s"
    assert snapshot.tokens_limit == 18000
    assert snapshot.tokens_remaining == 17950
    assert snapshot.tokens_reset == "166ms"
    assert snapshot.retry_after == 2
    assert snapshot.status_label == "Connected"


def test_groq_usage_adapter_empty_headers_marks_unavailable():
    adapter = GroqUsageAdapter()
    snapshot = adapter.extract_rate_limits({})
    assert snapshot.quota_available is False
    assert snapshot.status_label == "Provider quota unavailable"
    assert snapshot.requests_limit is None


def test_anthropic_usage_adapter_parses_headers():
    adapter = AnthropicUsageAdapter()
    headers = {
        "anthropic-ratelimit-requests-limit": "1000",
        "anthropic-ratelimit-requests-remaining": "995",
        "anthropic-ratelimit-requests-reset": "2026-09-16T12:00:00Z",
        "anthropic-ratelimit-tokens-limit": "400000",
        "anthropic-ratelimit-tokens-remaining": "395000",
        "anthropic-ratelimit-tokens-reset": "2026-09-16T12:00:00Z",
    }
    snapshot = adapter.extract_rate_limits(headers)
    assert snapshot.provider_id == "anthropic"
    assert snapshot.quota_available is True
    assert snapshot.requests_limit == 1000
    assert snapshot.requests_remaining == 995
    assert snapshot.tokens_limit == 400000
    assert snapshot.tokens_remaining == 395000


def test_ollama_usage_adapter_local_no_cloud_quota():
    adapter = OllamaUsageAdapter()
    assert adapter.is_local is True
    snapshot = adapter.extract_rate_limits({})
    assert snapshot.provider_id == "ollama"
    assert snapshot.quota_available is False
    assert snapshot.status_label == "Local / none"
    assert snapshot.requests_limit is None
    assert snapshot.tokens_limit is None


def test_nvidia_nim_usage_adapter_defaults_to_unavailable_when_no_headers():
    adapter = NvidiaNimUsageAdapter()
    assert adapter.is_local is False
    snapshot = adapter.extract_rate_limits({})
    assert snapshot.quota_available is False
    assert snapshot.status_label == "Not exposed by provider"


def test_get_usage_adapter_routing():
    assert isinstance(get_usage_adapter("groq"), GroqUsageAdapter)
    assert isinstance(get_usage_adapter("anthropic"), StandardOpenAIUsageAdapter)
    assert isinstance(get_usage_adapter("ollama"), OllamaUsageAdapter)
    assert isinstance(get_usage_adapter("nvidia_nim"), NvidiaNimUsageAdapter)
    assert isinstance(get_usage_adapter("together"), StandardOpenAIUsageAdapter)
    assert isinstance(get_usage_adapter("custom_unknown"), StandardOpenAIUsageAdapter)

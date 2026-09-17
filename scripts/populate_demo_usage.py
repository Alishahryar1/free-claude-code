"""Populate realistic sample telemetry into FCC's usage database for testing and demonstration."""

import asyncio
import random
import time

from free_claude_code.application.usage.models import RateLimitSnapshot, UsageRecord
from free_claude_code.config.paths import usage_database_path
from free_claude_code.runtime.usage_sqlite import UsageSqliteStore


async def main():
    store = UsageSqliteStore(usage_database_path())
    now = time.time()

    print(f"Populating usage data into {usage_database_path()}...")

    # 1. Quota snapshots for providers
    groq_quota = RateLimitSnapshot(
        provider_id="groq",
        updated_at=now,
        quota_available=True,
        status_label="Connected",
        requests_limit=14400,
        requests_remaining=13840,
        requests_reset="4s",
        tokens_limit=18000,
        tokens_remaining=16200,
        tokens_reset="180ms",
        retry_after=None,
    )
    await store.update_quota(groq_quota)

    anthropic_quota = RateLimitSnapshot(
        provider_id="anthropic",
        updated_at=now,
        quota_available=True,
        status_label="Connected",
        requests_limit=1000,
        requests_remaining=945,
        requests_reset="2026-09-16T12:00:00Z",
        tokens_limit=400000,
        tokens_remaining=382500,
        tokens_reset="2026-09-16T12:00:00Z",
        retry_after=None,
    )
    await store.update_quota(anthropic_quota)

    nvidia_quota = RateLimitSnapshot(
        provider_id="nvidia_nim",
        updated_at=now,
        quota_available=False,
        status_label="Not exposed by provider",
    )
    await store.update_quota(nvidia_quota)

    ollama_quota = RateLimitSnapshot(
        provider_id="ollama",
        updated_at=now,
        quota_available=False,
        status_label="Local / none",
    )
    await store.update_quota(ollama_quota)

    # 2. Add realistic records across past 24 hours
    models = [
        (
            "groq",
            "llama-3.3-70b-versatile",
            "groq:llama-3.3-70b-versatile",
            250.0,
            140.0,
        ),
        (
            "anthropic",
            "claude-3-5-sonnet-20241022",
            "anthropic:claude-3-5-sonnet-20241022",
            1200.0,
            45.0,
        ),
        (
            "nvidia_nim",
            "meta/llama-3.1-405b-instruct",
            "nvidia_nim:meta/llama-3.1-405b-instruct",
            900.0,
            60.0,
        ),
        ("ollama", "qwen2.5-coder:7b", "ollama:qwen2.5-coder:7b", 650.0, 85.0),
    ]

    for i in range(45):
        prov_id, prov_mod, mod_ref, base_dur, base_speed = random.choice(models)
        rec_time = now - random.uniform(60, 22 * 3600)
        dur = base_dur * random.uniform(0.7, 1.4)
        in_tok = random.randint(300, 2500)
        out_tok = random.randint(50, 800)
        cached = int(in_tok * 0.35) if prov_id in ("anthropic", "groq") else 0
        reasoning = int(out_tok * 0.25) if prov_id == "groq" else 0
        speed = base_speed * random.uniform(0.85, 1.15)
        status = "failed" if random.random() < 0.05 else "success"

        rec = UsageRecord(
            request_id=f"req-demo-{i:03d}",
            timestamp=rec_time,
            wire_api="messages",
            gateway_model="claude-3-5-sonnet",
            provider_id=prov_id,
            provider_model=prov_mod,
            provider_model_ref=mod_ref,
            is_primary=True,
            status=status,
            http_status=429 if status == "failed" else 200,
            duration_ms=round(dur, 1),
            ttft_ms=round(dur * 0.25, 1),
            input_tokens=in_tok,
            output_tokens=out_tok if status == "success" else 0,
            total_tokens=in_tok + (out_tok if status == "success" else 0),
            cached_tokens=cached,
            reasoning_tokens=reasoning,
            tokens_per_second=round(speed, 1),
        )
        await store.insert_record(rec)

        # If failed, add a fallback execution that succeeded on Ollama
        if status == "failed":
            fb_rec = UsageRecord(
                request_id=f"req-demo-{i:03d}",
                timestamp=rec_time + 1.2,
                wire_api="messages",
                gateway_model="claude-3-5-sonnet",
                provider_id="ollama",
                provider_model="qwen2.5-coder:7b",
                provider_model_ref="ollama:qwen2.5-coder:7b",
                is_primary=False,
                fallback_index=1,
                fallback_from_ref=mod_ref,
                fallback_reason="rate_limit",
                status="success",
                http_status=200,
                duration_ms=750.0,
                ttft_ms=180.0,
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=in_tok + out_tok,
                tokens_per_second=78.5,
            )
            await store.insert_record(fb_rec)

    summary = await store.query_summary(since=now - 86400, time_range_label="24h")
    print("Sample data loaded successfully!")
    print(f"Total requests: {summary.total_requests}")
    print(
        f"Successful: {summary.successful_requests}, Failed: {summary.failed_requests}"
    )
    print(f"Total tokens: {summary.total_tokens:,}")
    print(f"Fallbacks: {summary.fallback_count}")


if __name__ == "__main__":
    asyncio.run(main())

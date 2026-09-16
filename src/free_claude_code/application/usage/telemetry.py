"""Lightweight stream telemetry extraction for Anthropic and Responses wire protocols."""

import json
from time import monotonic, time
from typing import Any

from free_claude_code.application.routing import ProviderModelTarget

from .models import UsageRecord


class StreamTelemetryTracker:
    """Tracks one candidate attempt's stream timing, tokens, and outcome."""

    def __init__(
        self,
        *,
        request_id: str,
        wire_api: str,
        gateway_model: str,
        target: ProviderModelTarget,
        is_primary: bool,
        fallback_index: int = 0,
        fallback_from_ref: str | None = None,
        fallback_reason: str | None = None,
        estimated_input_tokens: int = 0,
    ) -> None:
        self.request_id = request_id
        self.wire_api = wire_api
        self.gateway_model = gateway_model
        self.target = target
        self.is_primary = is_primary
        self.fallback_index = fallback_index
        self.fallback_from_ref = fallback_from_ref
        self.fallback_reason = fallback_reason
        self.estimated_input_tokens = estimated_input_tokens

        self._wall_start = time()
        self._mono_start = monotonic()
        self._mono_first_token: float | None = None

        self._input_tokens = estimated_input_tokens
        self._output_tokens = 0
        self._cached_tokens = 0
        self._reasoning_tokens = 0
        self._has_exact_usage = False
        self._accumulated_text_chars = 0

    def on_chunk(self, chunk: str) -> None:
        if not chunk:
            return

        if self._mono_first_token is None:
            self._mono_first_token = monotonic()

        # Fast heuristic check for SSE usage payloads before full JSON parsing
        if (
            "usage" in chunk
            or "message_start" in chunk
            or "message_delta" in chunk
            or "response.completed" in chunk
        ):
            self._try_parse_chunk_usage(chunk)
        else:
            self._accumulated_text_chars += len(chunk)

    def _try_parse_chunk_usage(self, chunk: str) -> None:
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                payload = json.loads(data_str)
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            # Anthropic message_start
            if payload.get("type") == "message_start":
                msg = payload.get("message")
                if isinstance(msg, dict):
                    usage = msg.get("usage")
                    if isinstance(usage, dict):
                        if "input_tokens" in usage:
                            self._input_tokens = int(usage["input_tokens"])
                            self._has_exact_usage = True
                        self._cached_tokens = int(
                            usage.get("cache_read_input_tokens")
                            or usage.get("cache_creation_input_tokens")
                            or 0
                        )

            # Anthropic message_delta
            elif payload.get("type") == "message_delta":
                usage = payload.get("usage")
                if isinstance(usage, dict) and "output_tokens" in usage:
                    self._output_tokens = int(usage["output_tokens"])
                    self._has_exact_usage = True

            # OpenAI Responses response.completed / response.done
            elif payload.get("type") in ("response.completed", "response.done"):
                resp = payload.get("response")
                if isinstance(resp, dict):
                    usage = resp.get("usage")
                    if isinstance(usage, dict):
                        self._input_tokens = int(
                            usage.get("input_tokens", self._input_tokens)
                        )
                        self._output_tokens = int(
                            usage.get("output_tokens", self._output_tokens)
                        )
                        details = usage.get("output_token_details")
                        if isinstance(details, dict):
                            self._reasoning_tokens = int(
                                details.get("reasoning_tokens", 0)
                            )
                        cached_details = usage.get("input_token_details")
                        if isinstance(cached_details, dict):
                            self._cached_tokens = int(
                                cached_details.get("cached_tokens", 0)
                            )
                        self._has_exact_usage = True

    def on_success(self) -> UsageRecord:
        now_mono = monotonic()
        dur_ms = round((now_mono - self._mono_start) * 1000.0, 2)
        ttft_ms = (
            round((self._mono_first_token - self._mono_start) * 1000.0, 2)
            if self._mono_first_token is not None
            else None
        )

        out_tokens = self._output_tokens
        is_est = False
        if not self._has_exact_usage:
            # Estimate output tokens from characters if upstream did not report usage
            out_tokens = max(1, self._accumulated_text_chars // 4)
            is_est = True

        total_tokens = self._input_tokens + out_tokens
        dur_s = dur_ms / 1000.0
        tok_s = (
            round(out_tokens / dur_s, 2) if (out_tokens > 0 and dur_s > 0.05) else None
        )

        return UsageRecord(
            request_id=self.request_id,
            timestamp=self._wall_start,
            wire_api=self.wire_api,
            gateway_model=self.gateway_model,
            provider_id=self.target.provider_id,
            provider_model=self.target.provider_model,
            provider_model_ref=self.target.provider_model_ref,
            is_primary=self.is_primary,
            fallback_index=self.fallback_index,
            fallback_from_ref=self.fallback_from_ref,
            fallback_reason=self.fallback_reason,
            status="success",
            http_status=200,
            duration_ms=dur_ms,
            ttft_ms=ttft_ms,
            input_tokens=self._input_tokens,
            output_tokens=out_tokens,
            total_tokens=total_tokens,
            cached_tokens=self._cached_tokens,
            reasoning_tokens=self._reasoning_tokens,
            is_estimated=is_est,
            tokens_per_second=tok_s,
        )

    def on_failure(self, error: Any) -> UsageRecord:
        now_mono = monotonic()
        dur_ms = round((now_mono - self._mono_start) * 1000.0, 2)
        ttft_ms = (
            round((self._mono_first_token - self._mono_start) * 1000.0, 2)
            if self._mono_first_token is not None
            else None
        )

        http_status = getattr(error, "status_code", None)
        if http_status is None:
            resp = getattr(error, "response", None)
            http_status = getattr(resp, "status_code", None)

        err_kind = type(error).__name__
        if hasattr(error, "kind") and error.kind:
            err_kind = str(error.kind)

        err_msg = str(
            getattr(error, "message", None) or error or "Unknown execution error"
        )

        return UsageRecord(
            request_id=self.request_id,
            timestamp=self._wall_start,
            wire_api=self.wire_api,
            gateway_model=self.gateway_model,
            provider_id=self.target.provider_id,
            provider_model=self.target.provider_model,
            provider_model_ref=self.target.provider_model_ref,
            is_primary=self.is_primary,
            fallback_index=self.fallback_index,
            fallback_from_ref=self.fallback_from_ref,
            fallback_reason=self.fallback_reason or err_msg,
            status="failed",
            http_status=http_status or 500,
            error_category=err_kind,
            duration_ms=dur_ms,
            ttft_ms=ttft_ms,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self._input_tokens + self._output_tokens,
            cached_tokens=self._cached_tokens,
            reasoning_tokens=self._reasoning_tokens,
            is_estimated=True,
            tokens_per_second=None,
        )

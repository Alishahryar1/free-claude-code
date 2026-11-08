"""Seal native continuation replay against output actually released to the client."""

from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

from .continuation_request import ContinuationRequest
from .history_replay import (
    HistoryReplayError,
    decode_gemini_replay,
    decode_replay,
    encode_gemini_replay,
    encode_replay,
    is_replay,
)
from .recovery_limits import RECOVERY_RETENTION_BYTES
from .stream_recovery import DeliveredPrefix


def _carrier_fields(payload: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    block = payload.get("content_block")
    if isinstance(block, dict):
        if block.get("type") == "redacted_thinking":
            yield block, "data"
        elif block.get("type") == "thinking":
            yield block, "signature"
    for item in [
        payload.get("item"),
        *((payload.get("response") or {}).get("output") or []),
    ]:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            yield item, "encrypted_content"
    for choice in payload.get("choices") or []:
        for detail in (choice.get("delta") or {}).get("reasoning_details") or []:
            if isinstance(detail, dict) and detail.get("type") == "reasoning.encrypted":
                yield detail, "data"
    for candidate in payload.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            yield part, "thoughtSignature"


def finalize_replay(
    payload: dict[str, Any],
    prefix: Callable[[], DeliveredPrefix],
    sealed: dict[str, str],
) -> bool:
    """Replace draft carriers in known protocol fields, before their publication."""
    changed = False
    for container, key in _carrier_fields(payload):
        value = container.get(key)
        if not isinstance(value, str):
            continue
        if value in sealed:
            container[key] = sealed[value]
            changed = True
            continue
        native = key == "thoughtSignature"
        record = (
            decode_gemini_replay(value)
            if native
            else decode_replay(value)
            if is_replay(value)
            else None
        )
        if record is None:
            continue
        continuation = record.continuation
        if continuation is None or continuation.echo is not None:
            continue
        seed = prefix().continuation
        if seed is None:
            raise HistoryReplayError(
                "The public response cannot be retained for continuation replay."
            )
        # These calls are completed output. Their results arrive next turn.
        history = ContinuationRequest("gemini").history(
            seed, {"model": record.origin.model}, allow_pending_calls=True
        )
        if len(history) != 1:
            raise HistoryReplayError(
                "Continuation replay requires one public assistant turn."
            )
        encoded = (encode_gemini_replay if native else encode_replay)(
            replace(
                record,
                continuation=replace(continuation, echo=tuple(history[0]["parts"])),
            )
        )
        if (
            sum(len(item) for item in sealed.values()) + len(encoded)
            > RECOVERY_RETENTION_BYTES
        ):
            raise HistoryReplayError(
                "Continuation replay exceeded the retention limit."
            )
        sealed[value] = encoded
        container[key] = encoded
        changed = True
    return changed

"""Prepare tolerant computation intent without guessing backend identity."""

import json
import re
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.diagnostics import extract_upstream_error_detail
from free_claude_code.core.reasoning import (
    ReasoningCapability,
    ReasoningControl,
    ReasoningPolicy,
)


def prepare_messages_reasoning(
    request: MessagesRequest,
    reasoning: ReasoningPolicy,
    *,
    model_info: ProviderModelInfo | None,
    can_disable: bool,
    normal_max_tokens: int | None,
) -> tuple[MessagesRequest, ReasoningPolicy]:
    """Resolve only tolerant intent; retain original intent in execution owners."""
    if reasoning.control is not ReasoningControl.PREFER_OFF:
        return request, reasoning
    capability = (
        model_info.reasoning_capability if model_info else ReasoningCapability.UNKNOWN
    )
    disable = can_disable and capability not in {
        ReasoningCapability.NONE,
        ReasoningCapability.REQUIRED,
    }
    normal_allowance = capability not in {
        ReasoningCapability.NONE,
        ReasoningCapability.OPTIONAL,
    } or (capability is ReasoningCapability.OPTIONAL and not can_disable)
    output_config = dict(request.output_config or {})
    output_config.pop("effort", None)
    limit = request.max_tokens
    if normal_allowance:
        limit = (
            max(limit or 0, normal_max_tokens)
            if normal_max_tokens is not None
            else None
        )
    return request.model_copy(
        update={
            "thinking": None,
            "output_config": output_config or None,
            "max_tokens": limit,
        }
    ), ReasoningPolicy.off() if disable else ReasoningPolicy.provider_default()


@dataclass(frozen=True, slots=True)
class ReasoningCorrection:
    """One request's owned OFF field and normal output allowance."""

    off_fields: tuple[tuple[str, ...], ...]
    limit_field: str
    normal_max_tokens: int | None
    output_cap: int | None = None
    provider_rejection: Callable[[Exception], bool] | None = None

    def retry_body(
        self,
        error: Exception,
        body: dict[str, Any],
        *,
        sent_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Remove the sent control only after an explicit validation rejection."""
        present = []
        for path in self.off_fields:
            target: object = body
            sent: object = body if sent_body is None else sent_body
            for key in path:
                if (
                    not isinstance(target, Mapping)
                    or key not in target
                    or not isinstance(sent, Mapping)
                    or key not in sent
                ):
                    break
                target = target[key]
                sent = sent[key]
            else:
                if target == sent:
                    present.append(path)
        if not present or not (
            any(_disable_rejected(error, path) for path in present)
            or (self.provider_rejection is not None and self.provider_rejection(error))
        ):
            return None
        result = deepcopy(body)
        for path in present:
            parents = []
            current = result
            for key in path[:-1]:
                parents.append((current, key))
                current = current[key]
            del current[path[-1]]
            for parent, key in reversed(parents):
                if not parent[key]:
                    del parent[key]
        if self.normal_max_tokens is None:
            result.pop(self.limit_field, None)
        else:
            existing = result.get(self.limit_field)
            limit = max(
                existing if isinstance(existing, int) else 0, self.normal_max_tokens
            )
            result[self.limit_field] = (
                min(limit, self.output_cap) if self.output_cap is not None else limit
            )
        return result


def _disable_rejected(error: Exception, field: tuple[str, ...]) -> bool:
    detail = extract_upstream_error_detail(error)
    if not detail.is_invalid_request(code=getattr(error, "code", None)):
        return False
    text = detail.body_text or detail.exception_text or ""
    try:
        payload = json.loads(text)
    except ValueError:
        payload = {"message": text}
    return reasoning_control_rejected(payload, field)


def reasoning_control_rejected(payload: object, field: tuple[str, ...]) -> bool:
    """Match an explicit rejection tied to an owned field in an error payload."""
    return any(
        _record_rejects_control(record, field) for record in _error_records(payload)
    )


def _error_records(payload: object) -> Iterator[Mapping[str, object]]:
    if isinstance(payload, Mapping):
        if any(
            isinstance(payload.get(key), str) for key in ("message", "msg", "detail")
        ):
            yield payload
        for key in ("error", "errors", "detail"):
            if key in payload:
                yield from _error_records(payload[key])
    elif isinstance(payload, list):
        for item in payload:
            yield from _error_records(item)


def _record_rejects_control(
    record: Mapping[str, object], field: tuple[str, ...]
) -> bool:
    message = next(
        (
            record[key]
            for key in ("message", "msg", "detail")
            if isinstance(record.get(key), str)
        ),
        "",
    )
    text = str(message).lower()
    param = record.get("param", record.get("loc"))
    rejected = r"(?:not supported|unsupported|unknown parameter|unrecognized|extra inputs are not permitted|must be one of|not allowed|invalid value)"
    if param is not None:
        parts = re.findall(r"[a-z_][a-z_0-9]*", str(param).lower())
        while parts and parts[0] in {"body", "request", "extra_body"}:
            parts.pop(0)
        expected = [part for part in field if part != "extra_body"]
        matches = parts[: len(expected)] == expected or parts == [field[-1]]
        if not matches:
            return False
        return bool(re.search(rejected, text) or _mandatory_reasoning(text))
    if _mandatory_reasoning(text):
        return True
    name = re.escape(field[-1]) + r"(?:\.[a-z_][a-z_0-9]*)*"
    return bool(
        re.search(
            rf"\b{name}\b['\"\s:]*(?:(?:parameter|field|value)\s+)?(?:is\s+)?{rejected}"
            rf"|\b(?:unsupported|unrecognized|unknown)(?:\s+(?:parameter|field|argument))?[:\s'\"]+{name}\b"
            rf"|\bdoes not support\s+['\"]?{name}\b",
            text,
        )
    )


def _mandatory_reasoning(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:reasoning|thinking)\s+(?:is\s+)?(?:mandatory|required|cannot be disabled|can't be disabled)\b",
            text,
        )
    )

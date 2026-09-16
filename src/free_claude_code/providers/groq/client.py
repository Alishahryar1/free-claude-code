"""Groq chat-completions provider with per-model reasoning negotiation."""

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import openai
from loguru import logger

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.reasoning import (
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    NamedEffortReasoning,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    OpenAIModelListing,
    validate_extra_body_does_not_override_reasoning_fields,
)

from .tpm import correct_tpm_completion_budget

_GROQ_EFFORTS = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "medium"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "high"),
    (ReasoningEffort.MAX, "high"),
)
_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="GROQ",
    reasoning_replay=ReasoningReplayMode.THINK_TAGS,
    include_extra_body=True,
    extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
    max_tokens_field="max_completion_tokens",
    strip_message_names=True,
    unsupported_body_keys=frozenset({"logprobs", "logit_bias", "top_logprobs"}),
    normalize_n_to_one=True,
)
_PROFILE = OpenAIChatProfile(
    _REQUEST_POLICY,
    NamedEffortReasoning(
        _GROQ_EFFORTS,
        disabled_value="none",
        enabled_value="medium",
    ),
    model_listing=OpenAIModelListing(
        context_window_tokens_path=("context_window",),
        max_output_tokens_path=("max_completion_tokens",),
    ),
)

_REASONING_FIELD = "reasoning_effort"
_KNOWN_VALUES = frozenset({"none", "default", "low", "medium", "high"})
_FIELD_PATTERN = re.compile(r"(?<![a-z0-9_])reasoning_effort(?![a-z0-9_])", re.I)
_VALUE_PATTERN = re.compile(
    r"(?<![a-z0-9_])(none|default|low|medium|high)(?![a-z0-9_])",
    re.I,
)
_ALLOWED_MARKER = re.compile(
    r"(?:"
    r"(?:must|should|needs?\s+to)\s+be\s+one\s+of"
    r"|expected(?:\s+to\s+be)?\s+one\s+of"
    r"|(?:allowed|valid|accepted)\s+(?:values?|options?)"
    r")",
    re.I,
)
_CLAUSE_END = re.compile(r"(?:\r?\n|;|[.!?](?:\s|$))")
_UNSUPPORTED_PHRASE = re.compile(
    r"(?:"
    r"unsupported\s+(?:parameter|field)"
    r"|(?:parameter|field)(?:\s+\S+){0,3}\s+is\s+not\s+supported"
    r"|unknown\s+(?:parameter|field)"
    r"|unrecognized\s+(?:parameter|field)"
    r"|does\s+not\s+support"
    r"|is\s+not\s+supported"
    r")",
    re.I,
)


@dataclass(frozen=True, slots=True)
class _ErrorCandidate:
    text: str
    field_context: bool = False


class GroqChatBehavior(OpenAIChatBehavior):
    """Groq Chat adaptation without HTTP ownership."""

    def __init__(self) -> None:
        super().__init__(_PROFILE)
        self._model_reasoning_vocabularies: dict[str, frozenset[str]] = {}

    def finalize_chat_body(
        self,
        body: dict[str, Any],
        *,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        model = body.get("model")
        if not isinstance(model, str):
            return body
        accepted = self._model_reasoning_vocabularies.get(model)
        if accepted is None:
            return body
        return _rewrite_reasoning_effort(body, accepted) or body

    def reasoning_disable_rejected(self, error: Exception) -> bool:
        accepted = _parse_reasoning_vocabulary(error)
        return accepted is not None and "none" not in accepted

    def retry_request_body(
        self, error: Exception, body: dict[str, Any]
    ) -> dict[str, Any] | None:
        accepted = _parse_reasoning_vocabulary(error)
        model = body.get("model")
        if accepted is None or not isinstance(model, str):
            return None

        retry_body = _rewrite_reasoning_effort(body, accepted)
        if retry_body is None:
            return None

        self._model_reasoning_vocabularies[model] = accepted
        action = (
            "omitting reasoning_effort"
            if _REASONING_FIELD not in retry_body
            else "adjusting reasoning_effort"
        )
        logger.warning("GROQ_STREAM: {} after upstream rejection", action)
        return retry_body

    def retry_after_standard_corrections(
        self,
        error: Exception,
        body: JsonObject,
        used_retry_kinds: set[str],
    ) -> JsonObject | None:
        if "groq_tpm" in used_retry_kinds:
            return None

        correction = correct_tpm_completion_budget(error, body)
        if correction is None:
            return None
        used_retry_kinds.add("groq_tpm")
        logger.warning(
            "GROQ_STREAM: TPM limit={} requested={}; retrying "
            "max_completion_tokens {} -> {}",
            correction.limit,
            correction.requested,
            correction.previous_max_completion_tokens,
            correction.corrected_max_completion_tokens,
        )
        return correction.body


class GroqProvider(OpenAIChatProvider):
    """Groq API with model-agnostic reasoning-vocabulary learning."""

    def __init__(
        self, config: ProviderConfig, *, admission: ProviderAdmissionController
    ) -> None:
        super().__init__(config, behavior=GroqChatBehavior(), admission=admission)


def _parse_reasoning_vocabulary(error: Exception) -> frozenset[str] | None:
    """Return Groq's accepted known values, or an understood empty vocabulary."""
    if not _is_bad_request(error):
        return None

    candidates = _error_candidates(error)
    for candidate in candidates:
        for marker in _ALLOWED_MARKER.finditer(candidate.text):
            if not _marker_targets_reasoning(candidate, marker):
                continue
            tail = candidate.text[marker.end() : marker.end() + 512]
            clause = _CLAUSE_END.split(tail, maxsplit=1)[0]
            return frozenset(
                match.group(1).lower()
                for match in _VALUE_PATTERN.finditer(clause)
                if match.group(1).lower() in _KNOWN_VALUES
            )

    for candidate in candidates:
        for phrase in _UNSUPPORTED_PHRASE.finditer(candidate.text):
            if _marker_targets_reasoning(candidate, phrase):
                return frozenset()

    return None


def _rewrite_reasoning_effort(
    body: dict[str, Any], accepted: frozenset[str]
) -> dict[str, Any] | None:
    """Clone ``body`` with a compatible known effort, or omit the field."""
    current = body.get(_REASONING_FIELD)
    if not isinstance(current, str):
        return None

    normalized = current.lower()
    if normalized in accepted:
        return None

    rewritten = dict(body)
    if normalized in {"low", "medium", "high"} and "default" in accepted:
        rewritten[_REASONING_FIELD] = "default"
    else:
        rewritten.pop(_REASONING_FIELD, None)
    return rewritten


def _is_bad_request(error: Exception) -> bool:
    if isinstance(error, openai.BadRequestError):
        return True
    if getattr(error, "status_code", None) == 400:
        return True
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) == 400


def _error_candidates(error: Exception) -> tuple[_ErrorCandidate, ...]:
    body = getattr(error, "body", None)
    if body is not None:
        return _payload_error_candidates(body)

    response = getattr(error, "response", None)
    try:
        response_text = getattr(response, "text", None)
    except Exception:
        response_text = None
    if isinstance(response_text, str) and response_text.strip():
        return _payload_error_candidates(response_text)

    try:
        error_text = str(error)
    except Exception:
        error_text = ""
    return _payload_error_candidates(error_text)


def _payload_error_candidates(value: Any) -> tuple[_ErrorCandidate, ...]:
    if isinstance(value, str):
        parsed = _parse_json_container(value)
        if parsed is not None:
            value = parsed
    return _deduplicate_candidates(_body_error_candidates(value))


def _deduplicate_candidates(
    candidates: Iterator[_ErrorCandidate],
) -> tuple[_ErrorCandidate, ...]:
    result: list[_ErrorCandidate] = []
    positions: dict[str, int] = {}
    for candidate in candidates:
        text = candidate.text.strip()
        if not text:
            continue
        normalized = _ErrorCandidate(text, candidate.field_context)
        position = positions.get(text)
        if position is None:
            positions[text] = len(result)
            result.append(normalized)
        elif candidate.field_context and not result[position].field_context:
            result[position] = normalized
    return tuple(result)


def _parse_json_container(text: str) -> Mapping[Any, Any] | list[Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, (Mapping, list)) else None


def _body_error_candidates(value: Any) -> Iterator[_ErrorCandidate]:
    if isinstance(value, str):
        yield _ErrorCandidate(value)
        return

    if isinstance(value, Mapping):
        local_context = _mapping_names_reasoning_field(value)
        for key, child in value.items():
            if str(key).lower() in {"message", "detail"} and isinstance(child, str):
                yield _ErrorCandidate(child, local_context)
        for child in value.values():
            if isinstance(child, (Mapping, list, tuple)):
                yield from _body_error_candidates(child)
        return

    if isinstance(value, (list, tuple)):
        for child in value:
            yield from _body_error_candidates(child)


def _mapping_names_reasoning_field(value: Mapping[Any, Any]) -> bool:
    for key, parameter in value.items():
        if (
            str(key).lower() in {"param", "parameter"}
            and isinstance(parameter, str)
            and parameter.strip().lower() == _REASONING_FIELD
        ):
            return True
    return False


def _marker_targets_reasoning(
    candidate: _ErrorCandidate, marker: re.Match[str]
) -> bool:
    if candidate.field_context:
        return True
    for field in _FIELD_PATTERN.finditer(candidate.text):
        start = min(field.start(), marker.start())
        end = max(field.end(), marker.end())
        between = candidate.text[start:end]
        if len(between) <= 128 and _CLAUSE_END.search(between) is None:
            return True
    return False

"""NVIDIA NIM provider implementation."""

import json
import re
from collections.abc import Mapping
from typing import Any, override

import openai

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import NVIDIA_NIM_DEFAULT_BASE
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.canonical_failures import (
    context_window_exceeded_provider_failure,
    overloaded_provider_failure,
)
from free_claude_code.providers.history_negotiation import HistoryModeCache
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
)
from free_claude_code.providers.request_correction import CorrectionStep, Proposal, Rejection

from .models import HOSTED_MODEL_EXCLUSIONS
from .request_options import (
    NIM_REQUEST_POLICY,
    apply_nim_request_options,
    build_nim_request_body,
)
from .retry import (
    clone_body_without_chat_template,
    clone_body_without_reasoning_effort,
)
from .tool_schema import (
    body_without_nim_tool_argument_aliases,
    nim_tool_argument_aliases_from_body,
)

_DEGRADED_FUNCTION_STATE = "degraded function cannot be invoked"
_NEGATIVE_MAX_TOKENS_PATTERN = re.compile(
    r"\bmax_tokens must be at least 1,\s*got\s+-[1-9]\d*\b",
    re.IGNORECASE,
)
_PROFILE = OpenAIChatProfile(
    NIM_REQUEST_POLICY,
    NO_REASONING,
)


class NvidiaNimChatBehavior(OpenAIChatBehavior):
    """NVIDIA NIM Chat adaptation without HTTP ownership."""

    def __init__(self) -> None:
        super().__init__(_PROFILE)

    @property
    @override
    def reasoning_off_fields(self) -> tuple[tuple[str, ...], ...]:
        return (("reasoning_effort",),)

    @override
    def build_messages_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict[str, Any]:
        """Internal helper for tests and shared building."""
        return build_nim_request_body(
            request,
            reasoning=reasoning,
        )

    @override
    def finalize_chat_body(
        self,
        body: dict[str, Any],
        *,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        """Apply NIM policy after either client-protocol translation."""
        apply_nim_request_options(body, reasoning)
        return body

    @override
    def prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Strip private request metadata before calling NVIDIA NIM."""
        return body_without_nim_tool_argument_aliases(body)

    @override
    def tool_argument_aliases(self, body: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Return NIM tool argument aliases captured while building this request."""
        return nim_tool_argument_aliases_from_body(body)

    @override
    def correction_steps(self) -> tuple[CorrectionStep, ...]:
        return (
            CorrectionStep("nim_chat_template", self._retry_chat_template),
            CorrectionStep("nim_reasoning_effort", self._retry_reasoning_effort),
        )

    def _retry_chat_template(self, rejection: Rejection) -> Proposal | None:
        if not _is_bad_request(rejection.error):
            return None
        error_text = _error_text(rejection.error)
        if "chat_template" not in error_text:
            return None
        retry_body = clone_body_without_chat_template(rejection.body)
        if retry_body is None:
            return None
        return Proposal(retry_body, "dropped chat_template")

    def _retry_reasoning_effort(self, rejection: Rejection) -> Proposal | None:
        if not _is_bad_request(rejection.error):
            return None
        error_text = _error_text(rejection.error)
        if not _is_nim_reasoning_rejection(rejection.error, error_text=error_text):
            return None
        retry_body = clone_body_without_reasoning_effort(rejection.body)
        if retry_body is None:
            return None
        return Proposal(retry_body, "dropped reasoning_effort")

    @override
    def retry_request_body(
        self, error: Exception, body: dict[str, Any]
    ) -> Proposal | None:
        """Retry once with a downgraded body when NIM rejects a known field."""
        if not _is_bad_request(error):
            return None

        error_text = _error_text(error)

        if "chat_template" in error_text:
            retry_body = clone_body_without_chat_template(body)
            if retry_body is not None:
                return Proposal(retry_body, "dropped chat_template")

        if _is_nim_reasoning_rejection(error, error_text=error_text):
            retry_body = clone_body_without_reasoning_effort(body)
            if retry_body is not None:
                return Proposal(retry_body, "dropped reasoning_effort")

        return None

    @override
    def reasoning_disable_rejected(self, error: Exception) -> bool:
        """Recognize NIM rejection when disabling reasoning."""
        if not _is_bad_request(error):
            return False
        error_text = _error_text(error)
        if _identifies_other_rejected_field(error, error_text):
            return False
        return _is_nim_reasoning_rejection(error, error_text=error_text)

    @override
    def failure_override(self, error: Exception) -> ExecutionFailure | None:
        """Classify NVIDIA-specific 400/500 responses by their actual semantics."""
        if not isinstance(error, openai.BadRequestError | openai.InternalServerError):
            return None
        if error.status_code not in (400, 500):
            return None
        bodies = _nim_error_bodies(error)
        if any(_is_context_window_exhaustion(body) for body in bodies):
            return context_window_exceeded_provider_failure()
        if isinstance(error, openai.BadRequestError) and any(
            _is_degraded_function(body) for body in bodies
        ):
            return overloaded_provider_failure()
        return None


class NvidiaNimProvider(OpenAIChatProvider):
    """NVIDIA NIM provider using official OpenAI client."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        admission: ProviderAdmissionController,
        history_cache: HistoryModeCache | None = None,
    ):
        super().__init__(
            config,
            behavior=NvidiaNimChatBehavior(),
            admission=admission,
            history_cache=history_cache,
        )

    @override
    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Filter known unavailable and non-chat models from hosted discovery."""
        infos = await super().list_model_infos()
        if self._base_url != NVIDIA_NIM_DEFAULT_BASE:
            return infos
        return frozenset(
            info for info in infos if info.model_id not in HOSTED_MODEL_EXCLUSIONS
        )


def _nim_error_bodies(error: Exception) -> tuple[Mapping[str, Any], ...]:
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        return ()
    nested = body.get("error")
    if isinstance(nested, Mapping):
        return body, nested
    return (body,)


def _is_context_window_exhaustion(body: Mapping[str, Any]) -> bool:
    message = body.get("message")
    return (
        body.get("param") == "max_tokens"
        and isinstance(message, str)
        and _NEGATIVE_MAX_TOKENS_PATTERN.search(message) is not None
    )


def _is_degraded_function(body: Mapping[str, Any]) -> bool:
    detail = body.get("detail")
    if not isinstance(detail, str):
        return False
    function_ref, separator, state = detail.lower().partition(": ")
    function_id = function_ref.removeprefix("function id ").strip(" '\"")
    return bool(
        separator
        and function_ref.startswith("function id ")
        and function_id
        and state.strip() == _DEGRADED_FUNCTION_STATE
    )


def _is_bad_request(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    return isinstance(error, openai.BadRequestError) or status_code == 400


def _error_text(error: Exception) -> str:
    error_text = str(error)
    error_body = getattr(error, "body", None)
    if error_body is not None:
        error_text = f"{error_text} {json.dumps(error_body, default=str)}"
    return error_text.lower()


def _identifies_other_rejected_field(error: Exception, error_text: str) -> bool:
    for body in _nim_error_bodies(error):
        param = body.get("param")
        if isinstance(param, str) and param and param != "reasoning_effort":
            return True
    return "chat_template" in error_text


def _is_nim_reasoning_rejection(
    error: Exception | str, *, error_text: str | None = None
) -> bool:
    """Recognize NIM rejection of reasoning_effort by message or named field."""
    if isinstance(error, str):
        text = error.lower()
        if "must be one of" in text or "chat_template" in text:
            return False
        return "reasoning_effort" in text or "invalid request" in text

    bodies = _nim_error_bodies(error)
    for body in bodies:
        param = body.get("param")
        if param == "reasoning_effort":
            message = str(body.get("message", "")).lower()
            return "must be one of" not in message
        if isinstance(param, str) and param:
            return False

    text = error_text if error_text is not None else _error_text(error)
    if "must be one of" in text or "chat_template" in text:
        return False
    if "reasoning_effort" in text:
        return True

    for body in bodies:
        message = body.get("message")
        if isinstance(message, str) and message.strip().lower() == "invalid request":
            return True
    return False

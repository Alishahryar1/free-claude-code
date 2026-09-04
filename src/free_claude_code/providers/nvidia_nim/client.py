"""NVIDIA NIM provider implementation."""

import json
import re
from collections.abc import Mapping
from typing import Any

import openai
from loguru import logger

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.failure_policy import (
    context_window_exceeded_provider_failure,
    overloaded_provider_failure,
)
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
)

from .native_tool_stream import normalize_nim_native_tool_stream
from .request_options import (
    NIM_REQUEST_POLICY,
    apply_nim_request_options,
    build_nim_request_body,
)
from .retry import (
    clone_body_without_chat_template,
    clone_body_without_reasoning_budget,
    clone_body_without_reasoning_content,
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


class NvidiaNimProvider(OpenAIChatProvider):
    """NVIDIA NIM provider using official OpenAI client."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        nim_settings: NimSettings,
        admission: ProviderAdmissionController,
    ):
        super().__init__(
            config,
            profile=_PROFILE,
            admission=admission,
        )
        self._nim_settings = nim_settings

    def _build_request_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict:
        """Internal helper for tests and shared building."""
        return build_nim_request_body(
            request,
            self._nim_settings,
            reasoning=reasoning,
        )

    def _finalize_chat_body(
        self,
        body: dict[str, Any],
        *,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        """Apply NIM policy after either client-protocol translation."""
        apply_nim_request_options(body, reasoning, nim=self._nim_settings)
        return body

    def _prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Strip private request metadata before calling NVIDIA NIM."""
        return body_without_nim_tool_argument_aliases(body)

    def _normalize_stream(self, stream: Any, _body: Mapping[str, Any]) -> Any:
        """Repair model-native MiniMax tool markup leaked by NVIDIA NIM."""
        return normalize_nim_native_tool_stream(stream, _body)

    def _tool_argument_aliases(self, body: dict[str, Any]) -> dict[str, dict[str, str]]:
        """Return NIM tool argument aliases captured while building this request."""
        return nim_tool_argument_aliases_from_body(body)

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry once with a downgraded body when NIM rejects a known field."""
        status_code = getattr(error, "status_code", None)
        bad_request_like = isinstance(error, openai.BadRequestError) or (
            status_code == 400
        )

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        error_text = error_text.lower()

        if _is_reasoning_budget_rejection(error_text) and (
            bad_request_like or status_code == 500
        ):
            retry_body = clone_body_without_reasoning_budget(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning budget after upstream rejection"
            )
            return retry_body

        if not bad_request_like:
            return None

        if "chat_template" in error_text:
            retry_body = clone_body_without_chat_template(body)
            if retry_body is None:
                return None
            logger.warning("NIM_STREAM: retrying without chat_template after 400 error")
            return retry_body

        if "reasoning_content" in error_text:
            retry_body = clone_body_without_reasoning_content(body)
            if retry_body is None:
                return None
            logger.warning(
                "NIM_STREAM: retrying without reasoning_content after 400 error"
            )
            return retry_body

        return None

    def _provider_failure_override(self, error: Exception) -> ExecutionFailure | None:
        """Classify NVIDIA-specific 400/500 responses by their actual semantics."""
        if isinstance(error, openai.APIConnectionError | openai.APITimeoutError):
            return overloaded_provider_failure()
        if not isinstance(error, openai.BadRequestError | openai.InternalServerError):
            return None
        status = getattr(error, "status_code", None)
        if status not in (400, 500):
            return None
        bodies = _nim_error_bodies(error)
        if any(_is_context_window_exhaustion(body) for body in bodies):
            return context_window_exceeded_provider_failure()
        if isinstance(error, openai.BadRequestError) and any(
            _is_degraded_function(body) for body in bodies
        ):
            return overloaded_provider_failure()
        return None


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


def _is_reasoning_budget_rejection(error_text: str) -> bool:
    """Return whether NIM rejected optional thinking budget control."""
    if "reasoning_budget" in error_text:
        return True
    return "thinking_token_budget" in error_text and "reasoning_config" in error_text
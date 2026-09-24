"""Native Google Antigravity provider for FCC."""

from typing import cast

from openai import AsyncOpenAI

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningEffort
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    NamedEffortReasoning,
    OpenAIChatBehavior,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)

from .auth import AntigravityAuthManager
from .chat_adapter import AntigravityChatAdapter
from .client import AntigravityClient, AntigravityUpstreamError

_ANTIGRAVITY_EFFORTS = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "medium"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "high"),
    (ReasoningEffort.MAX, "high"),
)

ANTIGRAVITY_PROFILE = OpenAIChatProfile(
    request_policy=OpenAIChatRequestPolicy(
        provider_name="ANTIGRAVITY",
        reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    ),
    reasoning=NamedEffortReasoning(
        efforts=_ANTIGRAVITY_EFFORTS,
        disabled_value="off",
        field="reasoning_effort",
    ),
    reasoning_delta_field="reasoning_content",
    reasoning_delta_fallback_field="reasoning",
)


class AntigravityBehavior(OpenAIChatBehavior):
    """Keep Cloud Code HTTP status semantics visible to FCC's failure mapper."""

    def failure_override(self, error: Exception) -> ExecutionFailure | None:
        if not isinstance(error, AntigravityUpstreamError):
            return None

        status = error.status_code
        if status == 400:
            return ExecutionFailure(
                kind=FailureKind.INVALID_REQUEST,
                status_code=400,
                message="Invalid request sent to provider.",
                retryable=False,
            )
        if status == 401:
            return ExecutionFailure(
                kind=FailureKind.AUTHENTICATION,
                status_code=401,
                message="Provider authentication failed. Check API key.",
                retryable=False,
            )
        if status == 402:
            return ExecutionFailure(
                kind=FailureKind.PERMISSION,
                status_code=402,
                message=(
                    "Provider requires payment or additional credits. "
                    "Add credits or resolve billing."
                ),
                retryable=False,
            )
        if status == 403:
            return ExecutionFailure(
                kind=FailureKind.PERMISSION,
                status_code=403,
                message=(
                    "Provider denied access. Check credential permissions and "
                    "model access."
                ),
                retryable=False,
            )
        if status == 413:
            return ExecutionFailure(
                kind=FailureKind.INVALID_REQUEST,
                status_code=413,
                message="Provider rejected the request as too large.",
                retryable=False,
            )
        if status == 429:
            return ExecutionFailure(
                kind=FailureKind.RATE_LIMIT,
                status_code=429,
                message="Provider rate limit reached. Please retry shortly.",
                retryable=True,
            )
        if status in {502, 503, 504}:
            return ExecutionFailure(
                kind=FailureKind.OVERLOADED,
                status_code=529,
                message="Provider is currently overloaded. Please retry.",
                retryable=True,
            )
        if 500 <= status <= 599:
            return ExecutionFailure(
                kind=FailureKind.UPSTREAM,
                status_code=status,
                message="Provider API request failed.",
                retryable=True,
            )
        return ExecutionFailure(
            kind=FailureKind.UPSTREAM,
            status_code=status,
            message="Provider API request failed.",
            retryable=False,
        )


class AntigravityProvider(OpenAIChatProvider):
    """Reuse FCC's mature chat conversion around a Cloud Code-backed adapter."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        auth: AntigravityAuthManager,
        admission: ProviderAdmissionController,
    ) -> None:
        cloud_code = AntigravityClient(
            auth=auth,
            base_url=config.base_url,
            read_timeout=config.http_read_timeout,
            write_timeout=config.http_write_timeout,
            connect_timeout=config.http_connect_timeout,
            proxy=config.proxy,
        )
        adapter = AntigravityChatAdapter(cloud_code, base_url=config.base_url)
        self._cloud_code = cloud_code
        super().__init__(
            config,
            behavior=AntigravityBehavior(ANTIGRAVITY_PROFILE),
            admission=admission,
            client=cast(AsyncOpenAI, adapter),
        )

    async def cleanup(self) -> None:
        await self._cloud_code.close()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        payload = await self._cloud_code.fetch_available_models()
        models = payload.get("models")
        if not isinstance(models, dict):
            return frozenset()
        return frozenset(
            ProviderModelInfo(model_id=model_id)
            for model_id in models
            if isinstance(model_id, str) and model_id
        )

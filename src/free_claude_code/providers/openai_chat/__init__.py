"""OpenAI-compatible provider family."""

from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.history_negotiation import HistoryModeCache

from .base_url import openai_v1_base_url
from .behavior import OpenAIChatBehavior
from .client import OpenAIAsyncCredentialProvider, create_chat_client
from .extra_body import (
    validate_extra_body_does_not_override_canonical_fields,
    validate_extra_body_does_not_override_reasoning_fields,
)
from .native_output import NativeChatStreamOutput
from .profiles import OPENAI_CHAT_PROFILES, OpenAIChatProfile, OpenAIModelListing
from .provider import OpenAIChatProvider
from .reasoning import (
    NO_REASONING,
    ChatTemplateReasoning,
    NamedEffortReasoning,
    ReasoningEncoder,
    ReasoningObject,
    ThinkingObjectReasoning,
)
from .request_policy import (
    OpenAIChatRequestPolicy,
    apply_openai_chat_body_policy,
    build_openai_chat_request_body,
)
from .stream_output import (
    AnthropicChatStreamOutput,
    ChatStreamOutput,
    ChatStreamUsage,
    ResponsesChatStreamOutput,
)
from .transport import OpenAIChatTransport
from .usage import usage_int


def create_openai_chat_provider(
    provider_id: str,
    config: ProviderConfig,
    admission: ProviderAdmissionController,
    history_cache: HistoryModeCache | None = None,
) -> OpenAIChatProvider:
    """Construct one profile-driven provider."""
    profile = OPENAI_CHAT_PROFILES.get(provider_id)
    if profile is None:
        raise KeyError(f"No declarative OpenAI-chat profile for {provider_id!r}")
    return OpenAIChatProvider(
        config,
        profile=profile,
        admission=admission,
        history_cache=history_cache,
        default_headers=(
            {"User-Agent": profile.user_agent} if profile.user_agent else None
        ),
    )


__all__ = [
    "NO_REASONING",
    "OPENAI_CHAT_PROFILES",
    "AnthropicChatStreamOutput",
    "ChatStreamOutput",
    "ChatStreamUsage",
    "ChatTemplateReasoning",
    "NamedEffortReasoning",
    "NativeChatStreamOutput",
    "OpenAIAsyncCredentialProvider",
    "OpenAIChatBehavior",
    "OpenAIChatProfile",
    "OpenAIChatProvider",
    "OpenAIChatRequestPolicy",
    "OpenAIChatTransport",
    "OpenAIModelListing",
    "ReasoningEncoder",
    "ReasoningObject",
    "ResponsesChatStreamOutput",
    "ThinkingObjectReasoning",
    "apply_openai_chat_body_policy",
    "build_openai_chat_request_body",
    "create_chat_client",
    "create_openai_chat_provider",
    "openai_v1_base_url",
    "usage_int",
    "validate_extra_body_does_not_override_canonical_fields",
    "validate_extra_body_does_not_override_reasoning_fields",
]

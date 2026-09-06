"""Declarative profiles for ordinary OpenAI-compatible providers."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import (
    InputModalityBooleanPaths,
    ModelTokenLimitResolver,
    RequiredPathValues,
    live_provider_context_window_consensus,
)

from .base_url import openai_v1_base_url
from .extra_body import (
    validate_extra_body_does_not_override_canonical_fields,
    validate_extra_body_does_not_override_reasoning_fields,
)
from .reasoning import (
    LLAMACPP_REASONING,
    NO_REASONING,
    SPLIT_REASONING_OUTPUT,
    ChatTemplateReasoning,
    NamedEffortReasoning,
    ReasoningEncoder,
    ReasoningObject,
    ThinkingObjectReasoning,
)
from .reasoning_details import apply_reasoning_details_replay
from .request_policy import OpenAIChatPostprocessor, OpenAIChatRequestPolicy

_ALL_EFFORTS = tuple((effort, effort.value) for effort in ReasoningEffort)
_LOW_MEDIUM_HIGH = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "medium"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "high"),
    (ReasoningEffort.MAX, "high"),
)
_MINIMAL_TO_XHIGH = (
    (ReasoningEffort.MINIMAL, "minimal"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "medium"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "xhigh"),
    (ReasoningEffort.MAX, "xhigh"),
)
_LOW_TO_MAX = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "medium"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "max"),
    (ReasoningEffort.MAX, "max"),
)
_KIMI_CODE_EFFORTS = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "high"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "max"),
    (ReasoningEffort.MAX, "max"),
)
_TEXT_INPUT_MODALITIES = frozenset({ModelInputModality.TEXT})


@dataclass(frozen=True, slots=True)
class OpenAIModelPagination:
    """Bounded numbered-pagination metadata for a model-list endpoint."""

    page_param: str = "page"
    first_page: int = 1
    current_page_path: tuple[str, ...] = ("pagination", "current_page")
    total_pages_path: tuple[str, ...] = ("pagination", "total_pages")
    max_pages: int = 100


@dataclass(frozen=True, slots=True)
class OpenAIModelListing:
    """Declarative model-list endpoint and response shape."""

    path: str | None = None
    query_params: tuple[tuple[str, str], ...] = ()
    collection_field: str | None = "data"
    id_field: str = "id"
    aliases_field: str | None = None
    additional_model_ids: tuple[str, ...] = ()
    required_path_values: RequiredPathValues = ()
    required_null_field: str | None = None
    required_sequence_items: tuple[tuple[str, str], ...] = ()
    exclude_missing_sequence_fields: bool = False
    tags_field: str | None = None
    thinking_tag: str = "reasoning"
    non_thinking_tag: str | None = None
    thinking_boolean_path: tuple[str, ...] | None = None
    input_modalities_path: tuple[str, ...] | None = None
    thinking_sequence_path: tuple[str, ...] | None = None
    fixed_input_modalities: frozenset[ModelInputModality] | None = None
    input_modality_boolean_paths: InputModalityBooleanPaths = ()
    context_window_tokens_path: tuple[str, ...] | None = None
    max_output_tokens_path: tuple[str, ...] | None = None
    context_window_tokens_resolver: ModelTokenLimitResolver | None = None
    pagination: OpenAIModelPagination | None = None


@dataclass(frozen=True, slots=True)
class OpenAIChatProfile:
    """Immutable transport and reasoning behavior for one provider."""

    request_policy: OpenAIChatRequestPolicy
    reasoning: ReasoningEncoder
    postprocessors: tuple[OpenAIChatPostprocessor, ...] = ()
    model_ids_are_routable: bool = True
    model_listing: OpenAIModelListing = OpenAIModelListing()
    normalize_base_url: bool = False
    reasoning_delta_field: Literal["reasoning_content", "reasoning"] = (
        "reasoning_content"
    )
    reasoning_delta_fallback_field: Literal["reasoning_content", "reasoning"] | None = (
        None
    )
    structured_reasoning_details: bool = False
    user_agent: str | None = None

    @property
    def provider_name(self) -> str:
        return self.request_policy.provider_name

    def base_url(self, configured: str) -> str:
        return openai_v1_base_url(configured) if self.normalize_base_url else configured

    def reasoning_delta(self, delta: Any) -> str | None:
        value = getattr(delta, self.reasoning_delta_field, None)
        if isinstance(value, str) and value:
            return value
        fallback = self.reasoning_delta_fallback_field
        if fallback is None:
            return value if isinstance(value, str) else None
        fallback_value = getattr(delta, fallback, None)
        if isinstance(fallback_value, str):
            return fallback_value
        return value if isinstance(value, str) else None

    def apply_reasoning(
        self,
        body: dict[str, Any],
        _request: MessagesRequest,
        policy: ReasoningPolicy,
    ) -> None:
        self.apply_reasoning_to_body(body, policy)

    def apply_reasoning_to_body(
        self,
        body: dict[str, Any],
        policy: ReasoningPolicy,
    ) -> None:
        """Encode resolved reasoning policy after either client translation."""
        self.reasoning.encode(body, policy)

    @property
    def request_postprocessors(self) -> tuple[OpenAIChatPostprocessor, ...]:
        return (*self.postprocessors, self.apply_reasoning)


def _apply_cohere_request_quirks(
    body: dict[str, Any], request: MessagesRequest, _policy: ReasoningPolicy
) -> None:
    _merge_allowed_cohere_extra_body(body, request.extra_body)


_COHERE_EXTRA_BODY_KEYS = frozenset(
    {
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "seed",
    }
)


def _merge_allowed_cohere_extra_body(body: dict[str, Any], extra_body: Any) -> None:
    if extra_body in (None, {}):
        return
    if not isinstance(extra_body, Mapping):
        raise InvalidRequestError("Cohere extra_body must be an object when provided.")

    unsupported = sorted(
        str(key) for key in extra_body if key not in _COHERE_EXTRA_BODY_KEYS
    )
    if unsupported:
        raise InvalidRequestError(
            "Cohere extra_body supports only these keys: "
            f"{sorted(_COHERE_EXTRA_BODY_KEYS)}. Unsupported: {unsupported}"
        )
    body.update({str(key): deepcopy(value) for key, value in extra_body.items()})


def _policy(
    provider_name: str,
    replay: ReasoningReplayMode,
    **kwargs: Any,
) -> OpenAIChatRequestPolicy:
    return OpenAIChatRequestPolicy(
        provider_name=provider_name,
        reasoning_replay=replay,
        **kwargs,
    )


def _zai_profile(provider_name: str) -> OpenAIChatProfile:
    return OpenAIChatProfile(
        _policy(
            provider_name,
            ReasoningReplayMode.REASONING_CONTENT,
            reject_extra_body_message=(
                "Z.ai Chat Completions API does not support caller extra_body on "
                "requests."
            ),
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        ThinkingObjectReasoning(
            enabled={"type": "enabled", "clear_thinking": False},
            disabled={"type": "disabled"},
        ),
    )


OPENAI_CHAT_PROFILES: dict[str, OpenAIChatProfile] = {
    "cline_pass": OpenAIChatProfile(
        _policy("CLINE_PASS", ReasoningReplayMode.DISABLED),
        NO_REASONING,
        postprocessors=(apply_reasoning_details_replay,),
        model_listing=OpenAIModelListing(
            path="/ai/cline/recommended-models",
            collection_field="clinePass",
        ),
        reasoning_delta_field="reasoning",
        structured_reasoning_details=True,
    ),
    "xai": OpenAIChatProfile(
        _policy(
            "XAI",
            ReasoningReplayMode.REASONING_CONTENT,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NO_REASONING,
        model_listing=OpenAIModelListing(
            path="/language-models",
            collection_field="models",
            aliases_field="aliases",
            input_modalities_path=("input_modalities",),
        ),
    ),
    "qwencloud": OpenAIChatProfile(
        _policy(
            "QWENCLOUD",
            ReasoningReplayMode.REASONING_CONTENT,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NO_REASONING,
    ),
    "qwencloud_coding": OpenAIChatProfile(
        _policy(
            "QWENCLOUD_CODING",
            ReasoningReplayMode.REASONING_CONTENT,
        ),
        NO_REASONING,
    ),
    "together": OpenAIChatProfile(
        _policy(
            "TOGETHER",
            ReasoningReplayMode.REASONING,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NO_REASONING,
        model_listing=OpenAIModelListing(
            path="/models",
            collection_field=None,
            required_path_values=((("type",), ("chat",)),),
            context_window_tokens_path=("context_length",),
        ),
        reasoning_delta_field="reasoning",
    ),
    "deepinfra": OpenAIChatProfile(
        _policy(
            "DEEPINFRA",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            _ALL_EFFORTS,
            disabled_value="none",
            enabled_value="high",
            use_extra_body=True,
        ),
        model_listing=OpenAIModelListing(
            path="https://api.deepinfra.com/models/list",
            collection_field=None,
            id_field="model_name",
            required_path_values=((("reported_type",), ("text-generation",)),),
            required_null_field="deprecated",
            tags_field="tags",
            non_thinking_tag="non-reasoning",
            context_window_tokens_path=("max_tokens",),
        ),
    ),
    "siliconflow": OpenAIChatProfile(
        _policy(
            "SILICONFLOW",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NO_REASONING,
        model_listing=OpenAIModelListing(
            path="/models",
            query_params=(("sub_type", "chat"),),
        ),
    ),
    "nebius": OpenAIChatProfile(
        _policy(
            "NEBIUS",
            ReasoningReplayMode.REASONING_CONTENT,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            _MINIMAL_TO_XHIGH,
            disabled_value="none",
            enabled_value="xhigh",
        ),
        model_listing=OpenAIModelListing(
            path="/models",
            query_params=(("verbose", "true"),),
            required_path_values=((("architecture", "modality"), ("text->text",)),),
            fixed_input_modalities=_TEXT_INPUT_MODALITIES,
            context_window_tokens_path=("context_length",),
        ),
    ),
    "chutes": OpenAIChatProfile(
        _policy(
            "CHUTES",
            ReasoningReplayMode.DISABLED,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NO_REASONING,
        model_listing=OpenAIModelListing(
            path="/models",
            required_sequence_items=(
                ("input_modalities", "text"),
                ("output_modalities", "text"),
                ("supported_features", "tools"),
            ),
            exclude_missing_sequence_fields=True,
            tags_field="supported_features",
            input_modalities_path=("input_modalities",),
            context_window_tokens_path=("context_length",),
            max_output_tokens_path=("max_output_length",),
        ),
    ),
    "featherless": OpenAIChatProfile(
        _policy(
            "FEATHERLESS",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        ChatTemplateReasoning(field="enable_thinking"),
        model_listing=OpenAIModelListing(
            path="/models",
            query_params=(
                ("capabilities", "chat,tool-use"),
                ("available_on_current_plan", "true"),
                ("status", "active"),
                ("per_page", "1000"),
            ),
            required_path_values=(
                (("features", "tool_use"), (True,)),
                (("is_gated",), (False,)),
                (("available_on_current_plan",), (True,)),
            ),
            fixed_input_modalities=_TEXT_INPUT_MODALITIES,
            input_modality_boolean_paths=(
                (ModelInputModality.IMAGE, ("features", "image_input")),
            ),
            context_window_tokens_path=("context_length",),
            max_output_tokens_path=("max_completion_tokens",),
            pagination=OpenAIModelPagination(),
        ),
    ),
    "agnes": OpenAIChatProfile(
        _policy(
            "AGNES",
            ReasoningReplayMode.THINK_TAGS,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        ChatTemplateReasoning(field="enable_thinking"),
    ),
    "zenmux": OpenAIChatProfile(
        _policy(
            "ZENMUX",
            ReasoningReplayMode.REASONING,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            max_tokens_field="max_completion_tokens",
        ),
        ReasoningObject(_MINIMAL_TO_XHIGH),
        postprocessors=(apply_reasoning_details_replay,),
        model_listing=OpenAIModelListing(
            required_sequence_items=(
                ("input_modalities", "text"),
                ("output_modalities", "text"),
            ),
            thinking_boolean_path=("capabilities", "reasoning"),
            input_modalities_path=("input_modalities",),
            context_window_tokens_path=("context_length",),
        ),
        reasoning_delta_field="reasoning",
        reasoning_delta_fallback_field="reasoning_content",
        structured_reasoning_details=True,
    ),
    "wandb": OpenAIChatProfile(
        _policy(
            "WANDB",
            ReasoningReplayMode.DISABLED,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            max_tokens_field="max_completion_tokens",
        ),
        ChatTemplateReasoning(field="enable_thinking"),
        reasoning_delta_field="reasoning",
    ),
    "azure_openai": OpenAIChatProfile(
        _policy(
            "AZURE_OPENAI",
            ReasoningReplayMode.THINK_TAGS,
            max_tokens_field="max_completion_tokens",
        ),
        NamedEffortReasoning(
            _LOW_MEDIUM_HIGH,
            disabled_value="none",
            enabled_value="medium",
        ),
        model_ids_are_routable=False,
    ),
    "mistral_codestral": OpenAIChatProfile(
        _policy("CODESTRAL", ReasoningReplayMode.THINK_TAGS),
        NO_REASONING,
        model_listing=OpenAIModelListing(
            input_modality_boolean_paths=(
                (
                    ModelInputModality.TEXT,
                    ("capabilities", "completion_chat"),
                ),
                (ModelInputModality.IMAGE, ("capabilities", "vision")),
            ),
            context_window_tokens_path=("max_context_length",),
        ),
    ),
    "vercel": OpenAIChatProfile(
        _policy(
            "VERCEL",
            ReasoningReplayMode.THINK_TAGS,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
        ),
        ReasoningObject(_ALL_EFFORTS),
        model_listing=OpenAIModelListing(
            input_modalities_path=("modalities", "input"),
            thinking_sequence_path=("supported_parameters",),
            context_window_tokens_path=("context_window",),
            max_output_tokens_path=("max_tokens",),
        ),
    ),
    "bedrock": OpenAIChatProfile(
        _policy("BEDROCK", ReasoningReplayMode.THINK_TAGS),
        NO_REASONING,
        normalize_base_url=True,
    ),
    "huggingface": OpenAIChatProfile(
        _policy(
            "HUGGINGFACE",
            ReasoningReplayMode.DISABLED,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
        ),
        NO_REASONING,
        model_listing=OpenAIModelListing(
            input_modalities_path=("architecture", "input_modalities"),
            context_window_tokens_resolver=live_provider_context_window_consensus,
        ),
    ),
    "cohere": OpenAIChatProfile(
        _policy(
            "COHERE",
            ReasoningReplayMode.REASONING_CONTENT,
            strip_message_names=True,
            unsupported_body_keys=frozenset(
                {
                    "audio",
                    "logit_bias",
                    "metadata",
                    "modalities",
                    "n",
                    "parallel_tool_calls",
                    "prediction",
                    "service_tier",
                    "store",
                    "top_logprobs",
                }
            ),
        ),
        NamedEffortReasoning(
            tuple((effort, "high") for effort in ReasoningEffort),
            disabled_value="none",
            enabled_value="high",
        ),
        postprocessors=(_apply_cohere_request_quirks,),
    ),
    "wafer": OpenAIChatProfile(
        _policy(
            "WAFER",
            ReasoningReplayMode.REASONING_CONTENT,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            _LOW_TO_MAX,
            disabled_value="none",
            enabled_value="high",
        ),
    ),
    "kimi": OpenAIChatProfile(
        _policy(
            "KIMI",
            ReasoningReplayMode.REASONING_CONTENT,
            reject_extra_body_message=(
                "Kimi Chat Completions API does not support caller extra_body on requests."
            ),
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        ThinkingObjectReasoning(
            enabled={"type": "enabled"},
            disabled={"type": "disabled"},
        ),
    ),
    "kimi_code": OpenAIChatProfile(
        _policy(
            "KIMI_CODE",
            ReasoningReplayMode.REASONING_CONTENT,
            reject_extra_body_message=(
                "Kimi Code Chat Completions API does not support caller "
                "extra_body on requests."
            ),
            max_tokens_field="max_completion_tokens",
        ),
        NamedEffortReasoning(
            _KIMI_CODE_EFFORTS,
            disabled_value="none",
            enabled_value="max",
        ),
        user_agent="free-claude-code",
    ),
    "minimax": OpenAIChatProfile(
        _policy(
            "MINIMAX",
            ReasoningReplayMode.REASONING_CONTENT,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            max_tokens_field="max_completion_tokens",
        ),
        SPLIT_REASONING_OUTPUT,
    ),
    "cerebras": OpenAIChatProfile(
        _policy(
            "CEREBRAS",
            ReasoningReplayMode.THINK_TAGS,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            max_tokens_field="max_completion_tokens",
        ),
        NamedEffortReasoning(
            _LOW_MEDIUM_HIGH,
            disabled_value="none",
            enabled_value="medium",
        ),
        reasoning_delta_field="reasoning",
    ),
    "sambanova": OpenAIChatProfile(
        _policy(
            "SAMBANOVA",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
        ),
        NamedEffortReasoning(
            _LOW_MEDIUM_HIGH,
            enabled_value="medium",
        ),
    ),
    "fireworks": OpenAIChatProfile(
        _policy(
            "FIREWORKS",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            (
                (ReasoningEffort.MINIMAL, "low"),
                (ReasoningEffort.LOW, "low"),
                (ReasoningEffort.MEDIUM, "medium"),
                (ReasoningEffort.HIGH, "high"),
                (ReasoningEffort.XHIGH, "xhigh"),
                (ReasoningEffort.MAX, "max"),
            ),
            disabled_value="none",
            enabled_value="high",
            budget_field="reasoning_effort",
        ),
    ),
    "novita": OpenAIChatProfile(
        _policy(
            "NOVITA",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            (),
            disabled_value=False,
            enabled_value=True,
            field="enable_thinking",
            use_extra_body=True,
        ),
    ),
    "zai": _zai_profile("ZAI"),
    "zai_api": _zai_profile("ZAI_API"),
    "nararoute": OpenAIChatProfile(
        _policy(
            "NARAROUTE",
            ReasoningReplayMode.DISABLED,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            _LOW_MEDIUM_HIGH,
            enabled_value="medium",
        ),
        model_listing=OpenAIModelListing(
            thinking_boolean_path=("reasoning",),
        ),
    ),
    "xkiro": OpenAIChatProfile(
        _policy(
            "XKIRO",
            ReasoningReplayMode.REASONING_CONTENT,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        # xKiro translates reasoning_effort per vendor: every model declares the
        # levels it accepts (reasoning_efforts in GET /v1/models) and unsupported
        # values are adjusted downward, never rejected. Omitting the field is
        # not disabling — most models default to reasoning on — so explicit
        # effort keeps client intent and cost control intact.
        NamedEffortReasoning(
            _ALL_EFFORTS,
            disabled_value="none",
            enabled_value="high",
        ),
    ),
    "poolside": OpenAIChatProfile(
        _policy(
            "POOLSIDE",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        ChatTemplateReasoning(field="enable_thinking"),
    ),
    "llm7": OpenAIChatProfile(
        _policy(
            "LLM7",
            ReasoningReplayMode.REASONING_CONTENT,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NO_REASONING,
        model_listing=OpenAIModelListing(
            path="/models",
            additional_model_ids=("default", "fast", "pro"),
            required_path_values=(
                (("model_type",), ("chat",)),
                (("stream",), (True,)),
                (("tools_calling",), (True,)),
            ),
            thinking_boolean_path=("reasoning",),
            input_modalities_path=("modalities", "input"),
            context_window_tokens_path=("context_window", "tokens"),
        ),
    ),
    "ollama_cloud": OpenAIChatProfile(
        _policy(
            "OLLAMA_CLOUD",
            ReasoningReplayMode.REASONING,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            _LOW_TO_MAX,
            disabled_value="none",
            enabled_value="high",
        ),
        reasoning_delta_field="reasoning",
    ),
    "tokenrouter": OpenAIChatProfile(
        _policy(
            "TOKENROUTER",
            ReasoningReplayMode.DISABLED,
        ),
        NO_REASONING,
    ),
    "llamacpp": OpenAIChatProfile(
        _policy(
            "LLAMACPP",
            ReasoningReplayMode.THINK_TAGS,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        LLAMACPP_REASONING,
        normalize_base_url=True,
    ),
    "ollama": OpenAIChatProfile(
        _policy(
            "OLLAMA",
            ReasoningReplayMode.REASONING,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            _LOW_TO_MAX,
            disabled_value="none",
            enabled_value="high",
        ),
        normalize_base_url=True,
        reasoning_delta_field="reasoning",
    ),
}

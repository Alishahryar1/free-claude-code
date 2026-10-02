"""Pure, validated application settings schema."""

from typing import Annotated

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from .constants import HTTP_CONNECT_TIMEOUT_DEFAULT
from .nim import NimSettings
from .provider_catalog import (
    BEDROCK_DEFAULT_BASE,
    NARAROUTE_DEFAULT_BASE,
    SUPPORTED_PROVIDER_IDS,
    TOKENROUTER_DEFAULT_BASE,
)
from .reasoning import ReasoningPreference


def _empty_to_none(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    return value


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
OptionalNonEmptyString = Annotated[
    NonEmptyString | None,
    BeforeValidator(_empty_to_none),
]


class Settings(BaseModel):
    """Validated application settings with no file or process I/O."""

    model_config = ConfigDict(
        validate_default=True,
        populate_by_name=True,
        extra="ignore",
    )

    # ==================== Azure OpenAI ====================
    azure_openai_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="AZURE_OPENAI_API_KEY"
    )
    azure_openai_base_url: OptionalNonEmptyString = Field(
        default=None, validation_alias="AZURE_OPENAI_BASE_URL"
    )

    # ==================== OpenRouter Config ====================
    open_router_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )

    # ==================== Mistral La Plateforme ====================
    mistral_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="MISTRAL_API_KEY"
    )

    # ==================== Mistral Codestral (codestral.mistral.ai) ====================
    codestral_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="CODESTRAL_API_KEY"
    )

    # ==================== DeepSeek Config ====================
    deepseek_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="DEEPSEEK_API_KEY"
    )

    # ==================== Kimi Config ====================
    kimi_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="KIMI_API_KEY"
    )

    # ==================== Kimi Code Subscription ====================
    kimi_code_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="KIMI_CODE_API_KEY"
    )

    # ==================== Wafer Config ====================
    wafer_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="WAFER_API_KEY"
    )

    # ==================== MiniMax Config ====================
    minimax_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="MINIMAX_API_KEY"
    )

    # ==================== OpenCode Zen / OpenCode Go ====================
    # Same key from opencode.ai/auth; Zen uses ``opencode_zen/``, Go uses ``opencode_go/``.
    opencode_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="OPENCODE_API_KEY"
    )

    # ==================== Vercel AI Gateway ====================
    vercel_ai_gateway_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="AI_GATEWAY_API_KEY"
    )

    # ==================== Amazon Bedrock Mantle ====================
    bedrock_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="AWS_BEARER_TOKEN_BEDROCK"
    )
    bedrock_base_url: NonEmptyString = Field(
        default=BEDROCK_DEFAULT_BASE,
        validation_alias="BEDROCK_BASE_URL",
    )

    # ==================== Hugging Face Inference Providers ====================
    huggingface_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="HUGGINGFACE_API_KEY"
    )

    # ==================== Cohere Compatibility API ====================
    cohere_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="COHERE_API_KEY"
    )

    # ==================== GitHub Models ====================
    github_models_token: OptionalNonEmptyString = Field(
        default=None, validation_alias="GITHUB_MODELS_TOKEN"
    )

    # ==================== SambaNova Cloud ====================
    sambanova_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="SAMBANOVA_API_KEY"
    )

    # ==================== Kilo.ai Config ====================
    kilo_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="KILO_API_KEY"
    )

    # ==================== Z.ai Coding Plan / General API ====================
    zai_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="ZAI_API_KEY"
    )

    # ==================== TokenRouter Config ====================
    tokenrouter_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="TOKENROUTER_API_KEY"
    )
    tokenrouter_base_url: NonEmptyString = Field(
        default=TOKENROUTER_DEFAULT_BASE,
        validation_alias="TOKENROUTER_BASE_URL",
    )

    # ==================== NaraRoute Config ====================
    nararoute_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="NARAROUTE_API_KEY"
    )
    nararoute_base_url: NonEmptyString = Field(
        default=NARAROUTE_DEFAULT_BASE,
        validation_alias="NARAROUTE_BASE_URL",
    )

    # ==================== Fireworks AI Config ====================
    fireworks_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="FIREWORKS_API_KEY"
    )

    # ==================== Novita AI Config ====================
    novita_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="NOVITA_API_KEY"
    )

    # ==================== Cloudflare Workers AI Config ====================
    cloudflare_api_token: OptionalNonEmptyString = Field(
        default=None, validation_alias="CLOUDFLARE_API_TOKEN"
    )
    cloudflare_account_id: OptionalNonEmptyString = Field(
        default=None, validation_alias="CLOUDFLARE_ACCOUNT_ID"
    )

    # ==================== Google Gemini (Google AI Studio) ====================
    gemini_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="GEMINI_API_KEY"
    )

    # ==================== Google Vertex AI ====================
    vertex_project_id: OptionalNonEmptyString = Field(
        default=None, validation_alias="VERTEX_PROJECT_ID"
    )
    vertex_location: NonEmptyString = Field(
        default="global", validation_alias="VERTEX_LOCATION"
    )

    # ==================== Groq (OpenAI-compatible) ====================
    groq_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="GROQ_API_KEY"
    )

    # ==================== ClinePass (OpenAI-compatible) ====================
    cline_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="CLINE_API_KEY"
    )

    # ==================== xAI / Grok (OpenAI-compatible) ====================
    xai_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="XAI_API_KEY"
    )

    # ==================== QwenCloud Token Plan (OpenAI-compatible) ====================
    qwencloud_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="QWENCLOUD_API_KEY"
    )

    # ==================== QwenCloud Coding Plan (OpenAI-compatible) ====================
    qwencloud_coding_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="QWENCLOUD_CODING_API_KEY"
    )

    # ==================== Together AI (OpenAI-compatible) ====================
    together_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="TOGETHER_API_KEY"
    )

    # ==================== DeepInfra (OpenAI-compatible) ====================
    deepinfra_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="DEEPINFRA_API_KEY"
    )

    # ==================== SiliconFlow (OpenAI-compatible) ====================
    siliconflow_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="SILICONFLOW_API_KEY"
    )

    # ==================== Nebius Token Factory (OpenAI-compatible) ====================
    nebius_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="NEBIUS_API_KEY"
    )

    # ==================== Chutes (OpenAI-compatible) ====================
    chutes_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="CHUTES_API_KEY"
    )

    # ==================== Featherless AI (OpenAI-compatible) ====================
    featherless_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="FEATHERLESS_API_KEY"
    )

    # ==================== Agnes AI (OpenAI-compatible) ====================
    agnes_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="AGNES_API_KEY"
    )

    # ==================== ZenMux (OpenAI-compatible) ====================
    zenmux_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="ZENMUX_API_KEY"
    )

    # ==================== W&B Inference (OpenAI-compatible) ====================
    wandb_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="WANDB_API_KEY"
    )

    # ==================== Cerebras Inference (OpenAI-compatible) ====================
    cerebras_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="CEREBRAS_API_KEY"
    )

    # ==================== Ollama Cloud ====================
    ollama_api_key: OptionalNonEmptyString = Field(
        default=None, validation_alias="OLLAMA_API_KEY"
    )

    # ==================== Messaging Platform Selection ====================
    # Valid: "telegram" | "discord" | "none"
    messaging_platform: NonEmptyString = Field(
        default="discord", validation_alias="MESSAGING_PLATFORM"
    )
    messaging_rate_limit: int = Field(
        default=1, validation_alias="MESSAGING_RATE_LIMIT"
    )
    messaging_rate_window: float = Field(
        default=1.0, validation_alias="MESSAGING_RATE_WINDOW"
    )

    # ==================== NVIDIA NIM Config ====================
    nvidia_nim_api_key: OptionalNonEmptyString = Field(
        default=None,
        validation_alias="NVIDIA_NIM_API_KEY",
    )

    # ==================== LM Studio Config ====================
    lm_studio_base_url: NonEmptyString = Field(
        default="http://localhost:1234/v1",
        validation_alias="LM_STUDIO_BASE_URL",
    )

    # ==================== Llama.cpp Config ====================
    llamacpp_base_url: NonEmptyString = Field(
        default="http://localhost:8080/v1",
        validation_alias="LLAMACPP_BASE_URL",
    )

    # ==================== Ollama Config ====================
    ollama_base_url: NonEmptyString = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )

    # ==================== Model ====================
    # All Claude model requests are mapped to this single model (fallback)
    # Format: provider_type/model/name
    model: NonEmptyString = Field(
        default="nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        validation_alias="MODEL",
    )

    # Per-model overrides (optional, falls back to MODEL)
    # Each can use a different provider
    model_fable: OptionalNonEmptyString = Field(
        default=None, validation_alias="MODEL_FABLE"
    )
    model_opus: OptionalNonEmptyString = Field(
        default=None, validation_alias="MODEL_OPUS"
    )
    model_sonnet: OptionalNonEmptyString = Field(
        default=None, validation_alias="MODEL_SONNET"
    )
    model_haiku: OptionalNonEmptyString = Field(
        default=None, validation_alias="MODEL_HAIKU"
    )

    # ==================== Per-Provider Proxy ====================
    openai_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="OPENAI_PROXY"
    )
    xai_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="XAI_PROXY"
    )
    qwencloud_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="QWENCLOUD_PROXY"
    )
    qwencloud_coding_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="QWENCLOUD_CODING_PROXY"
    )
    together_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="TOGETHER_PROXY"
    )
    deepinfra_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="DEEPINFRA_PROXY"
    )
    siliconflow_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="SILICONFLOW_PROXY"
    )
    nebius_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="NEBIUS_PROXY"
    )
    chutes_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="CHUTES_PROXY"
    )
    featherless_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="FEATHERLESS_PROXY"
    )
    agnes_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="AGNES_PROXY"
    )
    zenmux_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="ZENMUX_PROXY"
    )
    wandb_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="WANDB_PROXY"
    )
    azure_openai_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="AZURE_OPENAI_PROXY"
    )
    nvidia_nim_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="NVIDIA_NIM_PROXY"
    )
    open_router_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="OPENROUTER_PROXY"
    )
    mistral_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="MISTRAL_PROXY"
    )
    codestral_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="CODESTRAL_PROXY"
    )
    lmstudio_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="LMSTUDIO_PROXY"
    )
    llamacpp_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="LLAMACPP_PROXY"
    )
    kimi_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="KIMI_PROXY"
    )
    kimi_code_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="KIMI_CODE_PROXY"
    )
    wafer_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="WAFER_PROXY"
    )
    minimax_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="MINIMAX_PROXY"
    )
    opencode_zen_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="OPENCODE_ZEN_PROXY"
    )
    opencode_go_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="OPENCODE_GO_PROXY"
    )
    vercel_ai_gateway_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="VERCEL_AI_GATEWAY_PROXY"
    )
    bedrock_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="BEDROCK_PROXY"
    )
    huggingface_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="HUGGINGFACE_PROXY"
    )
    cohere_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="COHERE_PROXY"
    )
    github_models_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="GITHUB_MODELS_PROXY"
    )
    sambanova_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="SAMBANOVA_PROXY"
    )
    kilo_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="KILO_PROXY"
    )
    zai_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="ZAI_PROXY"
    )
    zai_api_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="ZAI_API_PROXY"
    )
    tokenrouter_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="TOKENROUTER_PROXY"
    )
    nararoute_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="NARAROUTE_PROXY"
    )
    fireworks_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="FIREWORKS_PROXY"
    )
    novita_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="NOVITA_PROXY"
    )
    cloudflare_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="CLOUDFLARE_PROXY"
    )
    gemini_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="GEMINI_PROXY"
    )
    vertex_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="VERTEX_PROXY"
    )
    groq_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="GROQ_PROXY"
    )
    cline_pass_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="CLINE_PASS_PROXY"
    )
    cerebras_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="CEREBRAS_PROXY"
    )
    ollama_cloud_proxy: OptionalNonEmptyString = Field(
        default=None, validation_alias="OLLAMA_CLOUD_PROXY"
    )

    # ==================== Provider Rate Limiting ====================
    provider_rate_limit: int = Field(default=1, validation_alias="PROVIDER_RATE_LIMIT")
    provider_rate_window: int = Field(
        default=2, validation_alias="PROVIDER_RATE_WINDOW"
    )
    provider_max_concurrency: int = Field(
        default=2, validation_alias="PROVIDER_MAX_CONCURRENCY"
    )
    reasoning_policy: ReasoningPreference = Field(
        default=ReasoningPreference.CLIENT,
        validation_alias="REASONING_POLICY",
    )
    reasoning_fable: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_FABLE",
    )
    reasoning_opus: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_OPUS",
    )
    reasoning_sonnet: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_SONNET",
    )
    reasoning_haiku: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_HAIKU",
    )

    # ==================== HTTP Client Timeouts ====================
    http_read_timeout: float = Field(
        default=120.0, validation_alias="HTTP_READ_TIMEOUT"
    )
    http_write_timeout: float = Field(
        default=10.0, validation_alias="HTTP_WRITE_TIMEOUT"
    )
    http_connect_timeout: float = Field(
        default=HTTP_CONNECT_TIMEOUT_DEFAULT,
        validation_alias="HTTP_CONNECT_TIMEOUT",
    )

    # ==================== Fast Prefix Detection ====================
    fast_prefix_detection: bool = Field(
        default=True,
        validation_alias="FAST_PREFIX_DETECTION",
    )

    # ==================== Optimizations ====================
    enable_network_probe_mock: bool = Field(
        default=True,
        validation_alias="ENABLE_NETWORK_PROBE_MOCK",
    )
    enable_title_generation_skip: bool = Field(
        default=True,
        validation_alias="ENABLE_TITLE_GENERATION_SKIP",
    )
    enable_suggestion_mode_skip: bool = Field(
        default=True,
        validation_alias="ENABLE_SUGGESTION_MODE_SKIP",
    )
    enable_filepath_extraction_mock: bool = Field(
        default=True,
        validation_alias="ENABLE_FILEPATH_EXTRACTION_MOCK",
    )

    # ==================== Local web server tools (web_search / web_fetch) ====================
    # Off by default: these tools perform outbound HTTP from the proxy (SSRF risk).
    enable_web_server_tools: bool = Field(
        default=False, validation_alias="ENABLE_WEB_SERVER_TOOLS"
    )
    # Comma-separated URL schemes allowed for web_fetch (default: http,https).
    web_fetch_allowed_schemes: NonEmptyString = Field(
        default="http,https", validation_alias="WEB_FETCH_ALLOWED_SCHEMES"
    )
    # When true, skip private/loopback/link-local IP blocking for web_fetch (lab only).
    web_fetch_allow_private_networks: bool = Field(
        default=False, validation_alias="WEB_FETCH_ALLOW_PRIVATE_NETWORKS"
    )

    # ==================== Debug / diagnostic logging (avoid sensitive content) ====================
    # Minimum log level for the JSON file sink (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_level: NonEmptyString = Field(default="INFO", validation_alias="LOG_LEVEL")
    # When false (default), API and SSE helpers log only metadata (counts, lengths, ids).
    log_raw_api_payloads: bool = Field(
        default=False, validation_alias="LOG_RAW_API_PAYLOADS"
    )
    log_raw_sse_events: bool = Field(
        default=False, validation_alias="LOG_RAW_SSE_EVENTS"
    )
    # When false (default), unhandled exceptions log only type + route metadata (no message/traceback).
    log_api_error_tracebacks: bool = Field(
        default=False, validation_alias="LOG_API_ERROR_TRACEBACKS"
    )
    # When false (default), messaging logs omit text/transcription previews (metadata only).
    log_raw_messaging_content: bool = Field(
        default=False, validation_alias="LOG_RAW_MESSAGING_CONTENT"
    )
    # When true, log full Claude CLI stderr, non-JSON lines, and parser error text.
    log_raw_cli_diagnostics: bool = Field(
        default=False, validation_alias="LOG_RAW_CLI_DIAGNOSTICS"
    )
    # When true, log exception text / CLI error strings in messaging (may leak user content).
    log_messaging_error_details: bool = Field(
        default=False, validation_alias="LOG_MESSAGING_ERROR_DETAILS"
    )
    debug_platform_edits: bool = Field(
        default=False, validation_alias="DEBUG_PLATFORM_EDITS"
    )
    debug_subagent_stack: bool = Field(
        default=False, validation_alias="DEBUG_SUBAGENT_STACK"
    )

    # ==================== NIM Settings ====================
    nim: NimSettings = Field(default_factory=NimSettings)

    # ==================== Voice Note Transcription ====================
    voice_note_enabled: bool = Field(
        default=True, validation_alias="VOICE_NOTE_ENABLED"
    )
    # Device: "cpu" | "cuda" | "nvidia_nim"
    # - "cpu"/"cuda": local Whisper (requires voice_local extra: uv sync --extra voice_local)
    # - "nvidia_nim": NVIDIA NIM Whisper API (requires voice extra: uv sync --extra voice)
    whisper_device: NonEmptyString = Field(
        default="cpu", validation_alias="WHISPER_DEVICE"
    )
    # Whisper model ID or short name (for local Whisper) or NVIDIA NIM model (for nvidia_nim)
    # Local Whisper: "tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"
    # NVIDIA NIM: "nvidia/parakeet-ctc-1.1b-asr", "openai/whisper-large-v3", etc.
    whisper_model: NonEmptyString = Field(
        default="base", validation_alias="WHISPER_MODEL"
    )
    # ==================== Bot Wrapper Config ====================
    telegram_bot_token: OptionalNonEmptyString = Field(
        default=None,
        validation_alias="TELEGRAM_BOT_TOKEN",
    )
    allowed_telegram_user_id: OptionalNonEmptyString = Field(
        default=None,
        validation_alias="ALLOWED_TELEGRAM_USER_ID",
    )
    telegram_proxy_url: OptionalNonEmptyString = Field(
        default=None, validation_alias="TELEGRAM_PROXY_URL"
    )
    discord_bot_token: OptionalNonEmptyString = Field(
        default=None, validation_alias="DISCORD_BOT_TOKEN"
    )
    allowed_discord_channels: OptionalNonEmptyString = Field(
        default=None, validation_alias="ALLOWED_DISCORD_CHANNELS"
    )
    allowed_dir: OptionalNonEmptyString = Field(
        default=None,
        validation_alias="ALLOWED_DIR",
    )
    max_message_log_entries_per_chat: int | None = Field(
        default=None, validation_alias="MAX_MESSAGE_LOG_ENTRIES_PER_CHAT"
    )

    # ==================== Server ====================
    host: NonEmptyString = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8082, validation_alias="PORT")
    open_admin_browser: bool = Field(default=True, validation_alias="FCC_OPEN_BROWSER")
    proxy_auth_enabled: bool = Field(
        default=False,
        validation_alias="PROXY_AUTH_ENABLED",
    )
    proxy_auth_token: NonEmptyString = Field(
        default="freecc",
        validation_alias="ANTHROPIC_AUTH_TOKEN",
    )

    @field_validator("max_message_log_entries_per_chat", mode="before")
    @classmethod
    def parse_optional_log_cap(cls, v: object) -> object:
        if v == "" or v is None:
            return None
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(valid)}, got {v!r}")
        return upper

    @field_validator("reasoning_policy")
    @classmethod
    def validate_root_reasoning_policy(
        cls, value: ReasoningPreference
    ) -> ReasoningPreference:
        if value is ReasoningPreference.INHERIT:
            raise ValueError("REASONING_POLICY cannot inherit")
        return value

    @field_validator("whisper_device")
    @classmethod
    def validate_whisper_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda", "nvidia_nim"):
            raise ValueError(
                f"whisper_device must be 'cpu', 'cuda', or 'nvidia_nim', got {v!r}"
            )
        return v

    @field_validator("messaging_platform")
    @classmethod
    def validate_messaging_platform(cls, v: str) -> str:
        if v not in ("telegram", "discord", "none"):
            raise ValueError(
                f"messaging_platform must be 'telegram', 'discord', or 'none', got {v!r}"
            )
        return v

    @field_validator("messaging_rate_limit")
    @classmethod
    def validate_messaging_rate_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("messaging_rate_limit must be > 0")
        return v

    @field_validator("messaging_rate_window")
    @classmethod
    def validate_messaging_rate_window(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("messaging_rate_window must be > 0")
        return float(v)

    @field_validator("web_fetch_allowed_schemes")
    @classmethod
    def validate_web_fetch_allowed_schemes(cls, v: str) -> str:
        schemes = [part.strip().lower() for part in v.split(",") if part.strip()]
        if not schemes:
            raise ValueError("web_fetch_allowed_schemes must list at least one scheme")
        for scheme in schemes:
            if not scheme.isascii() or not scheme.isalpha():
                raise ValueError(
                    f"Invalid URL scheme in web_fetch_allowed_schemes: {scheme!r}"
                )
        return ",".join(schemes)

    @field_validator(
        "model", "model_fable", "model_opus", "model_sonnet", "model_haiku"
    )
    @classmethod
    def validate_model_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if "/" not in v:
            raise ValueError(
                f"Model must be prefixed with provider type. "
                f"Valid providers: {', '.join(SUPPORTED_PROVIDER_IDS)}. "
                f"Format: provider_type/model/name"
            )
        provider = v.split("/", 1)[0]
        if provider not in SUPPORTED_PROVIDER_IDS:
            supported = ", ".join(f"'{p}'" for p in SUPPORTED_PROVIDER_IDS)
            raise ValueError(f"Invalid provider: '{provider}'. Supported: {supported}")
        return v

    @model_validator(mode="after")
    def check_nvidia_nim_api_key(self) -> Settings:
        if (
            self.voice_note_enabled
            and self.whisper_device == "nvidia_nim"
            and self.nvidia_nim_api_key is None
        ):
            raise ValueError(
                "NVIDIA_NIM_API_KEY is required when WHISPER_DEVICE is 'nvidia_nim'. "
                "Set it in the Admin UI."
            )
        return self

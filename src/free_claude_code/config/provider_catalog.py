"""Neutral provider catalog: IDs, credentials, defaults, proxy and capability metadata.

Adapter factories live in :mod:`providers.runtime.factory`; this module stays free of
provider implementation imports (see contract tests).
"""

from dataclasses import dataclass
from enum import StrEnum

# Default upstream base URLs are owned here with the provider catalog.
NVIDIA_NIM_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
# Moonshot Kimi OpenAI-compatible Chat Completions API.
KIMI_DEFAULT_BASE = "https://api.moonshot.ai/v1"
# Kimi Code subscription OpenAI-compatible Chat Completions API.
KIMI_CODE_DEFAULT_BASE = "https://api.kimi.com/coding/v1"
WAFER_DEFAULT_BASE = "https://pass.wafer.ai/v1"
MINIMAX_DEFAULT_BASE = "https://api.minimax.io/v1"
# DeepSeek Chat Completions API; cache usage is reported on this endpoint.
DEEPSEEK_DEFAULT_BASE = "https://api.deepseek.com"
FIREWORKS_DEFAULT_BASE = "https://api.fireworks.ai/inference/v1"
NOVITA_DEFAULT_BASE = "https://api.novita.ai/openai/v1"
# Cloudflare account-scoped AI REST root; provider appends /accounts/{id}/ai/v1.
CLOUDFLARE_AI_REST_ROOT = "https://api.cloudflare.com/client/v4"
OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
MISTRAL_DEFAULT_BASE = "https://api.mistral.ai/v1"
# Codestral IDE/personal endpoint (distinct from La Plateforme ``api.mistral.ai`` keys).
CODESTRAL_DEFAULT_BASE = "https://codestral.mistral.ai/v1"
LMSTUDIO_DEFAULT_BASE = "http://localhost:1234/v1"
LLAMACPP_DEFAULT_BASE = "http://localhost:8080/v1"
OLLAMA_DEFAULT_BASE = "http://localhost:11434"
OLLAMA_CLOUD_DEFAULT_BASE = "https://ollama.com/v1"
OPENCODE_ZEN_DEFAULT_BASE = "https://opencode.ai/zen/v1"
OPENCODE_GO_DEFAULT_BASE = "https://opencode.ai/zen/go/v1"
VERCEL_AI_GATEWAY_DEFAULT_BASE = "https://ai-gateway.vercel.sh/v1"
# Amazon Bedrock Mantle OpenAI-compatible endpoint. The base URL remains
# configurable because API keys and model availability are region-scoped.
BEDROCK_DEFAULT_BASE = "https://bedrock-mantle.us-east-1.api.aws/v1"
HUGGINGFACE_DEFAULT_BASE = "https://router.huggingface.co/v1"
COHERE_DEFAULT_BASE = "https://api.cohere.ai/compatibility/v1"
GITHUB_MODELS_DEFAULT_BASE = "https://models.github.ai/inference"
# Z.ai OpenAI-compatible Chat Completions APIs. The endpoint selects billing.
ZAI_CODING_DEFAULT_BASE = "https://api.z.ai/api/coding/paas/v4"
ZAI_API_DEFAULT_BASE = "https://api.z.ai/api/paas/v4"
# Google AI Studio Gemini API OpenAI-compat layer (not Vertex AI).
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Vertex AI API root. The provider owns project/location endpoint composition.
VERTEX_AI_API_ROOT = "https://aiplatform.googleapis.com"
GROQ_DEFAULT_BASE = "https://api.groq.com/openai/v1"
# ClinePass subscription models through Cline's OpenAI-compatible API.
CLINE_DEFAULT_BASE = "https://api.cline.bot/api/v1"
CEREBRAS_DEFAULT_BASE = "https://api.cerebras.ai/v1"
SAMBANOVA_DEFAULT_BASE = "https://api.sambanova.ai/v1"
# Kilo.ai gateway OpenAI-compatible Chat Completions API.
KILO_DEFAULT_BASE = "https://api.kilo.ai/api/gateway"
OPENAI_CODEX_DEFAULT_BASE = "https://chatgpt.com/backend-api/codex"
# xAI OpenAI-compatible Chat Completions API.
XAI_DEFAULT_BASE = "https://api.x.ai/v1"
# QwenCloud Token Plan OpenAI-compatible Chat Completions API.
QWENCLOUD_DEFAULT_BASE = (
    "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
)
# QwenCloud Coding Plan OpenAI-compatible Chat Completions API.
QWENCLOUD_CODING_DEFAULT_BASE = "https://coding-intl.dashscope.aliyuncs.com/v1"
# Together AI OpenAI-compatible Chat Completions API.
TOGETHER_DEFAULT_BASE = "https://api.together.ai/v1"
# DeepInfra OpenAI-compatible Chat Completions API.
DEEPINFRA_DEFAULT_BASE = "https://api.deepinfra.com/v1/openai"
# SiliconFlow OpenAI-compatible Chat Completions API.
SILICONFLOW_DEFAULT_BASE = "https://api.siliconflow.com/v1"
# Nebius Token Factory OpenAI-compatible Chat Completions API.
NEBIUS_DEFAULT_BASE = "https://api.tokenfactory.nebius.com/v1"
# Chutes OpenAI-compatible Chat Completions API.
CHUTES_DEFAULT_BASE = "https://llm.chutes.ai/v1"
# Featherless AI OpenAI-compatible Chat Completions API.
FEATHERLESS_DEFAULT_BASE = "https://api.featherless.ai/v1"
# TokenRouter OpenAI-compatible Chat Completions gateway.
TOKENROUTER_DEFAULT_BASE = "https://api.tokenrouter.com/v1"
# NaraRoute OpenAI-compatible Chat Completions gateway.
NARAROUTE_DEFAULT_BASE = "https://router.bynara.id/v1"
# Agnes AI OpenAI-compatible Chat Completions API.
AGNES_DEFAULT_BASE = "https://apihub.agnes-ai.com/v1"
# ZenMux OpenAI-compatible Chat Completions gateway.
ZENMUX_DEFAULT_BASE = "https://zenmux.ai/api/v1"
# W&B Serverless Inference OpenAI-compatible API.
WANDB_INFERENCE_DEFAULT_BASE = "https://api.inference.wandb.ai/v1"


class ProviderAuthKind(StrEnum):
    """How a customer makes one provider available."""

    CONFIGURATION = "configuration"
    CONNECTED_ACCOUNT = "connected_account"


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Metadata for building :class:`~providers.base.ProviderConfig` and factory wiring."""

    provider_id: str
    display_name: str
    auth_kind: ProviderAuthKind = ProviderAuthKind.CONFIGURATION
    local: bool = False
    credential_env: str | None = None
    credential_url: str | None = None
    credential_attr: str | None = None
    static_credential: str | None = None
    default_base_url: str | None = None
    base_url_attr: str | None = None
    proxy_attr: str | None = None
    required_settings_attrs: tuple[str, ...] = ()

    def configuration_attrs(self) -> tuple[str, ...]:
        """Return settings fields whose non-empty values configure this provider."""
        if self.required_settings_attrs:
            return self.required_settings_attrs
        if self.credential_attr is not None:
            return (self.credential_attr,)
        if self.base_url_attr is not None:
            return (self.base_url_attr,)
        return ()


PROVIDER_CATALOG: dict[str, ProviderDescriptor] = {
    "nvidia_nim": ProviderDescriptor(
        provider_id="nvidia_nim",
        display_name="NVIDIA NIM",
        credential_env="NVIDIA_NIM_API_KEY",
        credential_url="https://build.nvidia.com/settings/api-keys",
        credential_attr="nvidia_nim_api_key",
        default_base_url=NVIDIA_NIM_DEFAULT_BASE,
        proxy_attr="nvidia_nim_proxy",
    ),
    "open_router": ProviderDescriptor(
        provider_id="open_router",
        display_name="OpenRouter",
        credential_env="OPENROUTER_API_KEY",
        credential_url="https://openrouter.ai/keys",
        credential_attr="open_router_api_key",
        default_base_url=OPENROUTER_DEFAULT_BASE,
        proxy_attr="open_router_proxy",
    ),
    "groq": ProviderDescriptor(
        provider_id="groq",
        display_name="Groq",
        credential_env="GROQ_API_KEY",
        credential_url="https://console.groq.com/keys",
        credential_attr="groq_api_key",
        default_base_url=GROQ_DEFAULT_BASE,
        proxy_attr="groq_proxy",
    ),
    "cline_pass": ProviderDescriptor(
        provider_id="cline_pass",
        display_name="ClinePass",
        credential_env="CLINE_API_KEY",
        credential_url="https://app.cline.bot",
        credential_attr="cline_api_key",
        default_base_url=CLINE_DEFAULT_BASE,
        proxy_attr="cline_pass_proxy",
    ),
    "openai": ProviderDescriptor(
        provider_id="openai",
        display_name="OpenAI / ChatGPT",
        auth_kind=ProviderAuthKind.CONNECTED_ACCOUNT,
        default_base_url=OPENAI_CODEX_DEFAULT_BASE,
        proxy_attr="openai_proxy",
    ),
    "xai": ProviderDescriptor(
        provider_id="xai",
        display_name="xAI (Grok)",
        credential_env="XAI_API_KEY",
        credential_url="https://console.x.ai/team/default/api-keys",
        credential_attr="xai_api_key",
        default_base_url=XAI_DEFAULT_BASE,
        proxy_attr="xai_proxy",
    ),
    "qwencloud": ProviderDescriptor(
        provider_id="qwencloud",
        display_name="QwenCloud Token Plan",
        credential_env="QWENCLOUD_API_KEY",
        credential_url="https://home.qwencloud.com/api-keys",
        credential_attr="qwencloud_api_key",
        default_base_url=QWENCLOUD_DEFAULT_BASE,
        proxy_attr="qwencloud_proxy",
    ),
    "qwencloud_coding": ProviderDescriptor(
        provider_id="qwencloud_coding",
        display_name="QwenCloud Coding Plan",
        credential_env="QWENCLOUD_CODING_API_KEY",
        credential_url="https://home.qwencloud.com/api-keys",
        credential_attr="qwencloud_coding_api_key",
        default_base_url=QWENCLOUD_CODING_DEFAULT_BASE,
        proxy_attr="qwencloud_coding_proxy",
    ),
    "together": ProviderDescriptor(
        provider_id="together",
        display_name="Together AI",
        credential_env="TOGETHER_API_KEY",
        credential_url="https://api.together.ai/settings/api-keys",
        credential_attr="together_api_key",
        default_base_url=TOGETHER_DEFAULT_BASE,
        proxy_attr="together_proxy",
    ),
    "deepinfra": ProviderDescriptor(
        provider_id="deepinfra",
        display_name="DeepInfra",
        credential_env="DEEPINFRA_API_KEY",
        credential_url="https://deepinfra.com/dash/api_keys",
        credential_attr="deepinfra_api_key",
        default_base_url=DEEPINFRA_DEFAULT_BASE,
        proxy_attr="deepinfra_proxy",
    ),
    "siliconflow": ProviderDescriptor(
        provider_id="siliconflow",
        display_name="SiliconFlow",
        credential_env="SILICONFLOW_API_KEY",
        credential_url="https://cloud.siliconflow.com/account/ak",
        credential_attr="siliconflow_api_key",
        default_base_url=SILICONFLOW_DEFAULT_BASE,
        proxy_attr="siliconflow_proxy",
    ),
    "nebius": ProviderDescriptor(
        provider_id="nebius",
        display_name="Nebius Token Factory",
        credential_env="NEBIUS_API_KEY",
        credential_url="https://tokenfactory.nebius.com/project/api-keys",
        credential_attr="nebius_api_key",
        default_base_url=NEBIUS_DEFAULT_BASE,
        proxy_attr="nebius_proxy",
    ),
    "chutes": ProviderDescriptor(
        provider_id="chutes",
        display_name="Chutes",
        credential_env="CHUTES_API_KEY",
        credential_url="https://chutes.ai/docs/getting-started/authentication",
        credential_attr="chutes_api_key",
        default_base_url=CHUTES_DEFAULT_BASE,
        proxy_attr="chutes_proxy",
    ),
    "featherless": ProviderDescriptor(
        provider_id="featherless",
        display_name="Featherless AI",
        credential_env="FEATHERLESS_API_KEY",
        credential_url="https://featherless.ai/account/api-keys",
        credential_attr="featherless_api_key",
        default_base_url=FEATHERLESS_DEFAULT_BASE,
        proxy_attr="featherless_proxy",
    ),
    "agnes": ProviderDescriptor(
        provider_id="agnes",
        display_name="Agnes AI",
        credential_env="AGNES_API_KEY",
        credential_url="https://agnes-ai.com/",
        credential_attr="agnes_api_key",
        default_base_url=AGNES_DEFAULT_BASE,
        proxy_attr="agnes_proxy",
    ),
    "zenmux": ProviderDescriptor(
        provider_id="zenmux",
        display_name="ZenMux",
        credential_env="ZENMUX_API_KEY",
        credential_url="https://zenmux.ai/platform/pay-as-you-go",
        credential_attr="zenmux_api_key",
        default_base_url=ZENMUX_DEFAULT_BASE,
        proxy_attr="zenmux_proxy",
    ),
    "wandb": ProviderDescriptor(
        provider_id="wandb",
        display_name="W&B Inference",
        credential_env="WANDB_API_KEY",
        credential_url="https://wandb.ai/settings",
        credential_attr="wandb_api_key",
        default_base_url=WANDB_INFERENCE_DEFAULT_BASE,
        proxy_attr="wandb_proxy",
    ),
    "azure_openai": ProviderDescriptor(
        provider_id="azure_openai",
        display_name="Azure OpenAI",
        credential_env="AZURE_OPENAI_API_KEY",
        credential_url="https://ai.azure.com/",
        credential_attr="azure_openai_api_key",
        base_url_attr="azure_openai_base_url",
        proxy_attr="azure_openai_proxy",
        required_settings_attrs=(
            "azure_openai_api_key",
            "azure_openai_base_url",
        ),
    ),
    "gemini": ProviderDescriptor(
        provider_id="gemini",
        display_name="Gemini",
        credential_env="GEMINI_API_KEY",
        credential_url="https://aistudio.google.com/apikey",
        credential_attr="gemini_api_key",
        default_base_url=GEMINI_DEFAULT_BASE,
        proxy_attr="gemini_proxy",
    ),
    "vertex": ProviderDescriptor(
        provider_id="vertex",
        display_name="Google Vertex AI",
        credential_url=(
            "https://cloud.google.com/docs/authentication/"
            "set-up-adc-local-dev-environment"
        ),
        default_base_url=VERTEX_AI_API_ROOT,
        proxy_attr="vertex_proxy",
        required_settings_attrs=("vertex_project_id",),
    ),
    "deepseek": ProviderDescriptor(
        provider_id="deepseek",
        display_name="DeepSeek",
        credential_env="DEEPSEEK_API_KEY",
        credential_url="https://platform.deepseek.com/api_keys",
        credential_attr="deepseek_api_key",
        default_base_url=DEEPSEEK_DEFAULT_BASE,
    ),
    "mistral": ProviderDescriptor(
        provider_id="mistral",
        display_name="Mistral",
        credential_env="MISTRAL_API_KEY",
        credential_url="https://console.mistral.ai/",
        credential_attr="mistral_api_key",
        default_base_url=MISTRAL_DEFAULT_BASE,
        proxy_attr="mistral_proxy",
    ),
    "mistral_codestral": ProviderDescriptor(
        provider_id="mistral_codestral",
        display_name="Mistral Codestral",
        credential_env="CODESTRAL_API_KEY",
        credential_url="https://console.mistral.ai/",
        credential_attr="codestral_api_key",
        default_base_url=CODESTRAL_DEFAULT_BASE,
        proxy_attr="codestral_proxy",
    ),
    "opencode_zen": ProviderDescriptor(
        provider_id="opencode_zen",
        display_name="OpenCode Zen",
        credential_env="OPENCODE_API_KEY",
        credential_url="https://opencode.ai/auth",
        credential_attr="opencode_api_key",
        default_base_url=OPENCODE_ZEN_DEFAULT_BASE,
        proxy_attr="opencode_zen_proxy",
    ),
    "opencode_go": ProviderDescriptor(
        provider_id="opencode_go",
        display_name="OpenCode Go",
        credential_env="OPENCODE_API_KEY",
        credential_url="https://opencode.ai/auth",
        credential_attr="opencode_api_key",
        default_base_url=OPENCODE_GO_DEFAULT_BASE,
        proxy_attr="opencode_go_proxy",
    ),
    "vercel": ProviderDescriptor(
        provider_id="vercel",
        display_name="Vercel AI Gateway",
        credential_env="AI_GATEWAY_API_KEY",
        credential_url="https://vercel.com/docs/ai-gateway",
        credential_attr="vercel_ai_gateway_api_key",
        default_base_url=VERCEL_AI_GATEWAY_DEFAULT_BASE,
        proxy_attr="vercel_ai_gateway_proxy",
    ),
    "bedrock": ProviderDescriptor(
        provider_id="bedrock",
        display_name="Amazon Bedrock",
        credential_env="AWS_BEARER_TOKEN_BEDROCK",
        credential_url="https://console.aws.amazon.com/bedrock/",
        credential_attr="bedrock_api_key",
        default_base_url=BEDROCK_DEFAULT_BASE,
        base_url_attr="bedrock_base_url",
        proxy_attr="bedrock_proxy",
    ),
    "huggingface": ProviderDescriptor(
        provider_id="huggingface",
        display_name="Hugging Face",
        credential_env="HUGGINGFACE_API_KEY",
        credential_url="https://huggingface.co/settings/tokens",
        credential_attr="huggingface_api_key",
        default_base_url=HUGGINGFACE_DEFAULT_BASE,
        proxy_attr="huggingface_proxy",
    ),
    "cohere": ProviderDescriptor(
        provider_id="cohere",
        display_name="Cohere",
        credential_env="COHERE_API_KEY",
        credential_url="https://dashboard.cohere.com/api-keys",
        credential_attr="cohere_api_key",
        default_base_url=COHERE_DEFAULT_BASE,
        proxy_attr="cohere_proxy",
    ),
    "github_models": ProviderDescriptor(
        provider_id="github_models",
        display_name="GitHub Models",
        credential_env="GITHUB_MODELS_TOKEN",
        credential_url="https://github.com/settings/tokens",
        credential_attr="github_models_token",
        default_base_url=GITHUB_MODELS_DEFAULT_BASE,
        proxy_attr="github_models_proxy",
    ),
    "wafer": ProviderDescriptor(
        provider_id="wafer",
        display_name="Wafer",
        credential_env="WAFER_API_KEY",
        credential_url="https://www.wafer.ai/pass",
        credential_attr="wafer_api_key",
        default_base_url=WAFER_DEFAULT_BASE,
        proxy_attr="wafer_proxy",
    ),
    "kimi": ProviderDescriptor(
        provider_id="kimi",
        display_name="Kimi",
        credential_env="KIMI_API_KEY",
        credential_url="https://platform.moonshot.cn/console/api-keys",
        credential_attr="kimi_api_key",
        default_base_url=KIMI_DEFAULT_BASE,
        proxy_attr="kimi_proxy",
    ),
    "kimi_code": ProviderDescriptor(
        provider_id="kimi_code",
        display_name="Kimi Code",
        credential_env="KIMI_CODE_API_KEY",
        credential_url="https://www.kimi.com/code/console",
        credential_attr="kimi_code_api_key",
        default_base_url=KIMI_CODE_DEFAULT_BASE,
        proxy_attr="kimi_code_proxy",
    ),
    "kilo": ProviderDescriptor(
        provider_id="kilo",
        display_name="Kilo.ai",
        credential_env="KILO_API_KEY",
        credential_url="https://app.kilo.ai",
        credential_attr="kilo_api_key",
        default_base_url=KILO_DEFAULT_BASE,
        proxy_attr="kilo_proxy",
    ),
    "minimax": ProviderDescriptor(
        provider_id="minimax",
        display_name="MiniMax",
        credential_env="MINIMAX_API_KEY",
        credential_url="https://platform.minimax.io/user-center/basic-information/interface-key",
        credential_attr="minimax_api_key",
        default_base_url=MINIMAX_DEFAULT_BASE,
        proxy_attr="minimax_proxy",
    ),
    "cerebras": ProviderDescriptor(
        provider_id="cerebras",
        display_name="Cerebras",
        credential_env="CEREBRAS_API_KEY",
        credential_url="https://cloud.cerebras.ai",
        credential_attr="cerebras_api_key",
        default_base_url=CEREBRAS_DEFAULT_BASE,
        proxy_attr="cerebras_proxy",
    ),
    "sambanova": ProviderDescriptor(
        provider_id="sambanova",
        display_name="SambaNova",
        credential_env="SAMBANOVA_API_KEY",
        credential_url="https://cloud.sambanova.ai/apis",
        credential_attr="sambanova_api_key",
        default_base_url=SAMBANOVA_DEFAULT_BASE,
        proxy_attr="sambanova_proxy",
    ),
    "fireworks": ProviderDescriptor(
        provider_id="fireworks",
        display_name="Fireworks",
        credential_env="FIREWORKS_API_KEY",
        credential_url="https://fireworks.ai/account/api-keys",
        credential_attr="fireworks_api_key",
        default_base_url=FIREWORKS_DEFAULT_BASE,
        proxy_attr="fireworks_proxy",
    ),
    "novita": ProviderDescriptor(
        provider_id="novita",
        display_name="Novita AI",
        credential_env="NOVITA_API_KEY",
        credential_url="https://novita.ai/settings/key-management",
        credential_attr="novita_api_key",
        default_base_url=NOVITA_DEFAULT_BASE,
        proxy_attr="novita_proxy",
    ),
    "cloudflare": ProviderDescriptor(
        provider_id="cloudflare",
        display_name="Cloudflare",
        credential_env="CLOUDFLARE_API_TOKEN",
        credential_url="https://dash.cloudflare.com/profile/api-tokens",
        credential_attr="cloudflare_api_token",
        default_base_url=CLOUDFLARE_AI_REST_ROOT,
        proxy_attr="cloudflare_proxy",
        required_settings_attrs=(
            "cloudflare_api_token",
            "cloudflare_account_id",
        ),
    ),
    "zai": ProviderDescriptor(
        provider_id="zai",
        display_name="Z.ai Coding Plan",
        credential_env="ZAI_API_KEY",
        credential_url="https://z.ai/manage-apikey/apikey-list",
        credential_attr="zai_api_key",
        default_base_url=ZAI_CODING_DEFAULT_BASE,
        proxy_attr="zai_proxy",
    ),
    "zai_api": ProviderDescriptor(
        provider_id="zai_api",
        display_name="Z.ai API",
        credential_env="ZAI_API_KEY",
        credential_url="https://z.ai/manage-apikey/apikey-list",
        credential_attr="zai_api_key",
        default_base_url=ZAI_API_DEFAULT_BASE,
        proxy_attr="zai_api_proxy",
    ),
    "tokenrouter": ProviderDescriptor(
        provider_id="tokenrouter",
        display_name="TokenRouter",
        credential_env="TOKENROUTER_API_KEY",
        credential_url="https://www.tokenrouter.com/",
        credential_attr="tokenrouter_api_key",
        default_base_url=TOKENROUTER_DEFAULT_BASE,
        base_url_attr="tokenrouter_base_url",
        proxy_attr="tokenrouter_proxy",
    ),
    "nararoute": ProviderDescriptor(
        provider_id="nararoute",
        display_name="NaraRoute",
        credential_env="NARAROUTE_API_KEY",
        credential_url="https://router.bynara.id/keys",
        credential_attr="nararoute_api_key",
        default_base_url=NARAROUTE_DEFAULT_BASE,
        base_url_attr="nararoute_base_url",
        proxy_attr="nararoute_proxy",
    ),
    "ollama_cloud": ProviderDescriptor(
        provider_id="ollama_cloud",
        display_name="Ollama Cloud",
        credential_env="OLLAMA_API_KEY",
        credential_url="https://ollama.com/settings/keys",
        credential_attr="ollama_api_key",
        default_base_url=OLLAMA_CLOUD_DEFAULT_BASE,
        proxy_attr="ollama_cloud_proxy",
    ),
    "lmstudio": ProviderDescriptor(
        provider_id="lmstudio",
        display_name="LM Studio",
        static_credential="lm-studio",
        default_base_url=LMSTUDIO_DEFAULT_BASE,
        base_url_attr="lm_studio_base_url",
        proxy_attr="lmstudio_proxy",
        local=True,
    ),
    "llamacpp": ProviderDescriptor(
        provider_id="llamacpp",
        display_name="llama.cpp",
        static_credential="llamacpp",
        default_base_url=LLAMACPP_DEFAULT_BASE,
        base_url_attr="llamacpp_base_url",
        proxy_attr="llamacpp_proxy",
        local=True,
    ),
    "ollama": ProviderDescriptor(
        provider_id="ollama",
        display_name="Ollama",
        static_credential="ollama",
        default_base_url=OLLAMA_DEFAULT_BASE,
        base_url_attr="ollama_base_url",
        local=True,
    ),
}

# Key order:
# NVIDIA NIM, OpenRouter, and Groq lead the customer-facing ranking;
# OpenCode gateways remain adjacent,
# Vercel / Hugging Face / Cohere / GitHub Models follow gateway-style remotes,
# then cloud gateways, Ollama Cloud, and local providers per project plan
# (github.com/cheahjs/free-llm-api-resources Free Providers TOC as rough guide
# beyond fixed slots).
# ``SUPPORTED_PROVIDER_IDS`` inherits this insertion order for UI and error-message listing.
SUPPORTED_PROVIDER_IDS: tuple[str, ...] = tuple(PROVIDER_CATALOG.keys())

if len(set(SUPPORTED_PROVIDER_IDS)) != len(SUPPORTED_PROVIDER_IDS):
    raise AssertionError("Duplicate provider ids in PROVIDER_CATALOG key order")

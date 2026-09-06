import asyncio
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.errors import (
    ApplicationUnavailableError,
    UnknownProviderError,
)
from free_claude_code.config.nim import NimSettings
from free_claude_code.config.provider_catalog import (
    AGNES_DEFAULT_BASE,
    BEDROCK_DEFAULT_BASE,
    CHUTES_DEFAULT_BASE,
    CLINE_DEFAULT_BASE,
    COHERE_DEFAULT_BASE,
    DEEPINFRA_DEFAULT_BASE,
    FEATHERLESS_DEFAULT_BASE,
    HUGGINGFACE_DEFAULT_BASE,
    KIMI_CODE_DEFAULT_BASE,
    LLM7_DEFAULT_BASE,
    MINIMAX_DEFAULT_BASE,
    NARAROUTE_DEFAULT_BASE,
    NEBIUS_DEFAULT_BASE,
    OLLAMA_CLOUD_DEFAULT_BASE,
    POOLSIDE_DEFAULT_BASE,
    PROVIDER_CATALOG,
    QWENCLOUD_CODING_DEFAULT_BASE,
    QWENCLOUD_DEFAULT_BASE,
    SILICONFLOW_DEFAULT_BASE,
    SUPPORTED_PROVIDER_IDS,
    TOGETHER_DEFAULT_BASE,
    TOKENROUTER_DEFAULT_BASE,
    VERCEL_AI_GATEWAY_DEFAULT_BASE,
    WANDB_INFERENCE_DEFAULT_BASE,
    XAI_DEFAULT_BASE,
    XKIRO_DEFAULT_BASE,
    ZAI_API_DEFAULT_BASE,
    ZAI_CODING_DEFAULT_BASE,
    ZENMUX_DEFAULT_BASE,
)
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.cloudflare import CloudflareProvider
from free_claude_code.providers.deepseek import DeepSeekProvider
from free_claude_code.providers.gemini import GeminiProvider
from free_claude_code.providers.github_copilot.provider import GitHubCopilotProvider
from free_claude_code.providers.groq import GroqProvider
from free_claude_code.providers.kilo import KiloProvider
from free_claude_code.providers.lmstudio import LMStudioProvider
from free_claude_code.providers.mistral import MistralProvider
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.openai_chat import (
    OPENAI_CHAT_PROFILES,
    OpenAIChatProvider,
)
from free_claude_code.providers.openai_codex import OpenAICodexProvider
from free_claude_code.providers.opencode import OpenCodeProvider
from free_claude_code.providers.runtime import (
    ProviderRuntime,
    build_provider_config,
    create_provider,
)
from free_claude_code.providers.vertex import VertexProvider


def _make_settings(**overrides):
    mock = MagicMock()
    mock.model = "nvidia_nim/meta/llama3"
    mock.model_fable = None
    mock.model_opus = None
    mock.model_sonnet = None
    mock.model_haiku = None
    mock.azure_openai_api_key = "test_azure_openai_key"
    mock.azure_openai_base_url = "https://test-resource.openai.azure.com/openai/v1/"
    mock.nvidia_nim_api_key = "test_key"
    mock.open_router_api_key = "test_openrouter_key"
    mock.xai_api_key = "test_xai_key"
    mock.qwencloud_api_key = "test_qwencloud_key"
    mock.qwencloud_coding_api_key = "test_qwencloud_coding_key"
    mock.together_api_key = "test_together_key"
    mock.deepinfra_api_key = "test_deepinfra_key"
    mock.siliconflow_api_key = "test_siliconflow_key"
    mock.nebius_api_key = "test_nebius_key"
    mock.chutes_api_key = "test_chutes_key"
    mock.featherless_api_key = "test_featherless_key"
    mock.mistral_api_key = "test_mistral_key"
    mock.codestral_api_key = "test_codestral_key"
    mock.deepseek_api_key = "test_deepseek_key"
    mock.wafer_api_key = "test_wafer_key"
    mock.minimax_api_key = "test_minimax_key"
    mock.opencode_api_key = "test_opencode_key"
    mock.vercel_ai_gateway_api_key = "test_vercel_key"
    mock.bedrock_api_key = "test_bedrock_key"
    mock.bedrock_base_url = BEDROCK_DEFAULT_BASE
    mock.huggingface_api_key = "test_huggingface_key"
    mock.cohere_api_key = "test_cohere_key"
    mock.zai_api_key = "test_zai_key"
    mock.tokenrouter_api_key = "test_tokenrouter_key"
    mock.tokenrouter_base_url = TOKENROUTER_DEFAULT_BASE
    mock.nararoute_api_key = "test_nararoute_key"
    mock.nararoute_base_url = NARAROUTE_DEFAULT_BASE
    mock.xkiro_api_key = "test_xkiro_key"
    mock.xkiro_base_url = XKIRO_DEFAULT_BASE
    mock.agnes_api_key = "test_agnes_key"
    mock.zenmux_api_key = "test_zenmux_key"
    mock.wandb_api_key = "test_wandb_key"
    mock.lm_studio_base_url = "http://localhost:1234/v1"
    mock.llamacpp_base_url = "http://localhost:8080/v1"
    mock.ollama_base_url = "http://localhost:11434"
    mock.ollama_api_key = "test_ollama_cloud_key"
    mock.poolside_api_key = "test_poolside_key"
    mock.llm7_api_key = "test_llm7_key"
    mock.nvidia_nim_proxy = None
    mock.open_router_proxy = None
    mock.lmstudio_proxy = None
    mock.llamacpp_proxy = None
    mock.mistral_proxy = None
    mock.codestral_proxy = None
    mock.kimi_proxy = None
    mock.kimi_api_key = "test_kimi_key"
    mock.kimi_code_proxy = None
    mock.kimi_code_api_key = "test_kimi_code_key"
    mock.wafer_proxy = None
    mock.minimax_proxy = None
    mock.opencode_zen_proxy = None
    mock.opencode_go_proxy = None
    mock.vercel_ai_gateway_proxy = None
    mock.bedrock_proxy = None
    mock.huggingface_proxy = None
    mock.cohere_proxy = None
    mock.zai_proxy = None
    mock.zai_api_proxy = None
    mock.tokenrouter_proxy = None
    mock.nararoute_proxy = None
    mock.xkiro_proxy = None
    mock.agnes_proxy = None
    mock.zenmux_proxy = None
    mock.wandb_proxy = None
    mock.fireworks_proxy = None
    mock.fireworks_api_key = "test_fireworks_key"
    mock.novita_proxy = None
    mock.novita_api_key = "test_novita_key"
    mock.cloudflare_api_token = "test_cloudflare_token"
    mock.cloudflare_account_id = "test_cloudflare_account"
    mock.cloudflare_proxy = None
    mock.gemini_api_key = ""
    mock.gemini_proxy = None
    mock.vertex_project_id = "test-vertex-project"
    mock.vertex_location = "global"
    mock.vertex_proxy = None
    mock.groq_api_key = ""
    mock.groq_proxy = None
    mock.cline_api_key = ""
    mock.cline_pass_proxy = None
    mock.cerebras_api_key = ""
    mock.cerebras_proxy = None
    mock.ollama_cloud_proxy = None
    mock.poolside_proxy = None
    mock.llm7_proxy = None
    mock.kilo_api_key = "test_kilo_key"
    mock.kilo_proxy = None
    mock.openai_proxy = None
    mock.xai_proxy = None
    mock.qwencloud_proxy = None
    mock.qwencloud_coding_proxy = None
    mock.together_proxy = None
    mock.deepinfra_proxy = None
    mock.siliconflow_proxy = None
    mock.nebius_proxy = None
    mock.chutes_proxy = None
    mock.featherless_proxy = None
    mock.azure_openai_proxy = None
    mock.provider_rate_limit = 40
    mock.provider_rate_window = 60
    mock.provider_max_concurrency = 5
    mock.http_read_timeout = 300.0
    mock.http_write_timeout = 10.0
    mock.http_connect_timeout = 10.0
    mock.log_raw_sse_events = False
    mock.log_api_error_tracebacks = False
    mock.nim = NimSettings()
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def test_importing_runtime_does_not_eager_load_other_adapters() -> None:
    """Runtime metadata must not import every provider adapter up front."""
    code = (
        "import sys\n"
        "import free_claude_code.providers.runtime\n"
        "assert 'free_claude_code.providers.open_router' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_provider_catalog_covers_advertised_provider_ids():
    assert set(PROVIDER_CATALOG) == set(SUPPORTED_PROVIDER_IDS)
    assert set(OPENAI_CHAT_PROFILES) < set(PROVIDER_CATALOG)
    for descriptor in PROVIDER_CATALOG.values():
        assert descriptor.provider_id


def test_ollama_descriptor_uses_local_openai_endpoint_semantics():
    descriptor = PROVIDER_CATALOG["ollama"]

    assert descriptor.default_base_url == "http://localhost:11434"
    assert descriptor.local is True


def test_ollama_cloud_descriptor_uses_direct_authenticated_endpoint():
    descriptor = PROVIDER_CATALOG["ollama_cloud"]

    assert descriptor.default_base_url == OLLAMA_CLOUD_DEFAULT_BASE
    assert descriptor.credential_env == "OLLAMA_API_KEY"
    assert descriptor.credential_attr == "ollama_api_key"
    assert descriptor.base_url_attr is None
    assert descriptor.local is False


def test_ollama_cloud_provider_config_uses_key_and_proxy():
    descriptor = PROVIDER_CATALOG["ollama_cloud"]
    settings = _make_settings(
        ollama_api_key="ollama-cloud-token",
        ollama_cloud_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "ollama-cloud-token"
    assert config.base_url == OLLAMA_CLOUD_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"


def test_poolside_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["poolside"]
    settings = _make_settings(
        poolside_api_key="poolside-token",
        poolside_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("poolside", settings)

    assert descriptor.display_name == "Poolside AI"
    assert descriptor.credential_env == "POOLSIDE_API_KEY"
    assert descriptor.credential_attr == "poolside_api_key"
    assert descriptor.credential_url == "https://platform.poolside.ai/"
    assert descriptor.default_base_url == POOLSIDE_DEFAULT_BASE
    assert descriptor.base_url_attr is None
    assert descriptor.proxy_attr == "poolside_proxy"
    assert config.api_key == "poolside-token"
    assert config.base_url == POOLSIDE_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_llm7_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["llm7"]
    settings = _make_settings(
        llm7_api_key="llm7-token",
        llm7_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("llm7", settings)

    assert descriptor.display_name == "LLM7.io"
    assert descriptor.credential_env == "LLM7_API_KEY"
    assert descriptor.credential_attr == "llm7_api_key"
    assert descriptor.credential_url == "https://dash.llm7.io/"
    assert descriptor.default_base_url == LLM7_DEFAULT_BASE
    assert descriptor.base_url_attr is None
    assert descriptor.proxy_attr == "llm7_proxy"
    assert config.api_key == "llm7-token"
    assert config.base_url == LLM7_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_xai_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["xai"]
    settings = _make_settings(
        xai_api_key="xai-token",
        xai_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("xai", settings)

    assert descriptor.display_name == "xAI (Grok)"
    assert descriptor.credential_env == "XAI_API_KEY"
    assert config.api_key == "xai-token"
    assert config.base_url == XAI_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_qwencloud_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["qwencloud"]
    settings = _make_settings(
        qwencloud_api_key="qwencloud-token",
        qwencloud_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("qwencloud", settings)

    assert descriptor.display_name == "QwenCloud Token Plan"
    assert descriptor.credential_env == "QWENCLOUD_API_KEY"
    assert config.api_key == "qwencloud-token"
    assert config.base_url == QWENCLOUD_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_cline_pass_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["cline_pass"]
    settings = _make_settings(
        cline_api_key="cline-programmatic-token",
        cline_pass_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("cline_pass", settings)

    assert descriptor.display_name == "ClinePass"
    assert descriptor.credential_env == "CLINE_API_KEY"
    assert descriptor.credential_attr == "cline_api_key"
    assert descriptor.credential_url == "https://app.cline.bot"
    assert descriptor.base_url_attr is None
    assert config.api_key == "cline-programmatic-token"
    assert config.base_url == CLINE_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)
    assert provider._provider_name == "CLINE_PASS"


def test_qwencloud_coding_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["qwencloud_coding"]
    settings = _make_settings(
        qwencloud_coding_api_key="qwencloud-coding-token",
        qwencloud_coding_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("qwencloud_coding", settings)

    assert descriptor.display_name == "QwenCloud Coding Plan"
    assert descriptor.credential_env == "QWENCLOUD_CODING_API_KEY"
    assert config.api_key == "qwencloud-coding-token"
    assert config.base_url == QWENCLOUD_CODING_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_together_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["together"]
    settings = _make_settings(
        together_api_key="together-token",
        together_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("together", settings)

    assert descriptor.display_name == "Together AI"
    assert descriptor.credential_env == "TOGETHER_API_KEY"
    assert config.api_key == "together-token"
    assert config.base_url == TOGETHER_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_deepinfra_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["deepinfra"]
    settings = _make_settings(
        deepinfra_api_key="deepinfra-token",
        deepinfra_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("deepinfra", settings)

    assert descriptor.display_name == "DeepInfra"
    assert descriptor.credential_env == "DEEPINFRA_API_KEY"
    assert config.api_key == "deepinfra-token"
    assert config.base_url == DEEPINFRA_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_siliconflow_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["siliconflow"]
    settings = _make_settings(
        siliconflow_api_key="siliconflow-token",
        siliconflow_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("siliconflow", settings)

    assert descriptor.display_name == "SiliconFlow"
    assert descriptor.credential_env == "SILICONFLOW_API_KEY"
    assert descriptor.base_url_attr is None
    assert config.api_key == "siliconflow-token"
    assert config.base_url == SILICONFLOW_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_nebius_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["nebius"]
    settings = _make_settings(
        nebius_api_key="nebius-token",
        nebius_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("nebius", settings)

    assert descriptor.display_name == "Nebius Token Factory"
    assert descriptor.credential_env == "NEBIUS_API_KEY"
    assert descriptor.credential_url == (
        "https://tokenfactory.nebius.com/project/api-keys"
    )
    assert descriptor.base_url_attr is None
    assert config.api_key == "nebius-token"
    assert config.base_url == NEBIUS_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_chutes_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["chutes"]
    settings = _make_settings(
        chutes_api_key="chutes-token",
        chutes_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("chutes", settings)

    assert descriptor.display_name == "Chutes"
    assert descriptor.credential_env == "CHUTES_API_KEY"
    assert descriptor.credential_url == (
        "https://chutes.ai/docs/getting-started/authentication"
    )
    assert descriptor.base_url_attr is None
    assert config.api_key == "chutes-token"
    assert config.base_url == CHUTES_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_featherless_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["featherless"]
    settings = _make_settings(
        featherless_api_key="featherless-token",
        featherless_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("featherless", settings)

    assert descriptor.display_name == "Featherless AI"
    assert descriptor.credential_env == "FEATHERLESS_API_KEY"
    assert descriptor.credential_url == "https://featherless.ai/account/api-keys"
    assert descriptor.base_url_attr is None
    assert config.api_key == "featherless-token"
    assert config.base_url == FEATHERLESS_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_agnes_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["agnes"]
    settings = _make_settings(
        agnes_api_key="agnes-token",
        agnes_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("agnes", settings)

    assert descriptor.display_name == "Agnes AI"
    assert descriptor.credential_env == "AGNES_API_KEY"
    assert descriptor.base_url_attr is None
    assert config.api_key == "agnes-token"
    assert config.base_url == AGNES_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_zenmux_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["zenmux"]
    settings = _make_settings(
        zenmux_api_key="zenmux-token",
        zenmux_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("zenmux", settings)

    assert descriptor.display_name == "ZenMux"
    assert descriptor.credential_env == "ZENMUX_API_KEY"
    assert descriptor.base_url_attr is None
    assert config.api_key == "zenmux-token"
    assert config.base_url == ZENMUX_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_wandb_provider_config_uses_key_base_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["wandb"]
    settings = _make_settings(
        wandb_api_key="wandb-token",
        wandb_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("wandb", settings)

    assert descriptor.display_name == "W&B Inference"
    assert descriptor.credential_env == "WANDB_API_KEY"
    assert descriptor.credential_url == "https://wandb.ai/settings"
    assert descriptor.base_url_attr is None
    assert config.api_key == "wandb-token"
    assert config.base_url == WANDB_INFERENCE_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)


def test_bedrock_provider_config_uses_regional_base_key_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["bedrock"]
    settings = _make_settings(
        bedrock_api_key="bedrock-token",
        bedrock_base_url="https://bedrock-mantle.eu-west-1.api.aws/v1",
        bedrock_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert descriptor.credential_env == "AWS_BEARER_TOKEN_BEDROCK"
    assert descriptor.default_base_url == BEDROCK_DEFAULT_BASE
    assert config.api_key == "bedrock-token"
    assert config.base_url == "https://bedrock-mantle.eu-west-1.api.aws/v1"
    assert config.proxy == "http://proxy.test:8080"


def test_azure_openai_provider_config_uses_resource_base_key_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["azure_openai"]
    settings = _make_settings(
        azure_openai_api_key="azure-token",
        azure_openai_base_url=("https://resource.openai.azure.com/openai/v1/"),
        azure_openai_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert descriptor.default_base_url is None
    assert descriptor.configuration_attrs() == (
        "azure_openai_api_key",
        "azure_openai_base_url",
    )
    assert config.api_key == "azure-token"
    assert config.base_url == "https://resource.openai.azure.com/openai/v1/"
    assert config.proxy == "http://proxy.test:8080"


def test_azure_openai_provider_config_reports_missing_resource_url() -> None:
    descriptor = PROVIDER_CATALOG["azure_openai"]

    with pytest.raises(
        ApplicationUnavailableError,
        match="AZURE_OPENAI_BASE_URL is not set",
    ):
        build_provider_config(
            descriptor,
            _make_settings(
                azure_openai_api_key="azure-token",
                azure_openai_base_url="",
            ),
        )


@pytest.mark.parametrize(
    ("provider_id", "expected_api_key"),
    [
        ("lmstudio", "lm-studio"),
        ("llamacpp", "llamacpp"),
        ("ollama", "ollama"),
    ],
)
def test_local_provider_factory_resolves_catalog_static_credential(
    provider_id: str,
    expected_api_key: str,
) -> None:
    descriptor = PROVIDER_CATALOG[provider_id]
    settings = _make_settings()

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider(provider_id, settings)

    assert config.api_key == expected_api_key
    assert isinstance(provider, OpenAIChatProvider)
    assert provider._api_key == expected_api_key


def test_zai_coding_descriptor_uses_fixed_cloud_base_url():
    descriptor = PROVIDER_CATALOG["zai"]

    assert descriptor.display_name == "Z.ai Coding Plan"
    assert descriptor.default_base_url == ZAI_CODING_DEFAULT_BASE
    assert descriptor.base_url_attr is None


def test_zai_provider_config_ignores_stale_base_url_setting():
    descriptor = PROVIDER_CATALOG["zai"]

    config = build_provider_config(
        descriptor,
        _make_settings(zai_base_url="https://custom.zai.invalid/v1"),
    )

    assert config.base_url == ZAI_CODING_DEFAULT_BASE


def test_zai_api_provider_config_uses_shared_key_general_base_and_own_proxy():
    descriptor = PROVIDER_CATALOG["zai_api"]
    settings = _make_settings(
        zai_api_key="shared-zai-key",
        zai_api_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("zai_api", settings)

    assert descriptor.display_name == "Z.ai API"
    assert descriptor.credential_env == "ZAI_API_KEY"
    assert descriptor.credential_attr == "zai_api_key"
    assert descriptor.default_base_url == ZAI_API_DEFAULT_BASE
    assert descriptor.base_url_attr is None
    assert descriptor.proxy_attr == "zai_api_proxy"
    assert config.api_key == "shared-zai-key"
    assert config.base_url == ZAI_API_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"
    assert isinstance(provider, OpenAIChatProvider)
    assert provider._provider_name == "ZAI_API"


def test_minimax_descriptor_uses_expected_endpoint_and_credential():
    descriptor = PROVIDER_CATALOG["minimax"]

    assert descriptor.default_base_url == MINIMAX_DEFAULT_BASE
    assert descriptor.credential_env == "MINIMAX_API_KEY"


def test_kimi_code_provider_config_uses_subscription_key_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["kimi_code"]
    settings = _make_settings(
        kimi_code_api_key="subscription-token",
        kimi_code_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert descriptor.credential_env == "KIMI_CODE_API_KEY"
    assert config.api_key == "subscription-token"
    assert config.base_url == KIMI_CODE_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"


def test_cloudflare_descriptor_uses_api_root_not_account_url():
    descriptor = PROVIDER_CATALOG["cloudflare"]

    assert descriptor.default_base_url == "https://api.cloudflare.com/client/v4"
    assert descriptor.base_url_attr is None


def test_create_cloudflare_provider_uses_account_scoped_base_url():
    settings = _make_settings(
        cloudflare_api_token="test_cloudflare_token",
        cloudflare_account_id="test-account",
    )

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("cloudflare", settings)

    assert isinstance(provider, CloudflareProvider)
    assert provider._base_url == (
        "https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1"
    )


def test_opencode_zen_provider_config_uses_explicit_id_and_name():
    with patch("httpx.AsyncClient"):
        provider = create_provider("opencode_zen", _make_settings())

    assert isinstance(provider, OpenCodeProvider)
    assert provider._base_url == "https://opencode.ai/zen/v1"
    assert provider._provider_name == "OPENCODE_ZEN"
    assert provider._api_key == "test_opencode_key"
    assert provider._responses._admission is provider._admission


def test_opencode_go_provider_config_uses_correct_base_url_and_name():
    with patch("httpx.AsyncClient"):
        provider = create_provider("opencode_go", _make_settings())

    assert isinstance(provider, OpenCodeProvider)
    assert provider._base_url == "https://opencode.ai/zen/go/v1"
    assert provider._provider_name == "OPENCODE_GO"
    assert provider._api_key == "test_opencode_key"
    assert provider._responses._admission is provider._admission


def test_opencode_go_catalog_uses_opencode_api_key() -> None:
    desc = PROVIDER_CATALOG["opencode_go"]

    assert desc.credential_env == "OPENCODE_API_KEY"
    assert desc.credential_attr == "opencode_api_key"


def test_build_provider_config_opencode_go_uses_opencode_api_key() -> None:
    descriptor = PROVIDER_CATALOG["opencode_go"]
    settings = _make_settings(opencode_api_key="shared-opencode-token")

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "shared-opencode-token"


def test_vercel_descriptor_uses_openai_chat_gateway() -> None:
    descriptor = PROVIDER_CATALOG["vercel"]

    assert descriptor.default_base_url == VERCEL_AI_GATEWAY_DEFAULT_BASE
    assert descriptor.credential_env == "AI_GATEWAY_API_KEY"
    assert descriptor.proxy_attr == "vercel_ai_gateway_proxy"


def test_huggingface_descriptor_uses_openai_chat_router() -> None:
    descriptor = PROVIDER_CATALOG["huggingface"]

    assert descriptor.default_base_url == HUGGINGFACE_DEFAULT_BASE
    assert descriptor.credential_env == "HUGGINGFACE_API_KEY"
    assert descriptor.proxy_attr == "huggingface_proxy"


def test_cohere_descriptor_uses_openai_chat_compatibility_api() -> None:
    descriptor = PROVIDER_CATALOG["cohere"]

    assert descriptor.default_base_url == COHERE_DEFAULT_BASE
    assert descriptor.credential_env == "COHERE_API_KEY"
    assert descriptor.proxy_attr == "cohere_proxy"


def test_build_provider_config_vercel_uses_gateway_key_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["vercel"]
    settings = _make_settings(
        vercel_ai_gateway_api_key="vercel-token",
        vercel_ai_gateway_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "vercel-token"
    assert config.proxy == "http://proxy.test:8080"


def test_build_provider_config_huggingface_uses_api_key_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["huggingface"]
    settings = _make_settings(
        huggingface_api_key="hf-token",
        huggingface_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "hf-token"
    assert config.proxy == "http://proxy.test:8080"


def test_build_provider_config_cohere_uses_api_key_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["cohere"]
    settings = _make_settings(
        cohere_api_key="cohere-token",
        cohere_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "cohere-token"
    assert config.proxy == "http://proxy.test:8080"


def test_create_provider_uses_openai_chat_openrouter_by_default():
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("open_router", _make_settings())

    assert isinstance(provider, OpenRouterProvider)


def test_create_provider_instantiates_each_builtin():
    settings = _make_settings(
        gemini_api_key="test_gemini_key",
        vertex_project_id="test-vertex-project",
        groq_api_key="test_groq_key",
        cline_api_key="test_cline_key",
        cerebras_api_key="test_cerebras_key",
        fireworks_api_key="test_fireworks_key",
        novita_api_key="test_novita_key",
        cloudflare_api_token="test_cloudflare_token",
        cloudflare_account_id="test_cloudflare_account",
        vercel_ai_gateway_api_key="test_vercel_key",
        bedrock_api_key="test_bedrock_key",
        huggingface_api_key="test_huggingface_key",
        cohere_api_key="test_cohere_key",
        kimi_api_key="test_kimi_key",
        kimi_code_api_key="test_kimi_code_key",
        provider_rate_limit=7,
        provider_rate_window=11,
        provider_max_concurrency=3,
        sambanova_api_key="test_sambanova_key",
    )
    cases = {
        "nvidia_nim": NvidiaNimProvider,
        "openai": OpenAICodexProvider,
        "github_copilot": GitHubCopilotProvider,
        "cline_pass": OpenAIChatProvider,
        "xai": OpenAIChatProvider,
        "qwencloud": OpenAIChatProvider,
        "qwencloud_coding": OpenAIChatProvider,
        "together": OpenAIChatProvider,
        "deepinfra": OpenAIChatProvider,
        "siliconflow": OpenAIChatProvider,
        "nebius": OpenAIChatProvider,
        "chutes": OpenAIChatProvider,
        "featherless": OpenAIChatProvider,
        "azure_openai": OpenAIChatProvider,
        "open_router": OpenRouterProvider,
        "mistral": MistralProvider,
        "mistral_codestral": OpenAIChatProvider,
        "deepseek": DeepSeekProvider,
        "kimi": OpenAIChatProvider,
        "kimi_code": OpenAIChatProvider,
        "minimax": OpenAIChatProvider,
        "fireworks": OpenAIChatProvider,
        "novita": OpenAIChatProvider,
        "cloudflare": CloudflareProvider,
        "lmstudio": LMStudioProvider,
        "llamacpp": OpenAIChatProvider,
        "ollama": OpenAIChatProvider,
        "ollama_cloud": OpenAIChatProvider,
        "wafer": OpenAIChatProvider,
        "opencode_zen": OpenCodeProvider,
        "poolside": OpenAIChatProvider,
        "llm7": OpenAIChatProvider,
        "opencode_go": OpenCodeProvider,
        "vercel": OpenAIChatProvider,
        "bedrock": OpenAIChatProvider,
        "huggingface": OpenAIChatProvider,
        "cohere": OpenAIChatProvider,
        "zai": OpenAIChatProvider,
        "zai_api": OpenAIChatProvider,
        "tokenrouter": OpenAIChatProvider,
        "nararoute": OpenAIChatProvider,
        "xkiro": OpenAIChatProvider,
        "agnes": OpenAIChatProvider,
        "zenmux": OpenAIChatProvider,
        "wandb": OpenAIChatProvider,
        "gemini": GeminiProvider,
        "vertex": VertexProvider,
        "groq": GroqProvider,
        "sambanova": OpenAIChatProvider,
        "kilo": KiloProvider,
        "cerebras": OpenAIChatProvider,
    }
    sentinel_admission = MagicMock(spec=ProviderAdmissionController)
    auth = MagicMock()
    injected_factories = {
        "openai": lambda config, _settings, admission: OpenAICodexProvider(
            config,
            auth=auth,
            admission=admission,
        ),
        "github_copilot": lambda config, _settings, admission: GitHubCopilotProvider(
            config,
            auth=auth,
            admission=admission,
        ),
    }

    with (
        patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"),
        patch("free_claude_code.providers.github_copilot.provider.AsyncOpenAI"),
        patch("free_claude_code.providers.openai_codex.provider.AsyncOpenAI"),
        patch("httpx.AsyncClient"),
        patch(
            "free_claude_code.providers.runtime.factory.ProviderAdmissionController",
            return_value=sentinel_admission,
        ) as admission_factory,
    ):
        for provider_id, provider_cls in cases.items():
            provider = create_provider(
                provider_id,
                settings,
                injected_factories=injected_factories,
            )

            assert isinstance(provider, provider_cls)
            assert provider._admission is sentinel_admission
            admission_factory.assert_called_once_with(
                provider_name=provider_id,
                rate_limit=7,
                rate_window=11,
                max_concurrency=3,
            )
            admission_factory.reset_mock()

    assert set(cases) == set(PROVIDER_CATALOG)


def test_provider_runtime_caches_by_provider_id():
    runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        first = runtime.resolve_provider("nvidia_nim")
        second = runtime.resolve_provider("nvidia_nim")

    assert first is second


def test_provider_runtime_provider_owns_one_admission_controller() -> None:
    runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        first = runtime.resolve_provider("nvidia_nim")
        second = runtime.resolve_provider("nvidia_nim")

    assert isinstance(first, NvidiaNimProvider)
    assert isinstance(second, NvidiaNimProvider)
    assert first._admission is second._admission


def test_separate_provider_runtimes_never_share_admission_controllers() -> None:
    first_runtime = ProviderRuntime(_make_settings())
    second_runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        first = first_runtime.resolve_provider("nvidia_nim")
        second = second_runtime.resolve_provider("nvidia_nim")

    assert isinstance(first, NvidiaNimProvider)
    assert isinstance(second, NvidiaNimProvider)
    assert first is not second
    assert first._admission is not second._admission


def test_different_providers_have_independent_admission_controllers() -> None:
    runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        nim = runtime.resolve_provider("nvidia_nim")
        open_router = runtime.resolve_provider("open_router")

    assert isinstance(nim, NvidiaNimProvider)
    assert isinstance(open_router, OpenRouterProvider)
    assert nim._admission is not open_router._admission


def test_unknown_provider_raises_unknown_provider_type_error():
    with pytest.raises(UnknownProviderError, match="Unknown provider_type"):
        create_provider("unknown", _make_settings())


@pytest.mark.asyncio
async def test_provider_runtime_cleanup_runs_all_even_if_one_fails() -> None:
    """Successful providers leave the cache while failed providers remain retryable."""
    p1 = MagicMock()
    p1.cleanup = AsyncMock(side_effect=RuntimeError("first"))
    p2 = MagicMock()
    p2.cleanup = AsyncMock()
    runtime = ProviderRuntime(_make_settings(), {"a": p1, "b": p2})

    with pytest.raises(RuntimeError, match="first"):
        await runtime.cleanup()

    p1.cleanup.assert_awaited_once()
    p2.cleanup.assert_awaited_once()
    assert runtime.is_cached("a")
    assert not runtime.is_cached("b")

    p1.cleanup = AsyncMock()
    await runtime.cleanup()

    p1.cleanup.assert_awaited_once()
    assert not runtime.is_cached("a")


@pytest.mark.asyncio
async def test_cancelled_cleanup_retains_current_and_unvisited_providers() -> None:
    first = MagicMock()
    second = MagicMock()
    third = MagicMock()
    second_started = asyncio.Event()
    second_attempts = 0

    async def cleanup_second() -> None:
        nonlocal second_attempts
        second_attempts += 1
        if second_attempts == 1:
            second_started.set()
            await asyncio.Event().wait()

    first.cleanup = AsyncMock()
    second.cleanup = AsyncMock(side_effect=cleanup_second)
    third.cleanup = AsyncMock()
    runtime = ProviderRuntime(
        _make_settings(),
        {"first": first, "second": second, "third": third},
    )
    cleanup_task = asyncio.create_task(runtime.cleanup())
    await second_started.wait()

    cleanup_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup_task

    assert runtime.is_cached("first") is False
    assert runtime.is_cached("second") is True
    assert runtime.is_cached("third") is True
    first.cleanup.assert_awaited_once_with()
    third.cleanup.assert_not_awaited()

    await runtime.cleanup()

    first.cleanup.assert_awaited_once_with()
    assert second.cleanup.await_count == 2
    third.cleanup.assert_awaited_once_with()
    assert runtime.is_cached("first") is False
    assert runtime.is_cached("second") is False
    assert runtime.is_cached("third") is False


@pytest.mark.asyncio
async def test_provider_runtime_cleanup_exceptiongroup_on_multiple_failures() -> None:
    p1 = MagicMock()
    p1.cleanup = AsyncMock(side_effect=RuntimeError("a"))
    p2 = MagicMock()
    p2.cleanup = AsyncMock(side_effect=RuntimeError("b"))
    runtime = ProviderRuntime(_make_settings(), {"x": p1, "y": p2})

    with pytest.raises(ExceptionGroup) as exc_info:
        await runtime.cleanup()

    assert len(exc_info.value.exceptions) == 2
    assert runtime.is_cached("x")
    assert runtime.is_cached("y")

    p1.cleanup = AsyncMock()
    p2.cleanup = AsyncMock()
    await runtime.cleanup()

    assert not runtime.is_cached("x")
    assert not runtime.is_cached("y")

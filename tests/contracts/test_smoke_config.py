from pathlib import Path
from types import SimpleNamespace

from smoke.conftest import (
    DISABLED_PROVIDER_MODEL,
    provider_model_params,
    provider_xdist_group,
)
from smoke.lib.config import (
    ALL_TARGETS,
    DEFAULT_TARGETS,
    MISTRAL_REASONING_SMOKE_DEFAULT_MODEL,
    NVIDIA_NIM_CLI_DEFAULT_MODELS,
    OPENROUTER_FREE_CLI_DEFAULT_MODELS,
    OPT_IN_TARGETS,
    PROVIDER_SMOKE_DEFAULT_MODELS,
    TARGET_REQUIRED_ENV,
    ProviderModel,
    SmokeConfig,
    nvidia_nim_cli_model_refs,
    openrouter_free_cli_model_refs,
)


def _settings(**overrides):
    values = {
        "model": "ollama/llama3.1",
        "model_fable": None,
        "model_opus": None,
        "model_sonnet": None,
        "model_haiku": None,
        "azure_openai_api_key": "",
        "azure_openai_base_url": "",
        "nvidia_nim_api_key": "",
        "open_router_api_key": "",
        "mistral_api_key": "",
        "codestral_api_key": "",
        "deepseek_api_key": "",
        "kimi_api_key": "",
        "kimi_code_api_key": "",
        "wafer_api_key": "",
        "minimax_api_key": "",
        "opencode_api_key": "",
        "vercel_ai_gateway_api_key": "",
        "bedrock_api_key": "",
        "bedrock_base_url": "https://bedrock-mantle.us-east-1.api.aws/v1",
        "huggingface_api_key": "",
        "cohere_api_key": "",
        "github_models_token": "",
        "zai_api_key": "",
        "kilo_api_key": "",
        "gemini_api_key": "",
        "vertex_project_id": "",
        "vertex_location": "global",
        "groq_api_key": "",
        "cline_api_key": "",
        "xai_api_key": "",
        "qwencloud_api_key": "",
        "qwencloud_coding_api_key": "",
        "together_api_key": "",
        "deepinfra_api_key": "",
        "siliconflow_api_key": "",
        "nebius_api_key": "",
        "chutes_api_key": "",
        "featherless_api_key": "",
        "wandb_api_key": "",
        "sambanova_api_key": "",
        "cerebras_api_key": "",
        "ollama_api_key": "",
        "fireworks_api_key": "",
        "novita_api_key": "",
        "cloudflare_api_token": "",
        "cloudflare_account_id": "",
        "lm_studio_base_url": "",
        "llamacpp_base_url": "",
        "ollama_base_url": "http://localhost:11434",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _smoke_config(**overrides) -> SmokeConfig:
    values = {
        "root": Path("."),
        "results_dir": Path(".smoke-results"),
        "live": False,
        "interactive": False,
        "targets": DEFAULT_TARGETS,
        "provider_matrix": frozenset(),
        "timeout_s": 45.0,
        "prompt": "Reply with exactly: FCC_SMOKE_PONG",
        "claude_bin": "claude",
        "worker_id": "main",
        "settings": _settings(),
    }
    values.update(overrides)
    return SmokeConfig(**values)


def test_ollama_is_default_smoke_target() -> None:
    assert "ollama" in DEFAULT_TARGETS
    assert "ollama" in TARGET_REQUIRED_ENV


def test_nvidia_nim_cli_is_opt_in_smoke_target() -> None:
    assert "nvidia_nim_cli" not in DEFAULT_TARGETS
    assert "nvidia_nim_cli" in OPT_IN_TARGETS
    assert "nvidia_nim_cli" in ALL_TARGETS
    assert "nvidia_nim_cli" in TARGET_REQUIRED_ENV
    assert "openrouter_free_cli" not in DEFAULT_TARGETS
    assert "openrouter_free_cli" in OPT_IN_TARGETS
    assert "openrouter_free_cli" in ALL_TARGETS
    assert "openrouter_free_cli" in TARGET_REQUIRED_ENV


def test_ollama_provider_configuration_uses_base_url() -> None:
    config = _smoke_config()

    assert config.has_provider_configuration("ollama")
    assert config.provider_models()[0].full_model == "ollama/llama3.1"


def test_ollama_provider_matrix_filters_models() -> None:
    config = _smoke_config(provider_matrix=frozenset({"ollama"}))

    assert [model.provider for model in config.provider_models()] == ["ollama"]


def test_ollama_cloud_provider_configuration_uses_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_OLLAMA_CLOUD", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            ollama_api_key="ollama-cloud-key",
        )
    )

    assert config.has_provider_configuration("ollama_cloud")
    models = config.provider_smoke_models()
    assert [model.provider for model in models] == ["ollama_cloud"]
    assert models[0].full_model == "ollama_cloud/qwen3-coder:480b"
    assert models[0].source == "provider_default"


def test_provider_smoke_models_cover_configured_providers_independent_of_model_mapping(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_DEEPSEEK", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            deepseek_api_key="deepseek-key",
            ollama_base_url="",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["deepseek"]
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["deepseek"]
    assert models[0].source == "provider_default"


def test_xai_provider_smoke_uses_current_grok_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_XAI", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            xai_api_key="xai-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["xai"]
    assert models[0].full_model == "xai/grok-4.5"
    assert models[0].source == "provider_default"


def test_cline_pass_provider_smoke_uses_low_cost_subscription_model(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_CLINE_PASS", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            cline_api_key="cline-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["cline_pass"]
    assert models[0].full_model == "cline_pass/cline-pass/deepseek-v4-flash"
    assert models[0].source == "provider_default"


def test_cline_pass_smoke_override_accepts_full_or_upstream_model_ref(
    monkeypatch,
) -> None:
    settings = _settings(
        model="ollama/llama3.1",
        ollama_base_url="",
        cline_api_key="cline-key",
    )
    for override in (
        "cline_pass/cline-pass/kimi-k3",
        "cline-pass/kimi-k3",
    ):
        monkeypatch.setenv("FCC_SMOKE_MODEL_CLINE_PASS", override)
        models = _smoke_config(settings=settings).provider_smoke_models()

        assert [model.provider for model in models] == ["cline_pass"]
        assert models[0].full_model == "cline_pass/cline-pass/kimi-k3"
        assert models[0].source == "FCC_SMOKE_MODEL_CLINE_PASS"


def test_zai_shared_key_enables_both_provider_smoke_surfaces(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_ZAI", raising=False)
    monkeypatch.delenv("FCC_SMOKE_MODEL_ZAI_API", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            zai_api_key="shared-zai-key",
        )
    )

    models = config.provider_smoke_models()

    assert [(model.provider, model.full_model, model.source) for model in models] == [
        ("zai", "zai/glm-5.2", "provider_default"),
        ("zai_api", "zai_api/glm-4.7-flash", "provider_default"),
    ]


def test_zai_api_smoke_override_is_independent_from_coding_plan(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_ZAI", raising=False)
    monkeypatch.setenv("FCC_SMOKE_MODEL_ZAI_API", "zai_api/glm-5.2")
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            zai_api_key="shared-zai-key",
        )
    )

    models = config.provider_smoke_models()

    assert [(model.provider, model.full_model, model.source) for model in models] == [
        ("zai", "zai/glm-5.2", "provider_default"),
        ("zai_api", "zai_api/glm-5.2", "FCC_SMOKE_MODEL_ZAI_API"),
    ]


def test_qwencloud_provider_smoke_uses_current_coding_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_QWENCLOUD", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            qwencloud_api_key="qwencloud-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["qwencloud"]
    assert models[0].full_model == "qwencloud/qwen3.7-plus"
    assert models[0].source == "provider_default"


def test_qwencloud_coding_provider_smoke_uses_recommended_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_QWENCLOUD_CODING", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            qwencloud_coding_api_key="qwencloud-coding-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["qwencloud_coding"]
    assert models[0].full_model == "qwencloud_coding/qwen3.7-plus"
    assert models[0].source == "provider_default"


def test_qwencloud_coding_smoke_override_is_independent_from_token_plan(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_QWENCLOUD", raising=False)
    monkeypatch.setenv(
        "FCC_SMOKE_MODEL_QWENCLOUD_CODING",
        "qwencloud_coding/kimi-k2.5",
    )
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            qwencloud_api_key="qwencloud-key",
            qwencloud_coding_api_key="qwencloud-coding-key",
        )
    )

    models = config.provider_smoke_models()

    assert [(model.provider, model.full_model, model.source) for model in models] == [
        ("qwencloud", "qwencloud/qwen3.7-plus", "provider_default"),
        (
            "qwencloud_coding",
            "qwencloud_coding/kimi-k2.5",
            "FCC_SMOKE_MODEL_QWENCLOUD_CODING",
        ),
    ]


def test_together_provider_smoke_uses_current_coding_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_TOGETHER", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            together_api_key="together-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["together"]
    assert models[0].full_model == "together/zai-org/GLM-5.2"
    assert models[0].source == "provider_default"


def test_deepinfra_provider_smoke_uses_current_coding_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_DEEPINFRA", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            deepinfra_api_key="deepinfra-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["deepinfra"]
    assert models[0].full_model == "deepinfra/deepseek-ai/DeepSeek-V4-Flash"
    assert models[0].source == "provider_default"


def test_siliconflow_provider_smoke_uses_documented_chat_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_SILICONFLOW", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            siliconflow_api_key="siliconflow-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["siliconflow"]
    assert models[0].full_model == "siliconflow/Qwen/Qwen3-32B"
    assert models[0].source == "provider_default"


def test_nebius_provider_smoke_uses_documented_agent_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_NEBIUS", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            nebius_api_key="nebius-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["nebius"]
    assert models[0].full_model == "nebius/Qwen/Qwen3-30B-A3B"
    assert models[0].source == "provider_default"


def test_chutes_provider_smoke_uses_documented_agent_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_CHUTES", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            chutes_api_key="chutes-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["chutes"]
    assert models[0].full_model == "chutes/Qwen/Qwen3-32B-TEE"
    assert models[0].source == "provider_default"


def test_featherless_provider_smoke_uses_documented_agent_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_FEATHERLESS", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            featherless_api_key="featherless-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["featherless"]
    assert models[0].full_model == "featherless/Qwen/Qwen3-32B"
    assert models[0].source == "provider_default"


def test_agnes_provider_smoke_uses_documented_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_AGNES", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            agnes_api_key="agnes-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["agnes"]
    assert models[0].full_model == "agnes/agnes-2.0-flash"
    assert models[0].source == "provider_default"


def test_zenmux_provider_smoke_uses_current_free_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_ZENMUX", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            zenmux_api_key="zenmux-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["zenmux"]
    assert models[0].full_model == "zenmux/deepseek/deepseek-v4-flash-free"
    assert models[0].source == "provider_default"


def test_wandb_provider_smoke_uses_documented_tool_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_WANDB", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            wandb_api_key="wandb-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["wandb"]
    assert models[0].full_model == "wandb/openai/gpt-oss-20b"
    assert models[0].source == "provider_default"


def test_wandb_provider_smoke_honors_model_override(monkeypatch) -> None:
    monkeypatch.setenv("FCC_SMOKE_MODEL_WANDB", "deepseek-ai/DeepSeek-V4-Flash")
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            wandb_api_key="wandb-key",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["wandb"]
    assert models[0].full_model == "wandb/deepseek-ai/DeepSeek-V4-Flash"
    assert models[0].source == "FCC_SMOKE_MODEL_WANDB"


def test_connected_account_provider_smoke_requires_explicit_model(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_OPENAI", raising=False)
    config = _smoke_config(
        provider_matrix=frozenset({"openai"}),
        settings=_settings(ollama_base_url=""),
    )

    assert not config.has_provider_configuration("openai")
    assert config.provider_smoke_models() == []

    monkeypatch.setenv("FCC_SMOKE_MODEL_OPENAI", "gpt-5.3-codex")

    assert config.has_provider_configuration("openai")
    assert config.provider_smoke_models() == [
        ProviderModel(
            provider="openai",
            full_model="openai/gpt-5.3-codex",
            source="FCC_SMOKE_MODEL_OPENAI",
        )
    ]


def test_openrouter_provider_smoke_uses_concrete_free_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_OPEN_ROUTER", raising=False)
    config = _smoke_config(
        settings=_settings(open_router_api_key="openrouter-key", ollama_base_url="")
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["open_router"]
    assert models[0].full_model == "open_router/moonshotai/kimi-k2.6:free"
    assert models[0].source == "provider_default"


def test_kilo_provider_smoke_uses_concrete_free_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_KILO", raising=False)
    config = _smoke_config(
        settings=_settings(kilo_api_key="anonymous", ollama_base_url="")
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["kilo"]
    assert models[0].full_model == "kilo/kilo-auto/free"
    assert models[0].source == "provider_default"


def test_bedrock_provider_configuration_uses_official_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_BEDROCK", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            bedrock_api_key="bedrock-key",
        )
    )

    assert config.has_provider_configuration("bedrock")
    models = config.provider_smoke_models()
    assert [model.provider for model in models] == ["bedrock"]
    assert models[0].full_model == "bedrock/openai.gpt-oss-120b"
    assert models[0].source == "provider_default"


def test_azure_openai_provider_configuration_requires_key_and_resource_url(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_AZURE_OPENAI", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            azure_openai_api_key="azure-key",
            azure_openai_base_url=("https://resource.openai.azure.com/openai/v1/"),
        )
    )

    assert config.has_provider_configuration("azure_openai")
    models = config.provider_smoke_models()
    assert [model.provider for model in models] == ["azure_openai"]
    assert models[0].full_model == "azure_openai/gpt-5.1"
    assert models[0].source == "provider_default"

    config.settings.azure_openai_base_url = ""
    assert not config.has_provider_configuration("azure_openai")


def test_vertex_provider_configuration_uses_project_id(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_VERTEX", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            vertex_project_id="vertex-project",
        )
    )

    assert config.has_provider_configuration("vertex")
    models = config.provider_smoke_models()
    assert [model.provider for model in models] == ["vertex"]
    assert models[0].full_model == "vertex/google/gemini-3.5-flash"
    assert models[0].source == "provider_default"


def test_wafer_provider_configuration_uses_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_WAFER", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            wafer_api_key="wafer-key",
        )
    )

    assert config.has_provider_configuration("wafer")
    models = config.provider_smoke_models()
    assert models[0].provider == "wafer"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["wafer"]


def test_kimi_code_provider_configuration_uses_subscription_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_KIMI_CODE", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            kimi_code_api_key="subscription-key",
        )
    )

    assert config.has_provider_configuration("kimi_code")
    models = config.provider_smoke_models()
    assert [model.provider for model in models] == ["kimi_code"]
    assert models[0].full_model == "kimi_code/k3"
    assert models[0].source == "provider_default"


def test_minimax_provider_configuration_uses_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_MINIMAX", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            minimax_api_key="minimax-key",
        )
    )

    assert config.has_provider_configuration("minimax")
    models = config.provider_smoke_models()
    assert models[0].provider == "minimax"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["minimax"]


def test_cloudflare_provider_configuration_requires_token_and_account(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_CLOUDFLARE", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            cloudflare_api_token="cf-token",
            cloudflare_account_id="cf-account",
        )
    )

    assert config.has_provider_configuration("cloudflare")
    models = config.provider_smoke_models()
    assert models[0].provider == "cloudflare"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["cloudflare"]


def test_cloudflare_provider_configuration_missing_account_is_unconfigured() -> None:
    config = _smoke_config(
        settings=_settings(
            ollama_base_url="",
            cloudflare_api_token="cf-token",
            cloudflare_account_id="",
        )
    )

    assert not config.has_provider_configuration("cloudflare")


def test_vercel_provider_configuration_uses_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_VERCEL", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            vercel_ai_gateway_api_key="vercel-key",
        )
    )

    assert config.has_provider_configuration("vercel")
    models = config.provider_smoke_models()
    assert models[0].provider == "vercel"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["vercel"]


def test_huggingface_provider_configuration_uses_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_HUGGINGFACE", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            huggingface_api_key="hf-key",
        )
    )

    assert config.has_provider_configuration("huggingface")
    models = config.provider_smoke_models()
    assert models[0].provider == "huggingface"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["huggingface"]


def test_cohere_provider_configuration_uses_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_COHERE", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            cohere_api_key="cohere-key",
        )
    )

    assert config.has_provider_configuration("cohere")
    models = config.provider_smoke_models()
    assert models[0].provider == "cohere"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["cohere"]


def test_github_models_provider_configuration_uses_token(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_GITHUB_MODELS", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            github_models_token="github-token",
        )
    )

    assert config.has_provider_configuration("github_models")
    models = config.provider_smoke_models()
    assert models[0].provider == "github_models"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["github_models"]


def test_sambanova_provider_configuration_uses_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_SAMBANOVA", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            sambanova_api_key="sambanova-key",
        )
    )

    assert config.has_provider_configuration("sambanova")
    models = config.provider_smoke_models()
    assert models[0].provider == "sambanova"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["sambanova"]


def test_provider_smoke_model_override_accepts_model_name_without_prefix(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FCC_SMOKE_MODEL_DEEPSEEK", "deepseek-reasoner")
    config = _smoke_config(
        settings=_settings(
            deepseek_api_key="deepseek-key",
            ollama_base_url="",
        ),
        provider_matrix=frozenset({"deepseek"}),
    )

    models = config.provider_smoke_models()

    assert models[0].full_model == "deepseek/deepseek-reasoner"
    assert models[0].source == "FCC_SMOKE_MODEL_DEEPSEEK"


def test_provider_smoke_model_override_accepts_owner_model_name(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "FCC_SMOKE_MODEL_NVIDIA_NIM", "nvidia/nemotron-3-super-120b-a12b"
    )
    config = _smoke_config(
        settings=_settings(
            model="deepseek/deepseek-chat",
            deepseek_api_key="",
            nvidia_nim_api_key="nim-key",
            ollama_base_url="",
        ),
        provider_matrix=frozenset({"nvidia_nim"}),
    )

    models = config.provider_smoke_models()

    assert models[0].full_model == "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
    assert models[0].source == "FCC_SMOKE_MODEL_NVIDIA_NIM"


def test_provider_smoke_model_override_preserves_namespaced_upstream_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FCC_SMOKE_MODEL_DEEPSEEK", "ollama/llama3.1")
    config = _smoke_config(
        settings=_settings(
            deepseek_api_key="deepseek-key",
            ollama_base_url="",
        ),
        provider_matrix=frozenset({"deepseek"}),
    )

    models = config.provider_smoke_models()

    assert models[0].full_model == "deepseek/ollama/llama3.1"


def test_mistral_reasoning_smoke_uses_reasoning_default(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_MISTRAL_REASONING", raising=False)
    config = _smoke_config(
        settings=_settings(mistral_api_key="mistral-key", ollama_base_url="")
    )

    model = config.mistral_reasoning_smoke_model()

    assert model is not None
    assert model.provider == "mistral"
    assert model.full_model == MISTRAL_REASONING_SMOKE_DEFAULT_MODEL
    assert model.source == "mistral_reasoning_default"


def test_mistral_reasoning_smoke_accepts_override(monkeypatch) -> None:
    monkeypatch.setenv("FCC_SMOKE_MODEL_MISTRAL_REASONING", "mistral-medium-3-5")
    config = _smoke_config(
        settings=_settings(mistral_api_key="mistral-key", ollama_base_url="")
    )

    model = config.mistral_reasoning_smoke_model()

    assert model is not None
    assert model.full_model == "mistral/mistral-medium-3-5"
    assert model.source == "FCC_SMOKE_MODEL_MISTRAL_REASONING"


def test_mistral_reasoning_smoke_respects_provider_matrix(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_MISTRAL_REASONING", raising=False)
    config = _smoke_config(
        settings=_settings(mistral_api_key="mistral-key", ollama_base_url=""),
        provider_matrix=frozenset({"deepseek"}),
    )

    assert config.mistral_reasoning_smoke_model() is None


def test_provider_smoke_matrix_filters_provider_catalog(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_DEEPSEEK", raising=False)
    config = _smoke_config(
        settings=_settings(
            deepseek_api_key="deepseek-key",
            nvidia_nim_api_key="nim-key",
            ollama_base_url="",
        ),
        provider_matrix=frozenset({"nvidia_nim"}),
    )

    assert [model.provider for model in config.provider_smoke_models()] == [
        "nvidia_nim"
    ]


def test_provider_smoke_collection_params_are_grouped_by_provider(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_DEEPSEEK", raising=False)
    monkeypatch.delenv("FCC_SMOKE_MODEL_NVIDIA_NIM", raising=False)
    config = _smoke_config(
        live=True,
        settings=_settings(
            deepseek_api_key="deepseek-key",
            nvidia_nim_api_key="nim-key",
            ollama_base_url="",
        ),
    )

    params = provider_model_params(config)

    assert [param.id for param in params] == ["nvidia_nim", "deepseek"]
    groups = [
        mark.args[0]
        for param in params
        for mark in param.marks
        if mark.name == "xdist_group"
    ]
    assert groups == ["provider:nvidia_nim", "provider:deepseek"]


def test_provider_smoke_collection_uses_disabled_placeholder_when_not_live() -> None:
    config = _smoke_config(live=False, settings=_settings(ollama_base_url=""))

    params = provider_model_params(config)

    assert [param.values[0] for param in params] == [DISABLED_PROVIDER_MODEL]
    assert provider_xdist_group(DISABLED_PROVIDER_MODEL) == "provider:smoke_disabled"


def test_provider_smoke_includes_local_provider_when_model_mapping_uses_it(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_OLLAMA", raising=False)
    config = _smoke_config()

    assert [model.provider for model in config.provider_smoke_models()] == ["ollama"]


def test_provider_smoke_does_not_include_default_local_urls_when_unmapped(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_OLLAMA", raising=False)
    config = _smoke_config(settings=_settings(model="nvidia_nim/test"))

    assert config.provider_smoke_models() == []


def test_nvidia_nim_cli_default_models_are_normalized() -> None:
    refs = nvidia_nim_cli_model_refs({})

    assert tuple(refs) == tuple(
        f"nvidia_nim/{model}" for model in NVIDIA_NIM_CLI_DEFAULT_MODELS
    )
    assert "nvidia_nim/deepseek-ai/deepseek-v4-pro" in refs
    assert "nvidia_nim/deepseek-ai/deepseek-v4-flash" in refs
    assert set(refs.values()) == {"nvidia_nim_cli_default"}


def test_nvidia_nim_cli_models_override_and_append() -> None:
    refs = nvidia_nim_cli_model_refs(
        {
            "FCC_SMOKE_NIM_MODELS": "z-ai/glm-5.2,nvidia_nim/custom/model",
            "FCC_SMOKE_NIM_EXTRA_MODELS": "moonshotai/kimi-k2.6,z-ai/glm-5.2",
        }
    )

    assert tuple(refs) == (
        "nvidia_nim/z-ai/glm-5.2",
        "nvidia_nim/custom/model",
        "nvidia_nim/moonshotai/kimi-k2.6",
    )
    assert refs["nvidia_nim/z-ai/glm-5.2"] == "FCC_SMOKE_NIM_MODELS"
    assert refs["nvidia_nim/moonshotai/kimi-k2.6"] == ("FCC_SMOKE_NIM_EXTRA_MODELS")


def test_nvidia_nim_cli_models_reject_empty_override() -> None:
    try:
        nvidia_nim_cli_model_refs({"FCC_SMOKE_NIM_MODELS": " , "})
    except ValueError as exc:
        assert "FCC_SMOKE_NIM_MODELS" in str(exc)
    else:
        raise AssertionError("expected empty NVIDIA NIM CLI model override to fail")


def test_nvidia_nim_cli_models_preserve_namespaced_upstream_model() -> None:
    refs = nvidia_nim_cli_model_refs({"FCC_SMOKE_NIM_MODELS": "open_router/model"})

    assert refs == {
        "nvidia_nim/open_router/model": "FCC_SMOKE_NIM_MODELS",
    }


def test_smoke_config_returns_nvidia_nim_cli_provider_models(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_NIM_MODELS", raising=False)
    monkeypatch.delenv("FCC_SMOKE_NIM_EXTRA_MODELS", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="nvidia_nim/z-ai/glm-5.2",
            nvidia_nim_api_key="nim-key",
            ollama_base_url="",
        )
    )

    models = config.nvidia_nim_cli_models()

    assert models[0].provider == "nvidia_nim"
    assert models[0].full_model == "nvidia_nim/z-ai/glm-5.2"
    assert models[0].source == "nvidia_nim_cli_default"


def test_openrouter_free_cli_default_models_are_normalized() -> None:
    refs = openrouter_free_cli_model_refs({})

    assert tuple(refs) == tuple(
        f"open_router/{model}" for model in OPENROUTER_FREE_CLI_DEFAULT_MODELS
    )
    assert "open_router/nvidia/nemotron-3-super-120b-a12b:free" in refs
    assert "open_router/poolside/laguna-m.1:free" in refs
    assert set(refs.values()) == {"openrouter_free_cli_default"}


def test_openrouter_free_cli_models_override_and_append() -> None:
    refs = openrouter_free_cli_model_refs(
        {
            "FCC_SMOKE_OPENROUTER_FREE_MODELS": (
                "openai/gpt-oss-120b:free,open_router/custom/model:free"
            ),
            "FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS": (
                "poolside/laguna-m.1:free,openai/gpt-oss-120b:free"
            ),
        }
    )

    assert tuple(refs) == (
        "open_router/openai/gpt-oss-120b:free",
        "open_router/custom/model:free",
        "open_router/poolside/laguna-m.1:free",
    )
    assert refs["open_router/openai/gpt-oss-120b:free"] == (
        "FCC_SMOKE_OPENROUTER_FREE_MODELS"
    )
    assert refs["open_router/poolside/laguna-m.1:free"] == (
        "FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS"
    )


def test_openrouter_free_cli_models_reject_empty_override() -> None:
    try:
        openrouter_free_cli_model_refs({"FCC_SMOKE_OPENROUTER_FREE_MODELS": " , "})
    except ValueError as exc:
        assert "FCC_SMOKE_OPENROUTER_FREE_MODELS" in str(exc)
    else:
        raise AssertionError("expected empty OpenRouter free CLI override to fail")


def test_openrouter_free_cli_models_preserve_namespaced_upstream_model() -> None:
    refs = openrouter_free_cli_model_refs(
        {"FCC_SMOKE_OPENROUTER_FREE_MODELS": "nvidia_nim/model"}
    )

    assert refs == {
        "open_router/nvidia_nim/model": "FCC_SMOKE_OPENROUTER_FREE_MODELS",
    }


def test_smoke_config_returns_openrouter_free_cli_provider_models(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_OPENROUTER_FREE_MODELS", raising=False)
    monkeypatch.delenv("FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="open_router/openai/gpt-oss-120b:free",
            open_router_api_key="openrouter-key",
            ollama_base_url="",
        )
    )

    models = config.openrouter_free_cli_models()

    assert models[0].provider == "open_router"
    assert models[0].full_model == "open_router/nvidia/nemotron-3-super-120b-a12b:free"
    assert models[0].source == "openrouter_free_cli_default"

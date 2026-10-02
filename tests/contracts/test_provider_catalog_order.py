"""Freeze ``PROVIDER_CATALOG`` insertion order used as canonical provider ranking."""

from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
)

_EXPECTED_PROVIDER_ORDER: tuple[str, ...] = (
    "nvidia_nim",
    "open_router",
    "groq",
    "cline_pass",
    "openai",
    "xai",
    "qwencloud",
    "qwencloud_coding",
    "together",
    "deepinfra",
    "siliconflow",
    "nebius",
    "chutes",
    "featherless",
    "agnes",
    "zenmux",
    "wandb",
    "azure_openai",
    "gemini",
    "vertex",
    "deepseek",
    "mistral",
    "mistral_codestral",
    "opencode_zen",
    "opencode_go",
    "vercel",
    "bedrock",
    "huggingface",
    "cohere",
    "github_models",
    "wafer",
    "kimi",
    "kimi_code",
    "kilo",
    "minimax",
    "cerebras",
    "sambanova",
    "fireworks",
    "novita",
    "cloudflare",
    "zai",
    "zai_api",
    "tokenrouter",
    "nararoute",
    "ollama_cloud",
    "lmstudio",
    "llamacpp",
    "ollama",
)


def test_provider_catalog_key_order_matches_canonical_plan() -> None:
    """Lead with NIM, OpenRouter, and Groq while preserving catalog groupings."""

    assert tuple(PROVIDER_CATALOG.keys()) == _EXPECTED_PROVIDER_ORDER
    assert SUPPORTED_PROVIDER_IDS == _EXPECTED_PROVIDER_ORDER

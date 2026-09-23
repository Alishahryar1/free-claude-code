import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import ConversationDriver, SmokeServerDriver, assert_product_stream
from smoke.lib.local_providers import first_local_provider_model_id

pytestmark = [pytest.mark.live]


@pytest.mark.smoke_target("lmstudio")
def test_lmstudio_messages_e2e(smoke_config: SmokeConfig) -> None:
    _local_provider_messages_e2e(
        smoke_config,
        provider="lmstudio",
        base_url=smoke_config.settings.lm_studio_base_url,
    )


@pytest.mark.smoke_target("llamacpp")
def test_llamacpp_openai_chat_e2e(smoke_config: SmokeConfig) -> None:
    _local_provider_messages_e2e(
        smoke_config,
        provider="llamacpp",
        base_url=smoke_config.settings.llamacpp_base_url,
    )


@pytest.mark.smoke_target("ollama")
def test_ollama_openai_chat_e2e(smoke_config: SmokeConfig) -> None:
    _local_provider_messages_e2e(
        smoke_config,
        provider="ollama",
        base_url=smoke_config.settings.ollama_base_url,
    )


def _local_provider_messages_e2e(
    smoke_config: SmokeConfig,
    *,
    provider: str,
    base_url: str,
) -> None:
    model_id = first_local_provider_model_id(
        provider,
        base_url,
        timeout_s=smoke_config.timeout_s,
    )

    with SmokeServerDriver(
        smoke_config,
        name=f"product-{provider}-messages",
        env_overrides={"MODEL": f"{provider}/{model_id}", "MESSAGING_PLATFORM": "none"},
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).ask(
            "Reply with one short sentence."
        )

    assert_product_stream(turn.events)

import time

import httpx
import pytest

from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    text_content,
    thinking_content,
)
from smoke.lib.config import ProviderModel, SmokeConfig, auth_headers
from smoke.lib.e2e import ProviderMatrixDriver
from smoke.lib.http import collect_message_stream, message_payload
from smoke.lib.server import start_server
from smoke.lib.skips import (
    skip_if_upstream_unavailable_events,
    skip_if_upstream_unavailable_exception,
)

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("providers")]


def test_model_mapping_configuration_is_consistent(smoke_config: SmokeConfig) -> None:
    models = smoke_config.provider_models()
    if not models:
        pytest.skip("no configured provider models with usable credentials/base URLs")
    for provider_model in models:
        assert "/" in provider_model.full_model
        assert provider_model.model_name


def test_mixed_provider_model_mapping_when_configured(
    smoke_config: SmokeConfig,
) -> None:
    models = smoke_config.provider_models()
    providers = {provider_model.provider for provider_model in models}
    if len(providers) < 2:
        pytest.skip("configure MODEL_* with at least two provider prefixes")

    sources = {provider_model.source for provider_model in models}
    assert sources <= {
        "MODEL",
        "MODEL_FABLE",
        "MODEL_OPUS",
        "MODEL_SONNET",
        "MODEL_HAIKU",
    }
    assert len(providers) >= 2


def test_configured_provider_models_stream_successfully(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    try:
        with start_server(
            smoke_config,
            env_overrides={
                "MODEL": provider_model.full_model,
                "MESSAGING_PLATFORM": "none",
            },
            name=f"provider-{provider_model.provider}",
        ) as server:
            events = collect_message_stream(
                server,
                message_payload(smoke_config.prompt, model="fcc-smoke-default"),
                smoke_config,
            )
            skip_if_upstream_unavailable_events(events)
            assert_anthropic_stream_contract(events)
            has_text = bool(text_content(events).strip())
            has_thinking = bool(thinking_content(events).strip())
            assert has_text or has_thinking, (
                "provider returned no visible text or thinking content"
            )
    except Exception as exc:
        skip_if_upstream_unavailable_exception(exc)
        raise AssertionError(
            f"{provider_model.source}={provider_model.full_model}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


@pytest.mark.smoke_target("rate_limit")
def test_client_disconnect_mid_stream_does_not_crash_server(
    smoke_config: SmokeConfig,
) -> None:
    provider_model = ProviderMatrixDriver(smoke_config).first_model()

    with start_server(
        smoke_config,
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
        name="disconnect",
    ) as server:
        with httpx.stream(
            "POST",
            f"{server.base_url}/v1/messages",
            headers=auth_headers(),
            json=message_payload(smoke_config.prompt, model="fcc-smoke-default"),
            timeout=smoke_config.timeout_s,
        ) as response:
            assert response.status_code == 200, response.read()
            for _line in response.iter_lines():
                break

        time.sleep(0.5)
        health = httpx.get(f"{server.base_url}/health", timeout=5)
        assert health.status_code == 200

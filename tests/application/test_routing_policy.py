import json

import pytest
from pydantic import ValidationError

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.model_refs import ModelFallback
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import Message, MessagesRequest
from free_claude_code.core.openai_responses import OpenAIResponsesRequest


def candidate(model="ollama/first", **updates):
    return {
        "model": model,
        "context_tokens": 32768,
        "supports_tools": True,
        "supports_thinking": True,
        **updates,
    }


def settings(mode="auto", candidates=None, **updates):
    return Settings.model_validate(
        {
            "MODEL": "groq/legacy",
            "ROUTING_MODE": mode,
            "ROUTING_CANDIDATES": json.dumps(
                [candidate()] if candidates is None else candidates
            ),
            **updates,
        }
    )


def message(model="claude-sonnet-4"):
    return MessagesRequest(
        model=model,
        max_tokens=100,
        messages=[Message(role="user", content="private request")],
    )


@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_auto_selects_without_changing_native_request_protocol(protocol):
    router = ModelRouter(settings())
    original = (
        message()
        if protocol == "messages"
        else OpenAIResponsesRequest(
            model="claude-sonnet-4", input="private request", max_output_tokens=100
        )
    )
    routed = (
        router.resolve_messages_request(original)
        if isinstance(original, MessagesRequest)
        else router.resolve_responses_request(original)
    )
    assert routed.request.model == "first"
    assert type(routed.request) is type(original)
    assert original.model == "claude-sonnet-4"
    assert routed.resolved.policy_reason == "ordered_candidates"


def test_off_keeps_upstream_fallback_behavior():
    route = ModelRouter(
        settings(mode="off", MODEL_FALLBACKS=(ModelFallback(model="groq/backup"),))
    ).resolve("groq/pinned")
    assert route.primary.provider_model == "pinned"
    assert len(route.fallbacks) == 1


@pytest.mark.parametrize("mode", ["shadow", "auto"])
@pytest.mark.parametrize("model", ["ollama/pinned", "anthropic/ollama/pinned"])
def test_enabled_policy_pins_explicit_model_without_configured_fallbacks(mode, model):
    router = ModelRouter(
        settings(mode=mode, MODEL_FALLBACKS=(ModelFallback(model="groq/backup"),))
    )
    routed = router.resolve_messages_request(message(model))
    assert routed.request.model == "pinned"
    assert routed.resolved.fallbacks == ()
    assert routed.resolved.policy_reason == "explicit_model"


def test_shadow_preserves_upstream_route_and_records_proposal():
    routed = ModelRouter(settings(mode="shadow")).resolve_messages_request(message())
    assert routed.request.model == "legacy"
    assert routed.resolved.proposed_model == "ollama/first"


@pytest.mark.parametrize(
    "overrides",
    [
        {"supports_tools": None},
        {"supports_thinking": None},
        {"context_tokens": 1},
    ],
)
def test_unknown_or_insufficient_capabilities_fail_closed(overrides):
    with pytest.raises(InvalidRequestError, match="eligible"):
        ModelRouter(
            settings(candidates=[candidate(**overrides)])
        ).resolve_messages_request(message())


@pytest.mark.parametrize("protocol", ["messages", "responses"])
def test_auto_rejects_uninspected_extra_body_overrides(protocol):
    router = ModelRouter(settings())
    extra = {"model": "unapproved", "max_tokens": 100000}
    with pytest.raises(InvalidRequestError, match="extra_body"):
        if protocol == "messages":
            router.resolve_messages_request(
                message().model_copy(update={"extra_body": extra})
            )
        else:
            router.resolve_responses_request(
                OpenAIResponsesRequest.model_validate(
                    {
                        "model": "claude-sonnet-4",
                        "input": "hello",
                        "max_output_tokens": 100,
                        "extra_body": extra,
                    }
                )
            )


@pytest.mark.parametrize(
    "url",
    [
        "https://remote.example",
        "http://localhost.evil.example",
        "http://localhost@remote.example",
        "not-a-url",
    ],
)
def test_local_only_checks_actual_endpoint(url):
    with pytest.raises(InvalidRequestError, match="eligible"):
        ModelRouter(settings(OLLAMA_BASE_URL=url)).resolve_messages_request(message())


def test_cloud_candidate_requires_opt_in_and_current_tuple_credentials():
    candidates = [candidate("deepseek/cloud"), candidate()]
    local = ModelRouter(settings(candidates=candidates, DEEPSEEK_API_KEY=["test-only"]))
    assert local.resolve_messages_request(message()).request.model == "first"
    cloud = ModelRouter(
        settings(
            candidates=candidates,
            ROUTING_PROFILE="allow-cloud",
            DEEPSEEK_API_KEY=["test-only"],
        )
    )
    assert cloud.resolve_messages_request(message()).request.model == "cloud"


def test_missing_cloud_credentials_are_ineligible():
    with pytest.raises(InvalidRequestError, match="eligible"):
        ModelRouter(
            settings(
                candidates=[candidate("deepseek/cloud")],
                ROUTING_PROFILE="allow-cloud",
            )
        ).resolve_messages_request(message())


def test_policy_limits_alternates_and_does_not_append_global_fallbacks():
    route = (
        ModelRouter(
            settings(
                candidates=[
                    candidate(),
                    candidate("ollama/second"),
                    candidate("ollama/third"),
                ],
                ROUTING_FAILOVER_ENABLED=True,
                MODEL_FALLBACKS=(ModelFallback(model="groq/unsafe"),),
            )
        )
        .resolve_messages_request(message())
        .resolved
    )
    assert [entry.target.provider_model for entry in route.fallbacks] == ["second"]


@pytest.mark.parametrize(
    "raw",
    [
        "not-json",
        "{}",
        json.dumps([candidate(), candidate()]),
        json.dumps([candidate("unknown/model")]),
        json.dumps([candidate(input_cost_per_million=1)]),
    ],
)
def test_policy_configuration_rejects_malformed_or_ambiguous_candidates(raw):
    with pytest.raises(ValidationError):
        settings(ROUTING_CANDIDATES=raw)


def test_local_policy_disables_environment_proxies_in_generation_config():
    from unittest.mock import patch

    from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
    from free_claude_code.providers.openai_chat.client import create_chat_client
    from free_claude_code.providers.runtime.config import build_provider_config

    config = build_provider_config(PROVIDER_CATALOG["ollama"], settings())
    assert config.trust_env is False
    with (
        patch(
            "free_claude_code.providers.openai_chat.client.DefaultAsyncHttpx2Client"
        ) as http,
        patch("free_claude_code.providers.openai_chat.client.AsyncOpenAI"),
    ):
        create_chat_client(config, base_url=config.base_url, provider_name="OLLAMA")
    assert http.call_args.kwargs["trust_env"] is False


def test_strict_request_client_disables_environment_proxies():
    from unittest.mock import MagicMock, patch

    from free_claude_code.core.stream_delivery import bind_strict_routing
    from free_claude_code.providers.endpoint_types import HttpEndpoint
    from free_claude_code.providers.openai_client import OpenAIRequestClient

    with (
        patch("free_claude_code.providers.openai_client.httpx2.AsyncClient") as http,
        bind_strict_routing(True),
    ):
        OpenAIRequestClient().for_endpoint(
            MagicMock(),
            HttpEndpoint(base_url="http://localhost", api_key="local", headers={}),
        )
    assert http.call_args.kwargs["trust_env"] is False

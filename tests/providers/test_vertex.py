"""Tests for Google Vertex AI authentication, discovery, and Chat Completions."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from google.auth.credentials import Credentials
from google.auth.exceptions import DefaultCredentialsError, TransportError
from google.auth.transport.requests import Request

from free_claude_code.application.errors import (
    ApplicationUnavailableError,
    InvalidRequestError,
)
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import VERTEX_AI_API_ROOT
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.vertex import VertexProvider
from free_claude_code.providers.vertex.auth import GoogleAccessTokenProvider
from free_claude_code.providers.vertex.endpoint import (
    vertex_openai_base_url,
    vertex_publisher_models_url,
    vertex_service_endpoint,
)
from free_claude_code.providers.vertex.models import extract_vertex_model_page
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    reasoning_for,
)

_PROJECT_ID = "my-project"
_GLOBAL_OPENAI_BASE = (
    "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/"
    "endpoints/openapi"
)
_GLOBAL_MODELS_URL = (
    "https://aiplatform.googleapis.com/v1beta1/publishers/google/models"
)


class FakeCredentials(Credentials):
    """Minimal mutable Google credentials for deterministic refresh tests."""

    def __init__(
        self,
        *,
        token: str | None = None,
        expired: bool = True,
        refresh_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.token = token
        self._expired = expired
        self._refresh_error = refresh_error
        self.refresh_count = 0
        self.refresh_request: object | None = None

    @property
    def expired(self) -> bool:
        return self._expired

    def refresh(self, request: object) -> None:
        self.refresh_request = request
        self.refresh_count += 1
        if self._refresh_error is not None:
            raise self._refresh_error
        self.token = "refreshed-token"
        self._expired = False


def _token_provider(
    credentials: FakeCredentials | None = None,
) -> GoogleAccessTokenProvider:
    value = credentials or FakeCredentials(token="access-token", expired=False)

    def loader() -> Credentials:
        return value

    return GoogleAccessTokenProvider(loader)


def _provider(
    *,
    location: str = "global",
    token_provider: GoogleAccessTokenProvider | None = None,
) -> VertexProvider:
    return VertexProvider(
        make_provider_config(api_key="", base_url=VERTEX_AI_API_ROOT),
        project_id=_PROJECT_ID,
        location=location,
        admission=immediate_admission(),
        access_token_provider=token_provider or _token_provider(),
    )


def _simulate_openai_sdk_wire_json(body: dict) -> dict:
    wire = {key: value for key, value in body.items() if key != "extra_body"}
    sdk_extra = body.get("extra_body")
    if isinstance(sdk_extra, dict):
        wire.update(sdk_extra)
    return wire


def _google_thinking_config(wire: dict) -> dict | None:
    literal_extra_body = wire.get("extra_body")
    if not isinstance(literal_extra_body, dict):
        return None
    google = literal_extra_body.get("google")
    if not isinstance(google, dict):
        return None
    thinking_config = google.get("thinking_config")
    return thinking_config if isinstance(thinking_config, dict) else None


@pytest.mark.parametrize(
    ("location", "service_endpoint"),
    [
        ("global", "https://aiplatform.googleapis.com"),
        ("us-central1", "https://us-central1-aiplatform.googleapis.com"),
    ],
)
def test_vertex_endpoints_use_global_or_regional_hosts(
    location: str, service_endpoint: str
) -> None:
    assert vertex_service_endpoint(location) == service_endpoint
    assert vertex_openai_base_url("project/name", location) == (
        f"{service_endpoint}/v1/projects/project%2Fname/locations/{location}/"
        "endpoints/openapi"
    )
    assert vertex_publisher_models_url(location) == (
        f"{service_endpoint}/v1beta1/publishers/google/models"
    )


@pytest.mark.parametrize("project_id", ["", "   "])
def test_vertex_endpoint_requires_project_id(project_id: str) -> None:
    with pytest.raises(ApplicationUnavailableError, match="VERTEX_PROJECT_ID"):
        vertex_openai_base_url(project_id, "global")


@pytest.mark.parametrize("location", ["", "US central1", "us/central1"])
def test_vertex_endpoint_rejects_unsafe_locations(location: str) -> None:
    with pytest.raises(ApplicationUnavailableError, match="VERTEX_LOCATION"):
        vertex_service_endpoint(location)


@pytest.mark.asyncio
async def test_access_token_provider_reuses_valid_token_without_refresh() -> None:
    credentials = FakeCredentials(token="cached-token", expired=False)
    token_provider = _token_provider(credentials)

    assert await token_provider() == "cached-token"
    assert await token_provider() == "cached-token"
    assert credentials.refresh_count == 0


@pytest.mark.asyncio
async def test_access_token_provider_coalesces_concurrent_refreshes() -> None:
    credentials = FakeCredentials()
    token_provider = _token_provider(credentials)

    tokens = await asyncio.gather(*(token_provider() for _ in range(10)))

    assert tokens == ["refreshed-token"] * 10
    assert credentials.refresh_count == 1


@pytest.mark.asyncio
async def test_access_token_refresh_uses_the_vertex_proxy() -> None:
    credentials = FakeCredentials()
    token_provider = GoogleAccessTokenProvider(
        lambda: credentials,
        proxy="socks5://proxy.test:1080",
    )

    assert await token_provider() == "refreshed-token"
    assert isinstance(credentials.refresh_request, Request)
    session = credentials.refresh_request.session
    assert session.proxies == {
        "http": "socks5://proxy.test:1080",
        "https": "socks5://proxy.test:1080",
    }


@pytest.mark.asyncio
async def test_missing_adc_is_non_retryable_authentication_failure() -> None:
    def missing_credentials() -> Credentials:
        raise DefaultCredentialsError("sensitive local path")

    token_provider = GoogleAccessTokenProvider(missing_credentials)

    with pytest.raises(ExecutionFailure) as exc_info:
        await token_provider()

    failure = exc_info.value
    assert failure.kind is FailureKind.AUTHENTICATION
    assert failure.status_code == 401
    assert failure.retryable is False
    assert "gcloud auth application-default login" in failure.message
    assert "sensitive local path" not in failure.message


@pytest.mark.asyncio
async def test_transient_adc_refresh_failure_is_retryable() -> None:
    credentials = FakeCredentials(
        refresh_error=TransportError("temporary auth service failure")
    )
    token_provider = _token_provider(credentials)

    with pytest.raises(ExecutionFailure) as exc_info:
        await token_provider()

    failure = exc_info.value
    assert failure.kind is FailureKind.UNAVAILABLE
    assert failure.status_code == 503
    assert failure.retryable is True
    assert "temporary auth service failure" not in failure.message


def test_vertex_provider_supplies_renewable_token_callback_to_openai() -> None:
    token_provider = _token_provider()
    with (
        patch(
            "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
        ) as openai_client,
        patch("free_claude_code.providers.vertex.client.httpx.AsyncClient"),
    ):
        provider = _provider(token_provider=token_provider)

    assert provider._provider_name == "VERTEX"
    assert provider._base_url == _GLOBAL_OPENAI_BASE
    assert openai_client.call_args.kwargs["api_key"] is token_provider
    assert openai_client.call_args.kwargs["default_headers"] == {
        "x-goog-user-project": _PROJECT_ID
    }


def test_vertex_request_uses_google_thinking_budget_without_named_effort() -> None:
    provider = _provider()
    request = make_messages_request(
        "google/gemini-3.5-flash",
        thinking={"type": "enabled", "budget_tokens": 2048},
    )

    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))

    assert body["model"] == "google/gemini-3.5-flash"
    assert "reasoning_effort" not in body
    assert body["extra_body"]["extra_body"]["google"]["thinking_config"] == {
        "include_thoughts": True,
        "thinking_budget": 2048,
    }


def test_vertex_request_maps_reasoning_off_to_zero_budget() -> None:
    provider = _provider()
    request = make_messages_request(
        "google/gemini-3.5-flash",
        thinking={"type": "disabled"},
    )

    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))

    assert body["extra_body"]["extra_body"]["google"]["thinking_config"] == {
        "thinking_budget": 0,
        "include_thoughts": False,
    }


@pytest.mark.parametrize(
    ("reasoning", "expected_thinking_config"),
    [
        (ReasoningPolicy.provider_default(), None),
        (ReasoningPolicy.off(), {"thinking_budget": 0, "include_thoughts": False}),
        (ReasoningPolicy.on(), {"include_thoughts": True}),
        (
            ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
            {"thinking_budget": 2048, "include_thoughts": True},
        ),
        (
            ReasoningPolicy.on(budget_tokens=777),
            {"thinking_budget": 777, "include_thoughts": True},
        ),
        (
            ReasoningPolicy.on(
                effort=ReasoningEffort.HIGH,
                budget_tokens=777,
            ),
            {"thinking_budget": 777, "include_thoughts": True},
        ),
    ],
)
def test_vertex_reasoning_has_one_google_wire_owner(
    reasoning: ReasoningPolicy,
    expected_thinking_config: dict | None,
) -> None:
    provider = _provider()

    body = provider._chat._build_request_body(
        make_messages_request("google/gemini", thinking=None),
        reasoning=reasoning,
    )
    wire = _simulate_openai_sdk_wire_json(body)

    assert "reasoning_effort" not in wire
    assert _google_thinking_config(wire) == expected_thinking_config


def test_vertex_preserves_caller_thinking_config_only_for_provider_default() -> None:
    provider = _provider()
    request = make_messages_request(
        "google/gemini",
        thinking=None,
        extra_body={
            "extra_body": {
                "google": {
                    "thinking_config": {
                        "thinking_level": "low",
                        "include_thoughts": False,
                    }
                }
            }
        },
    )

    body = provider._chat._build_request_body(
        request,
        reasoning=ReasoningPolicy.provider_default(),
    )
    wire = _simulate_openai_sdk_wire_json(body)

    assert _google_thinking_config(wire) == {
        "thinking_level": "low",
        "include_thoughts": False,
    }


def test_vertex_rejects_caller_thinking_config_with_fcc_reasoning_control() -> None:
    provider = _provider()
    request = make_messages_request(
        "google/gemini",
        thinking=None,
        extra_body={
            "extra_body": {"google": {"thinking_config": {"thinking_level": "low"}}}
        },
    )

    with pytest.raises(InvalidRequestError, match="thinking_config"):
        provider._chat._build_request_body(
            request,
            reasoning=ReasoningPolicy.on(effort=ReasoningEffort.HIGH),
        )


def test_vertex_model_page_translates_google_resource_names_generically() -> None:
    model_ids, page_token = extract_vertex_model_page(
        {
            "publisherModels": [
                {"name": "publishers/google/models/gemini-3.5-flash"},
                {"name": "publishers/acme/models/custom-chat"},
            ],
            "nextPageToken": "next-page",
        }
    )

    assert model_ids == frozenset({"google/gemini-3.5-flash", "acme/custom-chat"})
    assert page_token == "next-page"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"publisherModels": "not-a-list"},
        {"publisherModels": [{}]},
        {"publisherModels": [{"name": "models/missing-publisher"}]},
        {"publisherModels": [{"name": "publishers/google/models/   "}]},
        {"publisherModels": [], "nextPageToken": 123},
    ],
)
def test_vertex_model_page_rejects_malformed_responses(payload: object) -> None:
    with pytest.raises(ModelListResponseError, match="VERTEX model-list response"):
        extract_vertex_model_page(payload)


@pytest.mark.asyncio
async def test_vertex_model_discovery_follows_native_pagination() -> None:
    provider = _provider()
    responses = [
        httpx.Response(
            200,
            json={
                "publisherModels": [
                    {"name": "publishers/google/models/gemini-3.5-flash"}
                ],
                "nextPageToken": "page-2",
            },
            request=httpx.Request("GET", _GLOBAL_MODELS_URL),
        ),
        httpx.Response(
            200,
            json={
                "publisherModels": [{"name": "publishers/google/models/gemini-3.1-pro"}]
            },
            request=httpx.Request("GET", _GLOBAL_MODELS_URL),
        ),
    ]
    with patch.object(
        provider._model_list_client,
        "get",
        new_callable=AsyncMock,
        side_effect=responses,
    ) as get:
        model_infos = await provider.list_model_infos()

    assert model_infos == frozenset(
        {
            ProviderModelInfo("google/gemini-3.5-flash"),
            ProviderModelInfo("google/gemini-3.1-pro"),
        }
    )
    assert get.await_args_list[0].kwargs == {
        "params": None,
        "headers": {
            "Authorization": "Bearer access-token",
            "x-goog-user-project": _PROJECT_ID,
        },
    }
    assert get.await_args_list[1].kwargs == {
        "params": {"pageToken": "page-2"},
        "headers": {
            "Authorization": "Bearer access-token",
            "x-goog-user-project": _PROJECT_ID,
        },
    }
    assert all(response.is_closed for response in responses)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(
            200,
            json={"publisherModels": []},
            request=httpx.Request("GET", _GLOBAL_MODELS_URL),
        ),
        httpx.Response(
            200,
            content=b"not-json",
            request=httpx.Request("GET", _GLOBAL_MODELS_URL),
        ),
    ],
)
async def test_vertex_model_discovery_rejects_unusable_success_response(
    response: httpx.Response,
) -> None:
    provider = _provider()
    with (
        patch.object(
            provider._model_list_client,
            "get",
            new_callable=AsyncMock,
            return_value=response,
        ),
        pytest.raises(ModelListResponseError, match="VERTEX model-list response"),
    ):
        await provider.list_model_infos()

    assert response.is_closed


@pytest.mark.asyncio
async def test_vertex_model_discovery_rejects_repeated_page_token() -> None:
    provider = _provider()
    responses = [
        httpx.Response(
            200,
            json={"publisherModels": [], "nextPageToken": "same"},
            request=httpx.Request("GET", _GLOBAL_MODELS_URL),
        ),
        httpx.Response(
            200,
            json={"publisherModels": [], "nextPageToken": "same"},
            request=httpx.Request("GET", _GLOBAL_MODELS_URL),
        ),
    ]
    with (
        patch.object(
            provider._model_list_client,
            "get",
            new_callable=AsyncMock,
            side_effect=responses,
        ),
        pytest.raises(ModelListResponseError, match="repeated nextPageToken"),
    ):
        await provider.list_model_infos()

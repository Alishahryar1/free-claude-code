"""Tests for catalog-driven OpenCode Chat/Responses routing."""

import asyncio
import json
from collections.abc import Callable
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import httpx
import httpx2
import pytest
from jsonschema import Draft202012Validator
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.routing import (
    ProviderModelTarget,
    ResolvedModelRoute,
    RoutedMessagesRequest,
    RoutedResponsesRequest,
)
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningCapability,
)
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.opencode import (
    OpenCodeProvider,
    create_opencode_provider,
)
from free_claude_code.providers.opencode.catalog import (
    OPENCODE_CATALOG_URL,
    OpenCodeCatalog,
    OpenCodeUpstreamTransport,
    parse_open_code_catalog,
)
from tests.providers.support import (
    capture_openai_chat_wire_body,
    immediate_admission,
    make_provider_config,
    reasoning_for,
)


def _config():
    return make_provider_config(
        api_key="test_opencode_key",
        base_url="https://opencode.ai/zen/v1",
        rate_limit=100,
        rate_window=1,
    )


def _catalog_payload(
    models: dict[str, object] | None = None,
    *,
    provider_package: str | None = "@ai-sdk/openai-compatible",
    provider_key: str = "opencode",
) -> dict[str, object]:
    provider: dict[str, object] = {
        "models": models
        or {
            "chat-selector": {
                "id": "chat-upstream",
                "reasoning": False,
            },
            "responses-selector": {
                "id": "responses-upstream",
                "provider": {"npm": "@ai-sdk/openai"},
                "reasoning": True,
            },
        }
    }
    if provider_package is not None:
        provider["npm"] = provider_package
    return {provider_key: provider}


def _catalog_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _request(model: str, **overrides: object) -> MessagesRequest:
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 128,
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def _responses_request(
    model: str,
    **overrides: object,
) -> OpenAIResponsesRequest:
    payload: dict[str, object] = {
        "model": model,
        "input": "hello",
        "max_output_tokens": 128,
    }
    payload.update(overrides)
    return OpenAIResponsesRequest.model_validate(payload)


def _responses_event_stream(text: str) -> str:
    completed_response: dict[str, object] = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "background": False,
        "billing": {"payer": "developer"},
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": 128,
        "max_tool_calls": None,
        "model": "responses-upstream",
        "output": [],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": None,
        "reasoning": {"effort": None, "summary": None},
        "safety_identifier": None,
        "service_tier": "default",
        "store": False,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": 0,
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 3,
        },
        "user": None,
    }
    events: tuple[dict[str, object], ...] = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                **completed_response,
                "status": "in_progress",
                "output": [],
                "usage": None,
            },
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "item_text",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
            "logprobs": [],
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": completed_response,
        },
    )
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _chat_event_stream(text: str) -> str:
    chunks = (
        {
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "chat-upstream",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "chat-upstream",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        },
    )
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + (
        "data: [DONE]\n\n"
    )


def _provider_with_wire_transports(
    payload: dict[str, object],
    *,
    generation_response: Callable[[httpx2.Request], httpx2.Response] | None = None,
    max_attempts: int = 5,
) -> tuple[OpenCodeProvider, list[httpx2.Request], list[httpx.Request]]:
    generation_requests: list[httpx2.Request] = []
    catalog_requests: list[httpx.Request] = []

    def generation_handler(request: httpx2.Request) -> httpx2.Response:
        generation_requests.append(request)
        if generation_response is not None:
            return generation_response(request)
        if request.url.path.endswith("/responses"):
            body = _responses_event_stream("responses-ok")
        elif request.url.path.endswith("/chat/completions"):
            body = _chat_event_stream("chat-ok")
        else:
            raise AssertionError(f"unexpected generation path {request.url.path}")
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    def catalog_handler(request: httpx.Request) -> httpx.Response:
        catalog_requests.append(request)
        return httpx.Response(200, json=payload)

    generation_client = AsyncOpenAI(
        api_key="test_opencode_key",
        base_url="https://opencode.ai/zen/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(generation_handler)
        ),
    )
    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI",
        return_value=generation_client,
    ):
        provider = create_opencode_provider(
            "opencode_zen",
            _config(),
            immediate_admission(
                provider_name="opencode_zen",
                max_attempts=max_attempts,
            ),
            catalog_client=_catalog_client(catalog_handler),
        )
    return provider, generation_requests, catalog_requests


async def _collect(provider: OpenCodeProvider, model: str, **overrides: object) -> str:
    return "".join(
        [
            chunk
            async for chunk in provider.stream_messages(
                _request(model, **overrides),
                input_tokens=2,
                request_id="req_opencode",
            )
        ]
    )


async def _collect_responses(
    provider: OpenCodeProvider,
    model: str,
    **overrides: object,
) -> str:
    return "".join(
        [
            chunk
            async for chunk in provider.stream_responses(
                _responses_request(model, **overrides),
                input_tokens=2,
                request_id="req_opencode_responses",
            )
        ]
    )


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_client_identifies_as_first_party_opencode_user_agent(
    provider_id: str,
) -> None:
    with (
        patch(
            "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
        ) as mock_openai,
        patch("httpx.AsyncClient"),
    ):
        create_opencode_provider(
            provider_id,
            _config(),
            immediate_admission(provider_name=provider_id),
        )

    assert mock_openai.call_args.kwargs["default_headers"] == {"User-Agent": "opencode"}


def test_catalog_resolves_package_precedence_status_alias_and_reasoning() -> None:
    snapshot = parse_open_code_catalog(
        _catalog_payload(
            {
                "provider-default": {
                    "id": "provider-default",
                    "reasoning": False,
                    "modalities": {"input": ["text"]},
                    "limit": {"context": 131072, "output": 8192},
                },
                "responses-alias": {
                    "id": "actual-responses-id",
                    "provider": {"npm": "@ai-sdk/openai"},
                    "status": "beta",
                    "reasoning": True,
                    "modalities": {"input": ["text", "image"]},
                },
                "anthropic": {
                    "id": "anthropic-id",
                    "provider": {"npm": "@ai-sdk/anthropic"},
                },
                "google": {
                    "id": "google-id",
                    "provider": {"npm": "@ai-sdk/google"},
                },
                "unknown-package": {
                    "id": "unknown-id",
                    "provider": {"npm": "@vendor/future"},
                },
                "alpha": {"id": "alpha-id", "status": "alpha"},
                "deprecated": {"id": "old-id", "status": "deprecated"},
            }
        ),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
    )

    assert set(snapshot.routes) == {
        "provider-default",
        "responses-alias",
        "anthropic",
        "google",
        "unknown-package",
    }
    provider_default = snapshot.route("provider-default")
    assert provider_default is not None
    assert provider_default.transport is OpenCodeUpstreamTransport.CHAT_COMPLETIONS
    responses = snapshot.route("responses-alias")
    assert responses is not None
    assert responses.upstream_model_id == "actual-responses-id"
    assert responses.transport is OpenCodeUpstreamTransport.RESPONSES
    assert responses.supports_thinking is True
    for selector in ("anthropic", "google", "unknown-package"):
        route = snapshot.route(selector)
        assert route is not None
        assert route.transport is OpenCodeUpstreamTransport.CHAT_COMPLETIONS
    assert snapshot.model_infos == frozenset(
        {
            ProviderModelInfo(
                "provider-default",
                supports_thinking=False,
                reasoning_capability=ReasoningCapability.NONE,
                input_modalities=frozenset({ModelInputModality.TEXT}),
                context_window_tokens=131072,
                max_output_tokens=8192,
            ),
            ProviderModelInfo(
                "responses-alias",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
            ),
            ProviderModelInfo("anthropic"),
            ProviderModelInfo("google"),
            ProviderModelInfo("unknown-package"),
        }
    )


def test_catalog_defaults_missing_package_to_chat_completions() -> None:
    snapshot = parse_open_code_catalog(
        _catalog_payload(
            {"defaulted": {"id": "upstream"}},
            provider_package=None,
        ),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
    )

    route = snapshot.route("defaulted")
    assert route is not None
    assert route.transport is OpenCodeUpstreamTransport.CHAT_COMPLETIONS


def test_catalog_token_limits_degrade_independently_without_rejecting_model() -> None:
    snapshot = parse_open_code_catalog(
        _catalog_payload(
            {
                "partial": {
                    "id": "upstream",
                    "limit": {"context": "bad", "output": 8192},
                },
                "malformed": {"id": "other", "limit": "bad"},
            }
        ),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
    )

    infos = {info.model_id: info for info in snapshot.model_infos}
    assert infos["partial"].context_window_tokens is None
    assert infos["partial"].max_output_tokens == 8192
    assert infos["malformed"].context_window_tokens is None
    assert infos["malformed"].max_output_tokens is None


@pytest.mark.parametrize(
    "payload,match",
    [
        ({}, "provider section"),
        ({"opencode": {"models": []}}, "models object"),
        (_catalog_payload({" ": {"id": "x"}}), "model selector"),
        (_catalog_payload({"x": {"id": " "}}), "model 'x' id"),
        (_catalog_payload({"x": {"id": "x", "status": "future"}}), "status"),
        (
            _catalog_payload({"x": {"id": "x", "provider": {"npm": " "}}}),
            "provider npm",
        ),
        (
            _catalog_payload({"x": {"id": "x", "status": "deprecated"}}),
            "no active or beta models",
        ),
    ],
)
def test_catalog_rejects_malformed_route_critical_data(
    payload: object,
    match: str,
) -> None:
    with pytest.raises(ModelListResponseError, match=match):
        parse_open_code_catalog(
            payload,
            provider_key="opencode",
            provider_name="OPENCODE_ZEN",
        )


def test_catalog_degrades_malformed_optional_capabilities_to_unknown() -> None:
    snapshot = parse_open_code_catalog(
        _catalog_payload(
            {
                "x": {
                    "id": "x",
                    "reasoning": "yes",
                    "modalities": {"input": ["text", 7]},
                }
            }
        ),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
    )

    assert snapshot.model_infos == frozenset({ProviderModelInfo("x")})


def test_catalog_rejects_duplicate_normalized_selectors() -> None:
    with pytest.raises(ModelListResponseError, match="duplicate normalized"):
        parse_open_code_catalog(
            _catalog_payload(
                {
                    "same": {"id": "first"},
                    " same ": {"id": "second"},
                }
            ),
            provider_key="opencode",
            provider_name="OPENCODE_ZEN",
        )


@pytest.mark.asyncio
async def test_cold_catalog_load_is_coalesced_and_never_sends_api_key() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        await asyncio.sleep(0)
        return httpx.Response(200, json=_catalog_payload())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    catalog = OpenCodeCatalog(
        _config(),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
        admission=immediate_admission(provider_name="opencode_zen"),
        client=client,
    )
    try:
        first, second = await asyncio.gather(catalog.snapshot(), catalog.snapshot())
    finally:
        await catalog.cleanup()

    assert first is second
    assert len(requests) == 1
    assert str(requests[0].url) == OPENCODE_CATALOG_URL
    assert requests[0].headers["user-agent"] == "opencode"
    assert "authorization" not in requests[0].headers
    assert "test_opencode_key" not in repr(requests[0].headers)


@pytest.mark.asyncio
async def test_refresh_atomically_replaces_snapshot_and_retains_last_good_on_error() -> (
    None
):
    payloads: list[object] = [
        _catalog_payload({"first": {"id": "first-upstream"}}),
        _catalog_payload({"second": {"id": "second-upstream"}}),
        _catalog_payload({"broken": {"id": ""}}),
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payloads.pop(0))

    catalog = OpenCodeCatalog(
        _config(),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
        admission=immediate_admission(provider_name="opencode_zen"),
        client=_catalog_client(handler),
    )
    try:
        first = await catalog.refresh()
        second = await catalog.refresh()
        with pytest.raises(ModelListResponseError):
            await catalog.refresh()
        cached = await catalog.snapshot()
    finally:
        await catalog.cleanup()

    assert set(first.routes) == {"first"}
    assert set(second.routes) == {"second"}
    assert cached is second


@pytest.mark.asyncio
async def test_cold_catalog_failure_is_canonical_and_never_guesses_transport() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": "busy"})

    catalog = OpenCodeCatalog(
        _config(),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
        admission=immediate_admission(
            provider_name="opencode_zen",
            max_attempts=2,
        ),
        client=_catalog_client(handler),
    )
    try:
        with pytest.raises(ExecutionFailure) as exc_info:
            await catalog.snapshot(request_id="req_catalog")
    finally:
        await catalog.cleanup()

    assert attempts == 2
    assert exc_info.value.retryable is True
    assert catalog.current_snapshot is None


@pytest.mark.asyncio
async def test_direct_inference_routes_responses_without_prior_model_listing() -> None:
    provider, generation_requests, catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        body = await _collect(provider, "responses-selector")
    finally:
        await provider.cleanup()

    assert len(catalog_requests) == 1
    assert [request.url.path for request in generation_requests] == [
        "/zen/v1/responses"
    ]
    payload = json.loads(generation_requests[0].content)
    assert payload["model"] == "responses-upstream"
    events = parse_sse_text(body)
    assert_anthropic_stream_contract(events)
    assert text_content(events) == "responses-ok"


@pytest.mark.asyncio
async def test_chat_catalog_route_uses_existing_chat_transport_and_upstream_id() -> (
    None
):
    provider, generation_requests, _catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        body = await _collect(provider, "chat-selector")
    finally:
        await provider.cleanup()

    assert [request.url.path for request in generation_requests] == [
        "/zen/v1/chat/completions"
    ]
    payload = json.loads(generation_requests[0].content)
    assert payload["model"] == "chat-upstream"
    events = parse_sse_text(body)
    assert_anthropic_stream_contract(events)
    assert text_content(events) == "chat-ok"


@pytest.mark.asyncio
async def test_responses_ingress_uses_catalog_responses_transport_natively() -> None:
    provider, generation_requests, catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        body = await _collect_responses(provider, "responses-selector")
    finally:
        await provider.cleanup()

    assert len(catalog_requests) == 1
    assert [request.url.path for request in generation_requests] == [
        "/zen/v1/responses"
    ]
    payload = json.loads(generation_requests[0].content)
    assert payload["model"] == "responses-upstream"
    assert payload["input"] == "hello"
    assert payload["stream"] is True
    assert payload["store"] is False
    events = parse_sse_text(body)
    assert [event.event for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[0].data["response"]["id"] == "resp_1"
    assert events[-1].data["response"]["model"] == "responses-selector"


@pytest.mark.asyncio
async def test_responses_tool_search_accepts_optional_codex_arguments() -> None:
    # Captured from Codex 0.153.0: OpenCode rejects this client's optional limit.
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "number",
                "description": "Maximum number of tools to return. Defaults to 8.",
            },
            "query": {"type": "string"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    tools = [
        {"type": "tool_search", "execution": "client", "parameters": parameters},
        {
            "type": "function",
            "name": "ordinary_search",
            "parameters": deepcopy(parameters),
            "strict": False,
        },
    ]
    request = _responses_request("responses-selector", tools=tools)
    original = request.model_dump()

    def upstream(wire_request: httpx2.Request) -> httpx2.Response:
        schema = json.loads(wire_request.content)["tools"][0]["parameters"]
        if set(schema["required"]) != set(schema["properties"]):
            return httpx2.Response(
                400,
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "param": "parameters",
                        "message": "'required' must include every property. Missing 'limit'.",
                    }
                },
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_responses_event_stream("hello"),
        )

    provider, wire_requests, _ = _provider_with_wire_transports(
        _catalog_payload(), generation_response=upstream
    )
    try:
        await provider.list_model_infos()
        provider.preflight_responses(request)
        body = "".join([chunk async for chunk in provider.stream_responses(request)])
    finally:
        await provider.cleanup()

    assert len(wire_requests) == 1
    assert "hello" in body
    forwarded = json.loads(wire_requests[0].content)["tools"]
    validator = Draft202012Validator(forwarded[0]["parameters"])
    assert validator.is_valid({"query": "files", "limit": None})
    assert validator.is_valid({"query": "files", "limit": 3})
    assert not validator.is_valid({"query": "files", "limit": "three"})
    assert not validator.is_valid({"query": None, "limit": 3})
    assert forwarded[1] == tools[1]
    assert request.model_dump() == original


@pytest.mark.asyncio
async def test_codex_custom_tool_round_trip_uses_opencode_function_tools() -> None:
    patch_text = "*** Begin Patch\n*** Add File: hello.txt\n+hello 🌍\n*** End Patch"
    custom_tool = {
        "type": "custom",
        "name": "edit_file",
        "description": "Edit a file with a raw patch.",
        "format": {"type": "text"},
    }
    tools = [
        custom_tool,
        {
            "type": "web_search",
            "external_web_access": False,
            "search_content_types": ["text", "image"],
        },
    ]

    def upstream(wire_request: httpx2.Request) -> httpx2.Response:
        wire = json.loads(wire_request.content)
        if any(
            tool["type"] == "custom" or "search_content_types" in tool
            for tool in wire["tools"]
        ):
            return httpx2.Response(
                400,
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Unsupported tool format",
                    }
                },
            )
        packets = [
            json.loads(line[6:])
            for line in _responses_event_stream("done").splitlines()
            if line.startswith("data: ")
        ]
        item = {
            "type": "function_call",
            "id": "item_custom",
            "call_id": "call_custom",
            "name": wire["tools"][0]["name"],
            "arguments": json.dumps({"input": patch_text}),
            "status": "completed",
        }
        packets = [
            packets[0],
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**item, "arguments": "", "status": "in_progress"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"],
                "output_index": 0,
                "delta": item["arguments"],
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": 0,
                "arguments": item["arguments"],
            },
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {**packets[-1], "response": {**packets[-1]["response"], "output": [item]}},
        ]
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(
                "data: " + json.dumps({**packet, "sequence_number": index}) + "\n\n"
                for index, packet in enumerate(packets)
            ),
        )

    provider, wire_requests, _ = _provider_with_wire_transports(
        _catalog_payload(), generation_response=upstream
    )
    try:
        output = await _collect_responses(provider, "responses-selector", tools=tools)
        events = parse_sse_text(output)
        call = events[-1].data["response"]["output"][0]
        assert call == {
            "type": "custom_tool_call",
            "id": "item_custom",
            "call_id": "call_custom",
            "name": "edit_file",
            "input": patch_text,
            "status": "completed",
        }
        assert any(
            event.event == "response.custom_tool_call_input.done"
            and event.data["input"] == patch_text
            for event in events
        )
        assert not any(
            event.event.startswith("response.function_call_arguments")
            for event in events
        )
        sequences = [event.data["sequence_number"] for event in events]
        assert sequences == sorted(set(sequences))
        await _collect_responses(
            provider,
            "responses-selector",
            tools=tools,
            input=[
                call,
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_custom",
                    "output": "Patch applied",
                },
            ],
        )
    finally:
        await provider.cleanup()

    first = json.loads(wire_requests[0].content)
    assert first["tools"][0]["type"] == "function"
    assert first["tools"][1] == {"type": "web_search", "external_web_access": False}
    replay = json.loads(wire_requests[1].content)["input"]
    assert replay[0]["type"] == "function_call"
    assert json.loads(replay[0]["arguments"]) == {"input": patch_text}
    assert replay[1] == {
        "type": "function_call_output",
        "call_id": "call_custom",
        "output": "Patch applied",
    }
    assert tools[0] == custom_tool


@pytest.mark.asyncio
async def test_responses_ingress_uses_catalog_chat_transport_directly() -> None:
    provider, generation_requests, catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        body = await _collect_responses(provider, "chat-selector")
    finally:
        await provider.cleanup()

    assert len(catalog_requests) == 1
    assert [request.url.path for request in generation_requests] == [
        "/zen/v1/chat/completions"
    ]
    payload = json.loads(generation_requests[0].content)
    assert payload["model"] == "chat-upstream"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    events = parse_sse_text(body)
    assert events[0].event == "response.created"
    assert events[-1].event == "response.completed"
    assert events[-1].data["response"]["model"] == "chat-selector"
    assert events[-1].data["response"]["output"][0]["content"][0]["text"] == ("chat-ok")


@pytest.mark.asyncio
@pytest.mark.parametrize("execution", ["client", "server"])
async def test_discovered_custom_tools_survive_sdk_calls_and_replay(
    execution: str,
) -> None:
    definition = {
        "type": "namespace",
        "name": "editor",
        "description": "Editing tools",
        "tools": [
            {
                "type": "custom",
                "name": "edit",
                "format": {"type": "text"},
                "defer_loading": True,
                "allowed_callers": ["direct"],
                "extension": {"keep": True},
            }
        ],
    }
    expected_tool = definition["tools"][0]
    assert isinstance(expected_tool, dict)
    discovery = {
        "type": "tool_search_output",
        "call_id": "search_one",
        "execution": execution,
        "status": "completed",
        "tools": [definition],
    }
    history = [discovery, {**discovery, "call_id": "search_two"}]

    def upstream(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        for loaded in body["input"][:2]:
            if execution == "client":
                assert loaded["type"] == "function_call_output"
                tool = json.loads(loaded["output"])[0]
                assert "defer_loading" not in tool
                assert tool in body["tools"]
            else:
                assert loaded["execution"] == execution
                tool = loaded["tools"][0]
                assert tool["defer_loading"] is True
            assert tool["type"] == "function"
            assert tool["name"] == "editor__edit"
            assert "namespace" not in tool
            for field in ("allowed_callers", "extension"):
                assert tool[field] == expected_tool[field]
        if len(body["input"]) > 2:
            assert body["input"][2]["type"] == "function_call"
            assert body["input"][2]["name"] == tool["name"]
            assert "namespace" not in body["input"][2]
            assert json.loads(body["input"][2]["arguments"]) == {"input": "patch"}
            assert body["input"][3] == {
                "type": "function_call_output",
                "call_id": "edit_one",
                "output": "applied",
            }
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=_responses_event_stream("done"),
            )
        packets = [
            json.loads(line[6:])
            for line in _responses_event_stream("done").splitlines()
            if line.startswith("data: ")
        ]
        call = {
            "type": "function_call",
            "id": "edit_item",
            "call_id": "edit_one",
            "name": tool["name"],
            "arguments": '{"input":"patch"}',
            "status": "completed",
        }
        events = [
            packets[0],
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**call, "status": "in_progress", "arguments": ""},
            },
            {"type": "response.output_item.done", "output_index": 0, "item": call},
            {**packets[-1], "response": {**packets[-1]["response"], "output": [call]}},
        ]
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(
                "data: " + json.dumps({**event, "sequence_number": i}) + "\n\n"
                for i, event in enumerate(events)
            ),
        )

    provider, requests, _ = _provider_with_wire_transports(
        _catalog_payload(), generation_response=upstream
    )
    try:
        response = await _collect_responses(
            provider, "responses-selector", input=history
        )
        call = parse_sse_text(response)[-1].data["response"]["output"][0]
        assert call["type"] == "custom_tool_call"
        assert call["name"] == "edit" and call["namespace"] == "editor"
        reply = await _collect_responses(
            provider,
            "responses-selector",
            input=[
                *history,
                call,
                {
                    "type": "custom_tool_call_output",
                    "call_id": "edit_one",
                    "output": "applied",
                },
            ],
        )
        assert "done" in reply
    finally:
        await provider.cleanup()
    assert len(requests) == 2
    assert history[0]["tools"] == [definition]


@pytest.mark.parametrize(
    (
        "wire_api",
        "primary_model",
        "fallback_model",
        "failed_path",
        "fallback_path",
        "fallback_text",
    ),
    (
        (
            "responses",
            "responses-selector",
            "chat-selector",
            "/zen/v1/responses",
            "/zen/v1/chat/completions",
            "chat-fallback",
        ),
        (
            "messages",
            "chat-selector",
            "responses-selector",
            "/zen/v1/chat/completions",
            "/zen/v1/responses",
            "responses-fallback",
        ),
    ),
)
@pytest.mark.asyncio
async def test_candidate_fallback_resolves_each_opencode_transport(
    wire_api: str,
    primary_model: str,
    fallback_model: str,
    failed_path: str,
    fallback_path: str,
    fallback_text: str,
) -> None:
    def generation_response(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == failed_path:
            return httpx2.Response(503, json={"error": "temporary overload"})
        if request.url.path == "/zen/v1/responses":
            body = _responses_event_stream(fallback_text)
        elif request.url.path == "/zen/v1/chat/completions":
            body = _chat_event_stream(fallback_text)
        else:
            raise AssertionError(f"unexpected generation path {request.url.path}")
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    provider, generation_requests, catalog_requests = _provider_with_wire_transports(
        _catalog_payload(),
        generation_response=generation_response,
        max_attempts=2,
    )

    def target(model: str) -> ProviderModelTarget:
        return ProviderModelTarget(
            provider_id="opencode_zen",
            provider_model=model,
            provider_model_ref=f"opencode_zen/{model}",
        )

    def upstream_id(model: str) -> str:
        return (
            "responses-upstream" if model == "responses-selector" else "chat-upstream"
        )

    resolved = ResolvedModelRoute(
        original_model="public-model",
        primary=target(primary_model),
        fallbacks=(target(fallback_model),),
        reasoning_preference=ReasoningPreference.INHERIT,
    )
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        progress_timeout_seconds=60.0,
    )

    try:
        if wire_api == "responses":
            stream = executor.stream_responses(
                RoutedResponsesRequest(
                    request=_responses_request(primary_model),
                    resolved=resolved,
                    reasoning=DEFAULT_REASONING_POLICY,
                ),
                raw_log_payload={},
                request_id="req_opencode_cross_transport_responses",
            )
        else:
            stream = executor.stream_messages(
                RoutedMessagesRequest(
                    request=_request(primary_model),
                    resolved=resolved,
                    reasoning=DEFAULT_REASONING_POLICY,
                ),
                raw_log_payload={},
                request_id="req_opencode_cross_transport_messages",
            )
        body = "".join([chunk async for chunk in stream])
    finally:
        await provider.cleanup()

    assert len(catalog_requests) == 1
    assert [request.url.path for request in generation_requests] == [
        failed_path,
        failed_path,
        fallback_path,
    ]
    payloads = [json.loads(request.content) for request in generation_requests]
    assert [payload["model"] for payload in payloads] == [
        upstream_id(primary_model),
        upstream_id(primary_model),
        upstream_id(fallback_model),
    ]
    events = parse_sse_text(body)
    if wire_api == "responses":
        assert [event.event for event in events].count("response.created") == 1
        assert [event.event for event in events].count("response.completed") == 1
        assert events[-1].data["response"]["model"] == "public-model"
        assert [
            event.data["delta"]
            for event in events
            if event.event == "response.output_text.delta"
        ] == [fallback_text]
    else:
        assert_anthropic_stream_contract(events)
        assert events[0].data["message"]["model"] == "public-model"
        assert text_content(events) == fallback_text
    assert "temporary overload" not in body


@pytest.mark.asyncio
async def test_unknown_model_rejects_before_either_generation_endpoint() -> None:
    provider, generation_requests, catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        with pytest.raises(InvalidRequestError, match="does not advertise"):
            await _collect(provider, "not-advertised")
    finally:
        await provider.cleanup()

    assert len(catalog_requests) == 1
    assert generation_requests == []


@pytest.mark.asyncio
async def test_model_listing_and_dispatch_use_same_snapshot() -> None:
    provider, generation_requests, catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        infos = await provider.list_model_infos()
        body = await _collect(provider, "responses-selector")
    finally:
        await provider.cleanup()

    assert infos == frozenset(
        {
            ProviderModelInfo(
                "chat-selector",
                supports_thinking=False,
                reasoning_capability=ReasoningCapability.NONE,
            ),
            ProviderModelInfo("responses-selector", supports_thinking=True),
        }
    )
    assert len(catalog_requests) == 1
    assert generation_requests[0].url.path.endswith("/responses")
    assert text_content(parse_sse_text(body)) == "responses-ok"


@pytest.mark.asyncio
async def test_cold_route_specific_conversion_failure_precedes_generation() -> None:
    provider, generation_requests, _catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        with pytest.raises(InvalidRequestError, match="stop_sequences"):
            await _collect(
                provider,
                "responses-selector",
                stop_sequences=["done"],
            )
    finally:
        await provider.cleanup()

    assert generation_requests == []


@pytest.mark.asyncio
async def test_warm_preflight_rejects_unknown_and_route_specific_fields() -> None:
    provider, generation_requests, _catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        await provider.list_model_infos()
        with pytest.raises(InvalidRequestError, match="does not advertise"):
            provider.preflight_messages(_request("missing"))
        with pytest.raises(InvalidRequestError, match="stop_sequences"):
            provider.preflight_messages(
                _request("responses-selector", stop_sequences=["done"])
            )
    finally:
        await provider.cleanup()

    assert generation_requests == []


@pytest.mark.asyncio
async def test_cleanup_attempts_both_owned_clients_when_generation_close_fails() -> (
    None
):
    provider, _generation_requests, _catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    generation_close = AsyncMock(side_effect=RuntimeError("generation close failed"))
    catalog_close = AsyncMock()
    provider._client.close = generation_close
    provider._catalog._client.aclose = catalog_close

    with pytest.raises(RuntimeError, match="generation close failed"):
        await provider.cleanup()

    generation_close.assert_awaited_once()
    catalog_close.assert_awaited_once()


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_build_request_body_replays_tool_reasoning_natively(
    provider_id: str,
) -> None:
    with (
        patch("free_claude_code.providers.openai_chat.client.AsyncOpenAI"),
        patch("httpx.AsyncClient"),
    ):
        provider = create_opencode_provider(
            provider_id,
            _config(),
            immediate_admission(provider_name=provider_id),
        )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I should inspect the file.",
                            "signature": "sig",
                        },
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "file contents",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))

    assistant = body["messages"][0]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == "I should inspect the file."
    assert "<think>" not in assistant["content"]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert body["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "file contents",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
async def test_tool_only_history_sends_empty_reasoning_content_on_wire(
    provider_id: str,
) -> None:
    with (
        patch("free_claude_code.providers.openai_chat.client.AsyncOpenAI"),
        patch("httpx.AsyncClient"),
    ):
        provider = create_opencode_provider(
            provider_id,
            _config(),
            immediate_admission(provider_name=provider_id),
        )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_missing",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_missing",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }
    )

    body = provider._chat._build_request_body(request, reasoning=reasoning_for(request))
    wire = await capture_openai_chat_wire_body(body)

    assistant = wire["messages"][0]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == ""
    assert assistant["tool_calls"][0]["id"] == "call_missing"
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    assert wire["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_missing",
        "content": "file contents",
    }

"""Tests for the CodeBuddy connected-account provider."""

import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest

from free_claude_code.application.connected_accounts import ConnectedAccountLoginMode
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.codebuddy.auth import CodeBuddyAuthManager
from free_claude_code.providers.codebuddy.provider import CodeBuddyProvider
from free_claude_code.providers.codebuddy.sanitize import (
    NEUTRAL_SYSTEM_PROMPT,
    sanitize_system_text,
    sanitize_tool_parameters,
)
from free_claude_code.providers.codebuddy.site import default_site
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_DEFAULT,
    immediate_admission,
    make_provider_config,
)

_SITE = default_site()


def _jwt(payload: dict) -> str:
    def encode(part: dict) -> str:
        raw = json.dumps(part).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


def _write_credentials(
    path,
    *,
    access_token: str = "access-token",
    refresh_token: str = "refresh-token",
    expires_at: int | None = None,
    uid: str | None = "uid-1",
    enterprise_id: str | None = None,
    domain: str | None = None,
) -> None:
    credentials = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "uid": uid,
        "enterprise_id": enterprise_id,
        "domain": domain,
        "nickname": None,
    }
    path.write_text(
        json.dumps({"version": 1, "credentials": credentials}),
        encoding="utf-8",
    )


def _auth_manager(tmp_path, *, expires_at: int | None = None, client=None):
    credential_path = tmp_path / "auth" / "codebuddy.json"
    credential_path.parent.mkdir(parents=True)
    _write_credentials(credential_path, expires_at=expires_at)
    return CodeBuddyAuthManager(
        credential_path=credential_path,
        lock_path=tmp_path / "auth" / "codebuddy.lock",
        client=client
        or httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda r: httpx2.Response(200, json={"code": 0, "data": {}})
            )
        ),
    )


def _provider(auth, endpoint_transport=None) -> CodeBuddyProvider:
    return CodeBuddyProvider(
        make_provider_config(api_key=None, base_url=_SITE.api_base),
        auth=auth,
        admission=immediate_admission(provider_name="CODEBUDDY"),
        endpoint_transport=endpoint_transport,
    )


def _chat_sse(text: str) -> str:
    chunks = (
        {
            "id": "chatcmpl_cb",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-v4.1-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_cb",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-v4.1-flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
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


def test_init_uses_base_provider_and_codebuddy_behavior(tmp_path):
    auth = _auth_manager(tmp_path)
    provider = _provider(auth)

    assert isinstance(provider, BaseProvider)
    assert provider._site.site_id == "intl-cli"
    assert provider._behavior.profile.provider_name == "CODEBUDDY"


def test_provider_honors_configured_base_url(tmp_path):
    auth = _auth_manager(tmp_path)
    provider = CodeBuddyProvider(
        make_provider_config(api_key=None, base_url="https://codebuddy.example.test/"),
        auth=auth,
        admission=immediate_admission(provider_name="CODEBUDDY"),
    )

    assert provider._site.api_base == "https://codebuddy.example.test"
    assert provider._site.origin == "https://codebuddy.example.test"
    assert (
        str(provider._client.base_url).rstrip("/")
        == "https://codebuddy.example.test/v2"
    )


def test_provider_uses_site_preset_when_base_url_matches_default(tmp_path):
    auth = _auth_manager(tmp_path)
    provider = _provider(auth)

    assert provider._site is default_site()


def test_finalize_body_relies_on_transport_streaming(tmp_path):
    provider = _provider(_auth_manager(tmp_path))
    body = provider._chat._build_request_body(make_messages_request())

    # The shared transport requests the stream; the body must not repeat it.
    assert "stream" not in body
    assert body["messages"][0] == {"role": "system", "content": "System prompt"}


def test_finalize_body_sanitizes_channel_fingerprints(tmp_path):
    provider = _provider(_auth_manager(tmp_path))
    request = make_messages_request(
        system="You are Claude Agent SDK. cc_version=1.0 x-anthropic-billing-header",
    )
    body = provider._chat._build_request_body(request)

    assert body["messages"][0] == {
        "role": "system",
        "content": NEUTRAL_SYSTEM_PROMPT,
    }
    assert sanitize_system_text("ordinary prompt") == "ordinary prompt"


def test_finalize_body_prepends_system_when_first_message_is_user(tmp_path):
    provider = _provider(_auth_manager(tmp_path))
    request = make_messages_request(system=None)
    body = provider._chat._build_request_body(request)

    assert body["messages"][0] == {
        "role": "system",
        "content": NEUTRAL_SYSTEM_PROMPT,
    }
    assert body["messages"][1]["role"] == "user"


def test_finalize_body_rewrites_developer_role_to_system(tmp_path):
    provider = _provider(_auth_manager(tmp_path))
    body = provider._chat._build_request_body(make_messages_request())
    body["messages"] = [
        {"role": "user", "content": "hi"},
        {"role": "developer", "content": "extra rules"},
    ]

    finalized = provider._behavior.finalize_chat_body(body, reasoning=REASONING_DEFAULT)

    assert finalized["messages"][0] == {
        "role": "system",
        "content": NEUTRAL_SYSTEM_PROMPT,
    }
    roles = [message["role"] for message in finalized["messages"]]
    assert "developer" not in roles
    assert roles.count("system") == 2


def test_finalize_body_normalizes_tool_choice_to_plain_string(tmp_path):
    provider = _provider(_auth_manager(tmp_path))
    tools = [
        {
            "name": "lookup",
            "description": "look things up",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]

    auto = provider._chat._build_request_body(
        make_messages_request(
            tools=tools,
            tool_choice={"type": "auto"},
        )
    )
    assert auto["tool_choice"] == "auto"

    forced = provider._chat._build_request_body(
        make_messages_request(
            tools=tools,
            tool_choice={"type": "tool", "name": "lookup"},
        )
    )
    assert forced["tool_choice"] == "lookup"

    disabled = provider._chat._build_request_body(
        make_messages_request(
            tools=tools,
            tool_choice={"type": "none"},
        )
    )
    assert "tool_choice" not in disabled
    assert "tools" not in disabled


def test_finalize_body_sanitizes_tool_schemas(tmp_path):
    provider = _provider(_auth_manager(tmp_path))
    request = make_messages_request(
        tools=[
            {
                "name": "rebuild",
                "description": "rebuild the project",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$defs": {
                        "Target": {
                            "allOf": [
                                {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                },
                                {"required": ["path"]},
                            ],
                            "if": {"type": "object"},
                            "then": {"type": "object"},
                        }
                    },
                    "type": "object",
                    "properties": {
                        "target": {"$ref": "#/$defs/Target"},
                        "dangling": {"$ref": "#/$defs/Missing"},
                    },
                    "unevaluatedProperties": False,
                },
            }
        ]
    )
    body = provider._chat._build_request_body(request)

    parameters = body["tools"][0]["function"]["parameters"]
    assert "$defs" not in parameters
    assert "$schema" not in parameters
    assert "unevaluatedProperties" not in parameters
    target = parameters["properties"]["target"]
    assert target["type"] == "object"
    assert target["required"] == ["path"]
    assert "if" not in target and "then" not in target
    # Dangling refs are dropped, leaving an empty schema behind.
    assert parameters["properties"]["dangling"] == {}


def test_sanitize_tool_parameters_keeps_plain_schemas():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    assert sanitize_tool_parameters(schema) == schema


@pytest.mark.asyncio
async def test_stream_messages_sends_upstream_shape_and_streams_text(tmp_path):
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_chat_sse("codebuddy-ok"),
        )

    auth = _auth_manager(
        tmp_path,
        expires_at=int(time.time()) + 3600,
        client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    provider = _provider(auth, endpoint_transport=httpx2.MockTransport(handler))
    try:
        events = [
            event
            async for event in provider.stream_messages(
                make_messages_request(model="deepseek-v4.1-flash"),
                input_tokens=2,
                request_id="req_codebuddy_messages",
            )
        ]
    finally:
        await provider.cleanup()

    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/v2/chat/completions"
    assert request.headers["authorization"] == "Bearer access-token"
    assert request.headers["x-product"] == "SaaS"
    assert request.headers["x-user-id"] == "uid-1"
    assert "CLI/2.63.2 CodeBuddy/2.63.2" in request.headers["user-agent"]
    assert request.headers["x-request-id"]
    assert request.headers["x-request-trace-id"]

    payload = json.loads(request.content)
    assert payload["stream"] is True
    assert payload["model"] == "deepseek-v4.1-flash"
    assert payload["messages"][0] == {"role": "system", "content": "System prompt"}

    parsed = parse_sse_text("".join(events))
    assert_anthropic_stream_contract(parsed)
    assert text_content(parsed) == "codebuddy-ok"


@pytest.mark.asyncio
async def test_stream_messages_neutralizes_fingerprinted_system_prompt(tmp_path):
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_chat_sse("ok"),
        )

    auth = _auth_manager(tmp_path, expires_at=int(time.time()) + 3600)
    provider = _provider(auth, endpoint_transport=httpx2.MockTransport(handler))
    try:
        async for _ in provider.stream_messages(
            make_messages_request(system="You are Codex, cc_version=9"),
            request_id="req_codebuddy_sanitize",
        ):
            pass
    finally:
        await provider.cleanup()

    payload = json.loads(requests[0].content)
    assert payload["messages"][0]["content"] == NEUTRAL_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_stream_responses_accepts_responses_ingress(tmp_path):
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_chat_sse("responses-ok"),
        )

    auth = _auth_manager(tmp_path, expires_at=int(time.time()) + 3600)
    provider = _provider(auth, endpoint_transport=httpx2.MockTransport(handler))
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "deepseek-v4.1-flash",
            "input": "hello",
            "max_output_tokens": 64,
        }
    )
    try:
        events = [
            event
            async for event in provider.stream_responses(
                request,
                input_tokens=2,
                request_id="req_codebuddy_responses",
            )
        ]
    finally:
        await provider.cleanup()

    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["stream"] is True
    assert payload["messages"][0]["role"] == "system"

    parsed = parse_sse_text("".join(events))
    assert parsed[0].event == "response.created"
    assert parsed[-1].event == "response.completed"


@pytest.mark.asyncio
async def test_list_model_infos_maps_live_catalog(tmp_path):
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/console/enterprises/personal/models"
        assert request.headers["authorization"] == "Bearer access-token"
        return httpx2.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "agents": [{"name": "cli", "models": ["kimi-k3", "hy3"]}],
                    "models": [
                        {
                            "id": "kimi-k3",
                            "name": "Kimi-K3",
                            "disabled": False,
                            "maxInputTokens": 262144,
                            "maxOutputTokens": 8192,
                            "supportsImages": True,
                            "supportsReasoning": True,
                        },
                        {
                            "id": "hy3",
                            "name": "Hy3",
                            "disabled": False,
                            "supportsImages": False,
                        },
                        {
                            "id": "disabled-model",
                            "name": "Off",
                            "disabled": True,
                        },
                        {
                            "id": "not-in-cli",
                            "name": "Hidden",
                            "disabled": False,
                        },
                    ],
                },
            },
        )

    auth = _auth_manager(tmp_path, expires_at=int(time.time()) + 3600)
    provider = _provider(auth, endpoint_transport=httpx2.MockTransport(handler))

    infos = await provider.list_model_infos()

    assert infos == frozenset(
        {
            ProviderModelInfo(
                "kimi-k3",
                supports_thinking=True,
                input_modalities=frozenset(
                    {ModelInputModality.TEXT, ModelInputModality.IMAGE}
                ),
                context_window_tokens=262144,
                max_output_tokens=8192,
            ),
            ProviderModelInfo(
                "hy3",
                input_modalities=frozenset({ModelInputModality.TEXT}),
            ),
        }
    )


@pytest.mark.asyncio
async def test_list_model_infos_falls_back_to_static_ids(tmp_path):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, text="boom")

    auth = _auth_manager(tmp_path, expires_at=int(time.time()) + 3600)
    provider = _provider(auth, endpoint_transport=httpx2.MockTransport(handler))

    infos = await provider.list_model_infos()

    ids = {info.model_id for info in infos}
    assert "auto" in ids
    assert "deepseek-v4.1-flash" in ids
    assert "kimi-k3" in ids
    assert all(info.context_window_tokens is None for info in infos)


@pytest.mark.asyncio
async def test_list_model_infos_refreshes_and_retries_once_after_401(tmp_path):
    catalog_calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/v2/plugin/auth/token/refresh":
            return httpx2.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "accessToken": "new-token",
                        "refreshToken": "new-refresh",
                        "expiresIn": 7200,
                    },
                },
            )
        catalog_calls.append(request)
        if len(catalog_calls) == 1:
            return httpx2.Response(401, json={"code": 401, "msg": "unauthorized"})
        return httpx2.Response(
            200,
            json={
                "code": 0,
                "data": {"models": [{"id": "kimi-k3", "disabled": False}]},
            },
        )

    auth = _auth_manager(
        tmp_path,
        expires_at=int(time.time()) + 3600,
        client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    provider = _provider(auth, endpoint_transport=httpx2.MockTransport(handler))

    infos = await provider.list_model_infos()

    assert {info.model_id for info in infos} == {"kimi-k3"}
    assert len(catalog_calls) == 2
    assert catalog_calls[0].headers["authorization"] == "Bearer access-token"
    assert catalog_calls[1].headers["authorization"] == "Bearer new-token"


@pytest.mark.asyncio
async def test_list_model_infos_retries_401_only_once(tmp_path):
    catalog_calls: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/v2/plugin/auth/token/refresh":
            return httpx2.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "accessToken": "new-token",
                        "refreshToken": "new-refresh",
                        "expiresIn": 7200,
                    },
                },
            )
        catalog_calls.append(request)
        return httpx2.Response(401, json={"code": 401, "msg": "unauthorized"})

    auth = _auth_manager(
        tmp_path,
        expires_at=int(time.time()) + 3600,
        client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    provider = _provider(auth, endpoint_transport=httpx2.MockTransport(handler))

    infos = await provider.list_model_infos()

    assert len(catalog_calls) == 2
    assert {info.model_id for info in infos} >= {"auto", "kimi-k3"}


@pytest.mark.asyncio
async def test_cleanup_closes_openai_client(tmp_path):
    provider = _provider(_auth_manager(tmp_path))
    provider._client = MagicMock()
    provider._client.close = AsyncMock()

    await provider.cleanup()

    provider._client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_access_hydrates_identity_from_jwt_claims(tmp_path):
    credential_path = tmp_path / "auth" / "codebuddy.json"
    credential_path.parent.mkdir(parents=True)
    token = _jwt(
        {
            "sub": "user-42",
            "exp": int(time.time()) + 3600,
            "iss": "https://login.example.com/sso-ent-7",
        }
    )
    _write_credentials(
        credential_path,
        access_token=token,
        expires_at=None,
        uid=None,
        enterprise_id=None,
        domain=None,
    )
    manager = CodeBuddyAuthManager(
        credential_path=credential_path,
        lock_path=tmp_path / "auth" / "codebuddy.lock",
        client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda r: httpx2.Response(200, json={"code": 0, "data": {}})
            )
        ),
    )
    try:
        access = await manager.access()
    finally:
        await manager.close()

    assert access.access_token == token
    assert access.uid == "user-42"
    assert access.enterprise_id == "ent-7"
    assert access.domain == "login.example.com"


@pytest.mark.asyncio
async def test_access_refreshes_expired_token_and_persists(tmp_path):
    credential_path = tmp_path / "auth" / "codebuddy.json"
    credential_path.parent.mkdir(parents=True)
    _write_credentials(
        credential_path,
        access_token="old-token",
        refresh_token="old-refresh",
        expires_at=int(time.time()) - 10,
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v2/plugin/auth/token/refresh"
        assert request.headers["x-refresh-token"] == "old-refresh"
        assert request.headers["x-auth-refresh-source"] == "workbuddy"
        return httpx2.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "accessToken": "new-token",
                    "refreshToken": "new-refresh",
                    "expiresIn": 7200,
                },
            },
        )

    manager = CodeBuddyAuthManager(
        credential_path=credential_path,
        lock_path=tmp_path / "auth" / "codebuddy.lock",
        client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    try:
        access = await manager.access()
    finally:
        await manager.close()

    assert access.access_token == "new-token"
    saved = json.loads(credential_path.read_text(encoding="utf-8"))
    assert saved["credentials"]["access_token"] == "new-token"
    assert saved["credentials"]["refresh_token"] == "new-refresh"


@pytest.mark.asyncio
async def test_start_login_rejects_non_device_mode(tmp_path):
    manager = _auth_manager(tmp_path)
    try:
        with pytest.raises(InvalidRequestError, match="device-code"):
            await manager.start_login(ConnectedAccountLoginMode.BROWSER)
    finally:
        await manager.close()

    status = manager.status()
    assert status.attempt_id is None
    assert status.connected is True

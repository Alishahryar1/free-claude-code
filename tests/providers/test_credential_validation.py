"""Credential evidence must not consume inference or reject ambiguous failures."""

import asyncio

import httpx
import pytest

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers import credential_validation as validation

# Minimal published response shapes, independent of the implementation registry.
CASES = [
    (
        "open_router",
        "https://openrouter.ai/api/v1/key",
        {"data": {"label": "key"}},
        401,
    ),
    ("groq", "https://api.groq.com/openai/v1/models", {"data": []}, 401),
    ("together", "https://api.together.ai/v1/models", [], 401),
    (
        "deepseek",
        "https://api.deepseek.com/user/balance",
        {"is_available": True, "balance_infos": []},
        401,
    ),
    ("xai", "https://api.x.ai/v1/api-key", {"api_key_id": "id"}, 401),
    (
        "siliconflow",
        "https://api.siliconflow.com/v1/user/info",
        {"code": 20000, "status": True, "data": {}},
        401,
    ),
    (
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
        {"models": []},
        400,
    ),
    ("nebius", "https://api.tokenfactory.nebius.com/v1/models", {"data": []}, 401),
    (
        "vercel",
        "https://ai-gateway.vercel.sh/v1/credits",
        {"balance": "0", "total_used": "1"},
        401,
    ),
    ("huggingface", "https://huggingface.co/api/whoami-v2", {"name": "user"}, 401),
    ("cohere", "https://api.cohere.com/v1/models?page_size=1", {"models": []}, 498),
    (
        "wafer",
        "https://api.wafer.ai/v1/usage/me?period=1d&endpoint=pass.wafer.ai",
        {"user_id": "id"},
        401,
    ),
    (
        "kimi",
        "https://api.moonshot.ai/v1/users/me/balance",
        {"data": {"available_balance": 0}},
        401,
    ),
    ("nararoute", "https://router.bynara.id/v1/models", {"data": []}, 401),
    (
        "deepinfra",
        "https://api.deepinfra.com/v1/me",
        {"uid": "id", "email": None},
        None,
    ),
    ("mistral", "https://api.mistral.ai/v1/models", {"data": []}, None),
    ("wandb", "https://api.inference.wandb.ai/v1/models", {"data": []}, None),
    ("minimax", "https://api.minimax.io/v1/models", {"data": []}, None),
    (
        "novita",
        "https://api.novita.ai/openapi/v1/billing/balance/detail",
        {"availableBalance": "0"},
        None,
    ),
    ("cerebras", "https://api.cerebras.ai/v1/models", {"data": []}, None),
    ("sambanova", "https://api.sambanova.ai/v1/models", {"data": []}, None),
    (
        "fireworks",
        "https://api.fireworks.ai/v1/accounts?pageSize=1",
        {"accounts": []},
        None,
    ),
]


def _settings(provider_id):
    descriptor = PROVIDER_CATALOG[provider_id]
    assert descriptor.credential_attr is not None
    return Settings.model_construct(
        _fields_set=set(), **{descriptor.credential_attr: "secret-not-a-known-format"}
    )


def _mock_http(monkeypatch, handler):
    clients = []
    real_client = httpx.AsyncClient

    def factory(**kwargs):
        client = real_client(**kwargs, transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    monkeypatch.setattr(validation.httpx, "AsyncClient", factory)
    return clients


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_id", "url", "payload", "rejection"), CASES)
async def test_documented_probe_acceptance_and_rejection(
    monkeypatch, provider_id, url, payload, rejection
):
    requests = []
    status = 200
    body = payload

    def respond(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url) == url
        header = "x-goog-api-key" if provider_id == "gemini" else "Authorization"
        expected = (
            "secret-not-a-known-format"
            if provider_id == "gemini"
            else "Bearer secret-not-a-known-format"
        )
        assert request.headers[header] == expected
        assert "secret-not-a-known-format" not in str(request.url)
        assert not request.content
        return httpx.Response(status, json=body)

    clients = _mock_http(monkeypatch, respond)
    key = PROVIDER_CATALOG[provider_id].credential_env
    assert key is not None
    result = await validation.check_credentials(_settings(provider_id), (key, key))
    assert result[0].status == validation.CredentialStatus.VERIFIED
    assert len(requests) == 1
    assert all(client.is_closed for client in clients)

    status = rejection or 401
    body = {"error": {"message": "secret-not-a-known-format"}}
    if provider_id == "gemini":
        body = {"error": {"details": [{"reason": "API_KEY_INVALID"}]}}
    if provider_id == "siliconflow":
        body = "Invalid token"  # The documented StringData error body.
    result = await validation.check_credentials(_settings(provider_id), (key,))
    assert result[0].status == (
        validation.CredentialStatus.REJECTED
        if rejection
        else validation.CredentialStatus.UNVERIFIED
    )
    assert "secret-not-a-known-format" not in repr(result)
    assert len(requests) == 2
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 402, 403, 404, 429, 500, 503])
async def test_ambiguous_errors_warn_without_retry(monkeypatch, status, caplog):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={"error": "secret-not-a-known-format"})

    _mock_http(monkeypatch, respond)
    result = await validation.check_credentials(_settings("groq"), ("GROQ_API_KEY",))
    assert result[0].status == validation.CredentialStatus.UNVERIFIED
    assert len(requests) == 1
    assert "secret-not-a-known-format" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 401, 302])
async def test_non_api_responses_do_not_verify_or_reject(monkeypatch, status):
    _mock_http(
        monkeypatch,
        lambda request: httpx.Response(
            status,
            text="<html>Sign in</html>",
            headers={"Location": "https://other.invalid/"},
        ),
    )
    result = await validation.check_credentials(_settings("groq"), ("GROQ_API_KEY",))
    assert result[0].status == validation.CredentialStatus.UNVERIFIED


@pytest.mark.asyncio
async def test_unsupported_shared_credentials_never_send_http(monkeypatch):
    def unexpected(request):
        pytest.fail("Public catalogs must not be used to verify a credential")

    _mock_http(monkeypatch, unexpected)
    supported = {PROVIDER_CATALOG[row[0]].credential_env for row in CASES}
    all_keys = {d.credential_env for d in PROVIDER_CATALOG.values() if d.credential_env}
    keys = tuple(all_keys - supported)
    assert len(keys) == 21
    result = await validation.check_credentials(
        Settings.model_construct(), (*keys, "OPENCODE_API_KEY", "MODEL")
    )
    assert len(result) == 21
    assert all(
        check.status == validation.CredentialStatus.UNVERIFIED for check in result
    )


@pytest.mark.asyncio
async def test_batch_timeout_preserves_finished_rejection_and_closes_clients(
    monkeypatch,
):
    async def respond(request):
        if request.url.host == "api.groq.com":
            return httpx.Response(401, json={"error": "invalid key"})
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    clients = _mock_http(monkeypatch, respond)
    monkeypatch.setattr(validation, "MAX_CONCURRENCY", 1)
    monkeypatch.setattr(validation, "BATCH_TIMEOUT", 0.1)
    settings = Settings.model_construct(groq_api_key="key", open_router_api_key="key")
    results = await validation.check_credentials(
        settings, ("GROQ_API_KEY", "OPENROUTER_API_KEY")
    )
    assert [r.status for r in results] == [
        validation.CredentialStatus.REJECTED,
        validation.CredentialStatus.UNVERIFIED,
    ]
    assert len(clients) == 2
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_external_cancellation_propagates_and_closes_clients(monkeypatch):
    started = asyncio.Event()

    async def respond(request):
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    clients = _mock_http(monkeypatch, respond)
    task = asyncio.create_task(
        validation.check_credentials(_settings("groq"), ("GROQ_API_KEY",))
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_connection_failure_is_safe_and_does_not_retry(monkeypatch):
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("secret-not-a-known-format", request=request)

    clients = _mock_http(monkeypatch, respond)
    results = await validation.check_credentials(_settings("groq"), ("GROQ_API_KEY",))
    assert results[0].status == validation.CredentialStatus.UNVERIFIED
    assert "secret-not-a-known-format" not in repr(results)
    assert calls == 1
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_id", "payload"),
    [
        ("deepseek", {"is_available": False, "balance_infos": []}),
        ("xai", {"api_key_id": "id", "team_blocked": True}),
    ],
)
async def test_recognized_restricted_key_warns_instead_of_rejecting(
    monkeypatch, provider_id, payload
):
    _mock_http(monkeypatch, lambda request: httpx.Response(200, json=payload))
    key = PROVIDER_CATALOG[provider_id].credential_env
    assert key is not None
    result = await validation.check_credentials(_settings(provider_id), (key,))
    assert result[0].status == validation.CredentialStatus.UNVERIFIED
    assert "recognized" in result[0].message


@pytest.mark.asyncio
async def test_checks_are_concurrent_but_bounded(monkeypatch):
    first_wave = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0
    started = []

    async def check(settings, key, probe):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started.append(key)
        if active == 2:
            first_wave.set()
        await release.wait()
        active -= 1
        return validation.CredentialCheck(
            key, validation.CredentialStatus.VERIFIED, "accepted"
        )

    monkeypatch.setattr(validation, "_check_one", check)
    monkeypatch.setattr(validation, "MAX_CONCURRENCY", 2)
    task = asyncio.create_task(
        validation.check_credentials(
            Settings.model_construct(),
            ("GROQ_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY"),
        )
    )
    await first_wave.wait()
    assert len(started) == 2
    release.set()
    results = await task
    assert len(results) == 3
    assert peak == 2


@pytest.mark.asyncio
async def test_programming_errors_do_not_become_permission_to_save(monkeypatch):
    def broken(request):
        raise RuntimeError("unexpected bug")

    clients = _mock_http(monkeypatch, broken)
    with pytest.raises(ExceptionGroup) as error:
        await validation.check_credentials(_settings("groq"), ("GROQ_API_KEY",))
    assert isinstance(error.value.exceptions[0], RuntimeError)
    assert all(client.is_closed for client in clients)

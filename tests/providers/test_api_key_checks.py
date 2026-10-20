"""Save checks every submitted key with bounded work and stable row results."""

import asyncio

import httpx
import pytest
from pydantic import SecretStr

from free_claude_code.config.admin.credentials import CredentialSubmission
from free_claude_code.config.settings import Settings
from free_claude_code.providers import credential_validation as validation


def mock_http(monkeypatch, respond):
    original = httpx.AsyncClient
    clients = []

    def create(**kwargs):
        client = original(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(validation.httpx, "AsyncClient", create)
    return clients


@pytest.mark.asyncio
async def test_all_rows_use_their_own_credential_and_keep_result_order(monkeypatch):
    seen = []

    async def respond(request):
        credential = request.headers["Authorization"].removeprefix("Bearer ")
        seen.append(credential)
        if credential == "key-0":
            await asyncio.sleep(0.01)
        status = 401 if credential == "key-9" else 200
        return httpx.Response(
            status, json={"error": "invalid"} if status == 401 else {"data": []}
        )

    clients = mock_http(monkeypatch, respond)
    credentials = tuple(SecretStr(f"key-{index}") for index in range(14))
    results = await validation.check_credentials(
        Settings(GROQ_API_KEY="runtime-primary"),
        (CredentialSubmission("GROQ_API_KEY", credentials),),
    )
    assert set(seen) == {f"key-{index}" for index in range(14)}
    assert [result.index for result in results] == list(range(14))
    assert [
        result.index
        for result in results
        if result.status == validation.CredentialStatus.REJECTED
    ] == [9]
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_queued_rows_each_get_an_attempt_and_timeout(monkeypatch):
    active = peak = 0
    seen = []

    async def respond(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        seen.append(request.headers["Authorization"])
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1

    clients = mock_http(monkeypatch, respond)
    monkeypatch.setattr(validation, "MAX_CONCURRENCY", 2)
    monkeypatch.setattr(validation, "REQUEST_TIMEOUT", 0.03)
    credentials = tuple(SecretStr(f"key-{index}") for index in range(7))
    results = await validation.check_credentials(
        Settings(), (CredentialSubmission("GROQ_API_KEY", credentials),)
    )
    assert len(seen) == 7 and len(set(seen)) == 7
    assert peak <= 2 and active == 0
    assert len(results) == 7
    assert all(
        result.status == validation.CredentialStatus.UNVERIFIED for result in results
    )
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_custom_keys_get_individual_unverified_rows_without_public_probe(
    monkeypatch,
):
    def unexpected(request):
        pytest.fail("A custom public model list cannot verify a key")

    mock_http(monkeypatch, unexpected)
    provider_id = "custom_" + "1" * 32
    submission = CredentialSubmission(
        "api_keys", (SecretStr("one"), SecretStr("two")), provider_id
    )
    results = await validation.check_credentials(Settings(), (submission,))
    assert [result.index for result in results] == [0, 1]
    assert all(result.provider_id == provider_id for result in results)
    assert all(
        result.status == validation.CredentialStatus.UNVERIFIED for result in results
    )
    assert "one" not in repr(results) and "two" not in repr(results)


@pytest.mark.asyncio
async def test_cancelled_batch_closes_every_active_probe(monkeypatch):
    started = asyncio.Event()

    async def respond(request):
        started.set()
        await asyncio.Event().wait()

    clients = mock_http(monkeypatch, respond)
    credentials = tuple(SecretStr(f"key-{index}") for index in range(20))
    task = asyncio.create_task(
        validation.check_credentials(
            Settings(), (CredentialSubmission("GROQ_API_KEY", credentials),)
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert clients and len(clients) <= validation.MAX_CONCURRENCY
    assert all(client.is_closed for client in clients)

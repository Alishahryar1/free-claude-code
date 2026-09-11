"""Non-generating credential checks for Admin Apply, independent of inference.

Only documented authenticated endpoints belong here. Public model catalogs and
management endpoints requiring a different key cannot validate inference keys.
Failure defaults to unverified; rejection requires provider-specific evidence.
"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import httpx

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonValue
from free_claude_code.providers.runtime.config import string_setting

REQUEST_TIMEOUT = 5.0
BATCH_TIMEOUT = 10.0
MAX_CONCURRENCY = 8


class CredentialStatus(StrEnum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class CredentialCheck:
    key: str
    status: CredentialStatus
    message: str


def _field(payload: JsonValue, name: str) -> JsonValue:
    return payload.get(name) if isinstance(payload, Mapping) else None


def _list_field(name: str) -> Callable[[JsonValue], bool]:
    return lambda payload: isinstance(_field(payload, name), list)


def _identity(name: str) -> Callable[[JsonValue], bool]:
    return lambda payload: isinstance(_field(payload, name), str)


@dataclass(frozen=True, slots=True)
class _Probe:
    provider_id: str
    path: str
    accepts: Callable[[JsonValue], bool]
    rejected_statuses: frozenset[int] = frozenset()


_AUTH_401 = frozenset({401})
_MODELS = _list_field("data")

# Sources document both authenticated reads and (where enabled) rejection codes.
# https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key
# https://console.groq.com/docs/errors
# https://docs.together.ai/reference/models
# https://api-docs.deepseek.com/quick_start/error_codes/
# https://api.x.ai/api-docs/openapi.json
# https://docs.siliconflow.com/cn/api-reference/userinfo/get-user-info
# https://ai.google.dev/gemini-api/docs/generate-content/api-errors
# https://github.com/nebius/nebius-physical-ai/blob/main/docs/workbench/token-factory.md
# https://vercel.com/docs/ai-gateway/sdks-and-apis/rest-api
# https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hf_api.py
# https://docs.cohere.com/reference/list-models
# https://docs.wafer.ai/serverless/usage-api
# https://platform.kimi.ai/docs/api/errors
# https://router.bynara.id/id/docs
_PROBES = (
    _Probe(
        "open_router",
        "/key",
        lambda p: isinstance(_field(p, "data"), Mapping),
        _AUTH_401,
    ),
    _Probe("groq", "/models", _MODELS, _AUTH_401),
    _Probe("together", "/models", lambda p: isinstance(p, list), _AUTH_401),
    _Probe(
        "deepseek",
        "/user/balance",
        lambda p: (
            isinstance(_field(p, "is_available"), bool)
            and isinstance(_field(p, "balance_infos"), list)
        ),
        _AUTH_401,
    ),
    _Probe("xai", "/api-key", _identity("api_key_id"), _AUTH_401),
    _Probe(
        "siliconflow",
        "/user/info",
        lambda p: (
            _field(p, "code") == 20000
            and _field(p, "status") is True
            and isinstance(_field(p, "data"), Mapping)
        ),
        _AUTH_401,
    ),
    _Probe(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
        _list_field("models"),
    ),
    _Probe("nebius", "/models", _MODELS, _AUTH_401),
    _Probe(
        "vercel",
        "/credits",
        lambda p: isinstance(_field(p, "balance"), str | int | float),
        _AUTH_401,
    ),
    _Probe(
        "huggingface",
        "https://huggingface.co/api/whoami-v2",
        _identity("name"),
        _AUTH_401,
    ),
    _Probe(
        "cohere",
        "https://api.cohere.com/v1/models?page_size=1",
        _list_field("models"),
        frozenset({401, 498}),
    ),
    _Probe(
        "wafer",
        "https://api.wafer.ai/v1/usage/me?period=1d&endpoint=pass.wafer.ai",
        _identity("user_id"),
        _AUTH_401,
    ),
    _Probe(
        "kimi",
        "/users/me/balance",
        lambda p: isinstance(
            _field(_field(p, "data"), "available_balance"), int | float
        ),
        _AUTH_401,
    ),
    _Probe("nararoute", "/models", _MODELS, _AUTH_401),
    _Probe("orcarouter", "/models", _MODELS, _AUTH_401),
    # Positive evidence only: these errors can also reflect permissions, budget,
    # token type, or an undocumented response contract. Never reject on failure.
    # https://docs.deepinfra.com/api-reference/account/me
    # https://docs.mistral.ai/api/endpoint/models
    # https://docs.wandb.ai/inference/api-reference
    # https://docs.github.com/en/rest/users/users#get-the-authenticated-user
    # https://platform.minimax.io/docs/api-reference/models/openai/list-models
    # https://novita.ai/docs/api-reference/basic-get-user-balance
    # https://inference-docs.cerebras.ai/api-reference/models/list-models
    # https://docs.sambanova.ai/docs/api-reference/models/get-environments-available-model-list-metadata
    # https://docs.fireworks.ai/api-reference/list-accounts
    _Probe("deepinfra", "https://api.deepinfra.com/v1/me", _identity("uid")),
    _Probe("mistral", "/models", _MODELS),
    _Probe("wandb", "/models", _MODELS),
    _Probe("minimax", "/models", _MODELS),
    _Probe(
        "novita",
        "https://api.novita.ai/openapi/v1/billing/balance/detail",
        _identity("availableBalance"),
    ),
    _Probe("cerebras", "/models", _MODELS),
    _Probe("sambanova", "/models", _MODELS),
    _Probe(
        "fireworks",
        "https://api.fireworks.ai/v1/accounts?pageSize=1",
        _list_field("accounts"),
    ),
)
_PROBE_BY_KEY = {
    PROVIDER_CATALOG[probe.provider_id].credential_env: probe for probe in _PROBES
}


def _unverified(key: str, message: str) -> CredentialCheck:
    return CredentialCheck(key, CredentialStatus.UNVERIFIED, message)


def _invalid_google_key(payload: JsonValue) -> bool:
    details = _field(_field(payload, "error"), "details")
    return isinstance(details, list) and any(
        _field(detail, "reason") == "API_KEY_INVALID" for detail in details
    )


def _interpret(key: str, probe: _Probe, response: httpx.Response) -> CredentialCheck:
    try:
        payload: JsonValue = response.json()
    except ValueError:
        return _unverified(
            key, "Could not verify this key: unexpected provider response."
        )
    if response.is_success and probe.accepts(payload):
        if (
            probe.provider_id == "deepseek" and _field(payload, "is_available") is False
        ) or (
            probe.provider_id == "xai"
            and any(
                _field(payload, flag) is True
                for flag in ("api_key_disabled", "api_key_blocked", "team_blocked")
            )
        ):
            return _unverified(
                key,
                "The key was recognized, but the provider reports a billing or access restriction.",
            )
        return CredentialCheck(key, CredentialStatus.VERIFIED, "API key accepted.")
    # Require an API error body; an HTML intermediary's auth challenge is not
    # evidence that the provider rejected the submitted credential.
    api_error = isinstance(payload, Mapping) and bool(
        payload.get("error") or payload.get("message") or payload.get("detail")
    )
    # SiliconFlow's /user/info OpenAPI defines 401 JSON as StringData.
    if probe.provider_id == "siliconflow" and isinstance(payload, str):
        api_error = True
    if (api_error and response.status_code in probe.rejected_statuses) or (
        probe.provider_id == "gemini"
        and response.status_code == 400
        and _invalid_google_key(payload)
    ):
        return CredentialCheck(
            key,
            CredentialStatus.REJECTED,
            "The provider rejected this API key. Check it or create a new key.",
        )
    if response.status_code in {402, 403}:
        return _unverified(
            key,
            "Could not verify this key: check the provider's billing or permissions.",
        )
    return _unverified(
        key, "Could not verify this key with the provider. You can still save it."
    )


async def _check_one(settings: Settings, key: str, probe: _Probe) -> CredentialCheck:
    descriptor = PROVIDER_CATALOG[probe.provider_id]
    credential = string_setting(settings, descriptor.credential_attr)
    if not credential:
        return _unverified(key, "No API key to verify.")
    base = (
        string_setting(settings, descriptor.base_url_attr)
        or descriptor.default_base_url
    )
    if base != descriptor.default_base_url:
        return _unverified(
            key, "Key verification is unavailable for this custom endpoint."
        )
    assert base is not None
    url = (
        probe.path
        if probe.path.startswith("https://")
        else base.rstrip("/") + probe.path
    )
    headers = {"Accept": "application/json"}
    if probe.provider_id == "gemini":
        headers["x-goog-api-key"] = credential
    else:
        headers["Authorization"] = f"Bearer {credential}"
    if probe.provider_id == "novita":
        headers["Content-Type"] = "application/json"
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT):
            async with httpx.AsyncClient(
                proxy=string_setting(settings, descriptor.proxy_attr),
                timeout=REQUEST_TIMEOUT,
                follow_redirects=False,
            ) as client:
                response = await client.get(url, headers=headers)
                return _interpret(key, probe, response)
    except TimeoutError, httpx.RequestError, UnicodeError:
        return _unverified(key, "Could not reach the provider to verify this key.")


async def check_credentials(
    settings: Settings, changed_keys: tuple[str, ...]
) -> tuple[CredentialCheck, ...]:
    """Check distinct edited credentials without touching runtime or model state."""
    credential_keys = {
        descriptor.credential_env
        for descriptor in PROVIDER_CATALOG.values()
        if not descriptor.local and descriptor.credential_env is not None
    }
    keys = tuple(sorted(set(changed_keys) & credential_keys))
    results = {
        key: _unverified(
            key,
            "The key check timed out."
            if key in _PROBE_BY_KEY
            else "Automatic key verification is unavailable for this provider.",
        )
        for key in keys
    }
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def run(key: str, probe: _Probe) -> None:
        async with semaphore:
            results[key] = await _check_one(settings, key, probe)

    try:
        async with asyncio.timeout(BATCH_TIMEOUT):
            async with asyncio.TaskGroup() as group:
                for key in keys:
                    if probe := _PROBE_BY_KEY.get(key):
                        group.create_task(run(key, probe))
    except TimeoutError:
        pass  # Finished results survive; unfinished checks retain their warning.
    return tuple(results[key] for key in keys)

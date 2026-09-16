from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request

from free_claude_code.api.dependencies import (
    get_services,
    get_settings,
    require_anthropic_proxy_auth,
    require_proxy_auth,
    resolve_provider,
)
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.ports import RequestRuntimeLease
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app


def _request(
    *,
    headers: dict[str, str],
    token: str = "freecc",
    enabled: bool = True,
) -> tuple[Request, Settings]:
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [
                (key.lower().encode(), value.encode()) for key, value in headers.items()
            ],
        }
    )
    settings = Settings.model_construct(
        proxy_auth_enabled=enabled,
        proxy_auth_token=token,
    )
    return request, settings


def _lease(*, provider=None, error: Exception | None = None):
    lease = MagicMock(spec=RequestRuntimeLease)
    lease.is_provider_cached.return_value = False
    if error is None:
        lease.resolve_provider.return_value = provider or MagicMock()
    else:
        lease.resolve_provider.side_effect = error
    return lease


def test_get_services_reads_the_single_app_state_boundary() -> None:
    app = create_test_app()
    request = Request({"type": "http", "app": app})

    services = get_services(request)

    assert services is app.state.services
    assert isinstance(services, ApiServices)


def test_get_settings_reads_current_request_runtime_settings() -> None:
    app = create_test_app(
        Settings.model_construct(
            model="deepseek/test-model",
            proxy_auth_enabled=False,
            proxy_auth_token="freecc",
        )
    )

    assert get_settings(app.state.services).model == "deepseek/test-model"


@pytest.mark.asyncio
async def test_resolve_provider_uses_retained_lease_and_logs_first_initialization() -> (
    None
):
    provider = MagicMock()
    lease = _lease(provider=provider)

    with patch("free_claude_code.api.dependencies.logger.info") as log_info:
        result = await resolve_provider("nvidia_nim", lease=lease)

    assert result is provider
    lease.resolve_provider.assert_called_once_with("nvidia_nim")
    log_info.assert_called_once_with("Provider initialized: {}", "nvidia_nim")


@pytest.mark.asyncio
async def test_resolve_provider_skips_initialization_log_for_cached_provider() -> None:
    lease = _lease()
    lease.is_provider_cached.return_value = True

    with patch("free_claude_code.api.dependencies.logger.info") as log_info:
        await resolve_provider("nvidia_nim", lease=lease)

    log_info.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_provider_missing_key_preserves_readiness_error() -> None:
    lease = _lease(
        error=ApplicationUnavailableError(
            "OPENROUTER_API_KEY is required. Get one at https://openrouter.ai"
        )
    )

    with pytest.raises(ApplicationUnavailableError) as exc_info:
        await resolve_provider("open_router", lease=lease)

    assert exc_info.value.status_code == 503
    assert "OPENROUTER_API_KEY" in exc_info.value.message
    assert "openrouter.ai" in exc_info.value.message


@pytest.mark.asyncio
async def test_resolve_provider_unrelated_error_is_not_reclassified() -> None:
    lease = _lease(error=ValueError("unrelated config"))

    with pytest.raises(ValueError, match="unrelated config"):
        await resolve_provider("nvidia_nim", lease=lease)


def test_require_proxy_auth_allows_when_disabled_with_retained_token():
    request, settings = _request(headers={}, enabled=False)

    require_proxy_auth(request, settings)


def test_require_proxy_auth_rejects_missing_authorization():
    request, settings = _request(headers={}, token="secret")

    with pytest.raises(HTTPException) as exc_info:
        require_proxy_auth(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing proxy authentication token"


@pytest.mark.parametrize("header_name", ["x-api-key", "anthropic-auth-token"])
def test_require_proxy_auth_rejects_legacy_header_only(header_name: str):
    request, settings = _request(headers={header_name: "secret"}, token="secret")

    with pytest.raises(HTTPException) as exc_info:
        require_proxy_auth(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing proxy authentication token"


def test_require_proxy_auth_accepts_exact_bearer_token():
    request, settings = _request(
        headers={"authorization": "bEaReR secret"},
        token="secret",
    )

    require_proxy_auth(request, settings)


def test_require_proxy_auth_accepts_colons_in_configured_token():
    request, settings = _request(
        headers={"authorization": "Bearer secret:with:colons"},
        token="secret:with:colons",
    )

    require_proxy_auth(request, settings)


def test_require_proxy_auth_accepts_valid_bearer_with_conflicting_legacy_headers():
    request, settings = _request(
        headers={
            "authorization": "Bearer secret",
            "x-api-key": "wrong",
            "anthropic-auth-token": "also-wrong",
        },
        token="secret",
    )

    require_proxy_auth(request, settings)


@pytest.mark.parametrize(
    "authorization",
    [
        "secret",
        "Basic secret",
        "Bearer",
        "Bearer wrong",
        "Bearer secret:claude-sonnet",
    ],
)
def test_require_proxy_auth_rejects_malformed_or_invalid_authorization(
    authorization: str,
):
    request, settings = _request(
        headers={"authorization": authorization},
        token="secret",
    )

    with pytest.raises(HTTPException) as exc_info:
        require_proxy_auth(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid proxy authentication token"


def test_require_proxy_auth_rejects_invalid_bearer_when_legacy_header_matches():
    request, settings = _request(
        headers={"authorization": "Bearer wrong", "x-api-key": "secret"},
        token="secret",
    )

    with pytest.raises(HTTPException) as exc_info:
        require_proxy_auth(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid proxy authentication token"


def test_require_anthropic_proxy_auth_allows_when_disabled() -> None:
    request, settings = _request(
        headers={
            "authorization": "Basic wrong",
            "x-api-key": "wrong",
            "anthropic-auth-token": "wrong",
        },
        token="retained-token",
        enabled=False,
    )

    require_anthropic_proxy_auth(request, settings)


def test_require_anthropic_proxy_auth_accepts_exact_bearer_token() -> None:
    request, settings = _request(
        headers={"authorization": "  bEaReR secret:with:colons  "},
        token="secret:with:colons",
    )

    require_anthropic_proxy_auth(request, settings)


def test_require_anthropic_proxy_auth_accepts_exact_x_api_key() -> None:
    request, settings = _request(
        headers={"x-api-key": "  secret:with:colons  "},
        token="secret:with:colons",
    )

    require_anthropic_proxy_auth(request, settings)


def test_require_anthropic_proxy_auth_rejects_missing_credentials() -> None:
    request, settings = _request(headers={}, token="secret")

    with pytest.raises(HTTPException) as exc_info:
        require_anthropic_proxy_auth(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing proxy authentication token"


@pytest.mark.parametrize("x_api_key", ["", "   ", "wrong", "secret:model-suffix"])
def test_require_anthropic_proxy_auth_rejects_invalid_x_api_key(
    x_api_key: str,
) -> None:
    request, settings = _request(
        headers={"x-api-key": x_api_key},
        token="secret",
    )

    with pytest.raises(HTTPException) as exc_info:
        require_anthropic_proxy_auth(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid proxy authentication token"


@pytest.mark.parametrize(
    "authorization",
    ["", "   ", "secret", "Basic secret", "Bearer", "Bearer wrong"],
)
def test_require_anthropic_proxy_auth_rejects_authorization_before_x_api_key(
    authorization: str,
) -> None:
    request, settings = _request(
        headers={
            "authorization": authorization,
            "x-api-key": "secret",
        },
        token="secret",
    )

    with pytest.raises(HTTPException) as exc_info:
        require_anthropic_proxy_auth(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid proxy authentication token"


def test_require_anthropic_proxy_auth_prefers_valid_authorization() -> None:
    request, settings = _request(
        headers={
            "authorization": "Bearer secret",
            "x-api-key": "wrong",
        },
        token="secret",
    )

    require_anthropic_proxy_auth(request, settings)


def test_require_anthropic_proxy_auth_rejects_anthropic_auth_token_only() -> None:
    request, settings = _request(
        headers={"anthropic-auth-token": "secret"},
        token="secret",
    )

    with pytest.raises(HTTPException) as exc_info:
        require_anthropic_proxy_auth(request, settings)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing proxy authentication token"

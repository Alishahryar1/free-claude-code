import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from api.dependencies import get_provider, get_settings, cleanup_provider
from providers.lmstudio import LMStudioProvider
from providers.nvidia_nim import NvidiaNimProvider
from providers.open_router import OpenRouterProvider
from config.nim import NimSettings


def _make_mock_settings(**overrides):
    """Create a mock settings object with all required fields for get_provider()."""
    mock = MagicMock()
    mock.provider_type = "nvidia_nim"
    mock.nvidia_nim_api_key = "test_key"
    mock.provider_rate_limit = 40
    mock.provider_rate_window = 60
    mock.open_router_api_key = "test_openrouter_key"
    mock.lm_studio_base_url = "http://localhost:1234/v1"
    mock.nim = NimSettings()
    mock.http_read_timeout = 300.0
    mock.http_write_timeout = 10.0
    mock.http_connect_timeout = 2.0
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


@pytest.fixture(autouse=True)
def reset_provider():
    """Reset the global _provider singleton between tests."""
    with patch("api.dependencies._provider", None):
        yield


@pytest.mark.asyncio
async def test_get_provider_singleton():
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        p1 = get_provider()
        p2 = get_provider()

        assert isinstance(p1, NvidiaNimProvider)
        assert p1 is p2


@pytest.mark.asyncio
async def test_get_settings():
    settings = get_settings()
    assert settings is not None
    # Verify it calls the internal _get_settings
    with patch("api.dependencies._get_settings") as mock_get:
        get_settings()
        mock_get.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_provider():
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        provider = get_provider()
        assert isinstance(provider, NvidiaNimProvider)
        provider._client = AsyncMock()

        await cleanup_provider()

        provider._client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_provider_no_client():
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        provider = get_provider()
        if hasattr(provider, "_client"):
            del provider._client

        await cleanup_provider()
        # Should not raise


@pytest.mark.asyncio
async def test_get_provider_open_router():
    """Test that provider_type=open_router returns OpenRouterProvider."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(provider_type="open_router")

        provider = get_provider()

        assert isinstance(provider, OpenRouterProvider)
        assert provider._base_url == "https://openrouter.ai/api/v1"
        assert provider._api_key == "test_openrouter_key"


@pytest.mark.asyncio
async def test_get_provider_lmstudio():
    """Test that provider_type=lmstudio returns LMStudioProvider."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(provider_type="lmstudio")

        provider = get_provider()

        assert isinstance(provider, LMStudioProvider)
        assert provider._base_url == "http://localhost:1234/v1"
        assert provider._api_key == "lm-studio"


@pytest.mark.asyncio
async def test_get_provider_lmstudio_uses_lm_studio_base_url():
    """LM Studio provider uses lm_studio_base_url from settings."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(
            provider_type="lmstudio",
            lm_studio_base_url="http://custom:9999/v1",
        )

        provider = get_provider()

        assert isinstance(provider, LMStudioProvider)
        assert provider._base_url == "http://custom:9999/v1"


@pytest.mark.asyncio
async def test_get_provider_passes_http_timeouts_from_settings():
    """Provider receives http timeouts from settings when creating client."""
    with (
        patch("api.dependencies.get_settings") as mock_settings,
        patch("providers.openai_compat.AsyncOpenAI") as mock_openai,
    ):
        mock_settings.return_value = _make_mock_settings(
            http_read_timeout=600.0,
            http_write_timeout=20.0,
            http_connect_timeout=5.0,
        )
        provider = get_provider()
        assert isinstance(provider, NvidiaNimProvider)
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 20.0
        assert timeout.connect == 5.0


@pytest.mark.asyncio
async def test_get_provider_unknown_type():
    """Test that unknown provider_type raises ValueError."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(provider_type="unknown")

        with pytest.raises(ValueError, match="Unknown provider_type"):
            get_provider()


@pytest.mark.asyncio
async def test_cleanup_provider_aclose_raises():
    """cleanup_provider handles aclose() raising an exception."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        provider = get_provider()
        assert isinstance(provider, NvidiaNimProvider)
        provider._client = AsyncMock()
        provider._client.aclose = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        # Should propagate the error (current behavior - no try/except)
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await cleanup_provider()

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from api.dependencies import get_provider, get_settings, cleanup_provider
from providers.nvidia_nim import NvidiaNimProvider
from config.nim import NimSettings


def _make_mock_settings(**overrides):
    """Create a mock settings object with all required fields for get_provider()."""
    mock = MagicMock()
    mock.provider_type = "nvidia_nim"
    mock.nvidia_nim_api_key = "test_key"
    mock.nvidia_nim_rate_limit = 40
    mock.nvidia_nim_rate_window = 60
    mock.openrouter_api_key = "test_openrouter_key"
    mock.openrouter_base_url = "https://openrouter.ai/api/v1"
    mock.openrouter_rate_limit = 20
    mock.openrouter_rate_window = 60
    mock.nim = NimSettings()
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
async def test_get_provider_openrouter():
    """Test that openrouter provider_type returns OpenRouterProvider."""
    from providers.openrouter import OpenRouterProvider

    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(provider_type="openrouter")

        provider = get_provider()
        assert isinstance(provider, OpenRouterProvider)
        assert provider._api_key == "test_openrouter_key"
        assert provider._base_url == "https://openrouter.ai/api/v1"


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

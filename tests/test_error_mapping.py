"""Tests for providers/nvidia_nim/errors.py error mapping."""

import pytest
from unittest.mock import patch, MagicMock

import openai
from httpx import Response, Request

from providers.nvidia_nim.errors import map_error
from providers.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    OverloadedError,
    APIError,
)


def _make_openai_error(cls, message="test error", status_code=None):
    """Helper to create openai exceptions with required httpx objects."""
    response = Response(
        status_code=status_code or 500, request=Request("POST", "http://test")
    )
    body = {"error": {"message": message}}
    # openai.APIError base class has a different constructor signature
    if cls is openai.APIError:
        return cls(message, request=Request("POST", "http://test"), body=body)
    return cls(message, response=response, body=body)


class TestMapError:
    """Tests for map_error function."""

    def test_authentication_error(self):
        """openai.AuthenticationError -> AuthenticationError."""
        exc = _make_openai_error(openai.AuthenticationError, status_code=401)
        result = map_error(exc)
        assert isinstance(result, AuthenticationError)
        assert result.status_code == 401

    def test_rate_limit_error(self):
        """openai.RateLimitError -> RateLimitError and triggers global block."""
        exc = _make_openai_error(openai.RateLimitError, status_code=429)
        with patch("providers.nvidia_nim.errors.GlobalRateLimiter") as mock_rl:
            mock_instance = MagicMock()
            mock_rl.get_instance.return_value = mock_instance
            result = map_error(exc)
            assert isinstance(result, RateLimitError)
            assert result.status_code == 429
            mock_instance.set_blocked.assert_called_once_with(60)

    def test_bad_request_error(self):
        """openai.BadRequestError -> InvalidRequestError."""
        exc = _make_openai_error(openai.BadRequestError, status_code=400)
        result = map_error(exc)
        assert isinstance(result, InvalidRequestError)
        assert result.status_code == 400

    @pytest.mark.parametrize(
        "message",
        ["Server overloaded", "No capacity available"],
        ids=["overloaded", "capacity"],
    )
    def test_internal_server_error_overloaded(self, message):
        """InternalServerError with overloaded/capacity keywords -> OverloadedError."""
        exc = _make_openai_error(
            openai.InternalServerError, message=message, status_code=500
        )
        result = map_error(exc)
        assert isinstance(result, OverloadedError)
        assert result.status_code == 529

    def test_internal_server_error_generic(self):
        """InternalServerError without keywords -> APIError(500)."""
        exc = _make_openai_error(
            openai.InternalServerError, message="Unknown error", status_code=500
        )
        result = map_error(exc)
        assert isinstance(result, APIError)
        assert result.status_code == 500

    def test_generic_api_error(self):
        """openai.APIError -> APIError with original status_code."""
        exc = _make_openai_error(
            openai.APIError, message="Bad gateway", status_code=502
        )
        result = map_error(exc)
        assert isinstance(result, APIError)

    def test_unmapped_exception_passthrough(self):
        """Non-openai exceptions are returned as-is."""
        exc = RuntimeError("unexpected")
        result = map_error(exc)
        assert result is exc
        assert isinstance(result, RuntimeError)

    def test_value_error_passthrough(self):
        """ValueError passes through unchanged."""
        exc = ValueError("bad value")
        result = map_error(exc)
        assert result is exc

    @pytest.mark.parametrize(
        "exc_cls,expected_cls",
        [
            (openai.AuthenticationError, AuthenticationError),
            (openai.RateLimitError, RateLimitError),
            (openai.BadRequestError, InvalidRequestError),
        ],
        ids=["auth", "rate_limit", "bad_request"],
    )
    def test_mapping_parametrized(self, exc_cls, expected_cls):
        """Parametrized check of openai -> provider error mapping."""
        status_map = {
            openai.AuthenticationError: 401,
            openai.RateLimitError: 429,
            openai.BadRequestError: 400,
        }
        exc = _make_openai_error(exc_cls, status_code=status_map[exc_cls])
        with patch("providers.nvidia_nim.errors.GlobalRateLimiter"):
            result = map_error(exc)
        assert isinstance(result, expected_cls)

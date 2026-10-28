from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from api import skillopt
from api.model_router import ModelRouter
from api.models.anthropic import Message, MessagesRequest
from api.services import ClaudeProxyService
from config.settings import Settings
from providers.base import BaseProvider
from providers.exceptions import AuthenticationError, ServiceUnavailableError


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.model = "openai/gpt-4o"
    s.model_opus = None
    s.model_sonnet = None
    s.model_haiku = None
    s.enable_model_thinking = False
    s.anthropic_auth_token = ""
    return s


class _StubProvider(BaseProvider):
    """A minimal BaseProvider that lets a test control preflight behavior."""

    def __init__(self, provider_id: str, preflight_raises: BaseException | None = None):
        from providers.base import ProviderConfig

        super().__init__(ProviderConfig(api_key="stub", base_url="http://stub"))
        self.provider_id = provider_id
        self.preflight_called = False
        self.stream_called = False
        self._preflight_raises = preflight_raises

    def preflight_stream(
        self, request: Any, *, thinking_enabled: bool | None = None
    ) -> None:
        self.preflight_called = True
        if self._preflight_raises is not None:
            raise self._preflight_raises

    async def cleanup(self) -> None:
        return None

    async def list_model_ids(self) -> frozenset[str]:
        return frozenset()

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ):
        self.stream_called = True
        yield "data: {}\n\n"


def _enable_skillopt(monkeypatch, tmp_path, policy: dict[str, Any]) -> None:
    monkeypatch.setenv("SKILLOPT_ENABLED", "1")
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    (tmp_path / "skillopt_policy.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    skillopt.invalidate_cache()


def _make_service(settings, providers: dict[str, _StubProvider]) -> ClaudeProxyService:
    def _getter(provider_id: str) -> BaseProvider:
        return providers[provider_id]

    def _token_counter(messages, system, tools) -> int:
        return 100

    return ClaudeProxyService(
        settings,
        provider_getter=_getter,
        model_router=ModelRouter(settings),
        token_counter=_token_counter,
    )


def _request() -> MessagesRequest:
    # "fix the bug" → infer_skill returns "edit"
    return MessagesRequest(
        model="claude-3-opus-20240229",
        max_tokens=100,
        messages=[Message(role="user", content="fix the bug in this file")],
    )


def test_primary_succeeds_no_fallback_consulted(settings, monkeypatch, tmp_path):
    _enable_skillopt(
        monkeypatch,
        tmp_path,
        {
            "version": 1,
            "policies": {
                "edit": {
                    "primary": "openai/gpt-4o",
                    "fallbacks": ["deepseek/deepseek-chat"],
                }
            },
        },
    )
    primary = _StubProvider("openai")
    fallback = _StubProvider("deepseek")
    service = _make_service(settings, {"openai": primary, "deepseek": fallback})

    response = service.create_message(_request())

    assert primary.preflight_called is True
    assert fallback.preflight_called is False
    assert response is not None


def test_primary_auth_error_falls_back_to_next(settings, monkeypatch, tmp_path):
    _enable_skillopt(
        monkeypatch,
        tmp_path,
        {
            "version": 1,
            "policies": {
                "edit": {
                    "primary": "openai/gpt-4o",
                    "fallbacks": ["deepseek/deepseek-chat"],
                }
            },
        },
    )
    primary = _StubProvider(
        "openai", preflight_raises=AuthenticationError("key revoked")
    )
    fallback = _StubProvider("deepseek")
    service = _make_service(settings, {"openai": primary, "deepseek": fallback})

    response = service.create_message(_request())

    assert primary.preflight_called is True
    assert fallback.preflight_called is True
    assert response is not None


def test_primary_service_unavailable_falls_back(settings, monkeypatch, tmp_path):
    _enable_skillopt(
        monkeypatch,
        tmp_path,
        {
            "version": 1,
            "policies": {
                "edit": {
                    "primary": "openai/gpt-4o",
                    "fallbacks": ["deepseek/deepseek-chat"],
                }
            },
        },
    )
    primary = _StubProvider(
        "openai", preflight_raises=ServiceUnavailableError("down")
    )
    fallback = _StubProvider("deepseek")
    service = _make_service(settings, {"openai": primary, "deepseek": fallback})

    service.create_message(_request())

    assert fallback.preflight_called is True


def test_all_candidates_fail_re_raises_primary_error(
    settings, monkeypatch, tmp_path
):
    _enable_skillopt(
        monkeypatch,
        tmp_path,
        {
            "version": 1,
            "policies": {
                "edit": {
                    "primary": "openai/gpt-4o",
                    "fallbacks": ["deepseek/deepseek-chat"],
                }
            },
        },
    )
    primary = _StubProvider(
        "openai", preflight_raises=AuthenticationError("primary bad")
    )
    fallback = _StubProvider(
        "deepseek", preflight_raises=AuthenticationError("fallback bad")
    )
    service = _make_service(settings, {"openai": primary, "deepseek": fallback})

    with pytest.raises(AuthenticationError, match="primary bad"):
        service.create_message(_request())


def test_fallback_pointing_at_unsupported_provider_is_skipped(
    settings, monkeypatch, tmp_path
):
    _enable_skillopt(
        monkeypatch,
        tmp_path,
        {
            "version": 1,
            "policies": {
                "edit": {
                    "primary": "openai/gpt-4o",
                    "fallbacks": ["bogus/model", "deepseek/deepseek-chat"],
                }
            },
        },
    )
    primary = _StubProvider(
        "openai", preflight_raises=AuthenticationError("down")
    )
    fallback = _StubProvider("deepseek")
    service = _make_service(settings, {"openai": primary, "deepseek": fallback})

    service.create_message(_request())

    assert fallback.preflight_called is True


def test_skillopt_disabled_disables_fallback(settings, monkeypatch, tmp_path):
    """With SkillOpt off, a primary preflight error must propagate."""
    monkeypatch.delenv("SKILLOPT_ENABLED", raising=False)
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    skillopt.invalidate_cache()

    primary = _StubProvider(
        "openai", preflight_raises=AuthenticationError("boom")
    )
    service = _make_service(settings, {"openai": primary})

    with pytest.raises(AuthenticationError, match="boom"):
        service.create_message(_request())

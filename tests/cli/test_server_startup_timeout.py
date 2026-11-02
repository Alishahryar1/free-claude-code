"""Tests for the env-configurable launcher readiness budget."""

import functools
from types import SimpleNamespace

import pytest

from free_claude_code.cli.launchers import common


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from free_claude_code.config import clear_settings_cache

    clear_settings_cache()
    yield
    clear_settings_cache()


def _patch_settings(monkeypatch, value):
    stub = functools.lru_cache(maxsize=None)(
        lambda: SimpleNamespace(server_startup_timeout=value)
    )
    monkeypatch.setattr("free_claude_code.config.loader.get_settings", stub)


def test_setting_overrides_module_constant(monkeypatch) -> None:
    _patch_settings(monkeypatch, 45.0)
    monkeypatch.setattr(common, "SERVER_STARTUP_TIMEOUT_SECONDS", 30.0)
    assert common._server_startup_timeout() == 45.0


def test_unset_setting_falls_back_to_constant(monkeypatch) -> None:
    _patch_settings(monkeypatch, None)
    monkeypatch.setattr(common, "SERVER_STARTUP_TIMEOUT_SECONDS", 7.0)
    assert common._server_startup_timeout() == 7.0


def test_non_positive_setting_falls_back_to_constant(monkeypatch) -> None:
    _patch_settings(monkeypatch, 0)
    monkeypatch.setattr(common, "SERVER_STARTUP_TIMEOUT_SECONDS", 9.0)
    assert common._server_startup_timeout() == 9.0


def test_env_alias_round_trips_through_settings(monkeypatch) -> None:
    # FCC_SERVER_STARTUP_TIMEOUT_SECONDS reaches Settings via the loader's
    # process-env pass, so shell exports and .env/Admin values share one path.
    from free_claude_code.config import clear_settings_cache, get_settings

    monkeypatch.setenv("FCC_SERVER_STARTUP_TIMEOUT_SECONDS", "35")
    clear_settings_cache()
    try:
        assert get_settings().server_startup_timeout == 35.0
    finally:
        clear_settings_cache()


def test_default_budget_is_one_hundred_twenty_seconds() -> None:
    assert common.SERVER_STARTUP_TIMEOUT_SECONDS == 120.0

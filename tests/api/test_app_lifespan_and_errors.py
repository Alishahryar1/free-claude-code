import logging
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from free_claude_code.application.errors import (
    ApplicationUnavailableError,
    InvalidRequestError,
)
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.settings import Settings
from free_claude_code.messaging.transcription import TranscriptionService
from free_claude_code.providers.nvidia_nim.client import NvidiaNimProvider
from free_claude_code.providers.nvidia_nim.voice import NvidiaNimTranscriber
from free_claude_code.runtime.application import (
    ApplicationRuntime,
    startup_failure_message,
)
from free_claude_code.runtime.asgi import RuntimeASGIApp
from free_claude_code.runtime.bootstrap import _create_transcriber, build_asgi_app
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.integrations.discovery import LocalInstallations
from free_claude_code.runtime.integrations.service import IntegrationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.api.support import create_test_app


def _settings(**updates: object) -> Settings:
    return Settings().model_copy(update=updates)


@pytest.fixture(autouse=True)
def _redirect_fcc_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


@pytest.mark.asyncio
async def test_runtime_startup_logs_admin_url_without_printed_server_banner():
    settings = _settings(
        messaging_platform="none",
        host="127.0.0.1",
        port=9099,
    )
    manager = ProviderRuntimeManager(settings)
    runtime = ApplicationRuntime(
        manager,
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=None,
    )
    uvicorn_logger = MagicMock()

    with (
        patch("builtins.print") as printed,
        patch.object(
            manager,
            "warm_referenced_model_cache",
            new=AsyncMock(),
        ) as warm_cache,
        patch.object(manager, "start_model_list_refresh") as start_refresh,
        patch.object(manager, "close", new=AsyncMock()),
        patch(
            "free_claude_code.runtime.application.messaging_platform_factory.create_messaging_components",
            return_value=None,
        ),
        patch.object(logging, "getLogger", return_value=uvicorn_logger) as get_logger,
    ):
        await runtime.start()
        await runtime.close()

    printed.assert_not_called()
    warm_cache.assert_awaited_once()
    start_refresh.assert_called_once()
    get_logger.assert_any_call("uvicorn.error")
    uvicorn_logger.info.assert_called_once_with(
        "Admin UI: %s (local-only)",
        "http://127.0.0.1:9099/admin",
    )


def test_create_app_application_error_handler_returns_anthropic_format():
    app = create_test_app(_settings(log_api_error_tracebacks=False))

    @app.get("/raise_application")
    async def _raise_application():
        raise InvalidRequestError("bad request")

    response = TestClient(app).get("/raise_application")

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["request_id"] == response.headers["request-id"]
    assert "x-should-retry" not in response.headers


def test_application_error_handler_does_not_log_error_message():
    app = create_test_app(_settings(log_api_error_tracebacks=False))
    secret = "provider-upstream-secret-detail"

    @app.get("/raise_application_secret")
    async def _raise_application_secret():
        raise InvalidRequestError(secret)

    with patch("free_claude_code.api.app.logger.error") as log_error:
        response = TestClient(app).get("/raise_application_secret")

    assert response.status_code == 400
    blob = " ".join(
        str(value) for call in log_error.call_args_list for value in call.args
    )
    assert secret not in blob
    log_error.assert_not_called()


def test_create_app_general_exception_handler_returns_correlated_500():
    app = create_test_app(_settings(log_api_error_tracebacks=False))

    @app.get("/raise_general")
    async def _raise_general():
        raise RuntimeError("boom")

    response = TestClient(app, raise_server_exceptions=False).get("/raise_general")

    assert response.status_code == 500
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert body["request_id"] == response.headers["request-id"]


def test_general_exception_default_log_excludes_exception_message():
    app = create_test_app(_settings(log_api_error_tracebacks=False))
    secret = "user-provided-secret-token-xyzzy"

    @app.get("/raise_secret")
    async def _raise_secret():
        raise ValueError(secret)

    with patch("free_claude_code.api.app.logger.error") as log_error:
        response = TestClient(app, raise_server_exceptions=False).get("/raise_secret")

    assert response.status_code == 500
    blob = " ".join(
        str(value) for call in log_error.call_args_list for value in call.args
    )
    assert secret not in blob
    assert "ValueError" in blob


@pytest.mark.asyncio
async def test_runtime_startup_warms_catalog_before_background_refresh():
    settings = _settings(messaging_platform="none")
    manager = ProviderRuntimeManager(settings)
    runtime = ApplicationRuntime(
        manager,
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=None,
    )
    events: list[str] = []

    async def warm_cache() -> None:
        events.append("warm")

    def start_refresh() -> None:
        events.append("background")

    with (
        patch.object(
            manager,
            "warm_referenced_model_cache",
            side_effect=warm_cache,
        ) as warm,
        patch.object(
            manager,
            "start_model_list_refresh",
            side_effect=start_refresh,
        ) as refresh,
        patch.object(manager, "close", new=AsyncMock()),
        patch(
            "free_claude_code.runtime.application.messaging_platform_factory.create_messaging_components",
            return_value=None,
        ),
    ):
        await runtime.start()
        await runtime.close()

    warm.assert_awaited_once()
    refresh.assert_called_once()
    assert events == ["warm", "background"]


def test_startup_failure_message_preserves_existing_concise_contract():
    quiet = _settings(log_api_error_tracebacks=False)
    verbose = _settings(log_api_error_tracebacks=True)

    assert startup_failure_message(quiet, RuntimeError("secret")) == (
        "Server startup failed: exc_type=RuntimeError"
    )
    assert startup_failure_message(verbose, RuntimeError("visible")) == (
        "RuntimeError: visible"
    )
    assert (
        startup_failure_message(
            quiet,
            ApplicationUnavailableError("configured model is unavailable"),
        )
        == "configured model is unavailable"
    )


@pytest.mark.asyncio
async def test_runtime_asgi_app_starts_and_closes_owner_once():
    runtime = MagicMock(spec=ApplicationRuntime)
    runtime.settings = _settings()
    runtime.start = AsyncMock()
    runtime.close = AsyncMock(return_value=True)
    app = RuntimeASGIApp(AsyncMock(), runtime)
    received = iter(
        [
            {"type": "lifespan.startup"},
            {"type": "lifespan.shutdown"},
        ]
    )
    sent: list[dict[str, str]] = []

    async def receive():
        return next(received)

    async def send(message):
        sent.append(message)

    await app({"type": "lifespan"}, receive, send)

    runtime.start.assert_awaited_once()
    runtime.close.assert_awaited_once()
    assert sent == [
        {"type": "lifespan.startup.complete"},
        {"type": "lifespan.shutdown.complete"},
    ]


@pytest.mark.asyncio
async def test_runtime_asgi_app_reports_incomplete_owned_shutdown() -> None:
    runtime = MagicMock(spec=ApplicationRuntime)
    runtime.settings = _settings()
    runtime.start = AsyncMock()
    runtime.close = AsyncMock(return_value=False)
    app = RuntimeASGIApp(AsyncMock(), runtime)
    received = iter(
        [
            {"type": "lifespan.startup"},
            {"type": "lifespan.shutdown"},
        ]
    )
    sent: list[dict[str, str]] = []

    async def receive():
        return next(received)

    async def send(message):
        sent.append(message)

    await app({"type": "lifespan"}, receive, send)

    assert sent == [
        {"type": "lifespan.startup.complete"},
        {"type": "lifespan.shutdown.failed", "message": ""},
    ]


@pytest.mark.asyncio
async def test_runtime_asgi_app_reports_concise_startup_failure():
    runtime = MagicMock(spec=ApplicationRuntime)
    runtime.settings = _settings(log_api_error_tracebacks=False)
    runtime.start = AsyncMock(side_effect=RuntimeError("secret"))
    runtime.close = AsyncMock()
    app = RuntimeASGIApp(AsyncMock(), runtime)
    sent: list[dict[str, str]] = []

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):
        sent.append(message)

    await app({"type": "lifespan"}, receive, send)

    assert sent == [
        {
            "type": "lifespan.startup.failed",
            "message": "Server startup failed: exc_type=RuntimeError",
        }
    ]
    runtime.close.assert_not_awaited()


def test_bootstrap_configures_default_log_and_publishes_only_services(tmp_path):
    log_path = tmp_path / "server.log"
    settings = _settings()

    with (
        patch(
            "free_claude_code.runtime.bootstrap.server_log_path",
            return_value=log_path,
        ),
        patch("free_claude_code.runtime.bootstrap.configure_logging") as configure,
    ):
        asgi_app = build_asgi_app(settings)

    configure.assert_called_once_with(
        Path(log_path),
        level=settings.log_level,
        verbose_third_party=settings.log_raw_api_payloads,
    )
    api_app = cast(FastAPI, asgi_app.app)
    assert set(api_app.state._state) == {"services"}


def test_bootstrap_app_transparently_exposes_fastapi_interface() -> None:
    with patch("free_claude_code.runtime.bootstrap.configure_logging"):
        asgi_app = build_asgi_app(_settings())

    api_app = cast(FastAPI, asgi_app.app)
    assert asgi_app.router is api_app.router
    assert asgi_app.routes is api_app.routes
    assert asgi_app.state is api_app.state
    assert asgi_app.openapi == api_app.openapi


def test_bootstrap_wires_the_codex_catalog_publisher() -> None:
    publisher = MagicMock()

    with (
        patch("free_claude_code.runtime.bootstrap.configure_logging"),
        patch(
            "free_claude_code.runtime.bootstrap.CodexModelCatalogPublisher",
            return_value=publisher,
        ) as publisher_type,
    ):
        asgi_app = build_asgi_app(_settings())

    manager = asgi_app.runtime.provider_manager
    manager.cache_model_infos(
        "nvidia_nim",
        {ProviderModelInfo("published-model")},
    )

    publisher_type.assert_called_once_with()
    publisher.publish.assert_called_once_with(manager)


def test_bootstrap_honors_process_log_file_override(monkeypatch, tmp_path):
    log_path = tmp_path / "custom.log"
    monkeypatch.setenv("LOG_FILE", str(log_path))

    with patch("free_claude_code.runtime.bootstrap.configure_logging") as configure:
        build_asgi_app(_settings())

    assert configure.call_args.args[0] == log_path


def test_bootstrap_constructs_fresh_runtime_owned_transcribers() -> None:
    settings = _settings(voice_note_enabled=True, whisper_device="cpu")

    first = _create_transcriber(settings)
    second = _create_transcriber(settings)

    assert isinstance(first, TranscriptionService)
    assert isinstance(second, TranscriptionService)
    assert first is not second


@pytest.mark.asyncio
async def test_bootstrap_constructs_isolated_runtime_resource_graphs() -> None:
    settings = _settings(
        model="nvidia_nim/test-model",
        nvidia_nim_api_key="test-key",
        voice_note_enabled=True,
        whisper_device="cpu",
    )

    with patch("free_claude_code.runtime.bootstrap.configure_logging"):
        first = build_asgi_app(settings)
        second = build_asgi_app(settings)

    first_lease = await first.runtime.provider_manager.acquire()
    second_lease = await second.runtime.provider_manager.acquire()
    try:
        first_provider = first_lease.resolve_provider("nvidia_nim")
        second_provider = second_lease.resolve_provider("nvidia_nim")

        assert isinstance(first_provider, NvidiaNimProvider)
        assert isinstance(second_provider, NvidiaNimProvider)
        assert first_provider._admission is not second_provider._admission
        assert first.runtime._transcriber is not second.runtime._transcriber
    finally:
        await first_lease.release()
        await second_lease.release()
        await first.runtime.close()
        await second.runtime.close()


def test_bootstrap_selects_nvidia_transcriber_without_loading_riva() -> None:
    settings = _settings(
        voice_note_enabled=True,
        whisper_device="nvidia_nim",
        whisper_model="openai/whisper-large-v3",
        nvidia_nim_api_key="nvapi-test",
    )

    assert isinstance(_create_transcriber(settings), NvidiaNimTranscriber)


def test_bootstrap_disables_transcription_as_one_owned_resource() -> None:
    assert _create_transcriber(_settings(voice_note_enabled=False)) is None

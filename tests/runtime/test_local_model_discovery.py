"""Local discovery uses the shared catalog without requiring a selected model."""

import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from free_claude_code.application.model_catalog import read_model_catalog
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.loader import compose_settings_snapshot
from free_claude_code.config.model_refs import ModelFallback
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime import provider_manager
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.runtime.catalog_support import finish_discovery, finish_publication

LOCAL_IDS = ("lmstudio", "llamacpp", "ollama")


def provider(*ids, error=None):
    value = MagicMock(spec=BaseProvider)
    value.list_model_infos = AsyncMock(
        return_value=frozenset(ProviderModelInfo(model) for model in ids),
        side_effect=error,
    )
    value.cleanup = AsyncMock()
    return value


def manager_for(settings=None, providers=None, **kwargs):
    settings = settings or Settings()
    providers = providers or {name: provider(name + "-model") for name in LOCAL_IDS}
    all_providers = {
        "nvidia_nim": provider("cloud-model"),
        **{name: provider(error=ConnectionError()) for name in LOCAL_IDS},
        **providers,
    }
    return ProviderRuntimeManager(
        settings,
        runtime_factory=lambda snapshot, registry: ProviderRuntime(
            snapshot,
            registry,
            {(name, 1): value for name, value in all_providers.items()},
        ),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_startup_discovers_unselected_locals_once_and_publishes_them():
    providers = {name: provider(name + "-model") for name in LOCAL_IDS}
    publisher = MagicMock()
    manager = manager_for(providers=providers, model_catalog_publisher=publisher)
    try:
        manager.start_model_list_refresh()
        await finish_discovery(manager)
        manager.start_model_list_refresh()
        await finish_discovery(manager)
        await finish_publication(manager)
        assert set(LOCAL_IDS) <= manager.cached_model_ids().keys()
        assert all(
            value.list_model_infos.await_count == 1 for value in providers.values()
        )
        published = read_model_catalog(publisher.publish.call_args.args[0])
        assert {name + "/" + name + "-model" for name in LOCAL_IDS} <= {
            model.provider_model_ref for model in published.models
        }
        assert (
            published.default_model_id == read_model_catalog(manager).default_model_id
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cancelled_local_waiter_preserves_probe_and_refresh_recovers_inventory():
    entered, release = asyncio.Event(), asyncio.Event()

    async def held():
        entered.set()
        await release.wait()
        return frozenset({ProviderModelInfo("installed")})

    local = provider(error=held)
    manager = manager_for(providers={"ollama": local})
    waiter = asyncio.create_task(manager.refresh_provider("ollama"))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await finish_discovery(manager)
        assert local.list_model_infos.await_count == 1
        assert manager.cached_model_ids()["ollama"] == {"installed"}
        local.list_model_infos.side_effect = ConnectionError()
        assert (await manager.refresh_provider("ollama")).failed_provider_ids == (
            "ollama",
        )
        assert manager.cached_model_ids()["ollama"] == {"installed"}
        local.list_model_infos.side_effect = None
        local.list_model_infos.return_value = frozenset(
            {ProviderModelInfo("newly-installed")}
        )
        await manager.refresh_model_list_cache()
        assert manager.cached_model_ids()["ollama"] == {"newly-installed"}
    finally:
        release.set()
        await manager.close()


@pytest.mark.asyncio
async def test_shutdown_cancels_owned_local_probes_and_cleans_providers():
    started = set()
    cancelled = set()
    ready = asyncio.Event()

    def held(name):
        async def list_models():
            started.add(name)
            if started == set(LOCAL_IDS):
                ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(name)

        return list_models

    providers = {name: provider(error=held(name)) for name in LOCAL_IDS}
    manager = manager_for(providers=providers)
    try:
        manager.start_model_list_refresh()
        await asyncio.wait_for(ready.wait(), 1)
    finally:
        await asyncio.wait_for(manager.close(), 1)
    assert cancelled == set(LOCAL_IDS)
    for value in providers.values():
        value.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_deadlines_are_parallel_and_do_not_block_other_provider_readiness(
    monkeypatch,
):
    monkeypatch.setattr(
        provider_manager, "LOCAL_DISCOVERY_TIMEOUT_SECONDS", 0.05, raising=False
    )
    entered = set()
    cancelled = set()
    all_started = asyncio.Event()

    def held(name):
        async def list_models():
            entered.add(name)
            if len(entered) == len(LOCAL_IDS):
                all_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(name)

        return list_models

    providers = {name: provider(error=held(name)) for name in LOCAL_IDS}
    manager = manager_for(
        Settings.model_validate({"NVIDIA_NIM_API_KEY": "test"}), providers
    )
    try:
        manager.start_model_list_refresh()
        await asyncio.wait_for(all_started.wait(), 1)
        async with await manager.acquire() as lease:
            await asyncio.wait_for(lease.resolve_provider("nvidia_nim"), 1)
            assert lease.model_info("nvidia_nim", "cloud-model") is not None
        result = await asyncio.wait_for(manager.refresh_model_list_cache(), 1)
        assert not result.failed_provider_ids
        assert cancelled == set(LOCAL_IDS)
        status = manager.catalog_status()
        assert status["catalog"] == "ready"
        assert status["catalog_file"] == "ready"
        assert all(status["providers"][name] == "unavailable" for name in LOCAL_IDS)
    finally:
        await manager.close()


@pytest.mark.parametrize(
    "failure", [ConnectionError("private-url"), TimeoutError("private-url")]
)
@pytest.mark.asyncio
async def test_optional_absence_is_quiet_but_shared_manual_test_keeps_failure(
    failure, caplog
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def fail():
        entered.set()
        await release.wait()
        raise failure

    providers = {
        name: provider(error=fail if name == "ollama" else ConnectionError())
        for name in LOCAL_IDS
    }
    manager = manager_for(providers=providers)
    try:
        automatic = asyncio.create_task(manager.refresh_model_list_cache())
        await asyncio.wait_for(entered.wait(), 1)
        manual = asyncio.create_task(manager.refresh_provider("ollama"))
        await asyncio.sleep(0)
        release.set()
        assert not (await automatic).failed_provider_ids
        result = await manual
        assert result.failed_provider_ids == ("ollama",)
        assert "private-url" not in str(result.failure_messages)
        assert providers["ollama"].list_model_infos.await_count == 1
        assert manager.catalog_status()["providers"]["ollama"] == "unavailable"
        assert not [record for record in caplog.records if record.levelno >= 30]
    finally:
        release.set()
        await manager.close()


@pytest.mark.parametrize(
    "intent",
    ["managed_default", "process_default", "url", "proxy", "model", "fallback"],
)
@pytest.mark.asyncio
async def test_explicit_local_configuration_reports_failure(intent):
    default_url = Settings().ollama_base_url
    expected_provider = "lmstudio" if intent == "proxy" else "ollama"
    if intent == "managed_default":
        settings = compose_settings_snapshot(
            {"OLLAMA_BASE_URL": default_url}, {}
        ).settings
    elif intent == "process_default":
        settings = compose_settings_snapshot(
            {}, {"OLLAMA_BASE_URL": default_url}
        ).settings
    else:
        changes = {
            "url": {"ollama_base_url": "http://localhost:11435"},
            "proxy": {"lmstudio_proxy": "http://localhost:8000"},
            "model": {"model": "ollama/test-model"},
            "fallback": {
                "model_fallbacks": (ModelFallback(model="ollama/test-model"),)
            },
        }
        settings = Settings().model_copy(update=changes[intent])
    providers = {name: provider(error=ConnectionError()) for name in LOCAL_IDS}
    manager = manager_for(settings, providers)
    try:
        result = await manager.refresh_model_list_cache()
        assert result.failed_provider_ids == (expected_provider,)
        assert expected_provider in result.failure_messages
        assert manager.catalog_status()["providers"][expected_provider] == "failed"
    finally:
        await manager.close()


@pytest.mark.parametrize(
    "failure",
    [
        ModelListResponseError("private-response"),
        httpx.HTTPStatusError(
            "private-response",
            request=httpx.Request("GET", "http://localhost"),
            response=httpx.Response(401),
        ),
        RuntimeError("private-response"),
    ],
)
@pytest.mark.asyncio
async def test_responding_or_broken_optional_provider_reports_real_errors(failure):
    providers = {
        name: provider(error=failure if name == "ollama" else ConnectionError())
        for name in LOCAL_IDS
    }
    manager = manager_for(providers=providers)
    try:
        result = await manager.refresh_model_list_cache()
        assert result.failed_provider_ids == ("ollama",)
        assert "private-response" not in str(result.failure_messages)
        assert manager.catalog_status()["providers"]["ollama"] == "failed"
    finally:
        await manager.close()


@pytest.mark.parametrize(
    ("local_id", "attr"),
    [
        (name, attr)
        for name in LOCAL_IDS
        for attr in (
            PROVIDER_CATALOG[name].base_url_attr,
            PROVIDER_CATALOG[name].proxy_attr,
        )
        if attr is not None
    ],
)
@pytest.mark.asyncio
async def test_local_endpoint_change_discards_old_inventory_on_failed_rediscovery(
    local_id, attr
):
    providers = {name: provider(error=ConnectionError()) for name in LOCAL_IDS}
    manager = manager_for(providers=providers)
    try:
        manager.cache_model_infos(local_id, {ProviderModelInfo("old-server-model")})
        settings = manager.current_settings().model_copy(
            update={attr: "http://localhost:9999"}
        )
        await manager.replace(settings, commit=AsyncMock())
        await finish_discovery(manager)
        assert local_id not in manager.cached_model_ids()
        assert local_id + "/old-server-model" not in {
            model.provider_model_ref for model in read_model_catalog(manager).models
        }
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_optional_timeout_releases_real_admission_and_keeps_recovery_logs_quiet(
    monkeypatch, caplog
):
    monkeypatch.setattr(provider_manager, "LOCAL_DISCOVERY_TIMEOUT_SECONDS", 0.05)
    controller = ProviderAdmissionController(
        provider_name="ollama",
        rate_limit=1000,
        rate_window=1,
        max_concurrency=1,
        base_delay=0.1,
        max_delay=0.1,
        jitter=0,
    )

    async def offline():
        raise httpx.ConnectError("private-address")

    async def list_models():
        return await controller.start_execution().run_call(
            offline, operation_kind=ProviderOperationKind.MODEL_DISCOVERY
        )

    providers = {
        name: provider(error=list_models if name == "ollama" else ConnectionError())
        for name in LOCAL_IDS
    }
    manager = manager_for(providers=providers)
    try:
        result = await manager.refresh_model_list_cache()
        assert not result.failed_provider_ids
        assert not [record for record in caplog.records if record.levelno >= 30]
        async with await manager.acquire() as lease:
            assert await lease.resolve_provider("ollama") is providers["ollama"]
            assert (
                await asyncio.wait_for(
                    controller.start_execution().run_call(
                        AsyncMock(return_value="healthy"),
                        operation_kind=ProviderOperationKind.GENERATION,
                    ),
                    1,
                )
                == "healthy"
            )
    finally:
        await manager.close()


@pytest.mark.parametrize(
    "failure",
    [
        socket.gaierror("private-host"),
        httpx.HTTPStatusError(
            "private-response",
            request=httpx.Request("GET", "http://localhost"),
            response=httpx.Response(503),
        ),
    ],
)
@pytest.mark.asyncio
async def test_local_deadline_preserves_actual_failure_during_admission_backoff(
    monkeypatch, failure
):
    monkeypatch.setattr(provider_manager, "LOCAL_DISCOVERY_TIMEOUT_SECONDS", 0.05)
    controller = ProviderAdmissionController(
        provider_name="ollama",
        rate_limit=1000,
        rate_window=1,
        max_concurrency=1,
        base_delay=0.1,
        max_delay=0.1,
        jitter=0,
    )

    async def broken():
        if isinstance(failure, socket.gaierror):
            raise httpx.ConnectError("private-host") from failure
        raise failure

    async def list_models():
        return await controller.start_execution().run_call(
            broken, operation_kind=ProviderOperationKind.MODEL_DISCOVERY
        )

    manager = manager_for(providers={"ollama": provider(error=list_models)})
    try:
        result = await manager.refresh_model_list_cache()
        assert result.failed_provider_ids == ("ollama",)
        expected = (
            "DNS lookup failed" if isinstance(failure, socket.gaierror) else "HTTP 503"
        )
        assert expected in result.failure_messages["ollama"]
    finally:
        await manager.close()

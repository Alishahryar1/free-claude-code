import asyncio
import json
import threading
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.harnesses import antigravity_integration as native
from tests.runtime.test_integration_startup import runtime as runtime


@pytest.mark.asyncio
async def test_queued_connect_uses_replaced_catalog_generation(runtime):
    update = runtime._integrations._antigravity_update
    task = None
    try:
        async with update.access.read():
            task = asyncio.create_task(runtime.connect_antigravity())
            async with asyncio.timeout(5):
                while not update.access._waiting_writers:
                    await asyncio.sleep(0)
            await runtime.provider_manager.replace(
                runtime.settings.model_copy(update={"model": "nvidia_nim/replacement"}),
                commit=AsyncMock(),
            )
        assert (await asyncio.wait_for(task, 5))["connected"] is True
        models = json.loads(native.config_path().read_text())["customModelsConfig"][
            "customModels"
        ]
        assert "nvidia_nim/replacement" in models
    finally:
        if task:
            await asyncio.gather(task, return_exceptions=True)
        await runtime.close()


@pytest.mark.asyncio
async def test_cancelled_connect_finishes_before_disconnect(runtime, monkeypatch):
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    write = native._write_native

    def held(*args):
        if not entered.is_set():
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return write(*args)

    monkeypatch.setattr(native, "_write_native", held)
    first = asyncio.create_task(runtime.connect_antigravity())
    reader = second = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        first.cancel()
        reader = asyncio.create_task(runtime.antigravity_status())
        second = asyncio.create_task(runtime.disconnect_antigravity())
        assert (await asyncio.wait_for(runtime.jetbrains_acp_status(), 1))[
            "connected"
        ] is False
        assert not first.done() and not second.done() and not reader.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert (await asyncio.wait_for(second, 5))["connected"] is False
        await asyncio.wait_for(reader, 5)
        assert not native.state_path().exists()
    finally:
        release.set()
        await asyncio.gather(
            first, *[task for task in (reader, second) if task], return_exceptions=True
        )
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["antigravity_status", "connect_antigravity", "disconnect_antigravity"]
)
@pytest.mark.parametrize("condition", ["_is_draining", "_has_pending_restart"])
async def test_operations_reject_unavailable_runtime(
    runtime, monkeypatch, operation, condition
):
    monkeypatch.setattr(runtime._integrations, condition, lambda: True)
    try:
        with pytest.raises(ApplicationUnavailableError):
            await getattr(runtime, operation)()
        assert not native.state_path().exists()
    finally:
        await runtime.close()

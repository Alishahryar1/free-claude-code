import asyncio
import json

import pytest

from free_claude_code.harnesses import devin_acp_integration as devin
from tests.harnesses.test_devin_acp_integration import installed as installed
from tests.runtime.test_integration_startup import runtime as runtime


@pytest.mark.asyncio
async def test_startup_refreshes_owned_registration_only(runtime, installed):
    path = devin.config_path()
    devin.configure(path, True)
    document = json.loads(path.read_text())
    document["agents"][0]["distribution"]["binary"][devin.platform_key()]["cmd"] = (
        "old-fcc"
    )
    path.write_text(json.dumps(document))
    try:
        await runtime.start()
        await asyncio.wait_for(runtime._devin_update.task, 5)
        state = await runtime.devin_acp_status()
        assert state["connected"]
        assert state["update"]["changed"]
        before = path.read_bytes()
        await runtime.refresh_devin_acp()
        await asyncio.wait_for(runtime._devin_update.task, 5)
        assert not (await runtime.devin_acp_status())["update"]["changed"]
        assert path.read_bytes() == before
        await runtime.disconnect_devin_acp()
        await runtime.refresh_devin_acp()
        await asyncio.wait_for(runtime._devin_update.task, 5)
        assert not (await runtime.devin_acp_status())["connected"]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_unconnected_startup_does_not_discover_or_write(runtime, monkeypatch):
    def forbidden():
        pytest.fail("Unconnected startup tried to discover an agent")

    monkeypatch.setattr(devin, "_launch", forbidden)
    try:
        await runtime.start()
        await asyncio.wait_for(runtime._devin_update.task, 5)
        assert not devin.config_path().exists()
    finally:
        await runtime.close()

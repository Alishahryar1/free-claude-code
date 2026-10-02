"""Credential changes create fresh Anthropic state while retained leases drain."""

import httpx
import pytest

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.anthropic import AnthropicProvider
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.providers.runtime.config import build_provider_config
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.providers.support import immediate_admission
from tests.providers.test_anthropic_discovery import metadata


@pytest.mark.asyncio
async def test_workspace_replacement_retires_http_and_capabilities_with_lease():
    owners = []
    requests = []

    def factory(settings, registry):
        def handle(request):
            requests.append(request)
            if settings.anthropic_workspace_id == "new":
                return httpx.Response(403, json={"error": {"type": "permission_error"}})
            if request.url.path == "/v1/models":
                return httpx.Response(
                    200,
                    json={"data": [metadata("same-model", True)], "has_more": False},
                )
            return httpx.Response(200, json=metadata("same-model", True))

        owner = AnthropicProvider(
            build_provider_config(PROVIDER_CATALOG["anthropic"], settings),
            workspace_id=settings.anthropic_workspace_id,
            admission=immediate_admission(max_attempts=1),
            transport=httpx.MockTransport(handle),
        )
        owners.append(owner)
        return ProviderRuntime(settings, registry, {"anthropic": owner})

    async def commit():
        return None

    settings = Settings(
        MODEL="anthropic/same-model",
        NVIDIA_NIM_API_KEY=None,
        ANTHROPIC_API_KEY="old-key",
        ANTHROPIC_WORKSPACE_ID="old",
    )
    manager = ProviderRuntimeManager(settings, runtime_factory=factory)
    old_lease = await manager.acquire()
    new_lease = None
    try:
        old = await old_lease.resolve_provider("anthropic")
        assert isinstance(old, AnthropicProvider)
        record = await old.model_record("same-model")
        await manager.replace(
            settings.model_copy(
                update={"anthropic_api_key": "new-key", "anthropic_workspace_id": "new"}
            ),
            commit=commit,
        )
        new_lease = await manager.acquire()
        new = await new_lease.resolve_provider("anthropic")
        assert isinstance(new, AnthropicProvider)
        assert new is not old
        # Generic catalog rows may survive failed discovery; native facts do not.
        assert new_lease.model_info("anthropic", "same-model") is not None
        assert (await new.model_record("same-model")).messages.adaptive_thinking is None
        assert await old.model_record("same-model") is record
        assert (await old.endpoint()).headers["anthropic-workspace-id"] == "old"
        assert (await new.endpoint()).headers["Authorization"] == "Bearer new-key"
        assert not old._http.is_closed
        await old_lease.release()
        assert old._http.is_closed
        assert not new._http.is_closed
    finally:
        await old_lease.release()
        if new_lease is not None:
            await new_lease.release()
        await manager.close()
    assert all(owner._http.is_closed for owner in owners)
    assert {request.headers["anthropic-workspace-id"] for request in requests} == {
        "old",
        "new",
    }

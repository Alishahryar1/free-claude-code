"""Explicit test composition for the API adapter."""

from collections.abc import Mapping, MutableMapping

from fastapi import FastAPI

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.code_sessions import CodeApplicationPort
from free_claude_code.application.connected_accounts import ConnectedAccountPort
from free_claude_code.config.loader import ManagedConfigStore
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime, RestartCallback
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.integrations.discovery import LocalInstallations
from free_claude_code.runtime.integrations.service import IntegrationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


def create_test_app(
    settings: Settings | None = None,
    *,
    providers: MutableMapping[str, BaseProvider] | None = None,
    restart_callback: RestartCallback | None = None,
    connected_accounts: Mapping[str, ConnectedAccountPort] | None = None,
    code: CodeApplicationPort | None = None,
) -> FastAPI:
    """Build an API app with explicit in-memory runtime services."""
    store = ManagedConfigStore()
    store.initialize()  # API-only tests do not run the production ASGI lifespan.
    settings = settings or store.read().settings
    connected_accounts = dict(connected_accounts or {})

    def connected_provider_ids() -> tuple[str, ...]:
        return tuple(
            provider_id
            for provider_id, account in connected_accounts.items()
            if account.is_connected()
        )

    if providers is None:
        manager = ProviderRuntimeManager(
            settings,
            connected_provider_ids=connected_provider_ids,
        )
    else:
        manager = ProviderRuntimeManager(
            settings,
            runtime_factory=lambda snapshot: ProviderRuntime(
                snapshot,
                dict(providers),
            ),
            connected_provider_ids=connected_provider_ids,
        )
    runtime = ApplicationRuntime(
        manager,
        integrations=IntegrationService(LocalInstallations.current()),
        configuration=ConfigurationService(store),
        transcriber=None,
        restart_callback=restart_callback,
        connected_accounts=connected_accounts,
    )
    return create_app(
        ApiServices(
            requests=manager,
            admin=runtime,
            tasks=runtime,
            code=code,
        )
    )


def runtime_for_app(app: FastAPI) -> ApplicationRuntime:
    """Return the runtime supplied by :func:`create_test_app`."""
    runtime = app.state.services.admin
    if not isinstance(runtime, ApplicationRuntime):
        raise TypeError("Test app does not use ApplicationRuntime")
    return runtime


def provider_manager_for_app(app: FastAPI) -> ProviderRuntimeManager:
    """Return the provider manager supplied by :func:`create_test_app`."""
    manager = app.state.services.requests
    if not isinstance(manager, ProviderRuntimeManager):
        raise TypeError("Test app does not use ProviderRuntimeManager")
    return manager

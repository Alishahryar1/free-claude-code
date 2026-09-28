"""Own integration workflows while sharing configuration publication ordering."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

from loguru import logger

from free_claude_code.application.errors import (
    ApplicationError,
    ApplicationUnavailableError,
    InvalidRequestError,
)
from free_claude_code.application.model_catalog import ModelCatalog, read_model_catalog
from free_claude_code.application.readiness import InitializationWait
from free_claude_code.config.paths import (
    claude_desktop_disconnect_path,
    codex_model_catalog_path,
)
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.core.async_tasks import run_sync_owned
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses import (
    claude_desktop_integration,
    claude_integration,
    codex_integration,
    jetbrains_acp_integration,
    vscode_chat_integration,
)

from .provider_manager import ProviderRuntimeManager

IntegrationAction = Literal["status", "connect", "disconnect", "refresh"]


@dataclass
class _IntegrationUpdate:
    state: Literal["starting", "ready", "failed"] = "ready"
    changed: bool = False
    message: str | None = None
    task: asyncio.Task[None] | None = None

    def snapshot(self) -> JsonObject:
        return {"state": self.state, "changed": self.changed, "message": self.message}

    def complete(self, changed: bool = False) -> None:
        self.state = "ready"
        self.changed = changed
        self.message = None


class IntegrationService:
    """Own integration state and background refreshes, borrowing runtime resources."""

    def __init__(
        self,
        provider_manager: ProviderRuntimeManager,
        *,
        config_lock: asyncio.Lock,
        is_draining: Callable[[], bool],
        has_pending_restart: Callable[[], bool],
    ) -> None:
        self.provider_manager = provider_manager
        self._config_lock = config_lock
        self._is_draining = is_draining
        self._has_pending_restart = has_pending_restart
        self._claude_update = _IntegrationUpdate()
        self._desktop_update = _IntegrationUpdate()
        self._codex_update = _IntegrationUpdate()
        self._jetbrains_update = _IntegrationUpdate()
        self._vscode_update = _IntegrationUpdate()
        self._vscode_dirty = False

    @property
    def settings(self) -> Settings:
        return self.provider_manager.current_settings()

    def _updates(self) -> dict[str, _IntegrationUpdate]:
        return {
            "vscode-chat": self._vscode_update,
            "claude-vscode": self._claude_update,
            "claude-desktop": self._desktop_update,
            "codex": self._codex_update,
            "jetbrains-acp": self._jetbrains_update,
        }

    def snapshot(self) -> JsonObject:
        return {name: update.snapshot() for name, update in self._updates().items()}

    async def start(self) -> None:
        self.catalog_changed()
        await self.refresh_claude_vscode()
        await self.refresh_codex_integration()
        await self.refresh_claude_desktop()
        await self.refresh_jetbrains_acp()

    def cancel_background_tasks(self) -> None:
        for update in self._updates().values():
            if update.task is not None and not update.task.done():
                update.task.cancel()

    async def wait_background_tasks(self) -> None:
        tasks = tuple(
            update.task
            for update in self._updates().values()
            if update.task is not None
        )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning(
                    "Background initialization ended with exc_type={}",
                    type(result).__name__,
                )

    async def vscode_chat_status(self) -> JsonObject:
        return await self._vscode_chat("status")

    async def connect_vscode_chat(self) -> JsonObject:
        return await self._vscode_chat("connect")

    async def disconnect_vscode_chat(self) -> JsonObject:
        return await self._vscode_chat("disconnect")

    async def refresh_vscode_chat(self) -> JsonObject:
        self._check_integration_available()
        self.catalog_changed()
        return {"update": self._vscode_update.snapshot()}

    def catalog_changed(self) -> None:
        if self._is_draining():
            return
        self._vscode_dirty = True
        update = self._vscode_update
        if update.task is not None and not update.task.done():
            return
        update.state, update.changed, update.message = "starting", False, None

        async def drain() -> None:
            while self._vscode_dirty:
                self._vscode_dirty = False
                try:
                    result = await self._vscode_chat("refresh")
                    update.complete(update.changed or result.get("changed") is True)
                except ApplicationError as exc:
                    update.state, update.message = "failed", exc.message
                except Exception as exc:
                    update.state = "failed"
                    update.message = "Could not update VS Code models. Retry shortly."
                    logger.warning(
                        "VS Code integration update failed: exc_type={}",
                        type(exc).__name__,
                    )
                if self._vscode_dirty:
                    update.state = "starting"

        update.task = asyncio.create_task(drain(), name="fcc-vscode-models")

    async def _vscode_chat(self, action: IntegrationAction) -> JsonObject:
        try:
            if action == "refresh":
                async with self._config_lock:
                    self._check_integration_available()
                    status = await run_sync_owned(
                        lambda: vscode_chat_integration.status(
                            vscode_chat_integration.config_path()
                        )
                    )
                    if not status["connected"]:
                        return {"changed": False}
            while True:
                snapshot = (
                    await self.provider_manager.wait_for_catalog()
                    if action in {"connect", "refresh"}
                    else None
                )
                revision = self.provider_manager.catalog_status()["catalog_revision"]
                async with self._config_lock:
                    self._check_integration_available()
                    if snapshot is not None and (
                        snapshot.current_settings() is not self.settings
                        or revision
                        != self.provider_manager.catalog_status()["catalog_revision"]
                        or self.provider_manager.catalog_status()["catalog"] != "ready"
                    ):
                        continue
                    catalog = (
                        read_model_catalog(snapshot) if snapshot is not None else None
                    )
                    settings = self.settings

                    def operate(
                        catalog: ModelCatalog | None = catalog,
                        settings: Settings = settings,
                    ) -> JsonObject:
                        path = vscode_chat_integration.config_path()
                        changed = False
                        if action == "disconnect":
                            vscode_chat_integration.disconnect(path)
                        elif catalog is not None:
                            changed = vscode_chat_integration.configure(
                                path,
                                local_proxy_root_url(settings),
                                settings.proxy_auth_token,
                                catalog.models,
                                only_existing=action == "refresh",
                            )
                        if action == "refresh":
                            return {"changed": changed}
                        return vscode_chat_integration.status(path)

                    result = await run_sync_owned(operate)
                    if action in {"connect", "disconnect"}:
                        self._vscode_update.complete()
                    if action == "status":
                        result["update"] = self._vscode_update.snapshot()
                    return result
        except ValueError, UnicodeError:
            raise InvalidRequestError(
                "Could not configure VS Code Chat. Check chatLanguageModels.json for invalid JSON or conflicting FCC groups."
            ) from None
        except OSError:
            raise ApplicationUnavailableError(
                "Could not access chatLanguageModels.json. Finish configuration edits, check file permissions, and retry."
            ) from None

    async def claude_vscode_status(self) -> JsonObject:
        return await self._claude_vscode("status")

    async def connect_claude_vscode(self) -> JsonObject:
        return await self._claude_vscode("connect")

    async def disconnect_claude_vscode(self) -> JsonObject:
        return await self._claude_vscode("disconnect")

    def _check_integration_available(self) -> None:
        if self._is_draining() or self._has_pending_restart():
            raise ApplicationUnavailableError(
                "Wait for FCC to restart before changing the integration."
            )

    def _start_integration_update(
        self,
        update: _IntegrationUpdate,
        operation: Callable[[], Awaitable[JsonObject]],
    ) -> JsonObject:
        self._check_integration_available()
        if update.task is None or update.task.done():
            update.state, update.changed, update.message = "starting", False, None

            async def run() -> None:
                try:
                    result = await operation()
                    update.complete(result.get("changed") is True)
                except ApplicationError as exc:
                    update.state, update.message = "failed", exc.message
                except Exception as exc:
                    update.state = "failed"
                    update.message = (
                        "Could not update integration settings. Retry shortly."
                    )
                    logger.warning(
                        "Integration update failed: exc_type={}", type(exc).__name__
                    )

            update.task = asyncio.create_task(run(), name="fcc-integration-update")
        return {"update": update.snapshot()}

    async def refresh_claude_vscode(self) -> JsonObject:
        return self._start_integration_update(
            self._claude_update, partial(self._claude_vscode, "refresh")
        )

    async def refresh_codex_integration(self) -> JsonObject:
        return self._start_integration_update(
            self._codex_update, partial(self._codex_integration, "refresh")
        )

    async def jetbrains_acp_status(self) -> JsonObject:
        return await self._jetbrains_acp("status")

    async def connect_jetbrains_acp(self) -> JsonObject:
        return await self._jetbrains_acp("connect")

    async def disconnect_jetbrains_acp(self) -> JsonObject:
        return await self._jetbrains_acp("disconnect")

    async def refresh_jetbrains_acp(self) -> JsonObject:
        return self._start_integration_update(
            self._jetbrains_update, partial(self._jetbrains_acp, "refresh")
        )

    async def _jetbrains_acp(self, action: IntegrationAction) -> JsonObject:
        async with self._config_lock:
            self._check_integration_available()
            settings = self.settings

            def operate() -> JsonObject:
                path = jetbrains_acp_integration.config_path()
                url = local_proxy_root_url(settings)
                if action == "refresh":
                    return {
                        "changed": jetbrains_acp_integration.refresh_connected(
                            path, url, settings.proxy_auth_token
                        )
                    }
                return jetbrains_acp_integration.configure(
                    path,
                    url,
                    settings.proxy_auth_token,
                    None if action == "status" else action == "connect",
                )

            try:
                result = await run_sync_owned(operate)
            except jetbrains_acp_integration.SetupError as exc:
                raise InvalidRequestError(str(exc)) from None
            except ValueError, UnicodeError:
                raise InvalidRequestError(
                    "Could not read JetBrains ACP configuration. Check the JSON in acp.json and the installed Claude Agent metadata."
                ) from None
            except OSError:
                raise ApplicationUnavailableError(
                    "Could not access JetBrains ACP files. Finish any configuration edits, check file permissions, and retry."
                ) from None
            if action in {"connect", "disconnect"}:
                self._jetbrains_update.complete()
            if action == "status":
                result["update"] = self._jetbrains_update.snapshot()
            return result

    async def claude_desktop_status(self) -> JsonObject:
        return await self._claude_desktop("status")

    async def connect_claude_desktop(self) -> JsonObject:
        return await self._claude_desktop("connect")

    async def disconnect_claude_desktop(self) -> JsonObject:
        return await self._claude_desktop("disconnect")

    async def refresh_claude_desktop(self) -> JsonObject:
        return self._start_integration_update(
            self._desktop_update, partial(self._claude_desktop, "refresh")
        )

    async def _claude_desktop(self, action: IntegrationAction) -> JsonObject:
        async with self._config_lock:
            self._check_integration_available()
            settings = self.settings

            def operate() -> JsonObject:
                root = claude_desktop_integration.config_root()
                url = local_proxy_root_url(settings)
                if action == "refresh":
                    return {
                        "changed": claude_desktop_integration.refresh_connected(
                            root,
                            url,
                            settings.proxy_auth_token,
                            disconnect_path=claude_desktop_disconnect_path(),
                        )
                    }
                return claude_desktop_integration.configure(
                    root,
                    url,
                    settings.proxy_auth_token,
                    None if action == "status" else action == "connect",
                    disconnect_path=claude_desktop_disconnect_path(),
                )

            try:
                result = await run_sync_owned(operate)
            except claude_desktop_integration.ManagedDesktopError:
                raise InvalidRequestError(
                    "Claude Desktop is managed by an organization, or its policy could not be read. FCC can configure only unmanaged Desktop installations."
                ) from None
            except claude_desktop_integration.PendingDisconnectError:
                raise InvalidRequestError(
                    "Finish disconnecting Claude Desktop before connecting again."
                ) from None
            except claude_desktop_integration.PendingMigrationError:
                raise InvalidRequestError(
                    "Claude Desktop has data in its previous Windows location. Launch Claude Desktop once so it can migrate that data, fully quit it, then retry Connect."
                ) from None
            except ValueError, UnicodeError:
                raise InvalidRequestError(
                    "Could not configure Claude Desktop. Check its configuration JSON and FCC disconnect record, and ensure FCC uses a localhost address and a nonempty managed token."
                ) from None
            except OSError:
                raise ApplicationUnavailableError(
                    "Could not access Claude Desktop settings or the FCC disconnect record. Fully quit Claude Desktop, check file permissions, and retry."
                ) from None
            if action in {"connect", "disconnect"}:
                self._desktop_update.complete()
            if action == "status":
                if self._desktop_update.state != "ready":
                    result["connected"] = None
                result["update"] = self._desktop_update.snapshot()
            return result

    async def _claude_vscode(self, action: IntegrationAction) -> JsonObject:
        async with self._config_lock:
            self._check_integration_available()
            settings = self.settings

            def operate() -> JsonObject:
                path = claude_integration.settings_path()
                state_path = claude_integration.claude_state_path()
                if action == "status" and self._claude_update.state != "ready":
                    return {
                        "connected": None,
                        "paths": {
                            "vscode_settings": str(path.resolve()),
                            "claude_state": str(state_path.resolve()),
                        },
                    }
                if action == "refresh":
                    return {
                        "changed": claude_integration.refresh_connected(
                            path,
                            state_path,
                            local_proxy_root_url(settings),
                            settings.proxy_auth_token,
                        )
                    }
                return claude_integration.configure(
                    path,
                    state_path,
                    local_proxy_root_url(settings),
                    settings.proxy_auth_token,
                    None if action == "status" else action == "connect",
                )

            try:
                result = await run_sync_owned(operate)
            except ValueError, UnicodeError:
                raise InvalidRequestError(
                    "Could not read Claude integration settings. Check the JSON in VS Code settings.json and .claude.json."
                ) from None
            except OSError:
                raise ApplicationUnavailableError(
                    "Could not access VS Code settings.json or .claude.json. Check file permissions and try again."
                ) from None
            if action in {"connect", "disconnect"}:
                self._claude_update.complete()
            if action == "status":
                result["update"] = self._claude_update.snapshot()
            return result

    async def codex_integration_status(self) -> JsonObject:
        return await self._codex_integration("status")

    async def connect_codex(self) -> JsonObject:
        return await self._codex_integration("connect")

    async def disconnect_codex(self) -> JsonObject:
        return await self._codex_integration("disconnect")

    async def _codex_integration(self, action: IntegrationAction) -> JsonObject:
        wait = InitializationWait(None) if action == "refresh" else InitializationWait()
        needs_catalog = action in {"connect", "refresh"}
        try:
            if action == "refresh":
                async with self._config_lock:
                    self._check_integration_available()
                    url = local_proxy_root_url(self.settings)
                    if not await run_sync_owned(
                        lambda: codex_integration.recognizes_connection(
                            codex_integration.config_path(), url
                        )
                    ):
                        return {"changed": False}
            while True:
                generation_id = (
                    await self.provider_manager.wait_for_catalog_file(wait)
                    if needs_catalog
                    else None
                )
                async with self._config_lock:
                    self._check_integration_available()
                    if needs_catalog and (
                        generation_id != self.provider_manager.current_generation_id
                        or self.provider_manager.catalog_status()["catalog"] != "ready"
                    ):
                        continue
                    url = local_proxy_root_url(self.settings)

                    def operate(url: str = url) -> JsonObject:
                        path = codex_integration.config_path()
                        if action == "status" and self._codex_update.state != "ready":
                            return {
                                "connected": None,
                                "paths": {"codex_config": str(path.resolve())},
                            }
                        if action == "refresh":
                            return {
                                "changed": codex_integration.refresh_connected(
                                    path, codex_model_catalog_path(), url
                                )
                            }
                        return codex_integration.configure(
                            path,
                            codex_model_catalog_path(),
                            url,
                            None if action == "status" else action == "connect",
                        )

                    result = await run_sync_owned(operate)
                    if action in {"connect", "disconnect"}:
                        self._codex_update.complete()
                    if action == "status":
                        result["update"] = self._codex_update.snapshot()
                    return result
        except ValueError, UnicodeError:
            raise InvalidRequestError(
                "Could not read Codex settings. Check the TOML in config.toml."
            ) from None
        except OSError:
            raise ApplicationUnavailableError(
                "Could not access Codex config.toml. Check file permissions and try again."
            ) from None

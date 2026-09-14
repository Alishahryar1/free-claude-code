"""Opt-in FCC connection state over native Copilot account credentials."""

import asyncio
import json
import os
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.paths import github_copilot_auth_path
from free_claude_code.core.async_tasks import run_sync_owned
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.endpoint_types import HttpEndpoint

from .broker import CopilotBroker, CopilotLease
from .lifecycle import drain_owned
from .login import LOGIN_TIMEOUT_SECONDS, DeviceChallenge, device_login
from .types import (
    CopilotAuthenticationRequired,
    CopilotEgress,
    CopilotIdentity,
    CopilotModel,
    CopilotRuntime,
    CopilotUnavailable,
)

LoginFlow = Callable[[Callable[[DeviceChallenge], None]], Awaitable[None]]


async def _create_sdk_runtime() -> CopilotRuntime:
    def load():
        from .sdk import SdkRuntime

        return SdkRuntime

    runtime_type = await run_sync_owned(load)
    return runtime_type()


class AuthenticatedLease:
    """Map account loss at credential resolution to a canonical provider failure."""

    def __init__(
        self,
        manager: CopilotAuthManager,
        broker: CopilotBroker,
        lease: CopilotLease,
        revision: int,
    ) -> None:
        self._manager = manager
        self._broker = broker
        self._lease = lease
        self._revision = revision

    @property
    def model(self) -> CopilotModel:
        return self._lease.model

    @property
    def egress(self) -> CopilotEgress:
        return self._lease.egress

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        try:
            self._manager.ensure_current(self._broker, self._revision)
            endpoint = await self._lease.endpoint(force_refresh=force_refresh)
            self._manager.ensure_current(self._broker, self._revision)
            return endpoint
        except CopilotUnavailable as error:
            raise self._manager.account_failure(
                error, self._broker, self._revision
            ) from None


class CopilotAuthManager:
    """Own enabled state, login attempts, account revisions and draining brokers."""

    provider_id = "github_copilot"

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        runtime_factory: Callable[[], Awaitable[CopilotRuntime]] = _create_sdk_runtime,
        login_flow: LoginFlow = device_login,
        login_timeout_s: float = LOGIN_TIMEOUT_SECONDS,
    ) -> None:
        self._state_path = state_path or github_copilot_auth_path()
        self._runtime_factory = runtime_factory
        self._login_flow = login_flow
        self._login_timeout_s = login_timeout_s
        self._enabled = False
        self._identity: CopilotIdentity | None = None
        self._revision = 0
        self._last_error: str | None = None
        self._broker: CopilotBroker | None = None
        self._owners: list[CopilotBroker] = []
        self._operation_lock = asyncio.Lock()
        self._broker_lock = asyncio.Lock()
        self._login_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._attempt_id: str | None = None
        self._challenge: DeviceChallenge | None = None
        self._expires_at: int | None = None
        self._closed = False
        try:
            self._read_state()
        except OSError, ValueError, UnicodeError:
            self._enabled = False
            self._identity = None
            self._last_error = (
                "Saved Copilot connection state is unreadable. Connect again."
            )

    def is_connected(self) -> bool:
        return self._enabled and not self._closed

    def connected_provider_ids(self) -> tuple[str, ...]:
        return (self.provider_id,) if self.is_connected() else ()

    def status(self) -> ConnectedAccountStatus:
        connecting = self._login_task is not None and not self._login_task.done()
        state = (
            ConnectedAccountState.CONNECTING
            if connecting
            else ConnectedAccountState.ERROR
            if self._last_error
            else ConnectedAccountState.CONNECTED
            if self.is_connected()
            else ConnectedAccountState.DISCONNECTED
        )
        challenge = self._challenge if connecting else None
        return ConnectedAccountStatus(
            provider_id=self.provider_id,
            state=state,
            connected=self.is_connected(),
            revision=self._revision,
            display_identity=self._identity.display
            if self._identity is not None
            else None,
            supported_login_modes=(ConnectedAccountLoginMode.DEVICE,),
            default_login_mode=ConnectedAccountLoginMode.DEVICE,
            attempt_id=self._attempt_id if connecting else None,
            mode=ConnectedAccountLoginMode.DEVICE if connecting else None,
            verification_url=challenge.verification_url if challenge else None,
            user_code=challenge.user_code if challenge else None,
            expires_at=self._expires_at if connecting else None,
            model_count=len(self._broker.current_models) if self._broker else 0,
            message=self._last_error,
        )

    async def start_login(
        self, mode: ConnectedAccountLoginMode
    ) -> ConnectedAccountStatus:
        if mode is not ConnectedAccountLoginMode.DEVICE:
            raise InvalidRequestError("Copilot supports device-code login.")
        async with self._operation_lock:
            self._ensure_open()
            if self._login_task is not None and not self._login_task.done():
                return self.status()
            # Reconnect stops new requests under the old revision immediately.
            if not self._disable():
                return self.status()
            self._last_error = None
            attempt = uuid.uuid4().hex
            self._attempt_id = attempt
            self._challenge = None
            self._expires_at = int(time.time() + self._login_timeout_s)
            self._login_task = asyncio.create_task(self._complete_login(attempt))
            return self.status()

    async def cancel_login(self) -> ConnectedAccountStatus:
        async with self._operation_lock:
            await self._cancel_login()
            try:
                await self._drain_retired()
            except CopilotUnavailable:
                self._last_error = "Copilot cleanup could not finish. Retry Connect to close the previous attempt."
            else:
                self._last_error = None
            return self.status()

    async def disconnect(self) -> ConnectedAccountStatus:
        async with self._operation_lock:
            self._ensure_open()
            saved = self._disable()
            await self._cancel_login()
            async with self._broker_lock:
                pass  # Let any in-flight constructor publish its owned broker.
            await self._drain_retired()
            if saved:
                self._last_error = None
            return self.status()

    async def close(self) -> None:
        if self._close_task is None or self._close_task.done():
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
        await drain_owned(self._close_task)

    async def _close(self) -> None:
        async with self._operation_lock:
            # Preserve opt-in on process shutdown; credentials belong to the CLI.
            await self._cancel_login()
            async with self._broker_lock:
                self._broker = None
            await self._drain_retired()

    def _ensure_open(self) -> None:
        if self._closed:
            raise CopilotUnavailable("Copilot account manager is closing.")

    async def _new_broker(self, identity: CopilotIdentity | None) -> CopilotBroker:
        broker = CopilotBroker(
            await self._runtime_factory(), expected_identity=identity
        )
        self._owners.append(broker)
        return broker

    async def _current_broker(self) -> CopilotBroker:
        async with self._broker_lock:
            self._ensure_open()
            if not self._enabled:
                raise CopilotAuthenticationRequired(
                    "Connect your GitHub Copilot account in Admin."
                )
            if self._broker is None:
                revision = self._revision
                broker = await self._new_broker(self._identity)
                if self._closed or not self._enabled or self._revision != revision:
                    await self._drain_retired()
                    raise CopilotAuthenticationRequired(
                        "Copilot connection changed. Retry after connecting in Admin."
                    )
                self._broker = broker
            return self._broker

    def ensure_current(self, broker: CopilotBroker, revision: int) -> None:
        if (
            not self.is_connected()
            or broker is not self._broker
            or revision != self._revision
        ):
            raise CopilotAuthenticationRequired(
                "Copilot connection changed. Retry after connecting in Admin."
            )

    async def models(self, *, refresh: bool = True) -> Mapping[str, CopilotModel]:
        broker: CopilotBroker | None = None
        revision = self._revision
        try:
            broker = await self._current_broker()
            models = await broker.snapshot(refresh=refresh)
            self.ensure_current(broker, revision)
            return models
        except CopilotUnavailable as error:
            raise self.account_failure(error, broker, revision) from None

    @asynccontextmanager
    async def lease(self, model_id: str) -> AsyncIterator[AuthenticatedLease]:
        broker: CopilotBroker | None = None
        revision = self._revision
        try:
            broker = await self._current_broker()
            async with broker.lease(model_id) as lease:
                self.ensure_current(broker, revision)
                yield AuthenticatedLease(self, broker, lease, revision)
        except CopilotUnavailable as error:
            raise self.account_failure(error, broker, revision) from None
        except ExecutionFailure as error:
            if error.status_code == 401:
                self._invalidate(
                    broker,
                    revision,
                    "Copilot rejected refreshed credentials. Reconnect in Admin.",
                )
            raise

    def account_failure(
        self, error: CopilotUnavailable, broker: CopilotBroker | None, revision: int
    ) -> ExecutionFailure:
        if isinstance(error, CopilotAuthenticationRequired):
            self._invalidate(broker, revision, str(error))
            return ExecutionFailure(FailureKind.AUTHENTICATION, 401, str(error), False)
        return ExecutionFailure(FailureKind.UNAVAILABLE, 503, str(error), False)

    def _invalidate(
        self, broker: CopilotBroker | None, revision: int, message: str
    ) -> None:
        if self._enabled and revision == self._revision and broker is self._broker:
            if not self._disable():
                message += " FCC could not save the disconnected state."
            self._last_error = message

    def _disable(self) -> bool:
        self._enabled = False
        self._identity = None
        self._broker = None
        self._revision += 1
        try:
            self._write_state(enabled=False, identity=None, revision=self._revision)
        except OSError:
            self._last_error = "FCC could not save the disconnected state. Check state-directory permissions and retry Disconnect."
            return False
        finally:
            self._schedule_cleanup()
        return True

    async def _cancel_login(self) -> None:
        task = self._login_task
        self._attempt_id = None
        self._challenge = None
        self._expires_at = None
        if task is not None:
            task.cancel()
            # The login owns its cleanup; cancellation waits for that finally block.
            await drain_owned(asyncio.create_task(_settle(task)))
        self._login_task = None

    def _schedule_cleanup(self) -> asyncio.Task[None]:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_retired())
            self._cleanup_task.add_done_callback(_retrieve_failure)
        return self._cleanup_task

    async def _drain_retired(self) -> None:
        while True:
            await drain_owned(self._schedule_cleanup())
            if not any(owner is not self._broker for owner in self._owners):
                return

    async def _cleanup_retired(self) -> None:
        failures = False
        for broker in tuple(self._owners):
            if broker is self._broker:
                continue
            try:
                await broker.close()
            except Exception:
                failures = True
            else:
                if broker in self._owners:
                    self._owners.remove(broker)
        if failures:
            raise CopilotUnavailable(
                "Copilot cleanup could not finish. Retry disconnect."
            )

    async def _complete_login(self, attempt: str) -> None:
        broker: CopilotBroker | None = None

        def on_challenge(challenge: DeviceChallenge) -> None:
            if self._attempt_id == attempt and not self._closed:
                self._challenge = challenge

        try:
            async with asyncio.timeout(self._login_timeout_s):
                await self._drain_retired()
                broker = await self._new_broker(None)
                try:
                    await broker.snapshot(refresh=True)
                except CopilotAuthenticationRequired:
                    await broker.close()
                    if broker in self._owners:
                        self._owners.remove(broker)
                    broker = None
                    await self._login_flow(on_challenge)
                    broker = await self._new_broker(None)
                    await broker.snapshot(refresh=True)
                identity = broker.identity
                if identity is None:
                    raise CopilotAuthenticationRequired(
                        "Copilot did not report a signed-in profile."
                    )
                if self._attempt_id != attempt or self._closed:
                    return
                revision = self._revision + 1
                self._write_state(enabled=True, identity=identity, revision=revision)
                self._identity = identity
                self._revision = revision
                self._enabled = True
                self._broker = broker
                self._last_error = None
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._attempt_id == attempt and not self._closed:
                self._last_error = (
                    "Copilot sign-in exceeded FCC's time limit. Connect again."
                    if isinstance(error, TimeoutError)
                    else str(error)
                    if isinstance(error, CopilotUnavailable)
                    else "Copilot connection failed. Check the CLI installation, selected account and FCC state-directory permissions."
                )
        finally:
            if broker is not None and broker is not self._broker:
                try:
                    await drain_owned(asyncio.create_task(broker.close()))
                except Exception:
                    if self._attempt_id == attempt:
                        self._last_error = (
                            "Copilot cleanup could not finish. Retry disconnect."
                        )
                else:
                    if broker in self._owners:
                        self._owners.remove(broker)
            if self._login_task is asyncio.current_task():
                self._login_task = None
                self._attempt_id = None
                self._challenge = None
                self._expires_at = None

    def _read_state(self) -> None:
        try:
            payload: object = json.loads(self._state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Invalid Copilot state schema")
        revision = payload.get("revision")
        enabled = payload.get("enabled")
        if (
            not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 0
            or not isinstance(enabled, bool)
        ):
            raise ValueError("Invalid Copilot state")
        identity = payload.get("identity")
        if enabled:
            if not isinstance(identity, dict):
                raise ValueError("Missing Copilot identity")
            host, login = identity.get("host"), identity.get("login")
            if (
                not isinstance(host, str)
                or not host
                or not isinstance(login, str)
                or not login
            ):
                raise ValueError("Invalid Copilot identity")
            self._identity = CopilotIdentity(host, login)
        self._revision = revision
        self._enabled = enabled

    def _write_state(
        self, *, enabled: bool, identity: CopilotIdentity | None, revision: int
    ) -> None:
        payload: JsonObject = {
            "schema_version": 1,
            "enabled": enabled,
            "revision": revision,
            "identity": {"host": identity.host, "login": identity.login}
            if identity
            else None,
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._state_path.parent,
                prefix=".copilot-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._state_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


async def _settle(task: asyncio.Task[None]) -> None:
    await asyncio.gather(task, return_exceptions=True)


def _retrieve_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled():
        task.exception()

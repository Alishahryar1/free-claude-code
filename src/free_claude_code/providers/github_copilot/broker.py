"""Account-scoped discovery and inert endpoint sessions with draining leases."""

import asyncio
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from types import MappingProxyType

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.providers.endpoint_types import HttpEndpoint

from .lifecycle import drain_owned
from .types import (
    CopilotAuthenticationRequired,
    CopilotEgress,
    CopilotEndpoint,
    CopilotIdentity,
    CopilotModel,
    CopilotRuntime,
    CopilotSession,
    CopilotUnavailable,
)

_REFRESH_EARLY_SECONDS = 30


class _EndpointOwner:
    def __init__(self, runtime: CopilotRuntime, model_id: str) -> None:
        self._runtime = runtime
        self._model_id = model_id
        self._lock = asyncio.Lock()
        self._session_task: asyncio.Task[CopilotSession] | None = None
        self._snapshot: CopilotEndpoint | None = None

    async def endpoint(self, *, force_refresh: bool) -> CopilotEndpoint:
        async with self._lock:
            if self._session_task is None or (
                self._session_task.done()
                and (
                    self._session_task.cancelled()
                    or self._session_task.exception() is not None
                )
            ):
                self._session_task = asyncio.create_task(
                    self._runtime.session(self._model_id)
                )
            session = await asyncio.shield(self._session_task)
            snapshot = self._snapshot
            if (
                force_refresh
                or snapshot is None
                or (
                    snapshot.expires_at is not None
                    and snapshot.expires_at <= time.time() + _REFRESH_EARLY_SECONDS
                )
            ):
                snapshot = await session.endpoint()
                if (
                    snapshot.expires_at is not None
                    and snapshot.expires_at <= time.time()
                ):
                    raise CopilotUnavailable(
                        "Copilot returned an expired endpoint credential. Reconnect Copilot."
                    )
                self._snapshot = snapshot
            return snapshot

    async def drain_creation(self) -> None:
        if self._session_task is not None:
            await asyncio.gather(self._session_task, return_exceptions=True)


class CopilotLease:
    """Retain one concrete model and its routing family throughout inference."""

    def __init__(
        self,
        broker: CopilotBroker,
        owner: _EndpointOwner,
        model: CopilotModel,
        egress: CopilotEgress,
    ) -> None:
        self._broker = broker
        self._owner = owner
        self.model = model
        self.egress = egress
        self._released = False

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        if self._released:
            raise CopilotUnavailable("Copilot endpoint lease has already closed.")
        await self._broker.validate_identity()
        self._broker.validate_model(self.model.info.model_id)
        snapshot = await self._owner.endpoint(force_refresh=force_refresh)
        identity = await self._broker.validate_identity()
        self._broker.validate_model(self.model.info.model_id)
        if snapshot.egress is not self.egress:
            raise CopilotUnavailable(
                "Copilot changed this model's upstream protocol. Retry the request to use its new endpoint."
            )
        return replace(snapshot.http, account_id=f"{identity.host}/{identity.login}")

    def release(self) -> None:
        self._released = True


class CopilotBroker:
    """Coalesce SDK discovery and retain sessions until every response closes."""

    def __init__(
        self,
        runtime: CopilotRuntime,
        *,
        expected_identity: CopilotIdentity | None = None,
    ) -> None:
        self._runtime = runtime
        self._identity = expected_identity
        self._models: Mapping[str, CopilotModel] = MappingProxyType({})
        self._owners: dict[str, _EndpointOwner] = {}
        self._condition = asyncio.Condition()
        self._start_task: asyncio.Task[None] | None = None
        self._refresh_task: asyncio.Task[Mapping[str, CopilotModel]] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False
        self._leases = 0

    @property
    def identity(self) -> CopilotIdentity | None:
        return self._identity

    @property
    def current_models(self) -> Mapping[str, CopilotModel]:
        return self._models

    async def _start(self) -> None:
        async with self._condition:
            if self._closing:
                raise CopilotUnavailable("Copilot connection is closing.")
            if self._start_task is None:
                self._start_task = asyncio.create_task(self._runtime.start())
            task = self._start_task
        await asyncio.shield(task)

    async def validate_identity(self) -> CopilotIdentity:
        identity = await self._runtime.identity()
        if identity is None:
            raise CopilotAuthenticationRequired(
                "Copilot is signed out. Connect your GitHub account in Admin."
            )
        if self._identity is not None and identity != self._identity:
            raise CopilotAuthenticationRequired(
                "The selected GitHub profile changed. Reconnect Copilot in Admin."
            )
        self._identity = identity
        return identity

    async def snapshot(self, *, refresh: bool = False) -> Mapping[str, CopilotModel]:
        await self._start()
        await self.validate_identity()
        async with self._condition:
            if self._closing:
                raise CopilotUnavailable("Copilot connection is closing.")
            task = self._refresh_task
            if task is None or task.done():
                if self._models and not refresh:
                    return self._models
                task = asyncio.create_task(self._discover())
                self._refresh_task = task
        return await asyncio.shield(task)

    def validate_model(self, model_id: str) -> CopilotModel:
        model = self._models.get(model_id)
        if model is None:
            raise InvalidRequestError(
                "This model is not available from the connected Copilot account. Refresh the model list."
            )
        return model

    async def _discover(self) -> Mapping[str, CopilotModel]:
        await self.validate_identity()
        models = await self._runtime.models()
        await self.validate_identity()
        result = {model.info.model_id: model for model in models}
        if len(result) != len(models):
            raise CopilotUnavailable("Copilot returned conflicting model identifiers.")
        async with self._condition:
            if self._closing:
                raise CopilotUnavailable("Copilot connection closed during discovery.")
            self._models = MappingProxyType(result)
            if not result:
                raise CopilotUnavailable(
                    "This GitHub account exposes no enabled Copilot models. Check its subscription and model policies."
                )
            return self._models

    @asynccontextmanager
    async def lease(self, model_id: str) -> AsyncIterator[CopilotLease]:
        await self.snapshot()
        async with self._condition:
            if self._closing:
                raise CopilotUnavailable("Copilot connection is closing.")
            model = self.validate_model(model_id)
            self._leases += 1
            owner = self._owners.get(model_id)
            if owner is None:
                owner = _EndpointOwner(self._runtime, model_id)
                self._owners[model_id] = owner
        lease = None
        try:
            await self.validate_identity()
            endpoint = await owner.endpoint(force_refresh=False)
            await self.validate_identity()
            self.validate_model(model_id)
            lease = CopilotLease(self, owner, model, endpoint.egress)
            yield lease
        finally:
            if lease is not None:
                lease.release()
            async with self._condition:
                self._leases -= 1
                self._condition.notify_all()

    async def close(self) -> None:
        if self._closed:
            return
        if self._close_task is None or self._close_task.done():
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await drain_owned(self._close_task)

    async def _close(self) -> None:
        # Stop discovery publication; SDK-owned work itself has bounded RPC timeouts.
        tasks = [
            task for task in (self._start_task, self._refresh_task) if task is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._condition:
            await self._condition.wait_for(lambda: self._leases == 0)
        await asyncio.gather(
            *(owner.drain_creation() for owner in self._owners.values())
        )
        await self._runtime.close()
        self._owners.clear()
        self._models = MappingProxyType({})
        self._closed = True

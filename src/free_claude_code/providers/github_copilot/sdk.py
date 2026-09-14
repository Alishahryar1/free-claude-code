"""Pinned official Copilot SDK boundary; never run SDK agent turns."""

import asyncio
import re
import tempfile
import uuid
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

import httpx
from copilot import (
    CapiSessionOptions,
    CopilotClient,
    CopilotSession,
    PermissionRequest,
    RuntimeConnection,
)
from copilot.rpc import (
    AdaptiveThinkingSupport,
    Model,
    ModelPolicyState,
    ModelsListRequest,
    PermissionDecisionReject,
    ProviderEndpoint,
    ProviderTransport,
    ProviderType,
    ProviderWireAPI,
    SessionProviderGetEndpointRequest,
)
from copilot.session import PermissionInvocation

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
)
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.model_listing import reasoning_capability_from_options

from .lifecycle import drain_owned
from .native_cli import COPILOT_CLI_VERSION, profile_environment, verified_cli_path
from .types import (
    CopilotEgress,
    CopilotEndpoint,
    CopilotIdentity,
    CopilotModel,
    CopilotUnavailable,
)

SDK_RPC_TIMEOUT = 30.0
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")


def _deny_permission(
    _request: PermissionRequest, _invocation: PermissionInvocation
) -> PermissionDecisionReject:
    return PermissionDecisionReject(feedback="FCC owns tool execution.")


async def _delete_session_record(
    client: CopilotClient, session_id: str, *, allow_absent: bool = True
) -> None:
    try:
        async with asyncio.timeout(SDK_RPC_TIMEOUT):
            await client.delete_session(session_id)
    except Exception:
        # Endpoint-only sessions need not have a persisted record. Check this
        # exact owned UUID; neither error prose nor a failed lookup proves absence.
        if not allow_absent:
            raise
        async with asyncio.timeout(SDK_RPC_TIMEOUT):
            metadata = await client.get_session_metadata(session_id)
        if metadata is not None:
            raise


class SdkRuntime:
    """Own a hidden official runtime with native authentication and inert sessions."""

    def __init__(self) -> None:
        self._client: CopilotClient | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._sessions: list[SdkSession] = []
        self._pending_session_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._stopped = False

    async def start(self) -> None:
        async with self._lock:
            if self._closing:
                raise CopilotUnavailable("Copilot runtime has closed.")
            if self._client is not None:
                return
            env = profile_environment()
            path = await verified_cli_path(env)
            env["COPILOT_ALLOW_GET_PROVIDER_ENDPOINT"] = "true"
            directory = tempfile.TemporaryDirectory(prefix="fcc-copilot-")
            self._directory = directory
            try:
                client = CopilotClient(
                    connection=RuntimeConnection.for_stdio(
                        path=path, args=("--disable-builtin-mcps",)
                    ),
                    env=env,
                    working_directory=directory.name,
                    use_logged_in_user=True,
                    mode="copilot-cli",
                    log_level="error",
                )
                self._client = client
                async with asyncio.timeout(SDK_RPC_TIMEOUT):
                    await client.start()
                    status = await client.get_status()
                if status.version != COPILOT_CLI_VERSION:
                    raise CopilotUnavailable(
                        "Copilot runtime version changed. Reconnect with the supported CLI version."
                    )
            except BaseException:
                self._closing = True
                await drain_owned(asyncio.create_task(self._close_client()))
                raise

    def _running(self) -> CopilotClient:
        if self._client is None:
            raise CopilotUnavailable("Copilot runtime is not connected.")
        return self._client

    async def identity(self) -> CopilotIdentity | None:
        try:
            async with asyncio.timeout(SDK_RPC_TIMEOUT):
                status = await self._running().get_auth_status()
        except Exception:
            raise CopilotUnavailable(
                "Could not read Copilot profile status. Reconnect Copilot."
            ) from None
        if not status.isAuthenticated:
            return None
        if status.authType not in {"user", "gh-cli"}:
            raise CopilotUnavailable(
                "Copilot must use an existing GitHub profile. Sign in with copilot login --device-code."
            )
        if not status.login or not status.host:
            raise CopilotUnavailable(
                "Copilot did not report the selected GitHub profile."
            )
        host = urlsplit(
            status.host if "://" in status.host else f"https://{status.host}"
        ).hostname
        if not host:
            raise CopilotUnavailable("Copilot reported an invalid GitHub profile host.")
        return CopilotIdentity(host.lower(), status.login)

    async def models(self) -> tuple[CopilotModel, ...]:
        try:
            result = await self._running().rpc.models.list(
                ModelsListRequest(), timeout=SDK_RPC_TIMEOUT
            )
            return tuple(
                _model(model)
                for model in result.models
                if model.id.strip()
                and model.id.strip().casefold() != "auto"
                and (
                    model.policy is None
                    or model.policy.state is ModelPolicyState.ENABLED
                )
            )
        except Exception:
            raise CopilotUnavailable(
                "Could not discover Copilot subscription models. Check the selected account and its model policies."
            ) from None

    async def session(self, model_id: str) -> SdkSession:
        client = self._running()
        directory = self._directory
        if directory is None:
            raise CopilotUnavailable("Copilot working directory is unavailable.")
        session_id = str(uuid.uuid4())
        config = Path(directory.name) / session_id
        config.mkdir()
        self._pending_session_ids.add(session_id)
        try:
            async with asyncio.timeout(SDK_RPC_TIMEOUT):
                session = await client.create_session(
                    session_id=session_id,
                    model=model_id,
                    working_directory=directory.name,
                    config_directory=str(config),
                    available_tools=[],
                    on_permission_request=_deny_permission,
                    enable_config_discovery=False,
                    skip_custom_instructions=True,
                    custom_agents_local_only=True,
                    enable_file_hooks=False,
                    enable_on_demand_instruction_discovery=False,
                    enable_host_git_operations=False,
                    enable_skills=False,
                    included_builtin_skills=[],
                    request_extensions=False,
                    enable_session_store=False,
                    infinite_sessions={"enabled": False},
                    memory={"enabled": False},
                    skip_embedding_retrieval=True,
                    embedding_cache_storage="in-memory",
                    mcp_oauth_token_storage="in-memory",
                    enable_session_telemetry=False,
                    capi=CapiSessionOptions(enable_web_socket_responses=False),
                )
        except Exception:
            raise CopilotUnavailable(
                "Could not open a Copilot endpoint session. Reconnect Copilot."
            ) from None
        owned = SdkSession(client, session, model_id, session_id)
        self._sessions.append(owned)
        self._pending_session_ids.discard(session_id)
        return owned

    async def close(self) -> None:
        self._closing = True
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close())
        await drain_owned(self._close_task)

    async def _close(self) -> None:
        async with self._lock:
            await self._close_client()

    async def _close_client(self) -> None:
        client = self._client
        if client is None:
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None
            return
        records_only = self._stopped
        try:
            # A prior shutdown can stop the process while record deletion fails.
            # Retain that ownership and reconnect only to retry our own deletions.
            if self._stopped:
                self._stopped = False
                try:
                    async with asyncio.timeout(SDK_RPC_TIMEOUT):
                        await client.start()
                except Exception:
                    raise CopilotUnavailable(
                        "Could not reconnect Copilot to remove an FCC session record. Retry disconnect to finish cleanup."
                    ) from None
            failed_sessions: list[SdkSession] = []
            for session in self._sessions:
                try:
                    await session.close(runtime_stopped=records_only)
                except Exception:
                    failed_sessions.append(session)
            self._sessions = failed_sessions
            # A cancelled create RPC can still finish remotely. Do not accept
            # absence until the process that could create its record has stopped.
            for session_id in tuple(self._pending_session_ids) if records_only else ():
                try:
                    await _delete_session_record(client, session_id)
                    self._pending_session_ids.discard(session_id)
                except Exception:
                    pass
        finally:
            # start() may fail after spawning a process. Stop that partial runtime
            # too, without discarding record IDs still owned by this adapter.
            try:
                async with asyncio.timeout(SDK_RPC_TIMEOUT):
                    await client.stop()
            except Exception:
                try:
                    async with asyncio.timeout(SDK_RPC_TIMEOUT):
                        await client.force_stop()
                except Exception:
                    raise CopilotUnavailable(
                        "Could not stop the Copilot runtime. Retry disconnect to finish cleanup."
                    ) from None
            self._stopped = True
        if self._pending_session_ids and not records_only:
            # Reconnect solely to clean known UUIDs after their writer is gone.
            await self._close_client()
            return
        if self._sessions or self._pending_session_ids:
            raise CopilotUnavailable(
                "Copilot stopped, but an FCC session record could not be removed. Retry disconnect to finish cleanup."
            )
        self._client = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None


class SdkSession:
    def __init__(
        self,
        client: CopilotClient,
        session: CopilotSession,
        model_id: str,
        session_id: str,
    ) -> None:
        self._client = client
        self._session = session
        self._model_id = model_id
        self._session_id = session_id
        self._closed = False
        self._disconnected = False
        self._close_task: asyncio.Task[None] | None = None

    async def endpoint(self) -> CopilotEndpoint:
        if self._closed:
            raise CopilotUnavailable("Copilot endpoint session is closed.")
        try:
            endpoint = await self._session.rpc.provider.get_endpoint(
                SessionProviderGetEndpointRequest(model_id=self._model_id),
                timeout=SDK_RPC_TIMEOUT,
            )
            return _endpoint(endpoint, self._model_id)
        except CopilotUnavailable:
            raise
        except Exception:
            raise CopilotUnavailable(
                "Could not resolve this model's Copilot endpoint. Refresh models or reconnect Copilot."
            ) from None

    async def close(self, *, runtime_stopped: bool = False) -> None:
        if runtime_stopped:
            # Only SdkRuntime, after stopping the owning process, supplies this.
            self._disconnected = True
        if self._closed:
            return
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close())
        await drain_owned(self._close_task)

    async def _close(self) -> None:
        if not self._disconnected:
            try:
                async with asyncio.timeout(SDK_RPC_TIMEOUT):
                    await self._session.disconnect()
                self._disconnected = True
            except Exception:
                # Deletion and runtime stop also release server-side ownership.
                pass
        await _delete_session_record(
            self._client, self._session_id, allow_absent=self._disconnected
        )
        self._closed = True


def _model(model: Model) -> CopilotModel:
    limits = model.capabilities.limits
    supports = model.capabilities.supports
    adaptive: Literal["unsupported", "optional", "required"] | None = None
    if supports is not None and supports.adaptive_thinking is not None:
        modes: dict[
            AdaptiveThinkingSupport, Literal["unsupported", "optional", "required"]
        ] = {
            AdaptiveThinkingSupport.UNSUPPORTED: "unsupported",
            AdaptiveThinkingSupport.OPTIONAL: "optional",
            AdaptiveThinkingSupport.REQUIRED: "required",
        }
        adaptive = modes[supports.adaptive_thinking]
    vision = supports.vision if supports is not None else None
    effort_support = supports.reasoning_effort if supports is not None else None
    efforts = (
        tuple(model.supported_reasoning_efforts)
        if model.supported_reasoning_efforts is not None
        else None
    )
    output = limits.max_output_tokens if limits is not None else None
    context = limits.max_context_window_tokens if limits is not None else None
    info = ProviderModelInfo(
        model_id=model.id,
        reasoning_capability=reasoning_capability_from_options(
            {"supported_efforts": efforts}
        ),
        supports_thinking=True
        if effort_support is True or adaptive in {"optional", "required"}
        else None,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE})
        if vision is True
        else frozenset({ModelInputModality.TEXT})
        if vision is False
        else None,
        context_window_tokens=context if context is not None and context > 0 else None,
        max_output_tokens=output if output is not None and output > 0 else None,
    )
    return CopilotModel(
        info,
        MessagesModelCapabilities(
            max_output_tokens=info.max_output_tokens,
            adaptive_thinking=adaptive,
            supports_output_effort=effort_support,
            supported_efforts=efforts,
            supports_vision=vision,
        ),
        efforts,
        model.default_reasoning_effort,
    )


def _endpoint(endpoint: ProviderEndpoint, model_id: str) -> CopilotEndpoint:
    if endpoint.transport not in {None, ProviderTransport.HTTP}:
        raise CopilotUnavailable(
            "This Copilot model requires an unsupported upstream transport."
        )
    if endpoint.type is ProviderType.ANTHROPIC and endpoint.wire_api is None:
        egress = CopilotEgress.MESSAGES
    elif endpoint.type is ProviderType.OPENAI:
        egress = (
            CopilotEgress.RESPONSES
            if endpoint.wire_api is ProviderWireAPI.RESPONSES
            else CopilotEgress.CHAT
        )
    else:
        raise CopilotUnavailable(
            "This Copilot model requires an unsupported provider endpoint."
        )
    try:
        url = urlsplit(endpoint.base_url)
        valid_url = (
            url.scheme == "https"
            and bool(url.hostname)
            and url.username is None
            and url.password is None
            and not url.query
            and not url.fragment
            and not any(
                character.isspace() or ord(character) < 32
                for character in endpoint.base_url
            )
            and (url.port is None or url.port > 0)
        )
    except ValueError:
        valid_url = False
    if not valid_url:
        raise CopilotUnavailable("Copilot returned an invalid HTTPS endpoint.")
    headers = httpx.Headers(encoding="ascii")
    for key, value in endpoint.headers.items():
        if _HEADER_NAME.fullmatch(key) is None or any(
            character in value for character in "\r\n\0"
        ):
            raise CopilotUnavailable("Copilot returned invalid endpoint headers.")
        try:
            headers[key] = value
        except UnicodeError, ValueError:
            raise CopilotUnavailable(
                "Copilot returned invalid endpoint headers."
            ) from None
    expires = None
    token = endpoint.session_token
    if token is not None:
        if (
            token.model not in {None, model_id}
            or not token.token
            or _HEADER_NAME.fullmatch(token.header) is None
            or any(character in token.token for character in "\r\n\0")
        ):
            raise CopilotUnavailable(
                "Copilot returned an invalid or differently bound session token."
            )
        try:
            headers[token.header] = token.token
        except UnicodeError, ValueError:
            raise CopilotUnavailable(
                "Copilot returned an invalid session token."
            ) from None
        if token.expires_at is not None:
            if token.expires_at.tzinfo is None:
                raise CopilotUnavailable(
                    "Copilot returned a session expiry without a time zone."
                )
            expires = token.expires_at.timestamp()
    if endpoint.api_key is not None:
        if not endpoint.api_key or any(
            character in endpoint.api_key for character in "\r\n\0"
        ):
            raise CopilotUnavailable("Copilot returned an invalid endpoint credential.")
        try:
            endpoint.api_key.encode("ascii")
        except UnicodeError:
            raise CopilotUnavailable(
                "Copilot returned an invalid endpoint credential."
            ) from None
    return CopilotEndpoint(
        egress,
        HttpEndpoint(
            endpoint.base_url.rstrip("/"),
            MappingProxyType(dict(headers.items())),
            endpoint.api_key,
        ),
        expires,
    )

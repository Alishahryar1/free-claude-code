"""Deterministic native-account doubles; never read the real Copilot profile."""

import asyncio

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
)
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.github_copilot.types import (
    CopilotEgress,
    CopilotEndpoint,
    CopilotIdentity,
    CopilotModel,
)


def opened_gate() -> asyncio.Event:
    gate = asyncio.Event()
    gate.set()
    return gate


def model(name: str = "model") -> CopilotModel:
    return CopilotModel(ProviderModelInfo(model_id=name), MessagesModelCapabilities())


class FakeSession:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.value = CopilotEndpoint(
            CopilotEgress.CHAT,
            HttpEndpoint("https://copilot.invalid", {}, "credential"),
        )
        self.endpoint_calls = 0
        self.closed = False
        self.endpoint_gate = opened_gate()

    async def endpoint(self) -> CopilotEndpoint:
        self.endpoint_calls += 1
        await self.endpoint_gate.wait()
        return self.value

    async def close(self) -> None:
        self.closed = True


class FakeRuntime:
    def __init__(self) -> None:
        self.current_identity: CopilotIdentity | None = CopilotIdentity(
            "github.com", "octocat"
        )
        self.available: tuple[CopilotModel, ...] = (model(),)
        self.start_calls = 0
        self.model_calls = 0
        self.session_calls = 0
        self.close_calls = 0
        self.model_error: Exception | None = None
        self.sessions: list[FakeSession] = []
        self.model_gate = opened_gate()
        self.model_entered = asyncio.Event()
        self.session_gate = opened_gate()
        self.session_entered = asyncio.Event()
        self.close_gate = opened_gate()
        self.close_entered = asyncio.Event()

    async def start(self) -> None:
        self.start_calls += 1

    async def identity(self) -> CopilotIdentity | None:
        return self.current_identity

    async def models(self) -> tuple[CopilotModel, ...]:
        self.model_calls += 1
        self.model_entered.set()
        await self.model_gate.wait()
        if self.model_error is not None:
            raise self.model_error
        return self.available

    async def session(self, model_id: str) -> FakeSession:
        self.session_calls += 1
        self.session_entered.set()
        await self.session_gate.wait()
        result = FakeSession(model_id)
        self.sessions.append(result)
        return result

    async def close(self) -> None:
        self.close_calls += 1
        self.close_entered.set()
        await self.close_gate.wait()
        for session in self.sessions:
            await session.close()

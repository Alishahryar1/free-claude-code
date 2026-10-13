"""Official native Copilot device login; FCC never handles the resulting token."""

import asyncio
import os
import re
import subprocess
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import urlsplit

from .lifecycle import drain_owned
from .native_cli import profile_environment, verified_cli_path
from .types import CopilotUnavailable

LOGIN_TIMEOUT_SECONDS = 15 * 60
_CHALLENGE = re.compile(
    r"visit\s+(https://[^\s]+/login/device)\s+and enter code\s+([A-Z0-9]{4}-[A-Z0-9]{4})",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DeviceChallenge:
    verification_url: str
    user_code: str


def parse_challenge(text: str) -> DeviceChallenge | None:
    match = _CHALLENGE.search(text)
    if match is None:
        return None
    url, code = match.groups()
    parsed = urlsplit(url)
    if not parsed.hostname or parsed.username or parsed.password:
        return None
    return DeviceChallenge(url, code.upper())


async def device_login(on_challenge: Callable[[DeviceChallenge], None]) -> None:
    """Run the pinned CLI login and own its child and both bounded readers."""
    env = profile_environment()
    path = await verified_cli_path(env)
    spawning = asyncio.create_task(
        asyncio.create_subprocess_exec(
            path,
            "login",
            "--device-code",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    )
    readers: list[asyncio.Task[None]] = []

    async def read_output(stream: asyncio.StreamReader) -> None:
        buffer = ""
        previous: DeviceChallenge | None = None
        while chunk := await stream.read(1024):
            buffer = (buffer + chunk.decode("utf-8", errors="replace"))[-4096:]
            challenge = parse_challenge(buffer)
            if challenge is not None and challenge != previous:
                on_challenge(challenge)
                previous = challenge

    async def cleanup() -> None:
        try:
            process = await spawning
        except Exception:
            return
        # Cancellation can arrive before normal readers are installed, or cancel
        # their gather. Pipe transports must finish before process.wait can settle.
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)

        async def discard(stream: asyncio.StreamReader) -> None:
            while await stream.read(65536):
                pass

        draining = [
            asyncio.create_task(discard(stream))
            for stream in (process.stdout, process.stderr)
            if stream is not None
        ]
        try:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            await process.wait()
        finally:
            await asyncio.gather(*draining)

    try:
        process = await asyncio.shield(spawning)
        if process.stdout is None or process.stderr is None:
            raise CopilotUnavailable("Copilot login output is unavailable.")
        readers.extend(
            (
                asyncio.create_task(read_output(process.stdout)),
                asyncio.create_task(read_output(process.stderr)),
            )
        )
        code = await process.wait()
        await asyncio.gather(*readers)
        if code != 0:
            raise CopilotUnavailable(
                "Copilot sign-in did not complete. Retry Connect or run copilot login --device-code."
            )
    finally:
        await drain_owned(asyncio.create_task(cleanup()))

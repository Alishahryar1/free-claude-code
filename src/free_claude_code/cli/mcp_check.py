"""Bounded stdio readiness check for the separately installed Headroom tool."""

import asyncio
import json
import os
import signal
import subprocess
from collections.abc import Sequence
from contextlib import suppress

from free_claude_code.cli.process_registry import kill_pid_tree_best_effort
from free_claude_code.harnesses.headroom_mcp import McpSetupError

_TOOLS = {"headroom_compress", "headroom_retrieve", "headroom_stats"}


async def _discard_errors(stream: asyncio.StreamReader) -> None:
    # Keep stderr draining without retaining arbitrary output or credentials.
    while await stream.read(4096):
        pass


async def _stop(process: asyncio.subprocess.Process) -> None:
    if process.stdin is not None:
        process.stdin.close()
    try:
        async with asyncio.timeout(2):
            await process.wait()
            return
    except TimeoutError:
        pass
    if os.name == "nt":
        await asyncio.to_thread(kill_pid_tree_best_effort, process.pid)
    else:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def verify_mcp(command: Sequence[str], *, timeout: float = 30.0) -> None:
    """Require a live MCP handshake and the three stock Headroom tools."""
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name != "nt",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        limit=1024 * 1024,
    )
    assert (
        process.stdin is not None
        and process.stdout is not None
        and process.stderr is not None
    )
    errors = asyncio.create_task(_discard_errors(process.stderr))

    async def send(value: dict[str, object]) -> None:
        assert process.stdin is not None
        process.stdin.write(json.dumps(value).encode() + b"\n")
        await process.stdin.drain()

    async def result(identifier: int) -> dict[str, object]:
        assert process.stdout is not None
        while line := await process.stdout.readline():
            try:
                payload = json.loads(line)
            except ValueError:
                raise McpSetupError(
                    "Headroom MCP server wrote invalid protocol data."
                ) from None
            if not isinstance(payload, dict):
                raise McpSetupError("Headroom MCP server returned an invalid response.")
            if "id" not in payload:
                continue
            if payload.get("id") != identifier:
                raise McpSetupError(
                    "Headroom MCP server returned an unexpected response ID."
                )
            if "error" in payload:
                raise McpSetupError("Headroom MCP server returned a protocol error.")
            value = payload.get("result")
            if not isinstance(value, dict):
                raise McpSetupError("Headroom MCP server returned an invalid result.")
            return value
        raise McpSetupError(
            "Headroom MCP server stopped before verification completed. Check its MCP dependencies."
        )

    try:
        async with asyncio.timeout(timeout):
            await send(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "fcc-installer", "version": "1"},
                    },
                }
            )
            initialized = await result(1)
            if not isinstance(initialized.get("protocolVersion"), str):
                raise McpSetupError(
                    "Headroom MCP server did not complete initialization."
                )
            await send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            await send(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            )
            listed = (await result(2)).get("tools")
            names = (
                {
                    tool["name"]
                    for tool in listed
                    if isinstance(tool, dict) and isinstance(tool.get("name"), str)
                }
                if isinstance(listed, list)
                else set()
            )
            if not _TOOLS.issubset(names):
                raise McpSetupError(
                    "The installed command does not expose the required Headroom tools. Install headroom-ai[mcp]."
                )
    except TimeoutError:
        raise McpSetupError(
            "Headroom MCP verification timed out. Check the installation before retrying."
        ) from None
    except BrokenPipeError, ConnectionResetError:
        raise McpSetupError(
            "Headroom MCP server closed the connection during verification."
        ) from None
    finally:
        cleanup = asyncio.create_task(_stop(process))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
        finally:
            errors.cancel()
            await asyncio.gather(errors, return_exceptions=True)

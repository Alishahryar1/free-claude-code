import asyncio
import sys
from pathlib import Path

import pytest

from free_claude_code.cli.mcp_check import verify_mcp
from free_claude_code.harnesses.headroom_mcp import McpSetupError


def server(
    tmp_path: Path, *, tools: list[object], rpc_error: bool = False
) -> list[str]:
    script = tmp_path / "fake_mcp.py"
    script.write_text(
        "import json, sys\n"
        f"tools = {tools!r}\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if 'id' not in request: continue\n"
        "    result = {'protocolVersion': '2025-06-18', 'capabilities': {}, 'serverInfo': {'name': 'headroom', 'version': '1'}}\n"
        "    if request['method'] == 'tools/list': result = {'tools': [{'name': name} for name in tools]}\n"
        "    response = {'jsonrpc': '2.0', 'id': request['id'], 'result': result}\n"
        f"    if {rpc_error!r}: response = {{'jsonrpc': '2.0', 'id': request['id'], 'error': {{'code': -32000, 'message': 'broken'}}}}\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/message', 'params': {}}), flush=True)\n"
        "    print(json.dumps(response), flush=True)\n",
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def test_verifies_stdio_tools_and_ignores_notifications(tmp_path: Path):
    command = server(
        tmp_path, tools=["headroom_compress", "headroom_retrieve", "headroom_stats"]
    )
    asyncio.run(verify_mcp(command))


def test_missing_mcp_tools_fails(tmp_path: Path):
    with pytest.raises(McpSetupError, match="required Headroom tools"):
        asyncio.run(verify_mcp(server(tmp_path, tools=["other"])))


def test_rpc_error_fails(tmp_path: Path):
    with pytest.raises(McpSetupError, match=r"MCP.*error"):
        asyncio.run(verify_mcp(server(tmp_path, tools=[], rpc_error=True)))


def test_malformed_tool_names_fail_with_setup_error(tmp_path: Path):
    with pytest.raises(McpSetupError, match="required Headroom tools"):
        asyncio.run(verify_mcp(server(tmp_path, tools=[[]])))


def test_nonresponsive_server_is_stopped(tmp_path: Path):
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(McpSetupError, match="timed out"):
        asyncio.run(verify_mcp(command, timeout=0.2))


def test_missing_dependencies_are_reported(tmp_path: Path):
    command = [
        sys.executable,
        "-c",
        "import sys; print('MCP dependencies missing', file=sys.stderr); sys.exit(1)",
    ]
    with pytest.raises(McpSetupError, match="MCP server"):
        asyncio.run(verify_mcp(command))

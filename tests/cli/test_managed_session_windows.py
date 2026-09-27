"""Exercise managed messaging launches without a parent console on Windows."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console allocation")
@pytest.mark.parametrize("stop_early", [False, True])
def test_managed_session_does_not_allocate_console(tmp_path, stop_early):
    # Use a GUI parent to reproduce desktop launch semantics even under a terminal.
    parent = textwrap.dedent(r"""
        import asyncio
        import ctypes
        import json
        import os
        import sys
        from pathlib import Path
        from unittest.mock import patch

        from free_claude_code.cli import process_registry
        from free_claude_code.cli.managed.claude import ManagedClaudeInvocation
        from free_claude_code.cli.managed.session import ManagedClaudeSession

        child = (
            "import ctypes,json,sys,time; "
            "k=ctypes.WinDLL('kernel32'); k.GetConsoleWindow.restype=ctypes.c_void_p; "
            "print(json.dumps({'type':'probe','console':k.GetConsoleWindow()}),flush=True); "
            "time.sleep(3 if sys.argv[1]=='True' else 0)"
        )
        async def main():
            kernel = ctypes.WinDLL('kernel32')
            kernel.GetConsoleWindow.restype = ctypes.c_void_p
            assert kernel.GetConsoleWindow() is None
            session = ManagedClaudeSession(os.getcwd(), 'http://127.0.0.1:1', auth_token='probe')
            invocation = ManagedClaudeInvocation(
                argv=(str(Path(sys.executable).with_name('python.exe')), '-c', child, sys.argv[1]),
                env=dict(os.environ), cwd=os.getcwd(),
            )
            events = []
            with patch('free_claude_code.cli.managed.session.build_managed_claude_invocation', return_value=invocation):
                try:
                    async for event in session.start_task('probe'):
                        events.append(event)
                        if event['type'] == 'probe' and sys.argv[1] == 'True':
                            assert await session.stop()
                finally:
                    assert await session.stop()
            assert session.process.returncode is not None
            assert not process_registry._pids
            print(json.dumps(events), flush=True)
        asyncio.run(main())
    """)
    completed = subprocess.run(
        [
            str(Path(sys.executable).with_name("pythonw.exe")),
            "-c",
            parent,
            str(stop_early),
        ],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert completed.returncode == 0, completed.stderr
    events = json.loads(completed.stdout)
    assert (
        next(event for event in events if event["type"] == "probe")["console"] is None
    )
    assert events[-1]["type"] == "exit"
    if not stop_early:
        assert events[-1]["code"] == 0

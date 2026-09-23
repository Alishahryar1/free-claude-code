"""Own the native folder dialog for one client request at a time."""

import asyncio
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from loguru import logger

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.cli.process_registry import register_pid, unregister_pid

_UNAVAILABLE = "Could not open the folder picker. Enter the path manually."


def _unavailable(exc: Exception) -> ApplicationUnavailableError:
    logger.warning("Folder picker failed: {}: {}", type(exc).__name__, exc)
    return ApplicationUnavailableError(_UNAVAILABLE)


class NativeFolderPicker:
    def __init__(self) -> None:
        self._active: asyncio.Task[str | None] | None = None
        self._stop = asyncio.Event()
        self._closed = False

    async def pick_folder(self, initial_path: str | None) -> str | None:
        if self._closed:
            raise ApplicationUnavailableError("FCC is shutting down.")
        if self._active is not None:
            raise ApplicationUnavailableError("A folder picker is already open")
        self._stop = asyncio.Event()
        task = self._active = asyncio.create_task(
            self._select(initial_path, self._stop), name="fcc-folder-picker"
        )
        try:
            return await self._join(task)
        except ApplicationUnavailableError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            self._active = None

    def begin_shutdown(self) -> None:
        self._closed = True
        self._stop.set()

    async def close(self) -> None:
        self.begin_shutdown()
        if self._active is not None:
            # The request reports selection failures after the child is reaped.
            with suppress(ApplicationUnavailableError):
                await self._join(self._active)

    async def _join(self, task: asyncio.Task[str | None]) -> str | None:
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                # Never cancel spawn/cleanup: signal the owned operation instead.
                await asyncio.wait({task})
            except asyncio.CancelledError as exc:
                cancellation = exc
                self._stop.set()
        if cancellation is not None:
            if not task.cancelled():
                task.exception()
            raise cancellation
        return task.result()

    async def _spawn(self, initial_path: str | None) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "free_claude_code.runtime.native_folder_dialog",
            initial_path or "",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32",
        )

    async def _select(
        self, initial_path: str | None, stop: asyncio.Event
    ) -> str | None:
        try:
            process = await self._spawn(initial_path)
        except (OSError, ValueError) as exc:
            raise _unavailable(exc) from exc
        register_pid(process.pid)
        reading = asyncio.create_task(process.communicate())
        stopped = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait((reading, stopped), return_when=asyncio.FIRST_COMPLETED)
            if stop.is_set():
                return None
            stdout, stderr = reading.result()
            if process.returncode != 0:
                raise RuntimeError(stderr.decode("utf-8", errors="replace"))
            result = json.loads(stdout)
            if not isinstance(result, dict) or "path" not in result:
                raise ValueError("Missing folder selection")
            path = result["path"]
            if path is not None and (
                not isinstance(path, str)
                or not path
                or "\0" in path
                or not Path(path).is_absolute()
            ):
                raise ValueError("Invalid folder selection")
            return path
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            try:
                if process.returncode is None or not reading.done():
                    try:
                        if sys.platform == "win32":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                await process.wait()
                await reading
            finally:
                stopped.cancel()
                await asyncio.gather(stopped, return_exceptions=True)
                unregister_pid(process.pid)

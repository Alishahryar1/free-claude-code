"""Platform-neutral lifecycle for the FCC desktop shell."""

import os
import threading
from collections.abc import Callable
from typing import Protocol

from loguru import logger

from free_claude_code.cli.commands import (
    ServerStatus,
    ServerSupervisor,
    open_admin_when_ready,
)
from free_claude_code.config.loader import get_settings
from free_claude_code.config.paths import config_dir_path, desktop_port_lock_path
from free_claude_code.core.interprocess_lock import InterprocessFileLock


class DesktopTray(Protocol):
    """UI loop owned by the platform tray adapter."""

    def run(self, setup: Callable[[], None]) -> None: ...

    def stop(self) -> None: ...


class DesktopTrayFactory(Protocol):
    """Construct a tray adapter around a desktop controller."""

    def __call__(self, controller: DesktopController) -> DesktopTray: ...


class ServerOwner(Protocol):
    """Server lifecycle used by the desktop controller."""

    @property
    def status(self) -> ServerStatus: ...

    def schedule_run(self) -> bool: ...

    def run(self, *, open_admin_browser: bool | None = None) -> None: ...

    def request_restart(self) -> bool: ...

    def request_stop(self) -> None: ...


class DesktopController:
    """Coordinate one tray loop with one in-process FCC server owner."""

    def __init__(
        self,
        supervisor: ServerOwner,
        tray_factory: DesktopTrayFactory,
        open_admin: Callable[[], None],
        run_server: Callable[[], None] | None = None,
    ) -> None:
        self._supervisor = supervisor
        self._open_admin = open_admin
        self._server_run = run_server or supervisor.run
        self._thread_lock = threading.Lock()
        self._server_thread: threading.Thread | None = None
        self._tray = tray_factory(self)

    @property
    def status(self) -> ServerStatus:
        return self._supervisor.status

    def run(self) -> None:
        """Run the tray on this thread and the FCC server on its owned worker."""

        try:
            self._tray.run(self._start_server)
        finally:
            self._supervisor.request_stop()
            self._tray.stop()
            with self._thread_lock:
                thread = self._server_thread
            if thread is not None:
                thread.join()

    def open_admin(self) -> None:
        self._open_admin()

    def restart_server(self) -> None:
        """Restart an active server or relaunch one that exited unexpectedly."""

        with self._thread_lock:
            thread = self._server_thread
        if thread is not None and thread.is_alive():
            self._supervisor.request_restart()
            return
        self._start_server()

    def quit(self) -> None:
        """Close the server gracefully and end the platform tray loop."""

        self._supervisor.request_stop()
        self._tray.stop()

    def _start_server(self) -> None:
        with self._thread_lock:
            if self._server_thread is not None and self._server_thread.is_alive():
                return
            if not self._supervisor.schedule_run():
                return
            self._server_thread = threading.Thread(
                target=self._run_server,
                name="fcc-desktop-server",
            )
            self._server_thread.start()

    def _run_server(self) -> None:
        self._server_run()


def launch_desktop(tray_factory: DesktopTrayFactory) -> None:
    """Show the tray before acquiring configuration, singleton and server ownership."""
    supervisor = ServerSupervisor(console_logging=False)
    auto_started = os.environ.get("FCC_DESKTOP_STARTED_BY_LAUNCHER") == "1"

    def reuse_existing(settings) -> bool:
        if auto_started:
            controller.quit()
            return True
        reused = open_admin_when_ready(settings, stop_event=supervisor.stop_event)
        if reused:
            controller.quit()
        return reused

    def run_server() -> None:
        try:
            settings = get_settings()
        except Exception as exc:
            logger.error("Desktop server could not start: {}", exc)
            return
        port_lock = InterprocessFileLock(desktop_port_lock_path(settings.port))
        try:
            port_acquired = port_lock.acquire()
        except OSError as exc:
            logger.error("Could not acquire the desktop port lock: {}", exc)
            controller.quit()
            return
        if not port_acquired:
            try:
                if not auto_started:
                    open_admin_when_ready(settings, stop_event=supervisor.stop_event)
            finally:
                controller.quit()
            return
        instance_lock = InterprocessFileLock(config_dir_path() / "desktop.lock")
        try:
            acquired = instance_lock.acquire()
        except OSError as exc:
            logger.error("Could not acquire the desktop lock: {}", exc)
            port_lock.release()
            controller.quit()
            return
        if not acquired:
            port_lock.release()
            logger.error("Desktop is already running on a different port")
            controller.quit()
            return
        try:
            supervisor.run(existing_server=reuse_existing)
        except Exception as exc:
            logger.error("Desktop server could not start: {}", exc)
        finally:
            instance_lock.release()
            port_lock.release()

    controller = DesktopController(
        supervisor,
        tray_factory,
        supervisor.request_open_admin,
        run_server,
    )
    controller.run()

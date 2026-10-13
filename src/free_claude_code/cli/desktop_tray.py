"""pystray adapter for the Windows tray and macOS menu bar."""

from collections.abc import Callable
from io import BytesIO

from PIL import Image
from pystray import Icon, Menu, MenuItem

from free_claude_code.cli.desktop import DesktopController, launch_desktop
from free_claude_code.cli.desktop_assets import app_icon_bytes


class PystrayDesktopTray:
    """Render desktop lifecycle actions through the native status area."""

    def __init__(self, controller: DesktopController) -> None:
        self._controller = controller
        self._icon = Icon(
            "free-claude-code",
            _create_icon(),
            "Free Claude Code",
            Menu(
                MenuItem("Open Admin", self._open_admin, default=True),
                MenuItem("Check Server Status", self._check_status),
                MenuItem("Restart Server", self._restart_server),
                Menu.SEPARATOR,
                MenuItem("Quit", self._quit),
            ),
        )

    def run(self, setup: Callable[[], None]) -> None:
        def ready(icon: Icon) -> None:
            icon.visible = True
            setup()

        self._icon.run(setup=ready)

    def stop(self) -> None:
        self._icon.stop()

    def _open_admin(self, _icon: Icon, _item: MenuItem) -> None:
        self._controller.open_admin()

    def _check_status(self, _icon: Icon, _item: MenuItem) -> None:
        self._icon.notify(
            f"Server is {self._controller.status}.",
            "Free Claude Code",
        )

    def _restart_server(self, _icon: Icon, _item: MenuItem) -> None:
        self._controller.restart_server()

    def _quit(self, _icon: Icon, _item: MenuItem) -> None:
        self._controller.quit()


def _create_icon() -> Image.Image:
    """Load the same branded artwork used by native desktop launchers."""

    with Image.open(BytesIO(app_icon_bytes(".png"))) as image:
        return image.convert("RGBA")


def launch(*, auto_started: bool = False) -> None:
    """Launch the supported native tray adapter."""

    launch_desktop(PystrayDesktopTray, auto_started=auto_started)

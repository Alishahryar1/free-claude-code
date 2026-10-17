"""Test-only instrumentation installed in a disposable FCC environment."""

import asyncio
import multiprocessing
import os
import sys
import traceback
from importlib import import_module
from pathlib import Path
from typing import Never
from unittest.mock import patch

_installed = False


def _spawned_child(queue):
    import free_claude_code

    queue.put(
        {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "package": free_claude_code.__file__,
        }
    )


def subprocess_checks():
    from free_claude_code.runtime.folder_picker import NativeFolderPicker

    expected = str(Path.home())
    os.environ["FCC_NATIVE_DIALOG_RESULT"] = expected
    try:
        selected = asyncio.run(NativeFolderPicker().pick_folder(expected))
    finally:
        del os.environ["FCC_NATIVE_DIALOG_RESULT"]
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=_spawned_child, args=(queue,))
    child.start()
    try:
        result = queue.get(timeout=20)
        child.join(20)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.kill()
            child.join()
        queue.close()
        queue.join_thread()
    return {"folder": selected, "child": result}


def install():
    global _installed
    case = os.environ.get("FCC_NATIVE_CASE")
    if not case or _installed:
        return
    _installed = True
    if "FCC_NATIVE_DIALOG_RESULT" in os.environ:
        original_exec = os.execv

        def dialog_exec(path, argv, /) -> Never:
            if path == "/usr/bin/osascript":
                print(os.environ["FCC_NATIVE_DIALOG_RESULT"], flush=True)
                raise SystemExit(0)
            original_exec(path, argv)

        patch.object(os, "execv", dialog_exec).start()
    identity = import_module("fcc_identity_probe")
    if case == "identity":
        from free_claude_code.cli import desktop_entrypoint

        desktop_entrypoint.launch = identity.main
        return
    from free_claude_code.cli.desktop_tray import PystrayDesktopTray
    from free_claude_code.runtime.integration_service import IntegrationService
    from free_claude_code.runtime.provider_manager import ProviderRuntimeManager

    # Keep the real tray/server lifecycle, with external discovery and harness writes disabled.
    ProviderRuntimeManager.start_model_list_refresh = lambda self: None

    async def no_integrations(self):
        return None

    IntegrationService.start = no_integrations
    original_run = PystrayDesktopTray.run
    Foundation = import_module("Foundation")

    def observed_run(self, setup):
        output = Path(os.environ["FCC_MACOS_PROBE_OUTPUT"])
        control = output.with_suffix(".command")
        report = identity.process_report()
        identity.write_report(output, report)

        def observe(_timer):
            try:
                report.update(identity.tray_report(self._icon))
                report["server_status"] = str(self._controller.status)
                if control.exists():
                    command = control.read_text()
                    control.unlink()
                    if command == "subprocess":
                        report["subprocess"] = subprocess_checks()
                    else:
                        menu = self._icon._status_item.menu()
                        index = menu.indexOfItemWithTitle_(command)
                        assert index >= 0, command
                        menu.performActionForItemAtIndex_(index)
                        report.setdefault("actions", []).append(command)
                identity.write_report(output, report)
            except Exception:
                report["error"] = traceback.format_exc()
                identity.write_report(output, report)
                self._controller.quit()

        timer = Foundation.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            0.1, True, observe
        )
        try:
            original_run(self, setup)
        finally:
            timer.invalidate()
            report["loop_finished"] = True
            identity.write_report(output, report)

    PystrayDesktopTray.run = observed_run

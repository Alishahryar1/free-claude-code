"""Native update commands delegate to the installer and survive replacement."""

import contextlib
import http.server
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import override

import pytest

ROOT = Path(__file__).resolve().parents[2]
RAW_URL = "https://raw.githubusercontent.com/Alishahryar1/free-claude-code"
RELEASES_URL = "https://github.com/Alishahryar1/free-claude-code/releases"
SCRIPTS = ("fcc-update", "fcc-update.cmd")
INSTALLED = "1.0.0"
LATEST = {"current": INSTALLED, "newer-installed": "0.9.0"}


@contextlib.contextmanager
def installer_server(body: str, status: int, latest: str):
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            base = f"http://127.0.0.1:{server.server_port}"
            payload = (
                body.replace(RAW_URL, base)
                .replace(RELEASES_URL, base + "/releases")
                .encode("utf-8")
            )
            response_status = status
            if self.path == "/releases/latest":
                self.send_response(302)
                self.send_header("Location", f"/releases/tag/v{latest}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path.startswith("/releases/tag/"):
                payload, response_status = b"release page", 200
            elif self.path.endswith("/pyproject.toml"):
                payload = b'[project]\nrequires-python = "==3.15.2"\n[tool.uv]\nrequired-version = ">=0.14.2"\n'
                response_status = 200
            self.send_response(response_status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        @override
        def log_message(self, format: object, *args: object):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def fixture_wheel(area: Path, url: str):
    wheel = area / f"free_claude_code-{INSTALLED}-py3-none-any.whl"
    dist = f"free_claude_code-{INSTALLED}"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{dist}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: free-claude-code\nVersion: {INSTALLED}\n",
        )
        archive.writestr(
            f"{dist}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        for name in SCRIPTS:
            source = (ROOT / "scripts/update" / name).read_text(encoding="utf-8")
            archive.writestr(
                f"{dist}.data/scripts/{name}", source.replace(RAW_URL, url)
            )
        archive.writestr("free_claude_code/__init__.py", "")
        archive.writestr(
            f"{dist}.dist-info/RECORD",
            "".join(f"{name},,\n" for name in archive.namelist())
            + f"{dist}.dist-info/RECORD,,\n",
        )
    return wheel


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "exit",
        "error",
        "download-error",
        "real-installer",
        "current",
        "newer-installed",
    ],
)
def test_installed_update_delegates_and_survives_replacement(
    tmp_path: Path, outcome: str, powershell_module_paths: dict[str, str]
):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for native script lifecycle coverage")
    area = tmp_path / "José 测试 & (spaces) ! apostrophe'"
    area.mkdir()
    env = {
        key: value for key, value in os.environ.items() if key != "FCC_INSTALL_VERSION"
    }
    env.update(
        UV_TOOL_DIR=str(area / "tools"),
        UV_TOOL_BIN_DIR=str(area / "bin"),
        UV_CACHE_DIR=str(area / "cache"),
        UV_NO_CONFIG="1",
        UV_PYTHON_DOWNLOADS="never",
        FCC_TEST_LAUNCHER=str(
            area / "bin" / ("fcc-update.cmd" if os.name == "nt" else "fcc-update")
        ),
    )
    real_installer = outcome in {"real-installer", "current", "newer-installed"}
    if os.name == "nt":
        powershell = shutil.which("powershell")
        if powershell is None:
            pytest.skip("Windows PowerShell is required for the native update wrapper")
        env["PSMODULEPATH"] = powershell_module_paths[powershell]
        body = """param([switch] $Update, [switch] $VoiceLocal, [string] $TorchBackend)
$answer = Read-Host 'Continue'
Write-Output "installer:$Update|$VoiceLocal|$TorchBackend|$answer|$env:FCC_INSTALL_VERSION"
[IO.File]::WriteAllText($env:FCC_TEST_LAUNCHER, 'exit /b 99')
"""
        arguments = "-VoiceLocal -TorchBackend cu130"
        expected = "installer:True|True|cu130|yes|"
        body += {"exit": "exit 23", "error": "throw 'installer failed'"}.get(
            outcome, ""
        )
    else:
        body = """read -r answer
printf 'installer:%s|%s|%s|%s|%s|%s\\n' "$1" "$2" "$3" "$4" "$answer" "${FCC_INSTALL_VERSION-}"
printf 'exit 99\\n' > "$FCC_TEST_LAUNCHER"
"""
        arguments = ["--voice-local", "--torch-backend", "cu130"]
        expected = "installer:--update|--voice-local|--torch-backend|cu130|yes|"
        body += {"exit": "exit 23", "error": "exit 1"}.get(outcome, "")
    if real_installer:
        suffix = "ps1" if os.name == "nt" else "sh"
        source = (ROOT / f"scripts/install.{suffix}").read_text(encoding="utf-8")
        if os.name == "nt":
            # The real installer binds the options the stub reports.
            body = body.split("\n", 1)[1]
            body += 'Write-Output "marker:[$env:_FCC_INSTALLER_RELEASE]"\nreturn\n'
            boundary = "\nInitialize-InstallLog\n"
            expected = "installer:True|True|cu130|yes|2.0.0"
        else:
            body += 'printf "marker:[%s]\\n" "${_FCC_INSTALLER_RELEASE-}"\nexit 0\n'
            boundary = "\ninitialize_install_log\n"
            expected = (
                "installer:--update|--voice-local|--torch-backend|cu130|yes|2.0.0"
            )
        assert source.count(boundary) == 1
        body = source.replace(boundary, "\n" + body + boundary)
    latest = LATEST.get(outcome, "2.0.0")
    with installer_server(
        body, 503 if outcome == "download-error" else 200, latest
    ) as (url, requests):
        wheel = fixture_wheel(area, url)
        result = subprocess.run(
            [
                uv,
                "tool",
                "install",
                "--python",
                getattr(sys, "_base_executable", sys.executable),
                str(wheel),
            ],
            env=env,
            capture_output=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stderr
        launcher = env["FCC_TEST_LAUNCHER"]
        command = (
            f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /s /c ""{launcher}" {arguments}"'
            if os.name == "nt"
            else [launcher, *arguments]
        )
        result = subprocess.run(
            command,
            env=env,
            cwd=area,
            input="yes\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        suffix = "ps1" if os.name == "nt" else "sh"
        main_installer = [f"/main/scripts/install.{suffix}"]
        if not real_installer:
            assert requests == main_installer, output
            if outcome == "download-error":
                assert result.returncode != 0
                assert "installer:" not in result.stdout
            else:
                assert expected in result.stdout, output
                assert result.returncode == {"exit": 23, "error": 1}.get(outcome, 0), (
                    output
                )
        else:
            selected = [
                "/releases/latest",
                f"/releases/tag/v{latest}",
                f"/v{latest}/scripts/install.{suffix}",
            ]
            assert result.returncode == 0, output
            if outcome == "real-installer":
                assert requests == [
                    *main_installer,
                    *selected,
                    "/v2.0.0/pyproject.toml",
                ], output
                assert f"FCC update available: {INSTALLED} -> 2.0.0." in output
                assert "marker:[]" in result.stdout, output
                assert result.stdout.count(expected) == 1, output
                assert "Installer log:" not in result.stdout
            else:
                assert requests == [*main_installer, *selected], output
                message = (
                    f"FCC {INSTALLED} is already up to date."
                    if outcome == "current"
                    else f"FCC {INSTALLED} is newer than the published version {latest}. No update needed."
                )
                assert message in result.stdout, output
                assert "installer:" not in result.stdout
                assert "marker:" not in result.stdout
        result = subprocess.run(
            [uv, "tool", "uninstall", "free-claude-code"],
            env=env,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert not any((area / "bin" / name).exists() for name in SCRIPTS)

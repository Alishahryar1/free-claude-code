"""Native update commands delegate to the installer and survive replacement."""

import contextlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL_URL = "https://raw.githubusercontent.com/Alishahryar1/free-claude-code/main/scripts/install"
SCRIPTS = ("fcc-update", "fcc-update.cmd")


@contextlib.contextmanager
def installer_server(body, status, outcome):
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            payload = body.encode("utf-8")
            response_status = status
            if self.path == "/metadata":
                version = {"current": "1.0.0", "newer-installed": "0.9.0"}.get(
                    outcome, "2.0.0"
                )
                payload = json.dumps(
                    {"info": {"version": version}, "urls": [{"yanked": False}]}
                ).encode()
                response_status = 503 if outcome == "check-error" else 200
            self.send_response(response_status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/install", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def fixture_wheel(area, url):
    wheel = area / "free_claude_code-1.0.0-py3-none-any.whl"
    dist = "free_claude_code-1.0.0"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{dist}.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: free-claude-code\nVersion: 1.0.0\n"
            "Requires-Dist: httpx>=0.28.1\nRequires-Dist: packaging>=26.3\n",
        )
        archive.writestr(
            f"{dist}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        for name in SCRIPTS:
            source = (ROOT / "scripts/update" / name).read_text(encoding="utf-8")
            archive.writestr(
                f"{dist}.data/scripts/{name}", source.replace(INSTALL_URL, url)
            )
        archive.writestr(
            f"{dist}.dist-info/entry_points.txt",
            "[console_scripts]\n_fcc-update-check = free_claude_code.updater.check:main\n",
        )
        for name in ("__init__.py", "updater/__init__.py", "updater/check.py"):
            source = (ROOT / "src/free_claude_code" / name).read_text(encoding="utf-8")
            archive.writestr(
                f"free_claude_code/{name}",
                source.replace(
                    "https://pypi.org/pypi/free-claude-code/json",
                    url.removesuffix("/install") + "/metadata",
                ),
            )
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
        "current",
        "newer-installed",
        "check-error",
    ],
)
def test_installed_update_delegates_and_survives_replacement(tmp_path, outcome):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for native script lifecycle coverage")
    area = tmp_path / "José 测试 & (spaces) ! apostrophe'"
    area.mkdir()
    env = dict(os.environ)
    env.update(
        UV_TOOL_DIR=str(area / "tools"),
        UV_TOOL_BIN_DIR=str(area / "bin"),
        UV_CACHE_DIR=str(area / "cache"),
        UV_NO_CONFIG="1",
        UV_PYTHON_DOWNLOADS="never",
        FCC_TEST_LAUNCHER=str(
            area / "bin" / ("fcc-update.cmd" if os.name == "nt" else "fcc-update")
        ),
        FCC_TEST_CHECKER=str(
            area
            / "bin"
            / ("_fcc-update-check.exe" if os.name == "nt" else "_fcc-update-check")
        ),
    )
    if os.name == "nt":
        body = """param([switch] $VoiceLocal, [string] $TorchBackend)
$answer = Read-Host 'Continue'
Write-Output "installer:$VoiceLocal|$TorchBackend|$answer"
[IO.File]::WriteAllText($env:FCC_TEST_LAUNCHER, 'exit /b 99')
[IO.File]::WriteAllText($env:FCC_TEST_CHECKER, 'replaced after checker exited')
"""
        arguments = "-VoiceLocal -TorchBackend cu130"
        expected = "installer:True|cu130|yes"
        body += {
            "success": "",
            "exit": "exit 23",
            "error": "throw 'installer failed'",
            "download-error": "",
        }.get(outcome, "")
    else:
        body = """read -r answer
printf 'installer:%s|%s|%s|%s\\n' "$1" "$2" "$3" "$answer"
printf 'exit 99\\n' > "$FCC_TEST_LAUNCHER"
printf 'replaced after checker exited' > "$FCC_TEST_CHECKER"
"""
        arguments = ["--voice-local", "--torch-backend", "cu130"]
        expected = "installer:--voice-local|--torch-backend|cu130|yes"
        body += {
            "success": "",
            "exit": "exit 23",
            "error": "exit 1",
            "download-error": "",
        }.get(outcome, "")
    with installer_server(
        body, 503 if outcome == "download-error" else 200, outcome
    ) as (
        url,
        requests,
    ):
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
        skipped = outcome in {"current", "newer-installed", "check-error"}
        assert requests == ["/metadata"] + (
            [] if skipped else ["/install.ps1" if os.name == "nt" else "/install.sh"]
        ), result.stdout + result.stderr
        if skipped:
            assert result.returncode == (1 if outcome == "check-error" else 0)
            assert "installer:" not in result.stdout
        elif outcome == "download-error":
            assert result.returncode != 0
            assert "installer:" not in result.stdout
        else:
            assert expected in result.stdout, result.stdout + result.stderr
            assert (
                result.returncode == {"success": 0, "exit": 23, "error": 1}[outcome]
            ), result.stderr
        result = subprocess.run(
            [uv, "tool", "uninstall", "free-claude-code"],
            env=env,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert not any((area / "bin" / name).exists() for name in SCRIPTS)
        assert not any((area / "bin").glob("_fcc-update-check*"))

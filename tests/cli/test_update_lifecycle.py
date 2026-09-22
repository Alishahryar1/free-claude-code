"""Exercise real uv script ownership and self-replacement in disposable tools."""

import base64
import contextlib
import http.server
import io
import os
import shutil
import subprocess
import sys
import threading
import tomllib
import zipfile
from pathlib import Path

import pytest

from free_claude_code.cli import update

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ("fcc-update", "fcc-update.cmd", "fcc-update-windows.ps1")


def fixture_archive(url, version, marker, *, verify_exit=None):
    """Build offline fixtures; production has no alternate update-source switch."""
    buffer = io.BytesIO()
    dist = f"free_claude_code-{version}"
    source = Path(update.__file__).read_text(encoding="utf-8")
    source = source.replace(update.ARCHIVE_URL, url)
    # Match only the fixture's processes; never stop or interact with the real FCC.
    source = source.replace(
        'if __name__ == "__main__":',
        'FCC_COMMANDS = frozenset({"fcc-fixture-busy"})\nif __name__ == "__main__":',
    )
    if verify_exit is not None:
        source = source.replace(
            'if __name__ == "__main__":',
            f"original_prepare = prepare\ndef prepare(*args):\n"
            f"    plan = original_prepare(*args)\n"
            f"    plan['verify'] = [sys.executable, '-c', 'raise SystemExit({verify_exit})']\n"
            f"    return plan\nif __name__ == '__main__':",
        )
    with zipfile.ZipFile(buffer, "w") as wheel:
        wheel.writestr("free_claude_code/__init__.py", "")
        wheel.writestr("free_claude_code/cli/__init__.py", "")
        wheel.writestr("free_claude_code/cli/update.py", source)
        wheel.writestr("free_claude_code/marker.txt", marker)
        wheel.writestr(
            f"{dist}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: free-claude-code\nVersion: {version}\nProvides-Extra: voice\n",
        )
        wheel.writestr(
            f"{dist}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        for name in SCRIPTS:
            info = zipfile.ZipInfo(f"{dist}.data/scripts/{name}")
            # Match Hatch's wheel built on Windows; uv must make shell scripts executable.
            info.external_attr = (0o100755 if name.endswith(".cmd") else 0o100644) << 16
            wheel.writestr(info, (ROOT / "scripts/update" / name).read_bytes())
        wheel.writestr(
            f"{dist}.dist-info/RECORD",
            "".join(f"{name},,\n" for name in wheel.namelist())
            + f"{dist}.dist-info/RECORD,,\n",
        )
    # A dependency-free build backend serves the prepared wheel as a source zip.
    backend = f"import base64\nfrom pathlib import Path\ndef build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n    name = '{dist}-py3-none-any.whl'\n    Path(wheel_directory, name).write_bytes(base64.b64decode({base64.b64encode(buffer.getvalue())!r}))\n    return name\n"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as source_zip:
        source_zip.writestr(
            "fcc/pyproject.toml",
            '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n',
        )
        source_zip.writestr("fcc/backend.py", backend)
    return archive.getvalue()


@contextlib.contextmanager
def archive_server():
    payload = {"body": b""}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(payload["body"])))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload["body"])

        def log_message(self, format: str, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/main.zip", payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run(args, env, cwd):
    return subprocess.run(
        args, env=env, cwd=cwd, capture_output=True, text=True, timeout=90
    )


def test_installed_updater_replaces_itself_and_uv_removes_scripts(tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for real package-manager lifecycle coverage")
    area = tmp_path / "path with spaces & (brackets) ! apostrophe'"
    area.mkdir()
    env = dict(os.environ)
    env.update(
        UV_TOOL_DIR=str(area / "tools"),
        UV_TOOL_BIN_DIR=str(area / "bin"),
        UV_CACHE_DIR=str(area / "cache"),
        UV_NO_CONFIG="1",
        UV_HTTP_RETRIES="0",
        UV_PYTHON_DOWNLOADS="never",
    )
    env["PATH"] = str(Path(uv).parent) + os.pathsep + env["PATH"]
    sentinel = area / "settings.json"
    sentinel.write_text("keep me")
    with archive_server() as (url, payload):
        payload["body"] = fixture_archive(url, "1.0.0", "first")
        result = run(
            [
                uv,
                "tool",
                "install",
                "--python",
                getattr(sys, "_base_executable", sys.executable),
                f"free-claude-code[voice] @ {url}",
            ],
            env,
            area,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        bin_dir = area / "bin"
        assert all((bin_dir / name).is_file() for name in SCRIPTS)
        root = area / "tools/free-claude-code"
        python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        launcher = bin_dir / ("fcc-update.cmd" if os.name == "nt" else "fcc-update")
        command = (
            f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /s /c ""{launcher}""'
            if os.name == "nt"
            else [str(launcher)]
        )
        for version, marker in [
            ("1.1.0", "second"),
            ("1.1.0", "same-version-new-code"),
        ]:
            payload["body"] = fixture_archive(url, version, marker)
            result = run(command, env, area)
            assert result.returncode == 0, result.stdout + result.stderr
            assert f"FCC refreshed (version {version})" in result.stdout
            installed = run(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import free_claude_code; from pathlib import Path; print(Path(free_claude_code.__file__).with_name('marker.txt').read_text())",
                ],
                env,
                area,
            )
            assert installed.stdout.strip() == marker
            receipt = tomllib.loads((root / "uv-receipt.toml").read_text())
            assert receipt["tool"]["requirements"][0]["extras"] == ["voice"]
        payload["body"] = b"invalid archive"
        result = run(command, env, area)
        direct_failure = run(
            [
                uv,
                "tool",
                "install",
                "--force",
                "--refresh-package",
                "free-claude-code",
                f"free-claude-code[voice] @ {url}",
            ],
            env,
            area,
        )
        assert result.returncode == direct_failure.returncode != 0
        assert "FCC refreshed" not in result.stdout
        payload["body"] = fixture_archive(
            url, "1.2.0", "verification-error", verify_exit=23
        )
        result = run(
            [
                uv,
                "tool",
                "install",
                "--force",
                "--refresh-package",
                "free-claude-code",
                "--python",
                getattr(sys, "_base_executable", sys.executable),
                f"free-claude-code[voice] @ {url}",
            ],
            env,
            area,
        )
        assert result.returncode == 0, result.stderr
        payload["body"] = fixture_archive(
            url, "1.3.0", "installed-despite-verification-error"
        )
        result = run(command, env, area)
        assert result.returncode == 23, result.stdout + result.stderr
        assert "FCC refreshed" not in result.stdout
        assert sentinel.read_text() == "keep me"
        result = run([uv, "tool", "uninstall", "free-claude-code"], env, area)
        assert result.returncode == 0, result.stderr
        assert not any((bin_dir / name).exists() for name in SCRIPTS)

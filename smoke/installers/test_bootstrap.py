"""Exercise release selection and handoff without installing anything."""

import contextlib
import http.server
import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "https://github.com/Alishahryar1/free-claude-code"
RAW = "https://raw.githubusercontent.com/Alishahryar1/free-claude-code"


@contextlib.contextmanager
def release_server(body, target="v2.0.0", status=200, metadata=""):
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path == "/repo/releases/latest":
                self.send_response(302)
                # If incorrectly resolved again, the new latest release wins.
                selected = target if requests.count(self.path) == 1 else "v3.0.0"
                self.send_header("Location", f"/repo/releases/tag/{selected}")
                payload = b""
            else:
                installer = "/scripts/install." in self.path
                self.send_response(status if installer else 200)
                if self.path.endswith("/pyproject.toml"):
                    payload = metadata.encode()
                else:
                    payload = (
                        body.replace(
                            RAW, f"http://127.0.0.1:{server.server_port}/raw"
                        ).encode()
                        if installer
                        else b"release page"
                    )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
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


@pytest.fixture(params=["sh", "powershell", "pwsh"])
def bootstrap(request, tmp_path, powershell_module_paths):
    shell_name = request.param
    windows = shell_name != "sh"
    if windows != (os.name == "nt"):
        pytest.skip("bootstrap is tested on its native platform")
    shell = shutil.which(shell_name)
    if shell is None:
        pytest.skip(f"{shell_name} is unavailable")
    suffix = "ps1" if windows else "sh"
    script = tmp_path / f"bootstrap.{suffix}"
    env = os.environ | {
        "FCC_INSTALL_VERSION": "",
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
    }
    if windows:
        env["PSModulePath"] = powershell_module_paths[shell]
        env["LOCALAPPDATA"] = str(tmp_path / "local")
        env["APPDATA"] = str(tmp_path / "roaming")

    def run(url, *args, version="", input="yes\n", body=None):
        source = (
            body
            if body is not None
            else (ROOT / f"scripts/bootstrap.{suffix}").read_text(encoding="utf-8")
        )
        script.write_text(
            source.replace(RAW, url + "/raw").replace(REPOSITORY, url + "/repo"),
            encoding="utf-8",
        )
        command = (
            [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
            if windows
            else [shell]
        )
        return subprocess.run(
            [*command, str(script), *args],
            env=env | {"FCC_INSTALL_VERSION": version},
            input=input,
            capture_output=True,
            text=True,
            timeout=30,
        )

    return windows, suffix, run


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("exit_code", [0, 23])
def test_bootstrap_freezes_release_and_preserves_invocation(
    bootstrap, explicit, exit_code
):
    windows, suffix, run = bootstrap
    if windows:
        body = 'param([switch] $VoiceLocal, [string] $TorchBackend, [switch] $RtkHeadroom)\n$answer = Read-Host\nWrite-Output "installed:$env:FCC_INSTALL_VERSION|$VoiceLocal|$TorchBackend|$RtkHeadroom|$answer"\n'
        args = [
            "-VoiceLocal",
            "-TorchBackend",
            "custom backend 'quoted'",
            "-RtkHeadroom",
        ]
        expected = "installed:2.0.0|True|custom backend 'quoted'|True|yes"
    else:
        body = 'read -r answer\nprintf "installed:%s|%s|%s|%s|%s|%s\\n" "$FCC_INSTALL_VERSION" "$1" "$2" "$3" "$4" "$answer"\n'
        args = [
            "--voice-local",
            "--torch-backend",
            "custom backend 'quoted'",
            "--rtk-headroom",
        ]
        expected = "installed:2.0.0|--voice-local|--torch-backend|custom backend 'quoted'|--rtk-headroom|yes"
    body += f"exit {exit_code}\n"
    with release_server(body) as (url, requests):
        result = run(url, *args, version="2.0.0" if explicit else "")
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert expected in result.stdout
    assert requests == (
        [] if explicit else ["/repo/releases/latest", "/repo/releases/tag/v2.0.0"]
    ) + [f"/raw/v2.0.0/scripts/install.{suffix}"]


@pytest.mark.parametrize("mode", ["help", "dry-run", "invalid", "backend", "version"])
def test_bootstrap_preview_and_invalid_input_do_not_download(bootstrap, mode):
    windows, _, run = bootstrap
    options = {
        "help": ["-Help" if windows else "--help"],
        "dry-run": ["-DryRun" if windows else "--dry-run"],
        "invalid": ["--invalid"],
        "backend": ["-TorchBackend" if windows else "--torch-backend", "cu130"],
        "version": [],
    }
    with release_server("exit 0") as (url, requests):
        result = run(
            url, *options[mode], version="2.0.0/../main" if mode == "version" else ""
        )
    assert (result.returncode == 0) == (mode in {"help", "dry-run"})
    assert requests == []


@pytest.mark.parametrize(
    "version", ["2.0.0\n3.0.0", "2.0.0\ninvalid", "02.0.0", "v2.0.0", "2.0.0+local"]
)
def test_bootstrap_rejects_noncanonical_explicit_version(bootstrap, version):
    _, _, run = bootstrap
    with release_server("exit 0") as (url, requests):
        result = run(url, version=version)
    assert result.returncode != 0
    assert requests == []
    assert "canonical major.minor.patch" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "target",
    [
        "v2.0.0rc1",
        "v02.0.0",
        "v2.0.0?query=1",
        "v2.0.0#fragment",
        "v2.0.0/extra",
        "../../other",
    ],
)
def test_bootstrap_rejects_invalid_selected_release(bootstrap, target):
    _, _, run = bootstrap
    with release_server("exit 0", target=target) as (url, requests):
        result = run(url)
    assert result.returncode != 0
    assert not any("/scripts/" in request for request in requests)


@pytest.mark.parametrize("body,status", [("", 200), ("exit 0", 503), ("if (", 200)])
def test_bootstrap_rejects_bad_installer_download(bootstrap, body, status):
    _, _, run = bootstrap
    with release_server(body, status=status) as (url, requests):
        result = run(url)
    assert result.returncode != 0
    assert requests[0] == "/repo/releases/latest"
    assert "2.0.0" in result.stdout + result.stderr


def test_bootstrap_and_tagged_installer_share_release_metadata(bootstrap):
    windows, suffix, run = bootstrap
    source = (ROOT / f"scripts/install.{suffix}").read_text(encoding="utf-8")
    # Stop at the host-mutation boundary, after the real tagged metadata loader.
    if windows:
        source = source.replace(
            "\nInitialize-InstallLog\n",
            '\nWrite-Output "toolchain:$FccInstallVersion|$PythonRequest|$MinUvVersion"\nreturn\nInitialize-InstallLog\n',
        )
        expected = "toolchain:2.0.0|cpython-3.15.2-windows-x86_64-none|0.14.2"
    else:
        source = source.replace(
            "\ninitialize_install_log\n",
            '\nprintf "toolchain:%s|%s|%s\\n" "$fcc_install_version" "$PYTHON_VERSION" "$MIN_UV_VERSION"\nexit 0\ninitialize_install_log\n',
        )
        expected = "toolchain:2.0.0|3.15.2|0.14.2"
    metadata = '[project]\nrequires-python = "==3.15.2"\n[tool.uv]\nrequired-version = ">=0.14.2"\n'
    with release_server(source, metadata=metadata) as (url, requests):
        result = run(url)
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout
    assert requests == [
        "/repo/releases/latest",
        "/repo/releases/tag/v2.0.0",
        f"/raw/v2.0.0/scripts/install.{suffix}",
        "/raw/v2.0.0/pyproject.toml",
    ]


def test_powershell_bootstrap_restores_release_environment(bootstrap):
    windows, _, run = bootstrap
    if not windows:
        pytest.skip("POSIX exec does not modify its parent environment")
    source = (ROOT / "scripts/bootstrap.ps1").read_text(encoding="utf-8")
    source += '\nWrite-Output "restored:$env:FCC_INSTALL_VERSION"\n'
    with release_server('Write-Output "selected:$env:FCC_INSTALL_VERSION"') as (url, _):
        result = run(url, body=source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "selected:2.0.0" in result.stdout
    assert "restored:\n" in result.stdout

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

FCC_COMMANDS = (
    "fcc-desktop",
    "fcc-server",
    "fcc-claude",
    "fcc-codex",
    "fcc-pi",
    "fcc-opencode",
    "fcc-cline",
    "fcc-hermes",
    "fcc-dsh",
    "fcc-grok",
    "fcc-muse",
    "fcc-aider",
    "fcc-init",
    "free-claude-code",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_uv_ready_without_fcc_install(calls: list[str]) -> None:
    assert "uv:--version" in calls
    assert not any("--refresh-package free-claude-code" in call for call in calls)


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _braced_body(text: str, declaration: str) -> str:
    start = text.index(declaration)
    brace_start = text.index("{", start)
    depth = 0
    for index, char in enumerate(text[brace_start:], start=brace_start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : index]
    raise AssertionError(f"Unclosed function body for {declaration}")


def _posix_command(name: str, *, version_output: str | None = None) -> str:
    version = {
        "opencode": "1.18.18",
        "cline": "3.0.55",
        "hermes": "0.20.4",
        "dsh": "0.1.0-rc.8",
        "grok": "1.0.5",
        "muse": "0.2.1",
        "node": "22.19.0",
    }.get(name, "1.0.0")
    if version_output is None:
        if name == "hermes":
            version_output = f"Hermes Agent v{version} (test build)"
        elif name == "grok":
            version_output = f"grok {version} (5115b46bc9) [stable]"
        elif name == "muse":
            version_output = f"Muse Code {version} ({version}-R1215.1)"
        else:
            version_output = f"{name} {version}"
    version_command = f'''if [ "${{1:-}}" = "--version" ]; then
    echo "{version_output}"
fi'''
    help_output = (
        '    echo "  --extension, -e <path>  Load an extension"\n'
        '    echo "  --models <patterns>     Scope models"'
        if name == "pi"
        else "    :"
    )
    return f"""#!/bin/sh
echo "{name}:$*" >> "$CALL_LOG"
if [ "$FAIL_STEP" = "{name}-verify" ]; then
    exit 31
fi
{version_command}
if [ "${{1:-}}" = "--help" ]; then
{help_output}
fi
"""


def _posix_npm_command() -> str:
    return """#!/bin/sh
echo "npm:$*" >> "$CALL_LOG"
if [ "${1:-}" = "install" ] && [ "${2:-}" = "-g" ] && [ "${3:-}" = "cline" ]; then
    [ "$FAIL_STEP" = "cline-install" ] && exit 72
    mkdir -p "$FAKE_NPM_PREFIX/bin"
    cp "$FAKE_FIXTURES/cline-command.sh" "$FAKE_NPM_PREFIX/bin/cline"
    chmod +x "$FAKE_NPM_PREFIX/bin/cline"
    exit 0
fi
if [ "${1:-}" = "install" ] && [ "${2:-}" = "-g" ] && [ "${3:-}" = "@deepseek-ai/dsh@0.1.0-rc.8" ]; then
    [ "$FAIL_STEP" = "dsh-install" ] && exit 73
    mkdir -p "$FAKE_NPM_PREFIX/bin"
    cp "$FAKE_FIXTURES/dsh-command.sh" "$FAKE_NPM_PREFIX/bin/dsh"
    chmod +x "$FAKE_NPM_PREFIX/bin/dsh"
    exit 0
fi
if [ "${1:-}" = "prefix" ] && [ "${2:-}" = "-g" ]; then
    printf '%s\n' "$FAKE_NPM_PREFIX"
    exit 0
fi
if [ "${1:-}" = "config" ] && [ "${2:-}" = "get" ] && [ "${3:-}" = "prefix" ]; then
    printf '%s\n' "$FAKE_NPM_PREFIX"
    exit 0
fi
exit 71
"""


def _posix_uv_command(version: str) -> str:
    return f"""#!/bin/sh
echo "uv:$*" >> "$CALL_LOG"
tool_bin=${{UV_TOOL_BIN_DIR:-$FAKE_TOOL_BIN}}
if [ "${{1:-}}" = "--version" ]; then
    if [ "${{FCC_RUNNING_PHASE:-}}" = "late" ]; then
        : > "$FCC_PROCESS_MARKER"
    fi
    if [ "$FAIL_STEP" = "uv-verify" ]; then
        exit 32
    fi
    echo "uv {version}"
    exit 0
fi
if [ "${{1:-}}" = "tool" ] && [ "${{2:-}}" = "install" ]; then
    case " $* " in
        *" aider-chat@latest "*)
            if [ "$FAIL_STEP" = "aider-install" ]; then
                exit 29
            fi
            mkdir -p "$tool_bin"
            cp "$FAKE_FIXTURES/aider-command.sh" "$tool_bin/aider"
            chmod +x "$tool_bin/aider"
            exit 0
            ;;
    esac
    if [ "$FAIL_STEP" = "fcc-install" ]; then
        exit 33
    fi
    mkdir -p "$tool_bin"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-server"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-desktop"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-claude"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-pi"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-opencode"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-cline"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-hermes"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-dsh"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-grok"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-muse"
    cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-aider"
    if [ "$FAIL_STEP" != "fcc-missing" ]; then
        cp "$FAKE_FIXTURES/fcc-command.sh" "$tool_bin/fcc-codex"
    fi
    chmod +x "$tool_bin"/fcc-*
    exit 0
fi
if [ "${{1:-}}" = "tool" ] && [ "${{2:-}}" = "update-shell" ]; then
    if [ "$FAIL_STEP" = "path-update" ]; then
        exit 34
    fi
    exit 0
fi
if [ "${{1:-}}" = "tool" ] && [ "${{2:-}}" = "dir" ] && [ "${{3:-}}" = "--bin" ]; then
    printf '%s\n' "$tool_bin"
    exit 0
fi
exit 35
"""


def _posix_rtk_command() -> str:
    return """#!/bin/sh
echo "rtk:$*:telemetry=${RTK_TELEMETRY_DISABLED:-}" >> "$CALL_LOG"
if [ "$*" = "init --global --auto-patch" ]; then
    claude_config_directory=${CLAUDE_CONFIG_DIR:-$HOME/.claude}
    [ -d "$claude_config_directory" ] || exit 77
fi
case "${1:-}:$FAIL_STEP" in
    --version:rtk-verify|gain:rtk-verify) exit 72 ;;
esac
case "$*:$FAIL_STEP" in
    "init --global --auto-patch:rtk-init-claude") exit 73 ;;
    "init --global --codex:rtk-init-codex") exit 74 ;;
    "init --global --agent pi:rtk-init-pi") exit 75 ;;
    "init --global --opencode:rtk-init-opencode") exit 76 ;;
esac
if [ "${1:-}" = "--version" ]; then
    echo "rtk 0.44.2"
fi
exit 0
"""


@dataclass
class PosixHarness:
    root: Path
    bin_dir: Path
    fixtures: Path
    tool_bin: Path
    log: Path
    env: dict[str, str]

    def add_client(self, name: str) -> None:
        _write_executable(self.bin_dir / name, _posix_command(name))

    def add_unrelated_pi(self) -> None:
        _write_executable(self.bin_dir / "pi", _posix_command("unrelated-pi"))

    def add_npm_prefix(self, prefix: Path) -> None:
        prefix.mkdir(parents=True)
        self.env["FAKE_NPM_PREFIX"] = str(prefix)
        _write_executable(self.bin_dir / "npm", _posix_npm_command())

    def add_uv(self, version: str) -> None:
        _write_executable(self.bin_dir / "uv", _posix_uv_command(version))

    def add_rtk(self) -> None:
        _write_executable(self.bin_dir / "rtk", _posix_rtk_command())

    def add_unrelated_rtk(self) -> None:
        _write_executable(
            self.bin_dir / "rtk",
            """#!/bin/sh
echo "unrelated-rtk:$*" >> "$CALL_LOG"
[ "${1:-}" = "--version" ] && echo "rtk 1.0.0" && exit 0
exit 76
""",
        )

    def use_process_list_fallback(self, process_line: str) -> None:
        fallback_bin = self.root / "fallback-bin"
        fallback_bin.mkdir()
        _write_executable(
            fallback_bin / "ps",
            """#!/bin/sh
printf '%s\n' "$FCC_PS_OUTPUT"
""",
        )
        awk = shutil.which("awk", path=self.env["PATH"])
        if awk is None:
            pytest.skip("awk is required for the POSIX process fallback scenario")
        shutil.copy2(awk, fallback_bin / "awk")
        self.env["FCC_PS_OUTPUT"] = process_line
        self.env["PATH"] = str(fallback_bin)

    def run(self, *args: str, fail_step: str = "") -> subprocess.CompletedProcess[str]:
        env = self.env | {"FAIL_STEP": fail_step}
        return subprocess.run(
            ["/bin/sh", str(_repo_root() / "scripts" / "install.sh"), *args],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def run_interactive(
        self,
        answers: str,
        *args: str,
        fail_step: str = "",
    ) -> subprocess.CompletedProcess[str]:
        env = self.env | {
            "FAIL_STEP": fail_step,
            "FCC_INSTALLER": str(_repo_root() / "scripts" / "install.sh"),
        }
        command = [
            "/bin/sh",
            "-c",
            'cat "$FCC_INSTALLER" | /bin/sh -s -- "$@"',
            "fcc-installer",
            *args,
        ]
        return subprocess.run(
            [
                sys.executable,
                "-W",
                "error",
                str(Path(__file__).with_name("_pty_runner.py")),
                *command,
            ],
            input=answers,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    def calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return self.log.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def posix_harness(tmp_path: Path) -> PosixHarness:
    if os.name == "nt":
        pytest.skip("POSIX installer scenarios run on POSIX hosts")

    bin_dir = tmp_path / "bin"
    fixtures = tmp_path / "fixtures"
    tool_bin = tmp_path / "tool-bin"
    home = tmp_path / "home"
    log = tmp_path / "calls.log"
    for path in (bin_dir, fixtures, tool_bin, home):
        path.mkdir(parents=True)

    _write_executable(
        bin_dir / "pgrep",
        """#!/bin/sh
[ -n "${FCC_RUNNING_COMMAND:-}" ] || exit 1
if [ "${FCC_RUNNING_PHASE:-early}" = "late" ] && [ ! -e "$FCC_PROCESS_MARKER" ]; then
    exit 1
fi
case "$*" in
    *"$FCC_RUNNING_COMMAND"*) printf '4242\n'; exit 0 ;;
    *) exit 1 ;;
esac
""",
    )
    _write_executable(
        bin_dir / "getent",
        """#!/bin/sh
[ "${1:-}" = "passwd" ] || exit 61
printf 'fcc-test:x:1000:1000::%s:/bin/sh\n' "$FAKE_INFERRED_HOME"
""",
    )
    _write_executable(
        bin_dir / "curl",
        """#!/bin/sh
url=""
output=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o)
            shift
            output=$1
            ;;
        http*)
            url=$1
            ;;
    esac
    shift
done
echo "download:$url" >> "$CALL_LOG"
case "$url:$FAIL_STEP" in
    *claude.ai*:claude-download|*chatgpt.com*:codex-download|*pi.dev*:pi-download|*opencode.ai*:opencode-download|*hermes-agent.nousresearch.com*:hermes-download|*x.ai*:grok-download|*dev.meta.ai*:muse-download|*rtk-ai*:rtk-download|*astral.sh*:uv-download)
        exit 41
        ;;
esac
case "$url" in
    *claude.ai*) source="$FAKE_FIXTURES/claude-installer.sh" ;;
    *chatgpt.com*) source="$FAKE_FIXTURES/codex-installer.sh" ;;
    *pi.dev*) source="$FAKE_FIXTURES/pi-installer.sh" ;;
    *opencode.ai*) source="$FAKE_FIXTURES/opencode-installer.sh" ;;
    *hermes-agent.nousresearch.com*) source="$FAKE_FIXTURES/hermes-installer.sh" ;;
    *x.ai*) source="$FAKE_FIXTURES/grok-installer.sh" ;;
    *dev.meta.ai*) source="$FAKE_FIXTURES/muse-installer.sh" ;;
    *rtk-ai*)
        if [ "$FAIL_STEP" = "rtk-install" ]; then
            printf 'invalid archive\n' > "$output"
            exit 0
        fi
        source="$FAKE_FIXTURES/rtk-x86_64-unknown-linux-musl.tar.gz"
        ;;
    *astral.sh*) source="$FAKE_FIXTURES/uv-installer.sh" ;;
    *) exit 42 ;;
esac
cp "$source" "$output"
""",
    )
    _write_executable(
        fixtures / "claude-installer.sh",
        """#!/bin/sh
echo "claude-install" >> "$CALL_LOG"
[ "$FAIL_STEP" = "claude-install" ] && exit 21
mkdir -p "$HOME/.local/bin"
cp "$FAKE_FIXTURES/claude-command.sh" "$HOME/.local/bin/claude"
chmod +x "$HOME/.local/bin/claude"
""",
    )
    _write_executable(
        fixtures / "codex-installer.sh",
        """#!/bin/sh
echo "codex-install:$CODEX_NON_INTERACTIVE" >> "$CALL_LOG"
[ "$FAIL_STEP" = "codex-install" ] && exit 22
mkdir -p "$HOME/.local/bin"
cp "$FAKE_FIXTURES/codex-command.sh" "$HOME/.local/bin/codex"
chmod +x "$HOME/.local/bin/codex"
""",
    )
    _write_executable(
        fixtures / "pi-installer.sh",
        """#!/bin/sh
echo "pi-install" >> "$CALL_LOG"
[ "$FAIL_STEP" = "pi-install" ] && exit 24
[ "$FAIL_STEP" = "pi-skip" ] && exit 0
if [ -n "${FAKE_NPM_PREFIX:-}" ]; then
    pi_bin="$FAKE_NPM_PREFIX/bin"
else
    pi_bin="$HOME/.local/bin"
fi
mkdir -p "$pi_bin"
cp "$FAKE_FIXTURES/pi-command.sh" "$pi_bin/pi"
chmod +x "$pi_bin/pi"
""",
    )
    _write_executable(
        fixtures / "opencode-installer.sh",
        """#!/bin/sh
echo "opencode-install" >> "$CALL_LOG"
[ "$FAIL_STEP" = "opencode-install" ] && exit 25
mkdir -p "$HOME/.opencode/bin"
cp "$FAKE_FIXTURES/opencode-command.sh" "$HOME/.opencode/bin/opencode"
chmod +x "$HOME/.opencode/bin/opencode"
""",
    )
    _write_executable(
        fixtures / "hermes-installer.sh",
        """#!/bin/sh
echo "hermes-install:$*" >> "$CALL_LOG"
[ "$FAIL_STEP" = "hermes-install" ] && exit 26
mkdir -p "$HOME/.local/bin"
cp "$FAKE_FIXTURES/hermes-command.sh" "$HOME/.local/bin/hermes"
chmod +x "$HOME/.local/bin/hermes"
""",
    )
    _write_executable(
        fixtures / "grok-installer.sh",
        """#!/bin/sh
echo "grok-install" >> "$CALL_LOG"
[ "$FAIL_STEP" = "grok-install" ] && exit 27
grok_bin="${GROK_BIN_DIR:-$HOME/.grok/bin}"
mkdir -p "$grok_bin"
cp "$FAKE_FIXTURES/grok-command.sh" "$grok_bin/grok"
chmod +x "$grok_bin/grok"
""",
    )
    _write_executable(
        fixtures / "muse-installer.sh",
        """#!/bin/sh
echo "muse-install" >> "$CALL_LOG"
[ "$FAIL_STEP" = "muse-install" ] && exit 28
mkdir -p "$HOME/.local/bin"
cp "$FAKE_FIXTURES/muse-command.sh" "$HOME/.local/bin/muse"
chmod +x "$HOME/.local/bin/muse"
""",
    )
    _write_executable(
        fixtures / "uv-installer.sh",
        """#!/bin/sh
echo "uv-install" >> "$CALL_LOG"
[ "$FAIL_STEP" = "uv-install" ] && exit 23
inferred_home=${HOME:-}
if [ -z "$inferred_home" ]; then
    if [ -n "${USER:-}" ]; then
        inferred_home=$(getent passwd "$USER" | cut -d: -f6)
    else
        inferred_home=$(getent passwd "$(id -un)" | cut -d: -f6)
    fi
fi
force_install_dir=""
if [ -n "${UV_INSTALL_DIR:-}" ]; then
    force_install_dir=$UV_INSTALL_DIR
elif [ -n "${UV_UNMANAGED_INSTALL:-}" ]; then
    force_install_dir=$UV_UNMANAGED_INSTALL
fi
if [ -n "$force_install_dir" ]; then
    uv_bin=$force_install_dir
    if [ "$force_install_dir" = "${CARGO_HOME:-$inferred_home/.cargo}" ]; then
        uv_bin=$force_install_dir/bin
    fi
elif [ -n "${XDG_BIN_HOME:-}" ]; then
    uv_bin=$XDG_BIN_HOME
elif [ -n "${XDG_DATA_HOME:-}" ]; then
    uv_bin=$XDG_DATA_HOME/../bin
else
    uv_bin=$inferred_home/.local/bin
fi
mkdir -p "$uv_bin"
cp "$FAKE_FIXTURES/uv-command.sh" "$uv_bin/uv"
chmod +x "$uv_bin/uv"
""",
    )
    _write_executable(fixtures / "claude-command.sh", _posix_command("claude"))
    _write_executable(fixtures / "codex-command.sh", _posix_command("codex"))
    _write_executable(fixtures / "pi-command.sh", _posix_command("pi"))
    _write_executable(fixtures / "opencode-command.sh", _posix_command("opencode"))
    _write_executable(fixtures / "cline-command.sh", _posix_command("cline"))
    _write_executable(fixtures / "hermes-command.sh", _posix_command("hermes"))
    _write_executable(fixtures / "dsh-command.sh", _posix_command("dsh"))
    _write_executable(fixtures / "grok-command.sh", _posix_command("grok"))
    _write_executable(fixtures / "muse-command.sh", _posix_command("muse"))
    _write_executable(fixtures / "aider-command.sh", _posix_command("aider"))
    rtk_command = _posix_rtk_command().encode()
    with tarfile.open(
        fixtures / "rtk-x86_64-unknown-linux-musl.tar.gz", "w:gz"
    ) as archive:
        metadata = tarfile.TarInfo("rtk")
        metadata.mode = 0o755
        metadata.size = len(rtk_command)
        archive.addfile(metadata, io.BytesIO(rtk_command))
    _write_executable(fixtures / "uv-command.sh", _posix_uv_command("0.12.13"))
    _write_executable(
        fixtures / "fcc-command.sh",
        """#!/bin/sh
name=${0##*/}
echo "$name:$*" >> "$CALL_LOG"
if [ "$FAIL_STEP" = "fcc-verify" ]; then
    exit 36
fi
if [ "$name" = "fcc-desktop" ] && [ "${1:-}" = "--export-icon" ]; then
    [ "$FAIL_STEP" = "desktop-icon-export" ] && exit 37
    mkdir -p "$(dirname "$2")"
    printf 'fake icon\n' > "$2"
fi
if [ "$name" = "fcc-server" ] && [ "${1:-}" = "--version" ]; then
    echo "free-claude-code 3.5.18"
fi
""",
    )
    _write_executable(
        bin_dir / "uname",
        """#!/bin/sh
case "${1:-}" in
    -m) printf '%s\n' "${FAKE_UNAME_MACHINE:-x86_64}" ;;
    *) printf '%s\n' "${FAKE_UNAME:-Linux}" ;;
esac
""",
    )
    _write_executable(bin_dir / "opencode", _posix_command("opencode"))
    _write_executable(bin_dir / "node", _posix_command("node"))
    npm_prefix = tmp_path / "npm-prefix"
    npm_prefix.mkdir()
    _write_executable(bin_dir / "npm", _posix_npm_command())
    _write_executable(
        bin_dir / "sha256sum",
        """#!/bin/sh
echo "sha256sum:$*" >> "$CALL_LOG"
if [ "$FAIL_STEP" = "rtk-checksum" ]; then
    checksum="0000000000000000000000000000000000000000000000000000000000000000"
else
    checksum="d94cc2a3e57fa534892b5235a726e7eeb7523f205a5f8f48f853bfcae7be7e33"
fi
printf '%s  %s\n' "$checksum" "$1"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(home),
            "CALL_LOG": str(log),
            "FAKE_FIXTURES": str(fixtures),
            "FAKE_TOOL_BIN": str(tool_bin),
            "FAKE_INFERRED_HOME": str(home),
            "FCC_PROCESS_MARKER": str(tmp_path / "fcc-process-ready"),
            "FCC_RUNNING_COMMAND": "",
            "FCC_RUNNING_PHASE": "early",
            "FAKE_UNAME": "Linux",
            "FAKE_NPM_PREFIX": str(npm_prefix),
            "USER": "fcc-test",
            "CLAUDE_CONFIG_DIR": "",
            "FAIL_STEP": "",
        }
    )
    env.pop("XDG_BIN_HOME", None)
    env.pop("XDG_DATA_HOME", None)
    env.pop("UV_INSTALL_DIR", None)
    env.pop("UV_UNMANAGED_INSTALL", None)
    env.pop("UV_TOOL_BIN_DIR", None)
    env.pop("CARGO_HOME", None)
    env.pop("GROK_BIN_DIR", None)
    return PosixHarness(tmp_path, bin_dir, fixtures, tool_bin, log, env)


def test_install_sh_fresh_install_is_verified(posix_harness: PosixHarness) -> None:
    (posix_harness.bin_dir / "opencode").unlink()
    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "Free Claude Code is installed and verified." in result.stdout
    calls = posix_harness.calls()
    assert calls.index("claude-install") < calls.index("claude:--version")
    assert calls.index("codex-install:1") < calls.index("codex:--version")
    assert calls.index("pi-install") < calls.index("pi:--version")
    assert calls.index("opencode-install") < calls.index("opencode:--version")
    assert calls.index("npm:install -g cline") < calls.index("cline:--version")
    assert calls.index("hermes-install:--non-interactive --skip-setup") < calls.index(
        "hermes:--version"
    )
    assert calls.index("npm:install -g @deepseek-ai/dsh@0.1.0-rc.8") < calls.index(
        "dsh:--version"
    )
    assert calls.index("grok-install") < calls.index("grok:--version")
    assert calls.index("muse-install") < calls.index("muse:--version")
    assert calls.index("uv-install") < calls.index("uv:--version")
    aider_install = (
        "uv:tool install --force --python python3.12 --with pip aider-chat@latest"
    )
    assert calls.index("uv:--version") < calls.index("claude-install")
    assert calls.index(aider_install) < calls.index("aider:--version")
    assert any(
        call.startswith(
            "uv:tool install --force --refresh-package free-claude-code "
            "--python 3.14.0 free-claude-code @ "
            "https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
        )
        for call in calls
    )
    assert not any(call.startswith("git:") for call in calls)
    assert calls[-3:] == [
        "uv:tool update-shell",
        "uv:tool dir --bin",
        "fcc-server:--version",
    ]
    assert not any("hermes:setup" in call for call in calls)
    home = Path(posix_harness.env["HOME"])
    assert (home / ".grok" / "bin" / "grok").is_file()
    assert not (home / ".local" / "bin" / "grok").exists()


def test_install_sh_discovers_grok_in_custom_bin_directory(
    posix_harness: PosixHarness,
) -> None:
    custom_grok_bin = posix_harness.root / "custom-grok-bin"
    posix_harness.env["GROK_BIN_DIR"] = str(custom_grok_bin)

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert (custom_grok_bin / "grok").is_file()
    assert not (Path(posix_harness.env["HOME"]) / ".grok" / "bin" / "grok").exists()
    calls = posix_harness.calls()
    assert calls.index("grok-install") < calls.index("grok:--version")


def test_install_sh_installs_selected_hermes_without_setup(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run_interactive("n\nn\nn\nn\nn\ny\nn\nn\nn\nn\nn\n")

    assert result.returncode == 0, result.stdout
    calls = posix_harness.calls()
    assert "download:https://hermes-agent.nousresearch.com/install.sh" in calls
    assert "hermes-install:--non-interactive --skip-setup" in calls
    assert calls.index("hermes-install:--non-interactive --skip-setup") < calls.index(
        "hermes:--version"
    )
    assert "Run Hermes Agent with: fcc-hermes" in result.stdout
    assert not any("hermes:setup" in call for call in calls)
    assert not any(call.startswith("rtk:init") for call in calls)


@pytest.mark.parametrize("failure", ["hermes-download", "hermes-install"])
def test_install_sh_stops_when_selected_hermes_install_fails(
    posix_harness: PosixHarness,
    failure: str,
) -> None:
    result = posix_harness.run_interactive(
        "n\nn\nn\nn\nn\ny\nn\nn\nn\nn\nn\n", fail_step=failure
    )

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    _assert_uv_ready_without_fcc_install(posix_harness.calls())


def test_install_sh_rejects_unsupported_hermes_platform_before_download(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.env["FAKE_UNAME"] = "Darwin"
    posix_harness.env["FAKE_UNAME_MACHINE"] = "x86_64"

    result = posix_harness.run_interactive("n\nn\nn\nn\nn\ny\nn\nn\nn\nn\nn\n")

    assert result.returncode != 0
    assert "does not provide a supported release for Darwin x86_64" in result.stdout
    assert not any(
        "hermes-agent.nousresearch.com" in call for call in posix_harness.calls()
    )


@pytest.mark.parametrize(
    ("client", "install_call"),
    [
        ("opencode", "opencode-install"),
        ("cline", "npm:install -g cline"),
        ("hermes", "hermes-install:--non-interactive --skip-setup"),
        ("grok", "grok-install"),
        ("muse", "muse-install"),
        (
            "aider",
            "uv:tool install --force --python python3.12 --with pip aider-chat@latest",
        ),
    ],
)
def test_install_sh_preserves_upstream_managed_harness_without_parsing_version(
    posix_harness: PosixHarness,
    client: str,
    install_call: str,
) -> None:
    _write_executable(
        posix_harness.bin_dir / client,
        _posix_command(client, version_output="opaque upstream version output"),
    )

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    calls = posix_harness.calls()
    assert f"{client}:--version" in calls
    assert install_call not in calls


@pytest.mark.parametrize("failure", ["grok-download", "grok-install"])
def test_install_sh_stops_when_grok_install_fails(
    posix_harness: PosixHarness,
    failure: str,
) -> None:
    result = posix_harness.run_interactive(
        "n\nn\nn\nn\nn\nn\nn\ny\nn\nn\nn\n", fail_step=failure
    )

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    _assert_uv_ready_without_fcc_install(posix_harness.calls())


@pytest.mark.parametrize("failure", ["muse-download", "muse-install"])
def test_install_sh_stops_when_muse_install_fails(
    posix_harness: PosixHarness,
    failure: str,
) -> None:
    result = posix_harness.run_interactive(
        "n\nn\nn\nn\nn\nn\nn\nn\ny\nn\nn\n", fail_step=failure
    )

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    _assert_uv_ready_without_fcc_install(posix_harness.calls())


def test_install_sh_stops_when_aider_install_fails(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run_interactive(
        "n\nn\nn\nn\nn\nn\nn\nn\nn\ny\nn\n", fail_step="aider-install"
    )

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    calls = posix_harness.calls()
    aider_install = (
        "uv:tool install --force --python python3.12 --with pip aider-chat@latest"
    )
    assert calls.index("uv:--version") < calls.index(aider_install)
    assert not any("aider.chat/install" in call for call in calls)
    assert not any("--refresh-package free-claude-code" in call for call in calls)


def test_install_sh_accepts_aider_as_the_only_selected_agent(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run_interactive("n\nn\nn\nn\nn\nn\nn\nn\nn\ny\nn\n")

    assert result.returncode == 0, result.stdout
    calls = posix_harness.calls()
    aider_install = (
        "uv:tool install --force --python python3.12 --with pip aider-chat@latest"
    )
    assert calls.index("uv:--version") < calls.index(aider_install)
    assert calls.index(aider_install) < calls.index("aider:--version")
    assert not any("aider.chat/install" in call for call in calls)
    assert "Run Aider with: fcc-aider" in result.stdout
    assert "Select at least one coding agent." not in result.stdout


def test_install_sh_discovers_aider_in_custom_uv_tool_bin(
    posix_harness: PosixHarness,
) -> None:
    custom_tool_bin = posix_harness.root / "custom-tool-bin"
    posix_harness.env["UV_TOOL_BIN_DIR"] = str(custom_tool_bin)

    result = posix_harness.run_interactive("n\nn\nn\nn\nn\nn\nn\nn\nn\ny\nn\n")

    assert result.returncode == 0, result.stderr
    assert (custom_tool_bin / "aider").is_file()
    calls = posix_harness.calls()
    assert "uv:tool dir --bin" in calls
    assert "aider:--version" in calls


@pytest.mark.parametrize("fail_step", ("", "aider-verify"), ids=("valid", "broken"))
def test_install_sh_checks_existing_aider_in_custom_uv_tool_bin_before_installing(
    posix_harness: PosixHarness,
    fail_step: str,
) -> None:
    custom_tool_bin = posix_harness.root / "custom-tool-bin"
    existing_aider = custom_tool_bin / "aider"
    posix_harness.env["UV_TOOL_BIN_DIR"] = str(custom_tool_bin)
    _write_executable(
        existing_aider,
        _posix_command("aider", version_output="existing aider 1.0.0"),
    )
    original = existing_aider.read_bytes()

    result = posix_harness.run_interactive(
        "n\nn\nn\nn\nn\nn\nn\nn\nn\ny\nn\n", fail_step=fail_step
    )

    if fail_step:
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stderr
    calls = posix_harness.calls()
    assert calls.index("uv:tool dir --bin") < calls.index("aider:--version")
    assert not any("aider-chat@latest" in call for call in calls)
    assert existing_aider.read_bytes() == original


def test_install_sh_rejects_broken_existing_aider_without_replacing_it(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("aider")

    result = posix_harness.run(fail_step="aider-verify")

    assert result.returncode != 0
    calls = posix_harness.calls()
    assert "aider:--version" in calls
    assert not any("aider-chat@latest" in call for call in calls)
    assert not any("aider.chat" in call for call in calls)


def test_install_sh_installs_selected_dsh_at_exact_preview(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run_interactive("n\nn\nn\nn\nn\nn\ny\nn\nn\nn\nn\n")

    assert result.returncode == 0, result.stdout
    calls = posix_harness.calls()
    assert "npm:install -g @deepseek-ai/dsh@0.1.0-rc.8" in calls
    assert calls.index("npm:install -g @deepseek-ai/dsh@0.1.0-rc.8") < calls.index(
        "dsh:--version"
    )
    assert "Run DeepSeek Harness with: fcc-dsh" in result.stdout
    assert not any(call.startswith("rtk:init") for call in calls)


def test_install_sh_replaces_mismatched_dsh_preview(
    posix_harness: PosixHarness,
) -> None:
    _write_executable(
        posix_harness.bin_dir / "dsh",
        _posix_command("dsh").replace("0.1.0-rc.8", "0.1.0-rc.7"),
    )

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "does not match 0.1.0-rc.8" in result.stdout
    assert "npm:install -g @deepseek-ai/dsh@0.1.0-rc.8" in posix_harness.calls()


def test_install_sh_rejects_exact_dsh_on_unsupported_node(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("dsh")
    _write_executable(
        posix_harness.bin_dir / "node",
        _posix_command("node").replace("node 22.19.0", "node 23.9.0"),
    )

    result = posix_harness.run()

    assert result.returncode != 0
    assert "requires Node.js ^22.19.0 or >=24.0.0" in result.stderr
    _assert_uv_ready_without_fcc_install(posix_harness.calls())


@pytest.mark.parametrize("node_version", ["22.18.0", "23.9.0", "not-a-version"])
def test_install_sh_rejects_incompatible_node_for_selected_dsh(
    posix_harness: PosixHarness,
    node_version: str,
) -> None:
    _write_executable(
        posix_harness.bin_dir / "node",
        _posix_command("node").replace("node 22.19.0", f"node {node_version}"),
    )

    result = posix_harness.run_interactive("n\nn\nn\nn\nn\nn\ny\nn\nn\nn\nn\n")

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    _assert_uv_ready_without_fcc_install(posix_harness.calls())


def test_install_sh_noninteractive_skips_dsh_without_node(
    posix_harness: PosixHarness,
) -> None:
    (posix_harness.bin_dir / "node").unlink()
    (posix_harness.bin_dir / "npm").unlink()

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert (
        "fcc-dsh wrapper is ready after you install DeepSeek Harness" in result.stdout
    )
    assert not any("@deepseek-ai/dsh" in call for call in posix_harness.calls())


def test_install_sh_stops_when_selected_dsh_install_fails(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run_interactive(
        "n\nn\nn\nn\nn\nn\ny\nn\nn\nn\nn\n", fail_step="dsh-install"
    )

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    _assert_uv_ready_without_fcc_install(posix_harness.calls())


def test_install_sh_installs_and_configures_rtk_for_selected_agents(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run("--rtk")

    assert result.returncode == 0, result.stderr
    calls = posix_harness.calls()
    assert (
        "download:https://github.com/rtk-ai/rtk/releases/download/v0.44.2/rtk-x86_64-unknown-linux-musl.tar.gz"
        in calls
    )
    assert any(call.startswith("sha256sum:") for call in calls)
    assert calls.index("rtk:--version:telemetry=1") > next(
        index for index, call in enumerate(calls) if call.startswith("sha256sum:")
    )
    assert calls.index("rtk:--version:telemetry=1") < calls.index(
        "rtk:gain:telemetry=1"
    )
    assert [call for call in calls if call.startswith("rtk:init")] == [
        "rtk:init --global --auto-patch:telemetry=1",
        "rtk:init --global --codex:telemetry=1",
        "rtk:init --global --agent pi:telemetry=1",
        "rtk:init --global --opencode:telemetry=1",
    ]
    assert calls.index("uv:--version") < calls.index(
        "rtk:init --global --opencode:telemetry=1"
    )
    assert (Path(posix_harness.env["HOME"]) / ".claude").is_dir()


def test_install_sh_prepares_custom_claude_config_directory_for_rtk(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_rtk()
    custom_config = posix_harness.root / "custom-claude"
    posix_harness.env["CLAUDE_CONFIG_DIR"] = str(custom_config)

    result = posix_harness.run("--rtk")

    assert result.returncode == 0, result.stderr
    assert custom_config.is_dir()
    assert not (Path(posix_harness.env["HOME"]) / ".claude").exists()


@pytest.mark.parametrize(
    ("system", "machine", "asset"),
    [
        ("Linux", "x86_64", "rtk-x86_64-unknown-linux-musl.tar.gz"),
        ("Linux", "aarch64", "rtk-aarch64-unknown-linux-gnu.tar.gz"),
        ("Darwin", "x86_64", "rtk-x86_64-apple-darwin.tar.gz"),
        ("Darwin", "arm64", "rtk-aarch64-apple-darwin.tar.gz"),
    ],
)
def test_install_sh_selects_pinned_rtk_release_for_platform(
    posix_harness: PosixHarness,
    system: str,
    machine: str,
    asset: str,
) -> None:
    posix_harness.env["FAKE_UNAME"] = system
    posix_harness.env["FAKE_UNAME_MACHINE"] = machine

    result = posix_harness.run("--rtk", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert f"rtk-ai/rtk/releases/download/v0.44.2/{asset}" in result.stdout
    assert "raw.githubusercontent.com/rtk-ai/rtk" not in result.stdout


def test_install_sh_rejects_unsupported_rtk_platform(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("muse")
    posix_harness.env["FAKE_UNAME"] = "FreeBSD"
    posix_harness.env["FAKE_UNAME_MACHINE"] = "riscv64"

    result = posix_harness.run("--rtk", "--dry-run")

    assert result.returncode != 0
    assert "does not provide a release for FreeBSD riscv64" in result.stderr


def test_install_sh_preserves_existing_rtk_and_configures_only_selected_agent(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_rtk()

    result = posix_harness.run_interactive("n\ny\nn\nn\nn\nn\nn\nn\nn\nn\ny\n")

    assert result.returncode == 0, result.stdout
    assert "verifying it without updating it" in result.stdout
    assert not any("rtk-ai/rtk" in call for call in posix_harness.calls())
    assert [call for call in posix_harness.calls() if call.startswith("rtk:init")] == [
        "rtk:init --global --codex:telemetry=1"
    ]


def test_install_sh_rejects_conflicting_rtk_command(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_unrelated_rtk()

    result = posix_harness.run("--rtk")

    assert result.returncode != 0
    assert "not a compatible Rust Token Killer installation" in result.stderr
    assert not any("rtk-ai/rtk" in call for call in posix_harness.calls())
    _assert_uv_ready_without_fcc_install(posix_harness.calls())


@pytest.mark.parametrize(
    "failure",
    [
        "rtk-download",
        "rtk-checksum",
        "rtk-install",
        "rtk-verify",
        "rtk-init-claude",
    ],
)
def test_install_sh_stops_when_rtk_setup_fails(
    posix_harness: PosixHarness,
    failure: str,
) -> None:
    result = posix_harness.run("--rtk", fail_step=failure)

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    _assert_uv_ready_without_fcc_install(posix_harness.calls())


def test_install_sh_reprompts_then_installs_only_selected_agent(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run_interactive("n\n" * 11 + "y\n" + "n\n" * 9)

    assert result.returncode == 0, result.stdout
    assert "Select at least one coding agent." in result.stdout
    assert "Run Codex with: fcc-codex" in result.stdout
    assert "Run Claude Code with: fcc-claude" not in result.stdout
    assert "Run Pi with: fcc-pi" not in result.stdout
    assert "Run OpenCode with: fcc-opencode" not in result.stdout
    assert "Run Cline with: fcc-cline" not in result.stdout
    calls = posix_harness.calls()
    assert "codex-install:1" in calls
    assert not any("claude.ai" in call for call in calls)
    assert not any("pi.dev" in call for call in calls)
    assert not any("rtk-ai/rtk" in call for call in calls)


def test_install_sh_rejects_uninstalled_only_selection(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run_interactive(
        "n\nn\ny\nn\nn\nn\nn\nn\nn\nn\nn\n", fail_step="pi-skip"
    )

    assert result.returncode != 0
    assert "No selected coding agent was installed." in result.stdout
    _assert_uv_ready_without_fcc_install(posix_harness.calls())


def test_install_sh_creates_native_macos_app_and_desktop_link(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.env["FAKE_UNAME"] = "Darwin"
    tool_bin = posix_harness.root / "tool's bin"
    posix_harness.env["FAKE_TOOL_BIN"] = str(tool_bin)

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    app = posix_harness.root / "home" / "Applications" / "Free Claude Code.app"
    plist = app / "Contents" / "Info.plist"
    owner_file = app / "Contents" / ".free-claude-code-owner"
    launcher = app / "Contents" / "MacOS" / "fcc-desktop"
    icon = app / "Contents" / "Resources" / "AppIcon.icns"
    desktop_link = posix_harness.root / "home" / "Desktop" / "Free Claude Code.app"
    assert owner_file.read_text(encoding="utf-8").strip() == (
        "io.github.alishahryar1.free-claude-code"
    )
    plist_text = plist.read_text(encoding="utf-8")
    assert "<key>CFBundleIconFile</key>" in plist_text
    assert "<string>AppIcon</string>" in plist_text
    assert "<key>LSUIElement</key>" in plist_text
    assert "<key>LSMultipleInstancesProhibited</key>" in plist_text
    assert icon.read_bytes() == b"fake icon\n"
    assert launcher.stat().st_mode & 0o111
    expected_command = str(tool_bin / "fcc-desktop").replace("'", "'\\''")
    assert f"exec '{expected_command}'" in launcher.read_text(encoding="utf-8")
    assert desktop_link.is_symlink()
    assert desktop_link.readlink() == app
    assert any(
        call == f"fcc-desktop:--export-icon {icon}" for call in posix_harness.calls()
    )


def test_install_sh_stops_if_macos_icon_export_fails(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.env["FAKE_UNAME"] = "Darwin"

    result = posix_harness.run(fail_step="desktop-icon-export")

    assert result.returncode != 0
    assert "Command failed with exit code 37" in result.stderr
    assert not (
        posix_harness.root / "home" / "Desktop" / "Free Claude Code.app"
    ).exists()


def test_install_sh_rejects_unowned_macos_app_bundle(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.env["FAKE_UNAME"] = "Darwin"
    app = posix_harness.root / "home" / "Applications" / "Free Claude Code.app"
    contents = app / "Contents"
    contents.mkdir(parents=True)
    plist = contents / "Info.plist"
    plist.write_text("foreign app", encoding="utf-8")

    result = posix_harness.run()

    assert result.returncode != 0
    assert "not managed by Free Claude Code" in result.stderr
    assert plist.read_text(encoding="utf-8") == "foreign app"
    assert not (contents / ".free-claude-code-owner").exists()


def test_install_sh_preserves_unrelated_macos_desktop_link(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.env["FAKE_UNAME"] = "Darwin"
    desktop = posix_harness.root / "home" / "Desktop"
    desktop.mkdir()
    unrelated = posix_harness.root / "Unrelated.app"
    unrelated.mkdir()
    desktop_link = desktop / "Free Claude Code.app"
    desktop_link.symlink_to(unrelated, target_is_directory=True)

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "non-FCC link" in result.stdout
    assert desktop_link.readlink() == unrelated


@pytest.mark.parametrize("uv_version", ("0.12.13", "0.12.13+build.1", "0.12.14"))
def test_install_sh_preserves_valid_existing_tools(
    posix_harness: PosixHarness,
    uv_version: str,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_client("pi")
    posix_harness.add_client("cline")
    posix_harness.add_client("hermes")
    posix_harness.add_client("grok")
    posix_harness.add_client("muse")
    posix_harness.add_client("aider")
    posix_harness.add_uv(uv_version)

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert not any(call.startswith("download:") for call in posix_harness.calls())
    assert "leaving it unchanged" in result.stdout


def test_install_sh_replaces_unrelated_pi_command(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_unrelated_pi()
    posix_harness.add_uv("0.12.13")

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "is not Pi Coding Agent; installing Pi" in result.stdout
    assert "pi-install" in posix_harness.calls()


def test_install_sh_discovers_custom_pi_npm_prefix(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_npm_prefix(posix_harness.root / "custom-npm")
    posix_harness.add_uv("0.12.13")

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    calls = posix_harness.calls()
    assert "npm:prefix -g" in calls
    assert "pi:--help" in calls
    assert "pi:--version" in calls


def test_install_sh_continues_when_pi_is_not_installed(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run(fail_step="pi-skip")

    assert result.returncode == 0, result.stderr
    assert "Pi was not installed; continuing without it." in result.stdout
    assert "Run Pi with: fcc-pi" not in result.stdout
    calls = posix_harness.calls()
    assert "pi-install" in calls
    assert not any(call.startswith("pi:") for call in calls)
    assert "uv-install" in calls
    assert "fcc-server:--version" in calls


def test_install_sh_continues_when_unrelated_pi_is_unchanged(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_unrelated_pi()

    result = posix_harness.run(fail_step="pi-skip")

    assert result.returncode == 0, result.stderr
    assert "Pi was not installed; continuing without it." in result.stdout
    assert "Run Pi with: fcc-pi" not in result.stdout
    calls = posix_harness.calls()
    assert "unrelated-pi:--help" in calls
    assert "unrelated-pi:--version" not in calls
    assert "fcc-server:--version" in calls


def test_install_sh_continues_when_pi_resolution_changes_to_unrelated_command(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_unrelated_pi()
    npm_prefix = posix_harness.root / "custom-npm"
    posix_harness.add_npm_prefix(npm_prefix)
    _write_executable(
        npm_prefix / "bin" / "pi",
        _posix_command("other-unrelated-pi"),
    )

    result = posix_harness.run(fail_step="pi-skip")

    assert result.returncode == 0, result.stderr
    assert "Pi was not installed; continuing without it." in result.stdout
    assert "Run Pi with: fcc-pi" not in result.stdout
    calls = posix_harness.calls()
    assert "other-unrelated-pi:--help" in calls
    assert "other-unrelated-pi:--version" not in calls
    assert "fcc-server:--version" in calls


@pytest.mark.parametrize("uv_version", ("0.5.9", "0.11.16", "0.12.9", "0.12.12"))
def test_install_sh_replaces_obsolete_uv(
    posix_harness: PosixHarness,
    uv_version: str,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_client("pi")
    posix_harness.add_uv(uv_version)

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "uv-install" in posix_harness.calls()
    assert f"uv {uv_version} does not satisfy stable >=0.12.13" in result.stdout


def test_install_sh_prioritizes_replacement_uv_over_obsolete_cargo_uv(
    posix_harness: PosixHarness,
) -> None:
    home = Path(posix_harness.env["HOME"])
    cargo_bin = home / ".cargo" / "bin"
    local_bin = home / ".local" / "bin"
    _write_executable(cargo_bin / "uv", _posix_uv_command("0.5.9"))
    local_bin.mkdir(parents=True)
    posix_harness.env["PATH"] = f"{cargo_bin}:{local_bin}:{posix_harness.env['PATH']}"

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "Verified uv 0.12.13." in result.stdout
    assert "uv-install" in posix_harness.calls()


@pytest.mark.parametrize("install_variable", ("UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL"))
def test_install_sh_prioritizes_forced_cargo_home_uv_install_layout(
    posix_harness: PosixHarness,
    install_variable: str,
) -> None:
    home = Path(posix_harness.env["HOME"])
    cargo_home = home / ".cargo"
    cargo_bin = cargo_home / "bin"
    posix_harness.env["CARGO_HOME"] = str(cargo_home)
    posix_harness.env[install_variable] = str(cargo_home)
    posix_harness.env["PATH"] = f"{posix_harness.env['PATH']}:{cargo_bin}"
    posix_harness.add_uv("0.5.9")

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "Verified uv 0.12.13." in result.stdout
    assert "uv-install" in posix_harness.calls()


def test_install_sh_uv_install_dir_takes_precedence_over_unmanaged_install(
    posix_harness: PosixHarness,
) -> None:
    install_bin = posix_harness.root / "uv-install-bin"
    unmanaged_bin = posix_harness.root / "unmanaged-uv-bin"
    posix_harness.env["UV_INSTALL_DIR"] = str(install_bin)
    posix_harness.env["UV_UNMANAGED_INSTALL"] = str(unmanaged_bin)
    posix_harness.add_uv("0.5.9")

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert (install_bin / "uv").is_file()
    assert not (unmanaged_bin / "uv").exists()


def test_install_sh_prioritizes_replacement_uv_from_unmanaged_install_directory(
    posix_harness: PosixHarness,
) -> None:
    unmanaged_bin = posix_harness.root / "unmanaged-uv-bin"
    posix_harness.env["UV_UNMANAGED_INSTALL"] = str(unmanaged_bin)
    posix_harness.add_uv("0.5.9")

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert "Verified uv 0.12.13." in result.stdout
    assert "uv-install" in posix_harness.calls()


@pytest.mark.parametrize("cargo_layout", (False, True), ids=("default", "cargo-home"))
def test_install_sh_infers_home_for_replacement_uv(
    posix_harness: PosixHarness,
    cargo_layout: bool,
) -> None:
    inferred_home = Path(posix_harness.env["FAKE_INFERRED_HOME"])
    posix_harness.env.pop("HOME")
    if cargo_layout:
        posix_harness.env["UV_INSTALL_DIR"] = str(inferred_home / ".cargo")

    result = posix_harness.run_interactive("n\nn\nn\nn\nn\nn\nn\nn\nn\ny\nn\n")

    assert result.returncode == 0, result.stderr
    relative_uv = Path(".cargo/bin/uv") if cargo_layout else Path(".local/bin/uv")
    assert (inferred_home / relative_uv).is_file()
    assert "Verified uv 0.12.13." in result.stdout


@pytest.mark.parametrize("version", ("0.12.13-alpha.1", "0.13.0-rc.1"))
def test_install_sh_replaces_prerelease_uv(
    posix_harness: PosixHarness,
    version: str,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_client("pi")
    posix_harness.add_uv(version)

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr
    assert f"uv {version} does not satisfy stable >=0.12.13" in result.stdout
    assert "uv-install" in posix_harness.calls()


@pytest.mark.parametrize(
    "failure",
    [
        "claude-download",
        "claude-install",
        "claude-verify",
        "codex-download",
        "codex-install",
        "codex-verify",
        "pi-download",
        "pi-install",
        "pi-verify",
        "opencode-download",
        "opencode-install",
        "opencode-verify",
        "cline-install",
        "cline-verify",
        "uv-download",
        "uv-install",
        "uv-verify",
        "fcc-install",
        "path-update",
        "fcc-missing",
        "fcc-verify",
    ],
)
def test_install_sh_stops_without_success_on_each_failure(
    posix_harness: PosixHarness,
    failure: str,
) -> None:
    if failure.startswith("opencode-"):
        (posix_harness.bin_dir / "opencode").unlink()
    result = posix_harness.run(fail_step=failure)

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    calls = posix_harness.calls()
    if failure == "path-update":
        failure_index = calls.index("uv:tool update-shell")
        assert "uv:tool dir --bin" not in calls[failure_index + 1 :]

    forbidden = {
        "claude-download": "claude-install",
        "claude-install": "claude:--version",
        "claude-verify": "chatgpt.com",
        "codex-download": "codex-install",
        "codex-install": "codex:--version",
        "codex-verify": "pi.dev",
        "pi-download": "pi-install",
        "pi-install": "pi:--version",
        "pi-verify": "opencode:--version",
        "opencode-download": "opencode-install",
        "opencode-install": "opencode:--version",
        "opencode-verify": "npm:install -g cline",
        "cline-install": "cline:--version",
        "cline-verify": "hermes-agent.nousresearch.com",
        "uv-download": "uv-install",
        "uv-install": "uv:--version",
        "uv-verify": "uv:tool install",
        "fcc-install": "uv:tool update-shell",
        "fcc-missing": "fcc-server:--version",
    }.get(failure)
    if forbidden is not None:
        assert not any(forbidden in call for call in calls)


def test_install_sh_dry_run_never_executes_commands(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run("--dry-run")

    assert result.returncode == 0, result.stderr
    assert posix_harness.calls() == []
    assert "Dry run complete. No changes were made." in result.stdout
    assert "Free Claude Code is installed and verified." not in result.stdout


def test_install_sh_rejects_broken_existing_client_without_replacing_it(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")

    result = posix_harness.run(fail_step="claude-verify")

    assert result.returncode != 0
    calls = posix_harness.calls()
    _assert_uv_ready_without_fcc_install(calls)
    assert not any("claude.ai" in call for call in calls)


def test_install_sh_rejects_unparseable_existing_uv(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_client("pi")
    posix_harness.add_uv("not-a-version")

    result = posix_harness.run()

    assert result.returncode != 0
    assert not any("astral.sh" in call for call in posix_harness.calls())


def test_install_sh_voice_flags_only_change_fcc_spec(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run("--voice-all", "--torch-backend", "cu130")

    assert result.returncode == 0, result.stderr
    assert any(
        "--torch-backend cu130 free-claude-code[voice,voice_local] @ "
        "https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
        in call
        for call in posix_harness.calls()
    )


def test_install_sh_rejects_invalid_options_before_mutation(
    posix_harness: PosixHarness,
) -> None:
    result = posix_harness.run("--torch-backend", "cu130")

    assert result.returncode != 0
    assert posix_harness.calls() == []


@pytest.mark.parametrize("command_name", FCC_COMMANDS)
def test_install_sh_rejects_running_fcc_before_mutation(
    posix_harness: PosixHarness,
    command_name: str,
) -> None:
    posix_harness.env["FCC_RUNNING_COMMAND"] = command_name

    result = posix_harness.run()

    assert result.returncode != 0
    assert posix_harness.calls() == []
    assert f"{command_name} (PID 4242)" in result.stderr


def test_install_sh_rechecks_for_fcc_process_before_tool_replacement(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.add_client("claude")
    posix_harness.add_client("codex")
    posix_harness.add_client("pi")
    posix_harness.add_uv("0.12.13")
    posix_harness.env["FCC_RUNNING_COMMAND"] = "fcc-server"
    posix_harness.env["FCC_RUNNING_PHASE"] = "late"

    result = posix_harness.run()

    assert result.returncode != 0
    assert "fcc-server (PID 4242)" in result.stderr
    assert not any(
        "--refresh-package free-claude-code" in call for call in posix_harness.calls()
    )


def test_install_sh_ignores_similarly_named_process(
    posix_harness: PosixHarness,
) -> None:
    posix_harness.env["FCC_RUNNING_COMMAND"] = "fcc-server-helper"

    result = posix_harness.run()

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("command_name", "process_args"),
    (
        ("free-claude-code", "/home/user/.local/bin/free-claude-code"),
        ("fcc-server", "/usr/bin/python3 /home/user/.local/bin/fcc-server"),
    ),
)
def test_install_sh_process_fallback_reads_full_command_line(
    posix_harness: PosixHarness,
    command_name: str,
    process_args: str,
) -> None:
    posix_harness.use_process_list_fallback(f"4242 {process_args}")

    result = posix_harness.run()

    assert result.returncode != 0
    assert posix_harness.calls() == []
    assert f"{command_name} (PID 4242)" in result.stderr


def _powershells() -> tuple[str, ...]:
    candidates = (shutil.which("pwsh"), shutil.which("powershell"))
    return tuple(dict.fromkeys(path for path in candidates if path is not None))


def test_install_ps1_waits_for_gui_icon_export() -> None:
    installer = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")
    body = _braced_body(installer, "function Export-FccDesktopIcon")

    assert "Start-Process" in body
    assert "-WindowStyle Hidden" in body
    assert "-Wait" in body
    assert "-PassThru" in body
    assert "$process.ExitCode" in body


@pytest.mark.parametrize(
    "powershell",
    _powershells() or (None,),
    ids=lambda path: Path(path).name if path is not None else "unavailable",
)
def test_install_ps1_gui_icon_export_completes_before_returning(
    powershell: str | None,
    tmp_path: Path,
) -> None:
    if powershell is None or os.name != "nt":
        pytest.skip("PowerShell GUI process behavior runs on Windows hosts")

    installer = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")
    function_declarations = (
        "function Format-Argument",
        "function Format-Command",
        "function Export-FccDesktopIcon",
    )
    functions = "\n".join(
        f"{declaration} {{{_braced_body(installer, declaration)}}}"
        for declaration in function_declarations
    )
    installed_desktop_command = Path(sys.executable).with_name("fcc-desktop.exe")
    desktop_command = tmp_path / "icon-exporter.exe"
    shutil.copy2(installed_desktop_command, desktop_command)
    destination = tmp_path / "profile with spaces" / ".fcc" / "app-icon.ico"
    env = os.environ | {
        "FCC_TEST_DESKTOP_COMMAND": str(desktop_command),
        "FCC_TEST_ICON_PATH": str(destination),
    }
    script = "\n".join(
        (
            '$ErrorActionPreference = "Stop"',
            "$DryRun = $false",
            functions,
            (
                "Export-FccDesktopIcon "
                "-DesktopCommand $env:FCC_TEST_DESKTOP_COMMAND "
                "-IconPath $env:FCC_TEST_ICON_PATH"
            ),
        )
    )

    completed = subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        destination.read_bytes()
        == (
            _repo_root() / "src" / "free_claude_code" / "assets" / "app-icon.ico"
        ).read_bytes()
    )


def _create_windows_shortcut(
    powershell: str,
    shortcut_path: Path,
    target_path: Path,
) -> None:
    shortcut_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {
        "FCC_TEST_SHORTCUT": str(shortcut_path),
        "FCC_TEST_TARGET": str(target_path),
    }
    subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-Command",
            (
                "$shell = New-Object -ComObject WScript.Shell; "
                "$shortcut = $shell.CreateShortcut($env:FCC_TEST_SHORTCUT); "
                "$shortcut.TargetPath = $env:FCC_TEST_TARGET; "
                "$shortcut.Save()"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _windows_shortcut_icon(
    powershell: str,
    shortcut_path: Path,
    env: dict[str, str],
) -> str:
    process_env = env | {"FCC_TEST_SHORTCUT": str(shortcut_path)}
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-Command",
            (
                "$shell = New-Object -ComObject WScript.Shell; "
                "$shortcut = $shell.CreateShortcut($env:FCC_TEST_SHORTCUT); "
                "[Console]::Out.Write($shortcut.IconLocation)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=process_env,
    )
    return completed.stdout


def _batch_client(name: str, *, version_output: str | None = None) -> str:
    version = {
        "opencode": "1.18.18",
        "cline": "3.0.55",
        "hermes": "0.20.4",
        "dsh": "0.1.0-rc.8",
        "grok": "1.0.5",
        "muse": "0.2.1",
        "node": "22.19.0",
    }.get(name, "1.0.0")
    if version_output is not None:
        version_command = f'if "%1"=="--version" echo {version_output}'
    elif name == "hermes":
        version_command = (
            f'if "%1"=="--version" echo Hermes Agent v{version} '
            "(2026.8.18) · upstream deadbeef\n"
            'if "%1"=="--version" echo Install directory: C:\\fake-hermes\n'
            'if "%1"=="--version" echo Install method: git\n'
            'if "%1"=="--version" echo Python: 3.12.11\n'
            'if "%1"=="--version" echo OpenAI SDK: 2.15.0\n'
            'if "%1"=="--version" echo Up to date'
        )
    elif name == "grok":
        version_command = (
            f'if "%1"=="--version" echo grok {version} (5115b46bc9) [stable]'
        )
    elif name == "muse":
        version_command = (
            f'if "%1"=="--version" echo Muse Code {version} ({version}-R1215.1)'
        )
    else:
        version_command = f'if "%1"=="--version" echo {name} {version}'
    help_output = (
        "echo   --extension, -e ^<path^>  Load an extension\n"
        "echo   --models ^<patterns^>     Scope models"
        if name == "pi"
        else "rem no product help"
    )
    return f"""@echo off
echo {name}:%*>>"%CALL_LOG%"
if "%FAIL_STEP%"=="{name}-verify" exit /b 51
{version_command}
if "%1"=="--help" (
{help_output}
)
exit /b 0
"""


def _batch_npm() -> str:
    return r"""@echo off
echo npm:%*>>"%CALL_LOG%"
if "%1"=="install" if "%2"=="-g" if "%3"=="cline" goto install_cline
if "%1"=="install" if "%2"=="-g" if "%3"=="@deepseek-ai/dsh@0.1.0-rc.8" goto install_dsh
if "%1"=="prefix" if "%2"=="-g" echo %FAKE_NPM_PREFIX%& exit /b 0
if "%1"=="config" if "%2"=="get" if "%3"=="prefix" echo %FAKE_NPM_PREFIX%& exit /b 0
exit /b 71
:install_cline
if "%FAIL_STEP%"=="cline-install" exit /b 72
if not exist "%FAKE_NPM_PREFIX%" mkdir "%FAKE_NPM_PREFIX%"
copy /y "%FAKE_FIXTURES%\cline-command.cmd" "%FAKE_NPM_PREFIX%\cline.cmd" >nul
exit /b 0
:install_dsh
if "%FAIL_STEP%"=="dsh-install" exit /b 73
if not exist "%FAKE_NPM_PREFIX%" mkdir "%FAKE_NPM_PREFIX%"
copy /y "%FAKE_FIXTURES%\dsh-command.cmd" "%FAKE_NPM_PREFIX%\dsh.cmd" >nul
exit /b 0
"""


def _batch_uv(version: str) -> str:
    return rf"""@echo off
set "UV_BIN_DIR=%FAKE_TOOL_BIN%"
if defined UV_TOOL_BIN_DIR set "UV_BIN_DIR=%UV_TOOL_BIN_DIR%"
echo uv:%*>>"%CALL_LOG%"
if "%1"=="--version" goto version
if "%1"=="tool" if "%2"=="install" goto install
if "%1"=="tool" if "%2"=="update-shell" goto update_shell
if "%1"=="tool" if "%2"=="dir" if "%3"=="--bin" goto tool_bin
exit /b 59
:version
if "%FCC_RUNNING_PHASE%"=="late" type nul > "%FCC_PROCESS_MARKER%"
if "%FAIL_STEP%"=="uv-verify" exit /b 52
echo uv {version}
exit /b 0
:install
if "%5"=="python3.12" goto install_aider
if "%FAIL_STEP%"=="fcc-install" exit /b 53
if not exist "%UV_BIN_DIR%" mkdir "%UV_BIN_DIR%"
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-server.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-desktop.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-claude.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-pi.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-opencode.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-cline.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-hermes.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-dsh.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-grok.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-muse.cmd" >nul
copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-aider.cmd" >nul
if not "%FAIL_STEP%"=="fcc-missing" copy /y "%FAKE_FIXTURES%\fcc-command.cmd" "%UV_BIN_DIR%\fcc-codex.cmd" >nul
exit /b 0
:install_aider
if "%FAIL_STEP%"=="aider-install" exit /b 57
if not exist "%UV_BIN_DIR%" mkdir "%UV_BIN_DIR%"
copy /y "%FAKE_FIXTURES%\aider-command.cmd" "%UV_BIN_DIR%\aider.cmd" >nul
exit /b 0
:update_shell
if "%FAIL_STEP%"=="path-update" exit /b 54
exit /b 0
:tool_bin
echo %UV_BIN_DIR%
exit /b 0
"""


def _batch_rtk() -> str:
    return r"""@echo off
>>"%CALL_LOG%" echo rtk:%*:telemetry=%RTK_TELEMETRY_DISABLED%
if "%*"=="init --global --auto-patch" if defined CLAUDE_CONFIG_DIR if not exist "%CLAUDE_CONFIG_DIR%" exit /b 77
if "%*"=="init --global --auto-patch" if not defined CLAUDE_CONFIG_DIR if not exist "%USERPROFILE%\.claude" exit /b 77
if "%FAIL_STEP%"=="rtk-verify" if "%1"=="--version" exit /b 72
if "%FAIL_STEP%"=="rtk-verify" if "%1"=="gain" exit /b 72
if "%FAIL_STEP%"=="rtk-init-claude" if "%*"=="init --global --auto-patch" exit /b 73
if "%FAIL_STEP%"=="rtk-init-codex" if "%*"=="init --global --codex" exit /b 74
if "%FAIL_STEP%"=="rtk-init-pi" if "%*"=="init --global --agent pi" exit /b 75
if "%FAIL_STEP%"=="rtk-init-opencode" if "%*"=="init --global --opencode" exit /b 76
if "%1"=="--version" echo rtk 0.44.2
exit /b 0
"""


@dataclass
class PowerShellHarness:
    root: Path
    bin_dir: Path
    fixtures: Path
    tool_bin: Path
    log: Path
    env: dict[str, str]
    powershell: str
    wrapper: Path

    def add_client(self, name: str) -> None:
        _write_executable(self.bin_dir / f"{name}.cmd", _batch_client(name))

    def add_unrelated_pi(self) -> None:
        _write_executable(self.bin_dir / "pi.cmd", _batch_client("unrelated-pi"))

    def add_npm_prefix(self, prefix: Path) -> None:
        prefix.mkdir(parents=True)
        self.env["FAKE_NPM_PREFIX"] = str(prefix)
        _write_executable(self.bin_dir / "npm.cmd", _batch_npm())

    def add_uv(self, version: str) -> None:
        _write_executable(self.bin_dir / "uv.cmd", _batch_uv(version))

    def add_rtk(self) -> None:
        _write_executable(self.bin_dir / "rtk.cmd", _batch_rtk())

    def set_user_path(self, *entries: Path | str) -> None:
        Path(self.env["FAKE_USER_PATH_FILE"]).write_text(
            os.pathsep.join(str(entry) for entry in entries),
            encoding="utf-8",
        )

    def user_path_entries(self) -> list[str]:
        value = Path(self.env["FAKE_USER_PATH_FILE"]).read_text(encoding="utf-8")
        return value.split(os.pathsep) if value else []

    def add_unrelated_rtk(self) -> None:
        _write_executable(
            self.bin_dir / "rtk.cmd",
            r"""@echo off
echo unrelated-rtk:%*>>"%CALL_LOG%"
if "%1"=="--version" (
    echo rtk 1.0.0
    exit /b 0
)
exit /b 76
""",
        )

    def run(self, *args: str, fail_step: str = "") -> subprocess.CompletedProcess[str]:
        env = self.env | {"FAIL_STEP": fail_step}
        return subprocess.run(
            [
                self.powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.wrapper),
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return self.log.read_text(encoding="utf-8").splitlines()


@pytest.fixture(
    params=_powershells() or (None,),
    ids=lambda path: Path(path).name if path is not None else "unavailable",
)
def powershell_harness(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> PowerShellHarness:
    powershell = request.param
    if powershell is None or os.name != "nt":
        pytest.skip("PowerShell installer scenarios run on Windows hosts")

    bin_dir = tmp_path / "bin"
    fixtures = tmp_path / "fixtures"
    tool_bin = tmp_path / "tool-bin"
    home = tmp_path / "home"
    local_app_data = tmp_path / "local-app-data"
    app_data = tmp_path / "app-data"
    log = tmp_path / "calls.log"
    for path in (bin_dir, fixtures, tool_bin, home, local_app_data, app_data):
        path.mkdir(parents=True)

    (fixtures / "claude-command.cmd").write_text(
        _batch_client("claude"), encoding="utf-8"
    )
    (fixtures / "codex-command.cmd").write_text(
        _batch_client("codex"), encoding="utf-8"
    )
    (fixtures / "pi-command.cmd").write_text(_batch_client("pi"), encoding="utf-8")
    (fixtures / "opencode-command.cmd").write_text(
        _batch_client("opencode"), encoding="utf-8"
    )
    (fixtures / "cline-command.cmd").write_text(
        _batch_client("cline"), encoding="utf-8"
    )
    (fixtures / "hermes-command.cmd").write_text(
        _batch_client("hermes"), encoding="utf-8"
    )
    (fixtures / "dsh-command.cmd").write_text(_batch_client("dsh"), encoding="utf-8")
    (fixtures / "grok-command.cmd").write_text(_batch_client("grok"), encoding="utf-8")
    (fixtures / "muse-command.cmd").write_text(_batch_client("muse"), encoding="utf-8")
    (fixtures / "aider-command.cmd").write_text(
        _batch_client("aider"), encoding="utf-8"
    )
    (fixtures / "rtk-command.cmd").write_text(_batch_rtk(), encoding="utf-8")
    (fixtures / "uv-command.cmd").write_text(_batch_uv("0.12.13"), encoding="utf-8")
    (fixtures / "fcc-command.cmd").write_text(
        """@echo off
for %%I in ("%~f0") do set "FCC_NAME=%%~nI"
echo %FCC_NAME%:%*>>"%CALL_LOG%"
if "%FAIL_STEP%"=="fcc-verify" exit /b 55
if "%FCC_NAME%"=="fcc-desktop" if "%1"=="--export-icon" if "%FAIL_STEP%"=="desktop-icon-export" exit /b 56
if "%FCC_NAME%"=="fcc-desktop" if "%1"=="--export-icon" (
    if not exist "%~dp2" mkdir "%~dp2"
    echo fake icon>"%~2"
)
if "%FCC_NAME%"=="fcc-server" if "%1"=="--version" echo free-claude-code 3.5.18
exit /b 0
""",
        encoding="utf-8",
    )
    (fixtures / "claude-installer.ps1").write_text(
        r"""if ($env:FAIL_STEP -eq "claude-install") { exit 61 }
$bin = Join-Path $env:USERPROFILE ".local\bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "claude-command.cmd") (Join-Path $bin "claude.cmd") -Force
Add-Content -LiteralPath $env:CALL_LOG -Value "claude-install"
""",
        encoding="utf-8",
    )
    (fixtures / "codex-installer.ps1").write_text(
        r"""if ($env:FAIL_STEP -eq "codex-install") { exit 62 }
$bin = Join-Path $env:LOCALAPPDATA "Programs\OpenAI\Codex\bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "codex-command.cmd") (Join-Path $bin "codex.cmd") -Force
Add-Content -LiteralPath $env:CALL_LOG -Value "codex-install:$env:CODEX_NON_INTERACTIVE"
""",
        encoding="utf-8",
    )
    (fixtures / "pi-installer.ps1").write_text(
        r"""if ($env:FAIL_STEP -eq "pi-install") { exit 64 }
if ($env:FAIL_STEP -eq "pi-skip") {
    Add-Content -LiteralPath $env:CALL_LOG -Value "pi-install"
    return
}
$bin = if ($env:FAKE_NPM_PREFIX) { $env:FAKE_NPM_PREFIX } else { Join-Path $env:APPDATA "npm" }
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "pi-command.cmd") (Join-Path $bin "pi.cmd") -Force
Add-Content -LiteralPath $env:CALL_LOG -Value "pi-install"
""",
        encoding="utf-8",
    )
    (fixtures / "hermes-installer.ps1").write_text(
        r"""param(
    [switch] $NonInteractive,
    [switch] $SkipSetup
)
if ($env:FAIL_STEP -eq "hermes-install") { exit 65 }
$bin = Join-Path $env:LOCALAPPDATA "hermes\hermes-agent\bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "hermes-command.cmd") (Join-Path $bin "hermes.cmd") -Force
Add-Content -LiteralPath $env:CALL_LOG -Value "hermes-install:${NonInteractive}:${SkipSetup}"
""",
        encoding="utf-8",
    )
    (fixtures / "grok-installer.ps1").write_text(
        r"""if ($env:FAIL_STEP -eq "grok-install") { exit 66 }
$bin = if ($env:GROK_BIN_DIR) { $env:GROK_BIN_DIR } else { Join-Path $env:USERPROFILE ".grok\bin" }
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "grok-command.cmd") (Join-Path $bin "grok.cmd") -Force
Add-Content -LiteralPath $env:CALL_LOG -Value "grok-install"
""",
        encoding="utf-8",
    )
    (fixtures / "muse-installer.ps1").write_text(
        r"""if ($env:FAIL_STEP -eq "muse-install") { exit 68 }
$bin = if ([string]::IsNullOrWhiteSpace($env:MUSE_INSTALL_DIR)) {
    Join-Path $env:LOCALAPPDATA "Programs\muse"
}
else {
    $env:MUSE_INSTALL_DIR
}
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "muse-command.cmd") (Join-Path $bin "muse.cmd") -Force
if ([string]::IsNullOrWhiteSpace($env:MUSE_NO_MODIFY_PATH)) {
    $userPath = [IO.File]::ReadAllText($env:FAKE_USER_PATH_FILE)
    $entries = @($userPath.Split(';', [StringSplitOptions]::RemoveEmptyEntries))
    if ($entries -notcontains $bin) {
        $newUserPath = if ([string]::IsNullOrEmpty($userPath)) { $bin } else { "$bin;$userPath" }
        [IO.File]::WriteAllText($env:FAKE_USER_PATH_FILE, $newUserPath)
    }
}
Add-Content -LiteralPath $env:CALL_LOG -Value "muse-install:$bin"
""",
        encoding="utf-8",
    )
    (fixtures / "uv-installer.ps1").write_text(
        r"""if ($env:FAIL_STEP -eq "uv-install") { exit 63 }
$forceInstallDir = if ($env:UV_INSTALL_DIR) {
    $env:UV_INSTALL_DIR
}
elseif ($env:UV_UNMANAGED_INSTALL) {
    $env:UV_UNMANAGED_INSTALL
}
else {
    $null
}
$bin = if ($forceInstallDir) {
    if ($forceInstallDir -eq $(if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE ".cargo" })) {
        Join-Path $forceInstallDir "bin"
    }
    else {
        $forceInstallDir
    }
}
elseif ($env:XDG_BIN_HOME) {
    $env:XDG_BIN_HOME
}
elseif ($env:XDG_DATA_HOME) {
    Join-Path $env:XDG_DATA_HOME "..\bin"
}
else {
    Join-Path $env:USERPROFILE ".local\bin"
}
New-Item -ItemType Directory -Force -Path $bin | Out-Null
Copy-Item (Join-Path $env:FAKE_FIXTURES "uv-command.cmd") (Join-Path $bin "uv.cmd") -Force
Add-Content -LiteralPath $env:CALL_LOG -Value "uv-install"
""",
        encoding="utf-8",
    )
    rtk_archive = fixtures / "rtk-x86_64-pc-windows-msvc.zip"
    with zipfile.ZipFile(rtk_archive, "w") as archive:
        archive.writestr("rtk.exe", b"fake RTK executable")
    for asset_name in (
        "opencode-windows-x64-baseline.zip",
        "opencode-windows-arm64.zip",
    ):
        with zipfile.ZipFile(
            fixtures / asset_name, "w", zipfile.ZIP_DEFLATED
        ) as archive:
            archive.write(
                Path(sys.executable).with_name("fcc-server.exe"),
                arcname="opencode.exe",
            )
    (bin_dir / "opencode.cmd").write_text(_batch_client("opencode"), encoding="utf-8")
    (bin_dir / "node.cmd").write_text(_batch_client("node"), encoding="utf-8")
    npm_prefix = tmp_path / "npm-prefix"
    npm_prefix.mkdir()
    (bin_dir / "npm.cmd").write_text(_batch_npm(), encoding="utf-8")

    wrapper = tmp_path / "run-installer.ps1"
    wrapper.write_text(
        """Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
function Invoke-RestMethod {
    [CmdletBinding()]
    param([string] $Uri, [string] $OutFile)

    Add-Content -LiteralPath $env:CALL_LOG -Value "download:$Uri"
    if (
        ($env:FAIL_STEP -eq "claude-download" -and $Uri.Contains("claude.ai")) -or
        ($env:FAIL_STEP -eq "codex-download" -and $Uri.Contains("chatgpt.com")) -or
        ($env:FAIL_STEP -eq "pi-download" -and $Uri.Contains("pi.dev")) -or
        ($env:FAIL_STEP -eq "opencode-download" -and $Uri.Contains("anomalyco/opencode")) -or
        ($env:FAIL_STEP -eq "hermes-download" -and $Uri.Contains("hermes-agent.nousresearch.com")) -or
        ($env:FAIL_STEP -eq "grok-download" -and $Uri.Contains("x.ai/cli")) -or
        ($env:FAIL_STEP -eq "muse-download" -and $Uri.Contains("dev.meta.ai")) -or
        ($env:FAIL_STEP -eq "rtk-download" -and $Uri.Contains("rtk-ai/rtk")) -or
        ($env:FAIL_STEP -eq "uv-download" -and $Uri.Contains("astral.sh"))
    ) {
        throw "simulated download failure"
    }
    if ($Uri.Contains("claude.ai")) {
        $source = Join-Path $env:FAKE_FIXTURES "claude-installer.ps1"
    }
    elseif ($Uri.Contains("chatgpt.com")) {
        $source = Join-Path $env:FAKE_FIXTURES "codex-installer.ps1"
    }
    elseif ($Uri.Contains("pi.dev")) {
        $source = Join-Path $env:FAKE_FIXTURES "pi-installer.ps1"
    }
    elseif ($Uri.Contains("hermes-agent.nousresearch.com")) {
        $source = Join-Path $env:FAKE_FIXTURES "hermes-installer.ps1"
    }
    elseif ($Uri.Contains("x.ai/cli")) {
        $source = Join-Path $env:FAKE_FIXTURES "grok-installer.ps1"
    }
    elseif ($Uri -eq "https://dev.meta.ai/install.ps1") {
        $source = Join-Path $env:FAKE_FIXTURES "muse-installer.ps1"
    }
    elseif ($Uri.Contains("opencode-windows-")) {
        if ($env:FAIL_STEP -eq "opencode-archive") {
            Set-Content -LiteralPath $OutFile -Value "not a zip"
            return
        }
        $source = Join-Path $env:FAKE_FIXTURES ([IO.Path]::GetFileName($Uri))
    }
    elseif ($Uri.EndsWith("rtk-x86_64-pc-windows-msvc.zip")) {
        $source = Join-Path $env:FAKE_FIXTURES "rtk-x86_64-pc-windows-msvc.zip"
    }
    elseif ($Uri.Contains("astral.sh")) {
        $source = Join-Path $env:FAKE_FIXTURES "uv-installer.ps1"
    }
    else {
        throw "unexpected installer URL: $Uri"
    }
    Copy-Item -LiteralPath $source -Destination $OutFile -Force
}
function Get-Process {
    [CmdletBinding()]
    param([string[]] $Name)

    if ([string]::IsNullOrWhiteSpace($env:FCC_RUNNING_COMMAND)) {
        return
    }
    if (
        $env:FCC_RUNNING_PHASE -eq "late" -and
        -not (Test-Path -LiteralPath $env:FCC_PROCESS_MARKER)
    ) {
        return
    }
    foreach ($requestedName in $Name) {
        if ($requestedName -eq $env:FCC_RUNNING_COMMAND) {
            [pscustomobject] @{ Id = 4242; ProcessName = $requestedName }
        }
    }
}
$installerSource = [IO.File]::ReadAllText($env:FCC_INSTALLER)
$fakeMuseUserPath = @'
function Get-MuseUserPathValue {
    return [IO.File]::ReadAllText($env:FAKE_USER_PATH_FILE)
}

function Set-MuseUserPathValue {
    param([AllowEmptyString()][string] $Value)

    [IO.File]::WriteAllText($env:FAKE_USER_PATH_FILE, $Value)
}
'@
$museUserPathStart = $installerSource.IndexOf("function Get-MuseUserPathValue {")
$museUserPathEnd = $installerSource.IndexOf("function Set-MuseUpstreamPathPriority {", $museUserPathStart)
if ($museUserPathStart -lt 0 -or $museUserPathEnd -lt 0) {
    throw "failed to isolate the Muse user PATH helpers"
}
$installerSource = $installerSource.Substring(0, $museUserPathStart) + `
    $fakeMuseUserPath.TrimEnd() + "`r`n`r`n" + `
    $installerSource.Substring($museUserPathEnd)
$nativeVersionProbe = '    $output = Invoke-Utf8NativeCapture -FilePath $OpenCodePath -Arguments @("--version")'
$fakeArchiveVersionProbe = @'
    $output = if ([IO.Path]::GetFileName($OpenCodePath) -eq "opencode.exe") {
        "opencode 1.18.18"
    }
    else {
        Invoke-Utf8NativeCapture -FilePath $OpenCodePath -Arguments @("--version")
    }
'@
$installerSource = $installerSource.Replace($nativeVersionProbe, $fakeArchiveVersionProbe.TrimEnd())
$installer = [scriptblock]::Create($installerSource)
& $installer @args
""",
        encoding="utf-8",
    )

    system_root = os.environ["SYSTEMROOT"]
    user_path_file = tmp_path / "user-path.txt"
    user_path_file.write_text("", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PATH": os.pathsep.join(
                [str(bin_dir), str(Path(system_root) / "System32"), system_root]
            ),
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "USERPROFILE": str(home),
            "LOCALAPPDATA": str(local_app_data),
            "APPDATA": str(app_data),
            "CALL_LOG": str(log),
            "CLAUDE_CONFIG_DIR": "",
            "FAKE_FIXTURES": str(fixtures),
            "FAKE_TOOL_BIN": str(tool_bin),
            "FAKE_USER_PATH_FILE": str(user_path_file),
            "FAKE_NPM_PREFIX": str(npm_prefix),
            "FCC_INSTALLER": str(_repo_root() / "scripts" / "install.ps1"),
            "FCC_PROCESS_MARKER": str(tmp_path / "fcc-process-ready"),
            "FCC_RUNNING_COMMAND": "",
            "FCC_RUNNING_PHASE": "early",
            "PROCESSOR_ARCHITECTURE": "AMD64",
            "PROCESSOR_ARCHITEW6432": "",
            "FAIL_STEP": "",
        }
    )
    env.pop("XDG_BIN_HOME", None)
    env.pop("XDG_DATA_HOME", None)
    env.pop("UV_INSTALL_DIR", None)
    env.pop("UV_UNMANAGED_INSTALL", None)
    env.pop("UV_TOOL_BIN_DIR", None)
    env.pop("CARGO_HOME", None)
    env.pop("GROK_BIN_DIR", None)
    return PowerShellHarness(
        tmp_path, bin_dir, fixtures, tool_bin, log, env, powershell, wrapper
    )


def _install_legacy_muse(
    harness: PowerShellHarness,
    *,
    marker: str = "valid",
) -> tuple[Path, Path, Path, Path]:
    root = Path(harness.env["LOCALAPPDATA"]) / "Programs" / "Muse Code"
    bin_dir = root / "bin"
    executable = bin_dir / "muse.exe"
    record = root / ".fcc-muse-install.json"
    bin_dir.mkdir(parents=True)
    shutil.copy2(sys.executable, executable)
    ownership = {
        "schema_version": 1,
        "owner": "free-claude-code-muse-installer",
        "release_version": "1.0.3-R2198.1",
        "artifact_key": "x86_windows",
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "size": executable.stat().st_size,
    }
    if marker == "valid":
        record.write_text(json.dumps(ownership), encoding="utf-8")
    elif marker == "foreign":
        ownership["owner"] = "someone-else"
        record.write_text(json.dumps(ownership), encoding="utf-8")
    elif marker == "malformed":
        record.write_text("{", encoding="utf-8")
    elif marker != "missing":
        raise AssertionError(f"unknown marker fixture: {marker}")
    harness.env["PATH"] = f"{bin_dir}{os.pathsep}{harness.env['PATH']}"
    harness.set_user_path(bin_dir, harness.root / "unrelated-bin")
    return root, bin_dir, executable, record


def test_install_ps1_fresh_install_is_verified(
    powershell_harness: PowerShellHarness,
) -> None:
    (powershell_harness.bin_dir / "opencode.cmd").unlink()
    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "Free Claude Code is installed and verified." in result.stdout
    calls = powershell_harness.calls()
    assert calls.index("claude-install") < calls.index("claude:--version")
    assert calls.index("codex-install:1") < calls.index("codex:--version")
    assert calls.index("pi-install") < calls.index("pi:--version")
    assert any("anomalyco/opencode" in call for call in calls)
    assert calls.index("npm:install -g cline") < calls.index("cline:--version")
    assert any("hermes-agent.nousresearch.com/install.ps1" in call for call in calls)
    assert "hermes-install:True:True" in calls
    assert calls.index("npm:install -g @deepseek-ai/dsh@0.1.0-rc.8") < calls.index(
        "dsh:--version"
    )
    assert calls.index("grok-install") < calls.index("grok:--version")
    muse_install = next(call for call in calls if call.startswith("muse-install:"))
    assert calls.index(muse_install) < calls.index("muse:--version")
    muse_dir = Path(powershell_harness.env["LOCALAPPDATA"]) / "Programs" / "muse"
    assert (muse_dir / "muse.cmd").is_file()
    assert powershell_harness.user_path_entries()[0] == str(muse_dir)
    assert not any("hermes:setup" in call for call in calls)
    assert calls.index("uv-install") < calls.index("uv:--version")
    aider_install = (
        "uv:tool install --force --python python3.12 --with pip aider-chat@latest"
    )
    assert calls.index("uv:--version") < calls.index("claude-install")
    assert calls.index(aider_install) < calls.index("aider:--version")
    assert any(
        call.startswith(
            "uv:tool install --force --refresh-package free-claude-code "
            "--python cpython-3.14.0-windows-x86_64-none "
            '"free-claude-code @ '
            'https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"'
        )
        for call in calls
    )
    assert not any(call.startswith("git:") for call in calls)
    assert calls[-4:-1] == [
        "uv:tool update-shell",
        "uv:tool dir --bin",
        "fcc-server:--version",
    ]
    home = Path(powershell_harness.env["USERPROFILE"])
    app_data = Path(powershell_harness.env["APPDATA"])
    assert (home / ".grok" / "bin" / "grok.cmd").is_file()
    assert not (home / ".local" / "bin" / "grok.cmd").exists()
    icon = home / ".fcc" / "app-icon.ico"
    assert icon.read_text(encoding="utf-8").strip() == "fake icon"
    assert calls[-1] == f'fcc-desktop:--export-icon "{icon}"'
    desktop_shortcut = home / "Desktop" / "Free Claude Code.lnk"
    assert desktop_shortcut.is_file()
    assert (
        _windows_shortcut_icon(
            powershell_harness.powershell,
            desktop_shortcut,
            powershell_harness.env,
        )
        == f"{icon},0"
    )
    assert (
        app_data
        / "Microsoft"
        / "Windows"
        / "Start Menu"
        / "Programs"
        / "Free Claude Code.lnk"
    ).is_file()


def test_install_ps1_discovers_grok_in_custom_bin_directory(
    powershell_harness: PowerShellHarness,
) -> None:
    custom_grok_bin = powershell_harness.root / "custom-grok-bin"
    powershell_harness.env["GROK_BIN_DIR"] = str(custom_grok_bin)

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert (custom_grok_bin / "grok.cmd").is_file()
    assert not (
        Path(powershell_harness.env["USERPROFILE"]) / ".grok" / "bin" / "grok.cmd"
    ).exists()
    calls = powershell_harness.calls()
    assert calls.index("grok-install") < calls.index("grok:--version")


@pytest.mark.parametrize(
    ("client", "install_call"),
    [
        ("opencode", "anomalyco/opencode"),
        ("cline", "npm:install -g cline"),
        ("hermes", "hermes-install:True:True"),
        ("grok", "grok-install"),
        (
            "aider",
            "uv:tool install --force --python python3.12 --with pip aider-chat@latest",
        ),
    ],
)
def test_install_ps1_preserves_upstream_managed_harness_without_parsing_version(
    powershell_harness: PowerShellHarness,
    client: str,
    install_call: str,
) -> None:
    _write_executable(
        powershell_harness.bin_dir / f"{client}.cmd",
        _batch_client(client, version_output="opaque upstream version output"),
    )

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    calls = powershell_harness.calls()
    assert f"{client}:--version" in calls
    assert not any(install_call in call for call in calls)


def test_install_ps1_preserves_compatible_external_muse_without_adopting_it(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("muse")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    calls = powershell_harness.calls()
    assert "muse:--version" in calls
    assert not any(call.startswith("muse-install:") for call in calls)
    assert "download:https://dev.meta.ai/install.ps1" not in calls
    managed_root = (
        Path(powershell_harness.env["LOCALAPPDATA"]) / "Programs" / "Muse Code"
    )
    assert not managed_root.exists()


def test_install_ps1_migrates_owned_muse_after_upstream_verification(
    powershell_harness: PowerShellHarness,
) -> None:
    root, legacy_bin, executable, record = _install_legacy_muse(powershell_harness)
    unknown = root / "user-notes.txt"
    unknown.write_text("keep", encoding="utf-8")
    residue = legacy_bin / f".muse-{'a' * 32}.staging.exe"
    residue.write_bytes(b"staging")
    record_residue = root / f".fcc-muse-install.json.{'b' * 32}.backup"
    record_residue.write_bytes(b"backup")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    upstream = Path(powershell_harness.env["LOCALAPPDATA"]) / "Programs" / "muse"
    assert (upstream / "muse.cmd").is_file()
    assert not executable.exists()
    assert not record.exists()
    assert not residue.exists()
    assert not record_residue.exists()
    assert unknown.read_text(encoding="utf-8") == "keep"
    user_path = powershell_harness.user_path_entries()
    assert user_path[0] == str(upstream)
    assert str(legacy_bin) not in user_path
    calls = powershell_harness.calls()
    muse_install = next(call for call in calls if call.startswith("muse-install:"))
    assert calls.index(muse_install) < calls.index("muse:--version")


def test_install_ps1_failed_upstream_verification_preserves_owned_muse(
    powershell_harness: PowerShellHarness,
) -> None:
    _, legacy_bin, executable, record = _install_legacy_muse(powershell_harness)

    result = powershell_harness.run(fail_step="muse-verify")

    assert result.returncode != 0
    assert executable.is_file()
    assert record.is_file()
    assert str(legacy_bin) in powershell_harness.user_path_entries()
    assert "Free Claude Code is installed and verified." not in result.stdout


@pytest.mark.parametrize("marker", ("missing", "malformed", "foreign"))
def test_install_ps1_preserves_ambiguous_legacy_muse_and_prioritizes_upstream(
    powershell_harness: PowerShellHarness,
    marker: str,
) -> None:
    root, legacy_bin, executable, record = _install_legacy_muse(
        powershell_harness,
        marker=marker,
    )
    upstream = Path(powershell_harness.env["LOCALAPPDATA"]) / "Programs" / "muse"
    powershell_harness.set_user_path(legacy_bin, upstream)
    unknown = root / "user-notes.txt"
    unknown.write_text("keep", encoding="utf-8")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert executable.is_file()
    assert record.exists() is (marker != "missing")
    assert unknown.read_text(encoding="utf-8") == "keep"
    assert powershell_harness.user_path_entries()[:2] == [
        str(upstream),
        str(legacy_bin),
    ]
    assert "could not prove ownership" in result.stdout
    assert "was preserved" in result.stdout


def test_install_ps1_honors_custom_upstream_muse_directory(
    powershell_harness: PowerShellHarness,
) -> None:
    custom_dir = powershell_harness.root / "custom-muse"
    powershell_harness.env["MUSE_INSTALL_DIR"] = str(custom_dir)

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert (custom_dir / "muse.cmd").is_file()
    assert powershell_harness.user_path_entries()[0] == str(custom_dir)
    assert f"muse-install:{custom_dir}" in powershell_harness.calls()


def test_install_ps1_migrates_when_upstream_reuses_legacy_bin(
    powershell_harness: PowerShellHarness,
) -> None:
    _, legacy_bin, executable, record = _install_legacy_muse(powershell_harness)
    powershell_harness.env["MUSE_INSTALL_DIR"] = str(legacy_bin)

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert (legacy_bin / "muse.cmd").is_file()
    assert not executable.exists()
    assert not record.exists()
    assert powershell_harness.user_path_entries().count(str(legacy_bin)) == 1


def test_install_ps1_honors_muse_path_opt_out_during_migration(
    powershell_harness: PowerShellHarness,
) -> None:
    _, _, executable, record = _install_legacy_muse(powershell_harness)
    original_user_path = powershell_harness.user_path_entries()
    powershell_harness.env["MUSE_NO_MODIFY_PATH"] = "1"

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert not executable.exists()
    assert not record.exists()
    assert powershell_harness.user_path_entries() == original_user_path
    assert "MUSE_NO_MODIFY_PATH is set" in result.stdout


def test_install_ps1_muse_migration_preserves_empty_path_entries(
    powershell_harness: PowerShellHarness,
) -> None:
    _, legacy_bin, _, _ = _install_legacy_muse(powershell_harness)
    unrelated = powershell_harness.root / "unrelated-bin"
    powershell_harness.set_user_path("", legacy_bin, "", unrelated, "")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    upstream = Path(powershell_harness.env["LOCALAPPDATA"]) / "Programs" / "muse"
    assert powershell_harness.user_path_entries() == [
        str(upstream),
        "",
        "",
        str(unrelated),
        "",
    ]


@pytest.mark.parametrize("failure", ["muse-download", "muse-install"])
def test_install_ps1_stops_when_muse_install_fails(
    powershell_harness: PowerShellHarness,
    failure: str,
) -> None:
    result = powershell_harness.run(fail_step=failure)

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    _assert_uv_ready_without_fcc_install(powershell_harness.calls())


@pytest.mark.parametrize("failure", ["grok-download", "grok-install"])
def test_install_ps1_stops_when_grok_install_fails(
    powershell_harness: PowerShellHarness,
    failure: str,
) -> None:
    result = powershell_harness.run(fail_step=failure)

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    _assert_uv_ready_without_fcc_install(powershell_harness.calls())


def test_install_ps1_stops_when_aider_install_fails(
    powershell_harness: PowerShellHarness,
) -> None:
    result = powershell_harness.run(fail_step="aider-install")

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    calls = powershell_harness.calls()
    aider_install = (
        "uv:tool install --force --python python3.12 --with pip aider-chat@latest"
    )
    assert calls.index("uv:--version") < calls.index(aider_install)
    assert not any("aider.chat/install" in call for call in calls)
    assert not any("--refresh-package free-claude-code" in call for call in calls)


def test_install_ps1_discovers_aider_in_custom_uv_tool_bin(
    powershell_harness: PowerShellHarness,
) -> None:
    custom_tool_bin = powershell_harness.root / "custom-tool-bin"
    powershell_harness.env["UV_TOOL_BIN_DIR"] = str(custom_tool_bin)

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert (custom_tool_bin / "aider.cmd").is_file()
    calls = powershell_harness.calls()
    assert "uv:tool dir --bin" in calls
    assert "aider:--version" in calls


@pytest.mark.parametrize("fail_step", ("", "aider-verify"), ids=("valid", "broken"))
def test_install_ps1_checks_existing_aider_in_custom_uv_tool_bin_before_installing(
    powershell_harness: PowerShellHarness,
    fail_step: str,
) -> None:
    custom_tool_bin = powershell_harness.root / "custom-tool-bin"
    existing_aider = custom_tool_bin / "aider.cmd"
    custom_tool_bin.mkdir()
    powershell_harness.env["UV_TOOL_BIN_DIR"] = str(custom_tool_bin)
    existing_aider.write_text(
        _batch_client("aider", version_output="existing aider 1.0.0"),
        encoding="utf-8",
    )
    original = existing_aider.read_bytes()

    result = powershell_harness.run(fail_step=fail_step)

    if fail_step:
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stderr
    calls = powershell_harness.calls()
    assert calls.index("uv:tool dir --bin") < calls.index("aider:--version")
    assert not any("aider-chat@latest" in call for call in calls)
    assert existing_aider.read_bytes() == original


def test_install_ps1_rejects_broken_existing_aider_without_replacing_it(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("aider")

    result = powershell_harness.run(fail_step="aider-verify")

    assert result.returncode != 0
    calls = powershell_harness.calls()
    assert "aider:--version" in calls
    assert not any("aider-chat@latest" in call for call in calls)
    assert not any("aider.chat" in call for call in calls)


def test_install_ps1_preserves_exact_dsh_preview(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("dsh")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "already matches the supported preview" in result.stdout
    assert "npm:install -g @deepseek-ai/dsh@0.1.0-rc.8" not in (
        powershell_harness.calls()
    )


def test_install_ps1_replaces_mismatched_dsh_preview(
    powershell_harness: PowerShellHarness,
) -> None:
    (powershell_harness.bin_dir / "dsh.cmd").write_text(
        _batch_client("dsh").replace("0.1.0-rc.8", "0.1.0-rc.7"),
        encoding="utf-8",
    )

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "does not match 0.1.0-rc.8" in result.stdout
    assert "npm:install -g @deepseek-ai/dsh@0.1.0-rc.8" in (powershell_harness.calls())


def test_install_ps1_rejects_exact_dsh_on_unsupported_node(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("dsh")
    (powershell_harness.bin_dir / "node.cmd").write_text(
        _batch_client("node").replace("node 22.19.0", "node 23.9.0"),
        encoding="utf-8",
    )

    result = powershell_harness.run()

    assert result.returncode != 0
    assert "requires Node.js ^22.19.0 or >=24.0.0" in (
        f"{result.stdout}\n{result.stderr}"
    )
    _assert_uv_ready_without_fcc_install(powershell_harness.calls())


@pytest.mark.parametrize("node_version", ["22.18.0", "23.9.0", "not-a-version"])
def test_install_ps1_rejects_incompatible_node_for_selected_dsh(
    powershell_harness: PowerShellHarness,
    node_version: str,
) -> None:
    (powershell_harness.bin_dir / "dsh.cmd").write_text(
        _batch_client("dsh").replace("0.1.0-rc.8", "0.1.0-rc.7"),
        encoding="utf-8",
    )
    (powershell_harness.bin_dir / "node.cmd").write_text(
        _batch_client("node").replace("node 22.19.0", f"node {node_version}"),
        encoding="utf-8",
    )

    result = powershell_harness.run()

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    _assert_uv_ready_without_fcc_install(powershell_harness.calls())


def test_install_ps1_noninteractive_skips_dsh_without_node(
    powershell_harness: PowerShellHarness,
) -> None:
    (powershell_harness.bin_dir / "node.cmd").unlink()
    (powershell_harness.bin_dir / "npm.cmd").unlink()

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert (
        "fcc-dsh wrapper is ready after you install DeepSeek Harness" in result.stdout
    )
    assert not any("@deepseek-ai/dsh" in call for call in powershell_harness.calls())


def test_install_ps1_stops_when_selected_dsh_install_fails(
    powershell_harness: PowerShellHarness,
) -> None:
    result = powershell_harness.run(fail_step="dsh-install")

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    _assert_uv_ready_without_fcc_install(powershell_harness.calls())


def test_install_ps1_rejects_unsupported_hermes_architecture_before_download(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.env["PROCESSOR_ARCHITECTURE"] = "MIPS"
    powershell_harness.env["PROCESSOR_ARCHITEW6432"] = "MIPS"

    result = powershell_harness.run()

    assert result.returncode != 0
    assert "does not provide a supported Windows release" in result.stderr
    assert not any(
        "hermes-agent.nousresearch.com" in call for call in powershell_harness.calls()
    )


def test_install_ps1_selects_official_opencode_arm64_archive(
    powershell_harness: PowerShellHarness,
) -> None:
    (powershell_harness.bin_dir / "opencode.cmd").unlink()
    powershell_harness.env["PROCESSOR_ARCHITEW6432"] = "ARM64"

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert any(
        call.endswith("opencode-windows-arm64.zip")
        for call in powershell_harness.calls()
    )


def test_install_ps1_rejects_unsupported_opencode_architecture(
    powershell_harness: PowerShellHarness,
) -> None:
    (powershell_harness.bin_dir / "opencode.cmd").unlink()
    powershell_harness.env["PROCESSOR_ARCHITEW6432"] = "X86"

    result = powershell_harness.run()

    assert result.returncode != 0
    assert "does not provide a supported Windows release" in result.stderr
    assert not any("anomalyco/opencode" in call for call in powershell_harness.calls())


def test_install_ps1_preserves_existing_rtk_and_configures_selected_agents(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_rtk()

    result = powershell_harness.run("-Rtk")

    assert result.returncode == 0, result.stderr
    assert "verifying it without updating it" in result.stdout
    calls = powershell_harness.calls()
    assert not any("rtk-ai/rtk" in call for call in calls)
    assert "rtk:--version:telemetry=1" in calls
    assert "rtk:gain:telemetry=1" in calls
    assert [call for call in calls if call.startswith("rtk:init")] == [
        "rtk:init --global --auto-patch:telemetry=1",
        "rtk:init --global --codex:telemetry=1",
        "rtk:init --global --agent pi:telemetry=1",
        "rtk:init --global --opencode:telemetry=1",
    ]
    assert (Path(powershell_harness.env["USERPROFILE"]) / ".claude").is_dir()


def test_install_ps1_prepares_custom_claude_config_directory_for_rtk(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_rtk()
    custom_config = powershell_harness.root / "custom-claude"
    powershell_harness.env["CLAUDE_CONFIG_DIR"] = str(custom_config)

    result = powershell_harness.run("-Rtk")

    assert result.returncode == 0, result.stderr
    assert custom_config.is_dir()
    assert not (Path(powershell_harness.env["USERPROFILE"]) / ".claude").exists()


def test_install_ps1_rejects_conflicting_rtk_command(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_unrelated_rtk()

    result = powershell_harness.run("-Rtk")

    assert result.returncode != 0
    assert "not a compatible Rust Token Killer installation" in result.stderr
    assert not any("rtk-ai/rtk" in call for call in powershell_harness.calls())
    _assert_uv_ready_without_fcc_install(powershell_harness.calls())


def test_install_ps1_rtk_dry_run_prints_install_and_agent_setup(
    powershell_harness: PowerShellHarness,
) -> None:
    result = powershell_harness.run("-Rtk", "-DryRun")

    assert result.returncode == 0, result.stderr
    assert powershell_harness.calls() == []
    assert "releases/download/v0.44.2/rtk-x86_64-pc-windows-msvc.zip" in result.stdout
    assert "RTK_TELEMETRY_DISABLED=1 rtk init --global --auto-patch" in result.stdout
    assert "RTK_TELEMETRY_DISABLED=1 rtk init --global --codex" in result.stdout
    assert "RTK_TELEMETRY_DISABLED=1 rtk init --global --agent pi" in result.stdout


@pytest.mark.parametrize(
    "powershell",
    _powershells() or (None,),
    ids=lambda path: Path(path).name if path is not None else "unavailable",
)
@pytest.mark.parametrize("valid_checksum", [True, False])
def test_install_ps1_installs_only_checksum_verified_rtk_archive(
    powershell: str | None,
    tmp_path: Path,
    valid_checksum: bool,
) -> None:
    if powershell is None or os.name != "nt":
        pytest.skip("PowerShell RTK archive installation runs on Windows hosts")

    asset_name = "rtk-x86_64-pc-windows-msvc.zip"
    archive_path = tmp_path / asset_name
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("rtk.exe", b"verified RTK executable")
    checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if not valid_checksum:
        checksum = "0" * 64

    installer = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")
    format_argument = _braced_body(installer, "function Format-Argument")
    install_rtk = _braced_body(installer, "function Install-Rtk")
    script = f"""Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$DryRun = $false
$RtkReleaseBaseUrl = "https://example.test/releases/download/v0.44.2"
$RtkWindowsAssetName = "{asset_name}"
$RtkWindowsAssetSha256 = "{checksum}"
function Format-Argument {{{format_argument}}}
function Invoke-RestMethod {{
    [CmdletBinding()]
    param([string] $Uri, [string] $OutFile)
    Copy-Item -LiteralPath $env:RTK_TEST_ARCHIVE -Destination $OutFile
}}
function Install-Rtk {{{install_rtk}}}
Install-Rtk
"""
    home = tmp_path / "home"
    home.mkdir()
    env = os.environ | {
        "USERPROFILE": str(home),
        "RTK_TEST_ARCHIVE": str(archive_path),
    }
    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    installed = home / ".local" / "bin" / "rtk.exe"
    if valid_checksum:
        assert result.returncode == 0, result.stderr
        assert installed.read_bytes() == b"verified RTK executable"
    else:
        assert result.returncode != 0
        assert "checksum verification failed" in result.stderr
        assert not installed.exists()


def test_install_ps1_stops_if_windows_icon_export_fails(
    powershell_harness: PowerShellHarness,
) -> None:
    result = powershell_harness.run(fail_step="desktop-icon-export")

    assert result.returncode != 0
    assert "Command failed with exit code 56" in result.stderr
    home = Path(powershell_harness.env["USERPROFILE"])
    assert not (home / "Desktop" / "Free Claude Code.lnk").exists()


def test_install_ps1_preserves_unowned_desktop_shortcut(
    powershell_harness: PowerShellHarness,
) -> None:
    desktop_shortcut = (
        Path(powershell_harness.env["USERPROFILE"]) / "Desktop" / "Free Claude Code.lnk"
    )
    unrelated_target = powershell_harness.root / "unrelated.cmd"
    unrelated_target.write_text("@echo off\n", encoding="utf-8")
    _create_windows_shortcut(
        powershell_harness.powershell,
        desktop_shortcut,
        unrelated_target,
    )
    original_shortcut = desktop_shortcut.read_bytes()

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "not managed by Free Claude Code" in result.stdout
    assert desktop_shortcut.read_bytes() == original_shortcut


@pytest.mark.parametrize("uv_version", ("0.12.13", "0.12.13+build.1", "0.12.14"))
def test_install_ps1_preserves_valid_existing_tools(
    powershell_harness: PowerShellHarness,
    uv_version: str,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_client("pi")
    powershell_harness.add_client("cline")
    powershell_harness.add_client("hermes")
    powershell_harness.add_client("grok")
    powershell_harness.add_client("muse")
    powershell_harness.add_client("aider")
    powershell_harness.add_uv(uv_version)

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    download_calls = [
        call for call in powershell_harness.calls() if call.startswith("download:")
    ]
    assert download_calls == []
    assert not any(
        call.startswith("muse-install:") for call in powershell_harness.calls()
    )
    assert "leaving it unchanged" in result.stdout


def test_install_ps1_replaces_unrelated_pi_command(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_unrelated_pi()
    powershell_harness.add_uv("0.12.13")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "is not Pi Coding Agent; installing Pi" in result.stdout
    assert "pi-install" in powershell_harness.calls()


def test_install_ps1_discovers_custom_pi_npm_prefix(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_npm_prefix(powershell_harness.root / "custom-npm")
    powershell_harness.add_uv("0.12.13")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    calls = powershell_harness.calls()
    assert "npm:prefix -g" in calls
    assert "pi:--help" in calls
    assert "pi:--version" in calls


def test_install_ps1_continues_when_pi_is_not_installed(
    powershell_harness: PowerShellHarness,
) -> None:
    result = powershell_harness.run(fail_step="pi-skip")

    assert result.returncode == 0, result.stderr
    assert "Pi was not installed; continuing without it." in result.stdout
    assert "Run Pi with: fcc-pi" not in result.stdout
    calls = powershell_harness.calls()
    assert "pi-install" in calls
    assert not any(call.startswith("pi:") for call in calls)
    assert "uv-install" in calls
    assert "fcc-server:--version" in calls


def test_install_ps1_continues_when_unrelated_pi_is_unchanged(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_unrelated_pi()

    result = powershell_harness.run(fail_step="pi-skip")

    assert result.returncode == 0, result.stderr
    assert "Pi was not installed; continuing without it." in result.stdout
    assert "Run Pi with: fcc-pi" not in result.stdout
    calls = powershell_harness.calls()
    assert "unrelated-pi:--help" in calls
    assert "unrelated-pi:--version" not in calls
    assert "fcc-server:--version" in calls


def test_install_ps1_continues_when_pi_resolution_changes_to_unrelated_command(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_unrelated_pi()
    npm_prefix = powershell_harness.root / "custom-npm"
    powershell_harness.add_npm_prefix(npm_prefix)
    _write_executable(
        npm_prefix / "pi.cmd",
        _batch_client("other-unrelated-pi"),
    )

    result = powershell_harness.run(fail_step="pi-skip")

    assert result.returncode == 0, result.stderr
    assert "Pi was not installed; continuing without it." in result.stdout
    assert "Run Pi with: fcc-pi" not in result.stdout
    calls = powershell_harness.calls()
    assert "other-unrelated-pi:--help" in calls
    assert "other-unrelated-pi:--version" not in calls
    assert "fcc-server:--version" in calls


@pytest.mark.parametrize("uv_version", ("0.5.9", "0.11.16", "0.12.9", "0.12.12"))
def test_install_ps1_replaces_obsolete_uv(
    powershell_harness: PowerShellHarness,
    uv_version: str,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_client("pi")
    powershell_harness.add_uv(uv_version)

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "uv-install" in powershell_harness.calls()
    assert f"uv {uv_version} does not satisfy stable >=0.12.13" in result.stdout


def test_install_ps1_prioritizes_replacement_uv_from_custom_install_directory(
    powershell_harness: PowerShellHarness,
) -> None:
    custom_install_dir = powershell_harness.root / "custom-uv-bin"
    powershell_harness.env["UV_INSTALL_DIR"] = str(custom_install_dir)
    powershell_harness.add_uv("0.5.9")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "Verified uv 0.12.13." in result.stdout
    assert "uv-install" in powershell_harness.calls()


@pytest.mark.parametrize("install_variable", ("UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL"))
def test_install_ps1_prioritizes_forced_cargo_home_uv_install_layout(
    powershell_harness: PowerShellHarness,
    install_variable: str,
) -> None:
    cargo_home = Path(powershell_harness.env["USERPROFILE"]) / ".cargo"
    cargo_bin = cargo_home / "bin"
    powershell_harness.env["CARGO_HOME"] = str(cargo_home)
    powershell_harness.env[install_variable] = str(cargo_home)
    powershell_harness.env["PATH"] = (
        f"{powershell_harness.env['PATH']}{os.pathsep}{cargo_bin}"
    )
    powershell_harness.add_uv("0.5.9")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "Verified uv 0.12.13." in result.stdout
    assert "uv-install" in powershell_harness.calls()


def test_install_ps1_uv_install_dir_takes_precedence_over_unmanaged_install(
    powershell_harness: PowerShellHarness,
) -> None:
    install_bin = powershell_harness.root / "uv-install-bin"
    unmanaged_bin = powershell_harness.root / "unmanaged-uv-bin"
    powershell_harness.env["UV_INSTALL_DIR"] = str(install_bin)
    powershell_harness.env["UV_UNMANAGED_INSTALL"] = str(unmanaged_bin)
    powershell_harness.add_uv("0.5.9")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert (install_bin / "uv.cmd").is_file()
    assert not (unmanaged_bin / "uv.cmd").exists()


def test_install_ps1_prioritizes_replacement_uv_from_unmanaged_install_directory(
    powershell_harness: PowerShellHarness,
) -> None:
    unmanaged_bin = powershell_harness.root / "unmanaged-uv-bin"
    powershell_harness.env["UV_UNMANAGED_INSTALL"] = str(unmanaged_bin)
    powershell_harness.add_uv("0.5.9")

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert "Verified uv 0.12.13." in result.stdout
    assert "uv-install" in powershell_harness.calls()


@pytest.mark.parametrize("version", ("0.12.13-alpha.1", "0.13.0-rc.1"))
def test_install_ps1_replaces_prerelease_uv(
    powershell_harness: PowerShellHarness,
    version: str,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_client("pi")
    powershell_harness.add_uv(version)

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr
    assert f"uv {version} does not satisfy stable >=0.12.13" in result.stdout
    assert "uv-install" in powershell_harness.calls()


@pytest.mark.parametrize(
    "failure",
    [
        "claude-download",
        "claude-install",
        "claude-verify",
        "codex-download",
        "codex-install",
        "codex-verify",
        "pi-download",
        "pi-install",
        "pi-verify",
        "opencode-download",
        "opencode-archive",
        "opencode-verify",
        "cline-install",
        "cline-verify",
        "uv-download",
        "uv-install",
        "uv-verify",
        "fcc-install",
        "path-update",
        "fcc-missing",
        "fcc-verify",
    ],
)
def test_install_ps1_stops_without_success_on_each_failure(
    powershell_harness: PowerShellHarness,
    failure: str,
) -> None:
    if failure in {"opencode-download", "opencode-archive"}:
        (powershell_harness.bin_dir / "opencode.cmd").unlink()
    result = powershell_harness.run(fail_step=failure)

    assert result.returncode != 0
    assert "Free Claude Code is installed and verified." not in result.stdout
    calls = powershell_harness.calls()
    if failure == "path-update":
        failure_index = calls.index("uv:tool update-shell")
        assert "uv:tool dir --bin" not in calls[failure_index + 1 :]

    forbidden = {
        "claude-download": "claude-install",
        "claude-install": "claude:--version",
        "claude-verify": "chatgpt.com",
        "codex-download": "codex-install",
        "codex-install": "codex:--version",
        "codex-verify": "pi.dev",
        "pi-download": "pi-install",
        "pi-install": "pi:--version",
        "pi-verify": "opencode:--version",
        "opencode-download": "opencode:--version",
        "opencode-archive": "opencode:--version",
        "opencode-verify": "npm:install -g cline",
        "cline-install": "cline:--version",
        "cline-verify": "hermes-agent.nousresearch.com",
        "uv-download": "uv-install",
        "uv-install": "uv:--version",
        "uv-verify": "uv:tool install",
        "fcc-install": "uv:tool update-shell",
        "fcc-missing": "fcc-server:--version",
    }.get(failure)
    if forbidden is not None:
        assert not any(forbidden in call for call in calls)


def test_install_ps1_dry_run_never_executes_commands(
    powershell_harness: PowerShellHarness,
) -> None:
    result = powershell_harness.run("-DryRun")

    assert result.returncode == 0, result.stderr
    assert powershell_harness.calls() == []
    assert "Dry run complete. No changes were made." in result.stdout
    assert "Free Claude Code is installed and verified." not in result.stdout


def test_install_ps1_rejects_broken_existing_client_without_replacing_it(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")

    result = powershell_harness.run(fail_step="claude-verify")

    assert result.returncode != 0
    calls = powershell_harness.calls()
    _assert_uv_ready_without_fcc_install(calls)
    assert not any("claude.ai" in call for call in calls)


def test_install_ps1_rejects_unparseable_existing_uv(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_client("pi")
    powershell_harness.add_uv("not-a-version")

    result = powershell_harness.run()

    assert result.returncode != 0
    assert not any("astral.sh" in call for call in powershell_harness.calls())


def test_install_ps1_voice_flags_only_change_fcc_spec(
    powershell_harness: PowerShellHarness,
) -> None:
    result = powershell_harness.run("-VoiceAll", "-TorchBackend", "cu130")

    assert result.returncode == 0, result.stderr
    assert any(
        '--torch-backend cu130 "free-claude-code[voice,voice_local] @ '
        'https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"'
        in call
        for call in powershell_harness.calls()
    )


@pytest.mark.parametrize("command_name", FCC_COMMANDS)
def test_install_ps1_rejects_running_fcc_before_mutation(
    powershell_harness: PowerShellHarness,
    command_name: str,
) -> None:
    powershell_harness.env["FCC_RUNNING_COMMAND"] = command_name

    result = powershell_harness.run()

    assert result.returncode != 0
    assert powershell_harness.calls() == []
    assert f"{command_name} (PID 4242)" in result.stderr


def test_install_ps1_rechecks_for_fcc_process_before_tool_replacement(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.add_client("claude")
    powershell_harness.add_client("codex")
    powershell_harness.add_client("pi")
    powershell_harness.add_uv("0.12.13")
    powershell_harness.env["FCC_RUNNING_COMMAND"] = "fcc-server"
    powershell_harness.env["FCC_RUNNING_PHASE"] = "late"

    result = powershell_harness.run()

    assert result.returncode != 0
    assert "fcc-server (PID 4242)" in result.stderr
    assert not any(
        "--refresh-package free-claude-code" in call
        for call in powershell_harness.calls()
    )


def test_install_ps1_ignores_similarly_named_process(
    powershell_harness: PowerShellHarness,
) -> None:
    powershell_harness.env["FCC_RUNNING_COMMAND"] = "fcc-server-helper"

    result = powershell_harness.run()

    assert result.returncode == 0, result.stderr


def test_installers_use_native_clients_and_single_python_selection() -> None:
    shell = (_repo_root() / "scripts" / "install.sh").read_text(encoding="utf-8")
    powershell = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")

    for text in (shell, powershell):
        for command_name in FCC_COMMANDS:
            assert command_name in text
        assert "@anthropic-ai/claude-code" not in text
        assert "@openai/codex" not in text
        assert "@earendil-works/pi-coding-agent" not in text
        assert "git+" not in text
        assert "git --version" not in text
        assert (
            "https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
            in text
        )
        assert "python install" not in text
        assert "--refresh-package" in text
        assert "tool update-shell" in text
        assert "--python" in text

    assert "https://pi.dev/install.sh" in shell
    assert "https://pi.dev/install.ps1" in powershell
    assert "https://x.ai/cli/install.sh" in shell
    assert "https://x.ai/cli/install.ps1" in powershell
    assert "https://dev.meta.ai/install.sh" in shell
    assert "https://dev.meta.ai/install.ps1" in powershell
    assert "main/scripts/install-muse.ps1" not in powershell


def test_install_ps1_uses_x64_python_for_windows_arm_compatibility() -> None:
    powershell = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")

    assert '$PythonRequest = "cpython-3.14.0-windows-x86_64-none"' in powershell


@pytest.mark.parametrize("powershell", _powershells())
def test_install_ps1_rejects_invalid_download_before_execution(
    powershell: str,
) -> None:
    text = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")
    body = _braced_body(text, "function Invoke-DownloadedPowerShellInstaller")
    script = f"""Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$DryRun = $false
function Format-Argument {{ param([string] $Value) return $Value }}
function Invoke-RestMethod {{
    [CmdletBinding()]
    param([string] $Uri, [string] $OutFile)
    [IO.File]::WriteAllText($OutFile, "<style>div#box {{")
}}
function Get-PowerShellExecutable {{ throw "invalid installer reached execution" }}
function Invoke-DownloadedPowerShellInstaller {{{body}}}
Invoke-DownloadedPowerShellInstaller `
    -Url "https://example.test/install.ps1" `
    -Name "Example"
"""

    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert (
        "Example installer from 'https://example.test/install.ps1' is not valid PowerShell"
        in result.stderr
    )
    assert "network proxy or filter" in result.stderr
    assert "invalid installer reached execution" not in result.stderr


@pytest.mark.parametrize("powershell", _powershells())
@pytest.mark.parametrize(
    ("answers", "expected", "expected_messages"),
    [
        (
            ("", "", "", "", "", "", "", "", "", "", ""),
            "True,True,True,True,False,True,True,True,True,True,False",
            (),
        ),
        (
            (
                "maybe",
                "n",
                "n",
                "n",
                "n",
                "n",
                "n",
                "n",
                "n",
                "n",
                "n",
                "n",
                "y",
                "n",
                "n",
                "n",
                "n",
                "n",
                "n",
                "n",
                "n",
                "y",
            ),
            "False,True,False,False,False,False,False,False,False,False,True",
            ("Please answer Y or N.", "Select at least one coding agent."),
        ),
    ],
)
def test_install_ps1_selects_at_least_one_coding_agent(
    powershell: str,
    answers: tuple[str, ...],
    expected: str,
    expected_messages: tuple[str, ...],
) -> None:
    text = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")
    read_yes_no = _braced_body(text, "function Read-YesNo")
    select_agents = _braced_body(text, "function Select-CodingAgents")
    answer_array = ", ".join(repr(answer) for answer in answers)
    script = f"""Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:Answers = @({answer_array})
$script:AnswerIndex = 0
$script:InstallClaudeCode = $true
$script:InstallCodex = $true
$script:InstallPi = $true
$script:InstallOpenCode = $true
$script:InstallCline = $false
$script:InstallHermes = $true
$script:InstallDsh = $true
$script:InstallGrok = $true
$script:InstallMuse = $true
$script:InstallAider = $true
$script:EnableRtk = $false
function Read-Host {{
    param([string] $Prompt)
    $answer = $script:Answers[$script:AnswerIndex]
    $script:AnswerIndex += 1
    return $answer
}}
function Read-YesNo {{{read_yes_no}}}
function Select-CodingAgents {{{select_agents}}}
Select-CodingAgents
Write-Output "selection:$($script:InstallClaudeCode),$($script:InstallCodex),$($script:InstallPi),$($script:InstallOpenCode),$($script:InstallCline),$($script:InstallHermes),$($script:InstallDsh),$($script:InstallGrok),$($script:InstallMuse),$($script:InstallAider),$($script:EnableRtk)"
"""

    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"selection:{expected}" in result.stdout
    for message in expected_messages:
        assert message in result.stdout


@pytest.mark.parametrize("powershell", _powershells())
def test_install_ps1_runs_only_selected_coding_agents(powershell: str) -> None:
    text = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")
    body = _braced_body(text, "function Ensure-SelectedCodingAgents")
    script = f"""Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:InstallClaudeCode = $false
$script:InstallCodex = $true
$script:InstallPi = $false
$script:InstallOpenCode = $false
$script:InstallCline = $false
$script:InstallHermes = $false
$script:InstallDsh = $false
$script:InstallGrok = $false
$script:InstallMuse = $false
$script:InstallAider = $false
$script:PiAvailable = $false
$script:MuseAvailable = $false
$script:Calls = @()
function Write-Step {{ param([string] $Message) }}
function Ensure-ClaudeCode {{ $script:Calls += "claude" }}
function Ensure-Codex {{ $script:Calls += "codex" }}
function Ensure-Pi {{ $script:Calls += "pi"; $script:PiAvailable = $true }}
function Ensure-OpenCode {{ $script:Calls += "opencode" }}
function Ensure-Cline {{ $script:Calls += "cline" }}
function Ensure-Hermes {{ $script:Calls += "hermes" }}
function Ensure-Dsh {{ $script:Calls += "dsh" }}
function Ensure-Grok {{ $script:Calls += "grok" }}
function Ensure-Muse {{ $script:Calls += "muse"; $script:MuseAvailable = $true }}
function Ensure-Aider {{ $script:Calls += "aider" }}
function Ensure-SelectedCodingAgents {{{body}}}
Ensure-SelectedCodingAgents
Write-Output "calls:$($script:Calls -join ',')"
"""

    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "calls:codex" in result.stdout


@pytest.mark.parametrize("powershell", _powershells())
def test_install_ps1_configures_rtk_only_for_available_selected_agents(
    powershell: str,
) -> None:
    text = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")
    body = _braced_body(text, "function Configure-RtkForSelectedAgents")
    script = f"""Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:EnableRtk = $true
$script:InstallClaudeCode = $false
$script:InstallCodex = $true
$script:InstallPi = $true
$script:InstallOpenCode = $false
$script:InstallCline = $false
$script:InstallHermes = $false
$script:InstallDsh = $false
$script:InstallGrok = $false
$script:InstallMuse = $false
$script:InstallAider = $false
$script:PiAvailable = $false
$script:MuseAvailable = $false
$script:Calls = @()
function Write-Step {{ param([string] $Message) }}
function Ensure-Rtk {{ $script:Calls += "ensure" }}
function Invoke-RtkCommand {{
    param([string[]] $Arguments)
    $script:Calls += ($Arguments -join " ")
}}
function Configure-RtkForSelectedAgents {{{body}}}
Configure-RtkForSelectedAgents
Write-Output "calls:$($script:Calls -join ',')"
"""

    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "calls:ensure,init --global --codex" in result.stdout


@pytest.mark.parametrize("powershell", _powershells())
def test_install_ps1_rejects_uninstalled_only_selection(powershell: str) -> None:
    text = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")
    body = _braced_body(text, "function Ensure-SelectedCodingAgents")
    script = f"""Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$script:InstallClaudeCode = $false
$script:InstallCodex = $false
$script:InstallPi = $true
$script:InstallOpenCode = $false
$script:InstallCline = $false
$script:InstallHermes = $false
$script:InstallDsh = $false
$script:InstallGrok = $false
$script:InstallMuse = $false
$script:InstallAider = $false
$script:PiAvailable = $false
$script:MuseAvailable = $false
function Write-Step {{ param([string] $Message) }}
function Ensure-ClaudeCode {{ }}
function Ensure-Codex {{ }}
function Ensure-Pi {{ }}
function Ensure-OpenCode {{ }}
function Ensure-Cline {{ }}
function Ensure-Hermes {{ }}
function Ensure-Dsh {{ }}
function Ensure-Grok {{ }}
function Ensure-Muse {{ }}
function Ensure-Aider {{ }}
function Ensure-SelectedCodingAgents {{{body}}}
Ensure-SelectedCodingAgents
"""

    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "No selected coding agent was installed." in result.stderr


@pytest.mark.parametrize("powershell", _powershells())
def test_install_ps1_falls_back_when_pshome_executable_is_unavailable(
    tmp_path: Path,
    powershell: str,
) -> None:
    text = (_repo_root() / "scripts" / "install.ps1").read_text(encoding="utf-8")
    body = _braced_body(text, "function Get-PowerShellExecutable")
    fallback = tmp_path / "fallback" / "powershell.exe"
    script = tmp_path / "test-powershell-resolution.ps1"
    script.write_text(
        f"""Set-StrictMode -Version Latest
function Get-ApplicationCommand {{
    param([string] $Name)
    return [pscustomobject] @{{ Source = {str(fallback)!r} }}
}}
function Get-PowerShellExecutable {{
{body}
}}
$resolved = Get-PowerShellExecutable -PowerShellHome {str(tmp_path / "missing")!r}
if ($resolved -ne {str(fallback)!r}) {{
    throw "Unexpected fallback: $resolved"
}}
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [powershell, "-NoProfile", "-File", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

#!/usr/bin/env sh
set -eu

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

show_usage() {
    cat <<'USAGE'
Usage: install-termux.sh

Installs Free Claude Code into a private Termux Python environment and adds
FCC command links to $PREFIX/bin. It does not install coding agents or change
FCC configuration.
USAGE
}

if [ "${1:-}" = "--help" ]; then
    show_usage
    exit 0
fi
[ "$#" -eq 0 ] || fail "Unknown argument: $1"
[ -n "${TERMUX_VERSION:-}" ] ||
    fail "This installer must run inside the Termux app (TERMUX_VERSION is missing)."
[ -n "${PREFIX:-}" ] || fail "Termux PREFIX must be set by the Termux shell."
[ -n "${HOME:-}" ] || fail "HOME must be set by the Termux shell."
[ -x "$PREFIX/bin/pkg" ] ||
    fail "Termux pkg was not found under PREFIX; use Termux's native environment."
FCC_INSTALL_ROOT="$HOME/.local/share/free-claude-code"
FCC_VENV="$FCC_INSTALL_ROOT/termux-venv"
FCC_UPDATER="$FCC_INSTALL_ROOT/fcc-update"
FCC_COMMANDS="fcc-desktop fcc-server fcc-doctor fcc-claude fcc-codex fcc-pi fcc-opencode fcc-cline fcc-hermes fcc-dsh fcc-grok fcc-muse fcc-agy fcc-aider"

check_owned_link() {
    link_path=$1
    expected_target=$2
    if [ -e "$link_path" ] || [ -L "$link_path" ]; then
        existing_target=$(readlink "$link_path" 2>/dev/null || true)
        [ "$existing_target" = "$expected_target" ] ||
            fail "Refusing to overwrite an existing command: $link_path"
    fi
}

case "$(uname -m)" in
    aarch64|arm64) ;;
    *) fail "Native Termux installation requires ARM64 (AArch64); found $(uname -m)." ;;
esac

for command_name in $FCC_COMMANDS; do
    check_owned_link "$PREFIX/bin/$command_name" "$FCC_VENV/bin/$command_name"
done
check_owned_link "$PREFIX/bin/fcc-update" "$FCC_UPDATER"
updater_marker="# Managed by scripts/install-termux.sh"
if [ -e "$FCC_UPDATER" ]; then
    [ -f "$FCC_UPDATER" ] &&
        [ "$(sed -n '2p' "$FCC_UPDATER")" = "$updater_marker" ] ||
        fail "Refusing to overwrite the existing updater: $FCC_UPDATER"
fi

printf 'Installing native Termux build/runtime packages...\n'
pkg install -y python uv clang make pkg-config libffi openssl rust

command -v uv >/dev/null 2>&1 || fail "uv installation failed; install it with: pkg install uv"
python_bin="$PREFIX/bin/python"
[ -x "$python_bin" ] || fail "Termux Python is required. Install it with: pkg install python"
"$python_bin" -c '
from pathlib import Path
import os
import sys

supported = (3, 14) <= sys.version_info[:2] < (3, 15)
get_api_level = getattr(sys, "getandroidapilevel", None)
try:
    api_level = int(get_api_level()) if callable(get_api_level) else 0
except (OSError, TypeError, ValueError):
    api_level = 0
prefix = Path(os.environ["PREFIX"]).resolve()
executable = Path(sys.executable).resolve()
if not supported:
    raise SystemExit("FCC requires Termux Python 3.14.x.")
if api_level < 1:
    raise SystemExit("Python does not report an Android API level; install native Termux Python.")
if not executable.is_relative_to(prefix):
    raise SystemExit("Python must be installed inside the Termux PREFIX.")
'
mkdir -p "$FCC_INSTALL_ROOT"
if [ -e "$FCC_VENV" ] && [ ! -x "$FCC_VENV/bin/python" ]; then
    fail "The FCC environment path exists but is not a usable virtual environment: $FCC_VENV"
fi

if [ ! -x "$FCC_VENV/bin/python" ]; then
    printf 'Creating the isolated FCC Python environment...\n'
    uv venv --python "$python_bin" "$FCC_VENV" ||
        fail "Could not create the FCC environment with Termux Python."
fi

CFLAGS="${CFLAGS:+$CFLAGS }-I$PREFIX/include"
LDFLAGS="${LDFLAGS:+$LDFLAGS }-L$PREFIX/lib"
PKG_CONFIG_PATH="${PKG_CONFIG_PATH:+$PKG_CONFIG_PATH:}$PREFIX/lib/pkgconfig:$PREFIX/share/pkgconfig"
OPENSSL_DIR=${OPENSSL_DIR:-$PREFIX}
CC=${CC:-clang}
export CFLAGS LDFLAGS PKG_CONFIG_PATH OPENSSL_DIR CC

printf 'Installing Free Claude Code and dependencies (native builds may take a while)...\n'
if ! UV_PYTHON_DOWNLOADS=never uv pip install \
    --python "$FCC_VENV/bin/python" \
    --upgrade free-claude-code
then
    fail "FCC or a dependency could not be built for Android/Bionic. Check Termux package versions and the compiler output above."
fi
uv pip check --python "$FCC_VENV/bin/python" ||
    fail "The installed FCC environment has dependency conflicts."
"$FCC_VENV/bin/fcc-server" --version >/dev/null ||
    fail "The installed fcc-server command did not start."
"$FCC_VENV/bin/fcc-doctor" --version >/dev/null ||
    fail "The installed fcc-doctor command did not start."

for command_name in $FCC_COMMANDS; do
    target="$FCC_VENV/bin/$command_name"
    [ -x "$target" ] || fail "The FCC package did not install $command_name."
    link="$PREFIX/bin/$command_name"
    if [ -e "$link" ] || [ -L "$link" ]; then
        existing_target=$(readlink "$link" 2>/dev/null || true)
        [ "$existing_target" = "$target" ] ||
            fail "Refusing to overwrite an existing command: $link"
    else
        ln -s "$target" "$link" ||
            fail "Could not add the FCC command to Termux PATH: $link"
    fi
done

updater_tmp=$(mktemp "$FCC_INSTALL_ROOT/.fcc-update.XXXXXX") ||
    fail "Could not create the FCC updater."
trap 'rm -f "$updater_tmp"' EXIT
cat >"$updater_tmp" <<'UPDATER'
#!/usr/bin/env sh
# Managed by scripts/install-termux.sh
set -eu
: "${HOME:?HOME is not set}"
venv="$HOME/.local/share/free-claude-code/termux-venv"
[ -x "$venv/bin/python" ] || {
    printf 'error: FCC Termux environment is missing; rerun scripts/install-termux.sh.\n' >&2
    exit 1
}
command -v uv >/dev/null 2>&1 || {
    printf 'error: uv is required for FCC updates; install it with: pkg install uv\n' >&2
    exit 1
}
exec uv pip install --python "$venv/bin/python" --upgrade free-claude-code
UPDATER
chmod 700 "$updater_tmp"
mv -f "$updater_tmp" "$FCC_UPDATER"
trap - EXIT

updater_link="$PREFIX/bin/fcc-update"
if [ -e "$updater_link" ] || [ -L "$updater_link" ]; then
    existing_target=$(readlink "$updater_link" 2>/dev/null || true)
    [ "$existing_target" = "$FCC_UPDATER" ] ||
        fail "Refusing to overwrite an existing command: $updater_link"
else
    ln -s "$FCC_UPDATER" "$updater_link" ||
        fail "Could not add fcc-update to Termux PATH."
fi

printf '\nFree Claude Code installed in the native Termux environment.\n'
printf 'Start the server with: fcc-server\n'
printf 'Run Termux checks with: fcc-doctor --termux\n'
printf 'Update later with: fcc-update (stop FCC first)\n'
printf 'FCC configuration remains in: %s/.fcc\n' "$HOME"

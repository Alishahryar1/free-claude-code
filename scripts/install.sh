#!/bin/sh
set -eu

REPO_ARCHIVE_URL="https://github.com/Alishahryar1/free-claude-code/archive/refs/heads/main.zip"
PYTHON_VERSION="3.14.0"
MIN_UV_VERSION="0.12.13"
CLAUDE_INSTALL_URL="https://claude.ai/install.sh"
CODEX_INSTALL_URL="https://chatgpt.com/codex/install.sh"
PI_INSTALL_URL="https://pi.dev/install.sh"
OPENCODE_INSTALL_URL="https://opencode.ai/v2/install"
HERMES_INSTALL_URL="https://hermes-agent.nousresearch.com/install.sh"
DSH_VERSION="0.1.0-rc.8"
DSH_PACKAGE="@deepseek-ai/dsh@$DSH_VERSION"
GROK_INSTALL_URL="https://x.ai/cli/install.sh"
MUSE_INSTALL_URL="https://dev.meta.ai/install.sh"
RTK_VERSION="0.44.2"
RTK_RELEASE_BASE_URL="https://github.com/rtk-ai/rtk/releases/download/v$RTK_VERSION"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
FCC_MACOS_BUNDLE_ID="io.github.alishahryar1.free-claude-code"
FCC_MACOS_OWNER_FILE=".free-claude-code-owner"
# Include retired entry points so updates reject older FCC processes before replacement.
FCC_COMMANDS="fcc-desktop fcc-server fcc-claude fcc-codex fcc-pi fcc-opencode fcc-cline fcc-hermes fcc-dsh fcc-grok fcc-muse fcc-aider fcc-update fcc-init free-claude-code"

dry_run=0
voice_local=0
install_claude=1
install_codex=1
install_pi=1
install_opencode=1
install_cline=0
install_hermes=1
install_dsh=1
install_grok=1
install_muse=1
install_aider=1
enable_rtk=0
torch_backend=""
temporary_file=""
temporary_binary=""
tool_bin=""
pi_available=0
rtk_path=""

show_usage() {
    cat <<'USAGE'
Usage: install.sh [options]

Installs or updates Free Claude Code and lets you choose which coding agents to install or verify.

Options:
  --voice-local            Install local Whisper voice transcription support.
  --torch-backend VALUE    Use a uv PyTorch backend, such as cu130. Requires local voice.
  --rtk                    Install and configure RTK for the selected coding agents.
  --dry-run                Print commands without running them.
  --help                   Show this help text.
USAGE
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

installer_is_interactive() {
    [ -t 1 ] && ( : </dev/tty ) 2>/dev/null
}

prompt_yes_no() {
    question=$1
    default_answer=${2:-yes}
    case "$default_answer" in
        yes) prompt='[Y/n]' ;;
        no) prompt='[y/N]' ;;
        *) fail "Unsupported prompt default: $default_answer" ;;
    esac

    while :; do
        printf '%s %s ' "$question" "$prompt" >&4
        if ! IFS= read -r answer <&3; then
            fail "Could not read the installer selection."
        fi
        case "$answer" in
            '')
                if [ "$default_answer" = "yes" ]; then
                    return 0
                fi
                return 1
                ;;
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            [Nn]|[Nn][Oo]) return 1 ;;
            *) printf 'Please answer Y or N.\n' >&4 ;;
        esac
    done
}

find_installed_coding_agent() (
    # Lookup may prepare search paths, but must not change the installer's state.
    case "$1" in
        pi|cline|dsh)
            add_npm_bin_directories
            ;;
        aider)
            if ! command -v aider >/dev/null 2>&1; then
                if [ -n "${UV_TOOL_BIN_DIR:-}" ]; then
                    add_path_entry "$UV_TOOL_BIN_DIR"
                elif [ -n "${XDG_BIN_HOME:-}" ]; then
                    add_path_entry "$XDG_BIN_HOME"
                elif [ -n "${XDG_DATA_HOME:-}" ]; then
                    add_path_entry "$XDG_DATA_HOME/../bin"
                elif [ -n "${HOME:-}" ]; then
                    add_path_entry "$HOME/.local/bin"
                fi
            fi
            ;;
    esac

    if [ "$1" = opencode ] && [ -n "$original_opencode_path" ]; then
        printf '%s\n' "$original_opencode_path"
        return 0
    fi
    command_path=$(command -v "$1" 2>/dev/null) || return 1
    if [ "$1" = pi ] && [ "$dry_run" -eq 0 ]; then
        pi_command_is_compatible || return 1
    fi
    printf '%s\n' "$command_path"
)

select_coding_agent() {
    if find_installed_coding_agent "$1" >/dev/null; then
        printf '%s already installed; will verify.\n' "$2" >&4
        return 0
    fi
    prompt_yes_no "Install $2 for $3?" "${4:-yes}"
}

choose_coding_agents() {
    selection_input=$1
    selection_output=$2
    exec 3<"$selection_input"
    exec 4>"$selection_output"

    while :; do
        if select_coding_agent claude "Claude Code" fcc-claude; then
            install_claude=1
        else
            install_claude=0
        fi
        if select_coding_agent codex Codex fcc-codex; then
            install_codex=1
        else
            install_codex=0
        fi
        if select_coding_agent pi Pi fcc-pi; then
            install_pi=1
        else
            install_pi=0
        fi
        if select_coding_agent opencode OpenCode fcc-opencode; then
            install_opencode=1
        else
            install_opencode=0
        fi

        if [ "$install_cline" -eq 1 ]; then
            cline_default=yes
        else
            cline_default=no
        fi
        if select_coding_agent cline "Cline CLI" fcc-cline "$cline_default"; then
            install_cline=1
        else
            install_cline=0
        fi

        if [ "$install_hermes" -eq 1 ]; then
            hermes_default=yes
        else
            hermes_default=no
        fi
        if select_coding_agent hermes "Hermes Agent" fcc-hermes "$hermes_default"; then
            install_hermes=1
        else
            install_hermes=0
        fi

        if [ "$install_dsh" -eq 1 ]; then
            dsh_default=yes
        else
            dsh_default=no
        fi
        if select_coding_agent dsh "DeepSeek Harness" fcc-dsh "$dsh_default"; then
            install_dsh=1
        else
            install_dsh=0
        fi

        if [ "$install_grok" -eq 1 ]; then
            grok_default=yes
        else
            grok_default=no
        fi
        if select_coding_agent grok "Grok Build" fcc-grok "$grok_default"; then
            install_grok=1
        else
            install_grok=0
        fi

        if [ "$install_muse" -eq 1 ]; then
            muse_default=yes
        else
            muse_default=no
        fi
        if select_coding_agent muse "Muse Code" fcc-muse "$muse_default"; then
            install_muse=1
        else
            install_muse=0
        fi

        if [ "$install_aider" -eq 1 ]; then
            aider_default=yes
        else
            aider_default=no
        fi
        if select_coding_agent aider Aider fcc-aider "$aider_default"; then
            install_aider=1
        else
            install_aider=0
        fi

        if [ "$install_claude" -eq 1 ] || [ "$install_codex" -eq 1 ] || [ "$install_pi" -eq 1 ] || [ "$install_opencode" -eq 1 ] || [ "$install_cline" -eq 1 ] || [ "$install_hermes" -eq 1 ] || [ "$install_dsh" -eq 1 ] || [ "$install_grok" -eq 1 ] || [ "$install_muse" -eq 1 ] || [ "$install_aider" -eq 1 ]; then
            break
        fi
        printf 'Select at least one coding agent.\n\n' >&4
    done

    if [ "$enable_rtk" -eq 0 ]; then
        if command -v rtk >/dev/null 2>&1; then
            printf 'RTK already installed; will verify.\n' >&4
            enable_rtk=1
        elif prompt_yes_no "Enable RTK token optimization globally for the selected coding agents?" no; then
            enable_rtk=1
        fi
    fi

    exec 3<&-
    exec 4>&-
}

step() {
    printf '\n==> %s\n' "$1"
}

quote_arg() {
    case "$1" in
        *[!A-Za-z0-9_./:@%+=,-]*|"")
            escaped=$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')
            printf '"%s"' "$escaped"
            ;;
        *)
            printf '%s' "$1"
            ;;
    esac
}

print_command() {
    printf '+'
    for arg in "$@"; do
        printf ' '
        quote_arg "$arg"
    done
    printf '\n'
}

run() {
    print_command "$@"
    if [ "$dry_run" -eq 1 ]; then
        return 0
    fi

    if "$@"; then
        return 0
    else
        status=$?
    fi

    fail "Command failed with exit code $status: $1"
}

cleanup() {
    if [ -n "$temporary_file" ] && [ -e "$temporary_file" ]; then
        rm -f "$temporary_file"
    fi
    if [ -n "$temporary_binary" ] && [ -e "$temporary_binary" ]; then
        rm -f "$temporary_binary"
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM

add_path_entry() {
    [ -n "$1" ] || return 0
    case ":$PATH:" in
        *":$1:"*) ;;
        *) PATH="$1:$PATH" ;;
    esac
}

prioritize_path_entry() {
    [ -n "$1" ] || return 0
    PATH="$1:$PATH"
    export PATH
    hash -r 2>/dev/null || true
}

add_known_bin_directories() {
    if [ -n "${XDG_BIN_HOME:-}" ]; then
        add_path_entry "$XDG_BIN_HOME"
    fi

    if [ -n "${HOME:-}" ]; then
        add_path_entry "$HOME/.local/bin"
        add_path_entry "$HOME/.cargo/bin"
        add_path_entry "$HOME/.opencode/bin"
        add_path_entry "${XDG_DATA_HOME:-$HOME/.local/share}/pi-node/current/bin"
    fi

    if [ -n "${GROK_BIN_DIR:-}" ]; then
        add_path_entry "$GROK_BIN_DIR"
    elif [ -n "${HOME:-}" ]; then
        add_path_entry "$HOME/.grok/bin"
    fi

    export PATH
    hash -r 2>/dev/null || true
}

add_uv_tool_bin_directory() {
    print_command uv tool dir --bin
    if tool_bin=$(uv tool dir --bin); then
        :
    else
        status=$?
        fail "Could not determine the uv tool bin directory (exit code $status)."
    fi
    [ -n "$tool_bin" ] || fail "uv returned an empty tool bin directory."

    add_path_entry "$tool_bin"
    export PATH
    hash -r 2>/dev/null || true
}

add_npm_bin_directories() {
    [ "$dry_run" -eq 0 ] || return 0
    add_known_bin_directories
    if command -v npm >/dev/null 2>&1; then
        pi_npm_prefix=$(npm prefix -g 2>/dev/null || npm config get prefix 2>/dev/null || true)
        if [ -n "$pi_npm_prefix" ]; then
            add_path_entry "$pi_npm_prefix/bin"
            export PATH
            hash -r 2>/dev/null || true
        fi
    fi
}

fcc_process_ids() {
    command_name=$1

    if command -v pgrep >/dev/null 2>&1; then
        {
            pgrep -x "$command_name" 2>/dev/null || true
            pgrep -f "(^|/)${command_name}([[:space:]]|$)" 2>/dev/null || true
        } | sort -nu
        return 0
    fi

    ps -A -o pid= -o args= 2>/dev/null |
        awk -v command_name="$command_name" '
            BEGIN {
                pattern = "(^|/)" command_name "([[:space:]]|$)"
            }
            {
                process_id = $1
                sub(/^[[:space:]]*[0-9]+[[:space:]]+/, "")
                if ($0 ~ pattern) {
                    print process_id
                }
            }
        ' || true
}

assert_no_fcc_processes_running() {
    running=""
    for command_name in $FCC_COMMANDS; do
        process_ids=$(fcc_process_ids "$command_name")
        [ -n "$process_ids" ] || continue

        for process_id in $process_ids; do
            process="$command_name (PID $process_id)"
            if [ -n "$running" ]; then
                running="$running, $process"
            else
                running=$process
            fi
        done
    done

    if [ -n "$running" ]; then
        fail "Free Claude Code is still running ($running). Stop those processes, then rerun the installer."
    fi
}

require_command() {
    if [ "$dry_run" -eq 0 ] && ! command -v "$1" >/dev/null 2>&1; then
        fail "$1 is required. Install it first, then rerun this installer."
    fi
}

download_and_run() {
    url=$1
    interpreter=$2
    label=$3
    shift 3
    non_interactive=0
    if [ "$#" -gt 0 ]; then
        non_interactive=$1
        shift
    fi

    if [ "$dry_run" -eq 1 ]; then
        print_command curl -fsSL "$url" -o "<temporary-script>"
        if [ "$non_interactive" -eq 1 ]; then
            printf '+ CODEX_NON_INTERACTIVE=1 '
            quote_arg "$interpreter"
            printf ' <temporary-script>'
            for arg in "$@"; do
                printf ' '
                quote_arg "$arg"
            done
            printf '\n'
        else
            print_command "$interpreter" "<temporary-script>" "$@"
        fi
        return 0
    fi

    temporary_file=$(mktemp "${TMPDIR:-/tmp}/fcc-install.XXXXXX") || fail "Unable to create a temporary file for $label."
    print_command curl -fsSL "$url" -o "$temporary_file"
    if curl -fsSL "$url" -o "$temporary_file"; then
        :
    else
        status=$?
        fail "Could not download the $label installer (curl exit code $status)."
    fi

    if [ ! -s "$temporary_file" ]; then
        fail "The downloaded $label installer was empty."
    fi

    if [ "$non_interactive" -eq 1 ]; then
        printf '+ CODEX_NON_INTERACTIVE=1 '
        quote_arg "$interpreter"
        printf ' '
        quote_arg "$temporary_file"
        for arg in "$@"; do
            printf ' '
            quote_arg "$arg"
        done
        printf '\n'
        if CODEX_NON_INTERACTIVE=1 "$interpreter" "$temporary_file" "$@"; then
            :
        else
            status=$?
            fail "$label installation failed with exit code $status."
        fi
    else
        print_command "$interpreter" "$temporary_file" "$@"
        if "$interpreter" "$temporary_file" "$@"; then
            :
        else
            status=$?
            fail "$label installation failed with exit code $status."
        fi
    fi

    rm -f "$temporary_file"
    temporary_file=""
}

verify_command() {
    command_name=$1
    display_name=$2

    if [ "$dry_run" -eq 1 ]; then
        print_command "$command_name" --version
        return 0
    fi

    command_path=$(command -v "$command_name" 2>/dev/null) || fail "$display_name was installed, but '$command_name' is not available on PATH."
    run "$command_path" --version
}

pi_command_is_compatible() {
    pi_command_path=$(command -v pi 2>/dev/null) || return 1
    pi_help=$("$pi_command_path" --help 2>/dev/null) || return 1
    case "$pi_help" in
        *--extension*) ;;
        *) return 1 ;;
    esac
    case "$pi_help" in
        *--models*) return 0 ;;
        *) return 1 ;;
    esac
}

verify_pi_command() {
    if [ "$dry_run" -eq 1 ]; then
        printf '+ pi --help (verify --extension and --models support)\n'
        print_command pi --version
        return 0
    fi

    pi_command_path=$(command -v pi 2>/dev/null) || fail "Pi was installed, but 'pi' is not available on PATH."
    pi_command_is_compatible || fail "The 'pi' command at $pi_command_path is not a compatible Pi Coding Agent."
    run "$pi_command_path" --version
}

verify_rtk_command() {
    if [ "$dry_run" -eq 1 ]; then
        print_command env RTK_TELEMETRY_DISABLED=1 rtk --version
        print_command env RTK_TELEMETRY_DISABLED=1 rtk gain
        return 0
    fi

    rtk_path=$(command -v rtk 2>/dev/null) || fail "RTK was installed, but 'rtk' is not available on PATH."
    print_command env RTK_TELEMETRY_DISABLED=1 "$rtk_path" --version
    if ! RTK_TELEMETRY_DISABLED=1 "$rtk_path" --version; then
        fail "The 'rtk' command at $rtk_path is not a compatible Rust Token Killer installation. Remove the conflicting command from PATH, then rerun the installer."
    fi

    print_command env RTK_TELEMETRY_DISABLED=1 "$rtk_path" gain
    if ! RTK_TELEMETRY_DISABLED=1 "$rtk_path" gain; then
        fail "The 'rtk' command at $rtk_path is not a compatible Rust Token Killer installation. Remove the conflicting command from PATH, then rerun the installer."
    fi
}

select_rtk_release() {
    rtk_platform=$(uname -s)
    rtk_architecture=$(uname -m)
    case "$rtk_platform:$rtk_architecture" in
        Linux:x86_64|Linux:amd64)
            rtk_asset_name="rtk-x86_64-unknown-linux-musl.tar.gz"
            rtk_asset_sha256="d94cc2a3e57fa534892b5235a726e7eeb7523f205a5f8f48f853bfcae7be7e33"
            ;;
        Linux:aarch64|Linux:arm64)
            rtk_asset_name="rtk-aarch64-unknown-linux-gnu.tar.gz"
            rtk_asset_sha256="5cd3f7fa2697faf9e5b77a10ce4e699006e02d4752d792f06550697eb4b8e8a9"
            ;;
        Darwin:x86_64|Darwin:amd64)
            rtk_asset_name="rtk-x86_64-apple-darwin.tar.gz"
            rtk_asset_sha256="636f808db86b2cefab7db7dd9393da8b6e4721bb2ffaa0644e3ffa52d3420d81"
            ;;
        Darwin:aarch64|Darwin:arm64)
            rtk_asset_name="rtk-aarch64-apple-darwin.tar.gz"
            rtk_asset_sha256="b7c2218eca538b54e63fa594a8ce58bd3716851b01b3b0dc026515323baf6393"
            ;;
        *)
            fail "RTK $RTK_VERSION does not provide a release for $rtk_platform $rtk_architecture."
            ;;
    esac
}

install_rtk() {
    select_rtk_release
    rtk_archive_url="$RTK_RELEASE_BASE_URL/$rtk_asset_name"
    if [ "$dry_run" -eq 1 ]; then
        print_command curl -fsSL "$rtk_archive_url" -o "<temporary-archive>"
        printf '+ verify pinned SHA-256 for %s\n' "$rtk_asset_name"
        printf '+ extract rtk to %s\n' "${HOME:-~}/.local/bin/rtk"
        return 0
    fi

    [ -n "${HOME:-}" ] || fail "HOME is required to install RTK."
    temporary_file=$(mktemp "${TMPDIR:-/tmp}/fcc-rtk.XXXXXX") || fail "Unable to create a temporary RTK archive."
    print_command curl -fsSL "$rtk_archive_url" -o "$temporary_file"
    if curl -fsSL "$rtk_archive_url" -o "$temporary_file"; then
        :
    else
        status=$?
        fail "Could not download RTK $RTK_VERSION (curl exit code $status)."
    fi
    [ -s "$temporary_file" ] || fail "The downloaded RTK archive was empty."

    if command -v sha256sum >/dev/null 2>&1; then
        print_command sha256sum "$temporary_file"
        rtk_actual_sha256=$(sha256sum "$temporary_file") || fail "Could not hash the downloaded RTK archive."
    elif command -v shasum >/dev/null 2>&1; then
        print_command shasum -a 256 "$temporary_file"
        rtk_actual_sha256=$(shasum -a 256 "$temporary_file") || fail "Could not hash the downloaded RTK archive."
    else
        fail "RTK installation requires sha256sum or shasum for checksum verification."
    fi
    rtk_actual_sha256=${rtk_actual_sha256%% *}
    [ "$rtk_actual_sha256" = "$rtk_asset_sha256" ] || fail "RTK checksum verification failed for $rtk_asset_name."

    if rtk_archive_entries=$(tar -tzf "$temporary_file"); then
        :
    else
        fail "The verified RTK archive could not be inspected."
    fi
    [ "$rtk_archive_entries" = "rtk" ] || fail "The verified RTK archive did not contain exactly one root rtk executable."

    rtk_install_directory="$HOME/.local/bin"
    run mkdir -p "$rtk_install_directory"
    temporary_binary=$(mktemp "$rtk_install_directory/.rtk.XXXXXX") || fail "Unable to create a temporary RTK executable."
    print_command tar -xOzf "$temporary_file" rtk
    if tar -xOzf "$temporary_file" rtk >"$temporary_binary"; then
        :
    else
        fail "The verified RTK archive could not be extracted."
    fi
    [ -s "$temporary_binary" ] || fail "The verified RTK executable was empty."
    run chmod +x "$temporary_binary"
    run mv "$temporary_binary" "$rtk_install_directory/rtk"
    temporary_binary=""
    rm -f "$temporary_file"
    temporary_file=""
}

ensure_rtk() {
    if command -v rtk >/dev/null 2>&1; then
        printf 'RTK already found on PATH; verifying it without updating it.\n'
    else
        install_rtk
        add_known_bin_directories
    fi

    verify_rtk_command
}

run_rtk_init() {
    print_command env RTK_TELEMETRY_DISABLED=1 rtk "$@"
    if [ "$dry_run" -eq 1 ]; then
        return 0
    fi

    if RTK_TELEMETRY_DISABLED=1 "$rtk_path" "$@"; then
        return 0
    else
        status=$?
    fi

    fail "RTK configuration failed with exit code $status. Correct the reported RTK error, then rerun the installer."
}

ensure_rtk_claude_config_directory() {
    if [ -n "${CLAUDE_CONFIG_DIR:-}" ]; then
        rtk_claude_config_directory=$CLAUDE_CONFIG_DIR
    else
        [ -n "${HOME:-}" ] || fail "HOME is required to configure RTK for Claude Code."
        rtk_claude_config_directory="$HOME/.claude"
    fi
    run mkdir -p "$rtk_claude_config_directory"
}

configure_rtk_for_selected_agents() {
    [ "$enable_rtk" -eq 1 ] || return 0

    step "Installing and configuring RTK token optimization"
    ensure_rtk

    if [ "$install_claude" -eq 1 ]; then
        ensure_rtk_claude_config_directory
        run_rtk_init init --global --auto-patch
    fi
    if [ "$install_codex" -eq 1 ]; then
        run_rtk_init init --global --codex
    fi
    if [ "$install_pi" -eq 1 ] && [ "$pi_available" -eq 1 ]; then
        run_rtk_init init --global --agent pi
    fi
    if [ "$install_cline" -eq 1 ]; then
        printf 'Optional for each project: cd <project> && RTK_TELEMETRY_DISABLED=1 rtk init --agent cline\n'
    fi
}

ensure_claude() {
    if command -v claude >/dev/null 2>&1; then
        printf 'Claude Code already found on PATH; verifying it.\n'
    else
        download_and_run "$CLAUDE_INSTALL_URL" bash "Claude Code"
        add_known_bin_directories
    fi

    verify_command claude "Claude Code"
}

ensure_codex() {
    if command -v codex >/dev/null 2>&1; then
        printf 'Codex already found on PATH; verifying it.\n'
    else
        download_and_run "$CODEX_INSTALL_URL" sh "Codex" 1
        add_known_bin_directories
    fi

    verify_command codex "Codex"
}

ensure_pi() {
    pi_available=0
    add_npm_bin_directories
    existing_pi_path=$(command -v pi 2>/dev/null || true)

    if [ "$dry_run" -eq 1 ] && command -v pi >/dev/null 2>&1; then
        printf 'Pi already found on PATH; verifying it.\n'
    elif pi_command_is_compatible; then
        printf 'Pi already found on PATH; verifying it.\n'
    else
        if [ -n "$existing_pi_path" ]; then
            printf "The existing 'pi' command at %s is not Pi Coding Agent; installing Pi.\n" "$existing_pi_path"
        fi
        download_and_run "$PI_INSTALL_URL" sh "Pi"
        add_npm_bin_directories

        if [ "$dry_run" -eq 0 ]; then
            current_pi_path=$(command -v pi 2>/dev/null || true)
            if [ -z "$current_pi_path" ] ||
                { [ -n "$existing_pi_path" ] &&
                    [ "$current_pi_path" = "$existing_pi_path" ] &&
                    ! pi_command_is_compatible; }; then
                printf 'Pi was not installed; continuing without it.\n'
                return 0
            fi
        fi
    fi

    verify_pi_command
    pi_available=1
}

opencode_version() {
    opencode_output=$("$1" --version) || return 1
    printf '%s\n' "$opencode_output" | sed -nE 's/^[[:space:]]*(opencode([[:space:]]+version)?[[:space:]]+)?v?([0-9]+\.[0-9]+\.[0-9]+)(\+[0-9A-Za-z.-]+)?[[:space:]]*$/\3\4/p'
}

opencode_rtk_plugin() {
    opencode_plugin_path="$HOME/.config/opencode/plugins/rtk.ts"
    if [ ! -e "$opencode_plugin_path" ] && [ ! -L "$opencode_plugin_path" ]; then
        return 0
    fi
    [ -f "$opencode_plugin_path" ] && [ ! -L "$opencode_plugin_path" ] ||
        fail "Disable or migrate the RTK plugin at $opencode_plugin_path manually, then rerun the installer."
    # The backup lives in .config/opencode; check its parents as well as the
    # plugin directory so a linked parent cannot redirect either mutation.
    for opencode_parent in "$HOME/.config" "$HOME/.config/opencode" "$HOME/.config/opencode/plugins"; do
        [ ! -L "$opencode_parent" ] ||
            fail "The RTK plugin directory is linked: $opencode_parent. Disable or migrate it manually, then rerun the installer."
    done
    if command -v sha256sum >/dev/null 2>&1; then
        opencode_plugin_hash=$(sha256sum "$opencode_plugin_path") || return 1
    elif command -v shasum >/dev/null 2>&1; then
        opencode_plugin_hash=$(shasum -a 256 "$opencode_plugin_path") || return 1
    else
        fail "Checking the old OpenCode RTK plugin requires sha256sum or shasum."
    fi
    [ "${opencode_plugin_hash%% *}" = "6530c131946c84892f9522abd68d4e513e1e658d8ddbad1f59388c86ebbcb6bb" ] ||
        fail "The RTK plugin at $opencode_plugin_path was modified or is unrecognized. Disable or migrate it manually, then rerun the installer."
    printf '%s\n' "$opencode_plugin_path"
}

assert_no_opencode_processes_running() {
    [ -z "$(fcc_process_ids opencode)$(fcc_process_ids opencode2)" ] ||
        fail "Close OpenCode before replacing its executable or RTK plugin, then rerun the installer."
}

run_opencode_installer() {
    # Recheck after the script download. Upstream owns the actual installation.
    assert_no_opencode_processes_running
    VERSION= bash "$@"
}

ensure_opencode() {
    [ -n "${HOME:-}" ] || fail "HOME is required to install OpenCode."
    opencode_native="$HOME/.opencode/bin/opencode"
    opencode_path=${original_opencode_path:-$(command -v opencode || true)}
    if [ "$dry_run" -eq 1 ]; then
        print_command opencode --version
        printf 'Install stable OpenCode 2 if absent, or migrate v1 at %s; external v1 requires manual upgrade.\n' "$opencode_native"
        printf 'Check and back up the recognized old OpenCode RTK plugin if present.\n'
        if [ -z "$opencode_path" ]; then
            download_and_run "$OPENCODE_INSTALL_URL" bash "OpenCode"
        fi
        return 0
    fi

    opencode_install=1
    if [ -n "$opencode_path" ]; then
        opencode_current=$(opencode_version "$opencode_path") ||
            fail "Could not read OpenCode version at $opencode_path. Correct that installation, then rerun the installer."
        case "$opencode_current" in
            2.*) opencode_install=0 ;;
            1.*)
                [ "$opencode_path" = "$opencode_native" ] ||
                    fail "OpenCode 1 at $opencode_path requires manual migration. Remove it with its package manager (npm: npm uninstall -g opencode-ai), then rerun this installer. See https://opencode.ai/v2/docs/migrate-v1/"
                ;;
            *) fail "OpenCode at $opencode_path is not a recognized stable v1 or v2. Correct that installation, then rerun the installer. See https://opencode.ai/v2/docs/migrate-v1/" ;;
        esac
    fi
    opencode_plugin=$(opencode_rtk_plugin) || return $?
    if [ "$opencode_install" -eq 1 ] || [ -n "$opencode_plugin" ]; then
        assert_no_opencode_processes_running
    fi
    if [ "$opencode_install" -eq 1 ]; then
        for opencode_target in "$HOME/.opencode" "$HOME/.opencode/bin" "$opencode_native"; do
            [ ! -L "$opencode_target" ] || fail "OpenCode installation path is linked: $opencode_target. Migrate it manually."
        done
        download_and_run "$OPENCODE_INSTALL_URL" run_opencode_installer "OpenCode"
        add_known_bin_directories
        hash -r 2>/dev/null || true
        opencode_installed=$(opencode_version "$opencode_native") || fail "Could not verify installed OpenCode at $opencode_native."
        case "$opencode_installed" in
            2.*) ;;
            *) fail "The OpenCode installer did not install stable OpenCode 2. See https://opencode.ai/v2/docs/" ;;
        esac
    fi
    # Check the command selected after the installer's PATH additions.
    hash -r 2>/dev/null || true
    opencode_path=$(command -v opencode || true)
    [ -n "$opencode_path" ] || fail "OpenCode is not available on PATH after installation."
    opencode_current=$(opencode_version "$opencode_path") || fail "Could not verify OpenCode at $opencode_path."
    case "$opencode_current" in
        2.*) printf 'Verified OpenCode %s at %s.\n' "$opencode_current" "$opencode_path" ;;
        *) fail "OpenCode at $opencode_path is not stable OpenCode 2. Correct PATH, then rerun the installer." ;;
    esac
    if [ -n "$opencode_plugin" ]; then
        opencode_plugin=$(opencode_rtk_plugin) || return $?
        [ -n "$opencode_plugin" ] || return 0
        assert_no_opencode_processes_running
        opencode_backup=$(mktemp "$HOME/.config/opencode/rtk-v1-XXXXXX") || fail "Could not create an RTK plugin backup."
        if mv "$opencode_plugin" "$opencode_backup"; then
            printf 'OpenCode 2 RTK support is unavailable; the old plugin was saved at %s.\n' "$opencode_backup"
        else
            rm -f "$opencode_backup"
            fail "Could not disable the old RTK plugin at $opencode_plugin."
        fi
    fi
}

ensure_cline() {
    add_npm_bin_directories

    if command -v cline >/dev/null 2>&1; then
        printf 'Cline already found on PATH; verifying it.\n'
    else
        command -v npm >/dev/null 2>&1 || fail "Cline installation requires npm. Install Node.js from https://nodejs.org/en/download, then rerun the installer."
        run npm install -g cline
        add_npm_bin_directories
    fi

    verify_command cline "Cline"
}

hermes_platform_is_supported() {
    hermes_platform=$(uname -s)
    hermes_architecture=$(uname -m)
    case "$hermes_platform:$hermes_architecture" in
        Linux:x86_64|Linux:amd64|Linux:aarch64|Linux:arm64|Darwin:aarch64|Darwin:arm64)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

confirm_hermes_platform() {
    if hermes_platform_is_supported; then
        return 0
    fi
    fail "Hermes Agent does not provide a supported release for $hermes_platform $hermes_architecture."
}

install_hermes() {
    confirm_hermes_platform
    download_and_run "$HERMES_INSTALL_URL" bash "Hermes Agent" 0 --non-interactive --skip-setup
    add_known_bin_directories
}

ensure_hermes() {
    if command -v hermes >/dev/null 2>&1; then
        printf 'Hermes Agent already found on PATH; verifying it.\n'
    else
        install_hermes
    fi

    verify_command hermes "Hermes Agent"
}

current_dsh_version() {
    if output=$(dsh --version 2>/dev/null); then
        :
    else
        return 1
    fi

    version=$(printf '%s\n' "$output" | awk '
        match($0, /[0-9]+\.[0-9]+\.[0-9]+-[0-9A-Za-z][0-9A-Za-z.-]*/) {
            print substr($0, RSTART, RLENGTH)
            exit
        }
    ')
    [ -n "$version" ] || return 1
    printf '%s\n' "$version"
}

current_node_version() {
    if output=$(node --version 2>/dev/null); then
        :
    else
        return 1
    fi

    version=$(printf '%s\n' "$output" | awk '
        match($0, /[0-9]+\.[0-9]+\.[0-9]+/) {
            print substr($0, RSTART, RLENGTH)
            exit
        }
    ')
    [ -n "$version" ] || return 1
    printf '%s\n' "$version"
}

dsh_node_version_is_supported() {
    version=$1
    major=${version%%.*}
    rest=${version#*.}
    [ "$rest" != "$version" ] || return 1
    minor=${rest%%.*}
    case "$major:$minor" in
        *[!0-9:]*|:*) return 1 ;;
    esac
    if [ "$major" -eq 22 ]; then
        [ "$minor" -ge 19 ]
        return
    fi
    [ "$major" -ge 24 ]
}

dsh_toolchain_is_supported() {
    command -v node >/dev/null 2>&1 || return 1
    command -v npm >/dev/null 2>&1 || return 1
    version=$(current_node_version) || return 1
    dsh_node_version_is_supported "$version"
}

require_dsh_toolchain() {
    command -v node >/dev/null 2>&1 || fail "DeepSeek Harness requires Node.js ^22.19.0 or >=24.0.0 and npm. Install Node.js, then rerun the installer."
    command -v npm >/dev/null 2>&1 || fail "DeepSeek Harness requires npm. Install npm, then rerun the installer."
    version=$(current_node_version) || fail "DeepSeek Harness requires a readable Node.js version."
    dsh_node_version_is_supported "$version" || fail "DeepSeek Harness requires Node.js ^22.19.0 or >=24.0.0; found Node.js $version."
}

verify_dsh_command() {
    if [ "$dry_run" -eq 1 ]; then
        print_command dsh --version
        return 0
    fi

    command -v dsh >/dev/null 2>&1 || fail "DeepSeek Harness was installed, but 'dsh' is not available on PATH."
    version=$(current_dsh_version) || fail "DeepSeek Harness is present, but 'dsh --version' did not return its preview semantic version."
    [ "$version" = "$DSH_VERSION" ] || fail "DeepSeek Harness $DSH_VERSION is required; found $version after installation."
    printf 'Verified DeepSeek Harness %s.\n' "$version"
}

install_dsh_package() {
    require_dsh_toolchain
    run npm install -g "$DSH_PACKAGE"
    add_npm_bin_directories
}

ensure_dsh() {
    add_npm_bin_directories

    if [ "$dry_run" -eq 1 ]; then
        if command -v dsh >/dev/null 2>&1; then
            print_command dsh --version
            printf 'The exact supported DeepSeek Harness preview will be preserved; another version will be replaced.\n'
        else
            command -v node >/dev/null 2>&1 || fail "DeepSeek Harness requires Node.js ^22.19.0 or >=24.0.0 and npm. Install Node.js, then rerun the installer."
            command -v npm >/dev/null 2>&1 || fail "DeepSeek Harness requires npm. Install npm, then rerun the installer."
            print_command npm install -g "$DSH_PACKAGE"
        fi
        verify_dsh_command
        return 0
    fi

    require_dsh_toolchain
    if command -v dsh >/dev/null 2>&1; then
        version=$(current_dsh_version) || fail "DeepSeek Harness is present, but 'dsh --version' did not return its preview semantic version."
        if [ "$version" = "$DSH_VERSION" ]; then
            printf 'DeepSeek Harness %s already matches the supported preview; leaving it unchanged.\n' "$version"
            return 0
        fi
        printf 'DeepSeek Harness %s does not match %s; replacing it with the supported preview.\n' "$version" "$DSH_VERSION"
    fi

    install_dsh_package
    verify_dsh_command
}

install_grok_build() {
    download_and_run "$GROK_INSTALL_URL" bash "Grok Build"
    add_known_bin_directories
}

ensure_grok() {
    if command -v grok >/dev/null 2>&1; then
        printf 'Grok Build already found on PATH; verifying it.\n'
    else
        install_grok_build
    fi

    verify_command grok "Grok Build"
}

install_muse_code() {
    case "$(uname -s)" in
        Darwin|Linux) ;;
        *) fail "Meta's official Muse Code installer supports macOS, Linux, and WSL only." ;;
    esac
    download_and_run "$MUSE_INSTALL_URL" bash "Muse Code"
    add_known_bin_directories
}

ensure_muse() {
    if command -v muse >/dev/null 2>&1; then
        printf 'Muse Code already found on PATH; verifying it.\n'
    else
        install_muse_code
    fi

    verify_command muse "Muse Code"
}

install_aider_cli() {
    run uv tool install --force --python python3.12 --with pip aider-chat@latest
}

ensure_aider() {
    if ! command -v aider >/dev/null 2>&1 && [ "$dry_run" -eq 0 ]; then
        add_uv_tool_bin_directory
    fi

    if command -v aider >/dev/null 2>&1; then
        printf 'Aider already found on PATH; verifying it.\n'
    else
        install_aider_cli
    fi

    verify_command aider "Aider"
}

ensure_selected_coding_agents() {
    if [ "$install_claude" -eq 1 ]; then
        step "Ensuring Claude Code is installed"
        ensure_claude
    fi

    if [ "$install_codex" -eq 1 ]; then
        step "Ensuring Codex is installed"
        ensure_codex
    fi

    if [ "$install_pi" -eq 1 ]; then
        step "Checking or installing Pi"
        ensure_pi
    fi

    if [ "$install_opencode" -eq 1 ]; then
        step "Ensuring OpenCode is installed"
        ensure_opencode
    fi

    if [ "$install_cline" -eq 1 ]; then
        step "Ensuring Cline CLI is installed"
        ensure_cline
    fi

    if [ "$install_hermes" -eq 1 ]; then
        step "Ensuring Hermes Agent is installed"
        ensure_hermes
    fi

    if [ "$install_dsh" -eq 1 ]; then
        step "Ensuring DeepSeek Harness is installed"
        ensure_dsh
    fi

    if [ "$install_grok" -eq 1 ]; then
        step "Ensuring Grok Build is installed"
        ensure_grok
    fi

    if [ "$install_muse" -eq 1 ]; then
        step "Ensuring Muse Code is installed"
        ensure_muse
    fi

    if [ "$install_aider" -eq 1 ]; then
        step "Ensuring Aider is installed"
        ensure_aider
    fi

    if [ "$install_claude" -eq 0 ] && [ "$install_codex" -eq 0 ] && [ "$pi_available" -eq 0 ] && [ "$install_opencode" -eq 0 ] && [ "$install_cline" -eq 0 ] && [ "$install_hermes" -eq 0 ] && [ "$install_dsh" -eq 0 ] && [ "$install_grok" -eq 0 ] && [ "$install_muse" -eq 0 ] && [ "$install_aider" -eq 0 ]; then
        fail "No selected coding agent was installed. Re-run the installer and choose at least one."
    fi
}

current_uv_version() {
    if output=$(uv --version); then
        :
    else
        return 1
    fi

    case "$output" in
        uv\ *) version=${output#uv } ;;
        *) version=$output ;;
    esac
    version=${version%% *}

    case "$version" in
        [0-9]*.[0-9]*.[0-9]*) printf '%s\n' "$version" ;;
        *) return 1 ;;
    esac
}

stable_version_is_supported() {
    case "$1" in
        *-*) return 1 ;;
    esac

    current=${1%%+*}
    minimum=${2%%+*}

    old_ifs=$IFS
    IFS=.
    set -- $current
    current_major=${1:-0}
    current_minor=${2:-0}
    current_patch=${3:-0}
    set -- $minimum
    minimum_major=${1:-0}
    minimum_minor=${2:-0}
    minimum_patch=${3:-0}
    IFS=$old_ifs

    case "$current_major$current_minor$current_patch$minimum_major$minimum_minor$minimum_patch" in
        *[!0-9]*) return 1 ;;
    esac

    [ "$current_major" -gt "$minimum_major" ] && return 0
    [ "$current_major" -lt "$minimum_major" ] && return 1
    [ "$current_minor" -gt "$minimum_minor" ] && return 0
    [ "$current_minor" -lt "$minimum_minor" ] && return 1
    [ "$current_patch" -ge "$minimum_patch" ]
}

verify_uv() {
    if [ "$dry_run" -eq 1 ]; then
        print_command uv --version
        return 0
    fi

    command -v uv >/dev/null 2>&1 || fail "uv was installed, but it is not available on PATH."
    version=$(current_uv_version) || fail "uv is present, but 'uv --version' did not return a valid version."
    if ! stable_version_is_supported "$version" "$MIN_UV_VERSION"; then
        fail "Stable uv $MIN_UV_VERSION or newer is required; found uv $version after installation."
    fi

    printf 'Verified uv %s.\n' "$version"
}

uv_installer_home_directory() {
    if [ -n "${HOME:-}" ]; then
        printf '%s\n' "$HOME"
        return 0
    fi

    if [ -n "${USER:-}" ]; then
        user_name=$USER
    else
        user_name=$(id -un) || fail "Could not determine the current user for uv installation."
    fi
    home_directory=$(getent passwd "$user_name" | cut -d: -f6)
    [ -n "$home_directory" ] || fail "Could not determine the home directory for uv installation."
    printf '%s\n' "$home_directory"
}

uv_install_bin_directory() {
    force_install_directory=""
    if [ -n "${UV_INSTALL_DIR:-}" ]; then
        force_install_directory=$UV_INSTALL_DIR
    elif [ -n "${UV_UNMANAGED_INSTALL:-}" ]; then
        force_install_directory=$UV_UNMANAGED_INSTALL
    fi

    if [ -n "$force_install_directory" ]; then
        inferred_home=$(uv_installer_home_directory)
        cargo_home=${CARGO_HOME:-$inferred_home/.cargo}
        if [ "$force_install_directory" = "$cargo_home" ]; then
            printf '%s/bin\n' "$force_install_directory"
        else
            printf '%s\n' "$force_install_directory"
        fi
    elif [ -n "${XDG_BIN_HOME:-}" ]; then
        printf '%s\n' "$XDG_BIN_HOME"
    elif [ -n "${XDG_DATA_HOME:-}" ]; then
        printf '%s/../bin\n' "$XDG_DATA_HOME"
    else
        inferred_home=$(uv_installer_home_directory)
        printf '%s/.local/bin\n' "$inferred_home"
    fi
}

ensure_uv() {
    if [ "$dry_run" -eq 1 ]; then
        if command -v uv >/dev/null 2>&1; then
            print_command uv --version
            printf 'A compatible existing uv will be left unchanged; an obsolete one will be replaced by the standalone installer.\n'
        else
            printf 'uv is not installed; the current standalone uv would be installed.\n'
            download_and_run "$UV_INSTALL_URL" sh "uv"
            verify_uv
        fi
        return 0
    fi

    if command -v uv >/dev/null 2>&1; then
        version=$(current_uv_version) || fail "uv is present, but 'uv --version' did not return a valid version."
        if stable_version_is_supported "$version" "$MIN_UV_VERSION"; then
            printf 'uv %s already satisfies >=%s; leaving it unchanged.\n' "$version" "$MIN_UV_VERSION"
            return 0
        fi
        printf 'uv %s does not satisfy stable >=%s; installing the current standalone uv.\n' "$version" "$MIN_UV_VERSION"
    else
        printf 'uv is not installed; installing the current standalone uv.\n'
    fi

    download_and_run "$UV_INSTALL_URL" sh "uv"
    uv_bin=$(uv_install_bin_directory) || return $?
    prioritize_path_entry "$uv_bin"
    verify_uv
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --voice-local)
                voice_local=1
                ;;
            --torch-backend)
                shift
                [ "$#" -gt 0 ] || fail "--torch-backend requires a value."
                torch_backend=$1
                [ -n "$torch_backend" ] || fail "--torch-backend requires a non-empty value."
                ;;
            --torch-backend=*)
                torch_backend=${1#*=}
                [ -n "$torch_backend" ] || fail "--torch-backend requires a non-empty value."
                ;;
            --rtk)
                enable_rtk=1
                ;;
            --dry-run)
                dry_run=1
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                show_usage >&2
                fail "unknown option: $1"
                ;;
        esac
        shift
    done
}

validate_args() {
    if [ -n "$torch_backend" ] && [ "$voice_local" -ne 1 ]; then
        fail "--torch-backend requires --voice-local."
    fi
}

package_spec() {
    if [ "$voice_local" -eq 1 ]; then
        printf 'free-claude-code[voice_local] @ %s' "$REPO_ARCHIVE_URL"
    else
        printf 'free-claude-code @ %s' "$REPO_ARCHIVE_URL"
    fi
}

install_free_claude_code() {
    assert_no_fcc_processes_running
    spec=$(package_spec)

    if [ -n "$torch_backend" ]; then
        run uv tool install --force --refresh-package free-claude-code --python "$PYTHON_VERSION" --torch-backend "$torch_backend" "$spec"
    else
        run uv tool install --force --refresh-package free-claude-code --python "$PYTHON_VERSION" "$spec"
    fi
}

configure_and_verify_free_claude_code() {
    run uv tool update-shell

    if [ "$dry_run" -eq 1 ]; then
        print_command uv tool dir --bin
        printf '+ verify fcc-desktop, fcc-server, fcc-claude, fcc-codex, fcc-pi, fcc-opencode, fcc-cline, fcc-hermes, fcc-dsh, fcc-grok, fcc-muse, and fcc-aider in the uv tool bin directory\n'
        print_command fcc-server --version
        return 0
    fi

    add_uv_tool_bin_directory

    for command_name in fcc-desktop fcc-server fcc-claude fcc-codex fcc-pi fcc-opencode fcc-cline fcc-hermes fcc-dsh fcc-grok fcc-muse fcc-aider fcc-update; do
        [ -x "$tool_bin/$command_name" ] || fail "Free Claude Code installation did not create $tool_bin/$command_name."
    done

    run "$tool_bin/fcc-server" --version
}

shell_quote() {
    escaped=$(printf '%s' "$1" | sed "s/'/'\\\\''/g")
    printf "'%s'" "$escaped"
}

macos_app_is_fcc_owned() {
    app_dir=$1
    owner_file="$app_dir/Contents/$FCC_MACOS_OWNER_FILE"
    [ -d "$app_dir" ] &&
        [ ! -L "$app_dir" ] &&
        [ -f "$owner_file" ] &&
        [ "$(cat "$owner_file")" = "$FCC_MACOS_BUNDLE_ID" ]
}

install_macos_desktop_app() {
    [ "$(uname -s)" = "Darwin" ] || return 0

    app_dir="$HOME/Applications/Free Claude Code.app"
    contents_dir="$app_dir/Contents"
    owner_file="$contents_dir/$FCC_MACOS_OWNER_FILE"
    executable_dir="$contents_dir/MacOS"
    executable_path="$executable_dir/fcc-desktop"
    resources_dir="$contents_dir/Resources"
    icon_path="$resources_dir/AppIcon.icns"
    desktop_dir="$HOME/Desktop"
    desktop_link="$desktop_dir/Free Claude Code.app"

    if [ -e "$app_dir" ] || [ -L "$app_dir" ]; then
        macos_app_is_fcc_owned "$app_dir" ||
            fail "An app not managed by Free Claude Code already exists at $app_dir. Move it, then rerun the installer."
    fi

    if [ "$dry_run" -eq 1 ]; then
        print_command mkdir -p "$executable_dir" "$resources_dir" "$desktop_dir"
        print_command fcc-desktop --export-icon "$icon_path"
        printf '+ write %s, %s, and %s\n' "$owner_file" "$contents_dir/Info.plist" "$executable_path"
        print_command ln -s "$app_dir" "$desktop_link"
        return 0
    fi

    mkdir -p "$executable_dir" "$resources_dir" "$desktop_dir"
    run "$tool_bin/fcc-desktop" --export-icon "$icon_path"
    [ -f "$icon_path" ] || fail "Free Claude Code did not export its macOS app icon to $icon_path."
    printf '%s\n' "$FCC_MACOS_BUNDLE_ID" > "$owner_file"
    cat > "$contents_dir/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Free Claude Code</string>
    <key>CFBundleExecutable</key>
    <string>fcc-desktop</string>
    <key>CFBundleIdentifier</key>
    <string>io.github.alishahryar1.free-claude-code</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleName</key>
    <string>Free Claude Code</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMultipleInstancesProhibited</key>
    <true/>
    <key>LSUIElement</key>
    <true/>
</dict>
</plist>
PLIST
    desktop_command=$(shell_quote "$tool_bin/fcc-desktop")
    {
        printf '%s\n' '#!/bin/sh'
        printf 'exec %s\n' "$desktop_command"
    } > "$executable_path"
    chmod +x "$executable_path"

    if [ -L "$desktop_link" ]; then
        if [ "$(readlink "$desktop_link")" = "$app_dir" ]; then
            rm -f "$desktop_link"
        else
            printf 'A non-FCC link already exists at %s; leaving it unchanged.\n' "$desktop_link"
            return 0
        fi
    elif [ -e "$desktop_link" ]; then
        printf 'A non-FCC item already exists at %s; leaving it unchanged.\n' "$desktop_link"
        return 0
    fi
    ln -s "$app_dir" "$desktop_link"
}

parse_args "$@"
validate_args
# Preserve the user's winning command before adding installer search paths.
original_opencode_path=$(command -v opencode || true)
add_known_bin_directories
if command -v cline >/dev/null 2>&1 || command -v npm >/dev/null 2>&1; then
    install_cline=1
fi
if ! command -v hermes >/dev/null 2>&1 && ! hermes_platform_is_supported; then
    install_hermes=0
fi
step "Checking for running Free Claude Code processes"
assert_no_fcc_processes_running

if ! installer_is_interactive && ! command -v dsh >/dev/null 2>&1; then
    if [ "$dry_run" -eq 1 ] && command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
        install_dsh=1
    elif ! dsh_toolchain_is_supported; then
        install_dsh=0
    fi
fi

if installer_is_interactive; then
    step "Choosing coding agents"
    choose_coding_agents /dev/tty /dev/tty
fi

step "Checking installation prerequisites"
require_command curl
if [ "$install_claude" -eq 1 ] || [ "$install_opencode" -eq 1 ] || [ "$install_hermes" -eq 1 ] || [ "$install_grok" -eq 1 ] || [ "$install_muse" -eq 1 ]; then
    require_command bash
fi
require_command sh
require_command mktemp
if [ "$enable_rtk" -eq 1 ] && ! command -v rtk >/dev/null 2>&1; then
    require_command tar
    if [ "$dry_run" -eq 0 ] &&
        ! command -v sha256sum >/dev/null 2>&1 &&
        ! command -v shasum >/dev/null 2>&1; then
        fail "RTK installation requires sha256sum or shasum for checksum verification."
    fi
fi

step "Ensuring uv $MIN_UV_VERSION or newer is installed"
ensure_uv

ensure_selected_coding_agents
configure_rtk_for_selected_agents

step "Installing or updating Free Claude Code"
install_free_claude_code

step "Configuring PATH and verifying Free Claude Code"
configure_and_verify_free_claude_code

if [ "$(uname -s)" = "Darwin" ]; then
    step "Installing the Free Claude Code desktop launcher"
    install_macos_desktop_app
fi

if [ "$dry_run" -eq 1 ]; then
    printf '\nDry run complete. No changes were made.\n'
else
    if [ "$(uname -s)" = "Darwin" ]; then
        printf '\nFree Claude Code is installed and verified. Open Free Claude Code from Applications or the desktop to run it in the background.\n'
        printf 'For terminal use, start the proxy with: fcc-server\n'
    else
        printf '\nFree Claude Code is installed and verified. Start the proxy with: fcc-server\n'
    fi
    if [ "$install_claude" -eq 1 ]; then
        printf 'Run Claude Code with: fcc-claude\n'
    fi
    if [ "$install_codex" -eq 1 ]; then
        printf 'Run Codex with: fcc-codex\n'
    fi
    if [ "$pi_available" -eq 1 ]; then
        printf 'Run Pi with: fcc-pi\n'
    fi
    if [ "$install_opencode" -eq 1 ]; then
        printf 'Run OpenCode with: fcc-opencode\n'
    fi
    if [ "$install_cline" -eq 1 ]; then
        printf 'Run Cline with: fcc-cline\n'
    else
        printf 'The fcc-cline wrapper is ready after you install Cline CLI.\n'
    fi
    if [ "$install_hermes" -eq 1 ]; then
        printf 'Run Hermes Agent with: fcc-hermes\n'
    else
        printf 'The fcc-hermes wrapper is ready after you install Hermes Agent.\n'
    fi
    if [ "$install_dsh" -eq 1 ]; then
        printf 'Run DeepSeek Harness with: fcc-dsh\n'
    else
        printf 'The fcc-dsh wrapper is ready after you install DeepSeek Harness %s.\n' "$DSH_VERSION"
    fi
    if [ "$install_grok" -eq 1 ]; then
        printf 'Run Grok Build with: fcc-grok\n'
    else
        printf 'The fcc-grok wrapper is ready after you install Grok Build.\n'
    fi
    if [ "$install_muse" -eq 1 ]; then
        printf 'Run Muse Code with: fcc-muse\n'
    else
        printf 'The fcc-muse wrapper is ready after you install Muse Code.\n'
    fi
    if [ "$install_aider" -eq 1 ]; then
        printf 'Run Aider with: fcc-aider\n'
    else
        printf 'The fcc-aider wrapper is ready after you install Aider.\n'
    fi
fi

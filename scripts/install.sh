#!/bin/sh
set -eu

PYTHON_VERSION="3.14.7"
MIN_UV_VERSION="0.12.13"
CLAUDE_INSTALL_URL="https://claude.ai/install.sh"
CODEX_INSTALL_URL="https://chatgpt.com/codex/install.sh"
PI_INSTALL_URL="https://pi.dev/install.sh"
OPENCODE_INSTALL_URL="https://opencode.ai/v2/install"
HERMES_INSTALL_URL="https://hermes-agent.nousresearch.com/install.sh"
MIN_DSH_VERSION="0.2.0-rc.2"
DSH_PACKAGE="@deepseek-ai/dsh@latest"
GROK_INSTALL_URL="https://x.ai/cli/install.sh"
MUSE_INSTALL_URL="https://dev.meta.ai/install.sh"
RTK_VERSION="0.44.2"
RTK_RELEASE_BASE_URL="https://github.com/rtk-ai/rtk/releases/download/v$RTK_VERSION"
UV_INSTALL_URL="https://astral.sh/uv/install.sh"
FCC_MACOS_BUNDLE_ID="io.github.alishahryar1.free-claude-code"
FCC_MACOS_OWNER_FILE=".free-claude-code-owner"
# Include retired entry points so updates reject older FCC processes before replacement.
FCC_COMMANDS="fcc-desktop fcc-server fcc-claude fcc-codex fcc-pi fcc-opencode fcc-cline fcc-hermes fcc-dsh fcc-grok fcc-muse fcc-aider fcc-doctor fcc-update fcc-init free-claude-code"

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
install_log=""
install_log_warning=0
install_state_directory=""
install_report_started=0
install_report=""
pending_steps=""
current_install_step=""
install_step_outcome=Completed
installer_path=$0
# With a piped installer, $0 names the shell, which may be an existing binary.
case "${installer_path##*/}" in
    sh|-sh|bash|-bash|dash|ksh|zsh) installer_path="" ;;
esac

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
    write_install_log "error: $*"
    printf 'error: %s\n' "$*" >&2
    exit 1
}

write_install_log() {
    [ -n "$install_log" ] || return 0
    if [ -n "$install_state_directory" ] && [ -d "$install_state_directory/log-disabled" ]; then
        return 0
    fi
    if { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$install_log"; } 2>/dev/null; then
        return 0
    fi
    install_log=""
    if [ "$install_log_warning" -eq 0 ]; then
        install_log_warning=1
        # Probes run in subshells. Share the warning state so a failed log does
        # not print the same warning for every remaining probe.
        if [ -z "$install_state_directory" ] || mkdir "$install_state_directory/log-disabled" 2>/dev/null; then
            printf 'warning: Could not write the installer log. Installation will continue.\n' >&2
        fi
    fi
}

initialize_install_log() {
    [ "$dry_run" -eq 0 ] || return 0
    install_state_directory=$(umask 077; mktemp -d "${TMPDIR:-/tmp}/fcc-installer.XXXXXX" 2>/dev/null) || install_state_directory=""
    log_timestamp=$(date '+%Y%m%d-%H%M%S' 2>/dev/null) || log_timestamp=unknown
    install_log="${HOME:-}/.fcc/logs/install-$log_timestamp-$$.log"
    if [ -n "${HOME:-}" ] && (umask 077; mkdir -p "$HOME/.fcc/logs" && set -C && : >"$install_log") 2>/dev/null; then
        write_install_log "Starting Free Claude Code installer"
    else
        install_log=""
        install_log_warning=1
        printf 'warning: Could not create the installer log. Installation will continue.\n' >&2
    fi
}

complete_install_step() {
    [ -n "$current_install_step" ] || return 0
    install_step_result="${1:-$install_step_outcome}: $current_install_step"
    install_report="${install_report}${install_step_result}
"
    write_install_log "$install_step_result"
    current_install_step=""
}

start_install_step() {
    [ "$install_report_started" -eq 1 ] || return 0
    complete_install_step
    current_install_step=$1
    install_step_outcome=Completed
    install_newline='
'
    case "$pending_steps" in
        "$1") pending_steps="" ;;
        "$1$install_newline"*) pending_steps=${pending_steps#*'
'} ;;
    esac
    write_install_log "Starting: $1"
}

print_installer_retry() {
    printf 'Retry: '
    if [ -f "$installer_path" ]; then
        printf 'sh '
        shell_quote "$installer_path"
    else
        printf '%s' "curl -fsSL 'https://raw.githubusercontent.com/Alishahryar1/free-claude-code/main/scripts/install.sh' | sh -s --"
    fi
    [ "$voice_local" -eq 0 ] || printf ' --voice-local'
    if [ -n "$torch_backend" ]; then
        printf ' --torch-backend '
        shell_quote "$torch_backend"
    fi
    [ "$enable_rtk" -eq 0 ] || printf ' --rtk'
    printf '\n'
}

finish_install() {
    install_exit_code=$1
    trap - EXIT
    set +e
    if [ "$install_report_started" -eq 1 ]; then
        if [ "$install_exit_code" -ne 0 ]; then
            complete_install_step Failed
        else
            complete_install_step
        fi
        printf '\nInstallation summary:\n%s' "$install_report"
        if [ -n "$pending_steps" ]; then
            printf '%s\n' "$pending_steps" | while IFS= read -r pending; do
                printf 'Not attempted: %s\n' "$pending"
            done
        fi
        [ -z "$install_log" ] || printf 'Installer log: %s\n' "$install_log"
        if [ "$install_exit_code" -ne 0 ]; then
            printf 'When reporting this failure, include the terminal error and the installer log. Interactive installer output is only shown in the terminal.\n'
            printf 'Completed changes have been kept. Resolve the error, then rerun the installer.\n'
            print_installer_retry
        fi
    fi
    cleanup
    exit "$install_exit_code"
}

installer_is_interactive() {
    [ -t 1 ] && ( : </dev/tty ) 2>/dev/null
}

prompt_yes_no() {
    question=$1
    default_answer=${2:-no}
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
    prompt_yes_no "Install $2 for $3?"
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

        if select_coding_agent cline "Cline CLI" fcc-cline; then
            install_cline=1
        else
            install_cline=0
        fi

        if select_coding_agent hermes "Hermes Agent" fcc-hermes; then
            install_hermes=1
        else
            install_hermes=0
        fi

        if select_coding_agent dsh "DeepSeek Harness" fcc-dsh; then
            install_dsh=1
        else
            install_dsh=0
        fi

        if select_coding_agent grok "Grok Build" fcc-grok; then
            install_grok=1
        else
            install_grok=0
        fi

        if select_coding_agent muse "Muse Code" fcc-muse; then
            install_muse=1
        else
            install_muse=0
        fi

        if select_coding_agent aider Aider fcc-aider; then
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

    run_started=$(date +%s)
    write_install_log "$(print_command "$@")"
    if "$@"; then
        write_install_log "Exit code 0 after $(($(date +%s) - run_started))s: $1"
        return 0
    else
        status=$?
    fi

    fail "Command failed with exit code $status: $1"
}

run_logged() {
    if [ "$dry_run" -eq 1 ] || [ -z "$install_state_directory" ]; then
        run "$@"
        return
    fi
    print_command "$@"
    write_install_log "$(print_command "$@")"
    run_started=$(date +%s)
    # Save the command's status separately: a POSIX pipeline reports the last
    # stage's status. Log failures must not replace the package command's result.
    (
        if "$@"; then status=0; else status=$?; fi
        printf '%s\n' "$status" >"$install_state_directory/package-status"
    ) 2>&1 | while IFS= read -r line || [ -n "$line" ]; do
        printf '%s\n' "$line"
        write_install_log "$line"
    done
    read -r status <"$install_state_directory/package-status"
    rm -f "$install_state_directory/package-status"
    write_install_log "Exit code $status after $(($(date +%s) - run_started))s: $1"
    [ "$status" -eq 0 ] || fail "Command failed with exit code $status: $1"
}

# Keep terminal-attached installers in run(). Only noninteractive probes use this
# supervisor, whose private process group lets cancellation clean up descendants.
probe_capture() (
    probe_seconds=$1
    shift
    [ "$dry_run" -eq 0 ] || { print_command "$@" >&2; return 0; }
    probe_directory=$(umask 077; mktemp -d "${TMPDIR:-/tmp}/fcc-probe.XXXXXX") || return 125
    probe_supervisor=""
    probe_cleanup() {
        if [ -n "$probe_supervisor" ]; then
            kill -s TERM "$probe_supervisor" 2>/dev/null || :
            wait "$probe_supervisor" 2>/dev/null || :
        fi
        rm -rf "$probe_directory"
    }
    trap probe_cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' HUP TERM
    write_install_log "$(print_command "$@") (deadline ${probe_seconds}s)"
    probe_started=$(date +%s)
    BASH_ENV= ENV= bash --noprofile --norc -s -- "$probe_seconds" "$probe_directory" "$@" <<'PROBE' &
set +e
set -u
seconds=$1
scratch=$2
shift 2
probe_pid=
probe_group=
stop_probe() {
    if [ -n "$probe_group" ]; then
        if [ "$1" -eq 1 ]; then
            kill -s TERM -- "-$probe_group" 2>/dev/null || :
            sleep 1
        fi
        kill -s KILL -- "-$probe_group" 2>/dev/null || :
    elif [ -n "$probe_pid" ]; then
        kill -s KILL "$probe_pid" 2>/dev/null || :
    fi
    [ -z "$probe_pid" ] || wait "$probe_pid" 2>/dev/null || :
    probe_pid=
    probe_group=
}
cancel() {
    trap '' HUP INT TERM
    stop_probe 1
    exit "$1"
}
trap 'cancel 130' INT
trap 'cancel 143' HUP TERM
trap 'stop_probe 0' EXIT
mkfifo "$scratch/status.fifo" "$scratch/stdout.fifo" "$scratch/stderr.fifo" || exit 125
set -m
(
    set +m
    # The parent verifies our group before opening this FIFO. Keep its leader
    # alive after the command exits so cleanup cannot target a reused group ID.
    exec 3>"$scratch/status.fifo"
    trap ':' HUP INT TERM
    # Readers finish only after all inherited writers close their handles.
    # They stay in this group, so the deadline also bounds output draining.
    cat <"$scratch/stdout.fifo" >"$scratch/stdout" 3>&- &
    stdout_reader=$!
    cat <"$scratch/stderr.fifo" >"$scratch/stderr" 3>&- &
    stderr_reader=$!
    "$@" </dev/null >"$scratch/stdout.fifo" 2>"$scratch/stderr.fifo" 3>&-
    status=$?
    wait "$stdout_reader" || status=125
    wait "$stderr_reader" || status=125
    printf '%s\n' "$status" >&3
    while :; do sleep 60; done
) 2>/dev/null &
probe_pid=$!
set +m
group=$(ps -o pgid= -p "$probe_pid" 2>/dev/null) || exit 125
case "$group" in *[!0-9[:space:]]*|'') exit 125 ;; esac
[ "$group" -eq "$probe_pid" ] || exit 125
probe_group=$probe_pid
exec 3<"$scratch/status.fifo"
if IFS= read -r -t "$seconds" status <&3; then
    case "$status" in ''|*[!0-9]*) exit 125 ;; esac
    stop_probe 0
else
    : >"$scratch/timed-out"
    stop_probe 1
    status=124
fi
exit "$status"
PROBE
    probe_supervisor=$!
    if wait "$probe_supervisor"; then probe_status=0; else probe_status=$?; fi
    probe_supervisor=""
    for stream in stdout stderr; do
        if [ -f "$probe_directory/$stream" ]; then
            write_install_log "$stream: $(cat "$probe_directory/$stream")"
        fi
    done
    write_install_log "Exit code $probe_status after $(($(date +%s) - probe_started))s: $1"
    [ ! -f "$probe_directory/stdout" ] || cat "$probe_directory/stdout"
    [ ! -f "$probe_directory/stderr" ] || cat "$probe_directory/stderr" >&2
    if [ -f "$probe_directory/timed-out" ]; then
        printf 'Command timed out after %ss: %s\n' "$probe_seconds" "$1" >&2
        write_install_log "Command timed out: $1"
    fi
    exit "$probe_status"
)

run_probe() {
    probe_limit=$1
    shift
    print_command "$@"
    if probe_capture "$probe_limit" "$@"; then return 0; else probe_result=$?; fi
    fail "Command failed with exit code $probe_result: $1"
}

download_file() {
    download_limit=$3
    # curl resets a partial output file before a retry. Bound both each transfer
    # and the supervisor so even a long Retry-After cannot stall the installer.
    probe_capture "$((download_limit * 3 + 30))" curl -fsSL --connect-timeout 10 \
        --max-time "$download_limit" --retry 2 --retry-connrefused \
        --retry-max-time "$((download_limit * 2 + 30))" "$1" -o "$2"
}

cleanup() {
    if [ -n "$install_state_directory" ]; then
        rm -rf "$install_state_directory"
    fi
    if [ -n "$temporary_file" ] && [ -e "$temporary_file" ]; then
        rm -f "$temporary_file"
    fi
    if [ -n "$temporary_binary" ] && [ -e "$temporary_binary" ]; then
        rm -f "$temporary_binary"
    fi
}

trap 'finish_install "$?"' EXIT
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
    if tool_bin=$(probe_capture 10 uv tool dir --bin); then
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
        pi_npm_prefix=$(probe_capture 10 npm prefix -g 2>/dev/null || probe_capture 10 npm config get prefix 2>/dev/null || true)
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
    if download_file "$url" "$temporary_file" 30; then
        :
    else
        status=$?
        fail "Could not download the $label installer (curl exit code $status)."
    fi

    if [ ! -s "$temporary_file" ]; then
        fail "The downloaded $label installer was empty."
    fi

    installer_started=$(date +%s)
    write_install_log "$(print_command "$interpreter" "$temporary_file" "$@")"
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

    write_install_log "$label installer completed after $(($(date +%s) - installer_started))s"
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
    run_probe 10 "$command_path" --version
}

pi_command_is_compatible() {
    pi_command_path=$(command -v pi 2>/dev/null) || return 1
    pi_help=$(probe_capture 10 "$pi_command_path" --help 2>/dev/null) || return 1
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
    run_probe 10 "$pi_command_path" --version
}

verify_rtk_command() {
    if [ "$dry_run" -eq 1 ]; then
        print_command env RTK_TELEMETRY_DISABLED=1 rtk --version
        print_command env RTK_TELEMETRY_DISABLED=1 rtk gain
        return 0
    fi

    rtk_path=$(command -v rtk 2>/dev/null) || fail "RTK was installed, but 'rtk' is not available on PATH."
    print_command env RTK_TELEMETRY_DISABLED=1 "$rtk_path" --version
    if ! RTK_TELEMETRY_DISABLED=1 probe_capture 10 "$rtk_path" --version; then
        fail "The 'rtk' command at $rtk_path is not a compatible Rust Token Killer installation. Remove the conflicting command from PATH, then rerun the installer."
    fi

    print_command env RTK_TELEMETRY_DISABLED=1 "$rtk_path" gain
    if ! RTK_TELEMETRY_DISABLED=1 probe_capture 10 "$rtk_path" gain; then
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
    if download_file "$rtk_archive_url" "$temporary_file" 600; then
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

    rtk_started=$(date +%s)
    write_install_log "$(print_command "$rtk_path" "$@")"
    if RTK_TELEMETRY_DISABLED=1 "$rtk_path" "$@"; then
        write_install_log "RTK configuration completed after $(($(date +%s) - rtk_started))s"
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
        install_step_outcome=Reused
        printf 'Claude Code already found on PATH; verifying it.\n'
    else
        download_and_run "$CLAUDE_INSTALL_URL" bash "Claude Code"
        add_known_bin_directories
    fi

    verify_command claude "Claude Code"
}

ensure_codex() {
    if command -v codex >/dev/null 2>&1; then
        install_step_outcome=Reused
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
        install_step_outcome=Reused
        printf 'Pi already found on PATH; verifying it.\n'
    elif pi_command_is_compatible; then
        install_step_outcome=Reused
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
                install_step_outcome=Skipped
                printf 'Pi was not installed; continuing without it.\n'
                return 0
            fi
        fi
    fi

    verify_pi_command
    pi_available=1
}

opencode_version() {
    opencode_output=$(probe_capture 10 "$1" --version) || return 1
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

check_opencode_install() {
    [ -n "${HOME:-}" ] || fail "HOME is required to install OpenCode."
    opencode_native="$HOME/.opencode/bin/opencode"
    opencode_path=${original_opencode_path:-$(command -v opencode || true)}
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
    fi
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

    check_opencode_install
    if [ "$opencode_install" -eq 1 ]; then
        download_and_run "$OPENCODE_INSTALL_URL" run_opencode_installer "OpenCode"
        add_known_bin_directories
        hash -r 2>/dev/null || true
        opencode_installed=$(opencode_version "$opencode_native") || fail "Could not verify installed OpenCode at $opencode_native."
        case "$opencode_installed" in
            2.*) ;;
            *) fail "The OpenCode installer did not install stable OpenCode 2. See https://opencode.ai/v2/docs/" ;;
        esac
    fi
    if [ "$opencode_install" -eq 0 ] && [ -z "$opencode_plugin" ]; then
        install_step_outcome=Reused
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
        install_step_outcome=Reused
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
        install_step_outcome=Reused
        printf 'Hermes Agent already found on PATH; verifying it.\n'
    else
        install_hermes
    fi

    verify_command hermes "Hermes Agent"
}

current_dsh_version() {
    if output=$(probe_capture 10 dsh --version 2>/dev/null); then
        :
    else
        return 1
    fi

    version=$(printf '%s\n' "$output" | awk '
        { sub(/^[[:space:]]*(dsh[[:space:]]+)?v?/, ""); sub(/[[:space:]]*$/, "") }
        /^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$/ { print; exit }
    ')
    [ -n "$version" ] || return 1
    printf '%s\n' "$version"
}

dsh_version_is_supported() {
    LC_ALL=C awk -v version="$1" -v minimum="$MIN_DSH_VERSION" '
        function numeric(value) { return value ~ /^[0-9]+$/ }
        function compare(left, right) {
            if (left == right) return 0
            if (numeric(left) && numeric(right)) {
                if (length(left) != length(right)) return length(left) > length(right) ? 1 : -1
            } else if (numeric(left) != numeric(right)) return numeric(left) ? -1 : 1
            return ("x" left) > ("x" right) ? 1 : -1
        }
        BEGIN {
            sub(/\+.*/, "", version)
            dash = index(version, "-")
            core = dash ? substr(version, 1, dash - 1) : version
            count = dash ? split(substr(version, dash + 1), preview, ".") : 0
            if (split(core, release, ".") != 3) exit 1
            for (i = 1; i <= 3; i++) if (release[i] !~ /^(0|[1-9][0-9]*)$/) exit 1
            for (i = 1; i <= count; i++) {
                if (preview[i] !~ /^[0-9A-Za-z-]+$/ || preview[i] ~ /^0[0-9]+$/) exit 1
            }
            split(minimum, parts, "-")
            split(parts[1], base, ".")
            for (i = 1; i <= 3; i++) {
                diff = compare(release[i], base[i])
                if (diff) exit (diff < 0)
            }
            if (!count) exit 0
            floorCount = split(parts[2], floor, ".")
            for (i = 1; i <= count && i <= floorCount; i++) {
                diff = compare(preview[i], floor[i])
                if (diff) exit (diff < 0)
            }
            exit (count < floorCount)
        }
    '
}

current_node_version() {
    if output=$(probe_capture 10 node --version 2>/dev/null); then
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
    version=$(current_dsh_version) || fail "DeepSeek Harness is present, but 'dsh --version' did not return a semantic version."
    dsh_version_is_supported "$version" || fail "DeepSeek Harness requires >=$MIN_DSH_VERSION; found $version after installation."
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
            printf 'DeepSeek Harness >=%s will be preserved; an older version will be upgraded to latest.\n' "$MIN_DSH_VERSION"
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
        version=$(current_dsh_version) || fail "DeepSeek Harness is present, but 'dsh --version' did not return a semantic version."
        if dsh_version_is_supported "$version"; then
            install_step_outcome=Reused
            printf 'DeepSeek Harness %s already satisfies >=%s; leaving it unchanged.\n' "$version" "$MIN_DSH_VERSION"
            return 0
        fi
        printf 'DeepSeek Harness requires >=%s; upgrading %s to latest.\n' "$MIN_DSH_VERSION" "$version"
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
        install_step_outcome=Reused
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
        install_step_outcome=Reused
        printf 'Muse Code already found on PATH; verifying it.\n'
    else
        install_muse_code
    fi

    verify_command muse "Muse Code"
}

install_aider_cli() {
    run_logged uv tool install --force --python python3.12 --with pip aider-chat@latest
}

ensure_aider() {
    if ! command -v aider >/dev/null 2>&1 && [ "$dry_run" -eq 0 ]; then
        add_uv_tool_bin_directory
    fi

    if command -v aider >/dev/null 2>&1; then
        install_step_outcome=Reused
        printf 'Aider already found on PATH; verifying it.\n'
    else
        install_aider_cli
    fi

    verify_command aider "Aider"
}

ensure_selected_coding_agents() {
    if [ "$install_claude" -eq 1 ]; then
        start_install_step "Claude Code"
        step "Ensuring Claude Code is installed"
        ensure_claude
    fi

    if [ "$install_codex" -eq 1 ]; then
        start_install_step "Codex"
        step "Ensuring Codex is installed"
        ensure_codex
    fi

    if [ "$install_pi" -eq 1 ]; then
        start_install_step "Pi"
        step "Checking or installing Pi"
        ensure_pi
    fi

    if [ "$install_opencode" -eq 1 ]; then
        start_install_step "OpenCode"
        step "Ensuring OpenCode is installed"
        ensure_opencode
    fi

    if [ "$install_cline" -eq 1 ]; then
        start_install_step "Cline"
        step "Ensuring Cline CLI is installed"
        ensure_cline
    fi

    if [ "$install_hermes" -eq 1 ]; then
        start_install_step "Hermes"
        step "Ensuring Hermes Agent is installed"
        ensure_hermes
    fi

    if [ "$install_dsh" -eq 1 ]; then
        start_install_step "DeepSeek Harness"
        step "Ensuring DeepSeek Harness is installed"
        ensure_dsh
    fi

    if [ "$install_grok" -eq 1 ]; then
        start_install_step "Grok"
        step "Ensuring Grok Build is installed"
        ensure_grok
    fi

    if [ "$install_muse" -eq 1 ]; then
        start_install_step "Muse"
        step "Ensuring Muse Code is installed"
        ensure_muse
    fi

    if [ "$install_aider" -eq 1 ]; then
        start_install_step "Aider"
        step "Ensuring Aider is installed"
        ensure_aider
    fi

    if [ "$install_claude" -eq 0 ] && [ "$install_codex" -eq 0 ] && [ "$pi_available" -eq 0 ] && [ "$install_opencode" -eq 0 ] && [ "$install_cline" -eq 0 ] && [ "$install_hermes" -eq 0 ] && [ "$install_dsh" -eq 0 ] && [ "$install_grok" -eq 0 ] && [ "$install_muse" -eq 0 ] && [ "$install_aider" -eq 0 ]; then
        fail "No selected coding agent was installed. Re-run the installer and choose at least one."
    fi
}

current_uv_version() {
    if output=$(probe_capture 10 uv --version); then
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
            install_step_outcome=Reused
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
        printf '%s' 'free-claude-code[voice_local]'
    else
        printf '%s' 'free-claude-code'
    fi
}

install_free_claude_code() {
    assert_no_fcc_processes_running
    spec=$(package_spec)

    if [ -n "$torch_backend" ]; then
        run_logged uv tool install --force --refresh-package free-claude-code --python "$PYTHON_VERSION" --torch-backend "$torch_backend" "$spec"
    else
        run_logged uv tool install --force --refresh-package free-claude-code --python "$PYTHON_VERSION" "$spec"
    fi
}

configure_and_verify_free_claude_code() {
    start_install_step "PATH configuration"
    run uv tool update-shell

    if [ "$dry_run" -eq 1 ]; then
        print_command uv tool dir --bin
        printf '+ verify fcc-desktop, fcc-server, fcc-claude, fcc-codex, fcc-pi, fcc-opencode, fcc-cline, fcc-hermes, fcc-dsh, fcc-grok, fcc-muse, and fcc-aider in the uv tool bin directory\n'
        print_command fcc-server --version
        return 0
    fi

    add_uv_tool_bin_directory
    start_install_step "FCC verification"

    for command_name in fcc-desktop fcc-server fcc-claude fcc-codex fcc-pi fcc-opencode fcc-cline fcc-hermes fcc-dsh fcc-grok fcc-muse fcc-aider fcc-doctor fcc-update; do
        [ -x "$tool_bin/$command_name" ] || fail "Free Claude Code installation did not create $tool_bin/$command_name."
    done

    run_probe 10 "$tool_bin/fcc-server" --version
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
    run_probe 30 "$tool_bin/fcc-desktop" --export-icon "$icon_path"
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
initialize_install_log
if [ "$dry_run" -eq 0 ]; then install_report_started=1; fi
start_install_step Preflight
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
require_command bash

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

pending_steps=$(printf 'uv\n'
    [ "$install_claude" -eq 0 ] || printf 'Claude Code\n'
    [ "$install_codex" -eq 0 ] || printf 'Codex\n'
    [ "$install_pi" -eq 0 ] || printf 'Pi\n'
    [ "$install_opencode" -eq 0 ] || printf 'OpenCode\n'
    [ "$install_cline" -eq 0 ] || printf 'Cline\n'
    [ "$install_hermes" -eq 0 ] || printf 'Hermes\n'
    [ "$install_dsh" -eq 0 ] || printf 'DeepSeek Harness\n'
    [ "$install_grok" -eq 0 ] || printf 'Grok\n'
    [ "$install_muse" -eq 0 ] || printf 'Muse\n'
    [ "$install_aider" -eq 0 ] || printf 'Aider\n'
    printf 'FCC package\nPATH configuration\nFCC verification\n'
    [ "$(uname -s)" != Darwin ] || printf 'Desktop integration\n'
    [ "$enable_rtk" -eq 0 ] || printf 'RTK configuration\n'
)

step "Checking installation prerequisites"
require_command curl
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

if [ "$dry_run" -eq 0 ]; then
    # Pi can bootstrap Node/npm before the dependent agents are installed.
    if [ "$install_pi" -eq 0 ] || find_installed_coding_agent pi >/dev/null; then
        if [ "$install_dsh" -eq 1 ]; then require_dsh_toolchain; fi
        if [ "$install_cline" -eq 1 ] && ! command -v cline >/dev/null 2>&1; then
            command -v npm >/dev/null 2>&1 || fail "Cline installation requires npm. Install Node.js from https://nodejs.org/en/download, then rerun the installer."
        fi
    fi
    if [ "$install_hermes" -eq 1 ] && ! command -v hermes >/dev/null 2>&1; then confirm_hermes_platform; fi
    if [ "$install_opencode" -eq 1 ]; then check_opencode_install; fi
    if [ "$enable_rtk" -eq 1 ] && ! command -v rtk >/dev/null 2>&1; then select_rtk_release; fi
fi

start_install_step uv
step "Ensuring uv $MIN_UV_VERSION or newer is installed"
ensure_uv

ensure_selected_coding_agents

start_install_step "FCC package"
step "Installing or updating Free Claude Code"
install_free_claude_code

step "Configuring PATH and verifying Free Claude Code"
configure_and_verify_free_claude_code

if [ "$(uname -s)" = "Darwin" ]; then
    start_install_step "Desktop integration"
    step "Installing the Free Claude Code desktop launcher"
    install_macos_desktop_app
fi
if [ "$enable_rtk" -eq 1 ]; then
    start_install_step "RTK configuration"
    configure_rtk_for_selected_agents
fi
complete_install_step

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
        printf 'The fcc-dsh wrapper is ready after you install DeepSeek Harness >=%s.\n' "$MIN_DSH_VERSION"
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

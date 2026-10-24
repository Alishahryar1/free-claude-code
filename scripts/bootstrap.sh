#!/bin/sh
set -eu

# Select once, then delegate all installation to that release's installer.
repository="https://github.com/Alishahryar1/free-claude-code"
raw="https://raw.githubusercontent.com/Alishahryar1/free-claude-code"
release=""
dry_run=0
retry_options=""

quote_argument() {
    escaped=$(printf '%s' "$1" | sed "s/'/'\\\\''/g")
    printf "'%s'" "$escaped"
}

is_release_version() {
    case "$1" in ''|*[!0-9.]*) return 1 ;; esac
    printf '%s\n' "$1" | grep -Eq '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
}

fail() {
    printf 'error: %s\n' "$1" >&2
    if [ -n "$release" ]; then
        printf "Retry FCC %s: curl -fsSL '%s/v%s/scripts/install.sh' | FCC_INSTALL_VERSION=%s sh -s --%s\n" \
            "$release" "$raw" "$release" "$release" "$retry_options" >&2
    else
        printf 'Retry the bootstrap command after correcting the error.\n' >&2
    fi
    exit 1
}

validate_options() {
    voice=0
    backend=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --voice-local) voice=1 ;;
            --torch-backend)
                [ "$#" -ge 2 ] && [ -n "$2" ] || fail "--torch-backend requires a value."
                backend=1
                shift
                ;;
            --rtk-headroom) ;;
            --dry-run) dry_run=1 ;;
            --help|-h)
                printf '%s\n' 'Usage: bootstrap.sh [--voice-local] [--torch-backend VALUE] [--rtk-headroom] [--dry-run] [--help]' \
                    'Selects the latest completed FCC release and runs its installer.'
                exit 0
                ;;
            *) fail "unknown option: $1" ;;
        esac
        shift
    done
    [ "$backend" -eq 0 ] || [ "$voice" -eq 1 ] || fail "--torch-backend requires --voice-local."
}

for argument do retry_options="$retry_options $(quote_argument "$argument")"; done
validate_options "$@"
if [ -n "${FCC_INSTALL_VERSION:-}" ]; then
    is_release_version "$FCC_INSTALL_VERSION" ||
        fail "FCC_INSTALL_VERSION must be a canonical major.minor.patch release."
    release=$FCC_INSTALL_VERSION
fi
if [ "$dry_run" -eq 1 ]; then
    printf 'Select FCC release: %s\n' "${release:-<latest completed release>}"
    printf 'Download %s/v%s/scripts/install.sh and run with FCC_INSTALL_VERSION set%s\n' \
        "$raw" "${release:-<selected-release>}" "$retry_options"
    exit 0
fi
if [ -z "$release" ]; then
    selected_url=$(curl -fsSL --max-time 10 --max-redirs 5 -o /dev/null -w '%{url_effective}' "$repository/releases/latest") ||
        fail "Could not resolve the latest completed FCC release."
    case "$selected_url" in
        "$repository/releases/tag/v"*) candidate=${selected_url#"$repository/releases/tag/v"} ;;
        *) fail "GitHub returned an unexpected FCC release URL." ;;
    esac
    is_release_version "$candidate" ||
        fail "GitHub did not return a canonical FCC release URL."
    release=$candidate
fi
installer=$(curl -fsSL --max-time 10 --max-redirs 5 "$raw/v$release/scripts/install.sh") ||
    fail "Could not download the installer for FCC $release."
[ -n "$installer" ] || fail "The installer for FCC $release was empty."
sh -n -c "$installer" || fail "The installer for FCC $release was not valid shell."
FCC_INSTALL_VERSION=$release exec sh -c "$installer" sh "$@"

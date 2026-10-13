#!/bin/sh
set -eu

CHECK_ORDER="suppressions ruff-format ruff-check ty pytest playwright"

dry_run=0
only_checks=""
skip_checks=""

show_usage() {
    cat <<'USAGE'
Usage: ci.sh [options]

Runs the local sequence for the same check IDs enforced by GitHub CI.
Requires uv on PATH when running ruff, ty, pytest, or Playwright checks.
Local ruff checks repair formatting and autofixable lint before later checks.

Checks (in order):
  suppressions   Ban type ignores and legacy future annotations
  ruff-format    uv run ruff format
  ruff-check     uv run ruff check --fix
  ty             uv run ty check
  pytest         uv run pytest -v --tb=short
  playwright     Install Chromium and run deterministic Admin browser tests

Options:
  --only ID                Run only the given check (repeatable)
  --skip ID                Skip the given check (repeatable)
  --dry-run                Print commands without running them.
  --help                   Show this help text.
USAGE
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
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
    if [ "$dry_run" -eq 0 ]; then
        "$@"
    fi
}

valid_check_id() {
    case "$1" in
        suppressions | ruff-format | ruff-check | ty | pytest | playwright) return 0 ;;
        *) return 1 ;;
    esac
}

validate_check_id() {
    if ! valid_check_id "$1"; then
        fail "unknown check id: $1 (expected one of: $CHECK_ORDER)"
    fi
}

contains_check_id() {
    list=$1
    id=$2
    case " $list " in
        *" $id "*) return 0 ;;
        *) return 1 ;;
    esac
}

should_run_check() {
    check_id=$1

    if [ -n "$only_checks" ] && ! contains_check_id "$only_checks" "$check_id"; then
        return 1
    fi

    if contains_check_id "$skip_checks" "$check_id"; then
        return 1
    fi

    return 0
}

assert_uv_available() {
    if ! command -v uv >/dev/null 2>&1; then
        fail "uv is required but was not found on PATH. Install uv first (see README or scripts/install.sh)."
    fi
}

selected_checks_need_uv() {
    if [ "$dry_run" -ne 0 ]; then
        return 1
    fi

    for check_id in $CHECK_ORDER; do
        if should_run_check "$check_id" && [ "$check_id" != "suppressions" ]; then
            return 0
        fi
    done

    return 1
}

run_suppressions() {
    step "Ban suppressions and legacy annotations"
    pattern='# type: ignore|# ty: ignore|from __future__ import annotations'
    print_command grep -rE "$pattern" --include='*.py' . \
        --exclude-dir=.venv --exclude-dir=.git
    if [ "$dry_run" -eq 0 ]; then
        if grep -rE "$pattern" --include='*.py' . \
            --exclude-dir=.venv --exclude-dir=.git; then
            fail "type: ignore / ty: ignore comments and legacy future annotations are not allowed. Fix the underlying type/import issue instead."
        fi
    fi
}

run_ruff_format() {
    step "ruff format"
    run uv run ruff format
}

run_ruff_check() {
    step "ruff check --fix"
    run uv run ruff check --fix
}

run_ty() {
    step "ty check"
    run uv run ty check
}

run_pytest() {
    step "pytest"
    run uv run pytest -v --tb=short
}

run_playwright() {
    step "playwright"
    run uv run playwright install chromium
    run uv run pytest e2e -v --tb=short \
        --tracing=retain-on-failure --screenshot=only-on-failure \
        --full-page-screenshot --output=test-results
}

run_check() {
    case "$1" in
        suppressions) run_suppressions ;;
        ruff-format) run_ruff_format ;;
        ruff-check) run_ruff_check ;;
        ty) run_ty ;;
        pytest) run_pytest ;;
        playwright) run_playwright ;;
        *) fail "unknown check id: $1" ;;
    esac
}

parse_args() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --only)
                shift
                [ "$#" -gt 0 ] || fail "--only requires a check id."
                validate_check_id "$1"
                only_checks="${only_checks} $1"
                ;;
            --skip)
                shift
                [ "$#" -gt 0 ] || fail "--skip requires a check id."
                validate_check_id "$1"
                skip_checks="${skip_checks} $1"
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

parse_args "$@"
if selected_checks_need_uv; then
    assert_uv_available
fi

for check_id in $CHECK_ORDER; do
    if should_run_check "$check_id"; then
        run_check "$check_id"
    fi
done

printf '\nAll selected CI checks passed.\n'

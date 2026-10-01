#!/usr/bin/env bash
# install-service.sh — install the launchd plist for the fcc server.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TEMPLATE="$REPO/scripts/com.fcc.server.plist.template"
DEST="$HOME/Library/LaunchAgents/com.fcc.server.plist"

UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
    echo "error: uv not found on PATH. Install from https://astral.sh/uv" >&2
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents"

sed \
    -e "s|__UV__|$UV|g" \
    -e "s|__REPO__|$REPO|g" \
    -e "s|__PATH__|$PATH|g" \
    "$TEMPLATE" > "$DEST"

echo "installed $DEST"
echo "next: $REPO/scripts/fccd start"

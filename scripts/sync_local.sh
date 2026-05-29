#!/usr/bin/env bash
# Sync project to local CachyOS machine via rsync.
#
# Usage:
#   ./scripts/sync_local.sh              # sync only
#
# Requires: rsync, SSH access to shuo@cachyos-n.local

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-shuo@cachyos-n.local}"
REMOTE_DIR="${REMOTE_DIR:-/home/shuo/GitHub/sku-match}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ">>> Syncing $PROJECT_DIR/ → $REMOTE_HOST:$REMOTE_DIR/"
rsync -az --delete --exclude-from="$PROJECT_DIR/.rsyncignore" "$PROJECT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"
echo ">>> Sync complete."

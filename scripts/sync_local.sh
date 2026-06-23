#!/usr/bin/env bash
# Sync project to local CachyOS machine via rsync, then optionally restart the API server.
#
# Usage:
#   ./scripts/sync_local.sh              # sync only
#   ./scripts/sync_local.sh --restart    # sync + uv sync + restart API server
#
# Requires: rsync, SSH access to $REMOTE_HOST
# Override: REMOTE_HOST, REMOTE_DIR env vars.

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-shuo@cachyos-n.local}"
REMOTE_DIR="${REMOTE_DIR:-/home/shuo/GitHub/sku-match}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo ">>> Syncing $PROJECT_DIR/ → $REMOTE_HOST:$REMOTE_DIR/"
rsync -az --delete --exclude-from="$PROJECT_DIR/.rsyncignore" "$PROJECT_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"
echo ">>> Sync complete."

if [[ "${1:-}" == "--restart" ]]; then
  echo ">>> Running uv sync on $REMOTE_HOST..."
  ssh "$REMOTE_HOST" "cd $REMOTE_DIR && uv sync"

  echo ">>> Restarting sku-match service on $REMOTE_HOST..."
  ssh "$REMOTE_HOST" bash -s <<'RESTART'
    set -e
    systemctl restart sku-match
    echo "Service restarted. Waiting for health check..."

    for i in $(seq 1 30); do
      if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "API server is healthy."
        exit 0
      fi
      sleep 2
    done
    echo "WARNING: Health check timed out. Check: systemctl status sku-match"
    exit 1
RESTART
  echo ">>> Done."
fi

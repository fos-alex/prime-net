#!/usr/bin/env bash
# Sync prime-net training runs from the cloud VPS into the local project,
# merge them into the local registry, and regenerate the HTML dashboard.
#
# Designed to run from a systemd user timer (every 5 min). Safe to run manually.
# Overrides:
#   PRIMENET_DIR    local project root (default ~/prime-net)
#   PRIMENET_REMOTE remote ssh target (default root@cloud-openclaw)
set -euo pipefail

DIR="${PRIMENET_DIR:-$HOME/prime-net}"
REMOTE="${PRIMENET_REMOTE:-root@cloud-openclaw}"
REMOTE_RUNS="~/prime-net/runs/"
LOG="$DIR/runs/sync.log"

mkdir -p "$DIR/runs"
exec 9>"$DIR/runs/.sync.lock"
flock -n 9 || { echo "[$(date -Is)] previous sync still running, skipping" >> "$LOG"; exit 0; }

{
  echo "[$(date -Is)] syncing from $REMOTE"
  # registry.jsonl stays local-authoritative (new runs merge via --backfill);
  # progress/ is locally generated and must never be overwritten by the remote copy
  rsync -a --exclude registry.jsonl --exclude progress/ \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "$REMOTE:$REMOTE_RUNS" "$DIR/runs/"
  cd "$DIR"
  "$DIR/.venv/bin/python" -m primenet.board --backfill
  echo "[$(date -Is)] done"
} >> "$LOG" 2>&1

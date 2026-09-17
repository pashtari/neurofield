#!/usr/bin/env bash
# Pull the RunPod working copy (/workspace/neurofield) back to this Mac.
# Safe by default:
#   - never deletes anything local
#   - never overwrites a local file that is NEWER than the pod's copy
#     (so local edits made after a pod run always win)
# Usage:
#   scripts/pull_runpod.sh       # sync
#   scripts/pull_runpod.sh -n    # dry run: list what would change, touch nothing
set -euo pipefail

REMOTE=runpod
REMOTE_DIR=/workspace/neurofield/
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" true 2>/dev/null; then
  echo "ERROR: cannot reach host '$REMOTE'." >&2
  echo "If the pod was stopped/restarted, its IP and port have changed:" >&2
  echo "  RunPod console -> pod -> Connect -> 'SSH over exposed TCP'," >&2
  echo "  then update the HostName/Port lines under 'Host runpod' in ~/.ssh/config." >&2
  exit 1
fi

rsync -azui --no-owner --no-group \
  --exclude '__pycache__' --exclude '.ipynb_checkpoints' --exclude '.DS_Store' \
  --exclude '._*' --exclude '*.egg-info' \
  --stats "$@" \
  "${REMOTE}:${REMOTE_DIR}" "${LOCAL_DIR}"

echo "OK: synced ${REMOTE}:${REMOTE_DIR} -> ${LOCAL_DIR}"

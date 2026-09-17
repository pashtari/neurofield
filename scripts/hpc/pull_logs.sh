#!/usr/bin/env bash
# Copy benchmark logs from HPC-UGent into logs/ of this checkout. Never deletes
# local files or overwrites ones that are newer than the remote copy.
#     scripts/hpc/pull_logs.sh      # sync
#     scripts/hpc/pull_logs.sh -n   # dry run
set -euo pipefail
cd "$(dirname "$0")/../.."

host=${HPC_HOST:-hpc-ugent}
root=${NEUROFIELD_ROOT:-$(ssh "$host" 'echo $VSC_DATA')/projects/neurofield}

mkdir -p logs
rsync -azu --info=stats1 "$@" "$host:$root/logs/" logs/

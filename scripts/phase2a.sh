#!/usr/bin/env bash
# Phase 2a screening (docs/SPEC.md 1.1 section 12): U-Net + ONE phase-2 option each, compared with r2_unet.
# Resumable: rerun after a shutdown.   bash scripts/phase2a.sh [python] [threads per job]
set -u
PY=${1:-python}; THREADS=${2:-4}
cd "$(dirname "$0")/.."
mkdir -p runs/logs
run() { $PY -m training.run2 --family unet --prefix p2 --suffix "$1" --threads "$THREADS" --options "$2" >> "runs/logs/p2_unet_$1.log" 2>&1; echo "done $1 (exit $?)"; }
run edge "--edge-relax" &
run ll "--latlon" &
run d4 "--rain-head two_part" &
run sg "--shock-loss 5" &
wait
$PY -m training.report --prefix p2_unet

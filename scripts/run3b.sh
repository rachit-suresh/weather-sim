#!/usr/bin/env bash
# Run 3b (docs/SPEC.md 1.6): run 2 recipe + RAIN-3 two-part rain head + EDG-2 edge relax.
# U-Net's run 3b model is p2_unet_d4edge (already trained). Resumable: rerun after a shutdown.
#   bash scripts/run3b.sh [python] [parallel jobs] [threads per job]
set -u
PY=${1:-python}; JOBS=${2:-3}; THREADS=${3:-3}
cd "$(dirname "$0")/.."
mkdir -p runs/logs
printf '%s\n' hybrid convgru cnn physics | xargs -P "$JOBS" -I{} bash -c \
  "$PY -m training.run2 --family {} --prefix r3b --threads $THREADS '--options=--rain-head two_part --edge-relax' >> runs/logs/r3b_{}.log 2>&1; echo \"done {} (exit \$?)\""
$PY -m training.report --prefix r3b_

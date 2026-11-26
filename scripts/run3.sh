#!/usr/bin/env bash
# Run 3 (docs/SPEC.md 1.5, phase 2b): run 2 recipe + RAIN-3 two-part rain head on all 5 families.
# U-Net doubles as a reproducibility check of p2_unet_d4st (spec 1.5).
# Resumable: rerun after a shutdown.   bash scripts/run3.sh [python] [parallel jobs] [threads per job]
set -u
PY=${1:-python}; JOBS=${2:-5}; THREADS=${3:-3}
cd "$(dirname "$0")/.."
mkdir -p runs/logs
printf '%s\n' cnn unet hybrid physics convgru | xargs -P "$JOBS" -I{} bash -c \
  "$PY -m training.run2 --family {} --prefix r3 --threads $THREADS '--options=--rain-head two_part' >> runs/logs/r3_{}.log 2>&1; echo \"done {} (exit \$?)\""
$PY -m training.report --prefix r3_

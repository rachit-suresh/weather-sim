#!/usr/bin/env bash
# Run 2 (docs/SPEC.md section 12): spec v1 recipe on the top 5 families, then post-processing stages.
# Resumable: rerun the same command after a shutdown; finished stages are skipped.
#   bash scripts/run2.sh [python] [parallel jobs] [threads per job]
set -u
PY=${1:-python}; JOBS=${2:-5}; THREADS=${3:-3}
cd "$(dirname "$0")/.."
mkdir -p runs/logs
printf '%s\n' cnn unet hybrid physics convgru | xargs -P "$JOBS" -I{} bash -c \
  "$PY -m training.run2 --family {} --threads $THREADS >> runs/logs/r2_{}.log 2>&1; echo \"done {} (exit \$?)\""
$PY -m training.report --prefix r2_

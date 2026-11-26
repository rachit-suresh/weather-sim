#!/usr/bin/env bash
# Finish the family sweep after a shutdown: (re)train only the families whose final model is missing.
# Neural jobs use --resume, so a job that saved per-epoch state (runs/ckpt/) continues where it stopped;
# jobs started before resume support existed start from scratch.
#   bash scripts/resume_sweep.sh [python] [threads per job]
set -u
PY=${1:-python}; THREADS=${2:-4}
CUR="1:5,2:3,4:3"
RECIPE="--rain none --var-loss 0 --spec-loss 0 --heavy-weight 1"     # sweep 1 recipe (pre-spec)
cd "$(dirname "$0")/.."
mkdir -p runs/logs
declare -A NET=( [cnn]="" [unet]="" [convgru]="" [physics]="" [hybrid]="" [vit]="" [fno]="" [gnn]="width=48,rounds=4" )
for fam in "${!NET[@]}"; do
  [ -f "artifacts/models/sweep_$fam.pt" ] && grep -q "sweep_$fam," runs/leaderboard.csv && continue
  extra=""; [ -n "${NET[$fam]}" ] && extra="--net ${NET[$fam]}"
  echo "training $fam"
  $PY -m training.train_nn --family "$fam" --name "sweep_$fam" --curriculum "$CUR" --threads "$THREADS" $RECIPE $extra --resume \
    >> "runs/logs/sweep_$fam.log" 2>&1 &
done
if ! grep -q "sweep_lgbm," runs/leaderboard.csv 2>/dev/null; then
  echo "training lgbm"; $PY -m training.train_lgbm --name sweep_lgbm --threads "$THREADS" > runs/logs/sweep_lgbm.log 2>&1 &
fi
wait
echo "sweep complete"; $PY -m training.report

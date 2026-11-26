#!/usr/bin/env bash
# Train one baseline per model family with the SAME data split, inputs, loss and schedule, then score each
# with training/evaluate.py (rows land in runs/leaderboard.csv). Logs: runs/logs/<name>.log
#   bash scripts/sweep.sh [python] [parallel jobs] [threads per job]
set -u
PY=${1:-python}; JOBS=${2:-5}; THREADS=${3:-3}
CUR="1:5,2:3,4:3 --rain none --var-loss 0 --spec-loss 0 --heavy-weight 1"   # sweep 1 recipe (pre-spec)
cd "$(dirname "$0")/.."
mkdir -p runs/logs
jobs_list=(
  "-m training.train_nn --family cnn     --name sweep_cnn     --curriculum $CUR --threads $THREADS"
  "-m training.train_nn --family unet    --name sweep_unet    --curriculum $CUR --threads $THREADS"
  "-m training.train_nn --family convgru --name sweep_convgru --curriculum $CUR --threads $THREADS"
  "-m training.train_nn --family physics --name sweep_physics --curriculum $CUR --threads $THREADS"
  "-m training.train_nn --family hybrid  --name sweep_hybrid  --curriculum $CUR --threads $THREADS"
  "-m training.train_nn --family vit     --name sweep_vit     --curriculum $CUR --threads $THREADS"
  "-m training.train_nn --family fno     --name sweep_fno     --curriculum $CUR --threads $THREADS"
  "-m training.train_nn --family gnn     --name sweep_gnn     --curriculum $CUR --threads $THREADS --net width=48,rounds=4"   # lighter: full size was ~4x CNN cost
  "-m training.train_lgbm --name sweep_lgbm --threads $THREADS"
)
printf '%s\n' "${jobs_list[@]}" | xargs -P "$JOBS" -I{} bash -c \
  'name=$(echo "{}" | sed -E "s/.*--name ([^ ]+).*/\1/"); '"$PY"' {} > runs/logs/$name.log 2>&1; echo "done $name (exit $?)"'

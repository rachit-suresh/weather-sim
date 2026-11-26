#!/bin/bash
# V2-CYC scoring, one process at a time (leaderboard.csv has no lock)
PY=/c/Users/vangs/cursor/devforge_2/.venv/Scripts/python.exe
export TWIN_THREADS=4
cd /c/Users/vangs/cursor/devforge_2/wt-cyc
for v in a b; do
  $PY -m training.evaluate --spec runs/cyc_$v.json --split val --tag v2_cyc_$v > runs/logs/v2_cyc_${v}_eval_val.log 2>&1
  $PY -m training.evaluate --spec runs/cyc_$v.json --split test --tag test_v2_cyc_$v > runs/logs/v2_cyc_${v}_eval_test.log 2>&1
  $PY -m training.v2_checks --spec runs/cyc_$v.json --tag v2_cyc_$v > runs/logs/v2_cyc_${v}_checks.log 2>&1
done
echo ALLDONE >> runs/logs/v2_cyc_b_checks.log

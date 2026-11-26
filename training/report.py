"""Print the leaderboard as a markdown table (latest row per tag).   python -m training.report [--prefix sweep_]"""
from __future__ import annotations

import argparse
import csv

from .data import ROOT

COLS = [("auto_events_of_70", "auto /70 (events)"), ("auto_random_of_70", "auto /70 (random)"), ("auto_named_of_70", "auto /70 (named)"), ("gate", "gate"),
        ("ss_events", "error skill (events)"), ("csi_skill_events", "heavy-rain skill (events)"),
        ("blur_pts_events", "blur /10"), ("causal_pts", "shocks /15"), ("ms_per_rollout", "ms / rollout"),
        ("r95_day1", "reach r95 d1"), ("rmax_day1", "reach max d1"), ("r95_day4", "reach r95 d4"), ("rmax_day4", "reach max d4")]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="")
    a = ap.parse_args(argv)
    rows = {}
    for r in csv.DictReader((ROOT / "runs" / "leaderboard.csv").open(newline="")):
        key = r["tag"] or r["model"]
        if key.startswith(a.prefix):
            rows[key] = r
    ordered = sorted(rows.values(), key=lambda r: -float(r["auto_events_of_70"] or 0))
    print("| model | " + " | ".join(h for _, h in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for r in ordered:
        print(f"| {r['model']} | " + " | ".join(r.get(k, "") or "-" for k, _ in COLS) + " |")


if __name__ == "__main__":
    main()

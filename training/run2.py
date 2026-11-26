"""Run 2 (spec section 12) for one family, end to end and resumable:
  1. train with the spec v1 recipe (train_nn defaults), resuming from runs/ckpt if interrupted
  2. score the raw model                                     -> leaderboard tag r2_<family>
  3. fit post-processing on training years (fit_post)        -> artifacts/post/r2_<family>.npz
  4. score +blend, +blend+rescale, +blend+rescale+rainmap    -> tags r2_<family>+blend / +rescale / +rainmap
Stages already present in the leaderboard are skipped.

    python -m training.run2 --family cnn --threads 3
    python -m training.run2 --family unet --prefix p2 --suffix edge --options "--edge-relax"     (phase 2 screening)
"""
from __future__ import annotations

import argparse
import csv
import os

from twin.config import ARTIFACTS

from .data import ROOT


def done_tags() -> set[str]:
    f = ROOT / "runs" / "leaderboard.csv"
    return {r["tag"] for r in csv.DictReader(f.open(newline=""))} if f.exists() else set()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--prefix", default="r2")
    ap.add_argument("--net", default="{}")
    ap.add_argument("--curriculum", default=None, help="override (smoke tests only)")
    ap.add_argument("--suffix", default="", help="variant name, e.g. edge -> p2_unet_edge")
    ap.add_argument("--options", default="", help="extra train_nn flags, e.g. '--edge-relax --shock-loss 5'")
    a = ap.parse_args(argv)
    os.environ["TWIN_THREADS"] = str(a.threads)
    from . import fit_post, train_nn
    from .evaluate import evaluate

    name = f"{a.prefix}_{a.family}" + (f"_{a.suffix}" if a.suffix else "")
    ckpt = ARTIFACTS / "models" / f"{name}.pt"
    base = {"model": "neural", "checkpoint": f"models/{name}.pt"}
    tags = done_tags()
    if name not in tags:
        if not (ckpt.exists() and (ROOT / "runs" / "ckpt" / f"{name}_done").exists()):
            argv_t = ["--family", a.family, "--name", name, "--threads", str(a.threads), "--net", a.net,
                      "--no-eval", "--resume"] + (["--curriculum", a.curriculum] if a.curriculum else []) + a.options.split()
            train_nn.main(argv_t)
            (ROOT / "runs" / "ckpt" / f"{name}_done").touch()
        evaluate(base, "val", name)
    if not (ARTIFACTS / "post" / f"{name}.npz").exists():
        fit_post.main(["--checkpoint", f"models/{name}.pt", "--name", name])
    stages = {"+blend": dict(blend=True, rescale=False, rainmap=False),
              "+rescale": dict(blend=True, rescale=True, rainmap=False),
              "+rainmap": dict(blend=True, rescale=True, rainmap=True)}
    for suffix, flags in stages.items():
        if name + suffix not in done_tags():
            evaluate({"model": "post", "base": base, "file": f"post/{name}.npz", **flags}, "val", name + suffix)
    print(f"{name}: all stages done")


if __name__ == "__main__":
    main()

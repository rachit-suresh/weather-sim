"""Run the judge's static repo scan (monsoon_r1.repo_scan) on exactly what git would ship.

Copies the tracked + staged files (git ls-files) into a temp folder, scans that, prints points and flags.
    python scripts/check_repo.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    from monsoon_r1.repo_scan import scan
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split("\n")
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "repo"
        for f in filter(None, files):
            src = ROOT / f
            if src.is_file():
                (dst / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst / f)
        res = scan(dst)
    print(json.dumps(res, indent=1))
    print(f"\nREPO SCAN: {res['points']}/15  (anti-cheat {res['anti_cheat_points']}/8, code health {res['code_health_points']}/7)")
    return 0 if res["points"] >= 15 else 1


if __name__ == "__main__":
    sys.exit(main())

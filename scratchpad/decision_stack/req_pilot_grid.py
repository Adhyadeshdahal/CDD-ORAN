"""Kaggle shard driver for the request-level pilot (design: req_pilot/PILOT_DESIGN.md, predeclared; no rollouts
were run locally). Labeller = sibling module req_pilot_label.py (cloud.py bundles both flat into e6dev/).

  PYTHONPATH=. python req_pilot_grid.py list
  PYTHONPATH=. python req_pilot_grid.py run --part i/k --out FILE.jsonl   # episodes e with e % k == i; resumable
Output: one jsonl line per (episode, k) decision second (fields seed, k, t, J0, rows[...], numpy version).
Analysis: python scratchpad/decision_stack/req_pilot/analyze.py --in a.jsonl,b.jsonl,...
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import req_pilot_label as L  # noqa: E402


def run(part, out):
    i, k = map(int, part.split("/"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for e in range(len(L.episodes())):
        if e % k != i:
            continue
        t = time.time()
        L.run_ep(e, [0, 1, 2], out)
        print(json.dumps({"episode": e, "seed": L.episodes()[e][2], "secs": round(time.time() - t, 1)}), flush=True)


if __name__ == "__main__":
    cmd, kw = sys.argv[1], dict(zip(sys.argv[2::2], sys.argv[3::2], strict=False))
    if cmd == "list":
        for e, ep in enumerate(L.episodes()):
            print(e, ep)
    elif cmd == "run":
        run(kw["--part"], kw["--out"])
    else:
        raise SystemExit(f"unknown command {cmd}")

"""Generation CPU-s and peak RSS per (world, regime, lam) at n = 24000, one fresh process per cell (DEV seed)."""
import json
import os
import subprocess
import sys

CELLS = [("E1", "R1", 1.0), ("E1", "R2", 1.0), ("E2", "R1", 1.0), ("E2", "R2", 1.0), ("E3", "R1", 1.0),
         ("E3", "R2", 1.0), ("E4", "R1", 1.0), ("E4", "R2", 1.0), ("E4", "R3", 0.0), ("E4", "R3", 1.5),
         ("E4", "R4", 0.0), ("E4", "R4", 1.5), ("E5", "R1", 1.0), ("E5", "R2", 1.0)]
CODE = """
import json, time, sys
from cdd_oran.xmethod.runner import PeakRSS
from cdd_oran.xmethod.worlds import generate_dataset
w, r, lam = sys.argv[1], sys.argv[2], float(sys.argv[3])
with PeakRSS() as m:
    t = time.process_time()
    ds, _ = generate_dataset(w, r, 24000, 3000000, lam=lam)
    dt = time.process_time() - t
print(json.dumps({"world": w, "regime": r, "lam": lam, "shape_A": list(ds.X_action.shape),
                  "cpu_s": round(dt, 3), "peak_rss_mb": round(m.mb, 1), "rss_scope": m.scope}))
"""
out = os.environ.get("JOB_OUT", ".")
with open(os.path.join(out, "gen_timing.jsonl"), "w") as fh:
    for w, r, lam in CELLS:
        for rep in range(2):
            res = subprocess.run([sys.executable, "-c", CODE, w, r, str(lam)], capture_output=True, text=True)
            line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else json.dumps({"err": res.stderr[-500:]})
            print(line, flush=True)
            fh.write(line + "\n")

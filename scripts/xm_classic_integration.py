"""End-to-end integration of the xm-classic adapters on the harness worlds (orchestrator step after the harness merge).

For each method and each (world, regime) cell it applies to (granger: E3 only; E4 at lam 1.5): tune on DEV seeds
``--tune-seeds`` (score-only methods: placebo tau, CONTRACT sec 5 / R-2; p-value methods: untuned), write a runner
spec with the frozen configs, run ``cdd_oran.xmethod.runner run`` on the test seed (one dataset per cell) and print
a summary of ``cdd_oran.xmethod.score`` per job. DEV seeds only (3_000_000 - 3_000_199). ``--arm`` (eq | native;
default = each method's default arm) is used for both the tau and the run (audit-classic2 A).

  uv run python scripts/xm_classic_integration.py --methods pc,notears,corr,granger --out runs/xm-classic-int/a
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cdd_oran.xmethod.runner import cell_key  # noqa: E402
from cdd_oran.xmethod.worlds import REGIMES_OF, generate_dataset  # noqa: E402

CLASSES = {"pc": "pc:PC", "notears": "notears:Notears", "shap_dag": "shap_dag:ShapDag",
           "two_tower": "two_tower:TwoTowerM", "corr": "corr:Corr", "granger": "granger:Granger"}
LAM = 1.5


def cells(method: str):
    for w, regs in REGIMES_OF.items():
        if method == "granger" and w != "E3":
            continue
        for r in regs:
            yield w, r, (LAM if w == "E4" else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="pc,notears,corr,granger")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3_000_000)
    ap.add_argument("--tune-seeds", default="3000001,3000002,3000003")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kappa", type=float, required=True,
                    help="R-24 observation noise for every world (the runner requires it; 0 = noiseless)")
    ap.add_argument("--arm", choices=("eq", "native"), default=None,
                    help="R-17 / R-18 arm for tune and run (default: the method's default arm)")
    a = ap.parse_args(argv)
    from cdd_oran.xmethod.methods.classic import METHODS
    os.makedirs(a.out, exist_ok=True)
    tseeds = [int(s) for s in a.tune_seeds.split(",")]
    assert all(3_000_000 <= s < 3_000_200 for s in tseeds + [a.seed]) and a.seed not in tseeds
    for m in a.methods.split(","):
        t0 = time.process_time()
        meth = METHODS[m]()
        arm = {} if a.arm is None else {"arm": a.arm}
        configs: dict = {"default": dict(arm)}
        cl = list(cells(m))
        for w, r, lam in cl:
            if meth.uses_p:
                continue
            dev = [generate_dataset(w, r, a.n, s, lam=1.0 if lam is None else lam, kappa=a.kappa)[0]
                   for s in tseeds]
            cfg = meth.tune(dev, config=arm)
            configs[cell_key(w, r, lam, a.n, a.kappa)] = {k: cfg[k] for k in ("tau", "tau_rule", "n_placebo_scores",
                                                                     "dev_seeds")}
        tune_cpu = time.process_time() - t0
        ref = f"cdd_oran.xmethod.methods.{CLASSES[m]}"
        spec = {"methods": [ref], "worlds": sorted({w for w, _, _ in cl}),
                "regimes": sorted({r for _, r, _ in cl}), "ns": [a.n], "seeds": [a.seed], "lams": [LAM], "kappas": [a.kappa],
                "configs": {ref: configs}}
        sp = os.path.join(a.out, f"spec_{m}.json")
        json.dump(spec, open(sp, "w"), indent=1)
        out = os.path.join(a.out, f"res_{m}.jsonl")
        subprocess.run([sys.executable, "-m", "cdd_oran.xmethod.runner", "run", "--spec", sp, "--part", "0/1",
                        "--out", out], check=True, cwd=_ROOT)
        print(json.dumps({"method": m, "tune_cpu_s": round(tune_cpu, 1), "spec": sp, "results": out}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

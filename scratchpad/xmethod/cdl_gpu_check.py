"""cdl GPU vs CPU (R-26: a GPU version must reproduce the CPU reference before use; brief cdl step 3).

Same dataset, same seed, same batches (row indices are drawn from the CPU generator in both cases): the CUDA fit
differs only by float non-associativity (and torch's random mask draws come from the CUDA generator, so the masks
differ). Reports max |CMI_gpu - CMI_cpu|, the Spearman correlation of all candidate scores, and whether the planted
edges outrank every null edge in both. Synthetic data only (``_classic_synth``, DEV-block seed).

  python scratchpad/xmethod/cdl_gpu_check.py --n 1000 --seed 3000000 --out out/gpu_check.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cdd_oran.xmethod.methods import _classic_synth as SYN  # noqa: E402
from cdd_oran.xmethod.methods.classic import METHODS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=3_000_000)
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--regime", default="R1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    from scipy.stats import spearmanr
    d, tr = SYN.make(a.n, a.seed, b=a.b, regime=a.regime)
    m = METHODS["cdl"]()
    res = {}
    for dev in ("cpu", "cuda"):
        t0 = time.time()
        r = m.run(d, {"device": dev})
        res[dev] = (r, time.time() - t0)
    rc, rg = res["cpu"][0], res["cuda"][0]
    sc = np.array([e.score for e in rc.edges])
    sg = np.array([e.score for e in rg.edges])
    keys = [(e.source, e.target) for e in rc.edges]
    planted = [i for i, k in enumerate(keys) if k in tr.edges]
    null = [i for i, k in enumerate(keys) if k in tr.null_edges]
    out = {"n": a.n, "seed": a.seed, "b": a.b, "regime": a.regime, "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "wall_s": {k: round(v[1], 1) for k, v in res.items()}, "cpu_s": {k: round(v[0].cpu_s, 1) for k, v in res.items()},
           "max_abs_diff": float(np.abs(sc - sg).max()), "max_score_cpu": float(sc.max()),
           "spearman": float(spearmanr(sc, sg).statistic),
           "planted_min_cpu": float(sc[planted].min()), "null_max_cpu": float(sc[null].max()),
           "planted_min_gpu": float(sg[planted].min()), "null_max_gpu": float(sg[null].max()),
           "separates_cpu": bool(sc[planted].min() > sc[null].max()),
           "separates_gpu": bool(sg[planted].min() > sg[null].max()),
           "scores": {f"{s}->{t}": [round(float(x), 6), round(float(y), 6)] for (s, t), x, y in zip(keys, sc, sg,
                                                                                                    strict=True)}}
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "scores"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

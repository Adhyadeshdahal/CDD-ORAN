"""F3 reproduction of recorded results of completed older studies (ruling R-11: stored datasets read-only).

  python scratchpad/xmethod/citests/f3_frozen.py PART ROOT [--out DIR]     PART in mscr | rcot2 | pdcor
  ROOT = the main CDD-ORAN checkout (holds runs/ and scratchpad/p0k5_fp_calibration, never written).

Dataset: runs/e2slice-recovery/replicate-00/rows.npz (E2, n 4000, the frozen dataset of the e2slice studies) as an
api.Dataset: actions P0..P7 = x_params, lagged KPIs K0..K5 = x_kpis, Y = y_kpis, no placebo.
mscr : (data: the stored replicate-00 rows PREDATE the E2 operating-point redesign 9d87a60, so stage B's input is
       regenerated in memory by the frozen generator, E2DatasetConfig(4000, seed 0); dataset.py / e2.py sha256 equal
       the stage-B provenance) native config (B 2999 fixed, frozen bank streams [seed, j, 2999], seed 0), candidates = the 48 param->KPI
       (v2 family) -> must equal the stored stage-B seed-0 p-values (scratchpad/p0k5_fp_calibration/
       stageB_results.json, B = 2999, harness mscr_v2.py) EXACTLY.
rcot2: native config (B 299 fixed, rng_seed 0), all 84 candidates -> statistic, p-values, per-target BH mask must
       equal runs/e2slice-recovery/replicate-00/discovery_rcot_v2.json exactly (statistic to 1e-12 relative).
pdcor: native config (B 999 fixed, permutation_seed 0): signed pdCor of all 84 candidates == discovery.json
       (1e-12); permutation p-values on 6 candidates (fast loop; fp summation order may move a tie: |diff| <= 2/1000).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from cdd_oran.xmethod import api


def e2_dataset(root, rep="replicate-00", candidates="all", regenerate=False):
    if regenerate:      # the stage-B input: frozen generator, seed 0, in memory only (code sha256 == stage B's)
        from cdd_oran.e2slice.dataset import E2DatasetConfig, generate_rows

        z = generate_rows(E2DatasetConfig(n_rows_per_seed=4000, seed=0))
        z = {k: getattr(z, k) for k in ("x_params", "x_kpis", "y_kpis")}
    else:
        z = np.load(os.path.join(root, "runs", "e2slice-recovery", rep, "rows.npz"))
    xp, xk, y = z["x_params"].astype(float), z["x_kpis"].astype(float), z["y_kpis"].astype(float)
    acts = tuple(f"P{i}" for i in range(xp.shape[1]))
    kpis = tuple(f"K{j}" for j in range(y.shape[1]))
    srcs = acts if candidates == "params" else acts + kpis
    # candidate order = frozen grid order (target j, source i) so results map to [j][i]
    cands = tuple((s, k) for k in kpis for s in srcs)
    d = api.Design(kind="iid")
    ds = api.Dataset(world="E2-frozen", regime="R1", n=len(y), seed=0, action_names=acts, kpi_names=kpis,
                     X_action=xp, X_kpi_lag=xk, Y=y, designs=tuple(d for _ in acts), candidates=cands)
    return ds, acts + kpis


def part_mscr(root):
    from cdd_oran.xmethod.methods.mscr import MSCRMethod

    ds, _ = e2_dataset(root, candidates="params", regenerate=True)
    m = MSCRMethod()
    r = m.run(ds, {**m.native_config(), "seed": 0})
    ref = json.load(open(os.path.join(root, "scratchpad", "p0k5_fp_calibration", "stageB_results.json")))
    raw = next(x for x in ref["raw"] if x["seed"] == 0)["p"]
    ours = {(e.source, e.target): e.p for e in r.edges}
    diffs = [abs(ours[(f"P{i}", f"K{j}")] - raw[str(j)][i]) for j in range(6) for i in range(8)]
    return {"n_compared": len(diffs), "max_abs_diff": max(diffs), "exact": max(diffs) == 0.0, "cpu_s": r.cpu_s}


def part_rcot2(root):
    from cdd_oran.xmethod.methods.rcot2 import RCoT2Method

    ds, srcs = e2_dataset(root)
    m = RCoT2Method()
    r = m.run(ds, {**m.native_config(), "rng_seed": 0})
    ref = json.load(open(os.path.join(root, "runs", "e2slice-recovery", "replicate-00", "discovery_rcot_v2.json")))
    st, pv, mask = (np.array(ref[k], dtype=float) for k in ("statistic", "pvalues", "binary_mask"))
    nat = r.notes["declared_native"]
    rel, pd, md = [], [], 0
    for e, d in zip(r.edges, nat, strict=True):
        j, i = int(e.target[1:]), srcs.index(e.source)
        rel.append(abs(e.score - st[j, i]) / max(abs(st[j, i]), 1e-300))
        pd.append(abs(e.p - pv[j, i]))
        md += int(bool(d) != bool(mask[j, i]))
    return {"n_compared": len(rel), "max_rel_stat_diff": max(rel), "max_abs_p_diff": max(pd),
            "mask_mismatches": md, "pass": max(rel) <= 1e-12 and max(pd) == 0 and md == 0, "cpu_s": r.cpu_s}


def part_pdcor(root):
    import dataclasses

    from cdd_oran.xmethod.methods.pdcor import PDCorMethod

    ds, srcs = e2_dataset(root)
    ref = json.load(open(os.path.join(root, "runs", "e2slice-recovery", "replicate-00", "discovery.json")))
    sp = np.array([[np.nan if v is None else v for v in row] for row in ref["signed_pdcor"]])
    pp = np.array(ref["perm_pvalues"], dtype=float)
    m = PDCorMethod()
    r = m.run(ds, {**m.native_config(), "permutation_seed": 0, "b_perm": 1})     # statistics of all 84
    sd = [abs(s - sp[int(e.target[1:]), srcs.index(e.source)])
          for s, e in zip(r.notes["signed_pdcor"], r.edges, strict=True)]
    sub = (("P0", "K5"), ("P0", "K0"), ("P6", "K5"), ("P3", "K1"), ("K0", "K1"), ("K5", "K5"))
    r2 = m.run(dataclasses.replace(ds, candidates=sub), {**m.native_config(), "permutation_seed": 0})
    pdiff = {f"{e.source}->{e.target}": [e.p, float(pp[int(e.target[1:]), srcs.index(e.source)])] for e in r2.edges}
    worst = max(abs(a - b) for a, b in pdiff.values())
    return {"n_stat": len(sd), "max_abs_signed_diff": float(np.nanmax(sd)), "p_subset": pdiff, "max_abs_p_diff": worst,
            "pass": float(np.nanmax(sd)) <= 1e-12 and worst <= 2 / 1000, "cpu_s": r.cpu_s + r2.cpu_s}


if __name__ == "__main__":
    part, root = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "scratchpad/xmethod/results/citests"
    res = {"mscr": part_mscr, "rcot2": part_rcot2, "pdcor": part_pdcor}[part](root)
    print(json.dumps(res, indent=1, default=float))
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"F3_FROZEN_{part}.json"), "w"), indent=1, default=float)

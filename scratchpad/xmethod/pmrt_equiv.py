"""Equivalence of pmrt_core (cdd_oran/xmethod/methods/pmrt_core.py) with pmrt.py on E6-P cache samples.

  uv run python scratchpad/xmethod/pmrt_equiv.py --small DIR [--B 999] [--out F.json]

DIR holds the small caches placebo1.npz / plxc2.npz (split 9) and dev1.npz (split 0) of pmrt_equivalence.py
(main checkout: .tmp/mscr_plus/infra/cache). Params: the PMRT artifact docs/benchmark/artifacts/E6P_PMRT_V4.json.

(A) ENGINE (filter disabled, same weights; also (A') for the unclipped "plain" arm against pmrt's own counting): per family, the plain_c arm's predictable weight columns of pmrt
    (pmrt_bench.lean_columns, kernel 1 on bins [0, 90) + running centre + Huber clip, all frozen from ev2) are fed to
    pmrt_core.crt with the core's own logged-categorical assignment (values LEVEL_V2, propensity = the unit's pi0 row;
    sgn folded into the weights: sum sgn (L - mu) w = sum (L - mu) (sgn w)) and pmrt's RNG stream. Claim: p2, p_plus,
    p_minus identical to pmrt_bench.integrated_family's plain_c (bit-exact), z within 1e-12 (relative).
(B) OWN ADJUSTMENT: the core's past-only ridge + Huber on the unit table (Y = the unit outcome ud.y (post - pre, 90 s),
    X = pre-window sums of every (rel, kpi), ctx, hist, sgn, t0 / T) in pmrt's information order, same assignment and
    RNG stream. Not expected to be identical (adjustment fitted on the same episodes past-only vs on ev2 and frozen;
    no slots / bins; no episode x sgn strata): reported as z correlation, |dz| and p <= .05 agreement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, os.path.join(ROOT, "scratchpad", "e6_dev")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import pmrt_artifacts as PAR  # noqa: E402
import pmrt_bench as PB  # noqa: E402

from cdd_oran.decision import pmrt as PM  # noqa: E402
from cdd_oran.decision.crt_units import FAMILIES, LEVEL_ARR  # noqa: E402
from cdd_oran.decision.crt_units_v2 import LEVEL_V2_ARR  # noqa: E402
from cdd_oran.xmethod import api  # noqa: E402
from cdd_oran.xmethod.methods import pmrt_core as PC  # noqa: E402

SMALL = (("placebo1", 9), ("plxc2", 9), ("dev1", 0))
ART = os.path.join(ROOT, "docs", "benchmark", "artifacts", "E6P_PMRT_V4.json")


def family_compare(pd, f, params, cfg, split):
    ud = pd.ud
    fi = FAMILIES.index(f)
    rows0 = ud.rows_of(f)
    lv = LEVEL_ARR[ud.mode[rows0]]
    if (len(rows0) < cfg.min_units or (lv == 1.0).sum() < cfg.min_accept or (lv == 0.0).sum() < cfg.min_reject
            or len(np.unique(ud.episode[rows0])) < cfg.min_episodes):
        return None
    ref = PB.integrated_family(pd, f, params, cfg, split)
    rows, W, targets, meta = PB.lean_columns(pd, f, params, cfg)
    na = len(PB.COL_ARMS)
    ia = PB.COL_ARMS.index("plain_c")
    live = [t for t in range(len(targets)) if meta[t] is not None]
    Wc = W[:, [t * na + ia for t in live]]
    sg = ud.sgn[rows].astype(float)
    des = api.Design("logged", {"name": "categorical", "values": LEVEL_V2_ARR}, propensity=ud.probs[rows])
    asg = PC.assignment(des, LEVEL_V2_ARR[ud.mode[rows]])
    # (A) engine with pmrt's weights
    outA = PC.crt(asg.v, asg.var, sg[:, None] * Wc, asg.draw, PM._stream(cfg, fi, split), cfg.B, cfg.chunk,
                  cfg.max_chunk_bytes)
    # (B) the core's own predictable adjustment on the unit table
    Y = np.column_stack([ud.y[targets[t]][rows] for t in live])
    pre = np.column_stack([ud.pre[k][rows] for k in sorted(ud.pre)])
    X = np.column_stack([np.nan_to_num(pre), np.nan_to_num(pd.ctx[rows]), pd.hist[rows], sg, pd.tfrac[rows]])
    Wb, info = PC.predictable_weights(X, Y, PC.PmrtCoreConfig())
    outB = PC.crt(asg.v, asg.var, sg[:, None] * Wb, asg.draw, PM._stream(cfg, fi, split), cfg.B, cfg.chunk,
                  cfg.max_chunk_bytes)
    # (A') the same engine check for pmrt's unclipped "plain" arm (reference: the plain column's own counts)
    ip = PB.COL_ARMS.index("plain")
    Wp = W[:, [t * na + ip for t in live]]
    outP = PC.crt(asg.v, asg.var, sg[:, None] * Wp, asg.draw, PM._stream(cfg, fi, split), cfg.B, cfg.chunk,
                  cfg.max_chunk_bytes)
    refP = plain_reference(ud, rows, Wp, sg, cfg, fi, split)
    rec = []
    for j, t in enumerate(live):
        r = ref[targets[t]]["plain_c"]
        rec.append({"hyp": "|".join((f,) + targets[t]), "ref_p2": r["p2"], "ref_pp": r["p_plus"], "ref_pm": r["p_minus"],
                    "ref_z": r["z_raw"], "A_p2": float(outA["p2"][j]), "A_pp": float(outA["p_plus"][j]),
                    "A_pm": float(outA["p_minus"][j]), "A_z": float(outA["z"][j]), "B_p2": float(outB["p2"][j]),
                    "B_z": float(outB["z"][j]), "P_p2": float(outP["p2"][j]), "P_ref_p2": float(refP["p2"][j]),
                    "P_z": float(outP["z"][j]), "P_ref_z": float(refP["z"][j])})
    return {"n_units": int(len(rows)), "adjust": info, "rec": rec}


def plain_reference(ud, rows, W, sg, cfg, fi, split):
    """pmrt's own CRT arithmetic (family_tests_pmrt: v_design, PiAssignment draws, |V W| / sd counting) on the plain
    arm's weights, written out here because pmrt_bench.integrated_family reports plain_c / loadsp_c only."""
    from cdd_oran.decision.crt_units import PiAssignment
    v = PM.v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows])
    var = PM.v_var(ud.probs[rows])
    sd = np.sqrt(var @ (W * W))
    sd_safe = np.where(sd > 0, sd, 1.0)
    z = (v @ W) / sd_safe
    tol = 1e-9 * (1.0 + np.abs(z))
    pa, rng = PiAssignment(ud), PM._stream(cfg, fi, split)
    ch = int(max(1, min(cfg.chunk, cfg.max_chunk_bytes // (8 * max(len(rows), 1)))))
    mu, cnt, done = ud.probs[rows] @ LEVEL_V2_ARR, np.zeros(W.shape[1], np.int64), 0
    f = FAMILIES[fi]
    while done < cfg.B:
        b = min(ch, cfg.B - done)
        V = (LEVEL_V2_ARR[pa.conditional_draws(rng, f, b, rows=rows, what="mode")] - mu[None, :]) * sg[None, :]
        cnt += np.count_nonzero(np.abs(V @ W) / sd_safe[None, :] >= (np.abs(z) - tol)[None, :], axis=0)
        done += b
    return {"p2": (1 + cnt) / (cfg.B + 1.0), "z": z}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    params, _ = PAR.load_artifact(ART)
    cfg = PAR.pmrt_config(params, a.B)
    out = {"B": a.B, "artifact": os.path.relpath(ART, ROOT).replace("\\", "/"), "caches": {}}
    allrec = []
    for nm, split in SMALL:
        t = time.time()
        pd = PM.pmrt_data(PM.load_pmrt_pool([os.path.join(a.small, f"{nm}.npz")]))
        fam = {}
        for f in FAMILIES:
            r = family_compare(pd, f, params, cfg, split)
            if r is not None:
                fam[f] = r
                allrec += r["rec"]
        out["caches"][nm] = {"split": split, "n_units": int(pd.n), "families": fam, "wall_s": round(time.time() - t, 1)}
        print(f"{nm}: {len(fam)} families, {time.time() - t:.1f} s", flush=True)
    R = allrec
    exactA = all(r["A_p2"] == r["ref_p2"] and r["A_pp"] == r["ref_pp"] and r["A_pm"] == r["ref_pm"] for r in R)
    dzA = max(abs(r["A_z"] - r["ref_z"]) / max(1.0, abs(r["ref_z"])) for r in R)
    zr, zb = np.array([r["ref_z"] for r in R]), np.array([r["B_z"] for r in R])
    pr, pb = np.array([r["ref_p2"] for r in R]), np.array([r["B_p2"] for r in R])
    out["summary"] = {
        "n_hyp": len(R),
        "A_p_bit_exact": bool(exactA), "A_max_rel_dz": float(dzA),
        "Aplain_p_bit_exact": bool(all(r["P_p2"] == r["P_ref_p2"] for r in R)),
        "Aplain_max_rel_dz": float(max(abs(r["P_z"] - r["P_ref_z"]) / max(1.0, abs(r["P_ref_z"])) for r in R)),
        "B_corr_z": float(np.corrcoef(zr, zb)[0, 1]), "B_median_abs_dz": float(np.median(np.abs(zr - zb))),
        "B_max_abs_dz": float(np.max(np.abs(zr - zb))),
        "B_agree_p05": float(np.mean((pr <= 0.05) == (pb <= 0.05))),
        "B_n_p05_ref": int((pr <= 0.05).sum()), "B_n_p05_core": int((pb <= 0.05).sum()),
        "B_sign_agree_where_ref_p05": float(np.mean(np.sign(zr[pr <= 0.05]) == np.sign(zb[pr <= 0.05])))
        if (pr <= 0.05).any() else None,
    }
    print(json.dumps(out["summary"], indent=1))
    if a.out:
        with open(a.out, "w", newline="\n") as fh:
            json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()

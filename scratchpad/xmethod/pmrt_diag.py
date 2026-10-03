"""pmrt-diag: why is pmrt_eq liberal on E2 R2 truth-null edges at n 1000? (brief scratchpad/xmethod/briefs/pmrt-diag.md)

  uv run python scratchpad/xmethod/pmrt_diag.py --world E2 --regime R2 --n 1000 --kappa 0.5 \
      --seeds 3000100-3000119 --reps -1,0,1 --variants prod,noown,inv --out F.jsonl [--check]

Replicates (--reps): rep -1 = the DEV dataset itself; rep r >= 0 = the same DEV seed with every dither column (the
"actions" and "placebo" streams; R1: the whole i.i.d. action draw) redrawn from its KNOWN design on a diagnostic
stream SeedSequence([7801, 99, seed, world, regime, purpose, r]); setpoints and observation noise are the seed's own.
These are draws of the design itself (what the CRT null says the world would produce), not new corpus seeds.

E2 fact (envs/v2/e2.py): KPI_{t+1} = f(actions at t) + observation noise; no KPI memory. For a truth-null (a, k) the
series Y_k is therefore invariant to ALL of a's dithers (the global sharp null H0g(a, k) holds), so a W that is a
function of (Y_k, quantities independent of a's dither stream) makes the fixed-W CRT EXACT.

Variants of the weights W (everything else = production pmrt_core: ridge GCV, expanding window, no clip, BC h 20):
  prod      production eq: lag KPIs, setpoints, all actions @t-1/@t-2, concurrent designed actions (bit-identical
            to PmrtCore().run(), checked by --check)
  r3        production covariates="r3": lag KPIs + concurrent
  noown     prod minus the focal action's own @t-1/@t-2                    (H1, own-lag path)
  nolagkpi  prod minus every lag_kpi                                        (H1/H3, lagged-KPI path)
  inv       prod minus own lags, minus every lag_kpi except the target's own (per-target fit): W invariant to the
            focal dither under H0g -> exact CRT in E2
  inv_r3    r3 minus every lag_kpi except the target's own
  noconc    prod minus the concurrent actions                               (H4)
  nohist    prod minus every lagged action
  prodB     prod with fixed B = 9999 (seq_h None)                           (H5)
  invB      inv with fixed B = 9999
  pcorr_eq  NOT pmrt: the pcorr adapter's eq arm on the same dataset (same-data comparator for H6)
Records (JSON lines): one per (seed, rep): {seed, rep, variant: {edge: [p, z]}} for the truth-null action edges and
the P_placebo edges (--power: every primary candidate, plus "true" and "family" lists for recall / BY).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from cdd_oran.xmethod.covariates import design_covariates, info_order  # noqa: E402
from cdd_oran.xmethod.methods import pmrt_core as PC  # noqa: E402
from cdd_oran.xmethod.worlds import generate as G  # noqa: E402

DIAG_TAG = 99
VARIANTS = ("prod", "r3", "noown", "nolagkpi", "inv", "inv_r3", "noconc", "nohist", "prodB", "invB", "pcorr_eq")


def gen(world, regime, n, seed, kappa, rep):
    """The DEV dataset (rep -1) or a dither-regenerated replicate of it (module docstring)."""
    if rep < 0:
        return G.generate_dataset(world, regime, n, seed, kappa=kappa)
    orig = G._rng

    def patched(seed_, world_, regime_, purpose):
        if purpose in ("actions", "placebo"):
            return np.random.default_rng(np.random.SeedSequence(
                [PC.RNG_TAG, DIAG_TAG, int(seed_), G._WORLD_CODE[world_], G._REGIME_CODE[regime_],
                 G._PURPOSE[purpose], int(rep)]))
        return orig(seed_, world_, regime_, purpose)
    G._rng = patched
    try:
        return G.generate_dataset(world, regime, n, seed, kappa=kappa)
    finally:
        G._rng = orig


def _cols(names, keep):
    return [i for i, nm in enumerate(names) if keep(nm)]


def variant_X(data, ai, variant, k_name=None):
    """Covariate matrix (information order) of focal action ai for a variant (k_name: per-target variants)."""
    a = data.action_names[ai]
    base = variant.rstrip("B") if variant in ("prodB", "invB") else variant
    eq = base not in ("r3", "inv_r3")
    names, M, _ = design_covariates(data, include_setpoints=eq, include_lagged_actions=eq, lags=2, focal=ai)
    names = list(names)
    own = (f"{a}@t-1", f"{a}@t-2")
    if base in ("prod", "r3"):
        keep = lambda nm: True  # noqa: E731
    elif base == "noown":
        keep = lambda nm: nm not in own  # noqa: E731
    elif base == "nolagkpi":
        keep = lambda nm: not nm.startswith("lag_kpi:")  # noqa: E731
    elif base in ("inv", "inv_r3"):
        keep = lambda nm: nm not in own and (not nm.startswith("lag_kpi:") or nm == f"lag_kpi:{k_name}")  # noqa: E731
    elif base == "noconc":
        keep = lambda nm: not nm.startswith("concurrent:")  # noqa: E731
    elif base == "nohist":
        keep = lambda nm: "@t-" not in nm  # noqa: E731
    else:
        raise ValueError(variant)
    idx = _cols(names, keep)
    return M[info_order(data)][:, idx], [names[i] for i in idx]


PER_TARGET = ("inv", "inv_r3", "invB")


def run_pcorr_eq(data, targets_of):
    """Same-data comparator: the pcorr adapter's eq arm (partial-correlation t-test given Z_eq, R-17); z = the
    signed normal quantile of its p."""
    from scipy import stats

    from cdd_oran.xmethod.methods.pcorr import PCorrMethod
    res = PCorrMethod().run(data, {"arm": "eq"})
    want = {(a, k) for a, ks in targets_of.items() for k in ks}
    return {(e.source, e.target): (float(e.p), float(e.sign * stats.norm.isf(e.p / 2)))
            for e in res.edges if (e.source, e.target) in want and e.p is not None}


def run_variant(data, variant, targets_of, cfg=None):
    """{(a, k): (p2, z)} for the requested (action -> [kpi]) map, production arithmetic with variant weights."""
    if variant == "pcorr_eq":
        return run_pcorr_eq(data, targets_of)
    cfg = cfg or PC.PmrtCoreConfig()
    if variant in ("prodB", "invB"):
        cfg = dataclasses.replace(cfg, seq_h=None)
    order = info_order(data)
    kpis = list(data.kpi_names)
    Y = np.asarray(data.Y, float)[order]
    bad_t = ~np.all(np.isfinite(Y), 0) | (np.nanstd(Y, 0) <= 0)
    Yf = np.where(np.isfinite(Y), Y, 0.0)
    out = {}
    for ai, a in enumerate(data.action_names):
        tk = [k for k in targets_of.get(a, []) if not bad_t[kpis.index(k)]]
        if not tk:
            continue
        asg = PC.assignment(data.designs[ai], np.asarray(data.X_action, float)[:, ai])
        if isinstance(asg, str) or not np.any(asg.var > 0):
            continue
        ki = [kpis.index(k) for k in tk]
        if variant in PER_TARGET:
            W = np.column_stack([PC.predictable_weights(variant_X(data, ai, variant, k)[0], Yf[:, [kpis.index(k)]],
                                                        cfg)[0][:, 0] for k in tk])
        else:
            # the production fit is multi-target with per-target GCV: fitting all K columns and selecting = identical
            W = PC.predictable_weights(variant_X(data, ai, variant)[0], Yf, cfg)[0][:, ki]
        rng = np.random.default_rng([int(data.seed), PC.RNG_TAG, ai, int(cfg.split)])

        def draw_ordered(r, b, draw=asg.draw):
            return draw(r, b)[:, order]
        res = PC.crt(asg.v[order], asg.var[order], W, draw_ordered, rng, cfg.B, cfg.chunk, cfg.max_chunk_bytes,
                     cfg.seq_h)
        for j, k in enumerate(tk):
            out[(a, k)] = (float(res["p2"][j]), float(res["z"][j]))
    return out


def null_targets(ds, tr, power=False):
    """action -> [KPI] of the truth-null primary candidates (power: every primary candidate)."""
    t = {}
    for a, k in ds.meta["primary_candidates"]:
        if power or (a, k) in tr.null_edges:
            t.setdefault(a, []).append(k)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="E2")
    ap.add_argument("--regime", default="R2")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--seeds", default="3000100-3000119")
    ap.add_argument("--reps", default="-1")
    ap.add_argument("--variants", default="prod")
    ap.add_argument("--check", action="store_true", help="assert prod == PmrtCore().run() on every dataset")
    ap.add_argument("--power", action="store_true", help="every primary candidate (true edges too: B up to 9999)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    if any(not (3_000_000 <= s <= 3_000_199) for s in seeds):
        raise SystemExit("DEV seeds only (3_000_000 - 3_000_199)")
    reps = [int(r) for r in a.reps.split(",")] if "," in a.reps or a.reps.lstrip("-").isdigit() else None
    if reps is None:                                  # "a:b" range
        r0, r1 = a.reps.split(":")
        reps = list(range(int(r0), int(r1)))
    vs = a.variants.split(",")
    assert all(v in VARIANTS for v in vs), vs
    done = set()
    if os.path.exists(a.out):                         # resumable
        with open(a.out) as fh:
            for line in fh:
                d = json.loads(line)
                done.add((d["seed"], d["rep"]))
    t0 = time.time()
    with open(a.out, "a", newline="\n") as fh:
        for s in seeds:
            for r in reps:
                if (s, r) in done:
                    continue
                ds, tr = gen(a.world, a.regime, a.n, s, a.kappa, r)
                tg = null_targets(ds, tr, a.power)
                rec = {"world": a.world, "regime": a.regime, "n": a.n, "kappa": a.kappa, "seed": s, "rep": r}
                if a.power:
                    rec["true"] = sorted(f"{x}->{y}" for x, y in tr.edges if x in ds.action_names)
                    rec["family"] = [f"{x}->{y}" for x, y in ds.meta["primary_candidates"]]
                for v in vs:
                    res = run_variant(ds, v, tg)
                    rec[v] = {f"{x}->{y}": [round(pz[0], 6), round(pz[1], 5)] for (x, y), pz in res.items()}
                    if a.check and v == "prod":
                        ref = PC.PmrtCore().run(ds)
                        by = {(e.source, e.target): e.p for e in ref.edges}
                        bad = [(e, by[e], pz[0]) for e, pz in res.items() if by[e] != pz[0]
                               or abs(ref.notes["z"][f"{e[0]}->{e[1]}"] - pz[1]) > 1e-12 * (1 + abs(pz[1]))]
                        # z: column-subset BLAS order differs at ~1e-14; p must be identical
                        if bad:
                            raise AssertionError(f"prod != PmrtCore.run: {bad[:3]}")
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            print(f"seed {s} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

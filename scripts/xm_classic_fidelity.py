"""Fidelity gates of the xm-classic adapters (CONTRACT sec 4: F2, F3, F4) and the cost table.

  uv run python scripts/xm_classic_fidelity.py f3_e2     [--rows-root ../../CDD-ORAN/runs/e2slice-recovery]
  uv run python scripts/xm_classic_fidelity.py f3_notears          (needs igraph: uv run --with python-igraph ...)
  uv run python scripts/xm_classic_fidelity.py f3_pc
  uv run python scripts/xm_classic_fidelity.py f4 --method corr --reps 200 --n 1000
  uv run python scripts/xm_classic_fidelity.py cost --method pc --n 4000      (one run per process: clean peak RSS)

Every output is a JSON line on stdout. Synthetic data only (``_classic_synth``) and the read-only E2 replicate rows
of the earlier E2 study; no study world, no DEV / EVAL corpus is generated or read here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cdd_oran.xmethod import api  # noqa: E402
from cdd_oran.xmethod.methods import _classic_synth as SYN  # noqa: E402
from cdd_oran.xmethod.methods._classic_common import PLACEBO, peak_rss_mb, placebo_tau  # noqa: E402
from cdd_oran.xmethod.methods.classic import METHODS  # noqa: E402

E2_REF = {  # earlier E2 study outputs (main checkout scratchpad, read-only), tau_rel = .10, seed 0
    "two_tower": "scratchpad/gnn_baseline/gnn_results.json",
    "shap_dag": "scratchpad/shap_dag/shap_dag_lean.json",
}


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(c - h), 4), round(float(c + h), 4)]


# ------------------------------------------------------------------------------------------------ F3: E2 behaviour
def e2_dataset(path: str, rep: int) -> api.Dataset:
    d = np.load(path)
    X, K, Y = d["x_params"].astype(np.float64), d["x_kpis"].astype(np.float64), d["y_kpis"].astype(np.float64)
    an = tuple(f"P{i}" for i in range(X.shape[1]))
    kn = tuple(f"K{i}" for i in range(Y.shape[1]))
    return api.Dataset(world="E2", regime="E2-recovery", n=len(X), seed=rep, action_names=an, kpi_names=kn,
                       X_action=X, X_kpi_lag=K, Y=Y, designs=tuple(api.Design(kind="iid") for _ in an),
                       candidates=tuple((a, k) for a in an for k in kn))


def f3_e2(rows_root: str, methods=("shap_dag", "two_tower"), config: dict | None = None):
    """Replay the adapters on the earlier E2 replicate rows (seed 0; ``config`` overrides, e.g. shap_dag's
    regressor); yields one record per (method, replicate)."""
    paths = sorted(glob.glob(os.path.join(rows_root, "replicate-*", "rows.npz")))
    for name in methods:
        m = METHODS[name]()
        for path in paths:
            rep = int(os.path.basename(os.path.dirname(path)).split("-")[1])
            r = m.run(e2_dataset(path, rep), {"seed": 0, **(config or {})})
            sc = {(e.source, e.target): e.score for e in r.edges}
            nat = set(r.notes["native_declared"])
            row = {a: sc[(a, "K5")] for a in [f"P{i}" for i in range(8)]}   # shap score / sd: ratio unchanged
            yield {"method": name, "config": config or {}, "rep": rep, "has_P0_K5": "P0->K5" in nat, "n_param_edges": len(nat),
                   "P0_K5_ratio": round(row["P0"] / max(row.values()), 4), "cpu_s": round(r.cpu_s, 1)}


def f3_e2_compare(records: list[dict], ref_root: str) -> dict:
    out = {}
    for name in sorted({r["method"] for r in records}):
        per = sorted((r for r in records if r["method"] == name), key=lambda r: r["rep"])
        ref = json.load(open(os.path.join(ref_root, E2_REF[name])))
        if name == "two_tower":
            rref = [{"rep": int(k), "has_P0_K5": v["has_P0_K5"], "n_param_edges": v["n_param_edges"],
                     "P0_K5_ratio": round(v["P0_K5_score_ratio"], 4)} for k, v in ref["primary_per_replicate"].items()]
        else:
            rref = [{"rep": v["rep"], "has_P0_K5": v["has_P0K5"], "n_param_edges": v["n_edges"],
                     "P0_K5_ratio": round(v["shap_ratio"], 4)} for v in ref["per_rep"]]
        out[name] = {"ref_file": E2_REF[name], "n_reps": len(per),
                     "recovered_ours": sum(p["has_P0_K5"] for p in per),
                     "recovered_ref": sum(p["has_P0_K5"] for p in rref),
                     "n_param_edges_ours": [p["n_param_edges"] for p in per],
                     "n_param_edges_ref": [p["n_param_edges"] for p in rref],
                     "mean_ratio_ours": round(float(np.mean([p["P0_K5_ratio"] for p in per])), 4),
                     "mean_ratio_ref": round(float(np.mean([p["P0_K5_ratio"] for p in rref])), 4),
                     "cpu_s_per_rep": round(float(np.mean([p["cpu_s"] for p in per])), 1)}
    return out


# ------------------------------------------------------------------------------------------------ F3: NOTEARS README
def f3_notears(utils_path: str) -> dict:
    """Authors' README example: set_random_seed(1); n 100, d 20, s0 20, ER, gauss; lambda1 .1 ->
    {'fdr': 0.0, 'tpr': 1.0, 'fpr': 0.0, 'shd': 0, 'nnz': 20}."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("nt_utils", utils_path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    from cdd_oran.xmethod.methods._vendor.notears.linear import notears_linear
    from cdd_oran.xmethod.methods.notears import notears_linear_bk
    utils.set_random_seed(1)
    B = utils.simulate_dag(20, 20, "ER")
    W = utils.simulate_parameter(B)
    X = utils.simulate_linear_sem(W, 100, "gauss")
    W1 = notears_linear(X, lambda1=0.1, loss_type="l2")
    W2 = notears_linear_bk(X, lambda1=0.1, loss_type="l2")
    return {"vendored": utils.count_accuracy(B, W1 != 0), "bk_variant_no_forbid": utils.count_accuracy(B, W2 != 0),
            "max_abs_diff_vendored_vs_bk": float(np.abs(W1 - W2).max()),
            "readme": {"fdr": 0.0, "tpr": 1.0, "fpr": 0.0, "shd": 0, "nnz": 20}}


# ------------------------------------------------------------------------------------------------ F3: PC textbook
def f3_pc(n: int = 10000, seed: int = 3_000_000) -> dict:
    """Linear-Gaussian X1 -> X3 <- X2, X3 -> X4 -> X5: the CPDAG is fully oriented (v-structure + Meek R1);
    causal-learn ``pc`` (fisherz, alpha .05, defaults) must return it exactly."""
    from causallearn.search.ConstraintBased.PC import pc
    r = np.random.default_rng([7803, seed])
    e = r.normal(size=(n, 5))
    X1, X2 = e[:, 0], e[:, 1]
    X3 = 0.8 * X1 - 0.7 * X2 + e[:, 2]
    X4 = 0.9 * X3 + e[:, 3]
    X5 = -0.6 * X4 + e[:, 4]
    cg = pc(np.column_stack([X1, X2, X3, X4, X5]), 0.05, "fisherz", show_progress=False)
    G = cg.G.graph
    directed = sorted((i + 1, j + 1) for i in range(5) for j in range(5) if G[j, i] == 1 and G[i, j] == -1)
    undirected = sorted((i + 1, j + 1) for i in range(5) for j in range(i + 1, 5) if G[i, j] == -1 and G[j, i] == -1)
    want = [(1, 3), (2, 3), (3, 4), (4, 5)]
    return {"directed": directed, "undirected": undirected, "expected_directed": want,
            "pass": directed == want and not undirected}


# ------------------------------------------------------------------------------------------------ F4: synthetic
def f4(method: str, n: int, reps: int, regime: str = "R1", b: float = 0.5, n_tune: int = 10,
       nonlinear: bool = False) -> dict:
    """Power on planted edges (b) and null false-positive rate (b = 0, global null), primary declaration rule.
    Score-only methods: tau tuned on ``n_tune`` separate synthetic null-free 'DEV' datasets (placebo rule), then
    applied to ``reps`` fresh ones. Synthetic seeds are DEV-block integers (generator stream 9100, disjoint)."""
    m = METHODS[method]()
    tune_seeds = [3_000_000 + i for i in range(n_tune)]
    test_seeds = [3_000_100 + i for i in range(reps)]
    t0 = time.process_time()
    cfg = m.tune([SYN.make(n, s, b=b, regime=regime, nonlinear=nonlinear)[0] for s in tune_seeds]) \
        if not m.uses_p else m.default_config()
    cfg0 = m.tune([SYN.make(n, s, b=0.0, regime=regime)[0] for s in tune_seeds]) if not m.uses_p else cfg
    hits = {e: 0 for e in SYN.PLANTED}
    sign_ok = {e: 0 for e in SYN.PLANTED}
    fp_alt = n_null_alt = 0
    fp0 = n0 = 0
    p_null = []
    any_fp0 = 0
    for s in test_seeds:
        d, tr = SYN.make(n, s, b=b, regime=regime, nonlinear=nonlinear)
        r = m.run(d, cfg)
        for e in r.edges:
            k = (e.source, e.target)
            if k in tr.edges:
                hits[k] += e.declared
                sign_ok[k] += int(e.declared and tr.signs.get(k, e.sign) == e.sign)
            elif k in tr.null_edges:
                n_null_alt += 1
                fp_alt += e.declared
        d0, _ = SYN.make(n, s, b=0.0, regime=regime)
        r0 = m.run(d0, cfg0)
        fps = sum(e.declared for e in r0.edges)
        fp0 += fps
        n0 += len(r0.edges)
        any_fp0 += fps > 0
        p_null += [e.p for e in r0.edges if e.p is not None]
    out = {"method": method, "n": n, "reps": reps, "regime": regime, "b": b, "nonlinear": nonlinear,
           "rule": "BY per family q .05" if m.uses_p else f"placebo tau from {n_tune} synthetic DEV datasets",
           "tau_alt": cfg.get("tau"), "tau_null": cfg0.get("tau"),
           "power": {f"{a}->{t}": round(v / reps, 3) for (a, t), v in hits.items()},
           "sign_correct_given_declared": {f"{a}->{t}": (round(sign_ok[(a, t)] / hits[(a, t)], 3) if hits[(a, t)]
                                                         else None) for (a, t) in hits},
           "null_edge_fpr_alt": round(fp_alt / max(n_null_alt, 1), 4), "null_edge_fpr_alt_ci": _wilson(fp_alt,
                                                                                                       n_null_alt),
           "global_null_fpr": round(fp0 / max(n0, 1), 4), "global_null_fpr_ci": _wilson(fp0, n0),
           "global_null_fwer": round(any_fp0 / reps, 3), "global_null_fwer_ci": _wilson(any_fp0, reps),
           "cpu_s": round(time.process_time() - t0, 1)}
    if p_null:
        pn = np.asarray(p_null)
        out["raw_p_level_at_.05"] = round(float((pn <= 0.05).mean()), 4)
        out["raw_p_level_ci"] = _wilson(int((pn <= 0.05).sum()), len(pn))
        out["n_null_p"] = len(pn)
    return out


# ------------------------------------------------------------------------------------------------ cost
def cost(method: str, n: int, world: str = "E2", regime: str = "R2", seed: int = 3_000_000,
         config: dict | None = None) -> dict:
    """CPU-s / peak RSS of ONE ``method.run`` on one harness dataset (fresh process per call recommended: the RSS
    figure is the process-lifetime peak, incl. imports and generation). DEV seed only."""
    from cdd_oran.xmethod.worlds import generate_dataset
    assert 3_000_000 <= seed < 3_000_200
    g0 = time.process_time()
    d, _ = generate_dataset(world, regime, n, seed)
    gen = time.process_time() - g0
    m = METHODS[method]()
    base = peak_rss_mb()
    t0, w0 = time.process_time(), time.time()
    r = m.run(d, dict(config or {}))
    return {"method": method, "config": config or {}, "world": world, "regime": regime, "n": n,
            "gen_cpu_s": round(gen, 2),
            "p_actions": len(d.action_names), "k_kpis": len(d.kpi_names),
            "n_candidates": len(d.candidates), "cpu_s": round(r.cpu_s, 2), "cpu_s_total": round(time.process_time()
                                                                                                    - t0, 2),
            "wall_s": round(time.time() - w0, 2), "peak_rss_mb": round(peak_rss_mb(), 1),
            "rss_before_run_mb": round(base, 1), "threads": r.notes.get("threads")}


COST_NS = (500, 1000, 4000, 8000, 24000)


def cost_grid(methods: list[str], ns=COST_NS, timeout_s: int = 7200) -> None:
    """One fresh process per (method, n) (clean peak RSS); E2 R2 (largest world), granger on E3 R2; PC + KCI at
    n <= 1000 only. A run past ``timeout_s`` (the R-13 2 CPU-h budget) is reported as infeasible."""
    import subprocess
    jobs = []
    for m in methods:
        w = "E3" if m == "granger" else "E2"
        jobs += [(m, n, w, None) for n in ns]
        if m == "pc":
            jobs += [(m, n, w, {"indep_test": "kci"}) for n in (500, 1000)]
    for m, n, w, cfg in jobs:
        cmd = [sys.executable, __file__, "cost", "--method", m, "--n", str(n), "--world", w, "--regime", "R2"]
        if cfg:
            cmd += ["--config", json.dumps(cfg)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            line = [x for x in p.stdout.splitlines() if x.startswith("{")]
            print(line[-1] if line else json.dumps({"method": m, "n": n, "config": cfg, "error":
                                                    p.stderr[-800:]}), flush=True)
        except subprocess.TimeoutExpired:
            print(json.dumps({"method": m, "n": n, "config": cfg, "infeasible": f"> {timeout_s} s wall"}),
                  flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["f3_e2", "f3_e2_compare", "f3_notears", "f3_pc", "f4", "cost", "cost_grid",
                                        "tau_demo"])
    ap.add_argument("--config", default="")
    ap.add_argument("--inputs", default="")
    ap.add_argument("--method", default="corr")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--regime", default="R1")
    ap.add_argument("--world", default="E2")
    ap.add_argument("--b", type=float, default=0.5)
    ap.add_argument("--nonlinear", action="store_true")
    ap.add_argument("--rows-root", default=os.path.join(_ROOT, "..", "..", "CDD-ORAN", "runs", "e2slice-recovery"))
    ap.add_argument("--ref-root", default=os.path.join(_ROOT, "..", "..", "CDD-ORAN"))
    ap.add_argument("--notears-utils", default="")
    a = ap.parse_args(argv)
    if a.what == "f3_e2":
        for rec in f3_e2(a.rows_root, tuple(a.method.split(",")), json.loads(a.config) if a.config else None):
            print(json.dumps(rec), flush=True)
        return 0
    if a.what == "f3_e2_compare":
        recs = [json.loads(line) for line in open(a.inputs) if line.startswith("{")]
        res = f3_e2_compare(recs, a.ref_root)
    elif a.what == "f3_notears":
        res = f3_notears(a.notears_utils)
    elif a.what == "f3_pc":
        res = f3_pc()
    elif a.what == "f4":
        res = f4(a.method, a.n, a.reps, a.regime, a.b, nonlinear=a.nonlinear)
    elif a.what == "cost":
        res = cost(a.method, a.n, a.world, a.regime, config=json.loads(a.config) if a.config else None)
    elif a.what == "cost_grid":
        cost_grid(a.method.split(","))
        return 0
    else:
        res = placebo_tau([])
    print(json.dumps(res, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["PLACEBO", "cost", "f3_e2", "f3_notears", "f3_pc", "f4"]

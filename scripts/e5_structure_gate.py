"""E5 structure gate (docs/benchmark/GATE_CONTRACT_E5.md, STRUCTURE-ONLY).

  dev      D1 MSCR marginal calibration (column-permutation null) + D2 P0->K_harm power, seeds 0-19.
  confirm  S1-S5 on fresh seeds 900000-900099 (MSCR vs SHAP-GBDT proxy vs two-tower reconstruction,
           pooled |corr| reported). Provenance hashes are written BEFORE scoring.

Run detached (long): PYTHONPATH=. .venv/Scripts/python.exe -u
    scripts/e5_structure_gate.py {dev|confirm}
Truth is used only to score. Nothing here tunes anything.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from cdd_oran.discovery import discover_mscr, frozen_config  # noqa: E402
from cdd_oran.discovery.mscr import _build_bank, _pval, _s_star  # noqa: E402
from cdd_oran.envs.v2.e5 import E5V2Env  # noqa: E402
from scripts.e5_baselines import NK, NP, corpus, corr_mask, gnn_mask, shap_mask  # noqa: E402
from scripts.e5_spine import HARMFUL_EDGE, true_edges  # noqa: E402
from scripts.runtime_info import runtime_info  # noqa: E402

N_ROWS = 24_000
DEV_SEEDS = list(range(20))
CONFIRM_SEEDS = list(range(900_000, 900_100))
# Every seed any E5 corpus/data/state generator has used (repo scan 2026-09-24: corpus/rng seeds 0-19 in
# scripts/e5_*.py + scratchpad/e5_design/*; env/bank seeds 0-5999 via build_ingate_bank pool and
# e5_env_core_validate). The confirm seeds must be disjoint from all of it.
USED_E5_SEED_RANGES = [(0, 5999)]
N_COLPERM = 64
# Seed-level process pool, resolved at run time: $E5_GATE_WORKERS, else one per CPU. Each worker gets an
# equal share of the CPUs for torch and MSCR. Lower E5_GATE_WORKERS on memory-limited hosts.
N_WORKERS = int(os.environ.get("E5_GATE_WORKERS", 0)) or max(1, os.cpu_count() or 1)
OUT_DIR = os.path.join(_REPO, "runs", "e5-structure-gate")
_H4 = sum(1.0 / k for k in range(1, NP + 1))
ALPHA_GATED = [0.05 * k / (NP * _H4) for k in range(1, NP + 1)] + [0.02]


def _truth() -> set[tuple[int, int]]:
    p_true, _ = true_edges(E5V2Env(subdom=0.20, chain_gamma=0.0))
    return set(p_true)


def _boot(fn, n_units, reps=10_000, seed=12345):
    rng = np.random.default_rng(seed)
    return np.array([fn(rng.integers(0, n_units, n_units)) for _ in range(reps)])


# ------------------------------------------------------------------------------------ per-seed jobs
def _dev_job(seed: int) -> dict:
    x, y = corpus(n=N_ROWS, seed=seed)
    res = discover_mscr(x, y, n_params=NP, seed=seed)
    return {"seed": seed, "declared": res.declared.astype(int).tolist(), "pvals": res.pvals.tolist(),
            "colperm": _colperm_pvals(x, y, seed)}


def _confirm_job(seed: int) -> dict:
    import torch
    share = max(1, (os.cpu_count() or 1) // N_WORKERS)
    torch.set_num_threads(share)
    os.environ.setdefault("MSCR_NUM_THREADS", str(share))
    x, y = corpus(n=N_ROWS, seed=seed)
    res = discover_mscr(x, y, n_params=NP, seed=seed)
    shap_m, imps = shap_mask(x, y)
    out = {"seed": seed, "mscr": res.declared.astype(int).tolist(), "mscr_p": res.pvals.tolist(),
           "shap": shap_m[:, :NP].tolist(), "shap_imps": imps.tolist(),
           "twotower": gnn_mask(x, y)[:, :NP].tolist(), "corr": corr_mask(x, y)[:, :NP].tolist()}
    if seed in CONFIRM_SEEDS[:20]:
        out["colperm"] = _colperm_pvals(x, y, seed)
    return out


def _colperm_pvals(x, y, seed):
    """D1: p-values of each param slot after permuting that column (exact null for randomized params)."""
    cfg = frozen_config()
    ps = []
    for j in range(NK):
        denom, null, strata = _build_bank(x, y[:, j], cfg, np.random.default_rng([seed, j, cfg.n_perm]))
        for i in range(NP):
            for m in range(N_COLPERM):
                col = x[np.random.default_rng([seed, j, i, m, 7]).permutation(N_ROWS), i]
                ps.append(_pval(_s_star(col, i, denom, strata, cfg), i, null))
    return ps


def _run(job, seeds):
    t0, out = time.time(), []
    with ProcessPoolExecutor(N_WORKERS) as ex:
        for r in ex.map(job, seeds):
            out.append(r)
            print(f"seed {r['seed']} done ({time.time() - t0:.0f}s)", flush=True)
    return out


# ------------------------------------------------------------------------------------ scoring
def calibration(records) -> dict:
    per_seed = np.array([r["colperm"] for r in records])
    grid = []
    for a in ALPHA_GATED:
        hits = (per_seed <= a + 1e-12).mean(axis=1)
        ub = float(np.quantile(_boot(lambda idx, h=hits: h[idx].mean(), len(hits)), 0.95))
        grid.append({"alpha": a, "rate": float(hits.mean()), "ub95": ub, "pass": ub <= 1.5 * a})
    return {"grid": grid, "GO": all(g["pass"] for g in grid)}


def mscr_scores(masks, truth) -> dict:
    s = len(masks)
    fam = np.zeros((s, NK))
    v = np.zeros((s, NK))
    r = np.zeros((s, NK))
    tp = 0
    for a, m in enumerate(masks):
        for k in range(NK):
            dec = {p for p in range(NP) if m[k][p]}
            v[a, k] = len(dec - {e[1] for e in truth if e[0] == k})
            r[a, k] = len(dec)
            fam[a, k] = v[a, k] / max(r[a, k], 1)
            tp += len({p for p in dec if (k, p) in truth})
    fdr_b = _boot(lambda idx: fam[idx].mean(), s)
    fdp_b = _boot(lambda idx: v[idx].sum() / max(r[idx].sum(), 1), s)
    return {"family_FDR": float(fam.mean()), "family_FDR_ub95": float(np.quantile(fdr_b, 0.95)),
            "pooled_FDP": float(v.sum() / max(r.sum(), 1)),
            "pooled_FDP_ci95": [float(np.quantile(fdp_b, 0.025)), float(np.quantile(fdp_b, 0.975))],
            "FP_per_seed_mean": float(v.sum(axis=1).mean()),
            "total_recall": tp / (len(truth) * s)}


def edge_rate(masks, edge) -> float:
    k, p = edge
    return float(np.mean([m[k][p] for m in masks]))


def provenance() -> dict:
    files = ["scripts/e5_structure_gate.py", "scripts/e5_baselines.py", "cdd_oran/discovery/mscr.py",
             "cdd_oran/envs/v2/e5.py", "docs/benchmark/GATE_CONTRACT_E5.md"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", *files], capture_output=True, text=True,
                           cwd=_REPO).stdout.strip()
    sha = {f: hashlib.sha256(open(os.path.join(_REPO, f), "rb").read()).hexdigest() for f in files}
    return {"git_head": head, "uncommitted_changes": dirty, "sha256": sha, "runtime": runtime_info(),
            "n_workers": N_WORKERS}


def main_dev():
    os.makedirs(OUT_DIR, exist_ok=True)
    truth, prov = _truth(), provenance()
    recs = _run(_dev_job, DEV_SEEDS)
    d1 = calibration(recs)
    d2 = edge_rate([r["declared"] for r in recs], HARMFUL_EDGE)
    out = {"provenance": prov, "D1": d1, "D2_P0Kharm_rate": d2,
           "D2_GO": d2 * len(DEV_SEEDS) >= 19, "GO": bool(d1["GO"] and d2 * len(DEV_SEEDS) >= 19),
           "dev_mscr_descriptive": mscr_scores([r["declared"] for r in recs], truth), "raw": recs}
    json.dump(out, open(os.path.join(OUT_DIR, "dev.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "raw"}, indent=1))


def main_confirm():
    used = [sd for sd in CONFIRM_SEEDS if any(lo <= sd <= hi for lo, hi in USED_E5_SEED_RANGES)]
    assert not used, f"confirm seeds overlap previously used E5 seeds: {used[:5]}"
    os.makedirs(OUT_DIR, exist_ok=True)
    truth, prov = _truth(), provenance()
    json.dump(prov, open(os.path.join(OUT_DIR, "confirm_provenance.json"), "w"), indent=1)
    recs = _run(_confirm_job, CONFIRM_SEEDS)
    ms = mscr_scores([r["mscr"] for r in recs], truth)
    s1 = edge_rate([r["mscr"] for r in recs], HARMFUL_EDGE)
    s4_miss = 1 - edge_rate([r["shap"] for r in recs], HARMFUL_EDGE)
    s5_miss = 1 - edge_rate([r["twotower"] for r in recs], HARMFUL_EDGE)
    gates = {"S1_mscr_harm>=0.95": s1 >= 0.95, "S2_family_FDR_ub95<=0.05": ms["family_FDR_ub95"] <= 0.05,
             "S3_mscr_recall>=0.90": ms["total_recall"] >= 0.90, "S4_shap_miss>=0.95": s4_miss >= 0.95,
             "S5_twotower_miss>=0.95": s5_miss >= 0.95}
    k, p = HARMFUL_EDGE
    shap_ratio = [r["shap_imps"][k][p] / max(r["shap_imps"][k]) for r in recs]
    out = {"provenance": prov, "gates": gates, "verdict": "PASS" if all(gates.values()) else "FAIL",
           "S1_mscr_harm_rate": s1, "S4_shap_miss_rate": s4_miss, "S5_twotower_miss_rate": s5_miss,
           "mscr": ms, "corr_harm_rate": edge_rate([r["corr"] for r in recs], HARMFUL_EDGE),
           "shap_ratio_P0Kharm": {"mean": float(np.mean(shap_ratio)), "max": float(np.max(shap_ratio))},
           "per_edge_recall": {arm: {f"P{e[1]}->K{e[0]}": edge_rate([r[arm] for r in recs], e)
                                     for e in sorted(truth)} for arm in ("mscr", "shap", "twotower", "corr")},
           "calibration_fresh20": calibration([r for r in recs if "colperm" in r]), "raw": recs}
    json.dump(out, open(os.path.join(OUT_DIR, "confirm.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "raw"}, indent=1))


if __name__ == "__main__":
    {"dev": main_dev, "confirm": main_confirm}[sys.argv[1]]()

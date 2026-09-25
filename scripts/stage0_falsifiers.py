"""Stage 0 falsifiers (pre-declaration: docs/benchmark/STAGE0_FALSIFIERS.md).

  collect   one pass over every corpus; appends raw per-corpus records to runs/stage0/records.jsonl
            (MSCR masks, SHAP/|corr| importances, RCoT masks, learned-model decision losses, dither diagnostics).
            Resumable: a (check, world, seed, extra) key already present is skipped.
  score     applies the pre-declared rules (DEV tau / lambda selection, bootstrap bounds) -> runs/stage0/score.json

Truth is used only to score. MSCR is frozen. Thresholds/lambdas for baselines are chosen on DEV records only.
  PYTHONPATH=. .venv/Scripts/python.exe -u scripts/stage0_falsifiers.py {collect|score}
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from cdd_oran.analysis.v2_regret import score_grid  # noqa: E402
from cdd_oran.benchmark.learned_world_model import (  # noqa: E402
    MLPMember,
    fit_kpi_models,
    learned_env_class,
)
from cdd_oran.discovery import discover_mscr  # noqa: E402
from cdd_oran.e2slice.dataset import E2DatasetConfig, E2Rows, generate_rows  # noqa: E402
from cdd_oran.e2slice.discovery_rcot import discover_graph_rcot  # noqa: E402
from cdd_oran.e2slice.discovery_rcot_v2 import frozen_config_v2  # noqa: E402
from cdd_oran.envs.v2.e2 import E2V2Env  # noqa: E402
from cdd_oran.envs.v2.e5 import E5V2Env  # noqa: E402
from scripts import e2_decision_gate as g2  # noqa: E402
from scripts.e2_baseline_shap_dag import fit_shap_importances  # noqa: E402
from scripts.e5_baselines import corpus as e5_corpus  # noqa: E402
from scripts.runtime_info import runtime_info  # noqa: E402

OUT_DIR = os.path.join(_REPO, "runs", "stage0")
REC = os.path.join(OUT_DIR, "records.jsonl")
REGISTRY = os.path.join(_REPO, "docs", "benchmark", "SEED_REGISTRY.json")

TAUS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
DETECT = {"E2": {"dev": list(range(20)), "test": list(range(722000, 722020)), "n": 4000},
          "E5": {"dev": list(range(20)), "test": list(range(922000, 922020)), "n": 24000}}
GRAPH_DEV = list(range(5))
GRAPH_TEST = list(range(723000, 723010))
DITHER_TEST = list(range(724000, 724020))
DITHER_DELTAS = (0.05, 0.10, 0.20)
E2_TRUE = {k: sorted(s for kk, s in E2V2Env._TRUE_ADJACENCY if kk == k and s < 8) for k in range(6)}
E5_NP, E5_NK = E5V2Env.num_params, E5V2Env.num_kpis


def e5_truth():
    from scripts.e5_spine import true_edges
    p, _ = true_edges(E5V2Env(subdom=0.20, chain_gamma=0.0))
    return {k: sorted(s for kk, s in p if kk == k) for k in range(E5_NK)}


# ------------------------------------------------------------------------------------------ plumbing
def _done() -> set:
    if not os.path.exists(REC):
        return set()
    return {tuple(r["key"]) for r in map(json.loads, open(REC))}


def _append(rec: dict) -> None:
    with open(REC, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k in ("key", "elapsed_s")}), flush=True)


def _check_seeds() -> dict:
    reg = json.load(open(REGISTRY))

    def used(world):
        return reg[world]["used"]

    tests = {"E2": DETECT["E2"]["test"] + GRAPH_TEST + DITHER_TEST, "E5": DETECT["E5"]["test"]}
    for world, seeds in tests.items():
        bad = [s for s in seeds if any(lo <= s <= hi for lo, hi in used(world))]
        assert not bad, f"{world}: TEST seeds overlap the registry: {bad[:5]}"
    files = ["scripts/stage0_falsifiers.py", "docs/benchmark/STAGE0_FALSIFIERS.md", "docs/benchmark/SEED_REGISTRY.json",
             "cdd_oran/discovery/mscr.py", "cdd_oran/benchmark/learned_world_model.py"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", *files], capture_output=True, text=True, cwd=_REPO).stdout.strip()
    return {"git_head": head, "uncommitted": dirty, "runtime": runtime_info(),
            "sha256": {f: hashlib.sha256(open(os.path.join(_REPO, f), "rb").read()).hexdigest() for f in files}}


def _e2_x(rows: E2Rows) -> np.ndarray:
    return np.concatenate([rows.x_params, rows.x_kpis], axis=1)


def _imps(xp: np.ndarray, y: np.ndarray):
    """Per-KPI SHAP-GBDT mean|SHAP| and pooled |corr| over the param columns."""
    shap_i = [fit_shap_importances(xp, y[:, k], seed=0)[0].tolist() for k in range(y.shape[1])]
    corr_i = [[float(abs(np.corrcoef(xp[:, p], y[:, k])[0, 1])) for p in range(xp.shape[1])] for k in range(y.shape[1])]
    return shap_i, corr_i


# ------------------------------------------------------------------------------------------ F-detect
def collect_detect(done: set) -> None:
    for world, spec in DETECT.items():
        for split in ("dev", "test"):
            for seed in spec[split]:
                key = ["detect", world, seed, split]
                if tuple(key) in done:
                    continue
                t0 = time.time()
                if world == "E2":
                    rows = generate_rows(E2DatasetConfig(n_rows_per_seed=spec["n"], seed=seed))
                    xp, y = rows.x_params, rows.y_kpis
                    mscr = discover_mscr(_e2_x(rows), y, n_params=8, seed=seed).declared.astype(int).tolist()
                    rcot = None
                    if split == "test":
                        rcot = discover_graph_rcot(rows, frozen_config_v2()).binary_mask[:, :8].tolist()
                else:
                    xp, y = e5_corpus(n=spec["n"], seed=seed)
                    mscr = discover_mscr(xp, y, n_params=E5_NP, seed=seed).declared.astype(int).tolist()
                    rcot = None
                shap_i, corr_i = _imps(xp, y)
                _append({"key": key, "mscr": mscr, "rcot": rcot, "shap": shap_i, "corr": corr_i,
                         "elapsed_s": round(time.time() - t0, 1)})


# ------------------------------------------------------------------------------------------ F-graph
def _bank_and_losses():
    bank = g2.build_bank(E2V2Env)
    panel = g2.build_panel(E2V2Env(0), g2.FULL_PANEL_IDS)
    pos = bank.positives
    st = [score_grid(E2V2Env(env_seed=s), snap, g2.SHARED_PARAM, panel) for s, snap in pos]

    def losses(cls):
        return [float(t.max() - t[int(np.argmax(score_grid(cls(env_seed=s), snap, g2.SHARED_PARAM, panel)))])
                for (s, snap), t in zip(pos, st, strict=True)]
    return losses


def collect_graph_dev(done: set) -> None:
    """lambda selection data: validation MSE per (seed, KPI, lambda) on a fixed 80/20 split (prediction only)."""
    for seed in GRAPH_DEV:
        key = ["graph-dev", "E2", seed, ""]
        if tuple(key) in done:
            continue
        t0 = time.time()
        rows = generate_rows(E2DatasetConfig(n_rows_per_seed=4000, seed=seed))
        xp, y = rows.x_params, rows.y_kpis
        cut = int(0.8 * len(xp))
        val = {}
        for k in range(6):
            for lam in LAMBDAS:
                ens = [MLPMember(xp[:cut], y[:cut, k], seed=m, group_l1=lam) for m in range(3)]
                pred = np.mean([m.predict(xp[cut:]) for m in ens], axis=0)
                val[f"{k}|{lam}"] = float(np.mean((pred - y[cut:, k]) ** 2) / np.var(y[cut:, k]))
        _append({"key": key, "val_nmse": val, "elapsed_s": round(time.time() - t0, 1)})


def selected_lambdas() -> dict:
    recs = [r for r in map(json.loads, open(REC)) if r["key"][0] == "graph-dev"]
    assert len(recs) == len(GRAPH_DEV), "lambda selection needs all DEV records"
    return {k: min(LAMBDAS, key=lambda lam: np.mean([r["val_nmse"][f"{k}|{lam}"] for r in recs])) for k in range(6)}


def collect_graph_test(done: set) -> None:
    losses = _bank_and_losses()
    lam = selected_lambdas()
    for seed in GRAPH_TEST:
        rows = generate_rows(E2DatasetConfig(n_rows_per_seed=4000, seed=seed))
        xp, y = rows.x_params, rows.y_kpis
        mscr = discover_mscr(_e2_x(rows), y, n_params=8, seed=seed).declared
        shap_i, _ = _imps(xp, y)
        arms = {
            "mscr": ({k: [int(p) for p in np.nonzero(mscr[k])[0]] for k in range(6)}, 0.0),
            "l1": ({k: list(range(8)) for k in range(6)}, lam),
            "shap": ({k: [p for p in range(8) if shap_i[k][p] >= 0.10 * max(shap_i[k])] for k in range(6)}, 0.0),
            "all": ({k: list(range(8)) for k in range(6)}, 0.0),
            "true": (E2_TRUE, 0.0),
        }
        for arm, (parents, gl1) in arms.items():
            key = ["graph", "E2", seed, arm]
            if tuple(key) in done:
                continue
            t0 = time.time()
            cls = learned_env_class(E2V2Env, fit_kpi_models(xp, y, parents, group_l1=gl1), 8)
            _append({"key": key, "parents": {str(k): v for k, v in parents.items()}, "lambda": lam if arm == "l1" else None,
                     "loss": losses(cls), "elapsed_s": round(time.time() - t0, 1)})


# ------------------------------------------------------------------------------------------ F-dither
def dither_rows(n: int, seed: int, delta: float, n_blocks: int = 20):
    """Piecewise setpoint (inside +/-25% of range around the midpoint) + independent per-knob U(+/-delta*range)
    dither, clipped to the ID ranges; otherwise the generate_rows call list (two priming advances, lagged KPIs)."""
    lo, hi = np.array(E2V2Env.id_ranges, dtype=float).T
    mid, rng_ = (lo + hi) / 2, hi - lo
    env = E2V2Env(env_seed=seed, episode=seed)
    env.reset(episode=seed)
    rng = np.random.default_rng([seed, 724])
    setpoints = rng.uniform(mid - 0.25 * rng_, mid + 0.25 * rng_, size=(n_blocks, 8))
    clipped = 0

    def inject(row):
        nonlocal clipped
        sp = setpoints[min(row * n_blocks // n, n_blocks - 1)]
        cmd = sp + rng.uniform(-delta, delta, size=8) * rng_
        val = np.clip(cmd, lo, hi)
        clipped += int((val != cmd).sum())
        for i in range(8):
            env.apply_action(i, float(val[i]))

    for _ in range(2):  # the two priming advances of generate_rows
        inject(0)
        env.advance()
    xp, xk, y = [], [], []
    for r in range(n):
        xp.append(env.prev_params.copy())
        xk.append(env.prev_kpis.copy())
        inject(r)
        y.append(env.advance().copy())
    return np.array(xp), np.array(xk), np.array(y), clipped


def collect_dither(done: set) -> None:
    configs = [(d, 4000) for d in DITHER_DELTAS] + [(0.10, 12000)]
    for delta, n in configs:
        for seed in DITHER_TEST:
            key = ["dither", "E2", seed, f"{delta}|{n}"]
            if tuple(key) in done:
                continue
            t0 = time.time()
            xp, xk, y, clipped = dither_rows(n, seed, delta)
            res = discover_mscr(np.concatenate([xp, xk], axis=1), y, n_params=8, seed=seed)
            gate = y[:, 5] < -5.0
            blocks = np.arange(len(y)) * 20 // len(y)
            _append({"key": key, "mscr": res.declared.astype(int).tolist(), "gate_rows": int(gate.sum()),
                     "gate_blocks": int(len(set(blocks[gate].tolist()))), "clipped": clipped,
                     "elapsed_s": round(time.time() - t0, 1)})


# ------------------------------------------------------------------------------------------ scoring
def _boot(values_fn, n, reps=10_000, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([values_fn(rng.integers(0, n, n)) for _ in range(reps)])


def _ub95(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.quantile(_boot(lambda i: v[i].mean(), len(v)), 0.95))


def _lb95_diff(a, b) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.quantile(_boot(lambda i: d[i].mean(), len(d)), 0.05))


def _ci90(v) -> list:
    v = np.asarray(v, dtype=float)
    return [float(q) for q in np.quantile(_boot(lambda i: v[i].mean(), len(v)), [0.05, 0.95])]


def _fam_fdr(mask, truth, nparams) -> float:
    """Seed-level family FDR: mean over KPIs of V/max(R,1) over the param candidates."""
    out = []
    for k in range(len(mask)):
        dec = {p for p in range(nparams) if mask[k][p]}
        out.append(len(dec - set(truth[k])) / max(len(dec), 1))
    return float(np.mean(out))


def _thresh_mask(imps, tau):
    return [[int(v >= tau * max(row)) if max(row) > 0 else 0 for v in row] for row in imps]


def score() -> dict:
    recs = [json.loads(line) for line in open(REC)]
    out: dict = {}
    # ---- F-detect
    for world, spec in DETECT.items():
        truth = E2_TRUE if world == "E2" else e5_truth()
        npar = 8 if world == "E2" else E5_NP
        hk, hp = (5, 0) if world == "E2" else (1, 0)
        by = {(r["key"][3], r["key"][2]): r for r in recs if r["key"][0] == "detect" and r["key"][1] == world}
        dev = [by[("dev", s)] for s in spec["dev"] if ("dev", s) in by]
        test = [by[("test", s)] for s in spec["test"] if ("test", s) in by]
        if len(test) < len(spec["test"]) or len(dev) < len(spec["dev"]):
            out[f"F-detect_{world}"] = {"status": "INCOMPLETE"}
            continue
        mscr_hit = np.array([r["mscr"][hk][hp] for r in test], dtype=float)
        res = {"MSCR": {"recall": float(mscr_hit.mean()),
                        "fdr_ub": _ub95([_fam_fdr(r["mscr"], truth, npar) for r in test])}}
        falsified = []
        for det in ("shap", "corr"):
            valid = []
            for tau in TAUS:
                fdr = np.mean([_fam_fdr(_thresh_mask(r[det], tau), truth, npar) for r in dev])
                rec_ = np.mean([_thresh_mask(r[det], tau)[hk][hp] for r in dev])
                if fdr <= 0.05:
                    valid.append((rec_, -tau, tau))
            curve = {str(t): {"recall": float(np.mean([_thresh_mask(r[det], t)[hk][hp] for r in test])),
                              "fdr": float(np.mean([_fam_fdr(_thresh_mask(r[det], t), truth, npar) for r in test]))}
                     for t in TAUS}
            if not valid:
                res[det] = {"status": "no FDR-valid operating point on DEV", "test_curve": curve}
                continue
            tau = max(valid)[2]
            hit = np.array([_thresh_mask(r[det], tau)[hk][hp] for r in test], dtype=float)
            fdrs = np.array([_fam_fdr(_thresh_mask(r[det], tau), truth, npar) for r in test])
            lb, ub = _lb95_diff(hit, mscr_hit), _ub95(fdrs)
            fals = bool(lb >= -0.05 and ub <= 0.05)
            res[det] = {"tau": tau, "recall": float(hit.mean()), "paired_diff_lb95": lb, "fdr_ub": ub,
                        "falsifies": fals, "test_curve": curve}
            if fals:
                falsified.append(det)
        if world == "E2" and all(r["rcot"] is not None for r in test):
            hit = np.array([r["rcot"][hk][hp] for r in test], dtype=float)
            fdrs = np.array([_fam_fdr(r["rcot"], truth, npar) for r in test])
            lb, ub = _lb95_diff(hit, mscr_hit), _ub95(fdrs)
            res["rcot_v2"] = {"recall": float(hit.mean()), "paired_diff_lb95": lb, "fdr_ub": ub,
                              "falsifies": bool(lb >= -0.05 and ub <= 0.05)}
            if res["rcot_v2"]["falsifies"]:
                falsified.append("rcot_v2")
        res["FALSIFIED_BY"] = falsified
        out[f"F-detect_{world}"] = res
    # ---- F-graph
    g = {(r["key"][2], r["key"][3]): r for r in recs if r["key"][0] == "graph"}
    arms = ("mscr", "l1", "shap", "all", "true")
    if all((s, a) in g for s in GRAPH_TEST for a in arms):
        per = {a: np.array([np.mean(g[(s, a)]["loss"]) for s in GRAPH_TEST]) for a in arms}
        d = per["l1"] - per["mscr"]
        ub = _ub95(d)
        margin = max(0.20, 0.25 * float(per["mscr"].mean()))
        out["F-graph"] = {"per_seed": {a: v.tolist() for a, v in per.items()},
                          "mean": {a: float(v.mean()) for a, v in per.items()},
                          "frac_loss_ge_1": {a: float(np.mean([np.mean(np.array(g[(s, a)]["loss"]) >= 1) for s in GRAPH_TEST]))
                                             for a in arms},
                          "lambda": g[(GRAPH_TEST[0], "l1")]["lambda"], "d_mean": float(d.mean()), "d_ub95": ub,
                          "margin": margin, "FALSIFIED": bool(ub <= margin)}
    else:
        out["F-graph"] = {"status": "INCOMPLETE"}
    # ---- F-dither
    dith = {}
    for delta, n in [(d, 4000) for d in DITHER_DELTAS] + [(0.10, 12000)]:
        rs = [r for r in recs if r["key"][0] == "dither" and r["key"][3] == f"{delta}|{n}"]
        if len(rs) < len(DITHER_TEST):
            dith[f"{delta}|{n}"] = {"status": "INCOMPLETE"}
            continue
        visited = [r for r in rs if r["gate_rows"] >= 200]
        hit_all = np.array([r["mscr"][5][0] for r in rs], dtype=float)
        hit_vis = np.array([r["mscr"][5][0] for r in visited], dtype=float)
        entry = {"frac_corpora_gate_visited": len(visited) / len(rs),
                 "median_gate_rows": float(np.median([r["gate_rows"] for r in rs])),
                 "recall_all": float(hit_all.mean()),
                 "recall_visited": float(hit_vis.mean()) if len(hit_vis) else None,
                 "fdr_descriptive": float(np.mean([_fam_fdr(r["mscr"], E2_TRUE, 8) for r in rs])),
                 "clipped_mean": float(np.mean([r["clipped"] for r in rs]))}
        if len(hit_vis):
            entry["recall_visited_ci90"] = _ci90(hit_vis)
        if (delta, n) == (0.10, 4000):
            entry["METHOD_FALSIFIED"] = bool(hit_vis.mean() < 0.80) if len(hit_vis) else "NOT EVALUABLE (no corpus visited the gate)"
            entry["DATA_REGIME_RARELY_VISITS_GATE"] = bool(len(visited) / len(rs) < 0.25)
        dith[f"{delta}|{n}"] = entry
    out["F-dither"] = dith
    json.dump(out, open(os.path.join(OUT_DIR, "score.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


def collect() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    prov_path = os.path.join(OUT_DIR, "provenance.json")
    if not os.path.exists(prov_path):
        json.dump(_check_seeds(), open(prov_path, "w"), indent=1)
    else:
        _check_seeds()
    done = _done()
    collect_dither(done)
    collect_detect(done)
    collect_graph_dev(done)
    collect_graph_test(_done())
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    {"collect": collect, "score": score}[sys.argv[1]]()

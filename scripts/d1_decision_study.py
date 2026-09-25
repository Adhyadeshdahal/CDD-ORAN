"""Decision study D1 (pre-declaration: docs/benchmark/DECISION_STUDY_D1.md).

MSCR discovery -> learned per-KPI world model -> planner, on E2 (H=1, one knob) and E3 (H=3). Fresh seeds
710000-710009. One JSON line per (env, n, seed, arm) is appended to runs/d1-decision-study/records.jsonl as
it completes; `score` computes the gates from those records. Provenance is written before any scoring.

  PYTHONPATH=. .venv/Scripts/python.exe -u scripts/d1_decision_study.py {run|score}
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
from cdd_oran.benchmark.learned_world_model import fit_kpi_models, learned_env_class  # noqa: E402
from cdd_oran.benchmark.masked_world_model import MaskedE2WorldModel  # noqa: E402
from cdd_oran.discovery import discover_mscr  # noqa: E402
from cdd_oran.e2slice.dataset import E2DatasetConfig, generate_rows  # noqa: E402
from cdd_oran.envs.v2.e2 import E2V2Env  # noqa: E402
from cdd_oran.envs.v2.e3 import E3V2Env  # noqa: E402
from cdd_oran.planners.sequence import cem_sequence, mppi_sequence, qacm_v2  # noqa: E402
from scripts import e2_decision_gate as g2  # noqa: E402
from scripts import e3_decision_gate as g3  # noqa: E402
from scripts.runtime_info import runtime_info  # noqa: E402

SEEDS = list(range(710_000, 710_010))
USED = {"E2": [(0, 19), (100_000, 100_003), (500_000, 500_039), (600_000, 616_383), (800_000, 800_099)],
        "E3": [(0, 0)]}
E2_SIZES = (4000, 24000)
E3_N = 8000
OUT_DIR = os.path.join(_REPO, "runs", "d1-decision-study")
REC = os.path.join(OUT_DIR, "records.jsonl")


def _append(rec: dict) -> None:
    with open(REC, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("loss", "regret")}), flush=True)


def provenance() -> dict:
    files = ["scripts/d1_decision_study.py", "cdd_oran/benchmark/learned_world_model.py",
             "cdd_oran/discovery/mscr.py", "cdd_oran/planners/sequence.py",
             "docs/benchmark/DECISION_STUDY_D1.md"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", *files], capture_output=True, text=True,
                           cwd=_REPO).stdout.strip()
    return {"git_head": head, "uncommitted": dirty, "runtime": runtime_info(),
            "sha256": {f: hashlib.sha256(open(os.path.join(_REPO, f), "rb").read()).hexdigest() for f in files}}


# ------------------------------------------------------------------------------------------- E2
def run_e2(done: set) -> None:
    bank = g2.build_bank(E2V2Env)
    panel = g2.build_panel(E2V2Env(0), g2.FULL_PANEL_IDS)
    pos = bank.positives
    st = [score_grid(E2V2Env(env_seed=s), snap, g2.SHARED_PARAM, panel) for s, snap in pos]

    def losses(factory):
        return [float(t.max() - t[int(np.argmax(score_grid(factory(s), snap, g2.SHARED_PARAM, panel)))])
                for (s, snap), t in zip(pos, st, strict=True)]

    if ("E2", 0, 0, "missing-edge") not in done:
        _append({"env": "E2", "n": 0, "seed": 0, "arm": "missing-edge",
                 "loss": losses(lambda s: E2V2Env(env_seed=s, decoy_omit_p0_k5=True))})
    for n in E2_SIZES:
        for seed in SEEDS:
            if all(("E2", n, seed, a) in done for a in ("masked", "learned-mscr", "learned-none", "learned-true")):
                continue
            rows = generate_rows(E2DatasetConfig(n_rows_per_seed=n, seed=seed))
            x = np.concatenate([rows.x_params, rows.x_kpis], axis=1)
            res = discover_mscr(x, rows.y_kpis, n_params=8, seed=seed)
            edges = res.param_edges()
            graph = {k: sorted(p for kk, p in edges if kk == k) for k in range(6)}
            arms = {
                "masked": lambda s, e=edges: MaskedE2WorldModel(e, env_seed=s),
                "learned-mscr": graph,
                "learned-none": {k: list(range(8)) for k in range(6)},
                "learned-true": {k: sorted(p for kk, p in E2V2Env._TRUE_ADJACENCY if kk == k and p < 8)
                                 for k in range(6)},
            }
            for arm, spec in arms.items():
                if ("E2", n, seed, arm) in done:
                    continue
                if arm == "masked":
                    factory = spec
                else:
                    cls = learned_env_class(E2V2Env, fit_kpi_models(rows.x_params, rows.y_kpis, spec), 8)
                    factory = lambda s, c=cls: c(env_seed=s)  # noqa: E731
                _append({"env": "E2", "n": n, "seed": seed, "arm": arm,
                         "mscr_graph": {str(k): v for k, v in graph.items()}, "loss": losses(factory)})


# ------------------------------------------------------------------------------------------- E3
def e3_corpus(n: int, seed: int):
    rng = np.random.default_rng(seed)
    env = E3V2Env(truncate_fanout=False)
    p = rng.uniform(0.0, 1.0, size=(n, E3V2Env.num_params))
    x, y, k = [], [], np.zeros(E3V2Env.num_kpis)
    for t in range(n):
        k_next = env._update_kpis(p[t], k)
        x.append(np.concatenate([p[t], k]))
        y.append(k_next)
        k = k_next
    return np.array(x), np.array(y)


def _e3_regrets(bank, coef, factory, planners):
    regret = {pl: [] for pl in planners}
    for snap in bank.snaps:
        gg = g3.vectorized_G(snap, *coef)
        gs, d3 = float(gg.max()), float(gg.max() - gg.min())
        for pl, fn in planners.items():
            seq = fn(factory, snap, g3.SHARED_PARAM, g3.GRID.tolist(), g3.H, g3._R)
            regret[pl].append((gs - float(gg[g3._seq_to_indices(seq)])) / d3)
    return regret


def run_e3(done: set) -> None:
    bank = g3.build_bank(E3V2Env)
    full = E3V2Env(truncate_fanout=False)
    coef = (full.c10, full.c25, full.c35)
    nk, npar = E3V2Env.num_kpis, E3V2Env.num_params
    planners = {"qacm": qacm_v2, "cem": cem_sequence, "mppi": mppi_sequence}
    if ("E3", 0, 0, "true-model") not in done:  # diagnostic: planner quality with the true model
        _append({"env": "E3", "n": 0, "seed": 0, "arm": "true-model",
                 "regret": _e3_regrets(bank, coef, lambda: E3V2Env(truncate_fanout=False), planners)})
    for seed in SEEDS:
        if all(("E3", E3_N, seed, a) in done for a in ("learned-mscr", "learned-none", "learned-true")):
            continue
        x, y = e3_corpus(E3_N, seed)
        res = discover_mscr(x, y, n_params=npar + nk, seed=seed)
        graph = {k: [int(c) for c in np.nonzero(res.declared[k])[0]] for k in range(nk)}
        truth = {k: sorted(src for kk, src in full.adjacency_edges if kk == k) for k in range(nk)}
        for arm, par in (("learned-mscr", graph), ("learned-none", {k: list(range(npar + nk)) for k in range(nk)}),
                         ("learned-true", truth)):
            if ("E3", E3_N, seed, arm) in done:
                continue
            cls = learned_env_class(E3V2Env, fit_kpi_models(x, y, par), npar)
            regret = _e3_regrets(bank, coef, lambda c=cls: c(truncate_fanout=False), planners)
            _append({"env": "E3", "n": E3_N, "seed": seed, "arm": arm, "graph": {str(k): v for k, v in par.items()},
                     "mscr_graph_exact": graph == truth, "regret": regret})


# ------------------------------------------------------------------------------------------- score
def _boot_ub(v, reps=10_000, seed=0):
    """One-sided 95% percentile bootstrap upper bound of the mean; the unit is the corpus seed (v = the
    per-seed mean losses); 10,000 resamples of the seeds with replacement, RNG seed 0 (PREDECLARE_D1)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v)
    return float(np.quantile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(reps)], 0.95))


def score() -> dict:
    recs = [json.loads(line) for line in open(REC)]
    by = {(r["env"], r["n"], r["seed"], r["arm"]): r for r in recs}
    out = {"n_records": len(recs)}
    miss = by.get(("E2", 0, 0, "missing-edge"))
    out["E2_missing_edge_mean_loss"] = float(np.mean(miss["loss"])) if miss else None
    for n in E2_SIZES:
        per = {a: [float(np.mean(by[("E2", n, s, a)]["loss"])) for s in SEEDS if ("E2", n, s, a) in by]
               for a in ("masked", "learned-mscr", "learned-none", "learned-true")}
        tails = {a: [x for s in SEEDS if ("E2", n, s, a) in by for x in by[("E2", n, s, a)]["loss"]] for a in per}
        out[f"E2_n{n}"] = {a: {"mean": float(np.mean(v)), "per_seed": v,
                               "frac_states_loss_ge_1": float(np.mean(np.array(tails[a]) >= 1.0)),
                               "max_state_loss": float(np.max(tails[a]))} for a, v in per.items() if v}
        if n == 4000 and len(per["learned-mscr"]) == len(SEEDS) and len(per["learned-none"]) == len(SEEDS):
            m, nn = np.array(per["learned-mscr"]), np.array(per["learned-none"])
            out["D1"] = {"mean": float(m.mean()), "ub95": _boot_ub(m),
                         "PASS": bool(m.mean() <= 2.0 and _boot_ub(m) <= 4.0)}
            out["D2"] = {"mscr_beats_none_seeds": int((m < nn).sum()), "paired_diff_none_minus_mscr": (nn - m).tolist(),
                         "mean_paired_diff": float((nn - m).mean()), "PASS": bool((m < nn).sum() >= 8),
                         "note": "directional criterion (sign count); effect size reported, not gated"}
    tm = by.get(("E3", 0, 0, "true-model"))
    if tm:
        out["E3_true-model"] = {pl: float(np.mean(v)) for pl, v in tm["regret"].items()}
    e3 = {a: [by[("E3", E3_N, s, a)] for s in SEEDS if ("E3", E3_N, s, a) in by]
          for a in ("learned-mscr", "learned-none", "learned-true")}
    for a, rs in e3.items():
        if rs:
            out[f"E3_{a}"] = {}
            for pl in ("qacm", "cem", "mppi"):
                allr = np.concatenate([r["regret"][pl] for r in rs])
                out[f"E3_{a}"][pl] = {"mean": float(allr.mean()), "max": float(allr.max()),
                                      "frac_optimal": float(np.mean(allr < 1e-9)),
                                      "per_seed": [float(np.mean(r["regret"][pl])) for r in rs]}
            out[f"E3_{a}"]["mscr_graph_exact_rate"] = float(np.mean([r.get("mscr_graph_exact", False) for r in rs]))
    if len(e3["learned-mscr"]) == len(SEEDS):
        cem, qacm = out["E3_learned-mscr"]["cem"]["mean"], out["E3_learned-mscr"]["qacm"]["mean"]
        out["D3"] = {"cem_mean_regret": cem, "PASS": cem <= 0.01}
        out["D4"] = {"qacm_mean_regret": qacm, "PASS": qacm >= 0.10}
    gates = [out[g]["PASS"] for g in ("D1", "D2", "D3", "D4") if g in out]
    out["verdict"] = "PASS" if len(gates) == 4 and all(gates) else ("INCOMPLETE" if len(gates) < 4 else "FAIL")
    json.dump(out, open(os.path.join(OUT_DIR, "score.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


def main_run() -> None:
    for env, ranges in USED.items():
        bad = [s for s in SEEDS if any(lo <= s <= hi for lo, hi in ranges)]
        assert not bad, f"{env}: seeds overlap previously used seeds {bad[:3]}"
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(OUT_DIR, "provenance.json")):
        json.dump(provenance(), open(os.path.join(OUT_DIR, "provenance.json"), "w"), indent=1)
    done = set()
    if os.path.exists(REC):
        done = {(r["env"], r["n"], r["seed"], r["arm"]) for r in map(json.loads, open(REC))}
    t0 = time.time()
    run_e3(done)
    run_e2(done)
    print(f"ALL DONE ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    {"run": main_run, "score": score}[sys.argv[1]]()

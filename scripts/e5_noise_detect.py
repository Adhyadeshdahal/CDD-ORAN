"""E5 with observation noise: MSCR vs SHAP / |corr| detection (pre-declaration: docs/benchmark/E5_NOISE_DETECT.md).

  collect   per kappa in {0.1, 0.3}: DEV seeds 0-19 and TEST seeds 930000-930019 -> runs/e5-noise/records.jsonl
            (resumable; provenance written before any record)
  score     DEV tau selection (tie-break: higher recall, then lower FDR, then larger tau) and the three-way outcome
            per (kappa, baseline) -> runs/e5-noise/score.json

Device and threads are runtime choices (MSCR resolves them itself; override with MSCR_DEVICE / MSCR_NUM_THREADS).
  PYTHONPATH=. .venv/Scripts/python.exe -u scripts/e5_noise_detect.py {collect|score}
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

from cdd_oran.discovery import discover_mscr  # noqa: E402
from cdd_oran.envs.v2.e5 import E5V2Env  # noqa: E402
from scripts.e5_baselines import corpus as e5_corpus  # noqa: E402
from scripts.e5_spine import E5_KPI_MOMENTS  # noqa: E402
from scripts.runtime_info import runtime_info  # noqa: E402
from scripts.stage0_falsifiers import (  # noqa: E402
    TAUS,
    _fam_fdr,
    _imps,
    _lb95_diff,
    _thresh_mask,
    _ub95,
    e5_truth,
)

OUT_DIR = os.path.join(_REPO, "runs", "e5-noise")
REC = os.path.join(OUT_DIR, "records.jsonl")
REGISTRY = os.path.join(_REPO, "docs", "benchmark", "SEED_REGISTRY.json")
KAPPAS = (0.1, 0.3)
DEV = list(range(20))
TEST = list(range(930000, 930020))
N_ROWS = 24000
NP_ = E5V2Env.num_params
HARM = (1, 0)  # (K_harm, P0)
SIGMA = np.array([E5_KPI_MOMENTS[k][1] for k in range(E5V2Env.num_kpis)])


def noisy_corpus(seed: int, kappa: float, n: int = N_ROWS):
    """E5 CORE corpus + observation noise eps ~ N(0, (kappa*sigma_k)^2), i.i.d. per row and KPI, own RNG stream."""
    x, y = e5_corpus(n=n, seed=seed)
    rng = np.random.default_rng([int(seed), 5, int(round(1000 * kappa))])
    noise = rng.standard_normal(y.shape) * (kappa * SIGMA)[None, :]
    return x, y + noise, noise


def _provenance() -> dict:
    reg = json.load(open(REGISTRY))
    own = reg["E5"]["claimed_by"]["E5_NOISE_DETECT"]
    assert own == [[TEST[0], TEST[-1]]], f"registry claim {own} != TEST range"
    used = [r for r in reg["E5"]["used"] if r not in own]
    bad = [s for s in TEST if any(lo <= s <= hi for lo, hi in used)]
    assert not bad, f"TEST seeds overlap prior E5 use: {bad[:5]}"
    files = ["scripts/e5_noise_detect.py", "docs/benchmark/E5_NOISE_DETECT.md", "docs/benchmark/SEED_REGISTRY.json",
             "cdd_oran/discovery/mscr.py", "cdd_oran/discovery/mscr_torch.py", "scripts/e5_baselines.py",
             "scripts/stage0_falsifiers.py", "scripts/runtime_info.py"]
    run = lambda *a: subprocess.run(list(a), capture_output=True, text=True, cwd=_REPO).stdout.strip()  # noqa: E731
    return {"git_head": run("git", "rev-parse", "HEAD"), "uncommitted": run("git", "status", "--porcelain", *files),
            "sha256": {f: hashlib.sha256(open(os.path.join(_REPO, f), "rb").read()).hexdigest() for f in files},
            "runtime": runtime_info()}


def collect() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    prov = os.path.join(OUT_DIR, "provenance.json")
    if not os.path.exists(prov):
        json.dump(_provenance(), open(prov, "w"), indent=1)
    done = set()
    if os.path.exists(REC):
        done = {tuple(json.loads(line)["key"]) for line in open(REC)}
    for kappa in KAPPAS:
        for split, seeds in (("dev", DEV), ("test", TEST)):
            for seed in seeds:
                key = (kappa, split, seed)
                if key in done:
                    continue
                t0 = time.time()
                x, y, noise = noisy_corpus(seed, kappa)
                mscr = discover_mscr(x, y, n_params=NP_, seed=seed).declared.astype(int).tolist()
                shap_i, corr_i = _imps(x, y)
                rec = {"key": list(key), "mscr": mscr, "shap": shap_i, "corr": corr_i,
                       "realized_noise_frac": (noise.std(0) / SIGMA).tolist(), "elapsed_s": round(time.time() - t0, 1)}
                with open(REC, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(json.dumps({"key": rec["key"], "elapsed_s": rec["elapsed_s"]}), flush=True)
    print("ALL DONE", flush=True)


def _select_tau(dev, det, truth):
    """DEV rule: highest harmful-edge recall s.t. DEV family FDR <= 0.05; tie-break lower FDR, then larger tau."""
    hk, hp = HARM
    cands = []
    for tau in TAUS:
        fdr = float(np.mean([_fam_fdr(_thresh_mask(r[det], tau), truth, NP_) for r in dev]))
        rec = float(np.mean([_thresh_mask(r[det], tau)[hk][hp] for r in dev]))
        cands.append({"tau": tau, "dev_recall": rec, "dev_fdr": fdr})
    valid = [c for c in cands if c["dev_fdr"] <= 0.05]
    if not valid:
        return None, cands
    best = max(valid, key=lambda c: (c["dev_recall"], -c["dev_fdr"], c["tau"]))
    return best["tau"], cands


def score() -> dict:
    truth = e5_truth()
    hk, hp = HARM
    recs = [json.loads(line) for line in open(REC)]
    out = {}
    for kappa in KAPPAS:
        by = {(r["key"][1], r["key"][2]): r for r in recs if r["key"][0] == kappa}
        dev = [by[("dev", s)] for s in DEV if ("dev", s) in by]
        test = [by[("test", s)] for s in TEST if ("test", s) in by]
        if len(dev) < len(DEV) or len(test) < len(TEST):
            out[str(kappa)] = {"status": f"INCOMPLETE dev {len(dev)}/{len(DEV)} test {len(test)}/{len(TEST)}"}
            continue
        m_hit = np.array([r["mscr"][hk][hp] for r in test], dtype=float)
        m_fdr = [_fam_fdr(r["mscr"], truth, NP_) for r in test]
        mscr = {"recall": float(m_hit.mean()), "fdr_ub": _ub95(m_fdr), "fdr_mean": float(np.mean(m_fdr))}
        mscr_meets = mscr["recall"] >= 0.95 and mscr["fdr_ub"] <= 0.05
        cell = {"MSCR": mscr, "MSCR_meets_target": mscr_meets,
                "realized_noise_frac_mean": np.mean([r["realized_noise_frac"] for r in test], axis=0).tolist()}
        for det in ("shap", "corr"):
            tau, cands = _select_tau(dev, det, truth)
            curve = {str(t): {"recall": float(np.mean([_thresh_mask(r[det], t)[hk][hp] for r in test])),
                              "fdr": float(np.mean([_fam_fdr(_thresh_mask(r[det], t), truth, NP_) for r in test]))}
                     for t in TAUS}
            if tau is None:
                matches, res = False, {"status": "no FDR-valid operating point on DEV"}
                b_hit = None
            else:
                b_hit = np.array([_thresh_mask(r[det], tau)[hk][hp] for r in test], dtype=float)
                b_fdr = [_fam_fdr(_thresh_mask(r[det], tau), truth, NP_) for r in test]
                lb = _lb95_diff(b_hit, m_hit)
                ub = _ub95(b_fdr)
                matches = bool(lb >= -0.05 and ub <= 0.05)
                res = {"tau": tau, "recall": float(b_hit.mean()), "paired_lb95_base_minus_mscr": lb, "fdr_ub": ub}
            if matches and mscr_meets:
                outcome = "1: baseline matches MSCR"
            elif mscr_meets:
                outcome = "2: MSCR meets target while baseline fails the operating-point criterion (narrow)"
            else:
                outcome = "3: inconclusive or both fail" + (" (MSCR missed its own target)" if not mscr_meets else "")
            adv_lb = _lb95_diff(m_hit, b_hit) if b_hit is not None else None
            res.update({"dev_candidates": cands, "test_curve": curve, "OUTCOME": outcome,
                        "recall_advantage_lb95_mscr_minus_base": adv_lb,
                        "RECALL_ADVANTAGE": bool(adv_lb is not None and adv_lb > 0.10)})
            cell[det] = res
        out[str(kappa)] = cell
    json.dump(out, open(os.path.join(OUT_DIR, "score.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    {"collect": collect, "score": score}[sys.argv[1]]()

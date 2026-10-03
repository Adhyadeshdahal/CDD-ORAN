"""Dependence inputs of the F_max simulation (PROTOCOL_A section 10, ruling R-30); DEV seeds only.

    uv run python scratchpad/xmethod/fmax_calib.py dependence --out scratchpad/xmethod/results/fmax_sim/
    uv run python scratchpad/xmethod/fmax_calib.py table --dep .../dependence.json --out .../

``dependence``: on DEV seeds 3_000_100-3_000_199 (n 1000, kappa .25) every primary null candidate (action -> KPI,
incl. P_placebo and, in E4 R3, P_placebo_conf) is tested with a fast partial-correlation z statistic given the
equal-information set Z_eq (``covariates.design_covariates(data, focal=a)``). From the 100 x m matrix of z per
(world, regime) it writes the mean between-candidate correlation of z by pair type (same source / same target /
neither), the across-lambda correlation of z in E4 R3, and a check of the nested-n (Brownian) assumption,
corr(z_n1, z_n2) vs sqrt(n1 / n2), on E1 R1 / R2 at n 500, 1000, 4000. Truth is used only to list the null
candidates. The statistic is a stand-in for any test's null statistic: only its dependence enters the simulation.

``se4``: PROTOCOL_A T9 (R-39): S_E4 = the smallest of {200, 300, 400, 600} at which an exactly nominal pmrt_eq
passes C3 (E4 R3, 20 cells) with probability >= .90.

``table``: F_max and the probabilities of the C1 / C2 / C3 legs at an exactly nominal level for the pre-registered
cell sets and S in {40, 50, ..., 100} (E4 R3 at S_E4 = 200), with ``eval_analysis.fmax_simulate``.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from cdd_oran.xmethod.covariates import design_covariates  # noqa: E402
from cdd_oran.xmethod.worlds import REGIMES_OF, truth_for  # noqa: E402
from cdd_oran.xmethod.worlds.generate import generate_dataset  # noqa: E402

_spec = importlib.util.spec_from_file_location("eval_analysis", os.path.join(os.path.dirname(__file__),
                                                                             "eval_analysis.py"))
EA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EA)

SEEDS = range(3_000_100, 3_000_200)
N_DEP, KAPPA = 1000, 0.25
LAMS = (0.0, 0.5, 1.0, 1.5)


def null_candidates(world: str, regime: str) -> list[tuple[str, str]]:
    t = truth_for(world, regime)
    return sorted(e for e in t.null_edges if not e[0].startswith("K"))


def z_stats(data, cands) -> np.ndarray:
    """Partial-correlation z of every candidate given Z_eq of its source (normal scores of the t statistic)."""
    out = []
    for s, k in cands:
        ai, ki = data.action_names.index(s), data.kpi_names.index(k)
        _, Z, _ = design_covariates(data, focal=ai)
        D = np.column_stack([np.ones(data.n), Z]) if Z.size else np.ones((data.n, 1))
        x, y = data.X_action[:, ai].astype(float), data.Y[:, ki].astype(float)
        rx = x - D @ np.linalg.lstsq(D, x, rcond=None)[0]
        ry = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
        r = float(np.dot(rx, ry) / np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
        df = data.n - 2 - (D.shape[1] - 1)
        t = r * np.sqrt(df / max(1e-12, 1 - r * r))
        out.append(float(stats.norm.ppf(np.clip(stats.t.cdf(t, df), 1e-15, 1 - 1e-15))))
    return np.array(out)


def pair_type_corr(zm: np.ndarray, cands) -> dict:
    """Mean correlation (over candidate pairs) of z across seeds, by pair type."""
    c = np.corrcoef(zm.T) if zm.shape[1] > 1 else np.ones((1, 1))
    acc: dict[str, list] = {"same_source": [], "same_target": [], "other": []}
    for i, j in itertools.combinations(range(len(cands)), 2):
        typ = ("same_source" if cands[i][0] == cands[j][0] else "same_target" if cands[i][1] == cands[j][1]
               else "other")
        acc[typ].append(c[i, j])
    return {k: (float(np.mean(v)) if v else 0.0) for k, v in acc.items()} | \
        {f"n_pairs_{k}": len(v) for k, v in acc.items()}


def dependence() -> dict:
    out: dict = {"seeds": [SEEDS.start, SEEDS.stop - 1], "n": N_DEP, "kappa": KAPPA, "worlds": {}}
    for w in ("E1", "E2", "E3", "E4", "E5"):
        for r in REGIMES_OF[w]:
            if r == "R4":
                continue
            cands = null_candidates(w, r)
            lams = LAMS if (w == "E4" and r == "R3") else (1.0,)
            zl = {lam: np.array([z_stats(generate_dataset(w, r, N_DEP, s, lam=lam, kappa=KAPPA)[0], cands)
                                 for s in SEEDS]) for lam in lams}
            zm = zl[1.0]
            row = {"candidates": [f"{a}->{b}" for a, b in cands], "z_mean": float(zm.mean()),
                   "z_sd": float(zm.std(ddof=1)), **pair_type_corr(zm, cands)}
            if len(lams) > 1:
                cl = np.zeros((len(lams), len(lams)))
                for j in range(len(cands)):
                    cl += np.corrcoef(np.column_stack([zl[lam][:, j] for lam in lams]).T)
                row["lambda_corr"] = (cl / len(cands)).round(4).tolist()
                row["lambdas"] = list(lams)
            out["worlds"][f"{w}|{r}"] = row
            print(w, r, {k: v for k, v in row.items() if k != "candidates"}, flush=True)
    chk = {}
    for r in ("R1", "R2"):
        cands = null_candidates("E1", r)
        zn = {n: np.array([z_stats(generate_dataset("E1", r, n, s, kappa=KAPPA)[0], cands) for s in SEEDS])
              for n in (500, 1000, 4000)}
        for a, b in ((500, 1000), (1000, 4000), (500, 4000)):
            cc = float(np.mean([np.corrcoef(zn[a][:, j], zn[b][:, j])[0, 1] for j in range(len(cands))]))
            chk[f"E1|{r}|{a}-{b}"] = {"corr": round(cc, 4), "brownian": round(float(np.sqrt(a / b)), 4)}
    out["nested_n_check"] = chk
    print(chk, flush=True)
    return out


def table(dep: dict) -> dict:
    """F_max and leg probabilities for the pre-registered cell sets (PROTOCOL_A section 10)."""
    ns = (500, 1000, 4000, 8000, 24000)
    w5 = ("E1", "E2", "E3", "E4", "E5")
    sets = {
        "R1R2_all": [(w, r, 1.0 if w == "E4" else None, n) for w in w5 for r in ("R1", "R2") for n in ns],
        "R1_all": [(w, "R1", 1.0 if w == "E4" else None, n) for w in w5 for n in ns],
        "R2_all": [(w, "R2", 1.0 if w == "E4" else None, n) for w in w5 for n in ns],
        "R1R2_E3": [("E3", r, None, n) for r in ("R1", "R2") for n in ns],
        "R1_E3": [("E3", "R1", None, n) for n in ns],
        "E4R3": [("E4", "R3", lam, n) for lam in LAMS for n in ns],
    }
    out: dict = {"nsim": EA.FMAX_NSIM, "sim_seed": EA.FMAX_SEED, "sets": {}}
    for name, cells in sets.items():
        keys = ("plac_raw", "conf_raw") if name == "E4R3" else ("null_raw", "plac_raw")
        for S in ((200,) if name == "E4R3" else range(40, 101, 10)):
            res = EA.fmax_simulate(cells, S, keys, dep)
            out["sets"][f"{name}|S{S}"] = res
            print(name, S, {k: v for k, v in res.items() if k != "count_dist"}, flush=True)
    return out


def se4(dep: dict) -> dict:
    cells = [("E4", "R3", lam, n) for lam in LAMS for n in (500, 1000, 4000, 8000, 24000)]
    out = {"candidates": {}}
    for S in (200, 300, 400, 600):
        out["candidates"][str(S)] = EA.fmax_simulate(cells, S, ("plac_raw", "conf_raw"), dep)
        print(S, out["candidates"][str(S)]["p_pass"], flush=True)
    out["S_E4"] = next((int(S) for S, r in out["candidates"].items() if r["p_pass"] >= 0.90), None)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dependence", "table", "se4"))
    ap.add_argument("--dep")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "dependence":
        res, name = dependence(), "dependence.json"
    elif a.cmd == "se4":
        res, name = se4(EA.load_dependence(a.dep)), "s_e4_rule.json"
    else:
        res, name = table(EA.load_dependence(a.dep)), "fmax_table.json"
    with open(os.path.join(a.out, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(EA._clean(res), indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

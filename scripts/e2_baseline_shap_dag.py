"""SHAP-DAG baseline (Sharma et al. 2025, arXiv 2510.13031) through the E2 conflict spine.

Faithful reimplementation of "Towards xApp Conflict Evaluation with Explainable Machine Learning
and Causal Inference in O-RAN" (Sharma et al., 2025), Sec. III-IV:

  1. Fit an ML regressor of each KPI on the RCPs (params). The paper compares DecisionTree / RF /
     XGBoost / MLP / SVR and SELECTS XGBoost (best R^2, Table II). We substitute sklearn
     HistGradientBoostingRegressor -- the same gradient-boosted-decision-tree family as XGBoost,
     SHAP-TreeExplainer-compatible -- because xgboost is not in the env. (Faithful family match;
     stated substitution.)
  2. Explain with SHAP (TreeExplainer); rank RCPs by mean |SHAP| per KPI (paper Table III(b),
     Fig. 4).
  3. Build the causal DAG by "creating causal edges from the RCPs which are the most influential
     towards their associated KPIs" (Sec. IV-C.1). The paper's threshold is QUALITATIVE ("most
     influential"), so we use a relative-dominance rule -- keep param p as a parent of KPI k iff
     mean|SHAP|_p >= tau_rel * max_p' mean|SHAP|_p' for that KPI -- and SWEEP tau_rel to show the
     conclusion is not threshold-cherry-picked. Sharma regresses KPIs on RCPs ONLY (no lagged
     KPIs), so lagged-KPI mask columns (8..13) are always 0 -- faithful to their design and inert
     in the param-only spine.
  4. Descriptive ATE (their actual deliverable, Sec. IV-C.2, Table IV): DoWhy/EconML backdoor ATE.
     We substitute the identical estimand -- backdoor-adjusted OLS: regress y5 on P0 plus the other
     declared K5-parents (confounders) and read the P0 coefficient -- which is exactly DoWhy's
     linear-regression backdoor estimator / Sharma's LinearDML sensitivity check. (Stated
     substitution; dowhy/econml not in env.)

The DAG's param->KPI edges are then decoded and pushed through the SAME frozen decision spine as
M3/RCoT/pdCor (cdd_oran.benchmark.MaskedE2WorldModel -> do-propagation -> gap_norm regret), so
SHAP-DAG is scored on identical footing.

TRUTH-FREE discovery (SHAP fit uses only rows.npz); ground truth is used ONLY to grade recovery /
regret. No frozen module is edited.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import shap
from sklearn.ensemble import HistGradientBoostingRegressor

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cdd_oran.benchmark.masked_world_model import param_edges_from_binary_mask  # noqa: E402
from cdd_oran.envs.v2.e2 import E2V2Env  # noqa: E402
from scripts.e2_decision_gate import (  # noqa: E402
    FULL_PANEL_IDS,
    build_bank,
    build_panel,
)
from scripts.e2_spine import HARMFUL_EDGE, spine_regret  # noqa: E402

NUM_PARAMS = E2V2Env.num_params      # 8
NUM_KPIS = E2V2Env.num_kpis          # 6
NUM_CANDIDATES = NUM_PARAMS + NUM_KPIS  # 14 (params + lagged KPIs), to match RCoT/pdCor mask width
TAU_GRID = (0.05, 0.10, 0.20, 0.30)  # relative-dominance thresholds swept
TAU_PRIMARY = 0.10


def fit_shap_importances(X, y, seed=0):
    """Fit the GBDT regressor and return (mean|SHAP| per param feature, R^2)."""
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=seed)
    model.fit(X, y)
    r2 = float(model.score(X, y))
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    return np.abs(sv).mean(axis=0), r2


def shap_dag_mask(X_params, Y_kpis, tau_rel, seed=0):
    """Build the 6x14 SHAP-DAG binary_mask at a relative-dominance threshold tau_rel.

    Edge (kpi, p) = 1 iff mean|SHAP|_p >= tau_rel * max_p' mean|SHAP|_p' for that KPI. Lagged-KPI
    columns (>= NUM_PARAMS) stay 0 (Sharma regresses on RCPs only).
    """
    mask = np.zeros((NUM_KPIS, NUM_CANDIDATES), dtype=int)
    imps, r2s = [], []
    for k in range(NUM_KPIS):
        imp, r2 = fit_shap_importances(X_params, Y_kpis[:, k], seed=seed)
        imps.append(imp)
        r2s.append(r2)
        thr = tau_rel * float(imp.max())
        for p in range(NUM_PARAMS):
            if imp[p] >= thr:
                mask[k, p] = 1
    return mask, np.array(imps), np.array(r2s)


def backdoor_ate_p0_k5(X_params, y5, declared_k5_parents):
    """Sharma's descriptive ATE(P0->K5): backdoor-adjusted OLS coefficient on P0.

    Regress y5 on [1, P0, <other declared K5 parents as confounders>]; the P0 coefficient is the
    linear backdoor ATE (DoWhy's linear estimator). Returns (ate, n_confounders).
    """
    confounders = sorted(p for (k, p) in declared_k5_parents if k == 5 and p != 0)
    cols = [np.ones(len(y5)), X_params[:, 0]] + [X_params[:, c] for c in confounders]
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, y5, rcond=None)
    return float(beta[1]), len(confounders)


def load_replicates(repo_root=_REPO_ROOT):
    out = {}
    for path in sorted(glob.glob(os.path.join(repo_root, "runs", "e2slice-recovery",
                                              "replicate-*", "rows.npz"))):
        rep = int(os.path.basename(os.path.dirname(path)).split("-")[1])
        d = np.load(path)
        out[rep] = (d["x_params"], d["y_kpis"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "scratchpad", "shap_dag",
                                                  "shap_dag_results.json"))
    args = ap.parse_args(argv)

    bank = build_bank(E2V2Env)
    assert bank.feasible
    positives = bank.positives
    panel = build_panel(E2V2Env(0), FULL_PANEL_IDS)
    reps = load_replicates()
    print(f"replicates={list(reps)}  bank positives={len(positives)}")

    results = {"tau_primary": TAU_PRIMARY, "tau_grid": list(TAU_GRID),
               "predictor": "sklearn.HistGradientBoostingRegressor(max_iter=300, lr=0.1)",
               "note": "faithful Sharma 2025; GBDT<-XGBoost, OLS-backdoor<-DoWhy substitutions",
               "per_tau": {}, "primary_per_replicate": {}}

    for tau in TAU_GRID:
        pooled_gaps, harmful_hits, total_mism, p0k5_ratios, ates = [], 0, 0, [], []
        for rep, (X, Y) in reps.items():
            mask, imps, r2s = shap_dag_mask(X, Y, tau, seed=0)
            edges = param_edges_from_binary_mask(mask)
            gaps, mism = spine_regret(edges, positives, panel)
            has_h = HARMFUL_EDGE in edges
            harmful_hits += int(has_h)
            total_mism += mism
            pooled_gaps.extend(gaps)
            ratio = float(imps[5, 0] / imps[5].max())
            p0k5_ratios.append(ratio)
            ate, ncf = backdoor_ate_p0_k5(X, Y[:, 5], edges)
            ates.append(ate)
            if tau == TAU_PRIMARY:
                results["primary_per_replicate"][str(rep)] = {
                    "mean_pos": float(np.mean(gaps)), "has_P0_K5": has_h,
                    "action_mismatches": mism, "n_param_edges": len(edges),
                    "P0_K5_shap_ratio": ratio, "K5_R2": float(r2s[5]),
                    "ATE_P0_K5_backdoor": ate, "n_confounders": ncf,
                }
        results["per_tau"][str(tau)] = {
            "harmful_recovered": harmful_hits, "n_replicates": len(reps),
            "pooled_mean_pos": float(np.mean(pooled_gaps)),
            "action_mismatches_total": total_mism,
            "mean_P0_K5_shap_ratio": float(np.mean(p0k5_ratios)),
            "mean_ATE_P0_K5": float(np.mean(ates)),
        }
        print(f"tau_rel={tau}: P0->K5 recovered {harmful_hits}/{len(reps)}  "
              f"pooled mean_pos={np.mean(pooled_gaps):.6f}  mism={total_mism}  "
              f"mean SHAP ratio(P0/max for K5)={np.mean(p0k5_ratios):.3f}  "
              f"mean ATE(P0->K5)={np.mean(ates):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

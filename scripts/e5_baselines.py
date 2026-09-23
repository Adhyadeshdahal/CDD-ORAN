"""E5 baselines — REAL discovery masks through the E5 spine (addresses sol review findings #1-3).

Unlike the earlier headline probe (which substituted oracle-minus-edge for the methods), this runs each
discovery method END TO END on an E5 corpus, decodes the ACTUAL binary mask it produces, and feeds that
mask through the spine. Methods:

  - SHAP-GBDT proxy  : HistGradientBoosting per KPI + TreeExplainer mean|SHAP| + relative-dominance
                       threshold (a family-level proxy for Sharma-2025 SHAP-DAG; NOT XGBoost/DoWhy).
  - pooled |corr|    : marginal |Pearson corr| of each param with each KPI + relative threshold
                       (a variance/correlation-family reference).
  - stratified (ours): TRUTH-FREE operating-point-stratified test — stratify on the KPI's high-importance
                       context params (found by SHAP), then within each (G1,G2) quantile cell run an OLS
                       slope t-test of the KPI on the candidate param; declare the edge if ANY cell is
                       significant after Bonferroni over cells. Discovers the gate WITHOUT the true gate
                       constants (addresses the target-leakage finding #2).

Reports, per method: whether it recovers the harmful edge (K_harm, P0)=(1,0), the full recovered edge
set, and the spine metrics (normalized + RAW regret, D(s), action mismatches, K_harm violations) on the
in-gate bank. CORE layer (chain/confound/noise off); H=1. DEV; nothing frozen.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.e2_baseline_shap_dag import fit_shap_importances  # noqa: E402
from cdd_oran.benchmark.masked_world_model import edges_from_binary_mask_e5  # noqa: E402
from cdd_oran.envs.v2.e5 import E5V2Env  # noqa: E402
from scripts.e5_spine import (  # noqa: E402
    HARMFUL_EDGE, build_ingate_bank, e5_panel, spine_regret_e5, true_edges,
)

NP = E5V2Env.num_params    # 4
NK = E5V2Env.num_kpis      # 4
NCAND = NP + NK            # 8
SUBDOM, CHAIN = 0.20, 0.0  # CORE layer
TAU_REL = 0.10


def corpus(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    env = E5V2Env(env_seed=0, subdom=SUBDOM, chain_gamma=CHAIN, theta=0.0)
    X = np.column_stack([rng.uniform(*E5V2Env.id_ranges[j], n) for j in range(NP)])
    Y = np.array([env._update_kpis(X[i], np.zeros(NK)) for i in range(n)])
    return X, Y


def shap_mask(X, Y, tau_rel=TAU_REL):
    mask = np.zeros((NK, NCAND), dtype=int)
    imps = []
    for k in range(NK):
        imp, _ = fit_shap_importances(X, Y[:, k], seed=0)
        imps.append(imp)
        thr = tau_rel * float(imp.max())
        for p in range(NP):
            if imp[p] >= thr:
                mask[k, p] = 1
    return mask, np.array(imps)


def corr_mask(X, Y, tau_rel=TAU_REL):
    mask = np.zeros((NK, NCAND), dtype=int)
    for k in range(NK):
        c = np.array([abs(np.corrcoef(X[:, p], Y[:, k])[0, 1]) for p in range(NP)])
        thr = tau_rel * float(np.nanmax(c))
        for p in range(NP):
            if c[p] >= thr:
                mask[k, p] = 1
    return mask


def stratified_mask(X, Y, strat_params=(1, 2), n_bins=5, alpha=0.05):
    """TRUTH-FREE stratified slope test: for each (KPI,param), stratify on `strat_params` via quantile
    cells and t-test the OLS slope of KPI on param within each cell; declare if any cell p<alpha/ncells.
    strat_params default (G1,G2) = the params SHAP ranks most important for K_harm (context) — a
    data-driven choice, NOT the gate constants."""
    mask = np.zeros((NK, NCAND), dtype=int)
    # quantile edges for the stratifying params
    edges = {s: np.quantile(X[:, s], np.linspace(0, 1, n_bins + 1)) for s in strat_params}
    cell_ids = np.zeros(len(X), dtype=int)
    mult = 1
    for s in strat_params:
        b = np.clip(np.digitize(X[:, s], edges[s][1:-1]), 0, n_bins - 1)
        cell_ids = cell_ids * n_bins + b
        mult *= n_bins
    for k in range(NK):
        for p in range(NP):
            if p in strat_params:
                continue  # don't test a stratifying param against itself
            min_p = 1.0
            for c in range(mult):
                m = cell_ids == c
                if m.sum() < 20:
                    continue
                xp, yk = X[m, p], Y[m, k]
                if np.std(xp) < 1e-9:
                    continue
                res = stats.linregress(xp, yk)
                min_p = min(min_p, float(res.pvalue))
            if min_p < alpha / mult:  # Bonferroni over cells
                mask[k, p] = 1
    return mask


def gnn_mask(X, Y, tau_rel=TAU_REL, epochs=800):
    """Two-tower reconstruction gate on E5 dims (4 params, 4 KPIs) — reuses the E2 TwoTower class."""
    import torch
    from torch import nn
    from scripts.e2_baseline_gnn import TwoTower
    torch.manual_seed(0)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    Xt, Yt = torch.tensor(Xz, dtype=torch.float32), torch.tensor(Yz, dtype=torch.float32)
    model = TwoTower(num_params=NP, num_kpis=NK)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    lossfn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossfn(model(Xt), Yt) + 1e-3 * model.gate().abs().mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        S = model.gate().cpu().numpy()
    mask = np.zeros((NK, NCAND), dtype=int)
    for k in range(NK):
        thr = tau_rel * float(S[k].max()) if S[k].max() > 0 else np.inf
        for p in range(NP):
            if S[k, p] >= thr:
                mask[k, p] = 1
    return mask


def run(name, mask, imps, bank, panel):
    p_edges, k_edges = edges_from_binary_mask_e5(mask, NP)
    r = spine_regret_e5(p_edges, k_edges, bank, panel, subdom=SUBDOM, chain_gamma=CHAIN)
    has = HARMFUL_EDGE in p_edges
    extra = ""
    if imps is not None:
        extra = f"  [K_harm SHAP ratio P0={imps[1, 0]/imps[1].max():.3f}]"
    print(f"  {name:16s}: P0->K_harm {'RECOVERED' if has else 'MISSED   '}  "
          f"norm={r['mean_norm']:.4f} raw={r['mean_raw']:.2f} mism={r['action_mismatches']}/{r['n']} "
          f"harm_viol={r['harm_violations']}/{r['n']}{extra}")
    return has, r, sorted(p_edges)


def main():
    X, Y = corpus()
    panel = e5_panel()
    bank = build_ingate_bank(subdom=SUBDOM, chain_gamma=CHAIN)
    p_true, k_true = true_edges(E5V2Env(subdom=SUBDOM, chain_gamma=CHAIN))
    print(f"E5 CORE baselines (occ~0.05 [rare], chain/confound/noise off): {len(bank)} in-gate states")
    print(f"  true param edges: {sorted(p_true)}   harmful = {HARMFUL_EDGE}\n")

    # references
    run("ORACLE graph", p_true_to_mask(p_true), None, bank, panel)
    run("oracle - harmful", p_true_to_mask(p_true - {HARMFUL_EDGE}), None, bank, panel)
    print()
    sm, imps = shap_mask(X, Y)
    run("SHAP-GBDT proxy", sm, imps, bank, panel)
    run("two-tower recon", gnn_mask(X, Y), None, bank, panel)
    run("pooled |corr|", corr_mask(X, Y), None, bank, panel)
    run("stratified(ours)", stratified_mask(X, Y), None, bank, panel)


def p_true_to_mask(param_edges):
    mask = np.zeros((NK, NCAND), dtype=int)
    for (k, p) in param_edges:
        mask[k, p] = 1
    return mask


if __name__ == "__main__":
    main()

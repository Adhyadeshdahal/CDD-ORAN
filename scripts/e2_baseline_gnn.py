"""Correlational graph-learning baseline (two-tower interaction learning) through the E2 spine.

Faithful reimplementation of the correlational graph-reconstruction family for xApp conflict
detection, primarily the TWO-TOWER encoder of "Efficient Interaction Learning and Autonomous Graph
Reconstruction" (reference-papers/Two-tower.md), with the GRAPHICA GCN paper
(O-RAN_xApps_Conflict_Management_using_Graph_Convolutional_Networks) as the supervised sibling.

Why two-tower and not GRAPHICA's GCN: GRAPHICA is a SUPERVISED conflict CLASSIFIER trained on
labeled conflict graphs (direct/indirect/implicit), and needs conflict labels we do not have on the
observational discovery rows. The two-tower method is the UNSUPERVISED structure-reconstruction
member of the same correlational-graph family: it "models interaction learning as a ranking problem
between parameters and KPIs" and reconstructs a binary param/KPI adjacency A ∈ {0,1}^{Na×(Np+Nk)}
by scoring param↔KPI relevance (the paper's own scoring options include correlation / similarity)
and sparsity-thresholding. That maps DIRECTLY onto our param→KPI mask, on identical footing with
M3/RCoT/pdCor/SHAP-DAG.

Faithful realization (stated design choices where the paper is unspecified):
  - Two-tower encoder: a shared per-param MLP ψ:R^1→R^d plus a learnable per-param id embedding
    gives a param-sample embedding e_{n,p}=ψ(x_{n,p})+id_p (the PARAM TOWER); each KPI has a
    learnable embedding w_k∈R^d (the KPI TOWER). The interaction is the dot product <e_{n,p}, w_k>
    (the paper's two-tower similarity), and the reconstruction predicts ŷ_{n,k}=b_k+Σ_p<e_{n,p},w_k>.
    Trained by MSE(ŷ, y) on standardized data (self-supervised reconstruction, no truth/labels).
  - Interaction relevance score S[k,p] = std_n <e_{·,p}, w_k>: how much param p's learned
    tower-interaction moves KPI k's prediction (a param that does not drive KPI k has ~0-variance
    contribution). This is the paper's "score capturing the relevance of a param↔KPI pair".
  - Graph reconstruction: sparsity threshold — keep edge (k,p) iff S[k,p] >= tau_rel * max_p' S[k,p']
    (relative-dominance), swept over tau_rel to avoid cherry-picking (same rule/sweep as SHAP-DAG for
    comparability). Lagged-KPI columns (>= num_params) stay 0 (params-only reconstruction here).

TRUTH-FREE discovery (the fit sees only rows.npz); ground truth is used ONLY to grade recovery /
regret. No frozen module is edited. torch is a core dep.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cdd_oran.benchmark.masked_world_model import param_edges_from_binary_mask  # noqa: E402
from cdd_oran.envs.v2.e2 import E2V2Env  # noqa: E402
from scripts.e2_decision_gate import FULL_PANEL_IDS, build_bank, build_panel  # noqa: E402
from scripts.e2_spine import HARMFUL_EDGE, spine_regret  # noqa: E402

NUM_PARAMS = E2V2Env.num_params        # 8
NUM_KPIS = E2V2Env.num_kpis            # 6
NUM_CANDIDATES = NUM_PARAMS + NUM_KPIS  # 14
TAU_GRID = (0.05, 0.10, 0.20, 0.30)
TAU_PRIMARY = 0.10


class TwoTower(nn.Module):
    """Two-tower interaction gate + per-KPI nonlinear reconstruction head (sparsity-selected).

    Param tower V (P×r) × KPI tower U (K×r) → interaction gate a[k,p]=softplus(<U_k,V_p>); the param
    sample embedding e_{n,p}=ψ(x_{n,p})+id_p is aggregated per KPI as z_{n,k}=Σ_p a[k,p]·e_{n,p}, and a
    per-KPI head reconstructs ŷ_{n,k}=head_k(z_{n,k}) — so the head can fit the joint nonlinearity
    (bump) while the gate a[k,p] concentrates on the params that drive KPI k. Relevance S[k,p]=a[k,p].
    """

    def __init__(self, num_params=NUM_PARAMS, num_kpis=NUM_KPIS, d=16, r=8, hidden=32):
        super().__init__()
        self.psi = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, d))
        self.id_emb = nn.Parameter(torch.zeros(num_params, d))
        self.U = nn.Parameter(torch.randn(num_kpis, r) * 0.3)   # KPI tower
        self.V = nn.Parameter(torch.randn(num_params, r) * 0.3)  # param tower
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1)) for _ in range(num_kpis)]
        )
        self.num_params, self.num_kpis, self.d = num_params, num_kpis, d

    def gate(self):
        """a[k,p] = softplus(<U_k, V_p>) >= 0 — the learned param↔KPI interaction strength."""
        return torch.nn.functional.softplus(self.U @ self.V.t())   # (K, P)

    def forward(self, X):
        n = X.shape[0]
        e = self.psi(X.reshape(n * self.num_params, 1)).reshape(n, self.num_params, self.d)
        e = e + self.id_emb[None, :, :]                            # (N, P, d)
        a = self.gate()                                           # (K, P)
        z = torch.einsum("kp,npd->nkd", a, e)                     # (N, K, d) gate-weighted aggregate
        return torch.cat([self.heads[k](z[:, k, :]) for k in range(self.num_kpis)], dim=1)  # (N,K)


def fit_two_tower(X, Y, epochs=800, lr=0.01, l1=1e-3, seed=0, device="cpu"):
    """Train the two-tower reconstruction (+L1 sparsity on the gate); return scores S[k,p] and R^2."""
    torch.manual_seed(seed)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    Xt = torch.tensor(Xz, dtype=torch.float32, device=device)
    Yt = torch.tensor(Yz, dtype=torch.float32, device=device)
    model = TwoTower().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = lossfn(pred, Yt) + l1 * model.gate().abs().mean()   # sparsity-inducing selection
        loss.backward()
        opt.step()
    with torch.no_grad():
        S = model.gate().cpu().numpy()                            # (K, P): learned interaction gate
        resid = float(((model(Xt) - Yt) ** 2).mean())
        r2 = 1.0 - resid                                          # standardized -> pseudo-R^2
    return S, float(r2)


def two_tower_mask(S, tau_rel):
    """6x14 binary mask from interaction scores S (K x P) by relative-dominance threshold."""
    mask = np.zeros((NUM_KPIS, NUM_CANDIDATES), dtype=int)
    for k in range(NUM_KPIS):
        thr = tau_rel * float(S[k].max()) if S[k].max() > 0 else np.inf
        for p in range(NUM_PARAMS):
            if S[k, p] >= thr:
                mask[k, p] = 1
    return mask


def load_replicates(repo_root=_REPO_ROOT):
    out = {}
    for path in sorted(glob.glob(os.path.join(repo_root, "runs", "e2slice-recovery",
                                              "replicate-*", "rows.npz"))):
        rep = int(os.path.basename(os.path.dirname(path)).split("-")[1])
        d = np.load(path)
        out[rep] = (d["x_params"].astype(np.float64), d["y_kpis"].astype(np.float64))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "scratchpad", "gnn_baseline",
                                                  "gnn_results.json"))
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args(argv)

    bank = build_bank(E2V2Env)
    assert bank.feasible
    positives = bank.positives
    panel = build_panel(E2V2Env(0), FULL_PANEL_IDS)
    reps = load_replicates()
    print(f"replicates={list(reps)}  bank positives={len(positives)}  epochs={args.epochs}")

    # Fit each replicate ONCE; score all thresholds off the same interaction scores.
    scores = {}
    for rep, (X, Y) in reps.items():
        S, r2 = fit_two_tower(X, Y, epochs=args.epochs, seed=0)
        scores[rep] = S
        print(f"  rep {rep}: fit pseudo-R2~{r2:.3f}  S[K5]={np.round(S[5], 3)}  "
              f"P0/max(K5)={S[5,0]/S[5].max():.3f}", flush=True)

    results = {"tau_grid": list(TAU_GRID), "tau_primary": TAU_PRIMARY,
               "method": "two-tower interaction learning + sparsity graph reconstruction",
               "per_tau": {}, "primary_per_replicate": {}}
    for tau in TAU_GRID:
        pooled, harmful_hits, mism_total, ratios = [], 0, 0, []
        for rep, S in scores.items():
            mask = two_tower_mask(S, tau)
            edges = param_edges_from_binary_mask(mask)
            gaps, mism = spine_regret(edges, positives, panel)
            has_h = HARMFUL_EDGE in edges
            harmful_hits += int(has_h)
            mism_total += mism
            pooled.extend(gaps)
            ratios.append(float(S[5, 0] / S[5].max()))
            if tau == TAU_PRIMARY:
                results["primary_per_replicate"][str(rep)] = {
                    "mean_pos": float(np.mean(gaps)), "has_P0_K5": has_h,
                    "action_mismatches": mism, "n_param_edges": len(edges),
                    "P0_K5_score_ratio": float(S[5, 0] / S[5].max()),
                }
        results["per_tau"][str(tau)] = {
            "harmful_recovered": harmful_hits, "n_replicates": len(reps),
            "pooled_mean_pos": float(np.mean(pooled)), "action_mismatches_total": mism_total,
            "mean_P0_K5_score_ratio": float(np.mean(ratios)),
        }
        print(f"tau_rel={tau}: P0->K5 recovered {harmful_hits}/{len(reps)}  "
              f"pooled mean_pos={np.mean(pooled):.6f}  mism={mism_total}  "
              f"mean score ratio(P0/max K5)={np.mean(ratios):.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

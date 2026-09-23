"""E5 CHAIN LAYER — the 2-hop mediated harmful path needs H>=2 (DEV, plan 015 §3; sol review #4).

Layer: P0 -> K_mid -> K_harm (K_mid = C_CHAIN*P0, one-step lag; K_harm += chain_gamma*prev_K_mid, gated).
The acted P0 reaches K_harm only after flowing P0->K_mid (one advance) then K_mid->K_harm (another), so the
SCORED terminal needs 3 advances (H=2), not the H=1 convention's 2. Consequences this harness demonstrates:

  * At H=1 the chain edge (K_harm <- K_mid) is INERT: the scored terminal reads the PRE-action K_mid, so a
    world model that omits the chain edge scores identically to the full graph. H=1 cannot see the chain.
  * At H=2 the chain is ACTIVE: omitting the mediated path under-corrects (the in-gate harm is subdom (direct)
    + chain_gamma*C_CHAIN (chain) = 0.40, twice the direct-only 0.20), so the planner walks into the trap.
  * The chain edge is a KPI->KPI edge, which param->KPI-only discovery (SHAP-DAG/GNN/pooled-corr as run in
    scripts/e5_baselines.py) structurally cannot even propose — discovery must trace the 2-hop path.

Reuses the H-parameterized spine (scripts/e5_spine.spine_regret_e5(..., H=)). CORE otherwise (confound/noise
off), in-gate-conditional regret. Nothing frozen.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cdd_oran.envs.v2.e5 import E5V2Env  # noqa: E402
from scripts.e5_spine import (  # noqa: E402
    CHAIN_EDGE, HARMFUL_EDGE, build_ingate_bank, e5_panel, spine_regret_e5, true_edges,
)

SUBDOM, CHAIN = 0.20, 0.20


def main():
    panel = e5_panel()
    bank = build_ingate_bank(subdom=SUBDOM, chain_gamma=CHAIN)
    p_true, k_true = true_edges(E5V2Env(subdom=SUBDOM, chain_gamma=CHAIN))
    arms = {
        "oracle (direct+chain)": (p_true, k_true),
        "miss CHAIN edge": (p_true, frozenset(k_true) - {CHAIN_EDGE}),      # direct only
        "miss DIRECT edge": (p_true - {HARMFUL_EDGE}, k_true),              # chain only
        "miss BOTH": (p_true - {HARMFUL_EDGE}, frozenset(k_true) - {CHAIN_EDGE}),
    }
    print(f"E5 CHAIN layer (subdom={SUBDOM}, chain_gamma={CHAIN}) — {len(bank)} in-gate states")
    print(f"  true param edges {sorted(p_true)}  chain(KPI->KPI) edge {CHAIN_EDGE}")
    print(f"  in-gate total harm at H>=2 = subdom + chain_gamma*C_CHAIN = {SUBDOM + CHAIN*E5V2Env.C_CHAIN}\n")
    print(f"  {'world model':24s} {'H=1 norm':>9s} {'H=1 viol':>9s} {'H=2 norm':>9s} {'H=2 viol':>9s}")
    for name, (pe, ke) in arms.items():
        r1 = spine_regret_e5(pe, ke, bank, panel, subdom=SUBDOM, chain_gamma=CHAIN, H=1)
        r2 = spine_regret_e5(pe, ke, bank, panel, subdom=SUBDOM, chain_gamma=CHAIN, H=2)
        print(f"  {name:24s} {r1['mean_norm']:9.4f} {r1['harm_violations']:>4d}/{r1['n']:<4d} "
              f"{r2['mean_norm']:9.4f} {r2['harm_violations']:>4d}/{r2['n']:<4d}")
    print("\n  Read: miss-CHAIN is INERT at H=1 (== oracle) but incurs regret at H=2 — the mediated path is")
    print("  only scorable at H>=2. Discovery must recover the KPI->KPI chain edge, not just param edges.")


if __name__ == "__main__":
    main()

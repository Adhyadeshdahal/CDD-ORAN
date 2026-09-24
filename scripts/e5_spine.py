"""E5 spine — discovery structure → do-propagation → decision on the composed E5 env (DEV, plan 015).

Analogue of ``scripts/e2_spine.py`` for ``E5V2Env``. A discovered structure (param edges + chain
KPI-edges) is run through ``MaskedE5WorldModel`` (frozen mechanism, latent Z integrated out); the action
it selects is scored under the TRUE env (Z integrated out = E5V2Env(theta=0)) to give normalized regret:

    0.0    the planner picks the oracle-optimal action (it recovered the subdominant gated harmful edge)
    > 0    it walked into the trap (it pruned/missed the harmful edge on the in-gate states)

DEV: E5 is not frozen; panel is in raw units (standardization pending GATE_CONTRACT_E5). H=1 here isolates
the direct gated edge (the CORE layer); the chain layer needs H=2 (added when that layer is validated).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cdd_oran.analysis.v2_regret import P0_GRID, PanelXApp, reward, score_grid  # noqa: E402
from cdd_oran.benchmark.masked_world_model import MaskedE5WorldModel  # noqa: E402
from cdd_oran.envs.v2.e5 import E5V2Env, _gate  # noqa: E402

P0 = E5V2Env.P0
HARMFUL_EDGE = (E5V2Env.K_HARM, E5V2Env.P0)  # (1, 0): the subdominant gated direct edge
CHAIN_EDGE = (E5V2Env.K_HARM, E5V2Env.K_MID)  # KPI-source: the 2-hop chain edge (needs H>=2 to score)
NEUTRAL_STEPS = 3


def _rollout_terminal(env, param_id, v, H):
    """apply_action(param_id,v) then (H+1) advances; return the terminal (scored) latent KPI vector.
    H=1 is the frozen v2_regret call list (apply + 2 advances). H>=2 lets the acted param traverse a
    mediated KPI->KPI chain (each extra hop needs one more advance; sol review #4)."""
    env.apply_action(param_id, float(v))
    k = None
    for _ in range(H + 1):
        k = env.advance()
    return k


def score_grid_h(env, snap, param_id, panel, H, grid=P0_GRID):
    """``score_grid`` generalized to horizon H. H=1 delegates to the FROZEN v2_regret path (byte-identical);
    H>=2 rolls (H+1) advances so a 2-hop chain surfaces in the scored terminal."""
    if H == 1:
        return score_grid(env, snap, param_id, panel, grid)
    scores = np.empty(len(grid), dtype=float)
    for i, v in enumerate(grid):
        env.restore(snap)
        scores[i] = reward(_rollout_terminal(env, param_id, float(v), H), panel)
    return scores


# Per-KPI (mean, std) from scripts/e5_standardization.py (n=1e6, seed 0, uniform id_ranges, independent
# lagged P0 for the chain term, composed subdom=chain_gamma=0.2, theta=0 = Z at its mean, no noise).
E5_KPI_MOMENTS = {
    0: (4.161893, 4.634914),     # K_ben
    1: (23.514415, 24.662764),   # K_harm
    2: (0.075764, 57.680719),    # K_mid (conduit; not in the panel)
    3: (41.874536, 20.192141),   # K_dist
}


def e5_panel(standardized: bool = False) -> list[PanelXApp]:
    """Panel {K_ben above, K_harm below, K_dist above}; standardized by E5_KPI_MOMENTS if requested.

    Default stays RAW for now: the DEV reward geometry was tuned in raw units, and under the locked
    standardized convention the decision contrast collapses (CORE 0.378 -> 0.0095, CONFOUND DiD
    0.206 -> 0.0008; scratchpad/e5_design/std_rerun/). E5 must be re-balanced before its freeze.
    """
    return [
        PanelXApp((k,), *(E5_KPI_MOMENTS[k] if standardized else (0.0, 1.0)),
                  E5V2Env.kpi_thresholds[k], E5V2Env.directions[k])
        for k in E5V2Env.panel_kpi_ids
    ]


def true_edges(env: E5V2Env | None = None):
    """(param_edges, kpi_edges) of the FULL composed structure."""
    env = env or E5V2Env()
    param_edges, kpi_edges = set(), set()
    for kpi, src in env.adjacency_edges:
        if src < env.num_params:
            param_edges.add((kpi, src))
        else:
            kpi_edges.add((kpi, src - env.num_params))
    return frozenset(param_edges), frozenset(kpi_edges)


def committed_state(seed: int, subdom=0.20, chain_gamma=0.20):
    env = E5V2Env(env_seed=seed, theta=0.0, subdom=subdom, chain_gamma=chain_gamma)
    env.reset(episode=0)
    for _ in range(NEUTRAL_STEPS):
        env.neutral_step()
    return env.snapshot()


def is_in_gate(snap) -> bool:
    g1, g2 = float(snap[1][E5V2Env.G1]), float(snap[1][E5V2Env.G2])
    return _gate(g1, g2, E5V2Env.TAU1, E5V2Env.C2, E5V2Env.W_GATE)


def spine_regret_e5(param_edges, kpi_edges, states, panel, *, subdom=0.20, chain_gamma=0.20, H=1):
    """Per-state metrics of a masked-structure planner vs the true (Z-out) oracle. Layer config
    (subdom, chain_gamma) is applied to BOTH the true env and the world model so they differ ONLY by
    the mask (sol #4). ``H`` is the rollout horizon: H=1 scores the DIRECT edge (frozen path); H>=2 also
    scores the 2-hop chain P0->K_mid->K_harm (sol #4 — a mediated hop needs one more advance). Returns a
    dict with normalized + RAW regret, D(s), action mismatches, and the planner's K_harm
    constraint-violation rate under truth (sol asks for raw + violation metrics)."""
    norm, raw, ds, mism, viol = [], [], [], 0, 0
    for seed, snap in states:
        true_env = E5V2Env(env_seed=seed, theta=0.0, subdom=subdom, chain_gamma=chain_gamma)
        wm = MaskedE5WorldModel(param_edges, kpi_edges, env_seed=seed,
                                subdom=subdom, chain_gamma=chain_gamma)
        s_true = score_grid_h(true_env, snap, P0, panel, H)
        s_wm = score_grid_h(wm, snap, P0, panel, H)
        g_star = float(s_true.max())
        d = g_star - float(s_true.min())
        a_wm = int(np.argmax(s_wm))
        r_raw = g_star - float(s_true[a_wm])
        raw.append(r_raw)
        ds.append(d)
        norm.append(r_raw / d if d > 1e-9 else 0.0)
        mism += int(a_wm != int(np.argmax(s_true)))
        # K_harm (satisfy-below) violation at the planner's chosen action, under truth (horizon H):
        te2 = E5V2Env(env_seed=seed, theta=0.0, subdom=subdom, chain_gamma=chain_gamma)
        te2.restore(snap)
        k = _rollout_terminal(te2, P0, float(P0_GRID[a_wm]), H)
        viol += int(k[E5V2Env.K_HARM] > E5V2Env.kpi_thresholds[E5V2Env.K_HARM])
    n = len(states)
    return {"mean_norm": float(np.mean(norm)), "mean_raw": float(np.mean(raw)),
            "mean_Ds": float(np.mean(ds)), "action_mismatches": mism, "n": n,
            "harm_violations": viol, "per_state_norm": norm}


def build_ingate_bank(n=40, pool=6000, subdom=0.20, chain_gamma=0.20):
    states = []
    for seed in range(pool):
        snap = committed_state(seed, subdom=subdom, chain_gamma=chain_gamma)
        if is_in_gate(snap):
            states.append((seed, snap))
        if len(states) >= n:
            break
    return states


def main() -> int:
    # CORE layer sanity: oracle vs single-edge ablation (NOT a method comparison — see e5_baselines).
    subdom, chain = 0.20, 0.0
    panel = e5_panel()
    bank = build_ingate_bank(subdom=subdom, chain_gamma=chain)
    p_true, k_true = true_edges(E5V2Env(subdom=subdom, chain_gamma=chain))
    print(f"E5 spine CORE sanity (H=1): {len(bank)} in-gate states; harmful edge {HARMFUL_EDGE}")
    o = spine_regret_e5(p_true, k_true, bank, panel, subdom=subdom, chain_gamma=chain)
    m = spine_regret_e5(p_true - {HARMFUL_EDGE}, k_true, bank, panel, subdom=subdom, chain_gamma=chain)
    print(f"  oracle graph          : norm={o['mean_norm']:.4f} raw={o['mean_raw']:.3f} "
          f"mism={o['action_mismatches']} harm_viol={o['harm_violations']}/{o['n']}")
    print(f"  oracle minus P0->K_harm: norm={m['mean_norm']:.4f} raw={m['mean_raw']:.3f} "
          f"mism={m['action_mismatches']} harm_viol={m['harm_violations']}/{m['n']} D(s)~{m['mean_Ds']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

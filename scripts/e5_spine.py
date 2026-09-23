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

from cdd_oran.analysis.v2_regret import PanelXApp, score_grid  # noqa: E402
from cdd_oran.benchmark.masked_world_model import MaskedE5WorldModel  # noqa: E402
from cdd_oran.envs.v2.e5 import E5V2Env, _gate  # noqa: E402

P0 = E5V2Env.P0
HARMFUL_EDGE = (E5V2Env.K_HARM, E5V2Env.P0)  # (1, 0): the subdominant gated direct edge
NEUTRAL_STEPS = 3


def e5_panel() -> list[PanelXApp]:
    """Panel {K_ben above, K_harm below, K_dist above} in raw units (standardization pending)."""
    return [
        PanelXApp((k,), 0.0, 1.0, E5V2Env.kpi_thresholds[k], E5V2Env.directions[k])
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


def committed_state(seed: int):
    env = E5V2Env(env_seed=seed, theta=0.0)
    env.reset(episode=0)
    for _ in range(NEUTRAL_STEPS):
        env.neutral_step()
    return env.snapshot()


def is_in_gate(snap) -> bool:
    g1, g2 = float(snap[1][E5V2Env.G1]), float(snap[1][E5V2Env.G2])
    return _gate(g1, g2, E5V2Env.TAU1, E5V2Env.C2, E5V2Env.W_GATE)


def spine_regret_e5(param_edges, kpi_edges, states, panel):
    """Per-state normalized regret of a masked-structure planner vs the true (Z-out) oracle."""
    gaps, mism = [], 0
    for seed, snap in states:
        true_env = E5V2Env(env_seed=seed, theta=0.0)  # Z integrated out for scoring
        wm = MaskedE5WorldModel(param_edges, kpi_edges, env_seed=seed)
        s_true = score_grid(true_env, snap, P0, panel)
        s_wm = score_grid(wm, snap, P0, panel)
        g_star = float(s_true.max())
        d = g_star - float(s_true.min())
        a_wm = int(np.argmax(s_wm))
        gaps.append((g_star - float(s_true[a_wm])) / d if d > 1e-9 else 0.0)
        mism += int(a_wm != int(np.argmax(s_true)))
    return gaps, mism


def build_ingate_bank(n=40, pool=4000):
    states = []
    for seed in range(pool):
        snap = committed_state(seed)
        if is_in_gate(snap):
            states.append((seed, snap))
        if len(states) >= n:
            break
    return states


def main() -> int:
    panel = e5_panel()
    bank = build_ingate_bank()
    p_true, k_true = true_edges()
    print(f"E5 spine (CORE layer, H=1): {len(bank)} in-gate states; harmful edge {HARMFUL_EDGE}")

    o_gaps, o_m = spine_regret_e5(p_true, k_true, bank, panel)
    miss = (p_true - {HARMFUL_EDGE}, k_true)
    m_gaps, m_m = spine_regret_e5(miss[0], miss[1], bank, panel)
    print(f"  oracle (all true edges)   : mean_pos={np.mean(o_gaps):.4f}  mism={o_m}")
    print(f"  missed harmful edge (1,0) : mean_pos={np.mean(m_gaps):.4f}  mism={m_m}  "
          f"(frac>0 {np.mean(np.array(m_gaps) > 1e-6):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

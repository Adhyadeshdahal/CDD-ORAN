"""Synthetic api.Dataset / api.Truth for the xm-classic tests and fidelity gate F4 (used until xm-harness lands).

NOT a study world. A small time-series SCM with known planted edges, exact nulls and the ``P_placebo`` column:

    actions A0, A1, A2 (+ P_placebo), drawn i.i.d. U(0, 1) (R1, kind "iid") or as a bounded slow setpoint plus a
    U(-.25, .25) dither (R2-like, kind "dither");  K_t = KPIs of the previous row (lagged state, a real time series)
    Y0 = b * A0 + 0.5 b * K0_t + e        Y1 = -b * A1 + e        Y2 = 0.7 b * K1_t + e        e ~ N(0, s^2)
    (``nonlinear=True``: A0 enters as b * sin(2 pi A0) and A1 as -b * (A1 - .5)^2 * 4, a non-monotone bump.)

Planted edges: A0->Y0 (+), A1->Y1 (-), K0->Y0 (+), K1->Y2 (+). Exact nulls: A2->*, P_placebo->*, K2->* and every
other pair. ``b = 0`` gives the global null (every candidate exactly null).
"""
from __future__ import annotations

import numpy as np

from cdd_oran.xmethod import api

ACTIONS = ("A0", "A1", "A2", "P_placebo")
KPIS = ("Y0", "Y1", "Y2")
PLANTED = {("A0", "Y0"): +1, ("A1", "Y1"): -1, ("Y0", "Y0"): +1, ("Y1", "Y2"): +1}   # lag sources named as KPIs


def make(n: int, seed: int, b: float = 0.5, noise: float = 1.0, regime: str = "R1", nonlinear: bool = False,
         burn: int = 50, p_extra: int = 0, k_extra: int = 0) -> tuple[api.Dataset, api.Truth]:
    """``p_extra`` / ``k_extra`` add null actions A3.. (same design) and i.i.d. N(0, 1) null KPIs Y3.. (cost runs at
    world-like sizes; every candidate touching them is an exact null)."""
    r = np.random.default_rng([9_100, int(seed), int(n), int(b * 1000), int(regime == "R2"), int(nonlinear),
                               int(p_extra), int(k_extra)])
    T = n + burn
    actions = ACTIONS[:3] + tuple(f"A{3 + i}" for i in range(p_extra)) + ACTIONS[3:]
    kpis = KPIS + tuple(f"Y{3 + i}" for i in range(k_extra))
    p = len(actions)
    if regime == "R1":
        A = r.uniform(0.0, 1.0, (T, p))
        fixed = rand = None
    elif regime == "R2":
        sp = np.empty((T, p))
        sp[0] = 0.5
        for t in range(1, T):
            sp[t] = np.clip(sp[t - 1] + r.normal(0, 0.03, p), 0.25, 0.75)
        d = r.uniform(-0.25, 0.25, (T, p))
        A = sp + d
        fixed, rand = sp, d
    else:
        raise ValueError(regime)
    K = np.zeros((T + 1, len(kpis)))
    for t in range(T):
        a0, a1 = A[t, 0], A[t, 1]   # P_placebo is the last column, A2 / A3.. never enter
        if nonlinear:
            f0, f1 = np.sin(2 * np.pi * a0), -4.0 * (a1 - 0.5) ** 2
        else:
            f0, f1 = a0, -a1
        e = r.normal(0, noise, 3)
        K[t + 1, 0] = b * f0 + 0.5 * b * K[t, 0] + e[0]
        K[t + 1, 1] = b * f1 + e[1]
        K[t + 1, 2] = 0.7 * b * K[t, 1] + e[2]
        if k_extra:
            K[t + 1, 3:] = r.normal(0, 1, k_extra)
    sl = slice(burn, T)
    designs = []
    for j in range(p):
        if regime == "R1":
            designs.append(api.Design(kind="iid", dist={"name": "uniform", "lo": 0.0, "hi": 1.0}))
        else:
            designs.append(api.Design(kind="dither", dist={"name": "uniform", "lo": -0.25, "hi": 0.25},
                                      random_part=rand[sl, j].copy(), fixed_part=fixed[sl, j].copy()))
    cands = tuple((s, t) for s in actions + kpis for t in kpis)
    data = api.Dataset(world="SYN", regime=regime, n=n, seed=int(seed), action_names=actions, kpi_names=kpis,
                       X_action=A[sl].copy(), X_kpi_lag=K[sl].copy(), Y=K[burn + 1:T + 1].copy(),
                       designs=tuple(designs), candidates=cands, time_index=np.arange(n), context=None,
                       meta={"synthetic": True, "b": b, "nonlinear": nonlinear})
    planted = PLANTED if b != 0 else {}
    signs = {e: v for e, v in planted.items() if not (nonlinear and e[0].startswith("A"))}   # bumps: no sign
    truth = api.Truth(world="SYN", regime=regime, edges=frozenset(planted), signs=signs,
                      null_edges=frozenset(c for c in cands if c not in planted))
    return data, truth

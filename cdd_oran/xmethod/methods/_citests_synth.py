"""Synthetic api.Dataset objects for the citests fidelity gates (F4) and tests, until xm-harness lands.

Layout mirrors the harness: actions P0..P3 + P_placebo (i.i.d. uniform[-1, 1]), KPIs K0..K3 with lagged KPIs
(K at t, an AR(1) chain over rows) and targets Y = K at t+1. Candidates: every action x KPI, then every lagged KPI
x KPI (``generate.candidates_for`` order).

``planted``: K0 <- +0.6 P0 (linear), K1 <- -0.6 P1 (linear, negative), K2 <- 0.8 * P2 * 1{P3 > 0.4} (gated by a
co-parent; P3's own effect is also gated), K3 <- 0.5 * lagged K3 (KPI -> KPI), all + N(0, 1) noise.
``null``: no action effect at all; K_j(t+1) = 0.5 K_j(t) + noise (only the KPI -> KPI self edges are true).
Signs: +1 for P0->K0, P2->K2, P3->K2, K3->K3 (planted); -1 for P1->K1.
"""
from __future__ import annotations

import numpy as np

from cdd_oran.xmethod import api

ACTIONS = ("P0", "P1", "P2", "P3", "P_placebo")
KPIS = ("K0", "K1", "K2", "K3")


def candidates() -> tuple[tuple[str, str], ...]:
    return tuple((a, k) for a in ACTIONS for k in KPIS) + tuple((s, k) for s in KPIS for k in KPIS)


def make(kind: str, n: int, seed: int, world: str = "SYN") -> tuple[api.Dataset, api.Truth]:
    rng = np.random.default_rng([7802, 99, int(seed), n, 0 if kind == "planted" else 1])
    A = rng.uniform(-1.0, 1.0, size=(n, len(ACTIONS)))
    lag = np.zeros((n, len(KPIS)))
    Y = np.zeros((n, len(KPIS)))
    k_prev = rng.normal(size=len(KPIS))
    for t in range(n):
        lag[t] = k_prev
        eps = rng.normal(size=len(KPIS))
        if kind == "planted":
            y = np.array([0.6 * A[t, 0], -0.6 * A[t, 1], 0.8 * A[t, 2] * (A[t, 3] > 0.4), 0.5 * k_prev[3]]) + eps
        elif kind == "null":
            y = 0.5 * k_prev + eps
        else:
            raise ValueError(kind)
        Y[t] = y
        k_prev = y
    if kind == "planted":
        edges = {("P0", "K0"): 1, ("P1", "K1"): -1, ("P2", "K2"): 1, ("P3", "K2"): 1, ("K3", "K3"): 1}
    else:
        edges = {(k, k): 1 for k in KPIS}
    cands = candidates()
    designs = tuple(api.Design(kind="iid", dist={"name": "uniform", "lo": -1.0, "hi": 1.0}) for _ in ACTIONS)
    ds = api.Dataset(world=world, regime="R1", n=n, seed=int(seed), action_names=ACTIONS, kpi_names=KPIS,
                     X_action=A, X_kpi_lag=lag, Y=Y, designs=designs, candidates=cands,
                     time_index=np.arange(n), meta={"synthetic": kind})
    truth = api.Truth(world=world, regime="R1", edges=frozenset(edges), signs=dict(edges),
                      null_edges=frozenset(c for c in cands if c not in edges))
    return ds, truth

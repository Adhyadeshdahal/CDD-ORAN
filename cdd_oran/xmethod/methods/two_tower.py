"""``two_tower``: two-tower interaction learning (arXiv:2601.13213) via the repo's E2 port (xm-classic).

LABEL: an ADAPTATION (self-supervised E2 port), NOT the published supervised model. The paper trains its towers with
BCE against the ground-truth label matrix (sec VI-A, eq. 14), which a truth-free discovery method cannot use;
``notes['label']`` records this on every Result.

Wraps ``scripts/e2_baseline_gnn.fit_two_tower`` (unchanged) with its ``TwoTower`` re-sized by a temporary rebinding
(same wrapper as ``cdd_oran/decision/baselines_disc._two_tower_sized``): param tower = the action columns at t (incl.
``P_placebo``), KPI tower = the KPIs at t+1; inputs z-scored inside the script; learned non-negative interaction gate
a[KPI, action] = softplus(<U_k, V_p>).
Score (F7 audit fix): the ROW SHARE a[k, p] / sum_p' a[k, p'] over the action columns of KPI k. Each per-KPI head can
absorb a rescaling of its gate row, so raw gates are not comparable across KPIs, while the placebo tau is one cut
pooled over all KPIs; the share is invariant to that rescaling. Not the row max (a / max_p' a): it forces the top
entry of EVERY row to 1, also in rows with no true action parent, so the placebo is the row max with probability
~1 / p per such row and dataset and the 2nd-largest DEV placebo score (tau) saturates at 1 (nothing declarable). Not
sparsemax (the paper's row normalisation): it is not invariant to the row scale. Raw gates are kept in
``notes['gate']``.
Other adaptations (fidelity table): self-supervised MSE, scalar encoders; actions only (the port's "params-only
reconstruction"), so KPI -> KPI candidates are not scorable (NaN); ``native`` = the port's relative rule
a >= .10 * max_p a per KPI (unchanged by the share); sign: the gate is unsigned -> orchestrator rule ``pcorr_given_Z``.
Sizes d 16, r 8, hidden 32, lr .01, epochs 500 (the script's E2 defaults); l1 fixed a priori on SYNTHETIC data only
(F4 in the hand-back; value in ``defaults``).
R-37: the primary fit has no ``P_placebo_conf`` input (E4 R3 / R4 diagnostic); its own candidates are read from a
second fit with every action (``notes['diagnostic_fit']``, with its own gate).
"""
from __future__ import annotations

import contextlib
import functools
from typing import Any

import numpy as np

import scripts.e2_baseline_gnn as G
from cdd_oran.xmethod import api

from ._classic_common import (
    PLACEBO_CONF,
    ClassicBase,
    Scored,
    columns,
    int_seed,
    resolve,
    sign_pcorr_given_Z,
    target_index,
)


@contextlib.contextmanager
def _sized(mod, **sizes):
    orig = mod.TwoTower
    mod.TwoTower = functools.partial(orig, **sizes)
    try:
        yield
    finally:
        mod.TwoTower = orig


LABEL = "adaptation (self-supervised E2 port), not the published supervised model"


class TwoTowerM(ClassicBase):
    name = "two_tower"
    version = "scripts/e2_baseline_gnn.fit_two_tower (E2 port), row-share score"
    uses_p = False
    method_idx = 4
    defaults: dict[str, Any] = {"d": 16, "r": 8, "hidden": 32, "epochs": 500, "lr": 0.01, "l1": 1e-3,
                                "tau_rel_native": 0.10}

    def _score(self, data: api.Dataset, config: dict[str, Any]):
        cols = columns(data)
        Y = cols.Y
        ok_y = [k for k in range(Y.shape[1]) if np.std(Y[:, k]) > 0]
        seed = int_seed(data, self.method_idx) if config.get("seed") is None else int(config["seed"])

        def fit(with_conf: bool):
            """R-37: the primary fit has no ``P_placebo_conf`` input; the diagnostic fit (``with_conf``) has every
            action and is read only for its own candidates. Returns (action columns used, S, pseudo-R2)."""
            use = [j for j, a in enumerate(data.action_names) if with_conf or a != PLACEBO_CONF]
            X = cols.actions[:, use]
            with _sized(G, num_params=X.shape[1], num_kpis=len(ok_y), d=config["d"], r=config["r"],
                        hidden=config["hidden"]):
                S, r2 = G.fit_two_tower(X, Y[:, ok_y], epochs=config["epochs"], lr=config["lr"], l1=config["l1"],
                                        seed=seed)
            return use, S, r2

        use, S, r2 = fit(False)
        diag = fit(True) if any(s == PLACEBO_CONF for s, _ in data.candidates) else None
        out = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            u, Sx = (diag[0], diag[1]) if s == PLACEBO_CONF else (use, S)
            if fam != "action" or ti not in ok_y or j not in u:
                out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                continue
            row = Sx[ok_y.index(ti)]
            v = float(row[u.index(j)])
            tot = float(row.sum())
            native = bool(row.max() > 0 and v >= config["tau_rel_native"] * float(row.max()))
            share = v / tot if tot > 0 else float("nan")
            out.append(Scored(s, t, fam, share, sign_pcorr_given_Z(data, cols, fam, j, ti), None, native))
        gate = {data.kpi_names[k]: {data.action_names[a]: float(S[i, p]) for p, a in enumerate(use)}
                for i, k in enumerate(ok_y)}
        notes = {"sign_rule": "pcorr_given_Z", "pseudo_r2": float(r2), "seed": seed, "label": LABEL,
                 "score": "row share a[k, p] / sum_p' a[k, p'] (F7 fix; raw gate in notes['gate'])", "gate": gate}
        if diag is not None:
            notes["diagnostic_fit"] = {"rule": f"R-37: {PLACEBO_CONF} candidates read from a second fit with every "
                                               "action; primary fit without it", "pseudo_r2": float(diag[2]),
                                       "gate": {data.kpi_names[k]: {data.action_names[a]: float(diag[1][i, p])
                                                                    for p, a in enumerate(diag[0])}
                                                for i, k in enumerate(ok_y)}}
        return out, notes

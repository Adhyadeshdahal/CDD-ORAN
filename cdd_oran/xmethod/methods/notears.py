"""``notears``: linear NOTEARS (Zheng, Aragam, Ravikumar, Xing, NeurIPS 2018), authors' code (xm-classic).

Reference code: ``_vendor/notears/linear.py`` (xunzheng/notears @ 4a9ab19, Apache-2.0, byte-identical). Authors'
defaults: loss 'l2', lambda1 = 0.1, max_iter = 100, h_tol = 1e-8, rho_max = 1e16, w_threshold = 0.3.

Adaptations (all recorded in the fidelity table):
  1. Variables = actions (incl. ``P_placebo``) + non-NaN lagged KPIs at t + context (R3) + KPIs at t+1, each
     column STANDARDISED (z-score). The authors' code only centres; on these worlds the columns have arbitrary units,
     which would make the fixed lambda1 / w_threshold scale-dependent (and NOTEARS is known to exploit
     var-sortability, Reisach et al. 2021). Standardising is the common practice in comparisons.
  2. Background knowledge (same as pc): the L-BFGS-B bounds of forbidden entries W[i, j] are set to (0, 0), the
     authors' own mechanism for the diagonal. ``notears_linear_bk`` is the vendored function with ONLY the bounds line
     changed (tests: equal to the vendored function when nothing but the diagonal is forbidden).
     ``config['background_knowledge'] = False`` runs the unconstrained authors' problem.
  3. Score = |W[source, target]| of the UNTHRESHOLDED estimate (w_threshold applied afterwards for ``native``), sign =
     sign(W[source, target]) (orchestrator: NOTEARS uses its signed weight).
  4. R-37: the primary fit has no ``P_placebo_conf`` variable (E4 R3 / R4 diagnostic), so it cannot change any other
     candidate's weight; its own candidates are read from a second fit with every variable (``notes['diagnostic_fit']``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

from cdd_oran.xmethod import api

from ._classic_common import (
    PLACEBO_CONF,
    ClassicBase,
    Scored,
    columns,
    resolve,
    target_index,
    zscore,
)

VENDOR_COMMIT = "xunzheng/notears@4a9ab19fe502e392503da331773c5223d82d3666"


def notears_linear_bk(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3,
                      forbid=None):
    """Vendored ``notears_linear`` (Apache-2.0) with one change: ``forbid`` ([d, d] bool, True = W[i, j] fixed at 0)
    extends the authors' diagonal (0, 0) bounds. Everything else is verbatim."""
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    fb = np.eye(d, dtype=bool) if forbid is None else (np.asarray(forbid, dtype=bool) | np.eye(d, dtype=bool))
    bnds = [(0, 0) if fb[i, j] else (0, None) for _ in range(2) for i in range(d) for j in range(d)]   # CHANGED
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


class Notears(ClassicBase):
    name = "notears"
    version = f"linear, {VENDOR_COMMIT}"
    uses_p = False
    method_idx = 2
    defaults: dict[str, Any] = {"lambda1": 0.1, "loss_type": "l2", "max_iter": 100, "h_tol": 1e-8,
                                "rho_max": 1e16, "w_threshold": 0.3, "standardize": True,
                                "background_knowledge": True}

    def _fit(self, data: api.Dataset, cols, config: dict[str, Any], with_conf: bool) -> dict:
        """One NOTEARS fit. ``with_conf`` False (R-37, primary fit): no ``P_placebo_conf`` variable; True: every
        variable (read only for ``P_placebo_conf``'s own candidates)."""
        blocks = [("A", data.action_names, cols.actions), ("L", cols.lag_names, cols.lags),
                  ("C", tuple(f"c{i}" for i in range(cols.context.shape[1])), cols.context),
                  ("Y", data.kpi_names, cols.Y)]
        names, mats, kinds = [], [], []
        for kind, nm, M in blocks:
            for i, v in enumerate(nm):
                if kind == "A" and v == PLACEBO_CONF and not with_conf:
                    continue
                names.append(f"{kind}:{v}")
                mats.append(M[:, i])
                kinds.append(kind)
        D = np.column_stack(mats)
        keep = [i for i in range(D.shape[1]) if np.std(D[:, i]) > 0]
        D, kk = D[:, keep], [kinds[i] for i in keep]
        idx = {names[i]: k for k, i in enumerate(keep)}
        if config["standardize"]:
            D = zscore(D)
        d = D.shape[1]
        forbid = np.zeros((d, d), dtype=bool)
        if config["background_knowledge"]:
            for i, ki in enumerate(kk):
                for j, kj in enumerate(kk):
                    if ki == "Y" and kj != "Y":                  # t+1 -> t
                        forbid[i, j] = True
                    if kj == "A" and ki != "C":                  # into an action (context allowed, R3)
                        forbid[i, j] = True
                    if ki == "A" and kj in ("L", "C"):           # action -> state it was set in
                        forbid[i, j] = True
        W = notears_linear_bk(D, config["lambda1"], config["loss_type"], max_iter=config["max_iter"],
                              h_tol=config["h_tol"], rho_max=config["rho_max"], w_threshold=0.0, forbid=forbid)
        return {"W": W, "idx": idx, "d": d}

    def _score(self, data: api.Dataset, config: dict[str, Any]):
        cols = columns(data)
        main = self._fit(data, cols, config, with_conf=False)
        diag = (self._fit(data, cols, config, with_conf=True)
                if any(s == PLACEBO_CONF for s, _ in data.candidates) else None)
        out = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            g = diag if s == PLACEBO_CONF else main
            sname = None if j is None else (f"A:{data.action_names[j]}" if fam == "action" else f"L:{cols.lag_names[j]}")
            tname = f"Y:{data.kpi_names[ti]}"
            if sname not in g["idx"] or tname not in g["idx"]:
                out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                continue
            w = float(g["W"][g["idx"][sname], g["idx"][tname]])
            out.append(Scored(s, t, fam, abs(w), int(np.sign(w)), None, abs(w) >= config["w_threshold"]))
        W, d = main["W"], main["d"]
        h = float(np.trace(slin.expm(W * W)) - d)
        notes = {"sign_rule": "sign(W)", "h_final": h, "n_vars": d, "nnz_native": int((np.abs(W) >=
                 config["w_threshold"]).sum()), "score": "|W| unthresholded"}
        if diag is not None:
            notes["diagnostic_fit"] = {"rule": f"R-37: {PLACEBO_CONF} candidates read from a second fit with every "
                                               "variable; primary fit without it", "n_vars": diag["d"]}
        return out, notes

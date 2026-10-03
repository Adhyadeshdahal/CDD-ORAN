"""``shap_dag``: SHAP-DAG (Sharma et al., arXiv:2510.13031, NFV-SDN 2025), discovery stage (xm-classic).

One gradient-boosted regressor per KPI at t+1 on the RCPs (= the action columns at t, incl. ``P_placebo``),
explained with shap.TreeExplainer; mean|SHAP| per action per KPI.
  - Regressor (``config['regressor']``): "xgboost" (default; the paper's choice, sec V, Table II) =
    ``xgboost.XGBRegressor`` at the package defaults (the paper states no hyperparameters; R-13), seeded, no early
    stopping (XGBoost's default). "hgbdt" = the earlier E2 port's sklearn HistGradientBoostingRegressor(max_iter 300,
    lr .1) as a SENSITIVITY option, with ``early_stopping=False`` (sklearn's 'auto' switches it on at n > 10 000, which
    would change the estimator along the n grid; at n <= 10 000 identical to
    ``scripts/e2_baseline_shap_dag.fit_shap_importances``).
  - The paper regresses KPIs on RCPs ONLY, so lagged-KPI (KPI -> KPI) candidates are not scorable (NaN) by default;
    ``config['include_lag'] = True`` adds the lagged KPIs as features [F, not the paper].
  - score = mean|SHAP_source| / sd(target) (absolute, scale-free; the E6 adaptation in
    ``cdd_oran/decision/baselines_disc.score_shap``). The paper's rule is qualitative ("the most influential" RCPs, sec IV-C1;
    no numeric threshold, so every threshold here is ours); the E2 port
    keeps p iff mean|SHAP_p| >= tau_rel * max_p' (tau_rel = .10); that rule always declares >= 1 parent for EVERY
    KPI, null KPIs included, so the contract's placebo tau is applied to the absolute score. The E2 relative rule at
    tau_rel = .10 is reported as ``native`` (notes['native_declared']).
  - sign: SHAP importances are unsigned -> orchestrator rule ``pcorr_given_Z`` (the paper's DoWhy ATE is a linear
    backdoor coefficient on an adjustment set, so its sign is the partial-correlation sign on that set).
  - R-37: the primary regressors have no ``P_placebo_conf`` feature (E4 R3 / R4 diagnostic); its own candidates are
    read from a second set of regressors with every feature (``notes['diagnostic_fit']``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import shap
import sklearn

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


def make_regressor(regressor: str, seed: int, threads: int = 1):
    """The unfitted regressor of ``config['regressor']`` (module doc); never early-stopped."""
    if regressor == "xgboost":
        import xgboost
        return xgboost.XGBRegressor(random_state=seed, n_jobs=threads)
    if regressor == "hgbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=seed, early_stopping=False)
    raise ValueError(f"unknown regressor {regressor!r}")


def fit_shap(X: np.ndarray, y: np.ndarray, seed: int, regressor: str = "xgboost", threads: int = 1):
    """Fit one regressor (module doc) and return (mean|SHAP| per feature, training R^2)."""
    model = make_regressor(regressor, seed, threads)
    model.fit(X, y)
    r2 = float(model.score(X, y))
    sv = shap.TreeExplainer(model).shap_values(X)
    return np.abs(sv).mean(axis=0), r2


def _xgb_version() -> str:
    try:
        import xgboost
        return xgboost.__version__
    except ImportError:
        return "absent"


class ShapDag(ClassicBase):
    name = "shap_dag"
    version = "XGBRegressor (package defaults) + shap.TreeExplainer; hgbdt sensitivity option"
    uses_p = False
    method_idx = 3
    defaults: dict[str, Any] = {"include_lag": False, "tau_rel_native": 0.10, "regressor": "xgboost"}

    def _score(self, data: api.Dataset, config: dict[str, Any]):
        def fit(X, y, seed):
            return fit_shap(X, y, seed, config["regressor"], int(config.get("threads", 1)))
        cols = columns(data)
        conf = data.action_names.index(PLACEBO_CONF) if PLACEBO_CONF in data.action_names else None
        seed = int_seed(data, self.method_idx) if config.get("seed") is None else int(config["seed"])
        targets = sorted({target_index(data, t) for _, t in data.candidates})

        def fit_all(with_conf: bool):
            """One regressor per KPI. R-37: the primary fit has no ``P_placebo_conf`` feature; the diagnostic fit
            (``with_conf``) has every feature and is read only for its own candidates."""
            feats = [("action", j) for j in range(cols.actions.shape[1]) if with_conf or j != conf]
            if config["include_lag"]:
                feats += [("kpi", j) for j in range(cols.lags.shape[1])]
            X = np.column_stack([cols.actions[:, j] if f == "action" else cols.lags[:, j] for f, j in feats])
            imps, r2s = {}, {}
            for ti in targets:
                y = cols.Y[:, ti]
                if np.std(y) == 0:
                    continue
                imp, r2 = fit(X, y, seed=seed)
                imps[ti] = (np.asarray(imp, float), float(np.std(y)))
                r2s[data.kpi_names[ti]] = float(r2)
            return feats, imps, r2s

        feats, imps, r2s = fit_all(False)
        diag = fit_all(True) if any(s == PLACEBO_CONF for s, _ in data.candidates) else None
        out = []
        for s, t in data.candidates:
            fam, j = resolve(data, cols, s)
            ti = target_index(data, t)
            fe, im = (diag[0], diag[1]) if s == PLACEBO_CONF else (feats, imps)
            if j is None or (fam, j) not in fe or ti not in im:
                out.append(Scored(s, t, fam, float("nan"), 0, None, None))
                continue
            imp, sd = im[ti]
            v = float(imp[fe.index((fam, j))])
            native = bool(imp.max() > 0 and v >= config["tau_rel_native"] * float(imp.max()))
            out.append(Scored(s, t, fam, v / sd, sign_pcorr_given_Z(data, cols, fam, j, ti), None, native))
        notes = {"sign_rule": "pcorr_given_Z", "train_r2": r2s, "seed": seed,
                 "regressor": config["regressor"],
                 "packages": {"shap": shap.__version__, "sklearn": sklearn.__version__, "xgboost": _xgb_version()},
                 "score": "mean|SHAP| / sd(target)",
                 "native_rule": f"E2 relative rule, tau_rel = {config['tau_rel_native']}"}
        if diag is not None:
            notes["diagnostic_fit"] = {"rule": f"R-37: {PLACEBO_CONF} candidates read from a second fit with every "
                                               "feature; primary fit without it", "train_r2": diag[2]}
        return out, notes

"""audit-classic2, checks 2 / 4 / 5: arms (R-17 / R-18 / R-25), signs (R-4), declarations (R-2), two_tower row share,
shap_dag XGBoost defaults + SHAP additivity, Dataset-only inputs. Independent recomputation (statsmodels / numpy),
DEV seed 3_000_000, kappa .25 (R-27). Run: uv run --group baselines python scratchpad/xmethod/audit/classic2/arms_check.py
"""
from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from cdd_oran.xmethod.covariates import design_covariates
from cdd_oran.xmethod.methods import _classic_common as cc
from cdd_oran.xmethod.methods.classic import METHODS
from cdd_oran.xmethod.worlds.generate import generate_dataset

SEED, KAP, N = 3_000_000, 0.25, 1000
out: dict = {}


def ds(w, r, lam=1.0, n=N, kap=KAP):
    return generate_dataset(w, r, n, SEED, lam=lam, kappa=kap)[0]


def ols_p(y, X):
    f = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    return float(f.pvalues[-1]), float(f.params[-1])


# ---------------------------------------------------------------- 2a. cond_set == helper, both arms, every action
res = {}
for w, r in (("E2", "R2"), ("E3", "R2"), ("E4", "R3"), ("E4", "R4"), ("E1", "R1")):
    d = ds(w, r)
    bad = 0
    for j, a in enumerate(d.action_names):
        cs = cc.cond_set(d, "action", j, "eq")
        nm, M, mk = design_covariates(d, focal=a)                       # concurrent='designed' default
        bad += not (cs.names == nm and np.array_equal(cs.Z, M) and np.array_equal(cs.mask, mk))
        cs = cc.cond_set(d, "action", j, "native")
        nm, M, mk = design_covariates(d, False, False, focal=a, concurrent="all")
        bad += not (cs.names == nm and np.array_equal(cs.Z, M))
    # lagged-KPI source: base set w/o own lag + all actions at t
    k0 = cc.cond_set(d, "kpi", 0, "eq").names
    res[f"{w}{r}"] = {"mismatches": bad, "eq_P0": list(cc.cond_set(d, "action", 0, "eq").names),
                      "native_P0": list(cc.cond_set(d, "action", 0, "native").names),
                      "eq_lagK0_has_own": f"lag_kpi:{d.kpi_names[0]}" in k0,
                      "eq_lagK0_tail": list(k0[-len(d.action_names):])}
out["cond_set"] = res

# ---------------------------------------------------------------- 2b. granger eq == statsmodels OLS t-test given Z_eq
G = METHODS["granger"]()
gr = {}
for w, r in (("E3", "R2"), ("E2", "R2"), ("E1", "R2")):
    d = ds(w, r)
    cols = cc.columns(d)
    res_eq = G.run(d, {"arm": "eq"})
    res_nat = G.run(d, {"arm": "native"})
    maxrel, sign_bad, n = 0.0, 0, 0
    for e in res_eq.edges:
        if e.p is None:
            continue
        fam, j = cc.resolve(d, cols, e.source)
        x, src = cc.source_column(d, cols, fam, j)
        cs = cc.cond_set(d, fam, src, "eq")
        m = cs.mask
        p, b = ols_p(cols.Y[m, d.kpi_names.index(e.target)], np.column_stack([cs.Z[m], x[m]]))
        maxrel = max(maxrel, abs(p - e.p) / max(p, 1e-300))
        sign_bad += int(np.sign(b)) != e.sign
        n += 1
    # declarations vs independent BY (statsmodels fdr_by) per family
    dec_bad = 0
    for res_ in (res_eq, res_nat):
        fam = res_.notes["family"]
        for f in ("action", "kpi", "diagnostic"):
            es = [e for e in res_.edges if fam[f"{e.source}->{e.target}"] == f]
            ps = [e.p for e in es if e.p is not None]
            if not ps:
                continue
            rej = iter(multipletests(ps, 0.05, "fdr_by")[0])
            dec_bad += sum((next(rej) if e.p is not None else False) != e.declared for e in es)
    gr[f"{w}{r}"] = {"n_tested": n, "max_rel_p_diff_vs_statsmodels": maxrel, "sign_mismatch": sign_bad,
                     "BY_decl_mismatch": dec_bad, "n_not_testable_eq": res_eq.notes.get("n_not_testable", 0),
                     "n_rows_used": res_eq.notes.get("n_rows_used")}
out["granger_eq_vs_statsmodels"] = gr

# ---------------------------------------------------------------- 2c. pc eq node set and background knowledge
P = METHODS["pc"]()
pcr = {}
for w, r in (("E2", "R2"), ("E4", "R3"), ("E4", "R4")):
    d = ds(w, r)
    re_ = P.run(d, {"arm": "eq"})
    base = design_covariates(d)[0]
    # nodes into X forbidden: rebuild the node list like the adapter and inspect the final graph orientation
    pcr[f"{w}{r}"] = {"eq_X_nodes": re_.notes["eq_covariates"], "eq_dropped": re_.notes["eq_dropped"],
                      "n_rows_used": re_.notes["n_rows_used"], "n_vars": re_.notes["n_vars"],
                      "base_helper_names": list(base),
                      "designs": [x.kind for x in d.designs],
                      "Zeq_P0_concurrent": [x for x in design_covariates(d, focal=0)[0] if x.startswith("concurrent")],
                      "pinv_rate": re_.notes["fisherz_pinv_fallback_rate"]}
out["pc_eq"] = pcr

# ---------------------------------------------------------------- 4. signs: pcorr_given_Z rule vs independent OLS coef
sg = {}
for w, r in (("E2", "R2"), ("E1", "R2"), ("E4", "R3")):
    d = ds(w, r)
    cols = cc.columns(d)
    for mname, arm in (("pc", "eq"), ("pc", "native"), ("shap_dag", "native"), ("two_tower", "native")):
        if mname == "two_tower" and w != "E2":
            continue
        res_ = METHODS[mname]().run(d, {"arm": arm})
        bad = n = 0
        for e in res_.edges:
            if not np.isfinite(e.score):
                continue
            fam, j = cc.resolve(d, cols, e.source)
            x, src = cc.source_column(d, cols, fam, j)
            cs = cc.cond_set(d, fam, src, res_.notes["arm"])
            m = cs.mask
            _, b = ols_p(cols.Y[m, d.kpi_names.index(e.target)], np.column_stack([cs.Z[m], x[m]]))
            bad += int(np.sign(b)) != e.sign
            n += 1
        sg[f"{w}{r}:{mname}:{arm}"] = {"n": n, "sign_mismatch": bad, "arm_used": res_.notes["arm"]}
out["signs"] = sg

# ---------------------------------------------------------------- 5a. two_tower row share + label; 5b. shap_dag
d = ds("E2", "R2")
tt = METHODS["two_tower"]().run(d, {})
gate = tt.notes["gate"]
worst = 0.0
for e in tt.edges:
    if e.target in gate and e.source in gate[e.target]:
        row = gate[e.target]
        worst = max(worst, abs(e.score - row[e.source] / sum(row.values())))
out["two_tower"] = {"max_abs_share_err": worst, "label": tt.notes["label"], "arm": tt.notes["arm"],
                    "arm_note": tt.notes.get("arm_note"),
                    "row_sums_of_scores": {k: float(sum(e.score for e in tt.edges if e.target == k
                                                        and np.isfinite(e.score))) for k in d.kpi_names}}

import shap  # noqa: E402
import xgboost  # noqa: E402

m = cc.int_seed(d, 3)
mdl = xgboost.XGBRegressor(random_state=m, n_jobs=1).fit(d.X_action, d.Y[:, 5])
cfg = mdl.get_xgb_params()
ex = shap.TreeExplainer(mdl)
sv = ex.shap_values(d.X_action)
add_err = float(np.max(np.abs(sv.sum(1) + ex.expected_value - mdl.predict(d.X_action))))
sd_res = METHODS["shap_dag"]().run(d, {})
v = {e.source: e.score for e in sd_res.edges if e.target == "K5"}
ref = np.abs(sv).mean(0) / d.Y[:, 5].std()
out["shap_dag"] = {"booster_params": {k: cfg.get(k) for k in ("objective", "max_depth", "learning_rate",
                                                                 "tree_method", "base_score")},
                   "n_estimators": mdl.n_estimators, "num_boosted_rounds": mdl.get_booster().num_boosted_rounds(),
                   "shap_additivity_max_err": add_err, "pred_scale": float(np.std(mdl.predict(d.X_action))),
                   "adapter_vs_direct_K5_maxdiff": float(np.max(np.abs(
                       np.array([v[a] for a in d.action_names]) - ref))),
                   "regressor": sd_res.notes["regressor"], "packages": sd_res.notes["packages"]}

# ---------------------------------------------------------------- 5c. scores do not depend on meta beyond candidate lists
leak = {}
for mname, arm in (("pc", "eq"), ("granger", "eq"), ("shap_dag", "native"), ("two_tower", "native")):
    w = "E3" if mname == "granger" else "E2"
    d = ds(w, "R2", n=500)
    keep = {k: d.meta[k] for k in ("primary_candidates", "secondary_candidates", "diagnostic_candidates")}
    d2 = dataclasses.replace(d, meta=keep)
    a = METHODS[mname]().run(d, {"arm": arm})
    b = METHODS[mname]().run(d2, {"arm": arm})
    same = all((x.score == y.score or (np.isnan(x.score) and np.isnan(y.score))) and x.sign == y.sign
               for x, y in zip(a.edges, b.edges, strict=True))
    leak[mname] = {"identical_without_meta_extras": same, "meta_keys_dropped": sorted(set(d.meta) - set(keep))}
out["meta_independence"] = leak

json.dump(out, sys.stdout, indent=1, default=str)

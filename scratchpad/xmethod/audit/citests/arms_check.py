"""audit-citests, checks 2 / 3: arms (R-25 / R-28 / R-37), eq_dropped, masked rows, R-22 not-testable format and
counts, R-4 signs, R-2 BY per family, RNG streams, Dataset-only inputs. Independent recomputation (numpy /
statsmodels) against the shared helper, DEV seed 3_000_000, n 1000.
Run: uv run python scratchpad/xmethod/audit/citests/arms_check.py > scratchpad/xmethod/audit/citests/arms_check.json
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from cdd_oran.xmethod import score as sc
from cdd_oran.xmethod.covariates import design_covariates
from cdd_oran.xmethod.methods import _citests_common as cc
from cdd_oran.xmethod.methods._classic_common import cond_set
from cdd_oran.xmethod.methods.pcorr import PCorrMethod
from cdd_oran.xmethod.methods.rcot2 import RCoT2Method
from cdd_oran.xmethod.worlds.generate import REGIMES_OF, generate_dataset

SEED, N = 3_000_000, 1000
CONF = "P_placebo_conf"
out: dict = {}


def is_conf(nm: str) -> bool:
    b = nm.removesuffix(":missing")
    for pre in ("sp:", "concurrent:"):
        b = b.removeprefix(pre)
    return b.split("@t-")[0] == CONF


def cells():
    for w, regs in REGIMES_OF.items():
        for r in regs:
            yield w, r, 1.0


def expected_z(d, src, arm):
    """Independent expectation: helper set (R-25 action / R-28 lagged KPI), minus P_placebo_conf columns unless the
    source is P_placebo_conf (R-37), on mask rows, minus columns constant there; mapped by NAME."""
    if src in d.action_names:
        a = src
        if arm == "eq":
            nm, M, mk = design_covariates(d, focal=a)
        else:
            nm, M, mk = design_covariates(d, False, False, focal=a, concurrent="all")
    else:
        base_nm, base_M, mk = design_covariates(d, arm == "eq", arm == "eq")
        keep = [i for i, x in enumerate(base_nm) if x not in (f"lag_kpi:{src}", f"lag_kpi:{src}:missing")]
        nm = tuple(base_nm[i] for i in keep) + tuple(f"concurrent:{x}" for x in d.action_names)
        M = np.column_stack([base_M[:, keep], d.X_action])
    if arm == "native":
        mk = np.ones(d.n, bool)
    keep = [k for k, x in enumerate(nm) if (src == CONF or not is_conf(x)) and x != f"concurrent:{src}"]
    nm, M = [nm[k] for k in keep], M[:, keep]
    Mm = M[mk]
    nc = [k for k in range(Mm.shape[1]) if np.ptp(Mm[:, k]) > 0]
    return [nm[k] for k in nc], Mm[:, nc], mk


def canon(nm, d):
    """citests S names -> helper names (action -> concurrent:, lagged KPI -> lag_kpi:)."""
    if nm in d.action_names:
        return f"concurrent:{nm}"
    if nm in d.kpi_names:
        return f"lag_kpi:{nm}"
    return nm


# ---------------------------------------------------------------- 2. arms
arms = {}
for w, r, lam in list(cells()) + [("E4", "R3", 1.5), ("E4", "R4", 1.5)]:
    d, _t = generate_dataset(w, r, N, SEED, lam=lam, kappa=0.25)
    rec = {}
    for arm in ("eq", "native"):
        p = cc.prepare(d, arm)
        bad_names = bad_vals = 0
        conf_in_other = 0
        srcs = sorted({d.candidates[e][0] for e in range(len(d.candidates))})
        for s in srcs:
            i = d.action_names.index(s) if s in d.action_names else p.S_names.index(s)
            got = [canon(p.S_names[c], d) for c in p.z_cols(i)]
            exp_nm, exp_M, mk = expected_z(d, s, arm)
            if sorted(got) != sorted(exp_nm):
                bad_names += 1
                rec.setdefault(f"{arm}_diff", {})[s] = {"extra": sorted(set(got) - set(exp_nm)),
                                                         "missing": sorted(set(exp_nm) - set(got))}
            else:
                order = [got.index(x) for x in exp_nm]
                if not np.array_equal(p.S[:, p.z_cols(i)][:, order], exp_M):
                    bad_vals += 1
            if s != CONF:
                conf_in_other += sum(is_conf(x) or x == f"concurrent:{CONF}" for x in got)
        # classic cond_set (pre-R-37 rule on this branch) differs only by P_placebo_conf columns
        cs_extra = set()
        if arm == "eq":
            for j, a in enumerate(d.action_names):
                cs_extra |= {x for x in cond_set(d, "action", j, "eq").names} - set(
                    canon(p.S_names[c], d) for c in p.z_cols(j)) - set(p.eq_dropped)
        rec[arm] = {"sources": len(srcs), "name_mismatch": bad_names, "value_mismatch": bad_vals,
                    "conf_cols_in_other_Z": conf_in_other, "rows_used": int(p.S.shape[0]),
                    "eq_dropped": list(p.eq_dropped), "classic_minus_citests": sorted(cs_extra)}
    arms[f"{w}{r}@{lam}"] = rec
out["arms"] = arms

# ---------------------------------------------------------------- 3a. R-22 counts, kappa .25 and 0 (pcorr, both arms)
nt = {}
for kap in (0.25, 0.0):
    for w, r, lam in cells():
        d, t = generate_dataset(w, r, N, SEED, lam=lam, kappa=kap)
        for arm in ("eq", "native"):
            res = PCorrMethod().run(d, {"arm": arm})
            lst = sc.not_testable_edges(res)
            s = sc.score(res, t)
            nan_unlisted = sum(1 for e in res.edges if not np.isfinite(e.score) and (e.source, e.target) not in lst)
            nt[f"k{kap}:{w}{r}:{arm}"] = {"listed": len(lst), "nan_unlisted": nan_unlisted,
                                          "true": s["n_not_testable_true"], "null": s["n_not_testable_null"],
                                          "overridden": s["n_not_testable_overridden"],
                                          "rr_min_testable": res.notes.get("exact_fit_rr_min_testable"),
                                          "stop": res.notes.get("stop")}
out["not_testable"] = nt

# ---------------------------------------------------------------- 3b. signs (R-4) + BY per family (R-2)
sig = {}
for w, r in (("E2", "R2"), ("E1", "R2"), ("E4", "R3"), ("E5", "R1")):
    d, t = generate_dataset(w, r, N, SEED, lam=1.0, kappa=0.25)
    for mname, m in (("pcorr", PCorrMethod()), ("rcot2", RCoT2Method())):
        for arm in ("eq", "native"):
            res = m.run(d, {"arm": arm})
            p = cc.prepare(d, arm)
            sbad = n_s = 0
            maxrel = 0.0
            for e, ed in enumerate(res.edges):
                if ed.p is None:
                    continue
                i, j = p.pairs[e]
                X = np.column_stack([p.S[:, p.z_cols(i)], p.S[:, i]])
                f = sm.OLS(p.Y[:, j], sm.add_constant(X, has_constant="add")).fit()
                b = float(f.params[-1])
                sbad += int(np.sign(b)) != ed.sign
                n_s += 1
                if mname == "pcorr":
                    maxrel = max(maxrel, abs(f.pvalues[-1] - ed.p) / max(f.pvalues[-1], 1e-300))
            fam = np.asarray(res.notes["family_of_edge"])
            dbad = 0
            for fm in set(fam):
                idx = [e for e in np.nonzero(fam == fm)[0] if res.edges[e].p is not None]
                if idx:
                    rej = multipletests([res.edges[e].p for e in idx], 0.05, "fdr_by")[0]
                    dbad += sum(bool(a) != res.edges[e].declared for a, e in zip(rej, idx))
            sig[f"{w}{r}:{mname}:{arm}"] = {"n": n_s, "sign_mismatch": sbad, "by_mismatch": dbad,
                                           "families": sorted(set(fam)), "p_vs_ols_maxrel": maxrel,
                                           "sign_rule": res.notes["sign_rule"]}
out["signs_by"] = sig

# ---------------------------------------------------------------- 3c. Dataset-only inputs (meta stripped, truth absent)
d, t = generate_dataset("E2", "R2", N, SEED, kappa=0.25)
keep_meta = {k: d.meta[k] for k in d.meta if k in ("context_names",)}
d2 = dataclasses.replace(d, meta=keep_meta)
a = PCorrMethod().run(d, {}); b = PCorrMethod().run(d2, {})
out["meta_strip_equal"] = [(x.score, x.p, x.sign) for x in a.edges] == [(x.score, x.p, x.sign) for x in b.edges]
out["meta_keys"] = sorted(d.meta)

# ---------------------------------------------------------------- 3d. RNG seeds: tag 7802, distinct per method
out["method_seed"] = {k: cc.method_seed(SEED, k) for k in range(1, 6)}
print(json.dumps(out, indent=1, default=str))

"""Discovery baselines for the E6-P discovery study (scratchpad/e6_dev/decision/STEP1_MSCR_PLAN.md, "Methods";
protocol docs/benchmark/E6P_DISCOVERY_PROTOCOL.md), on the SAME unit table as MSCR-CRT (``crt_units.UnitData``), plus
the naive MSCR-rowperm comparator on the pooled obs-only panels and PACIFISTA's native xApp-level severity.

Every unit-table baseline returns a score >= 0 and a sign per hypothesis (f, rel, k) of the 60-edge space
(``edge_score.HYPOTHESES``); NaN = not scorable (the SAME support rule as the CRT, ``supported``: >= 30 units, >= 5
accepted, >= 5 rejected, >= 3 episodes; or a degenerate target). Treatment x = level * sgn ("dir"), target y =
relation KPI post - pre, pre = the pre-window value (``crt_units`` definitions).
  corr        |Pearson corr(x, y)| over the family's units; sign = sign(corr).
  granger     lagged F-test (Granger-style, 1 lag = the pre-window): OLS post = a + b pre + c x, F(1, n - 3) test of
              c; score = -log10 p, sign = sign(c). Reported twice: "granger_by" = BY at q = 0.05 over the scorable
              hypotheses (untuned) and "granger" = the DEV-tuned threshold.
  shap_gbdt   Sharma et al. 2025 via ``scripts/e2_baseline_shap_dag.fit_shap_importances`` (sklearn HistGBDT +
              shap.TreeExplainer), one fit per hypothesis on the family's units, features [x, sgn, pre]; score =
              mean|SHAP(x)| / std(y) (absolute, scale-free: the paper's per-KPI relative-dominance rule would declare
              >= 1 family for EVERY target, far included, so a common absolute threshold is used); sign = sign(corr(x,
              y)) (the importances are unsigned). Fallback when the script is not importable: sklearn
              GradientBoostingRegressor + permutation importance (recorded in ``notes``).
  two_tower   ``scripts/e2_baseline_gnn.fit_two_tower`` with its ``TwoTower`` re-parametrised by wrapping (not
              editing): num_params = 4 (x of each family, 0 on other families' units), num_kpis = 15 (relation x KPI
              targets), d / r / hidden / epochs / l1 declared (``TT_SIZES``); score = the learned gate a[target,
              family]; sign = sign(corr(x, y)). l1 = 0.5 (the script's E2 default 1e-3 leaves the softplus gate
              unidentified on this family-masked input: on the synthetic planted-edge check the planted gate ranked
              at chance for 3 / 3 seeds at 1e-3 and first for 3 / 3 at 0.5; fixed a priori from SYNTHETIC data, never
              from DEV / EVAL records).
  int         PACIFISTA-style (del Prever et al. 2025, Table 1 INT = ``envs.e6.published.int_distance``): ECDF
              distance of y * sgn between accepted and rejected units of the family (half excluded); sign =
              sign(mean_accept - mean_reject).
  qacm        QACM KPI predictor (Wadud et al. 2024 Sec. VI-B; ``published._ANN`` / ``_PR``, the better held-out R^2
              kept): own-cell post-window KPI (z-scored) on [applied change = level * step, cur, KPI now = pre-window
              (z-scored), own PRB util]; score = mean over units with a change of |pred(X) - pred(X | change 0)| (z
              units), sign = sign of mean (pred difference * sign(change)). OWN CELL ONLY (the method predicts the
              KPI of the acted-on cell): nbr / far are never declared (indirect recall 0 by construction). For the
              tau rule only, the same predictor is fitted on the far relation of DEV (never declared).
Thresholds (``tune_tau``): PRIMARY rule "far_fpr": tau = the (max_far + 1)-th largest DEV far-relation score (declare
iff score > tau), max_far = 1 = P1's far budget for MSCR, so every baseline gets the same DEV specificity on the
negative-control relation; fewer than max_far + 1 scorable far hypotheses -> untunable -> declares nothing.
Sensitivity rule "physics_f1": tau maximising DEV F1 vs ``edge_score.PHYSICS_PRIOR`` (declared proxy; reported,
never a verdict input). tau is frozen after DEV and applied to EVAL (pooled and per fold). MSCR-CRT gets no tuning.
"""
from __future__ import annotations

import contextlib
import dataclasses
import functools

import numpy as np

from .crt_units import FAMILIES, KPIS, RELATIONS, UnitData
from .edge_score import HYPOTHESES, PHYSICS_PRIOR, confusion, physics_reference

TT_SIZES = {"d": 16, "r": 8, "hidden": 32, "epochs": 400, "lr": 0.01, "l1": 0.5}   # l1: see module doc
MAX_FAR_DEV = 1
TAU_RULES = ("far_fpr", "physics_f1")
TUNED = ("shap_gbdt", "corr", "granger", "two_tower", "int", "qacm")      # the DEV-tuned baselines (P2)


@dataclasses.dataclass(frozen=True)
class Support:
    min_units: int = 30
    min_accept: int = 5
    min_reject: int = 5
    min_episodes: int = 3


SUPPORT = Support()


def supported(data: UnitData, family: str, sup: Support = SUPPORT) -> bool:
    rows = data.rows_of(family)
    lv = data.level[rows]
    return (len(rows) >= sup.min_units and int((lv == 1.0).sum()) >= sup.min_accept
            and int((lv == 0.0).sum()) >= sup.min_reject and len(np.unique(data.episode[rows])) >= sup.min_episodes)


def _empty(relations=RELATIONS):
    return {h: {"score": float("nan"), "sign": 0} for h in HYPOTHESES if h[1] in relations}


def _corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _each(data, sup, relations=RELATIONS):
    """(family, rows, rel, kpi, h) over the scorable family/hypotheses."""
    for f in FAMILIES:
        if not supported(data, f, sup):
            continue
        rows = data.rows_of(f)
        for r in relations:
            for k in KPIS:
                yield f, rows, r, k, (f, r, k)


# ------------------------------------------------------------------------------------------------ corr / granger
def score_corr(data: UnitData, sup: Support = SUPPORT) -> dict:
    out = _empty()
    for _f, rows, r, k, h in _each(data, sup):
        c = _corr(data.x[rows], data.y[(r, k)][rows])
        if np.isfinite(c):
            out[h] = {"score": abs(c), "sign": int(np.sign(c))}
    return out


def granger_pvalues(data: UnitData, sup: Support = SUPPORT) -> dict:
    from scipy.stats import f as fdist
    out = _empty()
    for _f, rows, r, k, h in _each(data, sup):
        pre = data.pre[(r, k)][rows]
        post = data.y[(r, k)][rows] + pre
        x = data.x[rows]
        n = len(rows)
        A = np.column_stack([np.ones(n), pre, x])
        if np.std(post) == 0 or np.linalg.matrix_rank(A) < 3:
            continue
        beta, *_ = np.linalg.lstsq(A, post, rcond=None)
        res = post - A @ beta
        s2 = float(res @ res) / (n - 3)
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = float(np.sqrt(max(cov[2, 2], 0.0)))
        if se <= 0:
            continue
        F = (beta[2] / se) ** 2
        p = float(fdist.sf(F, 1, n - 3))
        out[h] = {"score": float(-np.log10(max(p, 1e-300))), "sign": int(np.sign(beta[2])), "p": p}
    return out


def granger_by(scores: dict, q: float = 0.05) -> dict:
    """Untuned Granger: BY at q over the scorable hypotheses."""
    from cdd_oran.discovery.mscr import by_declare
    keys = [h for h, v in scores.items() if np.isfinite(v["score"])]
    dec = by_declare([scores[h]["p"] for h in keys], q) if keys else []
    out = {h: {"declared": False, "sign": v["sign"], "score": v["score"]} for h, v in scores.items()}
    for h, d in zip(keys, dec, strict=True):
        out[h]["declared"] = bool(d)
    return out


# ------------------------------------------------------------------------------------------------ SHAP-GBDT
def _shap_fn():
    """(fit(X, y, seed) -> (importances, r2), note)."""
    try:
        from scripts.e2_baseline_shap_dag import fit_shap_importances
        return fit_shap_importances, "scripts.e2_baseline_shap_dag.fit_shap_importances (HistGBDT + shap.TreeExplainer)"
    except Exception as e:                                           # noqa: BLE001  (fallback, recorded)
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.inspection import permutation_importance

        def fit(X, y, seed=0):
            m = GradientBoostingRegressor(random_state=seed).fit(X, y)
            pi = permutation_importance(m, X, y, n_repeats=5, random_state=seed)
            return np.abs(pi.importances_mean), float(m.score(X, y))
        return fit, f"FALLBACK sklearn GBDT + permutation importance (shap script not importable: {e!r})"


def score_shap(data: UnitData, sup: Support = SUPPORT, seed: int = 0) -> dict:
    fit, note = _shap_fn()
    out = _empty()
    for _f, rows, r, k, h in _each(data, sup):
        y = data.y[(r, k)][rows]
        sd = float(np.std(y))
        if sd == 0:
            continue
        X = np.column_stack([data.x[rows], data.sgn[rows], data.pre[(r, k)][rows]])
        imp, r2 = fit(X, y, seed=seed)
        c = _corr(data.x[rows], y)
        out[h] = {"score": float(imp[0]) / sd, "sign": int(np.sign(c)) if np.isfinite(c) else 0, "r2": float(r2)}
    out["_note"] = note
    return out


# ------------------------------------------------------------------------------------------------ two-tower
@contextlib.contextmanager
def _two_tower_sized(mod, **sizes):
    """Temporarily re-bind ``mod.TwoTower`` to a sized partial (``fit_two_tower`` builds ``TwoTower()``)."""
    orig = mod.TwoTower
    mod.TwoTower = functools.partial(orig, **sizes)
    try:
        yield
    finally:
        mod.TwoTower = orig


def score_two_tower(data: UnitData, sup: Support = SUPPORT, seed: int = 0, sizes: dict | None = None) -> dict:
    import scripts.e2_baseline_gnn as G
    sz = dict(TT_SIZES, **(sizes or {}))
    out = _empty()
    fams = [f for f in FAMILIES if supported(data, f, sup)]
    if not fams or data.n < 3:
        return out
    X = np.column_stack([data.x * (data.family == FAMILIES.index(f)) for f in fams])
    tg = [(r, k) for r in RELATIONS for k in KPIS]
    Y = np.column_stack([data.y[t] for t in tg])
    with _two_tower_sized(G, num_params=len(fams), num_kpis=len(tg), d=sz["d"], r=sz["r"], hidden=sz["hidden"]):
        S, r2 = G.fit_two_tower(X, Y, epochs=sz["epochs"], lr=sz["lr"], l1=sz["l1"], seed=seed)
    for j, f in enumerate(fams):
        rows = data.rows_of(f)
        for i, (r, k) in enumerate(tg):
            if np.std(data.y[(r, k)][rows]) == 0:
                continue
            c = _corr(data.x[rows], data.y[(r, k)][rows])
            out[(f, r, k)] = {"score": float(S[i, j]), "sign": int(np.sign(c)) if np.isfinite(c) else 0}
    out["_note"] = f"fit_two_tower with TwoTower sized {sz} (wrapped), pseudo-R2 {r2:.3f}"
    return out


# ------------------------------------------------------------------------------------------------ INT
def score_int(data: UnitData, sup: Support = SUPPORT) -> dict:
    from cdd_oran.envs.e6.published import int_distance
    out = _empty()
    for _f, rows, r, k, h in _each(data, sup):
        yd = data.y[(r, k)][rows] * data.sgn[rows]
        lv = data.level[rows]
        a, b = yd[lv == 1.0], yd[lv == 0.0]
        if np.ptp(np.concatenate([a, b])) == 0:
            continue
        out[h] = {"score": int_distance(a, b), "sign": int(np.sign(a.mean() - b.mean()))}
    return out


# ------------------------------------------------------------------------------------------------ QACM
def score_qacm(data: UnitData, sup: Support = SUPPORT, relations=("own",), seed: int = 0, holdout: float = 0.2,
               min_samples: int = 30, ann: bool = True) -> dict:
    from cdd_oran.envs.e6.published import _ANN, _PR, _r2
    out = _empty()
    util_all = data.z.get("ctx_own_prb_util", np.zeros(data.n))
    for f, rows, r, k, h in _each(data, sup, relations):
        pre = data.pre[(r, k)][rows]
        post = data.y[(r, k)][rows] + pre
        mu, sd = float(post.mean()), float(post.std())
        if sd == 0 or len(rows) < min_samples:
            continue
        delta = data.level[rows] * data.step[rows]
        cur = data.z.get("ctx_cur", np.zeros(data.n))[rows]
        util = util_all[rows]
        util = np.where(np.isfinite(util), util, np.nanmedian(util) if np.isfinite(util).any() else 0.0)
        X = np.column_stack([delta, cur, (pre - mu) / sd, util])
        fm, fs = X.mean(0), X.std(0) + 1e-9
        Xs = (X - fm) / fs
        X0 = X.copy()
        X0[:, 0] = 0.0
        X0s = (X0 - fm) / fs
        yz = (post - mu) / sd
        perm = np.random.default_rng([seed, 6616, 7, 7, *map(int, (FAMILIES.index(f), RELATIONS.index(r),
                                                                      KPIS.index(k)))]).permutation(len(yz))
        n_te = max(int(len(yz) * holdout), 1)
        te, tr = perm[:n_te], perm[n_te:]
        cands = {"PR": _PR().fit(Xs[tr], yz[tr])}
        if ann:
            cands["ANN"] = _ANN(Xs.shape[1], seed).fit(Xs[tr], yz[tr])
        r2 = {m: _r2(yz[te], g.predict(Xs[te])) for m, g in cands.items()}
        best = max(r2, key=r2.get)
        g = cands[best]
        chg = delta != 0
        if not chg.any():
            continue
        eff = g.predict(Xs[chg]) - g.predict(X0s[chg])
        out[h] = {"score": float(np.mean(np.abs(eff))), "sign": int(np.sign(np.mean(eff * np.sign(delta[chg])))),
                  "model": best, "r2": r2}
    return out


# ------------------------------------------------------------------------------------------------ tuning / declaring
def tune_tau(scores: dict, rule: str = "far_fpr", max_far: int = MAX_FAR_DEV, prior: dict | None = None) -> dict:
    """Threshold from DEV scores (module docstring). Returns {"tau", "rule", ...}; declare iff score > tau."""
    if rule == "far_fpr":
        far = sorted((v["score"] for h, v in scores.items() if isinstance(h, tuple) and h[1] == "far"
                      and np.isfinite(v["score"])), reverse=True)
        if len(far) <= max_far:
            return {"tau": float("inf"), "rule": rule, "max_far": max_far, "n_far_scorable": len(far),
                    "note": "untunable (too few scorable far hypotheses): declares nothing"}
        return {"tau": float(far[max_far]), "rule": rule, "max_far": max_far, "n_far_scorable": len(far)}
    if rule == "physics_f1":
        ref = physics_reference(prior)
        fin = sorted({v["score"] for h, v in scores.items() if isinstance(h, tuple) and np.isfinite(v["score"])})
        best = (-1.0, float("inf"))
        for tau in [-np.inf, *fin]:
            f1 = confusion(declare(scores, tau), ref)["f1"]
            f1 = 0.0 if not np.isfinite(f1) else f1
            if f1 > best[0] + 1e-12 or (abs(f1 - best[0]) <= 1e-12 and tau > best[1]):
                best = (f1, tau)
        return {"tau": float(best[1]), "rule": rule, "dev_proxy_f1": best[0], "prior": "edge_score.PHYSICS_PRIOR"}
    raise ValueError(f"rule in {TAU_RULES}")


def declare(scores: dict, tau: float, relations=RELATIONS) -> dict:
    """{h: {"declared", "sign", "score"}}; unscorable (NaN) -> not declared; hypotheses outside ``relations`` ->
    declared False (the method never declares them)."""
    out = {}
    for h in HYPOTHESES:
        v = scores.get(h, {"score": float("nan"), "sign": 0})
        s = v["score"]
        out[h] = {"declared": bool(h[1] in relations and np.isfinite(s) and s > tau), "sign": int(v["sign"]),
                  "score": s}
    return out


SCORERS = {"shap_gbdt": score_shap, "corr": score_corr, "granger": granger_pvalues, "two_tower": score_two_tower,
           "int": score_int, "qacm": score_qacm}
DECL_RELATIONS = {"qacm": ("own",)}


def run_baselines(dev: UnitData, splits: dict, methods=TUNED, sup: Support = SUPPORT, seed: int = 0,
                  log=None, ann: bool = True) -> dict:
    """Tune each baseline's tau on ``dev`` (primary far_fpr + physics sensitivity), freeze it, score ``splits``
    ({name: UnitData}) and declare. Also "granger_by" (untuned). Returns {method: {...}}."""
    import time
    out = {}
    for m in methods:
        t0, w0 = time.process_time(), time.time()
        fn = SCORERS[m]
        kw = {"sup": sup}
        if m in ("shap_gbdt", "two_tower", "qacm"):
            kw["seed"] = seed
        if m == "qacm":
            kw["ann"] = ann
        dev_kw = dict(kw, relations=("own", "far")) if m == "qacm" else kw
        dev_scores = fn(dev, **dev_kw)
        tau = tune_tau(dev_scores, "far_fpr")
        tau_phys = tune_tau(dev_scores, "physics_f1")
        rels = DECL_RELATIONS.get(m, RELATIONS)
        res = {"tau": tau, "tau_physics": tau_phys, "note": dev_scores.get("_note"), "decl_relations": list(rels),
               "dev_scores": {"|".join(h): v for h, v in dev_scores.items() if isinstance(h, tuple)}, "splits": {}}
        for name, d in splits.items():
            sc = dev_scores if d is dev else fn(d, **kw)
            res["splits"][name] = {"declared": declare(sc, tau["tau"], rels),
                                   "declared_physics_tau": declare(sc, tau_phys["tau"], rels),
                                   "scores": {"|".join(h): v for h, v in sc.items() if isinstance(h, tuple)}}
        res["cpu_s"], res["wall_s"] = round(time.process_time() - t0, 1), round(time.time() - w0, 1)
        out[m] = res
        if log:
            log(f"   baseline {m}: tau {tau['tau']:.4g} (physics {tau_phys['tau']:.4g}) wall {res['wall_s']} s "
                f"(cpu {res['cpu_s']} s)")
        if m == "granger":
            gb = {"tau": None, "note": "untuned: BY at q = 0.05 over the scorable hypotheses", "splits": {}}
            for name, d in splits.items():
                sc = dev_scores if d is dev else res["splits"][name]["scores"]
                sc = {tuple(k.split("|")) if isinstance(k, str) else k: v for k, v in sc.items()}
                gb["splits"][name] = {"declared": granger_by(sc)}
            out["granger_by"] = gb
    return out


# ------------------------------------------------------------------------------------------------ MSCR-rowperm
ROWPERM_COLUMNS = {"own_carrier": ("carrier", "own"), "own_sleep": ("sleep", "own"), "own_ptx": ("ptx", "own"),
                   "own_prot_min": ("prot_min", "own"), "nbr_carrier": ("carrier", "nbr"),
                   "nbr_sleep": ("sleep", "nbr"), "nbr_ptx": ("ptx", "nbr"), "nbr_prot_min": ("prot_min", "nbr")}
ROWPERM_KPIS = {"prot_viol": "pv", "energy_w": "e", "rlf": "rlf"}
ROWPERM_UNMAPPED = {"v": "no all-slice violated-UE KPI in the obs panel (ll_delay_p95 / embb_thp_p5 are not UE-s)",
                    "load": "no served-UE KPI in the obs panel (prb_util / act_ue are different quantities)",
                    "far": "the panel has no far-relation aggregate (knob_nbr = CIO-neighbour means only)"}


def mscr_rowperm(records, targets=tuple(ROWPERM_KPIS), seed: int = 0, n_perm: int | None = None,
                 n_jobs: int | None = None, device: str | None = "cpu") -> dict:
    """Naive comparator: frozen MSCR-v2 row-permutation template discovery (``discovery.discover_template``) on the
    pooled obs-only E6-P panels, mapped onto the 60-edge space where possible (module constants)."""
    from cdd_oran.discovery.mscr import MSCRConfig

    from .discovery import KPI_OWNER_P, DiscoveryConfig, discover_template
    from .features import concat_panels, drop_degenerate_p, panel_from_rec
    panels = [panel_from_rec(r["panel"], episode=i) for i, r in enumerate(records)]
    P = drop_degenerate_p(concat_panels(panels, drop_degenerate=False))
    mcfg = MSCRConfig() if n_perm is None else dataclasses.replace(MSCRConfig(), n_perm=int(n_perm))
    cfg = DiscoveryConfig(mscr=mcfg, kpi_owner=KPI_OWNER_P, targets=tuple(t for t in targets if t in P.data),
                          seed=seed)
    g = discover_template(P, cfg, n_jobs=n_jobs, device=device)
    decl = {h: {"declared": None, "sign": 0, "score": float("nan")} for h in HYPOTHESES}
    unmapped_edges = []
    for e in g.edges:
        fr = ROWPERM_COLUMNS.get(e["column"])
        k = ROWPERM_KPIS.get(e["kpi"])
        if fr is None or k is None:
            unmapped_edges.append({x: e[x] for x in ("column", "kpi", "p", "declared", "sign")})
            continue
        decl[(fr[0], fr[1], k)] = {"declared": bool(e["declared"]), "sign": int(e["sign"]), "p": e["p"],
                                   "score": float(-np.log10(max(e["p"], 1e-300))), "column": e["column"]}
    n_mapped = sum(1 for v in decl.values() if v["declared"] is not None)
    return {"declared": decl, "n_mapped": n_mapped, "n_unmapped": len(HYPOTHESES) - n_mapped,
            "unmapped_hypotheses": ROWPERM_UNMAPPED, "unmapped_panel_edges": unmapped_edges,
            "diagnostics": {k: {"status": v.status, "n_rows": v.n_rows, "dependence_ok": v.dependence_ok,
                                "power_floor_ok": v.power_floor_ok, "acf1_y": v.acf1_y}
                            for k, v in g.diagnostics.items()},
            "config": {"n_perm": mcfg.n_perm, "targets": list(cfg.targets), "thin_s": cfg.thin_s},
            "caveat": "row-permutation p-values on serially dependent, fed-back panels (REPORT_H.md): approximate"}


# ------------------------------------------------------------------------------------------------ PACIFISTA native
def pacifista_native(prof_records, kpis=KPIS, delta_tol: float = 0.25) -> dict:
    """PACIFISTA's own xApp-level severity from the PROF records (only profile xApp X's units accepted): per KPI the
    ECDF of the per-cell per-second scored values of each profile; sigma(a, b) = mean over KPIs of INT(a, b). Pairs
    with sigma > delta_tol are "in conflict" (the paper's deploy rule). xApp level: not mappable onto the edge space."""
    from cdd_oran.envs.e6.published import int_distance

    from .collect_p import dec
    from .crt_units import SERIES_OF
    samples = {}
    for rec in prof_records:
        x = rec.get("sub") or str(rec.get("policy", "")).split(":")[-1]
        ls = rec["lab_series"]
        D = dec(ls["data"]).astype(float)
        sc = dec(ls["scored"]) > 0.5
        s = samples.setdefault(x, {k: [] for k in kpis})
        for k in kpis:
            s[k].append(D[sc][:, list(ls["fields"]).index(SERIES_OF[k]), :].ravel())
    samples = {x: {k: np.concatenate(v) if v else np.zeros(0) for k, v in s.items()} for x, s in samples.items()}
    xs = sorted(samples)
    sigma, per_kpi = {}, {}
    for i, a in enumerate(xs):
        for b in xs[i + 1:]:
            d = {k: int_distance(samples[a][k], samples[b][k]) for k in kpis
                 if len(samples[a][k]) and len(samples[b][k])}
            per_kpi[f"{a}|{b}"] = d
            sigma[f"{a}|{b}"] = float(np.mean(list(d.values()))) if d else float("nan")
    return {"profiles": {x: int(sum(1 for r in prof_records if (r.get("sub") or "") == x)) for x in xs},
            "sigma": sigma, "per_kpi": per_kpi, "delta_tol": delta_tol,
            "conflicts": [p for p, v in sigma.items() if np.isfinite(v) and v > delta_tol],
            "means": {x: {k: float(np.mean(v)) if len(v) else float("nan") for k, v in s.items()}
                      for x, s in samples.items()},
            "mapping": "xApp-pair severity; no family / relation / KPI edge: not mapped onto the 60-edge space"}


__all__ = ["DECL_RELATIONS", "MAX_FAR_DEV", "PHYSICS_PRIOR", "ROWPERM_COLUMNS", "ROWPERM_KPIS", "SCORERS", "TT_SIZES",
           "TUNED", "Support", "declare", "granger_by", "granger_pvalues", "mscr_rowperm", "pacifista_native",
           "run_baselines", "score_corr", "score_int", "score_qacm", "score_shap", "score_two_tower", "supported",
           "tune_tau"]

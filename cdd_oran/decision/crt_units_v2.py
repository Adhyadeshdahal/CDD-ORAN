"""mscr-crt-units-v2: design-centred, outcome-residualised linear CRT on the E6-P unit table (discovery v2 re-test).

Same unit table (``crt_units.UnitData`` / ``crt_units.build_unit_data``: targets, relations, pre-windows, logged pi0
tables), same assignment model and null (``crt_units.PiAssignment.conditional_draws``), same support rule, statuses
and BY selection as mscr-crt-units-v1 (``crt_units``). What changes (step-1 diagnosis, .tmp/diag; agent B's "adj"):

  dose        ``LEVEL_V2`` = accept 1 / half 1 / reject 0 / accept+rb 1 (half converges to the requested value within
              the 60 s unit via the slew / alternation rule; applied 88-100 %): the treatment is "applied vs rejected".
  regressor   v_u = sgn_u * (LEVEL_V2[mode_u] - E_pi0u[LEVEL_V2]), E over the unit's LOGGED pi0 table (``probs`` row);
              sgn_u = sign(step), 0 -> +1 ("dir" orientation: a positive dependence = effect of a knob-value increase).
              Design-centred: E[v_u] = 0 under pi0 for every unit, so the non-randomized request direction sgn_u (and
              anything it correlates with) cannot leak into the statistic's null mean.
  outcome     for hypothesis (f, rel, kpi), over the family's units: r = y(rel, kpi) residualised by OLS on
              [episode x sgn fixed effects] + the same target's pre-window value pre(rel, kpi) (one pooled slope;
              Frisch-Waugh: within-stratum demeaning, then the within slope on pre). If the within-stratum pre is
              degenerate (constant / non-finite) -> fixed effects only. r depends on the outcomes and the pre-assignment
              skeleton only, so it is FIXED under the family's sharp null: computed once, never per draw.
  statistic   S = sum_u v_u r_u, T = |S|. Null: redraw the family's modes from the logged tables (every other family
              held fixed) exactly as ``PiAssignment.conditional_draws``; p = (1 + #{T_b >= T_obs - tol}) / (B + 1),
              finite-sample valid under the family's sharp null (the CRT argument of crt_units). tol = 1e-9 (1 + sum|r|)
              (conservative; only guards float round-off between the observed dot product and the batched one).
  effect      beta = S / sum v^2 (design-centred slope on the residualised outcome); sign = sign(beta). z_approx =
              S / sd(S_draws) (descriptive; |z_approx| = T_obs / sd of the signed null draws).

RNG: ``default_rng([seed, 6616, 2, family_idx, lag, split])`` (tag 6616, registered; the version component 2 makes
the streams differ from v1's ``[seed, 6616, family_idx, lag, split]``). No row permutation (the statistic has no
ties to break). Draws are generated and consumed in chunks of the same stream (identical p for any chunk size); the
chunk is capped so a (chunk x n) float64 block stays <= ``max_chunk_bytes`` (n ~ 42k units at 480 episodes -> chunk
~190, peak << 1 GB). All 15 targets of a family share the draws: S_chunk = V_chunk @ R, R = (n x n_targets).

Support / statuses / selection: exactly v1 (``UnitCRTConfigV2``: >= 30 units, >= 5 accepted (accept / accept+rb),
>= 5 rejected, >= 3 episodes, else undetermined "support"; a non-finite or degenerate outcome
(``features.degenerate_columns``) -> undetermined "degenerate_outcome"); BY at q over the TESTED hypotheses of one
call; statuses declared | not_detected | undetermined. Lag-1 carryover audit (``carryover_audit_v2``): optional, off
by default, descriptive only.

Input: ``crt_units.UnitData`` only (built by v1's ``build_unit_data``, which never reads ``gt_static``,
``gt_labels``, ``lab_outcome``); this module reads no record at all.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from cdd_oran.discovery import mscr

from .crt_units import (
    CRT_TAG,
    FAMILIES,
    KPIS,
    LEVEL_ARR,
    MODES,
    SPLIT_PLACEBO,
    SPLIT_POOLED,
    PiAssignment,
    UnitData,
    power_floor,
)
from .features import degenerate_columns

CRT_UNITS_V2_VERSION = "mscr-crt-units-v2"
V2_STREAM = 2                                          # RNG version component (v1 has none)
LEVEL_V2 = {"accept": 1.0, "half": 1.0, "reject": 0.0, "accept+rb": 1.0}
LEVEL_V2_ARR = np.array([LEVEL_V2[m] for m in MODES])
CLAIMS_V2 = {
    "declared": "sharp null of no ASSIGNED effect of the family's decisions on this KPI window rejected (design-"
                "centred residualised CRT, mscr-crt-units-v2); BY across the tested hypotheses of this call",
    "not_detected": "not detected at this support: this is NOT evidence that the edge is absent",
    "undetermined": "undetermined ({reason}): the edge is unknown, neither present nor absent",
}


@dataclasses.dataclass(frozen=True)
class UnitCRTConfigV2:
    B: int = 9999
    alpha: float = 0.05
    q: float = 0.05
    chunk: int = 1000                        # max draws per pass
    max_chunk_bytes: int = 64_000_000        # cap on one (chunk x n) float64 block
    audit_B: int = 999
    min_units: int = 30
    min_accept: int = 5
    min_reject: int = 5
    min_episodes: int = 3
    seed: int = 0


# ------------------------------------------------------------------------------------------------ pieces
def design_mean(data: UnitData, rows) -> np.ndarray:
    """E_pi0u[LEVEL_V2] per row (logged table)."""
    return data.probs[np.asarray(rows, int)] @ LEVEL_V2_ARR


def design_regressor(data: UnitData, rows) -> np.ndarray:
    """v_u = sgn_u (LEVEL_V2[mode_u] - E_pi0u[LEVEL_V2]) of the LOGGED modes."""
    rows = np.asarray(rows, int)
    return data.sgn[rows] * (LEVEL_V2_ARR[data.mode[rows]] - design_mean(data, rows))


def strata_of(data: UnitData, rows) -> np.ndarray:
    """episode x sgn stratum id per row."""
    rows = np.asarray(rows, int)
    return data.episode[rows].astype(np.int64) * 2 + (data.sgn[rows] > 0)


def _demean(a: np.ndarray, inv: np.ndarray, k: int) -> np.ndarray:
    s = np.bincount(inv, a, k)
    n = np.bincount(inv, minlength=k).astype(float)
    return a - (s / np.maximum(n, 1.0))[inv]


def residualise(y: np.ndarray, pre: np.ndarray | None, strata: np.ndarray) -> tuple[np.ndarray, dict]:
    """OLS residual of y on [stratum FE] + pre (one pooled slope), via within-stratum demeaning (Frisch-Waugh).
    Degenerate pre (None / non-finite / constant within strata) -> FE only."""
    y = np.asarray(y, float)
    _, inv = np.unique(strata, return_inverse=True)
    k = int(inv.max()) + 1 if len(inv) else 0
    yd = _demean(y, inv, k)
    info = {"resid": "fe", "pre_slope": float("nan"), "n_strata": k}
    if pre is not None:
        pre = np.asarray(pre, float)
        if np.all(np.isfinite(pre)):
            pd = _demean(pre, inv, k)
            ss = float(pd @ pd)
            scale = float(np.sum((pre - pre.mean()) ** 2)) if len(pre) else 0.0
            if ss > 1e-10 * max(scale, 1e-300) and ss > 0:
                b = float(pd @ yd) / ss
                return yd - b * pd, dict(info, resid="fe+pre", pre_slope=b)
    return yd, info


def _chunk(cfg: UnitCRTConfigV2, n: int) -> int:
    return int(max(1, min(int(cfg.chunk), int(cfg.max_chunk_bytes) // (8 * max(int(n), 1)))))


def _stream(cfg: UnitCRTConfigV2, fi: int, lag: int, split: int) -> np.random.Generator:
    return np.random.default_rng([int(cfg.seed), CRT_TAG, V2_STREAM, int(fi), int(lag), int(split)])


# ------------------------------------------------------------------------------------------------ test
def family_tests_v2(data: UnitData, family: str, targets, cfg: UnitCRTConfigV2 | None = None, lag: int = 0,
                    split: int = SPLIT_POOLED) -> list:
    """CRT v2 of family ``family`` against each target. ``targets``: [(name, (rel, kpi) key of data.y | ("pre",
    rel, kpi) for the pre-window value)]. One set of B draws for all targets. Returns one dict per target (status
    "tested" | "undetermined")."""
    cfg = cfg or UnitCRTConfigV2()
    fi = FAMILIES.index(family)
    rows = data.rows_of(family)
    if lag == 1:
        rows = rows[data.prev[rows] >= 0]
    elif lag != 0:
        raise ValueError("lag must be 0 or 1")
    src = rows if lag == 0 else data.prev[rows]
    lv = LEVEL_ARR[data.mode[src]]                     # v1 support counts (accept / accept+rb vs reject)
    n_acc, n_rej = int((lv == 1.0).sum()), int((lv == 0.0).sum())
    n_ep = len(np.unique(data.episode[src]))
    base = {"version": CRT_UNITS_V2_VERSION, "family": family, "lag": lag, "split": split, "n": len(rows),
            "n_accept": n_acc, "n_reject": n_rej, "n_half": int((lv == 0.5).sum()), "n_episodes": n_ep, "B": cfg.B,
            "p": np.nan, "t_obs": np.nan, "s_obs": np.nan, "beta": np.nan, "z_approx": np.nan, "sign": 0,
            "resid": None, "pre_slope": np.nan, "status": "undetermined", "reason": ""}

    def undetermined(name, key, reason):
        return dict(base, target=name, key=list(key), reason=reason,
                    claim=CLAIMS_V2["undetermined"].format(reason=reason))

    if (len(rows) < cfg.min_units or n_acc < cfg.min_accept or n_rej < cfg.min_reject
            or n_ep < cfg.min_episodes):
        return [undetermined(nm, k, "support") for nm, k in targets]
    strata = strata_of(data, rows)
    out = [None] * len(targets)
    cols, idx, infos = [], [], []
    for t, (nm, key) in enumerate(targets):
        if key[0] == "pre":
            y, pre = data.pre[tuple(key[1:])][rows], None
        else:
            y, pre = data.y[tuple(key)][rows], data.pre[tuple(key)][rows]
        if not np.all(np.isfinite(y)) or degenerate_columns({"y": y}):
            out[t] = undetermined(nm, key, "degenerate_outcome")
            continue
        r, info = residualise(y, pre, strata)
        cols.append(r)
        idx.append(t)
        infos.append(info)
    if not cols:
        return out
    R = np.column_stack(cols)                          # (n, m), fixed under the sharp null
    v = design_regressor(data, src)
    s_obs = v @ R
    tol = 1e-9 * (1.0 + np.abs(R).sum(0))
    t_obs = np.abs(s_obs)
    uniq, inv = np.unique(src, return_inverse=True)
    sg = data.sgn[src]
    mu = design_mean(data, src)
    pa = PiAssignment(data)
    rng = _stream(cfg, fi, lag, split)
    ch = _chunk(cfg, len(uniq))
    count = np.zeros(len(cols), np.int64)
    s1 = np.zeros(len(cols))
    s2 = np.zeros(len(cols))
    done = 0
    while done < cfg.B:                                # chunks of one stream: identical to one (B, n) draw
        b = min(ch, cfg.B - done)
        M = pa.conditional_draws(rng, family, b, rows=uniq, what="mode")
        V = LEVEL_V2_ARR[M]
        del M
        if len(uniq) != len(src) or np.any(uniq != src):
            V = V[:, inv]
        V -= mu[None, :]
        V *= sg[None, :]
        S = V @ R                                      # (b, m)
        del V
        count += np.count_nonzero(np.abs(S) >= (t_obs - tol)[None, :], axis=0)
        s1 += S.sum(0)
        s2 += (S * S).sum(0)
        done += b
    vv = float(v @ v)
    sd = np.sqrt(np.maximum(s2 / cfg.B - (s1 / cfg.B) ** 2, 0.0))
    for j, t in enumerate(idx):
        nm, key = targets[t]
        beta = float(s_obs[j] / vv) if vv > 0 else 0.0
        out[t] = dict(base, target=nm, key=list(key), status="tested",
                      p=float((1.0 + count[j]) / (cfg.B + 1.0)), t_obs=float(t_obs[j]), s_obs=float(s_obs[j]),
                      beta=beta, sign=int(np.sign(beta)),
                      z_approx=float(s_obs[j] / sd[j]) if sd[j] > 0 else float("nan"),
                      null_mean=float(s1[j] / cfg.B), null_sd=float(sd[j]), chunk=ch, **infos[j])
    return out


def crt_unit_test_v2(data: UnitData, family: str, relation: str, kpi: str, cfg: UnitCRTConfigV2 | None = None,
                     lag: int = 0, split: int = SPLIT_POOLED) -> dict:
    """One hypothesis. Identical p to the batched ``run_crt_units_v2`` (same draws)."""
    return family_tests_v2(data, family, [(f"{relation}_{kpi}", (relation, kpi))], cfg, lag, split)[0]


def _by(results, q=0.05) -> int:
    tested = [r for r in results if r["status"] in ("tested", "declared", "not_detected")]
    if not tested:
        return 0
    dec_ = mscr.by_declare([r["p"] for r in tested], q)
    for r, d in zip(tested, dec_, strict=True):
        r["status"] = "declared" if d else "not_detected"
        r["claim"] = CLAIMS_V2[r["status"]]
    return int(dec_.sum())


def run_crt_units_v2(data: UnitData, cfg: UnitCRTConfigV2 | None = None, split: int = SPLIT_POOLED,
                     audit: bool = False, families=FAMILIES) -> dict:
    """The 60 hypotheses (families x relations x KPIs) + BY at q over the tested ones (+ optional lag-1 audit)."""
    cfg = cfg or UnitCRTConfigV2()
    t0 = time.time()
    res = []
    for f in families:
        tg = [(f"{r}_{k}", (r, k)) for r in data.relations for k in data.kpis]
        for r in family_tests_v2(data, f, tg, cfg, 0, split):
            r["relation"], r["kpi"] = r["key"]
            res.append(r)
    n_dec = _by(res, cfg.q)
    n_tested = sum(r["status"] != "undetermined" for r in res)
    out = {"version": CRT_UNITS_V2_VERSION, "config": dataclasses.asdict(cfg), "split": split,
           "power_floor": power_floor(cfg.B, n_tested, cfg.q),
           "selection": f"BY-across-hypotheses ({CRT_UNITS_V2_VERSION}), q={cfg.q:g}, over the tested of {len(res)}",
           "n_declared": n_dec, "n_tested": n_tested, "n_units": int(data.n), "results": res,
           "wall_s": round(time.time() - t0, 2)}
    if audit:
        out["audit"] = carryover_audit_v2(data, cfg, split, families)
    return out


def carryover_audit_v2(data: UnitData, cfg: UnitCRTConfigV2 | None = None, split: int = SPLIT_POOLED,
                       families=FAMILIES) -> dict:
    """lag-1 (descriptive): the previous (episode, c, xApp, knob) unit's v2 regressor re-drawn against this unit's
    residualised y and its (FE-residualised) pre-window value, ``cfg.audit_B`` draws; BY across the audit's tests."""
    cfg = cfg or UnitCRTConfigV2()
    acfg = dataclasses.replace(cfg, B=int(cfg.audit_B))
    res = []
    for f in families:
        tg = [(f"{r}_{k}", (r, k)) for r in data.relations for k in data.kpis]
        tg += [(f"pre_{r}_{k}", ("pre", r, k)) for r in data.relations for k in data.kpis]
        res.extend(family_tests_v2(data, f, tg, acfg, 1, split))
    n_dec = _by(res, cfg.q)
    tested = [r for r in res if r["status"] != "undetermined"]
    return {"B": acfg.B, "n_declared": n_dec, "n_tested": len(tested),
            "rate_p_le_alpha": (sum(r["p"] <= cfg.alpha for r in tested) / len(tested)) if tested else None,
            "power_floor": power_floor(acfg.B, len(tested), cfg.q),
            "smallest": sorted((float(r["p"]), r["family"], r["target"]) for r in tested)[:8],
            "declared": [(r["family"], r["target"]) for r in res if r["status"] == "declared"],
            "results": res, "wording": "descriptive carryover audit (BY across the audit's own tests); never enters "
                                       "the verdict"}


def edges_from_crt_v2(run: dict) -> dict:
    """{(family, relation, kpi): {"declared", "sign", "score", "status", "p", "beta", "z_approx"}}."""
    out = {}
    for r in run["results"]:
        p = r["p"]
        out[(r["family"], r["relation"], r["kpi"])] = {
            "declared": r["status"] == "declared", "sign": int(r["sign"]), "status": r["status"], "p": p,
            "beta": r["beta"], "z_approx": r["z_approx"],
            "score": float(-np.log10(p)) if np.isfinite(p) else float("nan")}
    return out


def placebo_rejection_units_v2(data: UnitData, cfg: UnitCRTConfigV2 | None = None, binom_level: float = 0.01,
                               max_by: int = 1) -> dict:
    """K0 on PLACEBO records (logged pi0 modes, accept applied: a sharp null for every hypothesis), v1's rule:
    INVALID iff P(Binom(m, alpha) >= n_reject) < binom_level or > max_by BY declarations."""
    from scipy.stats import binom
    cfg = cfg or UnitCRTConfigV2()
    if np.any(data.applied != MODES.index("accept")):
        raise ValueError("placebo records must have applied_mode == accept for every unit")
    run = run_crt_units_v2(data, cfg, split=SPLIT_PLACEBO, audit=False)
    tested = [r for r in run["results"] if r["status"] != "undetermined"]
    m = len(tested)
    rej = [r for r in tested if r["p"] <= cfg.alpha]
    p_count = float(binom.sf(len(rej) - 1, m, cfg.alpha)) if m else float("nan")
    k_max = int(binom.isf(binom_level, m, cfg.alpha)) if m else 0
    return {"version": CRT_UNITS_V2_VERSION, "n_hypotheses": len(run["results"]), "n_tested": m, "alpha": cfg.alpha,
            "rate": len(rej) / m if m else float("nan"), "n_reject": len(rej), "n_by_declared": run["n_declared"],
            "declared": [(r["family"], r["relation"], r["kpi"], r["sign"], r["p"]) for r in run["results"]
                         if r["status"] == "declared"],
            "per_hypothesis": [{"family": r["family"], "relation": r["relation"], "kpi": r["kpi"],
                                "status": r["status"], "p": r["p"], "z_approx": r["z_approx"],
                                "reject": bool(r["status"] != "undetermined" and r["p"] <= cfg.alpha)}
                               for r in run["results"]],
            "p_count": p_count, "k_max": k_max,
            "rule": f"P(Binom(m, {cfg.alpha}) >= n_reject) >= {binom_level} and <= {max_by} BY declaration(s)",
            "pass": bool(m > 0 and p_count >= binom_level and run["n_declared"] <= max_by),
            "binomial_fail_prob_if_uniform_indep": float(binom.sf(k_max, m, cfg.alpha)) if m else None,
            "run": run}


__all__ = ["CLAIMS_V2", "CRT_UNITS_V2_VERSION", "FAMILIES", "KPIS", "LEVEL_V2", "LEVEL_V2_ARR", "V2_STREAM",
           "UnitCRTConfigV2", "carryover_audit_v2", "crt_unit_test_v2", "design_mean", "design_regressor",
           "edges_from_crt_v2", "family_tests_v2", "placebo_rejection_units_v2", "residualise", "run_crt_units_v2",
           "strata_of"]

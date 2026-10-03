"""Shared core of the independence-test adapters (mscr, pcorr, pdcor, rcot2, cmi_knn).

Contract: scratchpad/xmethod/CONTRACT.md; orchestrator decisions of 2026-10-02 (status file xm-citests.md):

- Each candidate (source s -> target y) is tested as ``s _||_ y | Z`` with Z = every OTHER observed source column:
  all action columns (incl. ``P_placebo``), all lagged KPI columns that are not all-NaN, and the observed context
  columns (R3). This is the convention of the original implementations ("condition on all other candidates").
- R-10: the diagnostic ``P_placebo_conf`` (R3 / R4) is scored apart in its own BY family ("placebo_conf"), is
  never used for tuning, and is NOT a conditioner of any other test (so the diagnostic cannot change the primary
  analysis); its own test conditions on all other columns.
- Sources resolve by name: ``action_names`` -> ``X_action`` column, ``kpi_names`` -> ``X_kpi_lag`` column (the
  lagged KPI); targets resolve to ``Y`` columns.
- Primary declaration (methods with p-values): Benjamini-Yekutieli at q = .05 pooled over the action -> KPI
  candidates (incl. ``P_placebo``) of the dataset; the lagged KPI -> KPI candidates are a secondary analysis with
  their own BY family (orchestrator ruling R-6). ``notes["family_of_edge"]`` labels each edge. The method's native rule (per-target BY / BH, largest gap) is kept in ``notes["declared_native"]``.
- Secondary declaration (score rule; R-29 replaces R-2's 2nd-largest): ``tau`` = conformal cutoff at .05 over the
  M pooled ``P_placebo`` scores of the (arm, cell) DEV tune seeds (``tau_from_scores``); declared iff score > tau;
  ``notes["declared_tau"]`` (aligned with ``edges``). Primary declaration stays BY on p-values.
- Every method reports a sign. Unsigned statistics take the sign of the partial correlation of (source, target)
  given the SAME Z (``notes["sign_rule"] = "pcorr_given_Z"``).
- Monte Carlo resolution (orchestrator ruling R-9): every resampling null uses at most B = 9999 draws with
  Besag-Clifford sequential stopping at h = 20 exceedances (p = h / k at stop k, else (1 + count) / (B + 1)).
  The null of each method is unchanged; native-B (no stopping) runs are kept only as a fidelity-gate check.
- Method RNG streams derive from tag 7802 (CONTRACT sec 6) and the dataset seed, never from EVAL seeds.
- Arms (rulings R-17 / R-18 / R-25, config ``arm``): "eq" (PRIMARY, equal information): an action source's WHOLE
  conditioning set is ``design_covariates(data, focal=<action>)`` (shared helper ``cdd_oran.xmethod.covariates``,
  concurrent="designed" as pmrt_core: lagged KPIs, context, setpoints, all actions at t-1 / t-2, the other
  designed actions at t). Helper columns that are R-3 columns (``concurrent:<P>``, ``lag_kpi:<K>``, ``ctx:<c>``) map
  onto them (values checked); the rest are appended once; rows with row_mask False are dropped; columns constant on
  the kept rows are left out (``notes["eq_dropped"]``). A lagged-KPI source (KPI -> KPI, secondary; the helper's
  focal is an action) conditions on the base helper set without its own column plus every action at t
  (concurrent "all", as R-3). Columns of ``P_placebo_conf`` (by name: ``sp:``, ``@t-L``, ``concurrent:``, and their
  ``:missing``) are conditioners of its own test only (R-10). "native" (SECONDARY, "as typically applied,
  design-blind") = the R-3 set (= helper with concurrent="all", no setpoints / lagged actions; equality tested).
  ``native_config()`` of every adapter (the fidelity gates) is arm "native". "eq_min" (R-33, secondary): the R-3
  set + the focal action's own setpoint ``sp:<focal>`` only (lagged-KPI sources: native set); like eq, taken
  from ``_classic_common.cond_set(data, family, src, arm)`` (which also applies R-37), so citests == classic.
- R-22 degenerate candidates (notes["not_testable"] = {"src->tgt": reason}, the score.py format): when the
  target is an EXACT fit of the test's conditioning information (source + Z;
  OLS residual variance / target variance < ``EXACT_FIT_RR`` = 1e-8; the deterministic linear worlds E1 / E3 give
  ~1e-29 in BOTH arms, the noisy worlds >= .08), the candidate is not tested: score NaN, p None, sign 0,
  declared False, ``notes["not_testable"]`` (+ ``notes["not_testable_edges"]``); it is left out of the BY
  families and the adapter never sees it. Linear check only (a nonlinear exact fit would not be detected).
"""
from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np

from cdd_oran.discovery.mscr import by_declare
from cdd_oran.xmethod import api

COVARIATES_HELPER = "cdd_oran.xmethod.covariates"

RNG_TAG = 7802
B_MAX = 9999            # R-9
BC_H = 20               # R-9 Besag-Clifford exceedance count
PLACEBO = "P_placebo"
PLACEBO_CONF = "P_placebo_conf"   # R-10 diagnostic negative control (R3 / R4): own family, not a conditioner
Q = 0.05
ARMS = ("eq", "native", "eq_min")
ARM_DEFAULT = "eq"      # R-17: equal information is the primary arm
EXACT_FIT_RR = 1e-8     # R-22 threshold on residual variance / target variance
NOT_TESTABLE = "exact fit: deterministic world under Z"
PER_EDGE_NOTES = ("mc_draws", "signed_pdcor")   # adapter notes aligned with the tested candidates


def method_seed(data_seed: int, method_key: int) -> int:
    """Deterministic 32-bit seed for one (dataset, adapter) pair from the citests RNG tag."""
    ss = np.random.SeedSequence(entropy=[RNG_TAG, int(data_seed), int(method_key)])
    return int(ss.generate_state(1)[0])


class SequentialP:
    """Besag-Clifford (1991) sequential Monte Carlo p-value: feed null draws in order with ``add``; ``done`` turns
    True after h exceedances (null >= observed). ``h=None`` = fixed-B Monte Carlo p (the native rule)."""

    def __init__(self, obs: float, b_max: int, h: int | None):
        self.obs, self.b_max, self.h = obs, int(b_max), h
        self.k = 0
        self.count = 0

    def add(self, null_value: float) -> bool:
        self.k += 1
        if null_value >= self.obs:
            self.count += 1
        return self.done

    @property
    def done(self) -> bool:
        return (self.h is not None and self.count >= self.h) or self.k >= self.b_max

    @property
    def p(self) -> float:
        if self.h is not None and self.count >= self.h:
            return self.h / self.k
        return (1.0 + self.count) / (self.b_max + 1.0)


def sequential_p_from_draws(obs: float, null: np.ndarray, h: int | None) -> tuple[float, int]:
    """The Besag-Clifford p of an ordered array of null draws (identical to feeding them to ``SequentialP``)."""
    b = len(null)
    exceed = np.cumsum(np.asarray(null) >= obs)
    if h is not None and exceed[-1] >= h:
        k = int(np.searchsorted(exceed, h)) + 1
        return h / k, k
    return (1.0 + float(exceed[-1])) / (b + 1.0), b


@dataclasses.dataclass(frozen=True)
class Prepared:
    """Column layout of one dataset for the CI tests."""
    S: np.ndarray                   # [n, d] all observed source columns (actions, lagged KPIs, context)
    S_names: tuple[str, ...]        # their names (context columns prefixed "ctx:")
    Y: np.ndarray                   # [n, k] targets
    pairs: tuple[tuple[int, int], ...]   # per candidate (source column i in S, target column j in Y)
    n_ctx: int
    family: tuple[str, ...] = ()    # per candidate: "action_kpi" (primary) | "kpi_kpi" | "placebo_conf"
    exclude_z: tuple[int, ...] = ()  # diagnostic columns of S (P_placebo_conf and, arm eq, its covariates): never
    #                                  conditioners of OTHER sources; a diagnostic source conditions on everything
    arm: str = "native"
    row_mask: np.ndarray | None = None   # arm eq: rows kept (S, Y are already restricted to them)
    eq_dropped: tuple[str, ...] = ()    # arm eq: helper columns left out (constant on the kept rows)
    z_map: dict[int, tuple[int, ...]] | None = None   # arm eq: source column -> its conditioning columns (R-25)

    def z_cols(self, i: int) -> list[int]:
        """Conditioning columns for source column i. Arm native: every other column, minus the diagnostic ones
        unless i is itself diagnostic. Arm eq: the helper's set for that source."""
        if self.z_map is not None:
            return list(self.z_map[i])
        return z_cols_of(self.S.shape[1], i, self.exclude_z)

    def column_groups(self) -> list[tuple[list[int], list[int]]]:
        """For methods that condition on "all other columns of their matrix": (columns, candidate indices) groups
        such that each candidate's source is in ``columns`` and its Z = the rest of ``columns`` = ``z_cols``."""
        if self.z_map is not None:
            by: dict[tuple[int, ...], list[int]] = {}
            for e, (i, _) in enumerate(self.pairs):
                by.setdefault(tuple(sorted(set(self.z_map[i]) | {i})), []).append(e)
            return [(list(k), es) for k, es in by.items()]
        base = [c for c in range(self.S.shape[1]) if c not in self.exclude_z]
        groups = [(base, [e for e, (i, _) in enumerate(self.pairs) if i not in self.exclude_z])]
        for x in self.exclude_z:
            groups.append((list(range(self.S.shape[1])), [e for e, (i, _) in enumerate(self.pairs) if i == x]))
        return [g for g in groups if g[1]]

    def z_of(self, a: np.ndarray, i: int) -> np.ndarray:
        """``a[:, z_cols(i)]`` for a matrix laid out like S (e.g. its standardized copy)."""
        return a[:, self.z_cols(i)]


def z_cols_of(d: int, i: int, exclude_z) -> list[int]:
    return [c for c in range(d) if c != i and (c not in exclude_z or i in exclude_z)]


def prepare(data: api.Dataset, arm: str = "native") -> Prepared:
    if arm not in ARMS:
        raise ValueError(f"citests: arm must be one of {ARMS}, got {arm!r}")
    cols, names = [data.X_action], list(data.action_names)
    lag_idx: dict[str, int] = {}
    if data.X_kpi_lag is not None and data.X_kpi_lag.size:
        keep = [j for j in range(data.X_kpi_lag.shape[1]) if not np.isnan(data.X_kpi_lag[:, j]).all()]
        for j in keep:
            lag_idx[data.kpi_names[j]] = len(names) + len(lag_idx)
        cols.append(data.X_kpi_lag[:, keep])
        names += [data.kpi_names[j] for j in keep]
    n_ctx = 0
    if data.context is not None:
        ctx = np.asarray(data.context, dtype=np.float64).reshape(data.n, -1)
        cnames = data.meta.get("context_names", tuple(f"c{i}" for i in range(ctx.shape[1])))
        cols.append(ctx)
        names += [f"ctx:{c}" for c in cnames]
        n_ctx = ctx.shape[1]
    S = np.concatenate([np.asarray(c, dtype=np.float64).reshape(data.n, -1) for c in cols], axis=1)
    Y = np.asarray(data.Y, dtype=np.float64)
    if not (np.isfinite(S).all() and np.isfinite(Y).all()):
        raise ValueError("citests: non-finite values in sources or targets")
    act_idx = {a: i for i, a in enumerate(data.action_names)}
    tgt_idx = {k: j for j, k in enumerate(data.kpi_names)}
    pairs = []
    for s, t in data.candidates:
        if s in act_idx:
            i = act_idx[s]
        elif s in lag_idx:
            i = lag_idx[s]
        else:
            raise ValueError(f"citests: candidate source {s!r} is neither an action nor a non-NaN lagged KPI")
        pairs.append((i, tgt_idx[t]))
    fam = tuple("placebo_conf" if s == PLACEBO_CONF else "action_kpi" if s in act_idx else "kpi_kpi"
                for s, _ in data.candidates)
    excl = [act_idx[a] for a in data.action_names if a == PLACEBO_CONF]
    if arm == "native":
        return Prepared(S=S, S_names=tuple(names), Y=Y, pairs=tuple(pairs), n_ctx=n_ctx, family=fam,
                        exclude_z=tuple(excl))
    return _prepare_eq(data, S, names, Y, pairs, fam, n_ctx, act_idx, lag_idx, excl, arm)


def _is_conf_column(name: str) -> bool:
    """R-10: helper columns of the diagnostic P_placebo_conf (sp:, @t-L, concurrent:, and their :missing)."""
    base = name.removesuffix(":missing")
    for pre in ("sp:", "concurrent:"):
        base = base.removeprefix(pre)
    return base.split("@t-")[0] == PLACEBO_CONF


def _prepare_eq(data, S, names, Y, pairs, fam, n_ctx, act_idx, lag_idx, excl, arm="eq") -> Prepared:
    """Arms eq (R-17 / R-25 / R-28) and eq_min (R-33): per-source conditioning sets from the shared rule
    ``_classic_common.cond_set(data, family, src, arm)`` (module docstring), mapped onto the S columns."""
    col_of = {f"concurrent:{a}": i for a, i in act_idx.items()}
    col_of.update({f"lag_kpi:{k}": i for k, i in lag_idx.items()})
    col_of.update({nm: c for c, nm in enumerate(names) if nm.startswith("ctx:")})
    extra: list[np.ndarray] = []
    state: dict[str, np.ndarray] = {}

    def mapped(cn, C, m) -> list[int]:
        C = np.asarray(C, dtype=np.float64).reshape(data.n, -1)
        m = np.asarray(m, dtype=bool)
        if len(cn) != C.shape[1] or m.shape != (data.n,):
            raise ValueError("citests: design_covariates returned inconsistent shapes")
        if "mask" not in state:
            state["mask"] = m
        elif not np.array_equal(state["mask"], m):
            raise ValueError("citests: design_covariates row_mask depends on the focal action")
        out = []
        for k, nm in enumerate(cn):
            v = C[:, k]
            if nm in col_of:
                c = col_of[nm]
                ref = S[:, c] if c < S.shape[1] else extra[c - S.shape[1]]
                if not np.array_equal(v, ref):
                    raise ValueError(f"citests: helper column {nm!r} differs from the dataset column")
            else:
                col_of[nm] = c = S.shape[1] + len(extra)
                extra.append(v)
                names.append(nm)
            out.append(c)
        return out

    from cdd_oran.xmethod.methods._classic_common import (
        cond_set,  # R-25 / R-28: the audited shared rule
    )

    z_map: dict[int, list[int]] = {}
    inv_act = {i: a for a, i in act_idx.items()}
    inv_lag = {i: k for k, i in lag_idx.items()}
    conf_cols = set(excl)
    for i in sorted({i for i, _ in pairs}):
        if i in inv_act:
            cs = cond_set(data, "action", data.action_names.index(inv_act[i]), arm)
            diag = inv_act[i] == PLACEBO_CONF
        else:                                   # R-28: base set - own lag_kpi + every action at t
            cs = cond_set(data, "kpi", data.kpi_names.index(inv_lag[i]), arm)
            diag = False
        z_map[i] = (mapped(cs.names, cs.Z, cs.mask), diag)
    conf_cols |= {c for nm, c in col_of.items() if _is_conf_column(nm)}
    zl = {i: [c for c in dict.fromkeys(z) if c != i and (diag or c not in conf_cols)] for i, (z, diag) in z_map.items()}
    mask = state["mask"]
    S = np.concatenate([S] + [v[:, None] for v in extra], axis=1)[mask]
    Y = Y[mask]
    if not np.isfinite(S).all():
        raise ValueError("citests: non-finite design covariates on kept rows")
    const = {c for c in range(S.shape[1]) if np.ptp(S[:, c]) == 0.0}
    dropped = sorted({names[c] for z in zl.values() for c in z if c in const})
    zm = {i: tuple(c for c in z if c not in const) for i, z in zl.items()}
    return Prepared(S=S, S_names=tuple(names), Y=Y, pairs=tuple(pairs), n_ctx=n_ctx, family=fam,
                    exclude_z=tuple(sorted(conf_cols)), arm=arm, row_mask=mask, eq_dropped=tuple(dropped),
                    z_map=zm)


def standardize(a: np.ndarray) -> np.ndarray:
    """z-standardize columns (population std); zero-variance columns are left centred (std := 1)."""
    sd = a.std(axis=0)
    return (a - a.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def pcorr_given_z(S: np.ndarray, Y: np.ndarray, pairs, exclude_z=(), z_cols=None) -> tuple[np.ndarray, np.ndarray]:
    """Partial correlation of S[:, i] and Y[:, j] given all OTHER columns of S except ``exclude_z`` (+ intercept),
    or given ``z_cols(i)`` when that callable is passed (``Prepared.z_cols``, arm eq).

    Returns (rho, df) with df = n - 2 - |Z|. Residuals by least squares, as in
    ``e1slice.discovery_v2.partial_correlation_scores`` (whose STOP branches are not applied here: a vanishing
    residual gives rho = 0, i.e. no evidence and sign 0).
    """
    n, d = S.shape
    rho = np.zeros(len(pairs))
    df = np.zeros(len(pairs))
    by_i: dict[int, list[int]] = {}
    for e, (i, _) in enumerate(pairs):
        by_i.setdefault(i, []).append(e)
    for i, es in by_i.items():
        zc = z_cols(i) if z_cols is not None else z_cols_of(d, i, exclude_z)
        others = np.concatenate([S[:, zc], np.ones((n, 1))], axis=1)
        js = sorted({pairs[e][1] for e in es})
        rhs = np.concatenate([S[:, [i]], Y[:, js]], axis=1)
        sol, *_ = np.linalg.lstsq(others, rhs, rcond=None)
        res = rhs - others @ sol
        rx = res[:, 0]
        for e in es:
            df[e] = n - 2 - len(zc)
            ry = res[:, 1 + js.index(pairs[e][1])]
            den = np.sqrt(rx @ rx) * np.sqrt(ry @ ry)
            vx, vy = rx @ rx / n, ry @ ry / n
            rho[e] = 0.0 if (den <= 0 or vx < 1e-12 or vy < 1e-12) else float(rx @ ry / den)
    return rho, df


def exact_fit(prep: Prepared) -> tuple[np.ndarray, np.ndarray]:
    """R-22: per candidate, OLS residual variance of the target on [1, source, Z] over its variance; and the mask
    rr < EXACT_FIT_RR (exact fit -> not testable)."""
    S, Y = standardize(prep.S), standardize(prep.Y)
    n = S.shape[0]
    rr = np.ones(len(prep.pairs))
    by_i: dict[int, list[int]] = {}
    for e, (i, _) in enumerate(prep.pairs):
        by_i.setdefault(i, []).append(e)
    for i, es in by_i.items():
        X = np.concatenate([np.ones((n, 1)), S[:, [i]], S[:, prep.z_cols(i)]], axis=1)
        js = sorted({prep.pairs[e][1] for e in es})
        sol, *_ = np.linalg.lstsq(X, Y[:, js], rcond=None)
        res = Y[:, js] - X @ sol
        tot = ((Y[:, js] - Y[:, js].mean(axis=0)) ** 2).sum(axis=0)
        for e in es:
            k = js.index(prep.pairs[e][1])
            rr[e] = float(res[:, k] @ res[:, k] / tot[k]) if tot[k] > 0 else 0.0
    return rr, rr < EXACT_FIT_RR


def by_pooled(p: np.ndarray, q: float = Q) -> np.ndarray:
    """BY at q pooled over all candidates (``mscr.by_declare``); NaN p (not scorable) never declared."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(len(p), dtype=bool)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = by_declare(p[ok], q)
    return out


def by_families(p: np.ndarray, family: tuple[str, ...], q: float = Q) -> np.ndarray:
    """Separate pooled-BY families (R-6): action -> KPI (primary) and KPI -> KPI (secondary)."""
    p = np.asarray(p, dtype=float)
    out = np.zeros(len(p), dtype=bool)
    fam = np.asarray(family)
    for f in set(family):
        idx = np.nonzero(fam == f)[0]
        out[idx] = by_pooled(p[idx], q)
    return out


def tau_from_scores(placebo_scores: list[float] | np.ndarray, alpha: float = 0.05) -> float:
    """R-29 conformal tau of a list of placebo scores: delegates to the ONE shared implementation
    ``score.placebo_tau`` (audit L1) by wrapping the scores as P_placebo edges of a dummy Result."""
    from cdd_oran.xmethod import score

    edges = tuple(api.EdgeResult(source=PLACEBO, target=f"k{i}", score=float(v), p=None, sign=0, declared=False)
                  for i, v in enumerate(placebo_scores) if v is not None)
    dummy = api.Result(method="tau", version="0", edges=edges, cpu_s=0.0, config={}, notes={})
    return float(score.placebo_tau([dummy], alpha=alpha))


def placebo_scores(result: api.Result) -> list[float]:
    return [e.score for e in result.edges if e.source == PLACEBO]


def tau_from_results(results: list[api.Result]) -> dict[str, Any]:
    """tau over one (arm, world, regime, n) cell's DEV results (truth-free): ``score.placebo_tau`` (R-29)."""
    from cdd_oran.xmethod import score

    sc = [v for r in results for v in placebo_scores(r)]
    return {"tau": float(score.placebo_tau(results, alpha=score.TAU_ALPHA)), "n_placebo_scores": len(sc),
            "tau_rule": score.tau_rule_name(score.TAU_ALPHA)}


def progress(cfg: dict[str, Any], done: int, total: int) -> None:
    """Optional cost logging (``cfg["progress_file"]``): one line "done total process_cpu_s" per finished edge, so a
    run stopped at the compute budget (R-13) still reports how far it got. Never affects results."""
    path = cfg.get("progress_file")
    if path:
        with open(path, "a") as f:
            f.write(f"{done} {total} {time.process_time():.3f}" + chr(10))


def cell_key(d: api.Dataset) -> tuple:
    return (d.world, d.regime, d.n, d.meta.get("lam"))


class CITestBase:
    """api.Method skeleton: subclasses implement ``_test(prep, data, cfg) -> dict`` returning arrays aligned with
    ``prep.pairs``: ``score``, ``p`` (or None), and optionally ``sign`` (native signed statistic),
    ``declared_native``, ``not_testable`` ({local index: reason}, e.g. a numerical guard; audit L2) and ``notes``.
    A ValueError from ``_test`` is a recorded STOP: every candidate it was given is listed as not testable with the
    STOP reason (audit L3)."""
    name = "citest"
    version = "0"
    method_key = 0
    native_sign = False

    def default_config(self) -> dict[str, Any]:
        return {}

    def native_config(self) -> dict[str, Any]:
        return {**self.default_config(), "arm": "native"}

    def tune(self, dev: list[api.Dataset], truth_free: bool = True,
             base_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """tau for one DEV cell; ``base_config`` selects the arm / null to tune (default: the primary config)."""
        if not truth_free:
            raise ValueError("citests: tuning is truth-free only (CONTRACT sec 5)")
        if len({cell_key(d) for d in dev}) != 1:
            raise ValueError("citests: tune() takes the DEV datasets of exactly one (world, regime, n) cell")
        cfg = {"arm": ARM_DEFAULT, **self.default_config(), **(base_config or {})}
        res = [self.run(d, cfg) for d in dev]
        out = dict(cfg)
        out.update(tau_from_results(res))
        out["tau_cell"] = list(cell_key(dev[0]))
        out["tau_dev_seeds"] = [d.seed for d in dev]
        return out

    def _test(self, prep: Prepared, data: api.Dataset, cfg: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def run(self, data: api.Dataset, config: dict[str, Any]) -> api.Result:
        cfg = {"arm": ARM_DEFAULT, **self.default_config(), **(config or {})}
        t0 = time.process_time()
        prep = prepare(data, cfg["arm"])
        notes: dict[str, Any] = {"arm": prep.arm, "conditioning": list(prep.S_names), "q": Q,
                                 "family": "pooled BY per family: action_kpi (primary), kpi_kpi (secondary), "
                                           "placebo_conf (R-10 diagnostic)",
                                 "family_of_edge": prep.family}
        if prep.arm in ("eq", "eq_min"):
            notes.update(covariates_helper=COVARIATES_HELPER, eq_dropped=list(prep.eq_dropped),
                         n_rows_used=int(prep.S.shape[0]), n_rows_dropped=int(data.n - prep.S.shape[0]))
        m_all = len(prep.pairs)
        rr, degenerate = exact_fit(prep)
        keep = np.nonzero(~degenerate)[0]
        nt: dict[int, str] = {int(e): NOT_TESTABLE for e in np.nonzero(degenerate)[0]}   # global index -> reason
        if degenerate.any():
            notes["exact_fit_rr_max"] = float(rr[degenerate].max())
        notes["exact_fit_rr_min_testable"] = float(rr[keep].min()) if len(keep) else None
        sub = dataclasses.replace(prep, pairs=tuple(prep.pairs[e] for e in keep),
                                  family=tuple(prep.family[e] for e in keep))
        if not len(keep):
            out = {"score": np.zeros(0), "p": np.zeros(0)}
        else:
            try:
                out = self._test(sub, data, cfg)
            except ValueError as e:      # a recorded STOP of the wrapped numerics: nothing declared (audit L3)
                out = {"score": np.full(len(keep), np.nan), "p": np.full(len(keep), np.nan),
                       "notes": {"stop": str(e)}, "not_testable": {k: f"STOP: {e}" for k in range(len(keep))}}
        for k, why in (out.get("not_testable") or {}).items():
            nt[int(keep[int(k)])] = str(why)
        dead = np.zeros(m_all, dtype=bool)
        dead[list(nt)] = True
        if nt:                                                  # score.py format + list (R-22 / R-23)
            notes["not_testable"] = {"{}->{}".format(*data.candidates[e]): why for e, why in sorted(nt.items())}
            notes["not_testable_edges"] = [list(data.candidates[e]) for e in sorted(nt)]
        score = np.full(m_all, np.nan)
        score[keep] = np.asarray(out["score"], dtype=float)
        score[dead] = np.nan
        p = None
        if out.get("p") is not None:
            p = np.full(m_all, np.nan)
            p[keep] = np.asarray(out["p"], dtype=float)
            p[dead] = np.nan                                    # never inside a BY family
        sign = np.zeros(m_all, dtype=int)
        if self.native_sign and out.get("sign") is not None:
            sign[keep] = np.sign(np.asarray(out["sign"], dtype=float)).astype(int)
            notes["sign_rule"] = "native"
        else:
            rho, _ = pcorr_given_z(standardize(sub.S), standardize(sub.Y), sub.pairs, z_cols=sub.z_cols)
            sign[keep] = np.sign(rho).astype(int)
            notes["sign_rule"] = "pcorr_given_Z"
        sign[dead] = 0
        declared = by_families(p, prep.family) if p is not None else np.zeros(m_all, dtype=bool)
        declared[dead] = False
        tau = cfg.get("tau")
        if tau is not None:
            notes["declared_tau"] = tuple(bool(np.isfinite(s) and s > tau) for s in score)
            if p is None:
                declared = np.array(notes["declared_tau"], dtype=bool)
        if out.get("declared_native") is not None:
            dn = np.zeros(m_all, dtype=bool)
            dn[keep] = np.asarray(out["declared_native"], dtype=bool)
            notes["declared_native"] = tuple(bool(v) for v in dn)
        notes["family_of_edge"] = prep.family
        for k, v in out.get("notes", {}).items():
            if k in PER_EDGE_NOTES and len(keep) < m_all:      # re-align with edges (untested -> None)
                full: list[Any] = [None] * m_all
                for e, x in zip(keep, v, strict=True):
                    full[e] = x
                v = full
            notes[k] = v
        edges = tuple(
            api.EdgeResult(source=s, target=t, score=float(score[e]),
                           p=None if p is None or not np.isfinite(p[e]) else float(p[e]),
                           sign=int(sign[e]), declared=bool(declared[e]))
            for e, (s, t) in enumerate(data.candidates))
        cpu = time.process_time() - t0
        return api.Result(method=self.name, version=self.version, edges=edges, cpu_s=float(cpu),
                          config={k: v for k, v in cfg.items()}, notes=notes)

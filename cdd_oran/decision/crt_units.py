"""mscr-crt-units-v1: unit-level conditional randomization test (CRT) for the E6-P discovery study
(scratchpad/e6_dev/decision/STEP1_MSCR_PLAN.md, "Methods / Primary"; protocol docs/benchmark/E6P_DISCOVERY_PROTOCOL.md).

Input: episode records of scratchpad/e6_dev/e6p_discovery.py (schema "e6p-disc-rec/1"; its module docstring is the
record contract). Read: ``units`` (c, x, knob, t0, mode, p, applied_mode, step, exp, ctx), ``pi0_table``,
``lab_series`` (the unit OUTCOMES), ``n_cells``, ``seed``, ``fold``, ``stage``. NEVER read: ``gt_static``,
``gt_labels``, ``lab_outcome`` (tests poison them).

Unit table (``build_unit_data``). One row per opened unit of a knob family f in ``FAMILIES`` whose windows fit in the
series (t0 - H_pre >= 0 and t0 + H <= len(series); others are dropped and counted):
  target      y(rel, kpi) = sum over the relation's cells of [KPI over rows t0 .. t0+H-1 (= env seconds t0+1 .. t0+H)
              minus KPI over rows t0-H_pre .. t0-1 (seconds t0-H_pre+1 .. t0)], H = H_pre = 90; KPIs ``KPIS`` = pv, v,
              e, rlf, load (= the series field "ue", served UE-s, as gt_p's "load");
  relations   own = {c}, nbr = N(c) - {c}, far = cells outside N(c) (``gt_p.relation_cells``; N(c) = the unit's logged
              obs-only exposure set);
  treatment   level = accept 1 / half 0.5 / reject 0 (``LEVEL``; the LOGGED pi0 mode, which on placebo episodes is
              NOT the applied one), oriented "dir" like gt_p: x = level * sgn, sgn = sign(prop - cur) of the unit's
              opening request (0 -> +1). A positive dependence of y on x = the effect of a knob-value INCREASE;
  conditioners the unit's numeric obs-only ctx (``units_p.UnitArbiter.context``: request cur / prop / step,
              mediators of c and aggregates over N(c) - c, since_change, n_exp, is_macro) plus the pre-window value of
              EVERY (relation, KPI) (15 columns). All pre-assignment. One conditioner set per family (shared by its 15
              hypotheses; the batched null below relies on it). NaN -> column median, degenerate columns dropped
              (``crt._conditioners``).

Per-unit rows (option (a), 2026-09-30): a unit that carries ``probs`` (its logged mode-probability row, e.g.
``collect_p.IncumbentPolicy``: context-dependent, drawn independently per unit from a keyed uniform) gets THAT row as
its ``probs`` (preferred over ``pi0_table[x]``; the p / row consistency check ``p_mismatch`` is unchanged). The
conditional draws below re-draw each unit from its own row, which is the exact conditional law under the family's
sharp null as long as the row is a function of pre-assignment obs only (then the trajectory, contexts and rows are
invariant to the family's assignments).

Assignment model (``PiAssignment``): pi0 is context-free and draws every unit independently (collect_p: one uniform
per unit, key [seed, 6612, c, x_idx, t0]). Conditional draws for family f: every f unit's mode is re-drawn from the
LOGGED table of its xApp (``pi0_table[x]``), every other family's modes are held fixed (they are not in the
statistic), outcomes are held fixed (true under the sharp null). Under the sharp null of no effect of family f's
assignments on the whole trajectory the skeleton (which units open, when, where) is invariant, so these draws are the
exact conditional law and p = (1 + #{T_draw >= T_obs}) / (B + 1) is finite-sample valid for ANY statistic T.
RNG: ``default_rng([seed, 6616, family_idx, lag, split])`` (tag 6616, registered); the row order of family f (MSCR
breaks ties by row order) is a seeded permutation ``default_rng([seed, 6616, 99, family_idx, lag, split])``.

B: 9999 (``UnitCRTConfig``). The plan's B = 999 gives a p floor 1e-3 above BY's rank-1 threshold over 60 hypotheses
(q / (m c_m) = 1.78e-4): no hypothesis is declarable unless >= 6 sit at the floor (``power_floor``); the minimum B
admitting a rank-1 declaration at m = 60 is 5615. Draws are generated and consumed in chunks of the same stream
(``chunk``; identical p-values, bounded memory).

Statistics (as crt.py, mscr-crt-v1): primary MSCR S* (``crt.MSCRStat``; observed via the frozen ``mscr._s_star``,
null draws via its vectorised replica, fidelity checked per hypothesis with ``crt.check_draws``) with CRTConfig
nc = 3, nb = 3, min_stratum = 10; secondary ``crt.signed_stat`` (episode-cluster |mean_e mean_u x (y - ybar_e)|).
The 15 hypotheses of a family share their conditioners and draws, so their null draws are computed in one pass
(``_joint_draws``: per conditioner stratum the argsort of the draws is computed once and applied to each target;
the arithmetic per target is exactly ``MSCRStat.draws``, asserted on the first draws of every run).
Sign / effect: beta = within-episode regression slope of y on x (sum xc yc / sum xc^2, xc, yc centred per episode);
sign = sign(beta). Declarations: BY at q over the TESTED hypotheses of one call (statuses declared | not_detected |
undetermined, strict ``crt.CLAIMS`` wording). Support rule (``UnitCRTConfig``): >= min_units units, >= min_accept
accepted and >= min_reject rejected units, >= min_episodes episodes; else undetermined.

Carryover audit (``lag=1``): the previous unit of the same (episode, cell, xApp) (same knob family) re-drawn against
this unit's outcome y and its pre-window value (baseline contamination), conditioners = the previous unit's
pre-assignment values; BY across the audit's own tests. Descriptive only (never enters the verdict).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from cdd_oran.discovery import mscr

from .collect_p import PI0_ORDER, dec
from .crt import CLAIMS, CRTConfig, MSCRStat, _conditioners, check_draws, signed_stat
from .features import degenerate_columns
from .gt_p import FAMILIES, RELATIONS, relation_cells

CRT_UNITS_VERSION = "mscr-crt-units-v1"
CRT_TAG = 6616
KPIS = ("pv", "v", "e", "rlf", "load")
SERIES_OF = {"pv": "pv", "v": "v", "e": "e", "rlf": "rlf", "load": "ue"}
MODES = PI0_ORDER                                   # accept, half, reject, accept+rb
LEVEL = {"accept": 1.0, "half": 0.5, "reject": 0.0, "accept+rb": 1.0}
LEVEL_ARR = np.array([LEVEL[m] for m in MODES])
H_UNIT = 90
HYPOTHESES = tuple((f, r, k) for f in FAMILIES for r in RELATIONS for k in KPIS)
SPLIT_POOLED, SPLIT_PLACEBO = 0, 9                  # split ids in the RNG key (folds: 1 + fold)
PRIVILEGED_KEYS = ("gt_static", "gt_labels", "lab_outcome")


@dataclasses.dataclass(frozen=True)
class UnitCRTConfig:
    """B: the plan's 999 cannot reach BY's rank-1 threshold over 60 hypotheses (q / (m c_m) = 1.78e-4 < 1 / 1000):
    nothing is declarable unless >= 6 hypotheses sit at the p floor (``power_floor``). Protocol draft: B = 9999."""
    B: int = 9999
    alpha: float = 0.05
    q: float = 0.05
    chunk: int = 1000                  # draws per pass (memory bound; the stream is chunk-invariant)
    audit_B: int = 999                 # lag-1 carryover audit (descriptive)
    nc: int = 3
    nb: int = 3
    min_stratum: int = 10
    min_units: int = 30
    min_accept: int = 5
    min_reject: int = 5
    min_episodes: int = 3
    seed: int = 0

    def crt_config(self) -> CRTConfig:
        return CRTConfig(nc=self.nc, nb=self.nb, min_stratum=self.min_stratum, B=self.B, alpha=self.alpha, q=self.q)


# ------------------------------------------------------------------------------------------------ unit table
@dataclasses.dataclass
class UnitData:
    """Pooled unit table (one row per unit); see the module docstring."""
    episode: np.ndarray        # (n,) episode index (position in the records passed)
    seed: np.ndarray           # (n,) protocol seed of the episode
    fold: np.ndarray           # (n,) EVAL fold (-1: none)
    family: np.ndarray         # (n,) index into FAMILIES
    xapp: np.ndarray           # (n,) object
    c: np.ndarray              # (n,) cell
    t0: np.ndarray             # (n,) int second
    mode: np.ndarray           # (n,) index into MODES (LOGGED pi0 draw)
    applied: np.ndarray        # (n,) index into MODES (what the arbiter applied)
    p: np.ndarray              # (n,) logged propensity
    probs: np.ndarray          # (n, len(MODES)) the logged pi0 table of the unit's xApp
    step: np.ndarray           # (n,) prop - cur of the opening request
    sgn: np.ndarray            # (n,) sign(step), 0 -> +1
    y: dict                    # (rel, kpi) -> (n,) post - pre
    pre: dict                  # (rel, kpi) -> (n,) pre-window sum
    ycell: dict                # kpi -> (n, C) per-cell post - pre
    rel_mask: dict             # rel -> (n, C) bool
    z: dict                    # conditioner name -> (n,)
    prev: np.ndarray           # (n,) row of the previous unit of the same (episode, c, xApp, knob), -1 if none
    n_cells: int
    H: int
    H_pre: int
    relations: tuple
    kpis: tuple
    meta: dict

    @property
    def n(self) -> int:
        return len(self.t0)

    @property
    def level(self) -> np.ndarray:
        return LEVEL_ARR[self.mode]

    @property
    def x(self) -> np.ndarray:
        return self.level * self.sgn

    def rows_of(self, family: str) -> np.ndarray:
        return np.nonzero(self.family == FAMILIES.index(family))[0]

    def subset(self, rows) -> UnitData:
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.nonzero(rows)[0]
        pos = np.full(self.n, -1)
        pos[rows] = np.arange(len(rows))
        prev = self.prev[rows]
        prev = np.where(prev >= 0, pos[np.maximum(prev, 0)], -1)
        pick = {f.name: getattr(self, f.name)[rows] for f in dataclasses.fields(self)
                if isinstance(getattr(self, f.name), np.ndarray) and f.name != "prev"}
        return dataclasses.replace(self, **pick, prev=prev, y={k: v[rows] for k, v in self.y.items()},
                                   pre={k: v[rows] for k, v in self.pre.items()},
                                   ycell={k: v[rows] for k, v in self.ycell.items()},
                                   rel_mask={k: v[rows] for k, v in self.rel_mask.items()},
                                   z={k: v[rows] for k, v in self.z.items()}, meta=dict(self.meta))


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def build_unit_data(records, H: int = H_UNIT, relations=RELATIONS, kpis=KPIS, H_pre: int | None = None) -> UnitData:
    """Unit table of episode records (module docstring). ``records``: parsed "episode" JSONL records (dicts)."""
    H = int(H)
    H_pre = H if H_pre is None else int(H_pre)
    relations, kpis = tuple(relations), tuple(kpis)
    cols = {k: [] for k in ("episode", "seed", "fold", "family", "xapp", "c", "t0", "mode", "applied", "p", "probs",
                            "step", "sgn", "prev")}
    ycell, precell, masks, ctxs = {k: [] for k in kpis}, {k: [] for k in kpis}, {r: [] for r in relations}, []
    n_cells = None
    dropped = {"window": 0, "family": 0}
    p_mismatch, n_rows, n_probs_rows = 0, 0, 0
    eps = []
    for e, rec in enumerate(records):
        nc = int(rec["n_cells"])
        if n_cells is None:
            n_cells = nc
        elif nc != n_cells:
            raise ValueError("episodes with different cell counts cannot be pooled")
        ls = rec["lab_series"]
        D = dec(ls["data"]).astype(np.float64)                       # (T, F, C)
        fidx = [list(ls["fields"]).index(SERIES_OF[k]) for k in kpis]
        T = D.shape[0]
        table = rec.get("pi0_table")
        last = {}
        eps.append({"seed": int(rec["seed"]), "stage": rec.get("stage"), "fold": rec.get("fold"),
                    "policy": rec.get("policy"), "smoke": bool(rec.get("smoke")), "n_units_logged": len(rec["units"])})
        for u in rec["units"]:
            if u["knob"] not in FAMILIES:
                dropped["family"] += 1
                continue
            t0 = int(round(float(u["t0"])))
            if t0 - H_pre < 0 or t0 + H > T:
                dropped["window"] += 1
                continue
            post = D[t0:t0 + H][:, fidx, :].sum(0)                    # (K, C)
            pre = D[t0 - H_pre:t0][:, fidx, :].sum(0)
            for j, k in enumerate(kpis):
                ycell[k].append(post[j] - pre[j])
                precell[k].append(pre[j])
            rel = relation_cells(int(u["c"]), u["exp"], nc)
            for r in relations:
                m = np.zeros(nc, bool)
                m[rel[r]] = True
                masks[r].append(m)
            row = u.get("probs")
            if row is not None:                                       # per-unit logged row (context-dependent pi0)
                pr = np.array([float(row.get(m, 0.0)) for m in MODES])
                if set(row) - set(MODES) or abs(pr.sum() - 1.0) > 1e-6:
                    raise ValueError(f"unit probs row {row} is not a distribution over {MODES}")
                n_probs_rows += 1
            elif table is not None:
                tab = table[u["x"]]
                pr = np.array([float(tab.get(m, 0.0)) for m in MODES])
            else:                                                     # deterministic policy (profiles): no CRT
                pr = np.array([1.0 if m == u["mode"] else 0.0 for m in MODES])
            if abs(pr[MODES.index(u["mode"])] - float(u["p"])) > 1e-6:
                p_mismatch += 1
            key = (int(u["c"]), u["x"], u["knob"])
            cols["prev"].append(last.get(key, -1))
            last[key] = n_rows
            step = float(u.get("step", u["ctx"].get("step", 0.0)))
            for k, v in (("episode", e), ("seed", int(rec["seed"])),
                         ("fold", -1 if rec.get("fold") is None else int(rec["fold"])),
                         ("family", FAMILIES.index(u["knob"])), ("xapp", u["x"]), ("c", int(u["c"])), ("t0", t0),
                         ("mode", MODES.index(u["mode"])), ("applied", MODES.index(u.get("applied_mode", u["mode"]))),
                         ("p", float(u["p"])), ("probs", pr), ("step", step), ("sgn", float(np.sign(step)) or 1.0)):
                cols[k].append(v)
            ctxs.append(u["ctx"])
            n_rows += 1
    n_cells = n_cells or 0
    n = n_rows
    arr = {k: np.asarray(v) for k, v in cols.items() if k not in ("probs", "xapp")}
    probs = np.asarray(cols["probs"], float).reshape(n, len(MODES))
    ycell_a = {k: np.asarray(v, float).reshape(n, n_cells) for k, v in ycell.items()}
    pre_a = {k: np.asarray(v, float).reshape(n, n_cells) for k, v in precell.items()}
    mask_a = {r: np.asarray(v, bool).reshape(n, n_cells) for r, v in masks.items()}
    y = {(r, k): (ycell_a[k] * mask_a[r]).sum(1) for r in relations for k in kpis}
    pre = {(r, k): (pre_a[k] * mask_a[r]).sum(1) for r in relations for k in kpis}
    keys = sorted({k for c in ctxs for k, v in c.items() if _num(v)})
    z = {f"ctx_{k}": np.array([float(c.get(k, np.nan)) if _num(c.get(k)) else np.nan for c in ctxs]) for k in keys}
    z.update({f"pre_{r}_{k}": pre[(r, k)] for r in relations for k in kpis})
    return UnitData(episode=arr["episode"].astype(int), seed=arr["seed"].astype(int), fold=arr["fold"].astype(int),
                    family=arr["family"].astype(int), xapp=np.asarray(cols["xapp"], dtype=object),
                    c=arr["c"].astype(int), t0=arr["t0"].astype(int), mode=arr["mode"].astype(int),
                    applied=arr["applied"].astype(int), p=arr["p"].astype(float), probs=probs,
                    step=arr["step"].astype(float), sgn=arr["sgn"].astype(float), y=y, pre=pre, ycell=ycell_a,
                    rel_mask=mask_a, z=z, prev=arr["prev"].astype(int), n_cells=n_cells, H=H, H_pre=H_pre,
                    relations=relations, kpis=kpis,
                    meta={"episodes": eps, "dropped": dropped, "p_mismatch": p_mismatch, "n_units": n,
                          "n_probs_rows": n_probs_rows,
                          "conditioners": list(z)})


# ------------------------------------------------------------------------------------------------ assignment
class PiAssignment:
    """Logged pi0 assignment model of a unit table (context-free, independent per unit)."""

    def __init__(self, data: UnitData):
        self.data = data

    def mode_draws(self, rng: np.random.Generator, rows: np.ndarray, B: int) -> np.ndarray:
        """(B, len(rows)) mode indices re-drawn from each row's logged table."""
        rows = np.asarray(rows, int)
        cum = np.cumsum(self.data.probs[rows], axis=1)
        cum[:, -1] = np.inf                                           # float round-off never falls off the end
        U = rng.random((int(B), len(rows)))
        out = np.zeros(U.shape, np.int64)
        for m in range(cum.shape[1] - 1):
            out += U >= cum[None, :, m]
        return out

    def conditional_draws(self, rng: np.random.Generator, family: str, B: int, rows=None,
                          what: str = "x") -> np.ndarray:
        """Re-draw family ``family``'s modes (every other family held fixed: not re-drawn, not returned).
        ``rows`` (default: every unit of the family, in table order) must all belong to the family. ``what``:
        "x" (level * sgn, the CRT candidate), "level" or "mode". Returns (B, len(rows))."""
        d = self.data
        fi = FAMILIES.index(family)
        rows = d.rows_of(family) if rows is None else np.asarray(rows, int)
        if len(rows) and not np.all(d.family[rows] == fi):
            raise ValueError("rows outside the family")
        M = self.mode_draws(rng, rows, B)
        if what == "mode":
            return M
        L = LEVEL_ARR[M]
        return L if what == "level" else L * d.sgn[rows][None, :]


# ------------------------------------------------------------------------------------------------ test
def _joint_draws(stats: list, X: np.ndarray) -> list:
    """``MSCRStat.draws`` for several targets sharing one strata plan (same Z): per stratum the argsort of X is
    computed once. Per target the arithmetic is exactly ``MSCRStat.draws``."""
    ref = stats[0]
    best = [np.full(X.shape[0], -np.inf) for _ in stats]
    for g, flat in enumerate(ref.flat):
        bss = [np.zeros(X.shape[0]) for _ in stats]
        live = [t for t, st in enumerate(stats) if st.denom[g + 1] > 0]
        if not live:
            continue
        for j, (rows, _, _sizes, _offsets) in enumerate(flat):
            order = np.argsort(X[:, rows], axis=1, kind="stable")
            for t in live:
                _, y_s, sz, off = stats[t].flat[g][j]
                gs = np.add.reduceat(y_s[order], off, axis=1)
                bss[t] += np.sum(gs ** 2 / sz[None, :], axis=1) - y_s.sum() ** 2 / len(y_s)
        for t in live:
            best[t] = np.maximum(best[t], bss[t] / stats[t].denom[g + 1])
    return best


def _beta(x: np.ndarray, y: np.ndarray, ep: np.ndarray) -> float:
    xc, yc = x.astype(float).copy(), y.astype(float).copy()
    for e in np.unique(ep):
        m = ep == e
        xc[m] -= xc[m].mean()
        yc[m] -= yc[m].mean()
    den = float(np.dot(xc, xc))
    return float(np.dot(xc, yc) / den) if den > 0 else 0.0


def family_tests(data: UnitData, family: str, targets, cfg: UnitCRTConfig | None = None, lag: int = 0,
                 split: int = SPLIT_POOLED, check: bool = True) -> list:
    """CRT of family ``family`` against each target. ``targets``: [(name, (rel, kpi) key of data.y | ("pre", rel,
    kpi) for the pre-window value)]. One conditioner set and one set of B draws for all targets. Returns one result
    dict per target (status "tested" | "undetermined")."""
    cfg = cfg or UnitCRTConfig()
    ccfg = cfg.crt_config()
    fi = FAMILIES.index(family)
    rows = data.rows_of(family)
    if lag == 1:
        rows = rows[data.prev[rows] >= 0]
    elif lag != 0:
        raise ValueError("lag must be 0 or 1")
    perm = np.random.default_rng([int(cfg.seed), CRT_TAG, 99, fi, lag, split]).permutation(len(rows))
    rows = rows[perm]
    src = rows if lag == 0 else data.prev[rows]
    lv = data.level[src]
    n_acc, n_rej = int((lv == 1.0).sum()), int((lv == 0.0).sum())
    n_ep = len(np.unique(data.episode[src]))
    base = {"version": CRT_UNITS_VERSION, "family": family, "lag": lag, "split": split, "n": len(rows),
            "n_accept": n_acc, "n_reject": n_rej, "n_half": int((lv == 0.5).sum()), "n_episodes": n_ep, "B": cfg.B,
            "p_mscr": np.nan, "p_signed": np.nan, "s_obs": np.nan, "t_obs": np.nan, "beta": np.nan, "sign": 0,
            "status": "undetermined", "reason": ""}

    def yvec(key):
        return data.pre[key[1:]] if key[0] == "pre" else data.y[key]

    def undetermined(name, key, reason):
        r = dict(base, target=name, key=list(key), reason=reason)
        r["claim"] = CLAIMS["undetermined"].format(reason=reason)
        return r

    out = [None] * len(targets)
    if (len(rows) < cfg.min_units or n_acc < cfg.min_accept or n_rej < cfg.min_reject
            or n_ep < cfg.min_episodes):
        return [undetermined(nm, k, "support") for nm, k in targets]
    Z, names = _conditioners(data.z, src)
    stats, idx = [], []
    for t, (nm, key) in enumerate(targets):
        y = yvec(key)[rows]
        if not np.all(np.isfinite(y)) or degenerate_columns({"y": y}):
            out[t] = undetermined(nm, key, "degenerate_outcome")
            continue
        st = MSCRStat(Z, y, ccfg)
        if not st.valid:
            out[t] = undetermined(nm, key, "no_conditioner_stratum")
            continue
        stats.append(st)
        idx.append(t)
    if not stats:
        return out
    x = data.x[src]
    rng = np.random.default_rng([int(cfg.seed), CRT_TAG, fi, lag, split])
    uniq, inv = np.unique(src, return_inverse=True)
    ep = data.episode[rows]
    pa = PiAssignment(data)
    s_null_all = [np.empty(cfg.B) for _ in stats]
    t_null_all = [np.empty(cfg.B) for _ in stats]
    X0, done = None, 0
    while done < cfg.B:                    # chunks of the same stream: identical to one (B, n) draw, bounded memory
        b = min(int(cfg.chunk), cfg.B - done)
        Xc = pa.conditional_draws(rng, family, b, rows=uniq)[:, inv]
        for j, v in enumerate(_joint_draws(stats, Xc)):
            s_null_all[j][done:done + b] = v
            t_null_all[j][done:done + b] = signed_stat(Xc, stats[j].y, ep)
        if X0 is None:
            X0 = Xc[:2]
        done += b
    s_obs_all = _joint_draws(stats, x[None, :])
    for st, t, s_null, t_null, s_o in zip(stats, idx, s_null_all, t_null_all, s_obs_all, strict=True):
        nm, key = targets[t]
        y = st.y
        s_obs = float(s_o[0])
        rep = {}
        if check:
            s_frozen = st.observed(x)
            joint_err = float(np.max(np.abs(st.draws(X0) - s_null[:len(X0)])))
            rep = {"s_obs_mscr_reuse": s_frozen, "joint_err": joint_err,
                   "replica_err": max(abs(s_obs - s_frozen), check_draws(st, X0, k=2))}
        t_obs = float(signed_stat(x, y, ep)[0])
        tol = 1e-12 * (1.0 + abs(s_obs))
        beta = _beta(x, y, ep)
        out[t] = dict(base, target=nm, key=list(key), status="tested",
                      p_mscr=(1.0 + np.count_nonzero(s_null >= s_obs - tol)) / (cfg.B + 1.0),
                      p_signed=(1.0 + np.count_nonzero(t_null >= t_obs * (1 - 1e-12))) / (cfg.B + 1.0),
                      s_obs=s_obs, t_obs=t_obs, beta=beta, sign=int(np.sign(beta)), n_conditioners=len(names),
                      **rep)
    return out


def crt_unit_test(data: UnitData, family: str, relation: str, kpi: str, cfg: UnitCRTConfig | None = None,
                  lag: int = 0, split: int = SPLIT_POOLED) -> dict:
    """One hypothesis (family -> relation KPI). Identical p-values to the batched ``run_crt_units`` (same draws)."""
    return family_tests(data, family, [(f"{relation}_{kpi}", (relation, kpi))], cfg, lag, split)[0]


def power_floor(B: int, m: int, q: float = 0.05) -> dict:
    """BY feasibility of a Monte-Carlo p floor 1 / (B + 1) over m hypotheses: rank-k threshold k q / (m c_m)."""
    c_m = sum(1.0 / k for k in range(1, max(int(m), 1) + 1))
    thr1 = q / (max(int(m), 1) * c_m)
    floor = 1.0 / (int(B) + 1.0)
    return {"B": int(B), "m": int(m), "p_floor": floor, "by_rank1_threshold": thr1,
            "min_hypotheses_at_floor_to_declare": int(np.ceil(floor / thr1 - 1e-12)), "rank1_ok": floor <= thr1,
            "min_B_rank1": int(np.ceil(1.0 / thr1 - 1.0))}


def _by(results, key="p_mscr", q=0.05, field="status"):
    tested = [r for r in results if r["status"] in ("tested", "declared", "not_detected")]
    if not tested:
        return 0
    dec_ = mscr.by_declare([r[key] for r in tested], q)
    for r, d in zip(tested, dec_, strict=True):
        if field == "status":
            r["status"] = "declared" if d else "not_detected"
            r["claim"] = CLAIMS[r["status"]]
        else:
            r[field] = bool(d)
    return int(dec_.sum())


def run_crt_units(data: UnitData, cfg: UnitCRTConfig | None = None, split: int = SPLIT_POOLED, audit: bool = True,
                  families=FAMILIES) -> dict:
    """The 60 hypotheses (families x relations x KPIs) + BY at q over the tested ones (+ the lag-1 audit)."""
    cfg = cfg or UnitCRTConfig()
    res = []
    for f in families:
        tg = [(f"{r}_{k}", (r, k)) for r in data.relations for k in data.kpis]
        for r in family_tests(data, f, tg, cfg, 0, split):
            r["relation"], r["kpi"] = r["key"]
            res.append(r)
    n_dec = _by(res, "p_mscr", cfg.q)
    _by(res, "p_signed", cfg.q, field="declared_signed")
    n_tested = sum(r["status"] != "undetermined" for r in res)
    out = {"version": CRT_UNITS_VERSION, "config": dataclasses.asdict(cfg), "split": split,
           "power_floor": power_floor(cfg.B, n_tested, cfg.q),
           "selection": f"BY-across-hypotheses (mscr-crt-units-v1), q={cfg.q:g}, over the tested of "
                        f"{len(res)}; NO MSCR FDR claim",
           "n_declared": n_dec, "n_tested": sum(r["status"] != "undetermined" for r in res), "results": res,
           "max_replica_err": max((r.get("replica_err", 0.0) for r in res), default=0.0),
           "max_joint_err": max((r.get("joint_err", 0.0) for r in res), default=0.0)}
    if audit:
        out["audit"] = carryover_audit(data, cfg, split, families)
    return out


def carryover_audit(data: UnitData, cfg: UnitCRTConfig | None = None, split: int = SPLIT_POOLED,
                    families=FAMILIES) -> dict:
    """lag-1: the previous (episode, c, xApp, knob) unit's assignment vs this unit's y and pre-window value, with
    ``cfg.audit_B`` draws. Reported: the share of tests at p <= alpha (vs alpha) and BY declarations (whose power
    floor is reported: at audit_B = 999 BY over 120 tests is not reachable at rank 1)."""
    cfg = cfg or UnitCRTConfig()
    acfg = dataclasses.replace(cfg, B=int(cfg.audit_B))
    res = []
    for f in families:
        tg = [(f"{r}_{k}", (r, k)) for r in data.relations for k in data.kpis]
        tg += [(f"pre_{r}_{k}", ("pre", r, k)) for r in data.relations for k in data.kpis]
        res.extend(family_tests(data, f, tg, acfg, 1, split, check=False))
    n_dec = _by(res, "p_mscr", cfg.q)
    tested = [r for r in res if r["status"] != "undetermined"]
    return {"B": acfg.B, "n_declared": n_dec, "n_tested": len(tested),
            "rate_p_le_alpha": (sum(r["p_mscr"] <= cfg.alpha for r in tested) / len(tested)) if tested else None,
            "power_floor": power_floor(acfg.B, len(tested), cfg.q),
            "smallest": sorted((float(r["p_mscr"]), r["family"], r["target"]) for r in tested)[:8],
            "declared": [(r["family"], r["target"]) for r in res if r["status"] == "declared"],
            "results": res, "wording": "descriptive carryover audit (BY across the audit's own tests); "
                                       "never enters the verdict"}


def edges_from_crt(run: dict) -> dict:
    """{(family, relation, kpi): {"declared", "sign", "score", "status", "p"}} over the 60 hypotheses."""
    out = {}
    for r in run["results"]:
        p = r["p_mscr"]
        out[(r["family"], r["relation"], r["kpi"])] = {
            "declared": r["status"] == "declared", "sign": int(r["sign"]), "status": r["status"], "p": p,
            "score": float(-np.log10(p)) if np.isfinite(p) else float("nan")}
    return out


def placebo_rejection_units(data: UnitData, cfg: UnitCRTConfig | None = None, binom_level: float = 0.01,
                            max_by: int = 1) -> dict:
    """K0 on PLACEBO records (logged pi0 modes, accept applied: a sharp null for every hypothesis). The logged modes
    are the replicate (a fresh re-draw would be uniform by construction and test nothing). Rejection rate = share of
    the tested hypotheses with p_mscr <= alpha; BY declarations at q. Also the chance of the rate rule failing under
    exactly-uniform independent p-values (Binomial reference; positively dependent p-values fail it more often)."""
    from scipy.stats import binom
    cfg = cfg or UnitCRTConfig()
    if np.any(data.applied != MODES.index("accept")):
        raise ValueError("placebo records must have applied_mode == accept for every unit")
    run = run_crt_units(data, cfg, split=SPLIT_PLACEBO, audit=False)
    tested = [r for r in run["results"] if r["status"] != "undetermined"]
    m = len(tested)
    rej = [r for r in tested if r["p_mscr"] <= cfg.alpha]
    rej_s = [r for r in tested if r["p_signed"] <= cfg.alpha]
    rate = len(rej) / m if m else float("nan")
    # K0 (frozen): INVALID if the rejection count is improbable under exactly-valid independent p-values,
    # P(Binom(m, alpha) >= n_rej) < binom_level (m = 60 -> invalid at >= 8 rejections, P = 0.0098), or > max_by BY declarations
    p_count = float(binom.sf(len(rej) - 1, m, cfg.alpha)) if m else float("nan")
    k_max = int(binom.isf(binom_level, m, cfg.alpha)) if m else 0
    return {"n_hypotheses": len(run["results"]), "n_tested": m, "alpha": cfg.alpha, "rate_mscr": rate,
            "rate_signed": len(rej_s) / m if m else float("nan"), "n_reject_mscr": len(rej),
            "n_by_declared": run["n_declared"],
            "per_hypothesis": [{"family": r["family"], "relation": r["relation"], "kpi": r["kpi"],
                                "status": r["status"], "p_mscr": r["p_mscr"], "p_signed": r["p_signed"],
                                "reject": bool(r["status"] != "undetermined" and r["p_mscr"] <= cfg.alpha)}
                               for r in run["results"]],
            "p_count": p_count, "k_max": k_max,
            "rule": f"P(Binom(m, {cfg.alpha}) >= n_reject) >= {binom_level} and <= {max_by} BY declaration(s)",
            "pass": bool(m > 0 and p_count >= binom_level and run["n_declared"] <= max_by),
            "binomial_fail_prob_if_uniform_indep": float(binom.sf(k_max, m, cfg.alpha)) if m else None,
            "run": run}


# ------------------------------------------------------------------------------------------------ localisation
def localise_receivers(data: UnitData, kpi: str = "load") -> dict:
    """Per (seed, pico): the method's top-1 receiving cell of a pico sleep. Sleep units with a sleep request (step
    > 0) of each (episode, cell): per other cell q, the within-group contrast of the unit's statistic applied cell by
    cell, sum_u (x_u - xbar) (y_uq - ybar_q) when the group's x varies, else sum_u x_u y_uq (dose-weighted post - pre
    change); top-1 = argmax over q != pico of the positive scores (None if none)."""
    fi = FAMILIES.index("sleep")
    rows = np.nonzero((data.family == fi) & (data.step > 0))[0]
    out = {}
    Y = data.ycell[kpi]
    for key in sorted({(int(data.seed[i]), int(data.c[i])) for i in rows}):
        g = rows[(data.seed[rows] == key[0]) & (data.c[rows] == key[1])]
        x = data.x[g]
        yq = Y[g]
        if np.ptp(x) > 0:
            s = ((x - x.mean())[:, None] * (yq - yq.mean(0))).sum(0)
        else:
            s = (x[:, None] * yq).sum(0)
        s[key[1]] = 0.0
        s = np.maximum(s, 0.0)
        out[key] = {"top1": int(np.argmax(s)) if s.max() > 0 else None, "n_units": len(g),
                    "varied": bool(np.ptp(x) > 0)}
    return out


# ------------------------------------------------------------------------------------------------ synthetic
def synthetic_records(rng: np.random.Generator, n_ep: int = 8, units_per_ep: int = 60, n_cells: int = 8,
                      effect: float = 0.0, family: str = "sleep", placebo: bool = False, T: int = 720,
                      far_effect: float = 0.0, seed0: int = 1000, strong_nuisance: float = 1.0) -> list:
    """Synthetic "e6p-disc-rec/1"-shaped episode records for mechanism tests. Cells on a ring; N(c) = {c-1, c, c+1}.
    Series: per cell per second AR(1) load around 20 UEs, Poisson pv / v / rlf, energy 1000 J +- noise. Units of the
    four families on random cells and seconds in [120, T - H); modes drawn from the pi0 table (applied = accept on
    placebo). A ``family`` unit adds effect * level * sgn / H per second to the load AND the pv of each neighbour
    (and minus that to its own load) over its window; ``far_effect`` does the same on the far cells. Another family
    (ptx) carries a fixed nuisance effect on own energy (conditional null exercised). No gt_static."""
    from .collect_p import PI0_HIGH_NO_RB, SERIES_FIELDS, enc
    xapp = {"carrier": "ES", "sleep": "ES", "ptx": "PowerES", "prot_min": "SliceGuarantee"}
    fam_p = {"carrier": 0.2, "sleep": 0.35, "ptx": 0.2, "prot_min": 0.25}
    fams = list(fam_p)
    F = len(SERIES_FIELDS)
    fi = {f: SERIES_FIELDS.index(f) for f in SERIES_FIELDS}
    recs = []
    for e in range(n_ep):
        D = np.zeros((T, F, n_cells))
        lvl = 20.0 + rng.normal(0, 2, n_cells)
        ar = np.zeros(n_cells)
        for t in range(T):
            ar = 0.95 * ar + rng.normal(0, 0.5, n_cells)
            D[t, fi["ue"]] = lvl + ar
        D[:, fi["pv"]] = rng.poisson(0.5, (T, n_cells))
        D[:, fi["v"]] = rng.poisson(1.0, (T, n_cells))
        D[:, fi["e"]] = 1000.0 + rng.normal(0, 5.0, (T, n_cells))
        D[:, fi["rlf"]] = rng.poisson(0.02, (T, n_cells))
        units = []
        t0s = np.sort(rng.integers(120, T - H_UNIT, units_per_ep))
        for i, t0 in enumerate(t0s):
            f = fams[int(rng.choice(len(fams), p=[fam_p[k] for k in fams]))]
            x = xapp[f]
            c = int(rng.integers(n_cells))
            exp = sorted({(c - 1) % n_cells, c, (c + 1) % n_cells})
            tab = PI0_HIGH_NO_RB[x]
            ms = [m for m in PI0_ORDER if m in tab]
            m = ms[int(rng.choice(len(ms), p=[tab[k] for k in ms]))]
            step = float(rng.choice([-1.0, 1.0]))
            applied = "accept" if placebo else m
            lev = LEVEL[applied] * (np.sign(step) or 1.0)
            w = slice(int(t0), int(t0) + H_UNIT)
            nb = [q for q in exp if q != c]
            far = [q for q in range(n_cells) if q not in exp]
            if f == family and effect:
                for q in nb:
                    D[w, fi["ue"], q] += effect * lev / H_UNIT
                    D[w, fi["pv"], q] += effect * lev / H_UNIT
                D[w, fi["ue"], c] -= effect * lev / H_UNIT
            if f == family and far_effect:
                for q in far:
                    D[w, fi["ue"], q] += far_effect * lev / H_UNIT
            if f == "ptx" and strong_nuisance:
                D[w, fi["e"], c] += strong_nuisance * 50.0 * lev / H_UNIT
            ctx = {"knob": f, "is_macro": float(f in ("carrier", "ptx")), "cur": 0.0, "prop": step, "step": step,
                   "since_change": float(rng.integers(0, 300)), "n_exp": len(exp),
                   "own_prb_util": float(rng.uniform(0.2, 0.9)), "nbr_mean_prb_util": float(rng.uniform(0.2, 0.9))}
            units.append({"i": i, "c": c, "x": x, "x_idx": ("ES", "PowerES", "SliceGuarantee").index(x), "knob": f,
                          "t0": float(t0), "mode": m, "p": float(tab[m]), "applied_mode": applied, "cur": 0.0,
                          "prop": step, "step": step, "exp": exp, "n_req": 1, "n_applied": 1, "rb_issued": False,
                          "ctx": ctx, "lab_kpi": None})
        recs.append({"kind": "episode", "schema": "e6p-disc-rec/1", "key": ["dev", "", seed0 + e], "stage": "dev",
                     "sub": "", "seed": seed0 + e, "cfg_seed": seed0 + e, "j": e, "fold": e % 3, "smoke": False,
                     "policy": "placebo" if placebo else "pi0", "pi0_table": PI0_HIGH_NO_RB, "T": 60.0,
                     "n_cells": n_cells, "units": units,
                     "lab_series": {"fields": list(SERIES_FIELDS), "t_first": 1,
                                    "scored": enc((np.arange(T) >= 120).astype(float)), "data": enc(D)}})
    return recs


__all__ = ["CRT_TAG", "CRT_UNITS_VERSION", "FAMILIES", "H_UNIT", "HYPOTHESES", "KPIS", "LEVEL", "MODES",
           "PRIVILEGED_KEYS", "RELATIONS", "SERIES_OF", "SPLIT_PLACEBO", "SPLIT_POOLED", "PiAssignment", "UnitCRTConfig",
           "power_floor",
           "UnitData", "build_unit_data", "carryover_audit", "crt_unit_test", "edges_from_crt", "family_tests",
           "localise_receivers", "placebo_rejection_units", "run_crt_units", "synthetic_records"]

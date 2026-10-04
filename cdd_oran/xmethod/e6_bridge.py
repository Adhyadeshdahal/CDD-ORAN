"""Experiment B bridge: the frozen E6-P v4 data as cross-method ``api.Dataset``s (docs/xmethod/EXP_B.md).

Read-only on everything frozen. Two steps:

1. ``build_table`` (cloud, where the v4 collection kernels are mounted as sources): the v4 loaders exactly as the v4
   analyzer / Experiment D (``e6p_disc_analyze_v4.build_caches`` -> ``pmrt.load_pmrt_pool`` (H 90, H_pre 90) ->
   ``pmrt.pmrt_data``) give the CRT unit table of every stage; it is written as ONE compact npz per stage
   (``E6UNITS_SCHEMA``). Nothing is re-simulated; the raw JSONL is never modified.
2. ``make_dataset`` (any platform): one Dataset per (slice, family) = the units of that family in the slice's
   episodes, plus its Truth from the stored v4 GT cells.

Mapping (one row = one randomised unit of family f; rows in information order (episode seed, t0)):
  action   ``A_<f>`` = sgn x L, L = crt_units_v2 LEVEL_V2[logged mode] (accept 1 / reject 0), sgn = sign(prop - cur):
           the signed applied change the arbiter's draw allows (intention to treat: the LOGGED mode, not the applied
           one). Design ``logged`` ``categorical_rows`` (the harness R3 convention: per-row probabilities in
           ``propensity``) over values (-1, 0, 1), propensity [P(L=1) 1{sgn<0}, P(L=0), P(L=1) 1{sgn>0}] from the
           logged pi0 row (v4: .5 / .5). Its centred value is exactly pmrt's v_design = sgn (L - E L).
  placebo  ``P_placebo`` = sgn x Bernoulli(P(L=1)), the SAME design, drawn independently (RNG
           default_rng([7800, 6, family index, dataset seed])): the truth-free negative control (CONTRACT R-10).
  KPIs     15 targets ``K_<rel>_<kpi>`` (rel own / nbr / far, kpi pv / v / e / rlf / load: the v4 hypothesis set).
           X_kpi_lag = the pre-window sum over [t0 - 90, t0), Y = the post-window sum over [t0, t0 + 90) (= pre + the
           crt_units outcome y = post - pre): Study A's state-at-t / KPI-at-t+1 convention. A lag column that is a
           linear combination of the earlier ones (fixed order) is set all-NaN, which every adapter drops
           (``meta["lag_dropped_collinear"]``): E6 holds the UE population fixed, so own + nbr + far load = 27 000
           in every window and the far-load lag is determined by the other two. Its target stays.
  context  every numeric obs-only ctx key at t0 (all-NaN keys dropped), ``hist`` (past design-centred modes of every
           family at the own cell / exposure neighbours, 150 s, past only), sgn, t0 / T: pmrt's predictable unit
           features. All are pre-assignment, so conditioning on them is valid for every method. Encoded per dataset
           (``encode_context``): all-NaN keys dropped, partly missing ones 0-filled + ``:missing`` indicator, then a
           full-rank subset in fixed order (``full_rank_columns``: E6 ctx has structural redundancies).
  time     ``time_index`` = 0..n-1 in information order, contiguous: the eq covariates' "actions at t-1, t-2" are
           the previous units of the family (for an episode's first units: the last units of the previous episode,
           an independent earlier draw, so still pre-assignment). No gap rows: Study A's adapters never saw gap
           indicators, and the citests stop on the constant gap columns they leave on kept rows.
  candidates  primary only: (A_f, K) and (P_placebo, K) for the 15 targets (30). No lagged-KPI -> KPI family (v4 has
           no truth for it).
Truth (``truth_from_gt``): v4 GT cell (f, rel, kpi) TRUE -> edge with the GT sign; NULL -> null edge (an EQUIVALENCE
label: GT CI inside +- delta, not a sharp null; Experiment D); INDET -> neither (unscored); P_placebo -> K null.
Placebo-log datasets (stage "placebo": accept applied whatever the logged mode) are an exact sharp null for A_f:
every candidate is a null edge.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from cdd_oran.xmethod import api

E6UNITS_SCHEMA = "xm-e6units/1"
WORLD = "E6"
FAMILIES = ("carrier", "sleep", "ptx", "prot_min")                 # cdd_oran.decision.gt_p (checked in build)
RELATIONS = ("own", "nbr", "far")
KPIS = ("pv", "v", "e", "rlf", "load")
TARGETS = tuple((r, k) for r in RELATIONS for k in KPIS)
KPI_NAMES = tuple(f"K_{r}_{k}" for r, k in TARGETS)
PLACEBO = "P_placebo"
VALUES = (-1.0, 0.0, 1.0)
RNG_TAG_HARNESS = 7800
SEEDS_V4 = {"dev": (188000, 20), "eval": (188100, 600), "placebo": (188700, 40), "gt": (188800, 40)}
SPLIT_V4 = {"pooled": 0, "60": 20, "120": 40, "300": 60, "placebo": 9}   # e6p_disc_analyze_v4.SPLIT (checked)


def action_name(family: str) -> str:
    return f"A_{family}"


# ================================================================================================ 1. unit table
def unit_table(pd) -> dict:
    """Compact arrays of a ``pmrt.PmrtData`` (all families, every unit of the H 90 / H_pre 90 set)."""
    from cdd_oran.decision.crt_units_v2 import LEVEL_V2_ARR
    ud = pd.ud
    p1 = np.asarray(ud.probs, float) @ LEVEL_V2_ARR                  # P(L = 1) under the logged pi0 row
    keep_ctx = [j for j in range(pd.ctx.shape[1]) if np.isfinite(pd.ctx[:, j]).any()]
    return {
        "schema": np.array(E6UNITS_SCHEMA),
        "seed": np.asarray(ud.seed, np.int64), "episode": np.asarray(ud.episode, np.int64),
        "family": np.asarray(ud.family, np.int8), "t0": np.asarray(ud.t0, np.int32), "c": np.asarray(ud.c, np.int16),
        "sgn": np.asarray(ud.sgn, np.float32), "mode": np.asarray(ud.mode, np.int8),
        "level": LEVEL_V2_ARR[np.asarray(ud.mode, int)].astype(np.float32), "p1": p1.astype(np.float64),
        "probs": np.asarray(ud.probs, np.float64),
        "pre": np.column_stack([ud.pre[t] for t in TARGETS]).astype(np.float64),
        "y": np.column_stack([ud.y[t] for t in TARGETS]).astype(np.float64),
        "ctx": pd.ctx[:, keep_ctx].astype(np.float64), "ctx_keys": np.array([pd.ctx_keys[j] for j in keep_ctx]),
        "hist": np.asarray(pd.hist, np.float64), "tfrac": np.asarray(pd.tfrac, np.float64),
        "families": np.array(FAMILIES), "targets": np.array([f"{r}|{k}" for r, k in TARGETS]),
    }


def _check_vocab() -> None:
    from cdd_oran.decision.crt_units import FAMILIES as F0
    from cdd_oran.decision.crt_units import KPIS as K0
    from cdd_oran.decision.crt_units import RELATIONS as R0
    if (tuple(F0), tuple(R0), tuple(K0)) != (FAMILIES, RELATIONS, KPIS):
        raise AssertionError(f"E6 vocabulary changed: {F0} {R0} {K0}")


def build_table(sources: dict[str, str], out_dir: str, workers: int = 4) -> dict:
    """``sources`` = {stage: comma-separated v4 collection dirs / jsonl}; writes ``<out_dir>/e6units_<stage>.npz``.
    Uses the v4 analyzer's own loaders (read-only import of scratchpad/e6_dev/e6p_disc_analyze_v4.py)."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(root, "scratchpad", "e6_dev"))
    import e6p_disc_analyze_v4 as AZ  # noqa: E402

    from cdd_oran.decision import pmrt as PM
    _check_vocab()
    if dict(AZ.SPLIT) | SPLIT_V4 != dict(AZ.SPLIT) or {k: tuple(v) for k, v in AZ.SEEDS.items()} != SEEDS_V4:
        raise AssertionError("v4 analyzer SPLIT / SEEDS differ from the bridge constants")
    os.makedirs(out_dir, exist_ok=True)
    info = {}
    for stage, src in sources.items():
        files = AZ.expand(src)
        cache = AZ.build_caches(files, stage, os.path.join(out_dir, "cache", stage), stage, workers)
        pool = PM.load_pmrt_pool(cache, stages={stage})
        pd = PM.pmrt_data(pool)
        tab = unit_table(pd)
        lo, n = SEEDS_V4[stage]
        seeds = sorted(set(int(s) for s in pool.eps["seed"]))
        tab["stage"] = np.array(stage)
        path = os.path.join(out_dir, f"e6units_{stage}.npz")
        np.savez_compressed(path, **tab)
        info[stage] = {"n_units": int(pd.n), "n_episodes": int(pool.n_eps), "seed_range": [seeds[0], seeds[-1]],
                       "complete": seeds == list(range(lo, lo + n)), "subs": sorted(set(map(str, pool.eps["sub"]))),
                       "by_family": {f: int((tab["family"] == i).sum()) for i, f in enumerate(FAMILIES)},
                       "pi0_rows": np.unique(tab["probs"], axis=0).tolist(), "file": os.path.basename(path)}
    json.dump(info, open(os.path.join(out_dir, "e6units_info.json"), "w"), indent=1)
    return info


# ================================================================================================ 2. datasets
class Table:
    """A loaded unit table (one stage)."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.a = {k: z[k] for k in z.files}
        z.close()
        if str(self.a["schema"]) != E6UNITS_SCHEMA:
            raise ValueError(f"{path}: schema {self.a['schema']} != {E6UNITS_SCHEMA}")
        self.stage = str(self.a["stage"])
        self.ctx_keys = [str(x) for x in self.a["ctx_keys"]]

    def rows(self, family: str, seeds) -> np.ndarray:
        """Row indices of ``family`` in the episodes ``seeds``, in information order (seed, t0, original order)."""
        a = self.a
        m = (a["family"] == FAMILIES.index(family)) & np.isin(a["seed"], np.asarray(list(seeds), np.int64))
        idx = np.nonzero(m)[0]
        return idx[np.lexsort((idx, a["t0"][idx], a["seed"][idx]))]


def dataset_seed(split: int, family: str) -> int:
    """Dataset identity (method RNG streams key on it): 60 000 000 + 10 x v4 split + family index. Disjoint from
    Study A's DEV / EVAL blocks (3 000 000+)."""
    return 60_000_000 + 10 * int(split) + FAMILIES.index(family)


def encode_context(M: np.ndarray, names: list[str], base: list | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """Per dataset, the shared helper's encoding (covariates._finite_block, R-25): an all-NaN column is dropped, a
    partly missing one is 0-filled and followed by its indicator ``<name>:missing``; then only the columns that add
    rank to (intercept, ``base`` = the action and kept lag columns, earlier context columns) are kept, so the R-3
    source design is non-singular (pcorr stops on a rank-deficient design). Every method then sees finite
    context (some E6 ctx keys are undefined for some units, e.g. own_carriers of a sleeping pico)."""
    cols, out = [], []
    for j, nm in enumerate(names):
        x = np.asarray(M[:, j], float)
        bad = ~np.isfinite(x)
        if bad.all():
            continue
        if bad.any():
            cols += [np.where(bad, 0.0, x), bad.astype(float)]
            out += [nm, f"{nm}:missing"]
        else:
            cols.append(x)
            out.append(nm)
    nb = len(base or [])                                         # base columns (actions, kept lags) come first
    allc = list(base or []) + cols
    keep = [i for i in full_rank_columns(allc) if i >= nb]
    keep = [i - nb for i in drop_near_collinear(allc, list(range(nb)), keep)]
    return (np.column_stack([cols[i] for i in keep]) if keep else np.zeros((len(M), 0))), tuple(out[i] for i in keep)


def full_rank_columns(cols: list, tol: float = 1e-7) -> list[int]:
    """Indices of a full-rank subset (with an intercept), greedy in the given order: a column is dropped if it is
    constant or a linear combination of the intercept and the columns kept before it (relative residual variance
    <= tol; 1e-7 sits above pcorr's own near-collinearity stop, 1e-8 on standardised columns). Removes E6's structural redundancies (sum-to-one shares, a key observed only as 0 = its indicator's
    complement), so every adapter gets a non-singular conditioning set. Truth-free, fixed order."""
    if not cols:
        return []
    n = len(cols[0])
    Q = np.ones((n, 1)) / math.sqrt(n)                           # orthonormal basis of the kept span
    keep = []
    for i, c in enumerate(cols):
        x = np.asarray(c, float) - np.mean(c)
        v = float(x @ x)
        if v <= 0.0:
            continue
        r = x - Q @ (Q.T @ x)
        r = r - Q @ (Q.T @ r)                                    # re-orthogonalise (numerical stability)
        if float(r @ r) <= tol * v:
            continue
        Q = np.column_stack([Q, r / math.sqrt(float(r @ r))])
        keep.append(i)
    return keep


def drop_near_collinear(cols: list, fixed: list[int], cand: list[int], tol: float = 1e-6) -> list[int]:
    """Second pass after ``full_rank_columns``: while some column of fixed + cand has standardised residual variance
    given all the others, 1 / diag(R^-1) (the quantity pcorr's near-collinearity stop checks at 1e-8), below
    ``tol``, drop the ``cand`` column with the smallest such value (ties: the later one). Returns the kept cand."""
    cand = list(cand)
    while cand:
        idx = list(fixed) + cand
        X = np.column_stack([np.asarray(cols[i], float) for i in idx])
        X = (X - X.mean(0)) / X.std(0)
        R = (X.T @ X) / len(X)
        d = 1.0 / np.clip(np.diag(np.linalg.pinv(R, hermitian=True)), 1e-300, None)
        if d.min() >= tol:
            break
        dc = d[len(fixed):]
        j = int(np.flatnonzero(dc == dc.min())[-1])
        if dc[j] >= tol:                                         # the dependence is among fixed columns only
            break
        cand.pop(j)
    return cand


def make_dataset(tab: Table, family: str, seeds, split: int, slice_name: str) -> api.Dataset:
    a = tab.a
    r = tab.rows(family, seeds)
    n = len(r)
    sgn = a["sgn"][r].astype(float)
    p1 = a["p1"][r]
    act = sgn * a["level"][r].astype(float) + 0.0                  # + 0.0: no -0.0 entries
    prop = np.column_stack([p1 * (sgn < 0), 1.0 - p1, p1 * (sgn > 0)])
    dseed = dataset_seed(split, family)
    rng = np.random.default_rng([RNG_TAG_HARNESS, 6, FAMILIES.index(family), dseed])
    plac = sgn * (rng.random(n) < p1).astype(float) + 0.0
    design = api.Design("logged", {"name": "categorical_rows", "values": list(VALUES)}, propensity=prop)
    ep = a["seed"][r]
    ti = np.arange(n, dtype=np.int64)                            # contiguous: no gap rows (see module doc)
    post = a["pre"][r] + a["y"][r]                               # Y from the ORIGINAL pre-window (before blanking)
    pre = a["pre"][r].copy()
    keep_lag = full_rank_columns([pre[:, j] for j in range(pre.shape[1])])
    lag_dropped = [KPI_NAMES[j] for j in range(pre.shape[1]) if j not in keep_lag]
    pre[:, [j for j in range(pre.shape[1]) if j not in keep_lag]] = np.nan   # adapters drop NaN-only lag columns
    ctx, cnames = encode_context(
        np.column_stack([a["ctx"][r], a["hist"][r], sgn, a["tfrac"][r]]),
        [f"ctx_{k}" for k in tab.ctx_keys] + [f"hist_{w}_{f}" for w in ("own", "nbr") for f in FAMILIES]
        + ["sgn", "tfrac"], base=[act, plac] + [pre[:, j] for j in keep_lag])
    an = action_name(family)
    cands = tuple((s, k) for s in (an, PLACEBO) for k in KPI_NAMES)
    meta = {"source": "e6p-v4", "stage": tab.stage, "slice": slice_name, "split": int(split), "family": family,
            "n_episodes": int(len(np.unique(ep))), "seed_range": [int(ep.min()), int(ep.max())] if n else None,
            "context_names": cnames, "lag_dropped_collinear": lag_dropped, "primary_candidates": cands,
            "secondary_candidates": (), "diagnostic_candidates": (), "placebo_conf": None}
    return api.Dataset(world=WORLD, regime=family, n=n, seed=dseed, action_names=(an, PLACEBO),
                       kpi_names=KPI_NAMES, X_action=np.column_stack([act, plac]), X_kpi_lag=pre,
                       Y=post, designs=(design, design), candidates=cands, time_index=ti, context=ctx,
                       meta=meta)


def load_gt(path: str) -> dict:
    """{(family, relation, kpi): cell} from a stored v4 GT block (analysis_v4.json["gt"], or a copy of it)."""
    g = json.load(open(path, encoding="utf-8"))
    g = g.get("gt", g)
    return {(c["family"], c["relation"], c["kpi"]): c for c in g["cells"]}


def truth_from_gt(gt: dict, family: str, placebo_log: bool = False) -> api.Truth:
    an = action_name(family)
    edges, signs, nulls = set(), {}, {(PLACEBO, k) for k in KPI_NAMES}
    for (r, k), name in zip(TARGETS, KPI_NAMES, strict=True):
        e = (an, name)
        if placebo_log:
            nulls.add(e)
            continue
        c = gt[(family, r, k)]
        if c["status"] == "TRUE":
            edges.add(e)
            if c.get("sign") in (1, -1):
                signs[e] = int(c["sign"])
        elif c["status"] == "NULL":
            nulls.add(e)
    return api.Truth(world=WORLD, regime=family, edges=frozenset(edges), signs=signs, null_edges=frozenset(nulls))


def slice_plan(kinds=("60", "120", "300", "pooled")) -> list[dict]:
    """The v4 analyzer's eval slices (e6p_disc_analyze_v4.slice_plan, non-pseudo) + the placebo logs."""
    lo, n = SEEDS_V4["eval"]
    out = []
    if "pooled" in kinds:
        out.append({"name": "pooled", "kind": "pooled", "split": SPLIT_V4["pooled"], "stage": "eval",
                    "seeds": list(range(lo, lo + n))})
    for kind in ("60", "120", "300"):
        if kind not in kinds:
            continue
        size = int(kind)
        for i in range(n // size):
            out.append({"name": f"s{kind}_{i}", "kind": kind, "split": SPLIT_V4[kind] + i, "stage": "eval",
                        "seeds": list(range(lo + i * size, lo + (i + 1) * size))})
    if "placebo" in kinds:
        plo, pn = SEEDS_V4["placebo"]
        out.append({"name": "placebo", "kind": "placebo", "split": SPLIT_V4["placebo"], "stage": "placebo",
                    "seeds": list(range(plo, plo + pn))})
    return out

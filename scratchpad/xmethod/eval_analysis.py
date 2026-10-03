"""EVAL analysis of Study A (docs/xmethod/PROTOCOL_A.md sections 8-13): tables V0-V11 as JSON + markdown.

    uv run python scratchpad/xmethod/eval_analysis.py --spec specs/eval/full.json --merged merged.jsonl[.gz] \
        --out results/eval/ --freeze-commit SHA [--amendments specs/eval/amendments.json] [--dev-merged DEV.jsonl.gz]
    uv run python scratchpad/xmethod/eval_analysis.py t1 --spec specs/dev/full.json --merged DEV.jsonl.gz [...]

Input: merged campaign records (`cdd_oran/xmethod/campaign.py`, one JSON record per unit = (arm, world, regime,
lam, n, kappa, seed), fields key / job / role / status / edges / notes / cpu_s / code / run_mode / pkgs / host /
dataset_sha256). Output: ``eval_tables.json`` and ``EVAL_TABLES.md`` in ``--out``. Deterministic: records sorted
by key, seeds sorted before every bootstrap, fixed bootstrap and simulation seeds, JSON with sorted keys.

Only records whose key is an expected unit of the spec with the expected role enter the tables (others are listed
in V11). Declarations: p arms (``declare: by``) use the adapter's BY flags (re-checked in V11); tau arms declare
``score > tau``, tau = the conformal placebo cutoff over the cell's TUNE-role records (R-29). Rates leave
not-testable candidates out of every denominator; raw-p rates also need a p (section 8). Validity is three-way per
(arm, cell, rate) (R-30); F_max is calibrated by simulation (``fmax_simulate``, section 10). Truth enters only the
scoring. R-42: the focal PMRT arm is the spec's ``focal``; PMRT power in R4 (no known design) is not applicable
(NA, never 0); V0 is the headline like-for-like table (raw p <= .05 and conformal tau for every p arm).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from typing import Any

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from cdd_oran.xmethod import api  # noqa: E402
from cdd_oran.xmethod import runner as R  # noqa: E402
from cdd_oran.xmethod.score import (  # noqa: E402
    PLACEBO,
    PLACEBO_CONF,
    apply_threshold,
    not_testable_edges,
    placebo_tau,
    score,
)
from cdd_oran.xmethod.worlds import REGIMES_OF, truth_for  # noqa: E402

ANALYSIS_VERSION = "xm-eval-analysis/5"
ALPHA = 0.05
VALID_UB = 0.075             # Bradley's liberal band 1.5 alpha (R-30)
Q_BY = 0.05
CONFORMAL_LEVEL = 0.05       # R-29
BOOT_REPS = 2000
BOOT_SEED = 20261002
MIN_FLAG_SEEDS = 10
MIN_COVERAGE = 0.9           # share of a component's pre-registered cells that must be counted (else NOT EVALUABLE)
PRIMARY_KAPPA = 0.25
SWEEP_KAPPAS = (0.125, 0.25, 0.5)
SWEEP_N = 1000
FOCAL = "pmrt_eq"            # focal arm of a spec without ``focal`` (DEV); EVAL names its primary PMRT arm (R-42)
NO_DESIGN_REGIMES = ("R4",)  # real actions of design kind 'none' (generate.py): PMRT power not applicable (R-42)
FMAX_NSIM = 2000
FMAX_SEED = 20261003
FMAX_Q = 0.95
DEP_PATH = os.path.join(ROOT, "scratchpad", "xmethod", "results", "fmax_sim", "dependence.json")
PROTOCOL_PATH = os.path.join(ROOT, "docs", "xmethod", "PROTOCOL_A.md")
FROZEN_MARK = "FROZEN: yes"
DEFAULT_E4_LAMS = {"R1": [1.0], "R2": [1.0], "R3": [0.0, 0.5, 1.0, 1.5], "R4": [0.0, 0.5, 1.0, 1.5]}
RATE_KEYS = ("null_raw", "null_decl", "plac_raw", "plac_decl", "conf_raw", "conf_decl")
LIKE_RAW_KEYS = ("null_decl", "plac_decl", "conf_decl")  # validity of the V0 raw p <= .05 scoring (no tuning column)
COUNTED = ("ok", "under_seeded")      # cell statuses that enter verdicts and tables
P_KEYS = ("null_raw", "plac_raw")     # C1 / C2 legs (BY declarations imply raw p <= .05, so raw is the binding leg)
C3_P_KEYS = ("plac_raw", "conf_raw")  # E4 has no real-action null candidate
C3_TAU_KEYS = ("conf_decl",)          # P_placebo is the tuning column of tau arms; the confounded placebo is not
EQ_TAU_KEYS = ("null_decl",)          # tau eq arms in R1 / R2 (unfiltered eq table, R-56): truth-null declarations


# ================================================================================================ input
def load_records(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    out = []
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return sorted(out, key=lambda r: (r.get("key", ""), json.dumps(r, sort_keys=True)))


def file_sha256(path: str) -> str:
    """LF-normalised sha256 (the hash the freeze commit records; campaign.protocol_sha256 uses the same rule)."""
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def _status(r: dict) -> str:
    return r.get("status") or ("ok" if r.get("error") is None else "error")


def _lam(x) -> float | None:
    return None if x is None else float(x)


def cell_of(world: str, regime: str, lam, n: int, kappa: float) -> str:
    return R.cell_key(world, regime, _lam(lam), int(n), float(kappa))


def _seeds(s) -> list[int]:
    if isinstance(s, str):
        raise ValueError(f"spec seeds are a placeholder ({s!r}): fill them at the freeze (PROTOCOL_A T1)")
    return list(range(int(s[0]), int(s[1]) + 1)) if (len(s) == 2 and s[1] > s[0] + 1) else [int(x) for x in s]


def planned_units(spec: dict) -> dict[str, dict]:
    """key -> unit for every unit of ``spec`` incl. those above an arm's T3 ``max_n`` (flag ``t3``); mirrors
    ``campaign.expand`` (which drops the T3 units), without its seed guard."""
    arms = spec["arms"]
    out: dict[str, dict] = {}
    for b in spec["blocks"]:
        names = list(arms) if b.get("arms", "all") == "all" else list(b["arms"])
        lam_map = {**DEFAULT_E4_LAMS, **b.get("e4_lams", {})}
        for w in b["worlds"]:
            for r in b["regimes"]:
                if r not in REGIMES_OF[w]:
                    continue
                for lam in ([float(x) for x in lam_map[r]] if w == "E4" else [None]):
                    for kap in b["kappas"]:
                        for n in b["ns"]:
                            for s in _seeds(b["seeds"]):
                                for a in names:
                                    d = arms[a]
                                    if "worlds" in d and w not in d["worlds"]:
                                        continue
                                    k = R.job_key(a, w, r, lam, int(n), int(s), float(kap))
                                    out.setdefault(k, {"arm": a, "world": w, "regime": r, "lam": lam, "n": int(n),
                                                       "kappa": float(kap), "seed": int(s), "role": b["role"],
                                                       "t3": "max_n" in d and int(n) > int(d["max_n"])})
    return out


def expected_units(spec: dict) -> dict[str, dict]:
    """key -> unit for every unit the campaign runs (T3-infeasible units excluded, as ``campaign.expand``)."""
    return {k: u for k, u in planned_units(spec).items() if not u["t3"]}


def unit_of(r: dict) -> dict:
    j = r["job"]
    return {"arm": r.get("arm") or r["key"].split("|")[0], "world": j["world"], "regime": j["regime"],
            "lam": _lam(j.get("lam")), "n": int(j["n"]), "kappa": float(j.get("kappa") or 0.0),
            "seed": int(j["seed"]), "role": r.get("role") or "measure"}


def to_result(r: dict) -> api.Result:
    edges = tuple(api.EdgeResult(e["source"], e["target"], math.nan if e.get("score") is None else float(e["score"]),
                                 e.get("p"), int(e.get("sign") or 0), bool(e.get("declared")))
                  for e in r.get("edges", []))
    return api.Result(r.get("method") or "?", r.get("version") or "?", edges, float(r.get("method_cpu_s") or 0.0),
                      dict(r.get("config") or {}), dict(r.get("notes") or {}))


def arm_kind(arm: dict) -> str:
    """'pmrt' (design-based), 'eq' / 'eq_min' (equal / minimal design information) or 'native'."""
    if "pmrt_core" in arm["ref"]:
        return "pmrt"
    a = (arm.get("config") or {}).get("arm")
    return a if a in ("eq", "eq_min") else "native"


def focal_arm(spec: dict) -> str:
    """The primary PMRT arm (C2a, C3, V3, T1): spec ``focal`` (EVAL: pmrt_nl_eq, statistic by the R-42 rule, T10),
    else pmrt_eq."""
    return spec.get("focal") or FOCAL


def power_not_applicable(arm: dict, regime: str) -> bool:
    """R-42: a PMRT arm where the real actions have no known design (R4, kind 'none') tests no true edge, so its
    power is 'not applicable' (NA): never recall 0, left out of every recall table and mean. Its placebo rates
    (P_placebo has an i.i.d. design in R4) are still read."""
    return arm_kind(arm) == "pmrt" and regime in NO_DESIGN_REGIMES


def cost_field(arm: dict) -> str:
    """Record field of an arm's budget: 'wall_s' for a GPU arm (spec ``budget_wall_s``; R-41: 2 h wall per (method,
    dataset) on a Kaggle T4, GPU time reported apart from CPU), else 'cpu_s' (R-13). Infeasible cost, T3 cost
    (spec ``t3_cost_<field>``) and V10 are read in this unit."""
    return "wall_s" if arm.get("budget_wall_s") else "cpu_s"


def _wall(r: dict) -> float | None:
    """Wall seconds of a unit: the forked child's (campaign isolation) if recorded, else the runner's."""
    x = r.get("child_wall_s") if r.get("child_wall_s") is not None else r.get("wall_s")
    return None if x is None else float(x)


def native_partner(name: str, spec: dict) -> str | None:
    """The native arm an eq / eq_min arm is paired with: spec ``native_partner``, else the native arm of the same
    adapter ref."""
    d = spec["arms"][name]
    if d.get("native_partner"):
        return d["native_partner"]
    return next((b for b, db in sorted(spec["arms"].items()) if db["ref"] == d["ref"] and arm_kind(db) == "native"),
                None)


def analysis_block(name: str, spec: dict) -> str:
    """'primary' (pmrt_eq, eq arms, native-only methods) or 'secondary' (spec ``analysis``; default by kind)."""
    d = spec["arms"][name]
    if d.get("analysis"):
        return d["analysis"]
    return "primary" if name == focal_arm(spec) or arm_kind(d) == "eq" else "secondary"


def arm_label(name: str, spec: dict) -> str | None:
    """The reporting label a ruling attaches to an arm (spec ``label``; R-40), printed next to every verdict line."""
    return spec["arms"][name].get("label")


def _is_primary(src: str) -> bool:
    return not src.startswith("K") and src not in (PLACEBO, PLACEBO_CONF)


# ================================================================================================ statistics
def cluster_ci(hits, cnt, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> list:
    """Cluster bootstrap 95 % percentile CI of sum(hits) / sum(cnt); clusters in the given (sorted) order."""
    hits, cnt = np.asarray(hits, float), np.asarray(cnt, float)
    keep = cnt > 0
    hits, cnt = hits[keep], cnt[keep]
    if not len(cnt):
        return [None, None]
    idx = np.random.default_rng(seed).integers(0, len(cnt), (reps, len(cnt)))
    bs = hits[idx].sum(1) / cnt[idx].sum(1)
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def design_effect(hits, cnt) -> float:
    h, c = np.asarray(hits, float), np.asarray(cnt, float)
    N, k = c.sum(), len(c)
    if N <= 0 or k < 2:
        return 1.0
    p = h.sum() / N
    if not 0 < p < 1:
        return 1.0
    v = k / (k - 1) * float(np.sum((h - p * c) ** 2)) / N ** 2
    return max(1.0, v / (p * (1 - p) / N))


def wilson_deff(hits, cnt, z: float = 1.959964) -> list:
    hits, cnt = np.asarray(hits, float), np.asarray(cnt, float)
    N = cnt.sum()
    if N <= 0:
        return [None, None]
    p = hits.sum() / N
    ne = N / design_effect(hits[cnt > 0], cnt[cnt > 0])
    c = z * z / ne
    mid = (p + c / 2) / (1 + c)
    half = z * math.sqrt(p * (1 - p) / ne + c / (4 * ne)) / (1 + c)
    return [max(0.0, mid - half), min(1.0, mid + half)]


def classify(ci) -> str | None:
    """Three-way validity of one rate (R-30): INVALID if the CI lower bound > .05 (takes precedence), VALID if the
    upper bound <= .075, else INCONCLUSIVE."""
    if not ci or ci[0] is None:
        return None
    return "INVALID" if ci[0] > ALPHA else "VALID" if ci[1] <= VALID_UB else "INCONCLUSIVE"


def rate(hits, cnt) -> dict | None:
    hits, cnt = np.asarray(hits, float), np.asarray(cnt, float)
    N = float(cnt.sum())
    if N <= 0:
        return None
    ci = cluster_ci(hits, cnt)
    return {"rate": float(hits.sum() / N), "hits": int(hits.sum()), "n": int(N), "n_clusters": int(np.sum(cnt > 0)),
            "ci": ci, "ci_wilson_deff": wilson_deff(hits, cnt), "deff": design_effect(hits[cnt > 0], cnt[cnt > 0]),
            "validity": classify(ci)}


def combine(classes) -> str:
    """Cell / pooled validity over several rates: INVALID if any is, VALID if all (>= 1) are, else INCONCLUSIVE."""
    cs = [c for c in classes if c is not None]
    if not cs:
        return "NO_READ"
    return "INVALID" if "INVALID" in cs else "VALID" if all(c == "VALID" for c in cs) else "INCONCLUSIVE"


def mean_ci(v) -> dict | None:
    v = np.array([x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if not len(v):
        return None
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else None
    ci = cluster_ci(v, np.ones(len(v))) if len(v) > 1 else [None, None]
    return {"mean": float(v.mean()), "sd": sd, "n": int(len(v)), "ci": ci}


def by_declare(ps: list[float], q: float = Q_BY, m: int | None = None) -> list[bool]:
    """Benjamini-Yekutieli step-up at q over ``ps``; ``m`` (>= len(ps)) counts untested family members too."""
    m = max(len(ps), m or 0)
    if not ps:
        return []
    cm = sum(1.0 / i for i in range(1, m + 1))
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    k = 0
    for rank, i in enumerate(order, 1):
        if ps[i] <= rank * q / (m * cm):
            k = rank
    out = [False] * len(ps)
    for i in order[:k]:
        out[i] = True
    return out


def _placebo_scores(results) -> list[float]:
    return sorted(e.score for r in results for e in r.edges
                  if e.source == PLACEBO and e.score is not None and not math.isnan(e.score))


def _conformal_tau_fallback(results, alpha: float = CONFORMAL_LEVEL) -> float:
    """Local R-29 rule, used only while the shared ``score.placebo_tau`` lacks its ``alpha`` argument (xm/hac2)."""
    s = _placebo_scores(results)
    if not s:
        return math.inf
    k = math.ceil(Fraction(len(s) + 1) * (1 - Fraction(str(alpha))))
    return float(s[min(k, len(s)) - 1])


def conformal_tau(results, alpha: float = CONFORMAL_LEVEL) -> float:
    """R-29: tau = the ceil((M + 1)(1 - alpha))-th smallest of the M finite P_placebo scores of the tune records
    (the largest if that index exceeds M); declaring ``score > tau`` has out-of-sample placebo level <= ``alpha``
    under exchangeability. M = 0 gives +inf (nothing declared). ONE implementation: the shared
    ``score.placebo_tau(results, alpha=...)`` (xm/hac2, R-38); the local copy only bridges until it is merged."""
    results = list(results)
    if "alpha" in inspect.signature(placebo_tau).parameters:
        return float(placebo_tau(results, alpha=alpha))
    return _conformal_tau_fallback(results, alpha)


def paired_seeds(sd_d: float, delta: float = 0.15, alpha: float = 0.05, power: float = 0.8,
                 s_max: int = 100_000) -> int:
    """Seeds for a two-sided paired t-test to detect a mean recall gap ``delta`` (= dev_power.paired_seeds)."""
    if not np.isfinite(sd_d) or sd_d <= 0:
        return 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    s = max(2, math.ceil(((za + zb) * sd_d / delta) ** 2))
    while s < s_max:
        if s >= ((stats.t.ppf(1 - alpha / 2, s - 1) + stats.t.ppf(power, s - 1)) * sd_d / delta) ** 2:
            return s
        s += 1
    return s_max


def min_detectable_gap(sd_d: float, s: int, alpha: float = 0.05, power: float = 0.8) -> float:
    return float((stats.t.ppf(1 - alpha / 2, s - 1) + stats.t.ppf(power, s - 1)) * sd_d / math.sqrt(s))


# ================================================================================================ F_max simulation
_FMAX_CACHE: dict = {}


def _boot_mult(k: int) -> np.ndarray:
    """reps x k multiplicities of the bootstrap draws ``cluster_ci`` makes for k clusters (same rng, same order)."""
    idx = np.random.default_rng(BOOT_SEED).integers(0, k, (BOOT_REPS, k))
    m = np.zeros((BOOT_REPS, k))
    np.add.at(m, (np.repeat(np.arange(BOOT_REPS), k), idx.ravel()), 1.0)
    return m


def _kinds(world: str, regime: str) -> list[tuple[str, str]]:
    """Null candidates of the primary family with their rate kind ('null' / 'plac' / 'conf'), sorted."""
    t = truth_for(world, regime)
    out = []
    for s, k in sorted(e for e in t.null_edges if not e[0].startswith("K")):
        out.append((f"{s}->{k}", "plac" if s == PLACEBO else "conf" if s == PLACEBO_CONF else "null"))
    return out


def _psd_unit(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh((c + c.T) / 2)
    c = v @ np.diag(np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


def _cand_corr(cands: list[tuple[str, str]], row: dict | None) -> np.ndarray:
    """Between-candidate correlation of null statistics: dependence.json pair-type means (identity without)."""
    m = len(cands)
    c = np.eye(m)
    if row:
        for i, j in itertools.combinations(range(m), 2):
            (si, ti), (sj, tj) = cands[i][0].split("->"), cands[j][0].split("->")
            typ = "same_source" if si == sj else "same_target" if ti == tj else "other"
            c[i, j] = c[j, i] = float(row.get(typ, 0.0))
    return _psd_unit(c)


def fmax_simulate(cells, S: int, keys, dep: dict | None, nsim: int = FMAX_NSIM, seed: int = FMAX_SEED) -> dict:
    """Null distribution of the number of INVALID cells among ``cells`` [(world, regime, lam, n)] for one arm whose
    tests all have exactly level .05, under the shared-seed structure: S seeds shared by every cell; data nested
    in n (null statistics Brownian in n: corr sqrt(n1 / n2)); E4 lambdas share the seed's streams (dependence.json
    lambda_corr); candidates of one dataset correlated by pair type (dependence.json); worlds and regimes
    independent (separate RNG streams). Per cell and rate the analysis' own seed-cluster bootstrap (same draws) is
    applied, so the result includes its coverage error. F_max = smallest f with P(#INVALID <= f) >= .95.
    Also returned: P(pooled rates VALID) (C2 / C3 leg b) and P(pooled rate INVALID) (C1 R2 leg)."""
    dep = dep or {}
    cells = sorted({(w, r, _lam(lam), int(n)) for w, r, lam, n in cells},
                   key=lambda c: (c[0], c[1], -1.0 if c[2] is None else c[2], c[3]))
    ck = (tuple(cells), int(S), tuple(keys), dep.get("_sha"), nsim, seed)
    if ck in _FMAX_CACHE:
        return _FMAX_CACHE[ck]
    kinds_of = sorted({k.split("_")[0] for k in keys})
    groups: dict[tuple, list] = defaultdict(list)
    for c in cells:
        groups[(c[0], c[1])].append(c)
    rng = np.random.default_rng(seed)
    mult_s = _boot_mult(S)
    zc = stats.norm.ppf(1 - ALPHA / 2)
    gspec = []
    for (w, r), gc in sorted(groups.items()):
        cands = [c for c in _kinds(w, r) if c[1] in kinds_of]
        if not cands:
            continue
        lams = sorted({c[2] for c in gc}, key=lambda x: -1.0 if x is None else x)
        ns = sorted({c[3] for c in gc})
        row = (dep.get("worlds") or {}).get(f"{w}|{r}")
        cl = np.eye(len(lams))
        if row and row.get("lambda_corr") and lams != [None]:
            li = [row["lambdas"].index(x) for x in lams]
            cl = np.asarray(row["lambda_corr"])[np.ix_(li, li)]
        cn = np.array([[math.sqrt(min(a, b) / max(a, b)) for b in ns] for a in ns])
        grid = [(lam, n) for lam in lams for n in ns]
        gspec.append({"w": w, "r": r, "cands": cands, "grid": grid, "cells": [grid.index((c[2], c[3])) for c in gc],
                      "Lc": np.linalg.cholesky(_cand_corr(cands, row)),
                      "Lg": np.linalg.cholesky(_psd_unit(np.kron(cl, cn)))})
    order = [(g["w"], g["r"], gi) for g in gspec for gi in g["cells"]]
    n_cells = len(order)
    world_cnt: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for g in gspec:
        for kind in kinds_of:
            m = sum(1 for c in g["cands"] if c[1] == kind)
            world_cnt[kind][g["w"]] += m * len(g["cells"])
    counts, pooled_valid, pooled_invalid = [], [], []
    cell_inv, cell_val = np.zeros(n_cells), np.zeros(n_cells)
    done = 0
    while done < nsim:
        b = min(200, nsim - done)
        inv_cols, val_cols = [], []
        pooled_hits: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for g in gspec:
            e = rng.standard_normal((b, S, len(g["cands"]), len(g["grid"])))
            z = np.einsum("ij,bsjg->bsig", g["Lc"], e)
            z = np.einsum("gh,bsih->bsig", g["Lg"], z)
            rej = np.abs(z) > zc
            for gi in g["cells"]:
                lo_any, hi_all = np.zeros(b, bool), np.ones(b, bool)
                for kind in kinds_of:
                    sel = [i for i, c in enumerate(g["cands"]) if c[1] == kind]
                    if not sel:
                        continue
                    h = rej[:, :, sel, gi].sum(2).astype(float)                       # b x S hits per seed
                    bs = (h @ mult_s.T) / (len(sel) * S)
                    lo_any |= np.quantile(bs, 0.025, axis=1) > ALPHA
                    hi_all &= np.quantile(bs, 0.975, axis=1) <= VALID_UB
                    acc = pooled_hits[kind]
                    acc[g["w"]] = acc.get(g["w"], 0.0) + h
                inv_cols.append(lo_any)
                val_cols.append(hi_all & ~lo_any)
        inv = np.column_stack(inv_cols)
        counts.append(inv.sum(1))
        cell_inv += inv.sum(0)
        cell_val += np.column_stack(val_cols).sum(0)
        pv, pi = np.ones(b, bool), np.zeros(b, bool)
        for kind in sorted(pooled_hits):
            acc = pooled_hits[kind]
            ws = sorted(acc)
            hp = np.concatenate([acc[w] for w in ws], axis=1)                          # clusters (world, seed)
            cnt = np.concatenate([np.full(S, world_cnt[kind][w]) for w in ws])
            mk = _boot_mult(len(cnt))
            bs = (hp @ mk.T) / (cnt @ mk.T)
            lo, hi = np.quantile(bs, 0.025, axis=1), np.quantile(bs, 0.975, axis=1)
            pi |= lo > ALPHA
            pv &= (hi <= VALID_UB) & ~(lo > ALPHA)
        pooled_valid.append(pv)
        pooled_invalid.append(pi)
        done += b
    counts = np.concatenate(counts)
    pv, pi = np.concatenate(pooled_valid), np.concatenate(pooled_invalid)
    f = 0
    while np.mean(counts <= f) < FMAX_Q:
        f += 1
    pw = defaultdict(list)
    for (w, _r, _), v in zip(order, cell_val / nsim, strict=True):
        pw[w].append(v)
    out = {"n_cells": n_cells, "S": int(S), "keys": list(keys), "nsim": nsim, "f_max": int(f),
           "p_count_le_fmax": float(np.mean(counts <= f)), "p_pooled_valid": float(pv.mean()),
           "p_pass": float(np.mean((counts <= f) & pv)), "p_pooled_invalid": float(pi.mean()),
           "p_cell_invalid": float(cell_inv.sum() / (nsim * max(1, n_cells))),
           "p_cell_valid_by_world": {w: float(np.mean(v)) for w, v in sorted(pw.items())},
           "count_dist": {str(k): int(v) for k, v in sorted(Counter(counts.tolist()).items())}}
    _FMAX_CACHE[ck] = out
    return out


def load_dependence(path: str | None = DEP_PATH) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    d["_sha"] = file_sha256(path)
    return d


# ================================================================================================ per seed
def raw_declare(result: api.Result) -> api.Result:
    """Copy of ``result`` with ``declared = p <= .05`` per edge, no multiplicity (V0 like-for-like scoring, R-42);
    an edge without a p is not declared."""
    edges = tuple(api.EdgeResult(e.source, e.target, e.score, e.p, e.sign, e.p is not None and float(e.p) <= ALPHA)
                  for e in result.edges)
    return api.Result(method=result.method, version=result.version, edges=edges, cpu_s=result.cpu_s,
                      config={**result.config, "declare": "raw_p"}, notes=result.notes)


def fixed_declare(result: api.Result, threshold: float) -> api.Result:
    """Copy of ``result`` with ``declared = score >= threshold`` per edge: a score-only arm's own fixed threshold
    (spec ``fixed_threshold``; cdl: the conference's CMI >= .16, ``CDL.get_binary_graph``, R-50), not tuned; an
    edge without a finite score is not declared. Secondary scoring (V0 rule 'fixed')."""
    edges = tuple(api.EdgeResult(e.source, e.target, e.score, e.p, e.sign,
                                 e.score is not None and math.isfinite(e.score) and float(e.score) >= threshold)
                  for e in result.edges)
    return api.Result(method=result.method, version=result.version, edges=edges, cpu_s=result.cpu_s,
                      config={**result.config, "declare": "fixed", "fixed_threshold": threshold}, notes=result.notes)


def seed_row(r: dict, decl: api.Result, truth: api.Truth) -> dict:
    """Validity / power numbers of one measurement record under declaration ``decl`` (PROTOCOL_A section 8)."""
    sc = score(decl, truth)
    nt = set(sc["not_testable_edges"])
    declared = set(sc["declared_edges"])
    nulls = {f"{s}->{t}" for s, t in truth.null_edges if _is_primary(s)}
    cnt = {k: [0, 0] for k in RATE_KEYS}
    for e in r.get("edges", []):
        s, t, p = e["source"], e["target"], e.get("p")
        name = f"{s}->{t}"
        kind = "plac" if s == PLACEBO else "conf" if s == PLACEBO_CONF else "null" if name in nulls else None
        if kind is None or name in nt:
            continue
        cnt[f"{kind}_decl"][1] += 1
        cnt[f"{kind}_decl"][0] += int(name in declared)
        if p is not None:
            cnt[f"{kind}_raw"][1] += 1
            cnt[f"{kind}_raw"][0] += int(float(p) <= ALPHA)
    na = (r.get("notes") or {}).get("not_applicable") or {}
    by_edge = {(e["source"], e["target"]): e for e in r.get("edges", [])}
    signed = [(k, v) for k, v in truth.signs.items() if k in truth.edges and _is_primary(k[0])]

    def sgn(k) -> int:
        return int(np.sign((by_edge.get(k) or {}).get("sign") or 0))
    sign_ok = sum(1 for k, v in signed if f"{k[0]}->{k[1]}" in declared and sgn(k) == v)
    sign_bad = sum(1 for k, v in signed if f"{k[0]}->{k[1]}" in declared and sgn(k) not in (0, v))
    sec = sc["secondary"]
    return {"seed": int(r["job"]["seed"]), "world": r["job"]["world"], "recall": sc["recall"], "fdp": sc["fdp"],
            "n_declared": sc["n_declared"], "tp": sc["tp"], "n_true": sc["n_true"], "cnt": cnt,
            "sign_ok": sign_ok, "sign_bad": sign_bad, "n_signed_true": len(signed),
            "nt_true": sc["n_not_testable_true"], "nt_null": sc["n_not_testable_null"],
            "n_not_applicable": sum(1 for k in na if _is_primary(str(k).partition("->")[0])),
            "sec": {"recall": sec["recall"], "fdp": sec["fdp"], "fp": sec["fp"],
                    "n_null_testable": sec["n_null"] - sec["n_not_testable_null"], "n_true": sec["n_true"]}}


def validity_keys(has_p: bool) -> tuple[str, ...]:
    """Rates that decide a cell's validity: p arms raw p and BY declarations; tau arms the truth-null and the
    confounded-placebo declarations (P_placebo is their tuning column: reported, not used; section 8)."""
    return ("null_raw", "plac_raw", "conf_raw", "null_decl", "plac_decl", "conf_decl") if has_p \
        else ("null_decl", "conf_decl")


def summarise(rows: list[dict], has_p: bool, *, vkeys=None, power_na: bool = False) -> dict:
    """Per-cell rates, validity over ``vkeys`` (default ``validity_keys(has_p)``) and power. ``power_na`` (R-42,
    ``power_not_applicable``): recall and sign are NA (None), never 0."""
    rows = sorted(rows, key=lambda x: x["seed"])
    out: dict[str, Any] = {"n_seeds": len(rows)}
    for k in RATE_KEYS:
        if k.endswith("_raw") and not has_p:
            continue
        out[k] = rate([x["cnt"][k][0] for x in rows], [x["cnt"][k][1] for x in rows])
    out["validity"] = ("few_seeds" if len(rows) < MIN_FLAG_SEEDS
                       else combine((out.get(k) or {}).get("validity") for k in (vkeys or validity_keys(has_p))))
    out["recall"] = mean_ci([x["recall"] for x in rows])
    out["fdp"] = mean_ci([x["fdp"] for x in rows])
    so, sb = [x["sign_ok"] for x in rows], [x["sign_ok"] + x["sign_bad"] for x in rows]
    out["sign_acc"] = rate(so, sb)
    if out["sign_acc"]:
        out["sign_acc"].pop("validity")
    nst = sum(x["n_signed_true"] for x in rows)
    out["wrong_sign_rate"] = sum(x["sign_bad"] for x in rows) / nst if nst else None
    out["n_declared"] = float(np.mean([x["n_declared"] for x in rows])) if rows else None
    out.update(nt_true=int(sum(x["nt_true"] for x in rows)), nt_null=int(sum(x["nt_null"] for x in rows)),
               n_not_applicable=int(sum(x["n_not_applicable"] for x in rows)))
    out["secondary"] = {"recall": mean_ci([x["sec"]["recall"] for x in rows]),
                        "fdp": mean_ci([x["sec"]["fdp"] for x in rows]),
                        "null_decl": rate([x["sec"]["fp"] for x in rows],
                                          [x["sec"]["n_null_testable"] for x in rows])}
    out["per_seed_recall"] = {str(x["seed"]): x["recall"] for x in rows}
    out["power_not_applicable"] = power_na
    if power_na:
        out.update(recall=None, sign_acc=None, wrong_sign_rate=None, per_seed_recall={})
    return out


# ================================================================================================ record screening
def screen(records: list[dict], spec: dict) -> tuple[list[dict], dict]:
    """Records that enter the tables: one per expected key with the expected role (others listed, never used).
    Without a known unit list (placeholder seeds) every record of a spec arm is used (output PROVISIONAL)."""
    try:
        want, why = expected_units(spec), None
    except ValueError as ex:
        want, why = None, str(ex)
    keys = Counter(r.get("key") for r in records)
    dup = sorted(k for k, v in keys.items() if v > 1)
    seen, use = set(), []
    unexpected, role_bad = [], []
    for r in records:
        k = r.get("key")
        if k in seen:
            continue
        seen.add(k)
        if want is None:
            if unit_of(r)["arm"] in spec["arms"]:
                use.append(r)
            continue
        if k not in want:
            unexpected.append(k)
        elif unit_of(r)["role"] != want[k]["role"]:
            role_bad.append(k)
        else:
            use.append(r)
    return use, {"expected": want, "expected_error": why, "duplicates": dup,
                 "unexpected": sorted(unexpected), "role_mismatch": sorted(role_bad)}


# ================================================================================================ cells
def _grid_cells(spec: dict) -> dict[str, dict]:
    """arm|cell -> planned measurement cell (incl. T3 cells) with its planned seed count and T3 flag."""
    out: dict[str, dict] = {}
    try:
        units = planned_units(spec)
    except ValueError:
        return out
    for u in units.values():
        if u["role"] != "measure":
            continue
        ck = f"{u['arm']}|{cell_of(u['world'], u['regime'], u['lam'], u['n'], u['kappa'])}"
        e = out.setdefault(ck, {**{f: u[f] for f in ("arm", "world", "regime", "lam", "n", "kappa")},
                                "planned_seeds": 0, "t3": u["t3"]})
        e["planned_seeds"] += 1
    return out


def build_cells(records: list[dict], spec: dict) -> tuple[dict, dict]:
    """(cells, rows): cells[arm|cell] = summary entry; rows[arm|cell] = per-seed rows of the primary declaration.
    ``records`` are screened records (``screen``). Planned cells without records appear with status 'missing',
    T3 cells with 'infeasible_t3' and their DEV cost (section 7)."""
    arms = spec["arms"]
    groups: dict[str, dict[str, list]] = defaultdict(lambda: {"tune": [], "measure": [], "infeasible": [],
                                                              "error": []})
    meta = {}
    for r in records:
        u = unit_of(r)
        ck = f"{u['arm']}|{cell_of(u['world'], u['regime'], u['lam'], u['n'], u['kappa'])}"
        meta[ck] = u
        st = _status(r)
        groups[ck][u["role"] if st == "ok" else st].append(r)
    plan = _grid_cells(spec)
    truths: dict[tuple, api.Truth] = {}
    cells, rows_out = {}, {}
    for ck in sorted(set(groups) | set(plan)):
        u = plan.get(ck) or meta[ck]
        if u["arm"] not in arms:
            continue
        g = groups[ck] if ck in groups else {"tune": [], "measure": [], "infeasible": [], "error": []}
        d = arms[u["arm"]]
        mode = d["declare"]
        truth = truths.setdefault((u["world"], u["regime"]), truth_for(u["world"], u["regime"]))
        tau = conformal_tau([to_result(r) for r in g["tune"]]) if g["tune"] else None
        rows, rows_tau, rows_raw, rows_fixed = [], [], [], []
        thr = d.get("fixed_threshold")
        for r in g["measure"]:
            res = to_result(r)
            if mode == "by":
                rows.append(seed_row(r, res, truth))
                rows_raw.append(seed_row(r, raw_declare(res), truth))
            if thr is not None:
                rows_fixed.append(seed_row(r, fixed_declare(res, float(thr)), truth))
            if tau is not None:
                rows_tau.append(seed_row(r, apply_threshold(res, tau), truth))
        planned = (plan.get(ck) or {}).get("planned_seeds")
        e = {"arm": u["arm"], "kind": arm_kind(d), "block": analysis_block(u["arm"], spec), "declare": mode,
             "world": u["world"], "regime": u["regime"], "lam": u["lam"], "n": u["n"], "kappa": u["kappa"],
             "tau": tau if tau is None or math.isfinite(tau) else None,
             "tau_is_pos_inf": tau == math.inf, "n_tune": len(g["tune"]), "n_measure": len(g["measure"]), "planned_seeds": planned,
             "n_infeasible": len(g["infeasible"]), "n_error": len(g["error"])}
        prim = rows if mode == "by" else rows_tau
        cf = cost_field(d)
        if (plan.get(ck) or {}).get("t3"):
            e["status"] = "infeasible_t3"
            e[f"infeasible_cost_{cf}"] = (d.get(f"t3_cost_{cf}") or {}).get(str(u["n"]))
        elif g["infeasible"]:
            cost = [x.get(cf) for x in g["infeasible"] if x.get(cf) is not None]
            e["status"] = "infeasible"
            e[f"infeasible_cost_{cf}"] = max(cost) if cost else None
        elif not g["measure"]:
            e["status"] = "missing"
        elif mode == "tau" and tau is None:
            e["status"] = "untuned"
        elif len(prim) < MIN_FLAG_SEEDS:
            e["status"] = "few_seeds"
        elif planned is not None and len(prim) < planned:
            e["status"] = "under_seeded"
        else:
            e["status"] = "ok"
        na = power_not_applicable(d, u["regime"])
        e["power_not_applicable"] = na
        e["primary"] = summarise(prim, mode == "by", power_na=na) if prim else None
        if mode == "by" and rows_tau:
            e["secondary_tau"] = summarise(rows_tau, False, power_na=na)
        if rows_raw:
            e["like_raw"] = summarise(rows_raw, False, vkeys=LIKE_RAW_KEYS, power_na=na)
        if rows_fixed:                         # untuned: all three declaration rates enter its validity (R-50)
            e["like_fixed"] = summarise(rows_fixed, False, vkeys=LIKE_RAW_KEYS, power_na=na)
        cells[ck] = e
        rows_out[ck] = prim
    return cells, rows_out


def counts(e: dict) -> bool:
    """An arm-cell enters verdicts / comparisons: counted status and a validity read."""
    return e["status"] in COUNTED and bool(e.get("primary")) and e["primary"]["validity"] != "few_seeds"


def valid(e: dict) -> bool:
    return counts(e) and e["primary"]["validity"] == "VALID"


def powered(e: dict) -> bool:
    """Enters power tables (R-39): counted and not INVALID (VALID-only is the sensitivity read)."""
    return counts(e) and e["primary"]["validity"] != "INVALID"


def cell_class(e: dict, keys) -> str:
    """Validity of a counted cell restricted to ``keys`` (a component's rates)."""
    p = e["primary"]
    return combine((p.get(k) or {}).get("validity") for k in keys)


# ================================================================================================ verdicts (V4)
def _pooled(rows: list[dict], key: str) -> dict | None:
    """Rate of ``key`` pooled over the given rows, clusters = (world, seed) (separate RNG streams), sorted."""
    agg: dict[tuple, list] = defaultdict(lambda: [0, 0])
    for x in rows:
        c = agg[(x["world"], x["seed"])]
        c[0] += x["cnt"][key][0]
        c[1] += x["cnt"][key][1]
    ks = sorted(agg)
    return rate([agg[k][0] for k in ks], [agg[k][1] for k in ks])


def _set_of(cells: dict, arm: str, pred) -> list[str]:
    """Pre-registered cells of ``arm`` (kappa .25, T3 cells excluded) satisfying ``pred``."""
    return sorted(k for k, e in cells.items() if e["arm"] == arm and e["kappa"] == PRIMARY_KAPPA
                  and e["status"] != "infeasible_t3" and pred(e))


def _fmax(cells: dict, keys_used: list[str], rate_keys, dep) -> dict:
    if not keys_used:
        return {"f_max": 0}
    S = max(cells[k]["primary"]["n_seeds"] for k in keys_used)
    cs = [(cells[k]["world"], cells[k]["regime"], cells[k]["lam"], cells[k]["n"]) for k in keys_used]
    return fmax_simulate(cs, S, [k.replace("_decl", "_raw") for k in rate_keys], dep)


def validity_component(cells: dict, rows: dict, planned: list[str], rate_keys, dep) -> dict:
    """C2 / C3 rule for one arm (section 10): (a) #INVALID counted cells <= F_max (simulated); (b) every pooled
    rate VALID. SUPPORTED if both, capped at PARTIAL when a planned cell is not counted or is under-seeded;
    NOT EVALUABLE if < 90 % of the planned cells are counted."""
    use = [k for k in planned if counts(cells[k])]
    cls = {k: cell_class(cells[k], rate_keys) for k in use}
    inv = sorted(k for k, c in cls.items() if c == "INVALID")
    pooled = {rk: _pooled([x for k in use for x in rows[k]], rk) for rk in rate_keys}
    pooled_cls = combine((p or {}).get("validity") for p in pooled.values())
    fm = _fmax(cells, use, rate_keys, dep)
    a_ok, b_ok = len(inv) <= fm["f_max"], pooled_cls == "VALID"
    not_counted = sorted(set(planned) - set(use))
    under = sorted(k for k in use if cells[k]["status"] == "under_seeded")
    out = {"n_planned": len(planned), "n_cells": len(use), "n_invalid": len(inv),
           "n_valid": sum(c == "VALID" for c in cls.values()),
           "n_inconclusive": sum(c in ("INCONCLUSIVE", "NO_READ") for c in cls.values()),
           "f_max": fm["f_max"], "f_max_sim": {k: fm.get(k) for k in ("S", "p_pass", "p_count_le_fmax",
                                                                      "p_pooled_valid")},
           "invalid_cells": inv, "pooled": pooled, "pooled_validity": pooled_cls, "a_ok": a_ok, "b_ok": b_ok,
           "not_counted": not_counted, "under_seeded": under}
    if not use or len(use) < MIN_COVERAGE * len(planned):
        v = "NOT EVALUABLE"
    elif a_ok and b_ok:
        v = "PARTIAL" if (not_counted or under) else "SUPPORTED"
    else:
        v = "NOT SUPPORTED"
    out["verdict"] = v
    rest = [k for k in use if cells[k]["n"] != 500]          # descriptive re-read, no verdict effect
    out["without_n500"] = {"n_cells": len(rest), "n_invalid": sum(cls[k] == "INVALID" for k in rest)}
    return out


def c1_arm(cells: dict, rows: dict, arm: str, dep) -> dict:
    """Section 10 C1 for one design-blind arm: FAILURE iff (i) INVALID in >= half of its planned R2 cells (fixed
    denominator: a cell not counted is not INVALID), (ii) a pooled R2 rate INVALID and (iii) #INVALID R1 cells <=
    F_max(R1). 'INVALID IN R1' if (iii) fails (invalid regardless of the design)."""
    p2 = _set_of(cells, arm, lambda e: e["regime"] == "R2")
    p1 = _set_of(cells, arm, lambda e: e["regime"] == "R1")
    u2 = [k for k in p2 if counts(cells[k])]
    u1 = [k for k in p1 if counts(cells[k])]
    n2 = sum(cell_class(cells[k], P_KEYS) == "INVALID" for k in u2)
    n1 = sum(cell_class(cells[k], P_KEYS) == "INVALID" for k in u1)
    pooled2 = {rk: _pooled([x for k in u2 for x in rows[k]], rk) for rk in P_KEYS}
    pooled1 = {rk: _pooled([x for k in u1 for x in rows[k]], rk) for rk in P_KEYS}
    pooled_inv = any((p or {}).get("validity") == "INVALID" for p in pooled2.values())
    fm1 = _fmax(cells, u1, P_KEYS, dep)
    complete = len(u2) == len(p2) and len(u1) == len(p1) and not any(
        cells[k]["status"] == "under_seeded" for k in u1 + u2)
    assessable = bool(p2) and len(u2) >= MIN_COVERAGE * len(p2) and len(u1) >= MIN_COVERAGE * len(p1)
    if not assessable:
        v = "NOT EVALUABLE"
    elif n1 > fm1["f_max"]:
        v = "INVALID IN R1"
    elif 2 * n2 >= len(p2) and pooled_inv:
        v = "FAILURE"
    else:
        v = "NOT A FAILURE"
    return {"verdict": v, "complete": complete, "assessable": assessable, "r2_invalid": n2, "r2_planned": len(p2),
            "r2_counted": len(u2), "r1_invalid": n1, "r1_planned": len(p1), "r1_counted": len(u1),
            "r1_f_max": fm1["f_max"], "pooled_r2": pooled2, "pooled_r1": pooled1, "pooled_r2_invalid": pooled_inv,
            "r2_legs": bool(assessable and 2 * n2 >= len(p2) and pooled_inv)}         # legs (i) + (ii) alone


def _majority(passes: list[bool], complete: bool, n_min: int = 1) -> str:
    if len(passes) < n_min:
        return "NOT EVALUABLE"
    k = sum(passes)
    v = "SUPPORTED" if 2 * k >= len(passes) else "NOT SUPPORTED" if k == 0 else "PARTIAL"
    return "PARTIAL" if (v == "SUPPORTED" and not complete) else v


def verdicts(cells: dict, rows: dict, spec: dict, dep) -> dict:
    arms = spec["arms"]
    focal = focal_arm(spec)
    # C1: D = design-blind p arms (native, declare by; the R-38 pcorr_hac variant), fixed by the frozen spec; an
    # arm with "set_D" other than true (the HAC variant not chosen under T8, or still undecided) is not in D: same
    # legs reported descriptively, no effect (pdcor, the former such arm, is dropped from Study A, R-48)
    blind = sorted(a for a, d in arms.items() if arm_kind(d) == "native" and d["declare"] == "by")
    D = [a for a in blind if arms[a].get("set_D", True) is True]
    c1_arms = {a: c1_arm(cells, rows, a, dep) for a in D}
    counted = [a for a in D if c1_arms[a]["assessable"]]
    c1 = {"D": D, "D_counted": counted, "arms": c1_arms,
          "descriptive": {a: c1_arm(cells, rows, a, dep) for a in blind if a not in D},
          "n_failures": sum(c1_arms[a]["verdict"] == "FAILURE" for a in counted),
          "verdict": _majority([c1_arms[a]["verdict"] == "FAILURE" for a in counted],
                               all(v["complete"] for v in c1_arms.values()), n_min=3)}
    # R-56: C1 also without the arms flagged "c1_sensitivity_drop" (mscr_native, R-54): sensitivity line, no effect
    drop = [a for a in D if arms[a].get("c1_sensitivity_drop")]
    if drop:
        keep = [a for a in counted if a not in drop]
        c1["sensitivity_without"] = {
            "dropped": drop, "D_counted": keep, "n_failures": sum(c1_arms[a]["verdict"] == "FAILURE" for a in keep),
            "verdict": _majority([c1_arms[a]["verdict"] == "FAILURE" for a in keep],
                                 all(c1_arms[a]["complete"] for a in D if a not in drop), n_min=3)}
    hac = [a for a in D if "hac" in a]
    hac_ok = [a for a in hac if c1_arms[a]["verdict"] == "NOT A FAILURE"]
    c1["wording"] = ("design-blind tests are invalid on the R2 (setpoint + dither) design tested" if not hac_ok else
                     "design-blind tests with i.i.d. / exchangeable nulls are invalid on the R2 design tested; the "
                     "serial-dependence-robust design-blind test (" + ", ".join(hac_ok) + ") is not")
    # C2a: the primary PMRT arm (spec focal; R-42) over R1 + R2
    c2a = validity_component(cells, rows, _set_of(cells, focal, lambda e: e["regime"] in ("R1", "R2")), P_KEYS, dep)
    # C2b: every eq arm whose native partner is a C1 FAILURE, same rule; an arm with "c2b" other than true (R-40:
    # mscr_eq, not equal information) is left out and its same-rule result reported under its label, no effect
    def r1r2(a: str) -> dict:
        return validity_component(cells, rows, _set_of(cells, a, lambda e: e["regime"] in ("R1", "R2")), P_KEYS, dep)
    c2b_arms, c2b_excluded = {}, {}
    for a in sorted(arms):
        if arm_kind(arms[a]) != "eq" or arms[a]["declare"] != "by":
            continue
        partner = native_partner(a, spec)
        if arms[a].get("c2b", True) is not True:
            c2b_excluded[a] = {"native": partner, "label": arm_label(a, spec), **r1r2(a)}
        elif partner in c1_arms and c1_arms[partner]["verdict"] == "FAILURE":
            c2b_arms[a] = {"native": partner, **r1r2(a)}
    ev = {a: v for a, v in c2b_arms.items() if v["verdict"] != "NOT EVALUABLE"}
    c2b = {"arms": c2b_arms, "excluded": c2b_excluded,
           "verdict": _majority([v["verdict"] in ("SUPPORTED", "PARTIAL") for v in ev.values()],
                                len(ev) == len(c2b_arms) and all(v["verdict"] != "PARTIAL" for v in ev.values()))}
    # R-56: C2b also with the eq arms whose native partner fails C1's R2 legs but is INVALID IN R1 (sensitivity)
    r1fail = {}
    for a in sorted(arms):
        if arm_kind(arms[a]) != "eq" or arms[a]["declare"] != "by" or arms[a].get("c2b", True) is not True:
            continue
        p = native_partner(a, spec)
        pv = c1_arms.get(p) or c1["descriptive"].get(p)
        if a not in c2b_arms and pv and pv["verdict"] == "INVALID IN R1" and pv["r2_legs"]:
            r1fail[a] = {"native": p, **r1r2(a)}
    both = {**c2b_arms, **r1fail}
    ev2 = {a: v for a, v in both.items() if v["verdict"] != "NOT EVALUABLE"}
    c2b["with_r1_failing_partners"] = {
        "added": sorted(r1fail), "arms": r1fail,
        "verdict": _majority([v["verdict"] in ("SUPPORTED", "PARTIAL") for v in ev2.values()],
                             len(ev2) == len(both) and all(v["verdict"] != "PARTIAL" for v in ev2.values()))}
    # R-56: unfiltered verdict of EVERY eq arm (same rule on R1 + R2; tau arms on their truth-null declarations)
    eq_table = {}
    for a in sorted(arms):
        if arm_kind(arms[a]) != "eq":
            continue
        p = native_partner(a, spec)
        pv = c1_arms.get(p) or c1["descriptive"].get(p)
        keys = P_KEYS if arms[a]["declare"] == "by" else EQ_TAU_KEYS
        x = validity_component(cells, rows, _set_of(cells, a, lambda e: e["regime"] in ("R1", "R2")), keys, dep)
        member = ("member" if a in c2b_arms else "excluded (R-40)" if a in c2b_excluded else
                  f"not a member (declare {arms[a]['declare']})" if arms[a]["declare"] != "by" else
                  f"not a member (native {p}: {pv['verdict'] if pv else 'not assessed'})")
        own = c1_arm(cells, rows, a, dep) if arms[a]["declare"] == "by" else None
        eq_table[a] = {"native": p, "native_c1": pv["verdict"] if pv else None, "membership": member,
                       "label": arm_label(a, spec), "verdict": x["verdict"], "n_cells": x["n_cells"],
                       "n_invalid": x["n_invalid"], "f_max": x["f_max"], "pooled": x["pooled"],
                       "r2_invalid": own["r2_invalid"] if own else None, "r2_planned": own["r2_planned"] if own else None,
                       "pooled_r1": own["pooled_r1"] if own else None, "pooled_r2": own["pooled_r2"] if own else None}
    # R-56: eq arms INVALID in R2 whose native partner is INVALID IN R1 are NAMED in the C2 text (pre-registered
    # sentence); not C2b members, so they never change its verdict
    named = []
    for a, x in eq_table.items():
        own = c1_arm(cells, rows, a, dep) if arms[a]["declare"] == "by" else None
        if (own and own["r2_legs"] and x["native_c1"] == "INVALID IN R1" and x["membership"] != "member"
                and x["membership"] != "excluded (R-40)"):
            named.append({"arm": a, "native": x["native"], "sentence": (
                f"{a} is INVALID in R2 (pooled truth-null raw rate {_rate_txt(own['pooled_r2'].get('null_raw'))}) "
                f"as already in R1 ({_rate_txt(own['pooled_r1'].get('null_raw'))}): the test is miscalibrated "
                f"without the design (its native partner {x['native']} is INVALID IN R1), so it is outside C2b's "
                f"membership rule, and adding design covariates does not make it valid")})
    restored = sorted(a for a, v in ev.items() if v["verdict"] in ("SUPPORTED", "PARTIAL"))
    if c2a["verdict"] in ("SUPPORTED", "PARTIAL") and c2b["verdict"] == "SUPPORTED":
        c2_wording = "using the design restores validity (design-based inference and design-covariate adjustment)"
    elif c2a["verdict"] in ("SUPPORTED", "PARTIAL"):
        c2_wording = ("design-based inference restores validity; adding design covariates does not suffice"
                      + (f" (it does for {', '.join(restored)})" if restored else ""))
    else:
        c2_wording = "not supported: the design-based test is not shown valid in R1 / R2"
    if named:
        c2_wording += ("; " + ", ".join(x["arm"] for x in named) + " INVALID in R2 as already in R1 (not restored "
                       "by design covariates)")

    # C3: the primary PMRT arm in E4 R3; reported for every eq arm (no claim effect)
    def c3_for(a: str) -> dict:
        keys = C3_P_KEYS if arms[a]["declare"] == "by" else C3_TAU_KEYS
        return validity_component(cells, rows, _set_of(cells, a, lambda e: e["world"] == "E4"
                                                       and e["regime"] == "R3"), keys, dep)
    c3 = c3_for(focal) if focal in arms else {"verdict": "NOT EVALUABLE"}
    c3_eq = {a: c3_for(a) for a in sorted(arms) if arm_kind(arms[a]) == "eq"}
    also = sorted(a for a, v in c3_eq.items() if v["verdict"] in ("SUPPORTED", "PARTIAL"))
    c3["wording"] = ("PMRT stays valid under the logged confounded policy (R3) by construction"
                     + (f"; so do {', '.join(also)}, so the property is not specific to PMRT" if also else "")
                     if c3["verdict"] in ("SUPPORTED", "PARTIAL") else "not supported")
    # other PMRT arms (R-42: the linear pmrt_eq is secondary once pmrt_nl_eq is focal; pmrt_r3): same C2a / C3
    # rules reported, no claim effect, never in C3's "so do <arms>"
    pmrt_sec = {a: {"label": arm_label(a, spec), "C2a": r1r2(a), "C3": c3_for(a)}
                for a in sorted(arms) if arm_kind(arms[a]) == "pmrt" and a != focal}
    comps = {"C1": c1["verdict"], "C2a": c2a["verdict"], "C2b": c2b["verdict"], "C3": c3["verdict"]}
    claim = "SUPPORTED" if all(v == "SUPPORTED" for v in comps.values()) else "NOT SUPPORTED"
    return {"focal": focal, "C1": c1, "C2a": c2a, "C2b": c2b, "C2_wording": c2_wording, "C2_named": named,
            "eq_arms_table": eq_table, "C3": c3,
            "C3_eq_arms": c3_eq, "pmrt_secondary": pmrt_sec, "components": comps, "claim": claim,
            "arm_labels": {a: arm_label(a, spec) for a in sorted(arms) if arm_label(a, spec)}}


# ================================================================================================ tables
def _rate_txt(x: dict | None) -> str:
    return "n/a" if not x or x.get("rate") is None else (f"{x['rate']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"
                                                        if x.get("ci") else f"{x['rate']:.3f}")


def _brief(x: dict | None) -> dict | None:
    return None if not x else {"rate": x["rate"], "ci": x["ci"], "n": x["n"], "validity": x["validity"]}


def _cell_row(k: str, e: dict) -> dict:
    p = e.get("primary") or {}
    row = {"key": k, **{f: e.get(f) for f in ("arm", "kind", "block", "declare", "world", "regime", "lam", "n",
                                              "kappa", "status", "tau", "tau_is_pos_inf", "n_tune", "n_measure", "planned_seeds",
                                              "n_infeasible", "n_error")}}
    row["validity"] = p.get("validity")
    row["n_seeds"] = p.get("n_seeds")
    for rk in RATE_KEYS:
        row[rk] = _brief(p.get(rk))
    row["power_not_applicable"] = bool(e.get("power_not_applicable"))
    for f in ("infeasible_cost_cpu_s", "infeasible_cost_wall_s"):
        if e.get(f) is not None:
            row[f] = e[f]
    return row


def _paired(ra: dict, rb: dict) -> list[float]:
    seeds = sorted((s for s in set(ra) & set(rb) if ra[s] is not None and rb[s] is not None
                    and not math.isnan(ra[s]) and not math.isnan(rb[s])), key=int)
    return [ra[s] - rb[s] for s in seeds]


def _gate_ok(validity: str | None, gate: str) -> bool:
    return validity == "VALID" if gate == "VALID" else validity in ("VALID", "INCONCLUSIVE", "NO_READ")


def paired_diffs(cells: dict, gate: str = "not_invalid", focal: str = FOCAL) -> list[dict]:
    """V3: recall of the focal PMRT arm minus arm per cell, both arms passing ``gate`` (R-39: 'not_invalid' = not
    INVALID; sensitivity read 'VALID'), under the SAME declaration rule: p arms against the focal arm's BY
    declarations, tau arms against the focal arm scored with tau (R-29). Per-cell 95 % paired-t CI, descriptive (no
    simultaneous inference); E4 rows descriptive (one true edge). Cells where either arm's power is not applicable
    (R-42, R4) have no per-seed recall and are left out."""
    out = []
    for k, e in sorted(cells.items()):
        if e["arm"] == focal or not counts(e) or not _gate_ok(e["primary"]["validity"], gate):
            continue
        f = cells.get(f"{focal}|{k.split('|', 1)[1]}")
        if f is None or not counts(f) or not _gate_ok(f["primary"]["validity"], gate):
            continue
        if e["declare"] == "by":
            fp, rule = f["primary"], "BY"
        else:
            fp, rule = f.get("secondary_tau"), "tau"
            if not fp or not _gate_ok(fp["validity"], gate):
                continue
        d = np.array(_paired(fp["per_seed_recall"], e["primary"]["per_seed_recall"]))
        if len(d) < 2:
            continue
        sd = float(np.std(d, ddof=1))
        half = float(stats.t.ppf(0.975, len(d) - 1) * sd / math.sqrt(len(d))) if sd > 0 else 0.0
        out.append({"cell": k.split("|", 1)[1], "arm": e["arm"], "rule": rule, "block": e["block"],
                    "world": e["world"], "regime": e["regime"], "lam": e["lam"], "n": e["n"], "kappa": e["kappa"],
                    "descriptive_only": e["world"] == "E4", "n_seeds": len(d), "mean_d": float(d.mean()),
                    "sd_d": sd, "ci": [float(d.mean()) - half, float(d.mean()) + half]})
    return out


def like_for_like(cells: dict, focal: str = FOCAL) -> list[dict]:
    """V0, the headline like-for-like table (R-42): every arm per cell under a common scoring, side by side.
    p arms twice: raw p <= .05 per edge (no multiplicity; rule 'raw_p') and the conformal placebo tau of the
    score-only arms (R-29; rule 'tau'); tau (score-only) arms with their tau, and an arm with a spec
    ``fixed_threshold`` (cdl, R-50) also at that fixed threshold (rule 'fixed', secondary, untuned). Per row: mean per-seed recall (NA where
    the arm's power is not applicable: PMRT in R4), truth-null and placebo declaration rates (not-applicable
    candidates count as not declared, section 8) with CI and three-way validity. The placebo is the tuning column
    of a tau scoring (reported, not in its validity: truth-null and confounded placebo); the raw_p and fixed
    validity use all three. Recall is shown for every counted cell next to its rates (descriptive; V2 / V3 keep the R-39 gate)."""
    out = []
    for k, e in sorted(cells.items()):
        rules = ([("raw_p", e.get("like_raw")), ("tau", e.get("secondary_tau"))] if e["declare"] == "by"
                 else [("tau", e.get("primary"))] + ([("fixed", e.get("like_fixed"))] if "like_fixed" in e else []))
        for rule, s in rules:
            ok = e["status"] in COUNTED and bool(s) and s["validity"] != "few_seeds"
            state = e["status"] if e["status"] not in COUNTED else "untuned" if not s else s["validity"]
            s = s or {}
            out.append({"key": k, "cell": k.split("|", 1)[1], "arm": e["arm"], "focal": e["arm"] == focal,
                        "kind": e["kind"], "block": e["block"], "declare": e["declare"], "rule": rule,
                        **{f: e[f] for f in ("world", "regime", "lam", "n", "kappa", "status")},
                        "state": state, "n_seeds": s.get("n_seeds", 0),
                        "power_not_applicable": bool(e.get("power_not_applicable")),
                        "recall": s.get("recall") if ok else None,
                        "null": _brief(s.get("null_decl")) if ok else None,
                        "placebo": _brief(s.get("plac_decl")) if ok else None,
                        "placebo_conf": _brief(s.get("conf_decl")) if ok else None,
                        "placebo_is_tuning_column": rule == "tau"})
    return out


def load_amendments(path: str | None) -> dict:
    if not path:
        return {"amendments": [], "persistent_errors": []}
    a = json.load(open(path, encoding="utf-8"))
    return {"amendments": a.get("amendments", []), "persistent_errors": a.get("persistent_errors", [])}


def protocol_check(spec: dict, protocol_path: str | None = PROTOCOL_PATH) -> dict:
    """The protocol file is frozen and its LF sha256 equals the spec's constant (as campaign.eval_authorised)."""
    want = spec.get("protocol_sha256")
    if not protocol_path or not os.path.exists(protocol_path):
        return {"ok": False, "why": "protocol file missing"}
    got = file_sha256(protocol_path)
    text = open(protocol_path, encoding="utf-8").read().replace("\r\n", "\n")
    frozen = any(line.startswith(FROZEN_MARK) for line in text.split("\n"))
    ok = bool(want) and str(want).lower() == got and frozen
    return {"ok": ok, "spec_protocol_sha256": want, "file_sha256": got, "frozen": frozen}


def integrity(records: list[dict], screened: dict, spec: dict, freeze_commit: str | None, *,
              spec_sha: str | None = None, protocol: dict | None = None, amendments: dict | None = None,
              dep: dict | None = None, dev_records: list[dict] | None = None) -> dict:
    """V11 and the FINAL / PROVISIONAL label (section 11): FINAL needs every check in ``checks`` to pass."""
    amendments = amendments or {"amendments": [], "persistent_errors": []}
    want = screened["expected"]
    use_keys = {r["key"] for r in records}
    missing = (sorted(set(want) - use_keys) if want is not None
               else [f"expected units unknown: {screened['expected_error']}"])
    allowed: dict[str, set] = {}
    for a in amendments["amendments"]:
        for k in a.get("keys", []):
            allowed.setdefault(k, set()).add(a.get("commit"))
    commits = sorted({str((r.get("code") or {}).get("commit")) for r in records})
    bad_commit = sorted(r["key"] for r in records if (r.get("code") or {}).get("commit") != freeze_commit
                        and (r.get("code") or {}).get("commit") not in allowed.get(r["key"], set()))
    dirty = sorted(r["key"] for r in records if (r.get("code") or {}).get("dirty") is not False)
    psha = spec.get("protocol_sha256")
    bad_stamp = sorted(r["key"] for r in records if (r.get("run_mode") or {}).get("protocol_sha256") != psha
                       or (spec_sha is not None and (r.get("run_mode") or {}).get("spec_sha256") != spec_sha))
    persist = {p["key"] for p in amendments["persistent_errors"]}
    errors = sorted(r["key"] for r in records if _status(r) == "error")
    errors_unlisted = sorted(set(errors) - persist)
    truths: dict = {}
    incomplete = []
    for r in records:
        if _status(r) != "ok":
            continue
        u = unit_of(r)
        t = truths.setdefault((u["world"], u["regime"]), truth_for(u["world"], u["regime"]))
        if {(e["source"], e["target"]) for e in r.get("edges", [])} != set(t.edges) | set(t.null_edges):
            incomplete.append(r["key"])
    # R-41a: a dataset's arms may run in separate shards (a GPU shard), so every arm of a dataset must carry
    # the same dataset_sha256 (runner.run_one: generate.dataset_hash). An ok record without it fails the check;
    # a not-ok record is compared when stamped (a unit not run has no dataset) and counted when not.
    ds: dict[tuple, set] = defaultdict(set)
    plat: dict[tuple, set] = defaultdict(set)
    hash_unstamped, hash_unstamped_not_ok = [], []
    for r in records:
        u = unit_of(r)
        dk = (u["world"], u["regime"], u["lam"], u["n"], u["kappa"], u["seed"])
        if r.get("dataset_sha256"):
            ds[dk].add(r["dataset_sha256"])
        elif _status(r) == "ok":
            hash_unstamped.append(r["key"])
        else:
            hash_unstamped_not_ok.append(r["key"])
        plat[dk].add(str((r.get("host") or {}).get("platform")))
    hash_mismatch = sorted(str(k) for k, v in ds.items() if len(v) > 1)
    multi_platform = sorted(str(k) for k, v in plat.items() if len(v) > 1)
    pkgs = Counter(json.dumps(r.get("pkgs"), sort_keys=True) for r in records if _status(r) == "ok")
    pk_lock = spec.get("pkgs_lock")
    pkgs_ok = len(pkgs) == 1 and (pk_lock is None or json.loads(next(iter(pkgs))) == pk_lock)
    dep_ok = spec.get("fmax_dependence_sha256") is None or (dep or {}).get("_sha") == spec["fmax_dependence_sha256"]
    # BY recheck (p arms): BY at .05 over the primary family (action sources incl. P_placebo, P_placebo_conf out)
    by_n = by_agree = 0
    by_bad: list[str] = []
    for r in records:
        if _status(r) != "ok" or spec["arms"].get(unit_of(r)["arm"], {}).get("declare") != "by":
            continue
        nt = {f"{s}->{t}" for s, t in not_testable_edges(to_result(r))}
        fam = [e for e in r.get("edges", []) if not e["source"].startswith("K") and e["source"] != PLACEBO_CONF
               and e.get("p") is not None and f"{e['source']}->{e['target']}" not in nt]
        mine = by_declare([float(e["p"]) for e in fam], m=(r.get("notes") or {}).get("by_family_m"))
        by_n += 1
        if all(bool(e.get("declared")) == m for e, m in zip(fam, mine, strict=True)):
            by_agree += 1
        elif len(by_bad) < 20:
            by_bad.append(r["key"])
    roles = Counter(f"{unit_of(r)['role']}:{_status(r)}" for r in records)
    repro = None
    if dev_records is not None:                   # R-34: EVAL tune re-run vs DEV tune records (report only)
        dev = {r["key"]: r for r in dev_records if _status(r) == "ok"}
        both = [(r, dev[r["key"]]) for r in records if r["key"] in dev and _status(r) == "ok"
                and unit_of(r)["role"] == "tune"]
        repro = {"compared": len(both),
                 "dataset_sha_equal": sum(a.get("dataset_sha256") == b.get("dataset_sha256") for a, b in both),
                 "declarations_equal": sum([bool(e.get("declared")) for e in a["edges"]]
                                           == [bool(e.get("declared")) for e in b["edges"]] for a, b in both)}
    protocol = protocol or {"ok": False, "why": "not checked"}
    checks = {"protocol_frozen_sha": protocol["ok"], "freeze_commit_given": freeze_commit is not None,
              "commits": not bad_commit, "clean": not dirty, "stamps": not bad_stamp, "missing": not missing,
              "unexpected": not screened["unexpected"], "role": not screened["role_mismatch"],
              "duplicates": not screened["duplicates"], "candidates_complete": not incomplete,
              "errors_listed": not errors_unlisted, "dataset_hash": not hash_mismatch,
              "dataset_hash_stamped": not hash_unstamped,
              "one_platform_per_dataset": not multi_platform, "pkgs_uniform": pkgs_ok, "fmax_dependence": dep_ok}
    return {"label": "FINAL" if all(checks.values()) else "PROVISIONAL", "checks": checks,
            "failed": sorted(k for k, v in checks.items() if not v),
            "freeze_commit": freeze_commit, "commits": commits, "commit_violations": bad_commit[:20],
            "n_commit_violations": len(bad_commit), "amendments": [a.get("id") for a in amendments["amendments"]],
            "not_clean": dirty[:20], "n_not_clean": len(dirty), "stamp_violations": bad_stamp[:20],
            "n_stamp_violations": len(bad_stamp), "protocol": protocol, "spec_sha256": spec_sha,
            "n_records_used": len(records), "n_expected": None if want is None else len(want),
            "missing": missing[:50], "n_missing": len(missing), "unexpected": screened["unexpected"][:20],
            "n_unexpected": len(screened["unexpected"]), "role_mismatch": screened["role_mismatch"][:20],
            "n_role_mismatch": len(screened["role_mismatch"]), "duplicates": screened["duplicates"][:20],
            "n_duplicates": len(screened["duplicates"]), "candidates_incomplete": incomplete[:20],
            "n_candidates_incomplete": len(incomplete), "by_role_status": dict(sorted(roles.items())),
            "errors": errors[:50], "errors_not_listed_persistent": errors_unlisted[:50],
            "infeasible": sorted(r["key"] for r in records if _status(r) == "infeasible")[:50],
            "dataset_hash_mismatch": hash_mismatch[:20], "n_dataset_hash_mismatch": len(hash_mismatch),
            "dataset_hash_unstamped": sorted(hash_unstamped)[:20], "n_dataset_hash_unstamped": len(hash_unstamped),
            "n_dataset_hash_unstamped_not_ok": len(hash_unstamped_not_ok),
            "multi_platform_datasets": multi_platform[:20],
            "pkgs_sets": len(pkgs), "fmax_dependence_sha256": (dep or {}).get("_sha"),
            "by_recheck": {"records": by_n, "agree": by_agree, "disagree_first": by_bad},
            "tune_reproducibility_vs_dev": repro}


def cost_table(records: list[dict], cells: dict, spec: dict) -> list[dict]:
    """V10 per (arm, world, regime, n). ``budget`` = the arm's budget field (``cost_field``): GPU arms (R-41) are
    budgeted and read in wall-s, reported apart from the CPU-s of the CPU arms."""
    g: dict[tuple, dict] = defaultdict(lambda: {"cpu": [], "wall": [], "rss": [], "gpu": [], "infeasible": [],
                                                "errors": 0})
    for r in records:
        u = unit_of(r)
        k = (u["arm"], u["world"], u["regime"], u["n"])
        st = _status(r)
        if st == "ok":
            for f, dst in (("cpu_s", "cpu"), ("peak_rss_mb", "rss"), ("gpu_s", "gpu")):
                if r.get(f) is not None:
                    g[k][dst].append(float(r[f]))
            if _wall(r) is not None:
                g[k]["wall"].append(_wall(r))
        elif st == "infeasible":
            g[k]["infeasible"].append({"seed": u["seed"], "cpu_s": r.get("cpu_s"), "wall_s": r.get("wall_s"),
                                       "reason": r.get("reason")})
        else:
            g[k]["errors"] += 1
    arms = spec["arms"]
    out = [{"arm": k[0], "world": k[1], "regime": k[2], "n": k[3], "budget": cost_field(arms[k[0]]),
            "n_runs": len(v["cpu"]), "cpu_s_mean": float(np.mean(v["cpu"])) if v["cpu"] else None,
            "cpu_s_max": max(v["cpu"]) if v["cpu"] else None,
            "wall_s_mean": float(np.mean(v["wall"])) if v["wall"] else None,
            "wall_s_max": max(v["wall"]) if v["wall"] else None, "peak_rss_mb_max": max(v["rss"]) if v["rss"] else None,
            "gpu_s_mean": float(np.mean(v["gpu"])) if v["gpu"] else None,
            "n_infeasible": len(v["infeasible"]), "infeasible": v["infeasible"][:5], "n_errors": v["errors"],
            "t3": False} for k, v in sorted(g.items()) if k[0] in arms]
    t3 = {(e["arm"], e["world"], e["regime"], e["n"]): e for e in cells.values() if e["status"] == "infeasible_t3"}
    for k, e in sorted(t3.items()):
        cf = cost_field(arms[k[0]])
        out.append({"arm": k[0], "world": k[1], "regime": k[2], "n": k[3], "budget": cf, "n_runs": 0,
                    "cpu_s_mean": None, "cpu_s_max": None, "wall_s_mean": None, "wall_s_max": None,
                    "peak_rss_mb_max": None, "gpu_s_mean": None, "n_infeasible": 0, "infeasible": [], "n_errors": 0,
                    "t3": True, f"t3_dev_cost_{cf}": e.get(f"infeasible_cost_{cf}")})
    return out


def analyse(records: list[dict], spec: dict, freeze_commit: str | None = None, *, dep: dict | None = None,
            spec_sha: str | None = None, protocol: dict | None = None, amendments: dict | None = None,
            dev_records: list[dict] | None = None) -> dict:
    use, screened = screen(records, spec)
    cells, rows = build_cells(use, spec)
    v11 = integrity(use, screened, spec, freeze_commit, spec_sha=spec_sha, protocol=protocol, amendments=amendments,
                    dep=dep, dev_records=dev_records)
    return assemble(cells, rows, use, spec, dep, v11)


def assemble(cells: dict, rows: dict, records: list[dict], spec: dict, dep: dict | None, v11: dict) -> dict:
    """Every table (V0-V10) from the cells / rows of ``build_cells`` and the screened ``records`` (V10 reads only
    their status / cost fields), with the given V11. ``analyse`` = screen + build_cells + integrity + this; the
    report generator (eval_report.py) builds cells arm by arm and calls this once."""
    focal = focal_arm(spec)
    prim = {k: e for k, e in cells.items() if e["kappa"] == PRIMARY_KAPPA}
    v1 = [_cell_row(k, e) for k, e in sorted(prim.items())]
    v2 = []
    for k, e in sorted(prim.items()):
        p = e.get("primary") or {}
        st = e["status"] if e["status"] not in COUNTED else (p.get("validity") or "none")
        v2.append({"key": k, "arm": e["arm"], "block": e["block"], "world": e["world"], "regime": e["regime"],
                   "lam": e["lam"], "n": e["n"], "state": st,
                   "power_not_applicable": bool(e.get("power_not_applicable")),
                   "recall": p.get("recall") if powered(e) else None, "fdp": p.get("fdp") if powered(e) else None,
                   "sign_acc": p.get("sign_acc") if powered(e) else None})
    with_info = {native_partner(a, spec) for a, d in spec["arms"].items() if arm_kind(d) in ("eq", "eq_min")}
    v5 = [_cell_row(k, e) | {"recall": (e.get("primary") or {}).get("recall")}
          for k, e in sorted(prim.items()) if e["regime"] == "R2"
          and (e["kind"] in ("pmrt", "eq", "eq_min") or e["arm"] in with_info)]
    v6 = []
    for k, e in sorted(prim.items()):
        if e["world"] != "E4":
            continue
        p = e.get("primary") or {}
        v6.append(_cell_row(k, e) | {"recall": p.get("recall"), "sign_acc": p.get("sign_acc"),
                                     "wrong_sign_rate": p.get("wrong_sign_rate")})
    v7 = [{"key": k, "nt_true": e["primary"]["nt_true"], "nt_null": e["primary"]["nt_null"],
           "n_not_applicable": e["primary"]["n_not_applicable"], "n_seeds": e["primary"]["n_seeds"]}
          for k, e in sorted(cells.items()) if e.get("primary") and (e["primary"]["nt_true"] or e["primary"]["nt_null"]
                                                                     or e["primary"]["n_not_applicable"])]
    v8 = [{"key": k, "state": e["primary"]["validity"] if counts(e) else e["status"], **e["primary"]["secondary"]}
          for k, e in sorted(prim.items()) if e.get("primary") and e["primary"]["secondary"]["recall"] is not None]
    v9 = [{"key": k, "arm": e["arm"], "world": e["world"], "regime": e["regime"], "lam": e["lam"], "n": e["n"],
           "kappa": e["kappa"], "status": e["status"], "validity": (e.get("primary") or {}).get("validity"),
           "recall": (e.get("primary") or {}).get("recall")}
          for k, e in sorted(cells.items())
          if e["regime"] == "R2" and e["n"] == SWEEP_N and e["kappa"] in SWEEP_KAPPAS]
    return {"analysis": ANALYSIS_VERSION, "spec_name": spec.get("name"), "focal": focal, "alpha": ALPHA,
            "valid_ub": VALID_UB,
            "boot_reps": BOOT_REPS, "boot_seed": BOOT_SEED, "min_flag_seeds": MIN_FLAG_SEEDS,
            "primary_kappa": PRIMARY_KAPPA, "fmax_nsim": FMAX_NSIM, "fmax_seed": FMAX_SEED,
            "V0_like_for_like": like_for_like(prim, focal), "V1_validity": v1, "V2_recall": v2,
            "V3_paired": paired_diffs(prim, focal=focal), "V3_paired_valid_only": paired_diffs(prim, "VALID", focal),
            "V4_verdicts": verdicts(cells, rows, spec, dep), "V5_information_R2": v5, "V6_E4": v6,
            "V7_not_testable": v7, "V8_secondary": v8, "V9_kappa_sweep": v9,
            "V10_cost": cost_table(records, cells, spec), "V11_integrity": v11,
            "secondary_tau_of_p_arms": {k: e["secondary_tau"]["validity"] for k, e in sorted(prim.items())
                                        if e.get("secondary_tau")}}


# ================================================================================================ markdown
def _f(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    s = f"{x:.{d}f}"
    return s.replace("0.", ".", 1) if abs(x) < 1 else s


def _wr(e: dict) -> str:
    return f"{e['world']} {e['regime']}" + ("" if e["lam"] is None else f" l{e['lam']:g}")


def _arms(out: dict) -> list[str]:
    seen = []
    for r in out["V1_validity"]:
        if r["arm"] not in seen:
            seen.append(r["arm"])
    focal = out.get("focal") or FOCAL
    return sorted(seen, key=lambda a: (a != focal, "pmrt" not in a, a))


def _pooled_txt(pooled: dict) -> str:
    return "; ".join(f"{k} {_f(p['rate'], 3)} [{_f(p['ci'][0], 3)}, {_f(p['ci'][1], 3)}] {p['validity']}"
                     for k, p in pooled.items() if p)


_STATE_TAG = {"INVALID": "inv", "NO_READ": "nr", "infeasible": "inf", "infeasible_t3": "inf", "untuned": "unt",
              "missing": "mis", "few_seeds": "few"}
_RATE_MARK = {"VALID": "", "INVALID": "*", "INCONCLUSIVE": "~", None: "?"}


def _md_like(rows: list[dict], arms: list[str]) -> list[str]:
    """V0 markdown: one table per (world, regime, lambda); rows arm x rule, columns n."""
    def rt(x: dict | None) -> str:
        return "-" if not x else _f(x["rate"], 3) + _RATE_MARK.get(x["validity"], "?")

    def txt(r: dict | None) -> str:
        if r is None:
            return "-"
        if r["recall"] is None and r["null"] is None and r["placebo"] is None:
            return _STATE_TAG.get(r["state"], r["state"])
        rec = "NA" if r["power_not_applicable"] else _f((r["recall"] or {}).get("mean"))
        return f"{rec} / {rt(r['null'])} / {rt(r['placebo'])}"
    L = ["## V0 like-for-like (headline, R-42): recall / truth-null rate / placebo rate per cell", "",
         "Every p arm scored at raw p <= .05 per edge (raw_p) and with the conformal placebo tau (tau), next to the "
         "score-only arms (tau; cdl also at the conference's fixed threshold, fixed). Rates are declaration rates; * INVALID, ~ INCONCLUSIVE; a tau row's placebo is "
         "its tuning column (reported, not in its validity). NA = not applicable (PMRT in R4: no known design, "
         "never recall 0); inv INVALID, inf infeasible (EVAL or T3), mis missing, unt untuned, few < 10 seeds."]
    ns = sorted({r["n"] for r in rows})
    idx = {(_wr(r), r["arm"], r["rule"], r["n"]): r for r in rows}
    order = sorted({(r["world"], r["regime"], -1.0 if r["lam"] is None else r["lam"], _wr(r)) for r in rows})
    for *_, w in order:
        L += ["", f"### {w}", "", "| arm | rule | " + " | ".join(f"n {n}" for n in ns) + " |",
              "|---|---|" + "---|" * len(ns)]
        for a in arms:
            for rule in ("raw_p", "tau", "fixed"):
                if not any((w, a, rule, n) in idx for n in ns):
                    continue
                L.append(f"| {a} | {rule} | " + " | ".join(txt(idx.get((w, a, rule, n))) for n in ns) + " |")
    return L


def markdown(out: dict) -> str:
    v11 = out["V11_integrity"]
    arms = _arms(out)
    focal = out.get("focal") or FOCAL
    L = [f"# Study A EVAL tables ({v11['label']})", "",
         f"`{out['analysis']}`, spec `{out['spec_name']}`, primary kappa {out['primary_kappa']}, primary PMRT arm "
         f"{focal}; commits {', '.join(c[:9] for c in v11['commits'])}. Protocol: docs/xmethod/PROTOCOL_A.md.", ""]
    L += _md_like(out.get("V0_like_for_like", []), arms) + [""]
    v4 = out["V4_verdicts"]
    c1 = v4["C1"]
    L += ["## V4 claim verdicts", "", f"- **Claim: {v4['claim']}** (components {v4['components']})",
          f"- C1: **{c1['verdict']}** ({c1['n_failures']} / {len(c1['D_counted'])} counted arms of D are "
          f"design-blind failures; |D| = {len(c1['D'])}). Wording: {c1['wording']}."]
    lab = v4.get("arm_labels", {})
    if c1.get("sensitivity_without"):
        sw = c1["sensitivity_without"]
        L.append(f"- C1 sensitivity without {', '.join(sw['dropped'])} (R-56, no effect): **{sw['verdict']}** "
                 f"({sw['n_failures']} / {len(sw['D_counted'])} counted arms fail)")
    L += [f"- Arm labels (R-40; every table): {a}: {x}" for a, x in lab.items()]

    def tag(a: str) -> str:
        return f" [{lab[a]}]" if a in lab else ""
    for a, v in [*c1["arms"].items(), *c1.get("descriptive", {}).items()]:
        nd = " (not in D, descriptive)" if a not in c1["arms"] else ""
        L.append(f"  - {a}{tag(a)}{nd}: **{v['verdict']}**; R2 INVALID {v['r2_invalid']}/{v['r2_planned']} (counted "
                 f"{v['r2_counted']}), pooled R2 {_pooled_txt(v['pooled_r2'])}; R1 INVALID {v['r1_invalid']}/"
                 f"{v['r1_counted']} (F_max {v['r1_f_max']})")

    def comp(name: str, x: dict, indent: str = "") -> None:
        L.append(f"{indent}- {name}: **{x['verdict']}**")
        if "n_cells" in x:
            L.append(f"{indent}  - cells {x['n_cells']}/{x['n_planned']}, INVALID {x['n_invalid']} (F_max "
                     f"{x['f_max']}), VALID {x['n_valid']}, INCONCLUSIVE {x['n_inconclusive']}; pooled "
                     f"{_pooled_txt(x['pooled'])}" + (f"; not counted {len(x['not_counted'])}"
                                                      if x["not_counted"] else ""))
    comp(f"C2a {focal} valid in R1 / R2", v4["C2a"])
    L.append(f"- C2b eq arms whose native partner is a C1 failure: **{v4['C2b']['verdict']}**")
    for a, x in v4["C2b"]["arms"].items():
        comp(f"{a}{tag(a)} (native {x['native']})", x, "  ")
    for a, x in v4["C2b"].get("excluded", {}).items():
        comp(f"{a} EXCLUDED from C2b (R-40) [{x['label']}], same rule reported (no effect)", x, "  ")
    w2 = v4["C2b"].get("with_r1_failing_partners")
    if w2:
        L.append(f"- C2b with the R1-failing partners (R-56, sensitivity, no effect): **{w2['verdict']}** (added: "
                 f"{', '.join(w2['added']) or 'none'})")
    L.append(f"- C2 wording: {v4['C2_wording']}.")
    for x in v4.get("C2_named", []):
        L.append(f"  - {x['sentence']}.")
    if v4.get("eq_arms_table"):
        L += ["", "Every eq arm, same rule on R1 + R2, unfiltered (R-56):", "",
              "| eq arm | C2b membership | native C1 | verdict | INVALID / cells (F_max) | R2 INVALID / planned | "
              "pooled R1 truth-null | pooled R2 truth-null |", "|---|---|---|---|---|---|---|---|"]
        for a, x in v4["eq_arms_table"].items():
            r2 = "-" if x["r2_invalid"] is None else f"{x['r2_invalid']}/{x['r2_planned']}"
            r1t = _rate_txt((x["pooled_r1"] or {}).get("null_raw")) if x["pooled_r1"] else "-"
            r2t = _rate_txt((x["pooled_r2"] or {}).get("null_raw")) if x["pooled_r2"] else "-"
            L.append(f"| {a}{tag(a)} | {x['membership']} | {x['native_c1'] or '-'} | {x['verdict']} | "
                     f"{x['n_invalid']}/{x['n_cells']} ({x['f_max']}) | {r2} | {r1t} | {r2t} |")
        L.append("")
    comp(f"C3 {focal} valid in E4 R3", v4["C3"])
    if v4["C3"].get("wording"):
        L.append(f"  - wording: {v4['C3']['wording']}.")
    for a, x in v4.get("pmrt_secondary", {}).items():
        comp(f"{a}{tag(a)} (secondary PMRT arm, same rules reported, no claim effect): C2a", x["C2a"])
        comp(f"{a}: C3", x["C3"], "  ")
    L.append("- C3 reported for every eq arm (no claim effect): " + ("; ".join(
        f"{a}{tag(a)} {x['verdict']}" for a, x in v4["C3_eq_arms"].items()) or "none"))
    # V1
    L += ["", "## V1 validity: INVALID / VALID / counted cells per world-regime (all n)", "",
          "| world regime | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    cnt: dict[tuple, list] = defaultdict(lambda: [0, 0, 0, 0])
    for r in out["V1_validity"]:
        c = cnt[(_wr(r), r["arm"])]
        if r["status"] in COUNTED and r["validity"] not in (None, "few_seeds"):
            c[2] += 1
            c[0] += r["validity"] == "INVALID"
            c[1] += r["validity"] == "VALID"
        else:
            c[3] += 1
    for w in sorted({_wr(r) for r in out["V1_validity"]}):
        cells = []
        for a in arms:
            c = cnt.get((w, a))
            cells.append("-" if not c else (f"**{c[0]}**" if c[0] else "0") + f"/{c[1]}/{c[2]}"
                         + (f" ({c[3]}x)" if c[3] else ""))
        L.append(f"| {w} | " + " | ".join(cells) + " |")
    L += ["", "(Nx) = planned cells not counted (missing, infeasible, T3-infeasible, untuned or < 10 seeds)."]
    flagged = [r for r in out["V1_validity"] if r["validity"] == "INVALID"]
    L += ["", f"INVALID cells ({len(flagged)}; first 30):"]
    for r in flagged[:30]:
        bad = [f"{k} {_f(r[k]['rate'], 3)} [{_f(r[k]['ci'][0], 3)}, {_f(r[k]['ci'][1], 3)}]"
               for k in RATE_KEYS if r.get(k) and r[k]["validity"] == "INVALID"]
        L.append(f"- {r['key']}: " + "; ".join(bad))
    nc = [r for r in out["V1_validity"] if r["status"] not in COUNTED]
    if nc:
        L += ["", f"Cells not counted ({len(nc)}; first 30):"]
        L += [f"- {r['key']}: {r['status']}" + "".join(f" (cost {_f(r[f'infeasible_cost_{f}'], 0)} {u})"
                                                       for f, u in (("cpu_s", "CPU-s"), ("wall_s", "GPU wall-s"))
                                                       if r.get(f"infeasible_cost_{f}") is not None)
              for r in nc[:30]]
    # V2
    ns = sorted({r["n"] for r in out["V2_recall"]})
    tag = _STATE_TAG
    L += ["", "## V2 recall among cells not INVALID (n " + " / ".join(map(str, ns)) + "; inv INVALID, "
          "inf infeasible, mis missing, unt untuned, few < 10 seeds; NA not applicable, R-42)", "",
          "| world regime | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    idx = {(r["arm"], _wr(r), r["n"]): r for r in out["V2_recall"]}
    for w in sorted({_wr(r) for r in out["V2_recall"]}):
        cells = []
        for a in arms:
            v = []
            for n in ns:
                r = idx.get((a, w, n))
                v.append("-" if r is None else _f(r["recall"]["mean"]) if r["recall"]
                         else "NA" if r.get("power_not_applicable") and r["state"] not in tag else tag.get(r["state"], "-"))
            cells.append("-" if all(x == "-" for x in v) else "/".join(v))
        L.append(f"| {w} | " + " | ".join(cells) + " |")
    # V3
    L += ["", f"## V3 paired recall difference {focal} - arm (neither INVALID, same declaration rule; per-cell 95 % "
          "paired-t CI, descriptive)", "", "| arm | rule | block | cells pmrt higher | pmrt lower | CI covers 0 |",
          "|---|---|---|---|---|---|"]
    by_arm: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])
    for d in out["V3_paired"]:
        c = by_arm[(d["arm"], d["rule"], d["block"])]
        c[0 if d["ci"][0] > 0 else 1 if d["ci"][1] < 0 else 2] += 1
    L += [f"| {a} | {r} | {b} | {c[0]} | {c[1]} | {c[2]} |" for (a, r, b), c in sorted(by_arm.items())]
    mark = {"VALID": "", "INVALID": "*", "INCONCLUSIVE": "~", "NO_READ": "~", "few_seeds": "?"}

    def cellv(r: dict, val: str) -> str:
        if r["status"] not in COUNTED:
            return tag.get(r["status"], r["status"])
        return val + mark.get(r.get("validity"), "?")

    def grid(title: str, rows: list[dict], rowkey, val) -> None:
        cols = [a for a in arms if any(r["arm"] == a for r in rows)]
        g = {(rowkey(r), r["arm"]): cellv(r, val(r)) for r in rows}
        L.extend(["", title, "", "| row | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)])
        order = sorted(rows, key=lambda r: (r["world"], r["regime"], -1 if r["lam"] is None else r["lam"],
                                            r["kappa"], r["n"]))
        for rk in dict.fromkeys(rowkey(r) for r in order):
            L.append(f"| {rk} | " + " | ".join(g.get((rk, a), "-") for a in cols) + " |")

    def rec(r: dict) -> str:
        return _f((r.get("recall") or {}).get("mean"))

    grid("## V5 information levels in R2 (pmrt, eq, eq_min, native): recall (* INVALID, ~ INCONCLUSIVE)",
         out["V5_information_R2"], lambda r: f"{r['world']} n{r['n']}", rec)
    grid("## V6 E4: placebo_conf rate (raw p, else declared) / wrong-sign rate of the true edge (* INVALID)",
         [r for r in out["V6_E4"] if r["n"] in (1000, 4000, 24000)],
         lambda r: f"{r['regime']} l{r['lam']:g} n{r['n']}",
         lambda r: f"{_f(((r.get('conf_raw') or r.get('conf_decl')) or {}).get('rate'))}/"
                   f"{'NA' if r.get('power_not_applicable') else _f(r['wrong_sign_rate'])}")
    L += ["", "## V7 not testable / not applicable", "", f"{len(out['V7_not_testable'])} (arm, cell) with counts > 0 "
          "(JSON V7).", "", "## V8 secondary KPI -> KPI family", "",
          f"{len(out['V8_secondary'])} (arm, cell) rows (JSON V8)."]
    grid("## V9 kappa sweep (R2, n 1000): recall", out["V9_kappa_sweep"], lambda r: f"{r['world']} k{r['kappa']:g}",
         rec)
    ns_c = sorted({r["n"] for r in out["V10_cost"]})
    budget = {r["arm"]: r["budget"] for r in out["V10_cost"]}
    for f, title in (("cpu_s", "CPU-s per dataset"),
                     ("wall_s", "GPU arms (R-41): wall-s per dataset on the GPU host, reported apart from CPU")):
        tab = [a for a in arms if budget.get(a, "cpu_s") == f]
        if not tab:
            continue
        L += ["", f"## V10 cost ({title}; mean / max over worlds, regimes; peak RSS MB; infeasible units; "
              "T3 = infeasible from DEV cost)", ""]
        L += ["| arm | " + " | ".join(f"n {n}" for n in ns_c) + " | RSS | infeasible |",
              "|---|" + "---|" * (len(ns_c) + 2)]
        for a in tab:
            rs = [r for r in out["V10_cost"] if r["arm"] == a]
            cells = []
            for n in ns_c:
                m = [r[f"{f}_mean"] for r in rs if r["n"] == n and r[f"{f}_mean"] is not None]
                x = [r[f"{f}_max"] for r in rs if r["n"] == n and r[f"{f}_max"] is not None]
                cells.append(f"{np.mean(m):.1f} / {max(x):.1f}" if m
                             else "T3" if any(r["n"] == n and r["t3"] for r in rs) else "-")
            rss = max([r["peak_rss_mb_max"] or 0 for r in rs], default=0)
            L.append(f"| {a} | " + " | ".join(cells) + f" | {rss:.0f} | {sum(r['n_infeasible'] for r in rs)} |")
    b = v11["by_recheck"]
    L += ["", "## V11 integrity", "", f"- label {v11['label']}; failed checks: {', '.join(v11['failed']) or 'none'}",
          f"- freeze commit {v11['freeze_commit']}; amendments {v11['amendments']}; records used "
          f"{v11['n_records_used']} / expected {v11['n_expected']}; missing {v11['n_missing']}; unexpected "
          f"{v11['n_unexpected']}; role mismatch {v11['n_role_mismatch']}; duplicates {v11['n_duplicates']}",
          f"- commit violations {v11['n_commit_violations']}; not clean {v11['n_not_clean']}; stamp violations "
          f"{v11['n_stamp_violations']}; candidates incomplete {v11['n_candidates_incomplete']}; pkgs sets "
          f"{v11['pkgs_sets']}",
          f"- role:status {v11['by_role_status']}; errors {len(v11['errors'])} (not listed persistent "
          f"{len(v11['errors_not_listed_persistent'])}); infeasible {len(v11['infeasible'])}; dataset-hash "
          f"mismatches {v11['n_dataset_hash_mismatch']} (R-41a: equal across all arms of a dataset); ok records "
          f"without a dataset hash {v11['n_dataset_hash_unstamped']} (not-ok {v11['n_dataset_hash_unstamped_not_ok']}); "
          f"multi-platform datasets "
          f"{len(v11['multi_platform_datasets'])}",
          f"- BY recheck (p arms): {b['agree']} / {b['records']} records agree; tune reproducibility vs DEV: "
          f"{v11['tune_reproducibility_vs_dev']}"]
    return "\n".join(L) + "\n"


# ================================================================================================ T1 (freeze)
S_FLOOR, S_CAP, T1_NS = 40, 60, (500, 1000, 4000)       # T6 cap 60 = stated compute ceiling (R-56)


def achieved_power(sd_d: float, s: int, delta: float = 0.15, alpha: float = 0.05) -> float:
    """Power of the two-sided paired t-test with s seeds for a mean gap ``delta`` (noncentral t)."""
    if not np.isfinite(sd_d) or sd_d <= 0:
        return 1.0
    df, nc = s - 1, delta * math.sqrt(s) / sd_d
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))


def t1_seed_count(records: list[dict], spec: dict, s_cap: int = S_CAP, focal: str | None = None) -> dict:
    """PROTOCOL_A T1 on the DEV merged records, with this module's own definitions: per kappa .25 cell at n in
    T1_NS (E4 excluded: one true edge), every V3 pair (pmrt_eq vs a primary-block arm, same declaration rule,
    neither arm INVALID on DEV) gives paired_seeds(sd_d; delta .15, alpha .05, power .8); S_power = the max;
    S = min(s_cap, max(40, S_power rounded up to a multiple of 10)), cap 60 (R-56). Also the minimum detectable gap
    and the achieved power (gap .15) at S per pair: reported when the cap binds (R-56).
    ``focal``: the PMRT arm of the pairs (default ``focal_arm(spec)``)."""
    focal = focal or focal_arm(spec)
    use, _ = screen(records, spec)
    cells, _ = build_cells(use, spec)
    pairs = []
    for k, e in sorted(cells.items()):
        if (e["arm"] == focal or e["block"] != "primary" or e["kappa"] != PRIMARY_KAPPA or e["n"] not in T1_NS
                or e["world"] == "E4" or not counts(e) or e["primary"]["validity"] == "INVALID"):
            continue
        f = cells.get(f"{focal}|{k.split('|', 1)[1]}")
        if f is None or not counts(f) or f["primary"]["validity"] == "INVALID":
            continue
        fp = f["primary"] if e["declare"] == "by" else f.get("secondary_tau")
        if not fp or fp["validity"] in ("INVALID", "few_seeds"):
            continue
        d = _paired(fp["per_seed_recall"], e["primary"]["per_seed_recall"])
        if len(d) < 2:
            continue
        sd = float(np.std(d, ddof=1))
        pairs.append({"cell": k.split("|", 1)[1], "arm": e["arm"], "n_seeds": len(d), "sd_d": sd,
                      "seeds_needed": paired_seeds(sd)})
    s_power = max((p["seeds_needed"] for p in pairs), default=None)
    s = S_FLOOR if s_power is None else min(s_cap, max(S_FLOOR, 10 * math.ceil(s_power / 10)))
    for p in pairs:
        p["mdg_at_S"] = min_detectable_gap(p["sd_d"], s) if p["sd_d"] > 0 else 0.0
        p["power_at_S"] = achieved_power(p["sd_d"], s)
    worst = sorted(pairs, key=lambda p: (-p["seeds_needed"], p["cell"], p["arm"]))[:15]
    return {"S": s, "focal": focal, "S_power": s_power, "cap": s_cap, "cap_binds": bool(s_power is not None and s_power > s_cap),
            "n_pairs": len(pairs), "largest": worst,
            "max_mdg_at_S": max((p["mdg_at_S"] for p in pairs), default=None),
            "min_power_at_S": min((p["power_at_S"] for p in pairs), default=None)}


# ================================================================================================ main
def _clean(x):
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return _clean(x.item())
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write(out: dict, out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    js = json.dumps(_clean(out), indent=1, sort_keys=True, allow_nan=False) + "\n"
    md = markdown(out)
    pj, pm = os.path.join(out_dir, "eval_tables.json"), os.path.join(out_dir, "EVAL_TABLES.md")
    with open(pj, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(js)
    with open(pm, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)
    return hashlib.sha256(js.encode()).hexdigest(), hashlib.sha256(md.encode()).hexdigest()


def _is_eval(spec: dict) -> bool:
    return str(spec.get("name", "")).lower().startswith("eval")


def relabel_tune(records: list[dict], seeds: set[int], spec: dict) -> tuple[list[dict], dict]:
    """DEV pilot testing only: records and spec blocks of ``seeds`` become tune (refused on an EVAL spec)."""
    if _is_eval(spec):
        raise SystemExit("--as-tune is for DEV pilot testing only; refused on an EVAL spec")
    blocks = []
    for b in spec["blocks"]:
        ss = _seeds(b["seeds"])
        if b["role"] == "measure" and set(ss) & seeds:
            blocks += [{**b, "role": "tune", "seeds": sorted(set(ss) & seeds)}]
            rest = sorted(set(ss) - seeds)
            blocks += [{**b, "seeds": rest}] if rest else []
        else:
            blocks.append(b)
    return ([{**r, "role": "tune"} if int(r["job"]["seed"]) in seeds else r for r in records],
            {**spec, "blocks": blocks})


def _load_many(paths: list[str]) -> list[dict]:
    return sorted((r for p in paths for r in load_records(p)),
                  key=lambda r: (r.get("key", ""), json.dumps(r, sort_keys=True)))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["t1"]:                    # eval_analysis.py t1 --spec DEV_SPEC --merged DEV.jsonl.gz [...]
        p = argparse.ArgumentParser(prog="eval_analysis t1")
        p.add_argument("--spec", required=True, nargs="+", help="DEV specs; arms and blocks are united")
        p.add_argument("--merged", required=True, nargs="+")
        p.add_argument("--cap", type=int, default=S_CAP)
        p.add_argument("--focal", default=None, help="PMRT arm of the pairs (default: a spec's focal, else pmrt_eq)")
        a = p.parse_args(argv[1:])
        specs = [json.load(open(x, encoding="utf-8")) for x in a.spec]
        spec = {"name": "+".join(str(x.get("name")) for x in specs),
                "focal": next((x["focal"] for x in specs if x.get("focal")), None),
                "arms": {k: v for x in specs for k, v in x["arms"].items()},
                "blocks": [{**b, "arms": b.get("arms", "all") if b.get("arms", "all") != "all" else list(x["arms"])}
                           for x in specs for b in x["blocks"]]}
        res = t1_seed_count(_load_many(a.merged), spec, a.cap, a.focal)
        res["inputs"] = {"spec": [[x, file_sha256(x)] for x in a.spec], "merged": [[m, file_sha256(m)] for m in a.merged]}
        print(json.dumps(_clean(res), indent=1))
        return 0
    ap = argparse.ArgumentParser(prog="eval_analysis")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--merged", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze-commit", default=None)
    ap.add_argument("--amendments", default=None)
    ap.add_argument("--protocol", default=PROTOCOL_PATH)
    ap.add_argument("--dependence", default=DEP_PATH)
    ap.add_argument("--dev-merged", default=None, nargs="+", help="DEV records: tune reproducibility check (V11)")
    ap.add_argument("--as-tune", default=None, help="comma-separated seeds relabelled tune (DEV testing only)")
    ap.add_argument("--min-flag-seeds", type=int, default=None, help="DEV testing only (EVAL: fixed at 10)")
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec, encoding="utf-8"))
    recs = _load_many(a.merged)
    if a.as_tune:
        recs, spec = relabel_tune(recs, {int(s) for s in a.as_tune.split(",")}, spec)
    if a.min_flag_seeds is not None:
        if _is_eval(spec):
            raise SystemExit("--min-flag-seeds is for DEV testing only; refused on an EVAL spec")
        global MIN_FLAG_SEEDS
        MIN_FLAG_SEEDS = a.min_flag_seeds
    out = analyse(recs, spec, a.freeze_commit, dep=load_dependence(a.dependence), spec_sha=file_sha256(a.spec),
                  protocol=protocol_check(spec, a.protocol), amendments=load_amendments(a.amendments),
                  dev_records=_load_many(a.dev_merged) if a.dev_merged else None)
    hj, hm = write(out, a.out)
    print(f"[eval] {out['V11_integrity']['label']}: claim {out['V4_verdicts']['claim']} "
          f"{out['V4_verdicts']['components']}; {len(out['V1_validity'])} primary cells; json {hj[:12]} md {hm[:12]}"
          f" -> {a.out}")
    return 0 if out["V11_integrity"]["label"] == "FINAL" else 2


if __name__ == "__main__":
    sys.exit(main())

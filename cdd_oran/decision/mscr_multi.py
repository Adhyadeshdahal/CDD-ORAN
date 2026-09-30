"""mscr-multi-v1: the DECLARATION layer of MSCR+ (multiplicity over the 60 E6-P hypotheses; MSCR+ agent M, 2026-09-30,
NOT frozen). Statistic-agnostic: it takes per-hypothesis p-values (two-sided p2 and, optionally, the one-sided p_plus /
p_minus of the SAME statistic) and / or e-values, and returns declarations {(f, rel, kpi): {"declared", "sign", ...}}
in edge_score's format.

Every procedure below controls its error rate under ARBITRARY dependence between the hypotheses' p- / e-values (the 60
CRT statistics share the family's draws and the outcome windows: no PRDS argument is available), provided
  (V1) each p_j is a valid p-value under H0_j (P(p_j <= t) <= t) and each e_j a valid e-value (E e_j <= 1), and
  (V2) every weight, direction, ordering and graph is FIXED before the test data are seen (a function of independent
      data: here the ev2 episodes, disjoint from the ev3 evaluation episodes, and of the knob / KPI names only).
Weights, directions and gates computed from the SAME data (e.g. the same-data load z) would break (V2): the load and pv
statistics share the assignment vector v, so a data-driven weight is correlated with the p-value it scales.

Procedures (``PROCEDURES``; validity condition in brackets):
  by           Benjamini-Yekutieli step-up (the v2 default)                          [FDR <= q; arbitrary dependence]
  bh_invalid   Benjamini-Hochberg: REFERENCE ONLY (needs PRDS, unproven here)       [not valid under arbitrary dep.]
  wby          weighted BY (Blanchard & Roquain 2008 weighted step-up with the BY reshaping beta(k) = k / c_m;
               Roeder & Wasserman 2009 weights, see ``prior_weights``): reject p_j <= q w_j beta(k) / m, sum w = m
                                                                                    [FDR <= q; arbitrary dependence]
  by1s         BY where hypothesis j uses the ONE-SIDED p in the prior direction d_j when |z_prior,j| >= thr (declared
               sign = d_j), the two-sided p2 otherwise. A one-sided p is valid under the sharp null whatever the
               direction, so (V1) holds for the mixed vector                        [FDR <= q; arbitrary dependence]
  wby1s        wby + by1s
  dagger       reshaped DAGGER (Ramdas, Chen, Wainwright & Jordan 2019, Biometrika, Thm 1(b); eq. 5b / 6), mediator-
               first DAG: the 12 (f, rel, load) hypotheses are the roots, each gates its 4 (f, rel, {pv, v, e, rlf})
               children (a child is testable only if its load was rejected). Reshaping measure nu ~ 1/k on
               {1..N}: beta(x) = min(x, N) / c_N (any fixed nu is allowed)       [FDR <= q; arbitrary dependence]
  dagger1s     dagger on the one-sided-where-directed p-values
  gholm        graphical Bonferroni-Holm (Bretz, Maurer, Brannath & Posch 2009): initial alpha on the 12 loads, a
               rejected load passes its alpha to its 4 children, children pass to siblings (1 - delta) and to the other
               loads (delta)                                                        [FWER <= q; arbitrary dependence]
  gholm1s      gholm on the one-sided-where-directed p-values
  pfilter_w1s  reshaped two-layer p-filter (Ramdas, Barber, Wainwright & Jordan 2019, AoS): layer 1 = the 60 leaves with
               the prior weights, layer 2 = the 12 (f, rel) groups with weighted-Bonferroni group p-values; BY
               reshaping on both layers        [leaf FDR <= q AND group FDR <= q simultaneously; arbitrary dependence]
  ebh          e-BH (Wang & Ramdas 2022) on the two-sided e-values                  [FDR <= q; arbitrary dependence]
  ebh1s        e-BH with the one-sided e-value (prior direction) where |z_prior| >= thr
  webh1s       weighted e-BH (e-BH on w_j e_j, sum w = m; FDR <= q sum_{H0} w_j / m <= q) + one-sided e-values

Prior (``Prior``): signed z of the SAME statistic on independent data (ev2: 480 episodes, sub "v2"). Weights
(``prior_weights``): Roeder-Wasserman optimal weights w_j = (m / alpha) Phibar(xi_j / 2 + c / xi_j) (sum = m) for the
projected mean xi_j = sqrt(max(z_j^2 - 1, 0)) sqrt(n_target / n_prior) (the -1 removes the chi-square bias of z^2),
alpha = the anticipated total BY level q k0 / c_m (k0 = the expected number of rejections under the projection, a fixed
point), capped (mean-preserving) at W_CAP = 5 (RW concentrates all mass on a few moderate xi and is brittle to a wrong
projection), then mixed with a floor: w = floor + (1 - floor) w_RW (mean 1 exactly, every hypothesis keeps >= floor x
its unweighted level). Directions (``prior_directions``): d_j = sign(z_prior,j) if |z_prior,j| >= thr else 0. n_target is
the number of episodes of the test data (known in advance).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.stats import norm

from .edge_score import HYPOTHESES

MULTI_VERSION = "mscr-multi-v1"
KPI_GATE = "load"
W_CAP = 5.0                                       # max prior weight (fixed before any ev3 evaluation)
PROCEDURES = ("by", "bh_invalid", "wby", "by1s", "wby1s", "dagger", "dagger1s", "gholm", "gholm1s", "pfilter_w1s",
              "ebh", "ebh1s", "webh1s")
_SPEC = {  # name: (kind, one_sided, weighted, structure)
    "by": ("p", False, False, None), "bh_invalid": ("p", False, False, "bh"), "wby": ("p", False, True, None),
    "by1s": ("p", True, False, None), "wby1s": ("p", True, True, None), "dagger": ("p", False, False, "dagger"),
    "dagger1s": ("p", True, False, "dagger"), "gholm": ("p", False, False, "gholm"),
    "gholm1s": ("p", True, False, "gholm"), "pfilter_w1s": ("p", True, True, "pfilter"),
    "ebh": ("e", False, False, None), "ebh1s": ("e", True, False, None), "webh1s": ("e", True, True, None)}


# ------------------------------------------------------------------------------------------------ inputs
@dataclasses.dataclass(frozen=True)
class HypStats:
    """Per-hypothesis output of ANY statistic. p2: two-sided p (NaN = untested); p_plus / p_minus: one-sided p for a
    positive / negative effect (NaN = not available); sign: sign of the estimate; e2 / e_plus / e_minus / e_sign: the
    same for e-values (NaN = not available)."""
    p2: float = float("nan")
    sign: int = 0
    p_plus: float = float("nan")
    p_minus: float = float("nan")
    e2: float = float("nan")
    e_plus: float = float("nan")
    e_minus: float = float("nan")
    e_sign: int = 0


def stats_from_json(d: dict) -> dict:
    """{"f|rel|kpi": {p2, p_plus, p_minus, sign, e2, e_plus, e_minus, e_sign, status}} -> {hyp: HypStats}."""
    def num(v, x, ok):
        return float(v[x]) if ok and v.get(x) is not None else float("nan")
    out = {}
    for k, v in d.items():
        ok = v.get("status", "tested") != "undetermined"
        eok = v.get("e_status", "tested") != "undetermined"
        out[tuple(k.split("|"))] = HypStats(
            p2=num(v, "p2", ok), sign=int(v.get("sign", 0) or 0), p_plus=num(v, "p_plus", ok),
            p_minus=num(v, "p_minus", ok), e2=num(v, "e2", eok), e_plus=num(v, "e_plus", eok),
            e_minus=num(v, "e_minus", eok), e_sign=int(v.get("e_sign", 0) or 0))
    return out


def stats_from_crt_v2(run: dict) -> dict:
    """crt_units_v2.run_crt_units_v2 output -> {hyp: HypStats} (two-sided only)."""
    out = {}
    for r in run["results"]:
        ok = r["status"] != "undetermined"
        out[(r["family"], r["relation"], r["kpi"])] = HypStats(p2=float(r["p"]) if ok else float("nan"),
                                                               sign=int(r["sign"]))
    return out


@dataclasses.dataclass(frozen=True)
class Prior:
    """Signed z of the statistic on INDEPENDENT data (``n_prior`` episodes)."""
    z: dict
    n_prior: int
    source: str = ""

    @classmethod
    def from_stats(cls, d: dict, n_prior: int, source: str = "") -> Prior:
        z = {}
        for k, v in d.items():
            x = v.get("z", float("nan")) if v.get("status", "tested") != "undetermined" else float("nan")
            z[tuple(k.split("|"))] = float(x) if x is not None and np.isfinite(x) else 0.0
        return cls(z=z, n_prior=int(n_prior), source=source)


# ------------------------------------------------------------------------------------------------ prior -> weights
def c_harm(m: int) -> float:
    return float(sum(1.0 / k for k in range(1, int(m) + 1)))


def projected_means(z, n_prior: int, n_target: int) -> np.ndarray:
    z = np.nan_to_num(np.asarray(z, float))
    return np.sqrt(np.maximum(z * z - 1.0, 0.0)) * math.sqrt(float(n_target) / float(n_prior))


def rw_weights(xi, alpha: float) -> np.ndarray:
    """Roeder & Wasserman (2009) optimal Bonferroni weights for one-sided means xi at total level alpha: w_j = (m / alpha)
    Phibar(xi_j / 2 + c / xi_j) (0 for xi_j <= 0), c solving sum w = m. All xi <= 0 -> ones."""
    xi = np.asarray(xi, float)
    m = len(xi)
    pos = xi > 1e-9
    if not pos.any():
        return np.ones(m)

    def tot(c):
        w = np.zeros(m)
        w[pos] = (m / alpha) * norm.sf(xi[pos] / 2.0 + c / xi[pos])
        return w

    lo, hi = -1.0, 1.0
    while tot(lo).sum() < m:
        lo *= 2.0
        if lo < -1e6:
            return np.ones(m)
    while tot(hi).sum() > m:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tot(mid).sum() > m:
            lo = mid
        else:
            hi = mid
    w = tot(hi)
    return w * (m / w.sum())


def expected_rejections(xi, q: float, max_iter: int = 100) -> float:
    """Fixed point k = sum_j Phi(xi_j - z_{1 - q k / (m c_m)}) (the anticipated number of BY rejections, >= 1)."""
    xi = np.asarray(xi, float)
    m = len(xi)
    cm = c_harm(m)
    k = 1.0
    for _ in range(max_iter):
        a = min(q * k / (m * cm), 0.5)
        k_new = max(1.0, float(norm.cdf(xi - norm.isf(a)).sum()))
        if abs(k_new - k) < 1e-6:
            break
        k = k_new
    return k


def cap_weights(w, cap: float) -> np.ndarray:
    """Mean-preserving cap: weights above ``cap`` are set to it and the excess is spread proportionally over the
    others (iterated). cap = inf -> unchanged."""
    w = np.asarray(w, float).copy()
    tot = w.sum()
    if not np.isfinite(cap) or cap * len(w) < tot:
        return w
    for _ in range(len(w)):
        hi = w > cap + 1e-12
        if not hi.any():
            break
        w[hi] = cap
        free = w < cap - 1e-12
        rest = tot - w[~free].sum()
        s = w[free].sum()
        w[free] = w[free] * rest / s if s > 0 else rest / max(free.sum(), 1)
    return w


def prior_weights(prior: Prior, n_target: int, q: float = 0.05, floor: float = 0.2, hyps=HYPOTHESES,
                  cap: float = W_CAP) -> dict:
    """Mean-1 weights (module docstring): floor + (1 - floor) x Roeder-Wasserman weights of the projected means, the RW
    part capped (mean-preserving) so that no final weight exceeds ``cap`` (robustness of RW to a wrong projection)."""
    z = np.array([prior.z.get(h, 0.0) for h in hyps])
    xi = projected_means(z, prior.n_prior, n_target)
    k0 = expected_rejections(xi, q)
    w_rw = rw_weights(xi, q * k0 / c_harm(len(hyps)))
    if np.isfinite(cap):
        w_rw = cap_weights(w_rw, (cap - floor) / (1.0 - floor))
    w = floor + (1.0 - floor) * w_rw
    return dict(zip(hyps, w.tolist(), strict=True))


def prior_directions(prior: Prior, thr: float = 3.0, hyps=HYPOTHESES) -> dict:
    return {h: (int(np.sign(prior.z.get(h, 0.0))) if abs(prior.z.get(h, 0.0)) >= thr else 0) for h in hyps}


# ------------------------------------------------------------------------------------------------ procedures
def weighted_step_up(p, w=None, q: float = 0.05, reshape: bool = True) -> np.ndarray:
    """Blanchard-Roquain weighted step-up: R = max{k: #{j: p_j <= q w_j beta(k) / m} >= k}; reject p_j <= q w_j
    beta(R) / m. beta(k) = k / c_m (BY; arbitrary dependence) or k (BH). w: nonnegative, sum = m (checked).
    NaN p -> never rejected."""
    p = np.asarray(p, float)
    m = len(p)
    w = np.ones(m) if w is None else np.asarray(w, float)
    if np.any(w < 0) or abs(w.sum() - m) > 1e-6 * m:
        raise ValueError("weights must be >= 0 with sum m")
    pp = np.where(np.isfinite(p), p, np.inf)
    cm = c_harm(m) if reshape else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(w > 0, pp / w, np.inf)
    order = np.argsort(r, kind="stable")
    ks = np.arange(1, m + 1)
    ok = r[order] <= q * ks / (m * cm)
    out = np.zeros(m, bool)
    if ok.any():
        k = int(np.nonzero(ok)[0].max()) + 1
        out[order[:k]] = True
    return out


def dag_depths(parents) -> np.ndarray:
    n = len(parents)
    depth = np.zeros(n, int)

    def d(i, seen=()):
        if depth[i]:
            return depth[i]
        if i in seen:
            raise ValueError("graph has a cycle")
        depth[i] = 1 + max((d(j, seen + (i,)) for j in parents[i]), default=0)
        return depth[i]
    for i in range(n):
        d(i)
    return depth


def dag_effective(parents) -> tuple[np.ndarray, np.ndarray]:
    """(l_a, m_a): effective numbers of leaves / nodes (DAGGER eq. 2: a child's counts split evenly over its parents)."""
    n = len(parents)
    children = [[] for _ in range(n)]
    for i, ps in enumerate(parents):
        for j in ps:
            children[j].append(i)
    depth = dag_depths(parents)
    ell, mm = np.zeros(n), np.zeros(n)
    for i in sorted(range(n), key=lambda i: -depth[i]):
        if not children[i]:
            ell[i] = mm[i] = 1.0
        else:
            ell[i] = sum(ell[b] / len(parents[b]) for b in children[i])
            mm[i] = 1.0 + sum(mm[b] / len(parents[b]) for b in children[i])
    return ell, mm


def dagger(p, parents, q: float = 0.05, reshape: bool = True) -> np.ndarray:
    """(Reshaped) DAGGER (module docstring). parents[i] = list of parent indices. Reshaped: beta(x) = min(x, N) / c_N
    (nu ~ 1/k on {1..N}), valid under arbitrary dependence (Thm 1(b))."""
    p = np.where(np.isfinite(np.asarray(p, float)), np.asarray(p, float), np.inf)
    N = len(p)
    depth = dag_depths(parents)
    ell, mm = dag_effective(parents)
    is_leaf = np.ones(N, bool)
    for ps in parents:
        for j in ps:
            is_leaf[j] = False
    L = float(is_leaf.sum())
    cN = c_harm(N)

    def beta(x):
        return min(x, N) / cN if reshape else x
    rej = np.zeros(N, bool)
    R_prev = 0
    for d in range(1, int(depth.max()) + 1):
        Hd = np.nonzero(depth == d)[0]
        cand = np.array([all(rej[j] for j in parents[i]) for i in Hd], bool)

        def thr(r, Hd=Hd, cand=cand, R_prev=R_prev):
            return np.array([q * ell[i] / L * beta(mm[i] + r + R_prev - 1) / mm[i] if c else -1.0
                             for i, c in zip(Hd, cand, strict=True)])
        Rd = 0
        for r in range(len(Hd), 0, -1):              # generalized step-up (eq. 6a)
            if np.sum(p[Hd] <= thr(r)) >= r:
                Rd = r
                break
        if Rd:
            rej[Hd[p[Hd] <= thr(Rd)]] = True
        R_prev += int(rej[Hd].sum())
    return rej


def graphical_holm(p, w0, G, alpha: float = 0.05) -> np.ndarray:
    """Bretz et al. (2009) Algorithm 1 (Bonferroni-based graph; FWER <= alpha under arbitrary dependence).
    w0: initial weights (sum <= 1); G: transition matrix (rows sum <= 1, zero diagonal)."""
    p = np.where(np.isfinite(np.asarray(p, float)), np.asarray(p, float), np.inf)
    w = np.asarray(w0, float).copy()
    G = np.asarray(G, float).copy()
    n = len(p)
    act = np.ones(n, bool)
    rej = np.zeros(n, bool)
    while True:
        cand = np.nonzero(act & (p <= w * alpha))[0]
        if not len(cand):
            break
        j = int(cand[np.argmin(p[cand] / np.maximum(w[cand], 1e-300))])
        rej[j], act[j] = True, False
        wn = np.where(act, w + w[j] * G[j], 0.0)
        den = 1.0 - G[:, j] * G[j, :]                  # G_lk <- (G_lk + G_lj G_jk) / (1 - G_lj G_jl)
        Gn = np.where(den[:, None] > 1e-12, (G + np.outer(G[:, j], G[j])) / np.maximum(den, 1e-12)[:, None], 0.0)
        Gn[~act, :] = 0.0
        Gn[:, ~act] = 0.0
        np.fill_diagonal(Gn, 0.0)
        w, G = wn, Gn
    return rej


def p_filter(p, groups, w=None, q: float = 0.05, reshape: bool = True, max_iter: int = 1000) -> np.ndarray:
    """Two-layer reshaped p-filter: layer 1 = the leaves (weights w, sum m), layer 2 = ``groups`` (a group label per
    leaf; unit group weights) with weighted-Bonferroni group p-values P_g = min_{i in g} p_i W_g / w_i (W_g = sum of w
    in g; valid under arbitrary dependence). Thresholds t_m <= q beta_m(|S_m(t)|) / G_m, beta_m(k) = k / c_{G_m}; the
    maximal fixed point is reached by monotone iteration from t = q."""
    p = np.where(np.isfinite(np.asarray(p, float)), np.asarray(p, float), np.inf)
    m = len(p)
    w = np.ones(m) if w is None else np.asarray(w, float)
    labels, gi = np.unique(np.asarray(groups), return_inverse=True)
    G = len(labels)
    Pg = np.full(G, np.inf)
    for g in range(G):
        sel = gi == g
        Wg = w[sel].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(w[sel] > 0, p[sel] * Wg / w[sel], np.inf)
        Pg[g] = min(1.0, float(v.min())) if len(v) else np.inf
    c1 = c_harm(m) if reshape else 1.0
    c2 = c_harm(G) if reshape else 1.0
    t1, t2 = q / c1, q / c2           # q beta(m) / m and q beta(G) / G: the largest grid points

    def S(t1_, t2_):
        return (p <= w * t1_) & (Pg[gi] <= t2_)
    for _ in range(max_iter):
        s = S(t1, t2)
        k1 = int(s.sum())
        n1 = q * k1 / (m * c1)
        s = S(n1, t2)
        k2 = len(np.unique(gi[s]))
        n2 = q * k2 / (G * c2)
        if n1 >= t1 - 1e-15 and n2 >= t2 - 1e-15:
            break
        t1, t2 = min(t1, n1), min(t2, n2)
    return S(t1, t2)


def e_bh(e, q: float = 0.05, w=None) -> np.ndarray:
    """(Weighted) e-BH: reject the k* largest w_j e_j, k* = max{k: (w e)_(k) >= m / (q k)}; w >= 0 with sum m.
    NaN e -> 0."""
    e = np.nan_to_num(np.asarray(e, float), nan=0.0)
    m = len(e)
    if w is not None:
        w = np.asarray(w, float)
        if np.any(w < 0) or abs(w.sum() - m) > 1e-6 * m:
            raise ValueError("weights must be >= 0 with sum m")
        e = e * w
    out = np.zeros(m, bool)
    order = np.argsort(-e, kind="stable")
    ok = e[order] >= m / (q * np.arange(1, m + 1))
    if ok.any():
        out[order[: int(np.nonzero(ok)[0].max()) + 1]] = True
    return out


# ------------------------------------------------------------------------------------------------ structure
def gate_parents(hyps=HYPOTHESES) -> list:
    """Mediator-first DAG: (f, rel, load) is the parent of (f, rel, k) for k != load; loads are roots."""
    idx = {h: i for i, h in enumerate(hyps)}
    return [[] if h[2] == KPI_GATE else [idx[(h[0], h[1], KPI_GATE)]] for h in hyps]


def gate_graph(hyps=HYPOTHESES, delta: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """(w0, G) of the graphical Holm design (module docstring)."""
    n = len(hyps)
    loads = [i for i, h in enumerate(hyps) if h[2] == KPI_GATE]
    w0 = np.zeros(n)
    w0[loads] = 1.0 / len(loads)
    G = np.zeros((n, n))
    for i, h in enumerate(hyps):
        grp = [j for j, g in enumerate(hyps) if g[:2] == h[:2] and j != i]
        if h[2] == KPI_GATE:
            G[i, grp] = 1.0 / len(grp)
        else:
            sib = [j for j in grp if hyps[j][2] != KPI_GATE]
            other = [j for j in loads if hyps[j][:2] != h[:2]]
            G[i, sib] = (1.0 - delta) / len(sib)
            G[i, other] = delta / len(other)
    return w0, G


# ------------------------------------------------------------------------------------------------ declare
def _vectors(stats: dict, dirs: dict, one_sided: bool, kind: str, hyps):
    vals, signs, used_1s = [], [], []
    for h in hyps:
        s = stats.get(h, HypStats())
        d = dirs.get(h, 0) if one_sided else 0
        if kind == "p":
            v = {1: s.p_plus, -1: s.p_minus}.get(d, s.p2)
            if d and not np.isfinite(v):          # one-sided p not available -> two-sided (still valid)
                v, d = s.p2, 0
            sg = d if d else s.sign
        else:
            v = {1: s.e_plus, -1: s.e_minus}.get(d, s.e2)
            if d and not np.isfinite(v):
                v, d = s.e2, 0
            sg = d if d else s.e_sign
        vals.append(v)
        signs.append(int(sg))
        used_1s.append(bool(d))
    return np.array(vals, float), signs, used_1s


def declare(stats: dict, procedure: str, prior: Prior | None = None, n_target: int | None = None, q: float = 0.05,
            floor: float = 0.2, thr: float = 3.0, hyps=HYPOTHESES, delta: float = 0.2) -> dict:
    """Apply ``procedure`` (``PROCEDURES``) to {hyp: HypStats}. Returns {hyp: {"declared", "sign", "p" | "e",
    "w", "one_sided", "procedure"}} over ``hyps`` (edge_score format; untested hypotheses are never declared)."""
    kind, one_sided, weighted, structure = _SPEC[procedure]
    if (one_sided or weighted) and prior is None:
        raise ValueError(f"{procedure} needs a Prior (independent data)")
    dirs = prior_directions(prior, thr, hyps) if one_sided else {}
    wd = prior_weights(prior, n_target, q, floor, hyps) if weighted else {h: 1.0 for h in hyps}
    w = np.array([wd[h] for h in hyps])
    x, signs, used = _vectors(stats, dirs, one_sided, kind, hyps)
    if kind == "e":
        dec = e_bh(x, q, w if weighted else None)
    elif structure is None:
        dec = weighted_step_up(x, w, q, reshape=True)
    elif structure == "bh":
        dec = weighted_step_up(x, None, q, reshape=False)
    elif structure == "dagger":
        dec = dagger(x, gate_parents(hyps), q, reshape=True)
    elif structure == "gholm":
        w0, G = gate_graph(hyps, delta)
        dec = graphical_holm(x, w0, G, q)
    elif structure == "pfilter":
        dec = p_filter(x, ["|".join(h[:2]) for h in hyps], w, q, reshape=True)
    else:
        raise ValueError(structure)
    out = {}
    for i, h in enumerate(hyps):
        ok = np.isfinite(x[i])
        out[h] = {"declared": bool(dec[i] and ok), "sign": signs[i], kind: float(x[i]), "w": float(w[i]),
                  "one_sided": used[i], "procedure": procedure,
                  "status": ("declared" if dec[i] and ok else "not_detected") if ok else "undetermined"}
        if kind == "p":
            out[h]["p"] = float(x[i])
    return out


__all__ = ["KPI_GATE", "MULTI_VERSION", "PROCEDURES", "W_CAP", "HypStats", "Prior", "c_harm", "cap_weights",
           "dag_depths", "dag_effective", "dagger", "declare", "e_bh", "expected_rejections", "gate_graph",
           "gate_parents", "graphical_holm", "p_filter", "prior_directions", "prior_weights", "projected_means",
           "rw_weights", "stats_from_crt_v2", "stats_from_json", "weighted_step_up"]

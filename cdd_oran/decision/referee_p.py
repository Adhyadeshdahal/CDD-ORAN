"""E6-P regime referee, K-L0 pieces (scratchpad/e6_dev/decision/STEP2_REFEREE_PLAN.md sections 1, 2 and 5).

Analysis-only building blocks of the step-2 referee: which energy-saving requests to DEFER (= reject for the unit)
and in which obs-only regime. No policy object yet (K-L1 builds ``RefereePolicy`` on top of these).

Energy-saving direction (``SAVE_SIGN``): a request is gated only when it moves its knob in the energy-SAVING
direction: macro carrier-off (carrier step < 0), pico sleep (sleep step > 0), PowerES ptx-down (ptx step < 0).
Wake, carrier-on, ptx-up and SliceGuarantee (prot_min) are always accepted. Actions: accept | defer (defer = the
unit's mode is reject; the knockout label's reject arm). PowerES "soften" has no paired label (the GT labels carry
accept / reject only), so it is not offered here.

Labels (``label_rows``): one row per GT knockout label of a gated request. y[kpi] = NETWORK-wide paired effect of
ACCEPTING vs rejecting the unit: mean over the 3 reseeds of the per-cell accept - reject deltas summed over all cells
(``gt_p`` label format; H = 90 s, AA continuation). The change caused by DEFERRING is -y[k]; for pv the deferral
benefit (reduction) is +y["pv"].

Features (``FEATURES``): obs-only keys of ``unit["ctx"]`` (``units_p.UnitArbiter.context``) restricted to the
mediators tied to the targeted KPIs (plan sec. 1): pv -> prot_act_ue, prot_below_frac, prot_dem_share; v / load ->
prb_util, act_ue; e -> carriers, asleep, ptx_db. A feature is read from ``ctx`` only; the privileged record keys
(``PRIVILEGED_KEYS``) are never read for features.

Honest regime leaves (``HonestTree``): depth <= 2 regression tree per family predicting y["pv"]; splits chosen on
the SPLIT half of the label episodes only (SSE reduction, thresholds = split-half quantiles), leaf values estimated
on the ESTIMATION half only. Support: every leaf needs >= ``MIN_LEAF_N`` labels and >= ``MIN_LEAF_EPS`` episodes in
BOTH halves; <= ``MAX_LEAVES`` leaves. If the estimation half does not support the fitted leaves, the tree is cut to
its root split, then to a single leaf. Half assignment: ``default_rng([REFEREE_TAG, swap])`` permutation of the
sorted label episodes (REFEREE_TAG = 6619, registered for step 2 in the plan).

Effect table (``effect_table``): per (family, leaf) the DEFER effect on pv, e, v, rlf (= -mean accept-vs-reject
label on the estimation half), episode-cluster percentile bootstrap 90 % CI, and empirical-Bayes shrinkage of the
leaf means toward the family mean (method-of-moments tau^2 across the family's leaves; one leaf -> unshrunk).

Knapsack (``knapsack``): per leaf accept / defer by enumeration (<= 12 leaves), maximising the expected pv reduction
per episode sum_l nbar_l * b_l subject to energy cost <= budget_e and v, rlf increases <= caps; a leaf may defer
only if its leaf is supported and its benefit's 90 % lower bound (5th percentile of the cluster bootstrap) is > 0.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np

REFEREE_TAG = 6619
SAVE_SIGN = {"carrier": -1, "sleep": +1, "ptx": -1}         # step sign of the energy-saving direction
GATED_FAMILIES = tuple(SAVE_SIGN)
ACTIONS = ("accept", "defer")
EFFECT_KPIS = ("pv", "e", "v", "rlf")
PRIVILEGED_KEYS = ("gt_static", "gt_labels", "lab_outcome", "lab_series", "lab_kpi", "delta", "d_net_reject")
MEDIATORS_OF = {"pv": ("prot_act_ue", "prot_below_frac", "prot_dem_share"), "v": ("prb_util", "act_ue"),
                "e": ("carriers", "asleep", "ptx_db")}
FEATURES = ("own_prot_act_ue", "nbr_sum_prot_act_ue", "own_prot_below_frac", "nbr_mean_prot_below_frac",
            "nbr_max_prot_below_frac", "own_prot_dem_share", "nbr_mean_prot_dem_share", "nbr_max_prot_dem_share",
            "own_prb_util", "nbr_mean_prb_util", "nbr_max_prb_util", "own_act_ue", "nbr_sum_act_ue",
            "own_carriers", "nbr_mean_carriers", "nbr_sum_asleep", "own_ptx_db", "nbr_mean_ptx_db")
MIN_LEAF_N, MIN_LEAF_EPS, MAX_LEAVES, MAX_DEPTH = 25, 6, 4, 2
N_QUANT = 9                                                  # candidate thresholds = deciles of the split half
N_BOOT = 4000
CI90 = (0.05, 0.95)


# ---------------------------------------------------------------------------------------------- units / features
def is_saving(knob: str, step: float) -> bool:
    """True iff the request moves ``knob`` in the energy-saving direction (gated families only)."""
    s = SAVE_SIGN.get(knob)
    return s is not None and float(step) * s > 0


def ctx_features(ctx: dict, features=FEATURES) -> np.ndarray:
    """Feature vector from an obs-only unit context. Refuses privileged key names; missing -> NaN."""
    bad = [f for f in features if any(p in f for p in PRIVILEGED_KEYS)]
    if bad:
        raise ValueError(f"privileged keys requested as features: {bad}")
    out = np.full(len(features), np.nan)
    for j, f in enumerate(features):
        v = ctx.get(f)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[j] = float(v)
    return out


def label_rows(records, dec, features=FEATURES) -> list:
    """GT knockout labels of gated (energy-saving-direction) requests -> rows
    {"ep", "family", "c", "t0", "x" (features, from the logged unit's ctx), "y": {kpi: network accept - reject}}.
    ``dec`` decodes the label's delta array (collect_p.dec)."""
    rows = []
    for rec in records:
        units = rec["units"]
        for L in rec["gt_labels"]:
            if not is_saving(L["knob"], L["step"]):
                continue
            u = units[int(L["i"])]
            if u["knob"] != L["knob"] or int(u["c"]) != int(L["c"]) or abs(float(u["t0"]) - float(L["t0"])) > 1e-6:
                raise ValueError(f"label {L['i']} does not match its logged unit")
            D = np.asarray(dec(L["delta"]), float)                   # (ks, kpis, cells) accept - reject
            net = D.sum(2).mean(0)
            kp = list(L["kpis"])
            rows.append({"ep": int(rec["seed"]), "family": L["knob"], "c": int(L["c"]), "t0": float(L["t0"]),
                         "x": ctx_features(u["ctx"], features), "y": {k: float(net[kp.index(k)]) for k in EFFECT_KPIS}})
    return rows


def unit_table(records, features=FEATURES) -> list:
    """Every logged gated unit of pi0 episode records -> {"ep", "family", "t0", "x"} (reads ``units`` only)."""
    out = []
    for rec in records:
        for u in rec["units"]:
            if is_saving(u["knob"], u["step"]):
                out.append({"ep": int(rec["seed"]), "family": u["knob"], "t0": float(u["t0"]),
                            "x": ctx_features(u["ctx"], features)})
    return out


def split_halves(episodes, swap: int = 0, tag: int = REFEREE_TAG) -> tuple[set, set]:
    """(split episodes, estimation episodes): a seeded permutation of the sorted episode ids cut in two; swap = 1
    exchanges the roles."""
    eps = sorted({int(e) for e in episodes})
    perm = np.random.default_rng([tag, 0]).permutation(len(eps))
    a = {eps[i] for i in perm[: len(eps) // 2]}
    b = set(eps) - a
    return (b, a) if swap else (a, b)


# ---------------------------------------------------------------------------------------------- honest tree
def _support(ep, idx, min_n, min_eps) -> bool:
    return len(idx) >= min_n and len(np.unique(ep[idx])) >= min_eps


def _best_split(X, y, ep, idx, min_n, min_eps, n_quant=N_QUANT):
    """(gain, feature, threshold) of the best SSE split of rows ``idx`` with support in both children, or None."""
    best = None
    yi = y[idx]
    sse0 = float(((yi - yi.mean()) ** 2).sum())
    for j in range(X.shape[1]):
        xj = X[idx, j]
        fin = np.isfinite(xj)
        if fin.sum() < 2 * min_n:
            continue
        for thr in np.unique(np.quantile(xj[fin], np.linspace(0, 1, n_quant + 2)[1:-1])):
            left = xj <= thr                                          # NaN -> right (both at fit and at predict)
            li, ri = idx[left], idx[~left]
            if not (_support(ep, li, min_n, min_eps) and _support(ep, ri, min_n, min_eps)):
                continue
            gain = sse0 - float(((y[li] - y[li].mean()) ** 2).sum()) - float(((y[ri] - y[ri].mean()) ** 2).sum())
            if best is None or gain > best[0] + 1e-12:
                best = (gain, j, float(thr))
    return best


@dataclasses.dataclass
class HonestTree:
    """Depth <= 2 tree; ``nodes``: {path: (feature index, threshold)}, path = "" (root), "L", "R"."""
    nodes: dict
    features: tuple
    split_eps: tuple = ()
    est_eps: tuple = ()
    note: str = ""

    def leaf_of(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        out = np.empty(len(X), dtype=object)
        for i, x in enumerate(X):
            p = ""
            while p in self.nodes and len(p) < MAX_DEPTH:
                j, thr = self.nodes[p]
                p += "L" if x[j] <= thr else "R"                      # NaN compares False -> "R"
            out[i] = p or "all"
        return out

    @property
    def leaves(self) -> list:
        if "" not in self.nodes:
            return ["all"]
        out = []
        for a in "LR":
            out += [a + b for b in "LR"] if a in self.nodes else [a]
        return out

    def describe(self) -> dict:
        return {p or "root": f"{self.features[j]} <= {t:.4g}" for p, (j, t) in self.nodes.items()}


def fit_honest_tree(X, y, ep, split_eps, est_eps, features=FEATURES, min_n=MIN_LEAF_N, min_eps=MIN_LEAF_EPS,
                    max_leaves=MAX_LEAVES) -> HonestTree:
    """Grow on the rows of ``split_eps`` ONLY; prune until every leaf is supported on the ``est_eps`` rows too."""
    X, y, ep = np.asarray(X, float), np.asarray(y, float), np.asarray(ep)
    s_idx = np.nonzero(np.isin(ep, list(split_eps)))[0]
    e_idx = np.nonzero(np.isin(ep, list(est_eps)))[0]
    nodes = {}
    root = _best_split(X, y, ep, s_idx, min_n, min_eps) if _support(ep, s_idx, 2 * min_n, min_eps) else None
    if root is not None:
        nodes[""] = root[1:]
        left = X[s_idx, root[1]] <= root[2]
        kids = []
        for side, idx in (("L", s_idx[left]), ("R", s_idx[~left])):
            b = _best_split(X, y, ep, idx, min_n, min_eps)
            if b is not None:
                kids.append((b[0], side, b[1:]))
        for _, side, node in sorted(kids, key=lambda k: -k[0]):
            if len(HonestTree(dict(nodes, **{side: node}), tuple(features)).leaves) <= max_leaves:
                nodes[side] = node
    tree = HonestTree(nodes, tuple(features), tuple(sorted(split_eps)), tuple(sorted(est_eps)))
    for cut in ("children", "root"):
        lf = tree.leaf_of(X[e_idx]) if len(e_idx) else np.array([], dtype=object)
        if all(_support(ep, e_idx[lf == leaf], min_n, min_eps) for leaf in tree.leaves) or tree.leaves == ["all"]:
            break
        tree.nodes = {"": tree.nodes[""]} if cut == "children" else {}
        tree.note = f"pruned ({cut}) for estimation-half support"
    if not tree.nodes and not tree.note:
        tree.note = "single leaf (split half cannot support a split)"
    return tree


# ---------------------------------------------------------------------------------------------- effect table
def cluster_boot(y, ep, rng, n_boot=N_BOOT, q=CI90):
    """(mean, (lo, hi)) with an episode-cluster percentile bootstrap of the pooled unit mean."""
    y, ep = np.asarray(y, float), np.asarray(ep)
    if len(y) == 0:
        return float("nan"), (float("nan"), float("nan")), np.full(n_boot, np.nan)
    eps, inv = np.unique(ep, return_inverse=True)
    S = np.bincount(inv, y, len(eps))
    N = np.bincount(inv, minlength=len(eps)).astype(float)
    idx = rng.integers(len(eps), size=(n_boot, len(eps)))
    bm = S[idx].sum(1) / np.maximum(N[idx].sum(1), 1e-12)
    lo, hi = np.quantile(bm, q)
    return float(y.mean()), (float(lo), float(hi)), bm


@dataclasses.dataclass
class Effect:
    family: str
    leaf: str
    n: int
    n_eps: int
    defer: dict           # kpi -> mean change caused by DEFER (= -mean accept-reject), raw
    ci90: dict            # kpi -> (lo, hi) of the raw DEFER change
    shrunk: dict          # kpi -> EB-shrunk DEFER change
    benefit_lb90: float   # 5th percentile of the pv reduction from deferring (= -defer["pv"])
    supported: bool = True  # >= MIN_LEAF_N labels and >= MIN_LEAF_EPS episodes

    @property
    def benefit(self) -> float:
        return -self.shrunk["pv"]


def effect_table(rows, trees: dict, use: str = "est", n_boot=N_BOOT, tag=REFEREE_TAG) -> list:
    """Effects per (family, leaf) on the estimation-half rows of each family's tree (use="est"), or on all rows."""
    out = []
    for fi, fam in enumerate(sorted(trees)):
        tree = trees[fam]
        fr = [r for r in rows if r["family"] == fam and (use == "all" or r["ep"] in set(tree.est_eps))]
        if not fr:
            continue
        X = np.array([r["x"] for r in fr])
        ep = np.array([r["ep"] for r in fr])
        lf = tree.leaf_of(X)
        leaf_stats = {}
        for li, leaf in enumerate(tree.leaves):
            m = lf == leaf
            st = {}
            for ki, k in enumerate(EFFECT_KPIS):
                y = -np.array([r["y"][k] for r in fr])[m]                  # change caused by DEFER
                rng = np.random.default_rng([tag, 1, fi, li, ki])
                mean, ci, bm = cluster_boot(y, ep[m], rng, n_boot)
                se = float(np.nanstd(bm)) if len(y) else float("nan")
                st[k] = (mean, ci, se, bm)
            leaf_stats[leaf] = (int(m.sum()), int(len(np.unique(ep[m]))), st)
        fam_mean = {k: float(np.mean([-r["y"][k] for r in fr])) for k in EFFECT_KPIS}
        for leaf, (n, ne, st) in leaf_stats.items():
            shr = {}
            for k in EFFECT_KPIS:
                means = np.array([s[2][k][0] for s in leaf_stats.values() if s[0] > 0])
                ses = np.array([s[2][k][2] for s in leaf_stats.values() if s[0] > 0])
                if len(means) < 2 or not np.isfinite(st[k][0]):
                    shr[k] = st[k][0]
                    continue
                tau2 = max(0.0, float(np.var(means, ddof=1) - np.mean(ses ** 2)))
                se2 = st[k][2] ** 2
                w = tau2 / (tau2 + se2) if tau2 + se2 > 0 else 1.0
                shr[k] = float(w * st[k][0] + (1 - w) * fam_mean[k])
            bm_pv = st["pv"][3]
            lb = float(np.nanquantile(-bm_pv, CI90[0])) if n else float("nan")
            out.append(Effect(fam, leaf, n, ne, {k: st[k][0] for k in EFFECT_KPIS},
                              {k: st[k][1] for k in EFFECT_KPIS}, shr, lb,
                              supported=n >= MIN_LEAF_N and ne >= MIN_LEAF_EPS))
    return out


# ---------------------------------------------------------------------------------------------- knapsack
def knapsack(effects, nbar: dict, budget_e: float, cap_v: float, cap_rlf: float, lb_rule: bool = True) -> dict:
    """Enumerate accept / defer per (family, leaf). ``nbar[(family, leaf)]`` = gated units per episode.
    Objective: max sum nbar * benefit (pv reduction / episode); constraints on the DEFER changes (shrunk means):
    sum nbar * e <= budget_e (energy cost), sum nbar * v <= cap_v, sum nbar * rlf <= cap_rlf."""
    items = [ef for ef in effects if nbar.get((ef.family, ef.leaf), 0.0) > 0]
    if len(items) > 16:
        raise ValueError("too many leaves for enumeration")
    allowed = [ef for ef in items if (not lb_rule) or (ef.supported and np.isfinite(ef.benefit_lb90)
                                                       and ef.benefit_lb90 > 0)]
    best = {"obj": 0.0, "defer": [], "e": 0.0, "v": 0.0, "rlf": 0.0}
    for r in range(1, len(allowed) + 1):
        for combo in itertools.combinations(allowed, r):
            tot = {k: sum(nbar[(ef.family, ef.leaf)] * ef.shrunk[k] for ef in combo) for k in EFFECT_KPIS}
            if tot["e"] > budget_e + 1e-9 or tot["v"] > cap_v + 1e-9 or tot["rlf"] > cap_rlf + 1e-9:
                continue
            obj = -tot["pv"]
            if obj > best["obj"] + 1e-12:
                best = {"obj": float(obj), "defer": [(ef.family, ef.leaf) for ef in combo], "e": float(tot["e"]),
                        "v": float(tot["v"]), "rlf": float(tot["rlf"])}
    best["policy"] = {(ef.family, ef.leaf): ("defer" if (ef.family, ef.leaf) in best["defer"] else "accept")
                      for ef in items}
    best["n_allowed"] = len(allowed)
    return best


def sign_agreement(rows, trees_by_half: dict, kpis=("pv",), supported_only: bool = False) -> dict:
    """Split-half sign agreement: for each half's tree, the sign of every (family, leaf) DEFER mean on its split half
    vs its estimation half (same tree). Returns {"agree", "n", "pairs"}; a zero mean counts as disagreement.
    ``supported_only``: keep only leaves with >= MIN_LEAF_N labels and >= MIN_LEAF_EPS episodes in BOTH halves."""
    pairs = []
    for h, trees in trees_by_half.items():
        for fam, tree in trees.items():
            fr = [r for r in rows if r["family"] == fam]
            if not fr:
                continue
            X = np.array([r["x"] for r in fr])
            lf = tree.leaf_of(X)
            ep = np.array([r["ep"] for r in fr])
            ha, hb = np.isin(ep, list(tree.split_eps)), np.isin(ep, list(tree.est_eps))
            for leaf in tree.leaves:
                for k in kpis:
                    y = -np.array([r["y"][k] for r in fr])
                    a, b = y[(lf == leaf) & ha], y[(lf == leaf) & hb]
                    if len(a) == 0 or len(b) == 0:
                        continue
                    if supported_only and not all(
                            _support(ep, np.nonzero((lf == leaf) & h_)[0], MIN_LEAF_N, MIN_LEAF_EPS) for h_ in (ha, hb)):
                        continue
                    pairs.append({"half": h, "family": fam, "leaf": leaf, "kpi": k, "mean_split": float(a.mean()),
                                  "mean_est": float(b.mean()), "n": (len(a), len(b)),
                                  "agree": bool(np.sign(a.mean()) == np.sign(b.mean()) != 0)})
    n = len(pairs)
    return {"agree": (sum(p["agree"] for p in pairs) / n) if n else float("nan"), "n": n, "pairs": pairs}


__all__ = ["ACTIONS", "CI90", "EFFECT_KPIS", "FEATURES", "GATED_FAMILIES", "MAX_LEAVES", "MEDIATORS_OF",
           "MIN_LEAF_EPS", "MIN_LEAF_N", "PRIVILEGED_KEYS", "REFEREE_TAG", "SAVE_SIGN", "Effect", "HonestTree",
           "cluster_boot", "ctx_features", "effect_table", "fit_honest_tree", "is_saving", "knapsack", "label_rows",
           "sign_agreement", "split_halves", "unit_table"]

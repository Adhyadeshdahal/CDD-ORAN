"""Selection-aware confidence gate for a learned WG3 arbiter (DESIGN.md REVISION v2, item 5).

The arbiter deviates from accept-all only when a CALIBRATED lower bound of the predicted improvement is > 0.
Calibration target = the whole argmax pipeline, not single predictions: at held-out decision epochs (held-out
EPISODES) the learned arbiter picks its best plan; we record its predicted improvement over accept-all and the
realized (oracle, same state, same future tape) improvement of that same plan. Nonconformity score
    s = predicted_improvement - realized_improvement            (the optimism of the selected candidate)
and the split-conformal quantile q gives LB = predicted_improvement - q with
    P(realized >= LB) >= 1 - alpha                                ("pooled": new decision exchangeable with calib.)
    P(all decisions of a new episode covered) >= 1 - alpha        ("episode_max": score = per-episode max; rigorous
                                                                   under episode exchangeability, conservative)
Default method = "episode_max" (SOL_BUILD_REVIEW.md item 4): epochs of one episode are dependent, so the
calibration unit is the EPISODE; calibration and final-evaluation episodes come from disjoint registered seed ranges
(docs/benchmark/E6_COLLECTION_CONTRACT.md). "pooled" (decisions exchangeable) stays selectable for diagnostics only.
Per-stratum quantiles (e.g. scenario x load x support bin) fall back to the pooled one below ``min_units`` units
(episodes under episode_max): with 30 calibration episodes pooled over the six strata the claim is POOLED.
Outside the training support (``SupportDetector``) or with a stale context the score carries ``ood=True`` and the
gate refuses (the arbiter then keeps accept-all). Guardrails are NOT covered by this bound: bound per-component
effects separately (``PolicyEffectModel`` exposes them).

Hook: ``ConformalGate(...).calibrate(...)`` is a ``confidence(score_candidate, score_accept_all) -> bool`` callable
for ``WG3Arbiter(confidence=...)``. Scores may carry ``members`` (per-member values, paired with the reference's),
``ood`` and ``stratum`` attributes (``effect_model.EffectScore``); plain ``Score`` works too.
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

import numpy as np


def conformal_quantile(scores, alpha: float) -> float:
    """Split-conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest score; +inf if n is too small."""
    v = np.sort(np.asarray(scores, float).ravel())
    n = len(v)
    k = math.ceil((n + 1) * (1.0 - alpha))
    return float("inf") if n == 0 or k > n else float(v[k - 1])


class SupportDetector:
    """Box support of the training contexts: per-column [q, 1 - q] quantiles widened by ``margin`` x range.
    A row is outside if any column is outside its box or non-finite. Constant columns get a zero-width box."""

    def __init__(self, q: float = 0.005, margin: float = 0.1):
        self.q, self.margin = q, margin
        self.lo = self.hi = None

    def fit(self, x) -> SupportDetector:
        x = np.asarray(x, float)
        lo, hi = np.nanquantile(x, self.q, axis=0), np.nanquantile(x, 1.0 - self.q, axis=0)
        span = self.margin * (hi - lo)
        self.lo, self.hi = lo - span - 1e-9, hi + span + 1e-9
        return self

    def outside(self, x) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        return ~np.isfinite(x).all(1) | (x < self.lo).any(1) | (x > self.hi).any(1)


class ConformalGate:
    """Calibrated paired-difference gate. ``member_agree`` (optional) additionally requires that at least that
    fraction of ensemble members predicts an improvement (paired member-wise vs the reference score)."""

    def __init__(self, alpha: float = 0.1, method: str = "episode_max", min_units: int = 30,
                 member_agree: float | None = None):
        if method not in ("pooled", "episode_max"):
            raise ValueError("method must be 'pooled' or 'episode_max'")
        self.alpha, self.method, self.min_units, self.member_agree = alpha, method, min_units, member_agree
        self.q_pooled, self.q_stratum, self.n_units = float("inf"), {}, {}

    def _units(self, s, episode):
        if self.method == "pooled":
            return s
        ep = np.asarray(episode)
        return np.array([s[ep == e].max() for e in np.unique(ep)])

    def calibrate(self, pred, realized, episode, stratum: Sequence[Hashable] | None = None) -> ConformalGate:
        """``pred``/``realized``: improvement over accept-all (> 0 = better) of the SELECTED plan per decision;
        ``episode``: calibration episode id per decision (never a training episode)."""
        s = np.asarray(pred, float) - np.asarray(realized, float)
        ep = np.asarray(episode)
        u = self._units(s, ep)
        self.q_pooled, self.n_units = conformal_quantile(u, self.alpha), {None: len(u)}
        self.q_stratum = {}
        if stratum is not None:
            st = np.asarray(stratum, dtype=object)
            for g in {x for x in st}:
                m = st == g
                ug = self._units(s[m], ep[m])
                self.n_units[g] = len(ug)
                if len(ug) >= self.min_units:
                    self.q_stratum[g] = conformal_quantile(ug, self.alpha)
        return self

    def q(self, stratum=None) -> float:
        return self.q_stratum.get(stratum, self.q_pooled)

    def lower_bound(self, pred, stratum=None):
        if stratum is None or np.ndim(stratum) == 0:
            return np.asarray(pred, float) - self.q(stratum)
        return np.asarray(pred, float) - np.array([self.q(g) for g in stratum])

    def confidence(self, s_cand, s_ref) -> bool:
        if getattr(s_cand, "ood", False):
            return False
        imp = float(s_ref.mean) - float(s_cand.mean)
        if self.member_agree is not None:
            mc = np.asarray(getattr(s_cand, "members", ()), float)
            mr = np.asarray(getattr(s_ref, "members", ()), float)
            if mc.size:
                d = (mr if mr.size else np.zeros_like(mc)) - mc          # paired member-wise improvement
                if float(np.mean(d > 0)) < self.member_agree:
                    return False
        return bool(self.lower_bound(imp, getattr(s_cand, "stratum", None)) > 0)

    __call__ = confidence


# -------------------------------------------------------------------------------------------- evaluation helpers
def coverage(lb, realized) -> float:
    """Share of decisions whose realized improvement is >= the lower bound."""
    return float(np.mean(np.asarray(realized, float) >= np.asarray(lb, float)))


def false_improvement_rate(lb, realized) -> float:
    """Among decisions where the gate deviates (LB > 0): share whose realized improvement is <= 0 (NaN if none)."""
    lb, r = np.asarray(lb, float), np.asarray(realized, float)
    m = lb > 0
    return float(np.mean(r[m] <= 0)) if m.any() else float("nan")


def gate_report(gate: ConformalGate, pred, realized, episode=None, stratum=None) -> dict:
    """Coverage / deviation / false-improvement / realized gain on an evaluation set (held-out episodes)."""
    lb = gate.lower_bound(pred, stratum)
    r = np.asarray(realized, float)
    dev = lb > 0
    out = {"n": int(len(r)), "coverage": coverage(lb, r), "deviate_rate": float(np.mean(dev)),
           "false_improvement": false_improvement_rate(lb, r),
           "realized_gain_sum": float(r[dev].sum()), "oracle_gain_sum": float(np.clip(r, 0, None).sum())}
    if episode is not None:
        ep = np.asarray(episode)
        out["episode_all_covered"] = float(np.mean([np.all(r[ep == e] >= lb[ep == e]) for e in np.unique(ep)]))
    return out


def oracle_record(arbiter, ctx, true_wm=None) -> dict:
    """One selection-aware calibration record at a decision epoch (PRIVILEGED: ``ctx.env`` between the phases).
    Runs the arbiter's own search with its learned world model and NO gate, then scores the chosen plan and
    accept-all with the learned model (predicted) and with the true simulator (realized)."""
    from . import plans as P
    from .world_model import TrueSimWM
    if ctx.env is None:
        raise ValueError("oracle_record needs the privileged env handle in ctx.env")
    conf, arbiter.confidence = arbiter.confidence, None
    try:
        plan = arbiter.choose(ctx)
    finally:
        arbiter.confidence = conf
    acc = P.accept_all(ctx.regions)
    sp = arbiter.wm.score(ctx, [plan, acc])
    so = (true_wm or TrueSimWM()).score(ctx, [plan, acc])
    return {"t": float(ctx.obs["t"]), "plan": plan, "pred": float(sp[1].mean - sp[0].mean),
            "realized": float(so[1].mean - so[0].mean), "stratum": getattr(sp[0], "stratum", None),
            "ood": bool(getattr(sp[0], "ood", False))}

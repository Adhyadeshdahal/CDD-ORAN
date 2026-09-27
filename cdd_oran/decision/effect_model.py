"""Learned policy-effect world model for the WG3 arbiter (DESIGN.md REVISION v2, items 1, 3, 5, 6).

Estimand per region r at decision epoch e (observable context x_r, neighbour policies pi_N(r)):
    Delta_r(pi | x) = E[C_H(pi_r, pi_N) - C_H(accept-all) | x]
for each priced component in COMPONENTS (violated UE-s per slice, RLF, energy kWh, applied changes = churn), summed
over the H s after the epoch start on the UEs/cells of region r. It is estimated from randomized joint-policy
collector episodes (``collect.collect_episode``), one row per (epoch, region), as a CONTRAST: every ensemble member
fits an outcome model m_k(x, a, a_N) and predicts m_k(x, pi) - m_k(x, accept-all). Accept-all is therefore exactly
0 and member-wise differences are paired (same member on both sides).

Confounding control = IPW-weighted outcome regression (the weighted-least-squares doubly-robust estimator, Robins
et al. 2007; Kang & Schafer 2007): each member is fitted with stabilized weights s(class)/p(code | x) from the
LOGGED propensities (class = default code vs the rest), so in the weighted population the drawn policy is
independent of the context; with the ridge learner and effect modifiers in the span of the baseline basis the
policy coefficients are consistent if EITHER the outcome basis OR the propensities are right. With the current
collector (context-free eps mixture) the weights are all exactly 1. ``aipw`` gives cross-fitted AIPW pseudo-outcomes
for one candidate vs default (the classic DR estimator; needs many rows per candidate class, i.e. few active xApps).

Policy features (13): per xApp one-hot of reject / half / lock (accept = 0) + rollback flag; the learner supplies
interactions with context. Context features: RUNTIME-observable only (never lab_*): per region OWN, per epoch GLOB.
``ContextSelector`` = the graph-as-soft-prior hook: which neighbouring regions (all / none / weights: graph mask,
SHAP mask, topology) and which context columns enter. Neighbour context and policies enter as weighted means.

Future-assignment covariates (v2 collector data, ``collect.RandomizedStepPolicy``): with H > D the label window
(t_e, t_e + H] also covers the region's own later epochs e+1 .. e+F (F = ceil(H/D) - 1), whose policies were drawn by
the same outcome-independent randomization, so they are valid covariates. ``fut`` (E, R, 13) = sum_j w_j phi(code
of epoch e+j), w_j = (H - j D) / H = share of the window the epoch's policy acts on (exposure-weighted future policy).
They enter as main effects (``use_future``; "auto" = on iff every episode carries them). At serve time the
continuation is declared by ``EffectWM(continuation=...)``: "accept_all" (default) sets fut = 0 for the candidate
AND the accept-all reference, i.e. the scored estimand is "pi for one epoch (D s), then accept-all" (DESIGN.md
REVISION v2 item 1); "hold" sets fut = sum_j w_j phi(pi) for the candidate (pi held for H, as TrueSimWM's rollout
does). With the linear ridge learner and accept_all the fut block cancels in the contrast; it still removes the
future draws' variance/bias from the policy coefficients.
Weighting: ``weighting="auto"`` (default) = "none" when every episode comes from the v2 collector (context-free
propensities by design: weights are not needed for consistency and only cost efficiency), else "ipw".

Train/serve parity: the context is built by ONE function (``assemble_features``) from the collector's own
``RandomizedJointPolicy._context`` code path (reused via a shim at serve time) + per-xApp request counts. The
collector keeps the latest delivered fast KPM report across seconds, which is NOT in the epoch's obs: at serve time
``EffectWM.observe(obs)`` must see EVERY second's obs (``run_episode`` below does it; WG3Arbiter.act does not call
it yet). A gap in observed seconds marks the context stale (ood) so the gate falls back to accept-all.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import collect as CO
from . import plans as P
from .gate import SupportDetector
from .trace import OUT, RB_CODES, Trace, region_labels
from .world_model import DecisionContext, Score

COMPONENTS = ("viol_LL", "viol_eMBB", "viol_BE", "rlf", "energy_kwh", "churn")
OWN = ("prb_util", "act_ue_LL", "act_ue_eMBB", "act_ue_BE", "ll_p95", "ll_missing",
       "nreq_MRO", "nreq_TS", "nreq_ES", "nreq_SLICE", "locked")
GLOB = ("report_age", "has_cap", "churn_left")
POLICY_NAMES = tuple(f"{x}:{m}" for x in P.XAPPS for m in P.MODES[1:]) + ("rb",)


def _phi_table() -> np.ndarray:
    t = np.zeros((CO.N_CODES, len(POLICY_NAMES)))
    for c in range(CO.N_CODES):
        rp = CO.decode(c)
        for i, x in enumerate(P.XAPPS):
            m = P.MODES.index(rp["mode"][x])
            if m:
                t[c, i * (len(P.MODES) - 1) + m - 1] = 1.0
        t[c, -1] = rp["rb"]
    return t


PHI = _phi_table()


def policy_features(codes) -> np.ndarray:
    """codes (...) -> (..., 13) policy features."""
    return PHI[np.asarray(codes, int)]


def policy_class(code: int, active) -> tuple:
    """Equivalence class of a code when only ``active`` xApps act (inert xApps' modes are marginalised)."""
    rp = CO.decode(code)
    return tuple(rp["mode"][x] for x in P.XAPPS if x in active) + (rp["rb"],)


# ------------------------------------------------------------------------------------------------ context features
class ReportTracker:
    """Latest delivered KPM report per granularity, with the collector's update rule (newer or equal t1 wins).
    ``complete`` = every second since t = 1 has been observed (else the context may be stale)."""

    def __init__(self):
        self.latest, self.last_t, self.first_t, self.gaps = {}, None, None, 0

    def observe(self, obs: Mapping) -> None:
        t = float(obs["t"])
        if self.last_t is not None and t <= self.last_t:
            return                                   # idempotent within a second
        if self.last_t is None:
            self.first_t = t
        elif t != self.last_t + 1:
            self.gaps += 1
        for rep in obs["new_reports"]:
            cur = self.latest.get(rep["gran"])
            if cur is None or rep["t1"] >= cur["t1"]:
                self.latest[rep["gran"]] = rep
        self.last_t = t

    @property
    def complete(self) -> bool:
        return self.first_t == 1.0 and self.gaps == 0


def assemble_features(c: Mapping, nreq_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collector context dict (``RandomizedJointPolicy._context`` layout, one epoch) + per-xApp request counts
    (R, 4) -> (own (R, len(OWN)), glob (len(GLOB),)). NaN kept where no report exists (caller flags ood)."""
    ll = np.asarray(c["ll_p95"], float)
    own = np.column_stack([np.asarray(c["prb_util"], float), np.asarray(c["act_ue"], float),
                           np.nan_to_num(ll, nan=0.0), (~np.isfinite(ll)).astype(float),
                           np.asarray(nreq_x, float), np.asarray(c["locked"], float)])
    cl = float(c["churn_left"])
    glob = np.array([float(c["report_age"]), float(cl >= 0), max(cl, 0.0)])
    return own, glob


def nreq_by_xapp(requests, site, regions) -> np.ndarray:
    rix = {g: i for i, g in enumerate(regions)}
    n = np.zeros((len(regions), len(P.XAPPS)))
    for r in requests:
        n[rix[int(site[r["knob"][1]])], P.XAPPS.index(r["xapp"])] += 1
    return n


def serve_features(obs: Mapping, tracker: ReportTracker, site, regions) -> tuple[np.ndarray, np.ndarray]:
    """Decision-time features from obs + the tracker, through the collector's own context code."""
    site = np.asarray(site)
    c = CO.region_context(obs, tracker.latest.get("fast"), site, list(regions))
    return assemble_features(c, nreq_by_xapp(obs["requests"], site, regions))


def horizon_outcomes(trace: Trace, t0s, H: int, site) -> tuple[np.ndarray, np.ndarray]:
    """PRIVILEGED labels: per epoch start t_e, components over (t_e, t_e + H] per region -> (Y (E, R, C), valid (E,)).
    churn = applied request changes (out "ok") + applied rollbacks decided in [t_e, t_e + H), by the knob's region."""
    a, site = trace.arrays, np.asarray(site)
    regions = sorted({int(x) for x in site})
    tmax = int(a["t"][-1]) if len(a["t"]) else 0
    kreg = np.searchsorted(regions, [int(site[k[1]]) for k in trace.meta["knobs"]])
    Y = np.zeros((len(t0s), len(regions), len(COMPONENTS)))
    valid = np.zeros(len(t0s), bool)
    for i, t0 in enumerate(np.asarray(t0s, int)):
        if t0 + H > tmax:
            continue
        valid[i] = True
        y = region_labels(trace, site, t0, t0 + H)
        ch = np.zeros(len(regions))
        m = (a["rq_t"] >= t0) & (a["rq_t"] < t0 + H) & (a["rq_out"] == OUT["ok"])
        np.add.at(ch, kreg[a["rq_knob"][m]], 1)
        m = (a["rb_t"] >= t0) & (a["rb_t"] < t0 + H) & (a["rb_out"] == RB_CODES.index("applied"))
        np.add.at(ch, kreg[a["rb_k"][m]], 1)
        Y[i] = np.column_stack([y["viol"], y["rlf"], y["energy_j"] / 3.6e6, ch])
    return Y, valid


def future_features(fut_codes, H: int, D: int) -> np.ndarray:
    """fut_codes (..., F) region codes of epochs e+1 .. e+F (-1 = none) -> (..., 13) exposure-weighted policy
    features sum_j (H - j D) / H * phi(code_j)."""
    fc = np.asarray(fut_codes, int)
    F = fc.shape[-1]
    w = np.array([(H - j * D) / H for j in range(1, F + 1)])
    ph = np.where((fc >= 0)[..., None], PHI[np.maximum(fc, 0)], 0.0)             # (..., F, 13)
    return np.einsum("...fk,f->...k", ph, w)


def hold_features(codes, H: int, D: int) -> np.ndarray:
    """Future features of holding ``codes`` (...) for the whole horizon -> (..., 13)."""
    F = max(int(np.ceil(H / D)) - 1, 0)
    return policy_features(codes) * sum((H - j * D) / H for j in range(1, F + 1))


def episode_from_trace(trace: Trace, H: int, future: bool | str = "auto") -> dict:
    """Collector trace -> episode dict for ``PolicyEffectModel.fit`` (valid epochs only, i.e. t_e + H <= end).
    Keys: own (E, R, F), glob (E, G), code, prop, prop_eff (E, R), y (E, R, C), t (E,), active (xApp names),
    version (collector version: 1 or 2), D; + fut (E, R, 13) future-assignment features when ``future`` is True or
    "auto" and the trace is v2 data (``collect.future_assignments`` for this H)."""
    a = trace.arrays
    site = np.asarray(trace.meta["cell_region"])
    regions = [int(g) for g in a["pol_region_ids"]]
    xn = trace.meta["xapps"]
    kreg = np.searchsorted(regions, [int(site[k[1]]) for k in trace.meta["knobs"]])
    own, glob = [], []
    for e, t0 in enumerate(a["pol_t"]):
        m = a["rq_t"] == t0                                       # requests offered at the epoch start
        nx = np.zeros((len(regions), len(P.XAPPS)))
        np.add.at(nx, (kreg[a["rq_knob"][m]], [P.XAPPS.index(xn[i]) for i in a["rq_xapp"][m]]), 1)
        c = {k: a["ctx_" + k][e] for k in ("prb_util", "act_ue", "ll_p95", "locked", "report_age", "churn_left")}
        o, g = assemble_features(c, nx)
        own.append(o)
        glob.append(g)
    Y, valid = horizon_outcomes(trace, a["pol_t"], H, site)
    col = trace.meta["collector"]
    out = {"own": np.array(own)[valid], "glob": np.array(glob)[valid], "code": a["pol_code"][valid].astype(int),
           "prop": a["pol_prop"][valid], "prop_eff": a["pol_prop_eff"][valid], "y": Y[valid],
           "t": a["pol_t"][valid], "active": tuple(col["active_xapps"]), "H": int(H),
           "version": int(col.get("version", 1)), "D": int(col.get("D", 20))}
    has_fut = "pol_fut_code" in a
    if future is True and not has_fut:
        raise ValueError("future=True needs v2 collector data (pol_fut_code)")
    if future is True or (future == "auto" and has_fut):
        out["fut"] = future_features(CO.future_assignments(trace, H)["code"], H, out["D"])[valid]
    return out


# ------------------------------------------------------------------------------------------------ context selector
@dataclass
class ContextSelector:
    """Which neighbouring regions' context / policies enter region r's effect model (graph-as-soft-prior hook).
    mode "none": own region only; "all": uniform mean over all other regions; "weights": ``weights`` (R, R)
    non-negative (graph mask, SHAP mask, topology, soft prior), diagonal ignored, rows normalised.
    ``own_columns`` / ``nbr_columns`` / ``modifiers``: names (of OWN or the model's own_names) to use; None = all
    own columns (nbr / modifiers default to the selected own columns)."""
    mode: str = "all"
    weights: np.ndarray | None = None
    own_columns: Sequence[str] | None = None
    nbr_columns: Sequence[str] | None = None
    modifiers: Sequence[str] | None = None
    nbr_policy: bool = True

    def matrix(self, R: int) -> np.ndarray:
        if self.mode == "none":
            return np.zeros((R, R))
        if self.mode == "all":
            return (1.0 - np.eye(R)) / max(R - 1, 1)
        if self.mode != "weights" or self.weights is None:
            raise ValueError("mode must be none | all | weights (with weights)")
        w = np.clip(np.asarray(self.weights, float), 0, None) * (1.0 - np.eye(R))
        s = w.sum(1, keepdims=True)
        return np.divide(w, s, out=np.zeros_like(w), where=s > 0)


def topology_weights(neighbours, site) -> np.ndarray:
    """Region adjacency (R, R) from the cell neighbour lists (obs["static"]["neighbours"]): 1 if any cell pair is
    adjacent. A physical prior (not learned), usable as ``ContextSelector("weights", ...)``."""
    site = np.asarray(site)
    regions = sorted({int(x) for x in site})
    rix = np.searchsorted(regions, site)
    A = np.zeros((len(regions), len(regions)))
    for c, nb in enumerate(neighbours):
        for n in nb:
            A[rix[c], rix[int(n)]] = 1.0
    return A * (1.0 - np.eye(len(regions)))


# ------------------------------------------------------------------------------------------------ the model
@dataclass
class _Basis:
    own_ix: np.ndarray
    nbr_ix: np.ndarray
    mod_ix: np.ndarray
    edges: list = field(default_factory=list)          # per modifier column: interior quantile edges
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None


class PolicyEffectModel:
    """Ensemble estimator of Delta_r(pi | x) per component. ``learner``: "ridge" (basis expansion with explicit
    policy x context interactions, RidgeCV per member; default) or "hgb" (sklearn HistGradientBoosting per
    component on [context, policy]). Members are fitted on bootstrap resamples of EPISODES (of epochs when fewer
    than ``min_boot_episodes`` episodes are given). ``weighting``: "ipw" (stabilized logged-propensity weights,
    clipped at ``clip_w``), "none" (unweighted regression) or "auto" (none for v2 data, else ipw). ``use_future``:
    True / False / "auto" (on iff every episode has ``fut``)."""

    def __init__(self, H: int, n_members: int = 5, learner: str = "ridge", selector: ContextSelector | None = None,
                 weighting: str = "auto", own_names: Sequence[str] = OWN, glob_names: Sequence[str] = GLOB,
                 n_bins: int = 4, alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0), clip_w: float = 30.0,
                 min_boot_episodes: int = 4, seed: int = 0, use_future: bool | str = "auto"):
        if learner not in ("ridge", "hgb") or weighting not in ("ipw", "none", "auto"):
            raise ValueError("learner in {ridge, hgb}, weighting in {ipw, none, auto}")
        self.use_future = use_future
        self.future_on, self.D, self.weighting_used = False, None, None
        self.H, self.K, self.learner, self.weighting = int(H), int(n_members), learner, weighting
        self.selector = selector if selector is not None else ContextSelector()
        self.own_names, self.glob_names = tuple(own_names), tuple(glob_names)
        self.n_bins, self.alphas, self.clip_w = n_bins, tuple(alphas), clip_w
        self.min_boot_episodes, self.seed = min_boot_episodes, seed
        self.members, self.W, self.basis, self.support = [], None, None, None
        self.boot_unit = None

    # ---- data
    @staticmethod
    def stack(episodes: Sequence[Mapping]) -> dict:
        keys = ("own", "glob", "code", "prop", "prop_eff", "y") + (("fut",) if all("fut" in e for e in episodes)
                                                                     else ())
        out = {k: np.concatenate([np.asarray(e[k]) for e in episodes]) for k in keys}
        out["ep"] = np.concatenate([np.full(len(e["code"]), i) for i, e in enumerate(episodes)])
        return out

    def weights(self, code, prop) -> np.ndarray:
        """Stabilized IPW weights s(class) / p(code | x), class = default code vs any other; mean 1 per class."""
        code, p = np.asarray(code).ravel(), np.asarray(prop, float).ravel()
        if (self.weighting_used or self.weighting) == "none":
            return np.ones_like(p)
        w = 1.0 / p
        for m in (code == 0, code != 0):
            if m.any():
                w[m] *= m.sum() / w[m].sum()
        return np.clip(w, 0, self.clip_w)

    def _cols(self, names):
        return np.array([self.own_names.index(n) for n in names], int)

    def _init_basis(self, own_flat):
        s = self.selector
        own_ix = self._cols(s.own_columns) if s.own_columns is not None else np.arange(len(self.own_names))
        nbr_ix = self._cols(s.nbr_columns) if s.nbr_columns is not None else own_ix
        mod_ix = self._cols(s.modifiers) if s.modifiers is not None else own_ix
        edges = []
        for j in mod_ix:
            v = own_flat[:, j]
            e = np.unique(np.nanquantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1])) if self.n_bins > 1 else []
            edges.append(np.asarray(e, float))
        self.basis = _Basis(own_ix, nbr_ix, mod_ix, edges)

    def _fut(self, fut, M, R):
        if not self.future_on:
            return np.zeros((M * R, 0))
        return np.zeros((M * R, len(POLICY_NAMES))) if fut is None else np.asarray(fut, float).reshape(M * R, -1)

    def _raw(self, own, glob, A, A_nbr=None):
        """own (M, R, F), glob (M, G), A (M, R, 13) own-region policies, A_nbr (default A) the policies whose
        neighbour aggregates W @ A_nbr enter -> flat blocks (M * R, ...)."""
        M, R = A.shape[:2]
        A_nbr = A if A_nbr is None else A_nbr
        b, W = self.basis, self.W
        own = np.nan_to_num(np.asarray(own, float))
        glob = np.nan_to_num(np.asarray(glob, float))
        o = own[..., b.own_ix].reshape(M * R, -1)
        g = np.repeat(glob, R, axis=0)
        use_nbr = self.selector.mode != "none"
        nb = np.einsum("rs,msf->mrf", W, own[..., b.nbr_ix]).reshape(M * R, -1) if use_nbr else np.zeros((M * R, 0))
        a = A.reshape(M * R, -1)
        na = np.einsum("rs,msk->mrk", W, A_nbr).reshape(M * R, -1) if use_nbr and self.selector.nbr_policy \
            else np.zeros((M * R, 0))
        mod = own[..., b.mod_ix].reshape(M * R, -1)
        return o, g, nb, a, na, mod

    def _design(self, own, glob, A, A_nbr=None, fut=None):
        o, g, nb, a, na, mod = self._raw(own, glob, A, A_nbr)
        fu = self._fut(fut, *A.shape[:2])
        if self.learner == "hgb":
            return np.hstack([o, g, nb, a, na, fu])
        b = self.basis
        bins = [(mod[:, i:i + 1] > e[None, :]).astype(float) for i, e in enumerate(b.edges) if len(e)]
        bins = np.hstack(bins) if bins else np.zeros((len(o), 0))
        zmod = (mod - b.mu[: mod.shape[1]]) / b.sd[: mod.shape[1]]
        h = np.hstack([np.ones((len(o), 1)), zmod, bins])                  # effect modifiers (span of the base)
        inter = (a[:, :, None] * h[:, None, :]).reshape(len(o), -1)
        ninter = (na[:, :, None] * h[:, None, : 1 + zmod.shape[1]]).reshape(len(o), -1)
        return np.hstack([o, g, nb, bins, a, na, inter, ninter, fu])

    # ---- fit / predict
    def fit(self, episodes: Sequence[Mapping]) -> PolicyEffectModel:
        for e in episodes:
            if int(e.get("H", self.H)) != self.H:
                raise ValueError(f"episode labels are for H={e.get('H')}, model H={self.H}")
        d = self.stack(episodes)
        M, R = d["code"].shape
        self.R = R
        if self.use_future is True and "fut" not in d:
            raise ValueError("use_future=True needs episodes with fut (v2 collector data)")
        self.future_on = "fut" in d and self.use_future in (True, "auto")
        self.D = int(episodes[0].get("D", 20))
        v2 = all(int(e.get("version", 1)) == 2 for e in episodes)
        self.weighting_used = ("none" if v2 else "ipw") if self.weighting == "auto" else self.weighting
        self.W = self.selector.matrix(R)
        own_flat = d["own"].reshape(M * R, -1)
        self._init_basis(own_flat)
        mod = np.nan_to_num(own_flat[:, self.basis.mod_ix])
        self.basis.mu, self.basis.sd = mod.mean(0), mod.std(0) + 1e-9
        self.support = SupportDetector().fit(np.hstack([own_flat, np.repeat(d["glob"], R, axis=0)]))
        X = self._design(d["own"], d["glob"], policy_features(d["code"]), fut=d.get("fut"))
        Y = d["y"].reshape(M * R, -1)
        w = self.weights(d["code"], d["prop"])
        ep = d["ep"]
        n_ep = len(np.unique(ep))
        self.boot_unit = "episode" if n_ep >= self.min_boot_episodes else "epoch"
        unit = np.repeat(ep if self.boot_unit == "episode" else np.arange(M), R)
        ids = np.unique(unit)
        rng = np.random.default_rng([self.seed, 4242])
        self.members = []
        for k in range(self.K):
            pick = rng.choice(ids, size=len(ids), replace=True)
            cnt = np.bincount(np.searchsorted(ids, pick), minlength=len(ids))
            wk = w * cnt[np.searchsorted(ids, unit)]                    # bootstrap multiplicity as a weight
            m = wk > 0
            self.members.append(self._fit_member(X[m], Y[m], wk[m], k))
        return self

    def _fit_member(self, X, Y, w, k):
        mu, sd = X.mean(0), X.std(0)
        sd[sd < 1e-12] = 1.0
        Z = (X - mu) / sd
        if self.learner == "ridge":
            from sklearn.linear_model import RidgeCV
            r = RidgeCV(alphas=self.alphas, alpha_per_target=True).fit(Z, Y, sample_weight=w)
            return ("ridge", mu, sd, r.coef_.T.copy(), np.asarray(r.intercept_, float).copy())
        from sklearn.ensemble import HistGradientBoostingRegressor
        ms = [HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, min_samples_leaf=20,
                                            random_state=self.seed * 1000 + k * 10 + j).fit(Z, Y[:, j], sample_weight=w)
              for j in range(Y.shape[1])]
        return ("hgb", mu, sd, ms, None)

    @staticmethod
    def _predict_member(mem, X):
        kind, mu, sd, a, b = mem
        Z = (X - mu) / sd
        if kind == "ridge":
            return Z @ a + b
        return np.column_stack([m.predict(Z) for m in a])

    def member_effects(self, own, glob, codes, continuation: str = "accept_all") -> np.ndarray:
        """Per-member contrasts vs accept-all. own (R, F) or (M, R, F); glob (G,) or (M, G); codes (M, R) of the
        region policies (neighbour policies = the other regions' codes in the same row) -> (K, M, R, C).
        ``continuation`` (only with future features): "accept_all" -> fut = 0 on both sides; "hold" -> the
        candidate's fut = its own policy held over H (the reference keeps fut = 0)."""
        if continuation not in ("accept_all", "hold"):
            raise ValueError("continuation in {accept_all, hold}")
        codes = np.atleast_2d(np.asarray(codes, int))
        M, R = codes.shape
        own = np.broadcast_to(own, (M, R, np.shape(own)[-1]))
        glob = np.broadcast_to(np.asarray(glob, float), (M, np.shape(glob)[-1]))
        fut = hold_features(codes, self.H, self.D) if continuation == "hold" and self.future_on else None
        Xa = self._design(own, glob, policy_features(codes), fut=fut)
        X0 = self._design(own, glob, np.zeros((M, R, len(POLICY_NAMES))))
        out = [(self._predict_member(m, Xa) - self._predict_member(m, X0)).reshape(M, R, -1) for m in self.members]
        return np.array(out)

    def effects(self, own, glob, codes) -> np.ndarray:
        return self.member_effects(own, glob, codes).mean(0)

    @staticmethod
    def cost_weights(lam_e: float, w_ll: float, w_rlf: float = 0.0, w_churn: float = 0.0) -> np.ndarray:
        """Component prices: world_model.objective (viol + w_ll * LL viol + lam_e * kWh) + optional RLF / churn."""
        return np.array([1.0 + w_ll, 1.0, 1.0, w_rlf, lam_e, w_churn])

    # ---- doubly-robust (AIPW) contrast for one candidate
    def aipw(self, episodes: Sequence[Mapping], candidate: int, n_folds: int = 2, seed: int = 0) -> dict:
        """Cross-fitted (by episode) AIPW pseudo-outcomes of ``candidate`` code vs accept-all, per row (M * R, C):
        psi = m(x, pi) - m(x, 0) + [A ~ pi] / p (Y - m(x, pi)) - [A ~ 0] / p (Y - m(x, 0)), with ~ = same policy
        class on the episodes' active xApps (inert xApps' modes marginalised), p = the logged class propensity
        (prop_eff), m = out-of-fold ensemble mean. Neighbour policies are kept as logged. Returns {"psi", "ep" (row episode)}; use ``clustered_mean`` for a mean and episode-clustered SE."""
        d = self.stack(episodes)
        M, R = d["code"].shape
        active = tuple(episodes[0]["active"])
        cls_pi, cls_0 = policy_class(candidate, active), policy_class(0, active)
        cls = [policy_class(c, active) for c in d["code"].ravel()]
        is_pi = np.array([c == cls_pi for c in cls], float)
        is_0 = np.array([c == cls_0 for c in cls], float)
        n_ep = len(episodes)
        fold = np.random.default_rng([seed, 99]).permutation(n_ep) % n_folds
        psi = np.zeros((M * R, d["y"].shape[-1]))
        y = d["y"].reshape(M * R, -1)
        pe = d["prop_eff"].ravel()
        for f in range(n_folds):
            tr = [episodes[i] for i in range(n_ep) if fold[i] != f]
            te = np.isin(d["ep"], np.nonzero(fold == f)[0])
            if not te.any():
                continue
            sub = PolicyEffectModel(self.H, self.K, self.learner, self.selector, self.weighting, self.own_names,
                                    self.glob_names, self.n_bins, self.alphas, self.clip_w,
                                    self.min_boot_episodes, self.seed + 1 + f, self.use_future).fit(tr)
            own, glob, code = d["own"][te], d["glob"][te], d["code"][te]
            fut = d["fut"][te] if "fut" in d else None         # logged future assignments kept as covariates
            A = policy_features(code)
            rows = np.repeat(te, R)

            def mu(A_own, own=own, glob=glob, A=A, sub=sub, fut=fut):
                # own region's policy replaced, neighbours' policies and future assignments as logged
                X = sub._design(own, glob, A_own, A, fut=fut)
                return np.mean([sub._predict_member(m, X) for m in sub.members], axis=0)

            mpi, m0 = mu(policy_features(np.full(code.shape, candidate))), mu(np.zeros_like(A))
            ipi, i0 = (is_pi[rows] / pe[rows])[:, None], (is_0[rows] / pe[rows])[:, None]
            psi[rows] = mpi - m0 + ipi * (y[rows] - mpi) - i0 * (y[rows] - m0)
        return {"psi": psi, "ep": np.repeat(d["ep"], R)}


def clustered_mean(psi, ep, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Mean of per-row values and its episode-clustered standard error (per column)."""
    psi, ep = np.asarray(psi, float), np.asarray(ep)
    if mask is not None:
        psi, ep = psi[mask], ep[mask]
    m = psi.mean(0)
    ids = np.unique(ep)
    s = np.array([(psi[ep == e] - m).sum(0) for e in ids])
    return m, np.sqrt((s ** 2).sum(0)) / len(psi)


# ------------------------------------------------------------------------------------------------ WM adapter
@dataclass(frozen=True)
class EffectScore(Score):
    members: tuple = ()
    ood: bool = False
    stratum: object = None


class EffectWM:
    """WorldModel adapter (non-privileged) for WG3Arbiter: J(plan) = sum over regions of the priced predicted
    contrast vs accept-all (so accept-all scores exactly 0). ``observe(obs)`` must be fed every second
    (``run_episode``). A plan is ``ood`` if it deviates from accept-all in a region whose context is outside the
    training support, or if the tracked context is stale. ``stratifier(own, glob) -> hashable`` tags scores with a
    calibration stratum for the gate. ``continuation`` = what the future-assignment features are set to
    (module docstring): "accept_all" (default; the arbiter's declared default continuation) or "hold"."""
    privileged = False

    def __init__(self, model: PolicyEffectModel, site, w_rlf: float = 0.0, w_churn: float = 0.0,
                 stratifier: Callable | None = None, check_support: bool = True, continuation: str = "accept_all"):
        if continuation not in ("accept_all", "hold"):
            raise ValueError("continuation in {accept_all, hold}")
        self.continuation = continuation
        self.model, self.site = model, np.asarray(site)
        self.regions = sorted({int(x) for x in self.site})
        self.w_rlf, self.w_churn, self.stratifier, self.check_support = w_rlf, w_churn, stratifier, check_support
        self.tracker = ReportTracker()
        self.last = None                                  # (plans, members (K, P), ood (P,)) of the last call
        self.n_score = 0

    def observe(self, obs: Mapping) -> None:
        self.tracker.observe(obs)

    def features(self, obs: Mapping):
        return serve_features(obs, self.tracker, self.site, self.regions)

    def score(self, ctx: DecisionContext, plans: Sequence[P.Plan]) -> list[Score]:
        if int(ctx.H) != self.model.H:
            raise ValueError(f"effect model was fitted for H={self.model.H}, arbiter asks H={ctx.H}")
        self.observe(ctx.obs)
        own, glob = self.features(ctx.obs)
        codes = np.array([[CO.encode(p[g]) for g in self.regions] for p in plans], int)
        cw = self.model.cost_weights(ctx.lam_e, ctx.w_ll, self.w_rlf, self.w_churn)
        mem = (self.model.member_effects(own, glob, codes, self.continuation) @ cw).sum(-1)   # (K, P)
        bad_region = np.zeros(len(self.regions), bool)
        if self.check_support and self.model.support is not None:
            bad_region = self.model.support.outside(np.hstack([own, np.repeat(glob[None], len(own), 0)]))
        stale = not self.tracker.complete or not np.isfinite(own).all()
        ood = stale | ((codes != 0) & bad_region[None, :]).any(1)
        ood &= (codes != 0).any(1)                                                  # accept-all is never ood
        st = self.stratifier(own, glob) if self.stratifier is not None else None
        self.last, self.n_score = (plans, mem, ood), self.n_score + len(plans)
        return [EffectScore(float(mem[:, j].mean()), float(mem[:, j].std()), tuple(map(float, mem[:, j])),
                            bool(ood[j]), st) for j in range(len(plans))]


def run_episode(env, arbiter) -> dict:
    """``arbiter.run_episode`` + feeds every second's obs to the world model's ``observe`` (if it has one)."""
    obs_hook = getattr(arbiter.wm, "observe", None)
    while env.sec < env.total_s:
        obs = env.step_propose()
        if obs_hook is not None:
            obs_hook(obs)
        dec = arbiter.act(obs, env.last_change, env)
        env.step_apply(dec)
        arbiter.record(dec, env.last_change)
    return env.score()

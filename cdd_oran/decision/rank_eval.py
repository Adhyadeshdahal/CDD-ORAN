"""Step-5 ranking evaluation of the learned policy-effect world model on held-out ORACLE PANELS.

SOL_REVIEW.md build order (5): evaluate the response model (+ support gate / calibrated gate) on held-out oracle
panels against accept-all, all-input and graph-selected baselines; the requirement is LOWER PAIRED DECISION REGRET
THAN ACCEPT-ALL with credible (episode-clustered) uncertainty and no guardrail failure. This module measures; it
claims nothing. It re-tests the D1 failure (scratchpad/pact/endtask/D012_SUMMARY.md: passive learners picked worse
candidates than accept-all, within-state rho ~ 0.3) on the E6 estimand of docs/benchmark/E6_COLLECTION_CONTRACT.md.

Pieces
  ``fit_from_npz``   PolicyEffectModel on v3 FIT episodes only (contract e6-collect-v3: j < 20 per stratum),
                     intention-to-treat on ALL randomized slots (no eligibility selection), UNWEIGHTED (context-free
                     propensities), with a ContextSelector for the graph ablation: none / topology / all / weights.
                     Also returns the first-stage support counts (``effect_model.support_counts``) so evaluation
                     applies the predeclared SupportRule(20, 20, 10) per stratum.
  ``oracle_panel``   on one (diagnostic) episode run under accept-all (= the v3 paired reference: wg3=True, every
                     request accepted, no rollback), at each slot start copy the env and score a FIXED candidate set
                     with ``TrueSimWM(continuation="accept_all")`` (plan for D s, then accept-all; horizon H):
                       accept-all (index 0), freeze (network reject), uniform single-xApp modes (one xApp in one
                       mode network-wide, the rest accept), network rollback, each learned model's top-k plans from
                       its own WG3Arbiter search (supported scores, no gate, incumbent = accept-all, the arbiter's
                       RNG [seed, t, 13]), and random plans (RNG [cfg.seed, PANEL_KEY, slot]: first half
                       network-uniform ``plans.random_region``, rest independent per region).
                     Every candidate's oracle components are recorded (violated UE-s per slice, outage, severe, RLF,
                     kWh, HO, ping-pong, churn) so any price vector can be re-scored offline; learned predictions
                     are recorded raw (no support gating), with the support gate's unidentified parts and the
                     context ``ood`` flag separately.
  ``slot_metrics``   per slot and model: picks (raw argmin / supported argmin / gated), decision regret vs the
                     panel's oracle best, accept-all regret, paired improvement over accept-all, within-slot
                     Spearman rho (all candidates with raw predictions; supported subset), abstentions.
  ``summarize``      episode-clustered bootstrap CIs (``cluster_bootstrap``; cluster = episode) of every metric and
                     of the guardrail component deltas (pick - accept-all), pooled and per stratum.
  ``shrink_rows``    offline re-scoring of stored slot records with EB shrinkage toward the accept-all contrast
                     (``effect_model.eb_factors`` from the recorded mean / member std; no simulation).
  ``beats_zero_contrast``  the precondition "the model's picks beat accept-all (zero contrast) on held-out DEV
                     slots": episode-clustered CI of the paired improvement pick vs accept-all; beats iff lo > 0.

Timing semantics = TrueSimWM's: at slot start t the second t has already been played; the rollout applies the plan
at t and plays seconds t+1 .. t+H, i.e. the window (t, t+H] -- the same window as effect_model's labels. The
accept-all candidate therefore equals the objective the accept-all main run realises over those seconds (checked per
slot: ``realized_accept_all``).
RNG tags: PANEL_KEY 1717 (random panel plans), BOOT_KEY 5353 (bootstrap). Arbiter search: 13 (unchanged).
"""
from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import collect as CO
from . import effect_model as EM
from . import plans as P
from .adapters.e6 import region_map
from .arbiter import WG3Arbiter
from .trace import Trace
from .world_model import DecisionContext, TrueSimWM, objective

RANK_VERSION = "e6-rank-eval/1"
PANEL_KEY, BOOT_KEY = 1717, 5353
LAM_E, W_LL = 1e4, 5.0                    # the priced primary objective of gate B (scratchpad/e6_dev/gate_b.py)
ORACLE_COMPONENTS = ("J", "viol", "viol_LL", "viol_eMBB", "viol_BE", "outage", "severe", "rlf", "energy_kwh",
                     "ho", "pingpong", "churn")
GUARDRAILS = ("viol_LL", "viol_eMBB", "viol_BE", "rlf", "severe", "energy_kwh", "churn")
SELECTORS = ("none", "topology", "all", "weights")
SUPPORT_RULE = EM.SupportRule()            # predeclared (20, 20, 10), contract section 4


# ================================================================================================ fitting
@dataclass
class Fitted:
    """A fitted model + what evaluation needs: first-stage support counts (per stratum) and a manifest."""
    model: EM.PolicyEffectModel
    counts: dict
    manifest: dict = field(default_factory=dict)

    def support(self, stratum, rule: EM.SupportRule | None = SUPPORT_RULE) -> dict | None:
        """{(xapp, mode): identified} for (scenario, load); None = no support gating (rule None)."""
        return None if rule is None else rule.identified(self.counts, tuple(stratum))


def _neighbours_from_knobs(knobs, n_cells: int) -> list:
    """Cell neighbour lists from the trace's CIO knobs ("cio", s, n) (= layout neighbours and their reverses)."""
    nb = [set() for _ in range(n_cells)]
    for k in knobs:
        if k[0] == "cio":
            nb[int(k[1])].add(int(k[2]))
    return [sorted(x) for x in nb]


def trace_topology(tr: Trace) -> np.ndarray:
    """Region adjacency (R, R) of one episode (the pico placement, hence the adjacency, varies by seed)."""
    site = np.asarray(tr.meta["cell_region"])
    return EM.topology_weights(_neighbours_from_knobs(tr.meta["knobs"], len(site)), site)


def _check_fit_trace(tr: Trace, path, require_fit: bool) -> None:
    col = tr.meta.get("collector", {})
    if int(col.get("version", 0)) != 3:
        raise ValueError(f"{path}: not a v3 collector trace (e6-collect-v3)")
    mix = col.get("mixture", {})
    if float(mix.get("p_accept", 0.0)) >= 1.0:
        raise ValueError(f"{path}: accept-all reference trace (collection cost only, never fitted)")
    if require_fit:
        st = CO.dev_stratum(int(tr.meta["cfg"]["seed"]))
        if st["kind"] != "policy" or st["split"] != "fit":
            raise ValueError(f"{path}: seed {tr.meta['cfg']['seed']} is not a v3 FIT seed ({st})")


def fit_from_npz(paths: Sequence, H: int = 90, selector: str | EM.ContextSelector = "all",
                 weights: np.ndarray | None = None, n_members: int = 5, learner: str = "ridge", seed: int = 0,
                 require_fit: bool = True, **model_kw) -> Fitted:
    """Fit a PolicyEffectModel on v3 collector npz traces (FIT seeds only when ``require_fit``; refs refused).
    ITT on every labelled randomized slot, unweighted (``weighting="none"``). ``selector``: "none" | "topology"
    (mean region adjacency over the fit episodes, a soft physical prior) | "all" | "weights" (``weights`` (R, R),
    e.g. MSCR / SHAP masks) | a ContextSelector."""
    trs = []
    for p in paths:
        tr = Trace.from_npz(p)
        _check_fit_trace(tr, p, require_fit)
        trs.append(tr)
    if not trs:
        raise ValueError("no traces")
    if isinstance(selector, EM.ContextSelector):
        sel, sel_name = selector, "custom"
    elif selector == "topology":
        sel, sel_name = EM.ContextSelector("weights", np.mean([trace_topology(t) for t in trs], axis=0)), "topology"
    elif selector == "weights":
        if weights is None:
            raise ValueError("selector='weights' needs weights (R, R)")
        sel, sel_name = EM.ContextSelector("weights", np.asarray(weights, float)), "weights"
    elif selector in ("none", "all"):
        sel, sel_name = EM.ContextSelector(selector), selector
    else:
        raise ValueError(f"selector in {SELECTORS} or a ContextSelector")
    eps = [EM.episode_from_trace(t, H) for t in trs]
    model = EM.PolicyEffectModel(H, n_members=n_members, learner=learner, selector=sel, weighting="none",
                                 seed=seed, **model_kw).fit(eps)
    strata = {}
    for t in trs:
        k = f"{t.meta['cfg'].get('scenario', 'base')}-{t.meta['cfg']['load']}"
        strata[k] = strata.get(k, 0) + 1
    man = {"version": RANK_VERSION, "selector": sel_name, "H": int(H), "learner": learner, "n_members": n_members,
           "seed": seed, "n_episodes": len(trs), "n_rows": int(sum(len(e["code"]) * e["code"].shape[1] for e in eps)),
           "seeds": sorted(int(t.meta["cfg"]["seed"]) for t in trs), "strata": strata,
           "weighting": model.weighting_used, "boot_unit": model.boot_unit}
    return Fitted(model, EM.support_counts(trs, H), man)


def save_bundle(path, fitted: Mapping[str, Fitted]) -> None:
    with open(path, "wb") as f:
        pickle.dump({"version": RANK_VERSION, "models": dict(fitted)}, f)


def load_bundle(path) -> dict[str, Fitted]:
    with open(path, "rb") as f:
        b = pickle.load(f)
    if b.get("version") != RANK_VERSION:
        raise ValueError(f"bundle version {b.get('version')} != {RANK_VERSION}")
    return b["models"]


# ================================================================================================ candidate panel
@dataclass(frozen=True)
class CandidateSpec:
    """Declared candidate panel (see module docstring). ``single_modes``: modes for the uniform single-xApp plans."""
    top_k: int = 4
    n_random: int = 6
    single_modes: tuple = ("reject", "half", "lock")
    freeze: bool = True
    rollback: bool = True
    search: Mapping = field(default_factory=lambda: {"n_glob": 6, "n_loc": 2})


def plan_key(plan: P.Plan, regions) -> tuple:
    return tuple(CO.encode(plan[g]) for g in regions)


def plan_of(codes, regions) -> P.Plan:
    return {int(g): CO.decode(int(c)) for g, c in zip(regions, codes, strict=True)}


def fixed_candidates(regions, spec: CandidateSpec) -> list[tuple[str, P.Plan]]:
    out = [("accept_all", P.accept_all(regions))]
    if spec.freeze:
        out.append(("freeze", P.network(regions, P.uniform("reject"))))
    for x in P.XAPPS:
        for m in spec.single_modes:
            rp = {"mode": {y: (m if y == x else "accept") for y in P.XAPPS}, "rb": 0}
            out.append((f"single:{x}:{m}", P.network(regions, rp)))
    if spec.rollback:
        out.append(("rollback", P.network(regions, P.uniform("accept", rb=1))))
    return out


def random_candidates(seed: int, slot: int, regions, n: int) -> list[tuple[str, P.Plan]]:
    r = np.random.default_rng([int(seed), PANEL_KEY, int(slot)])
    out = []
    for i in range(n):
        if i < (n + 1) // 2:
            out.append(("random:network", P.network(regions, P.random_region(r))))
        else:
            out.append(("random:region", {int(g): P.random_region(r) for g in regions}))
    return out


class _RecordingWM:
    """Delegating world model that records every (plan, score) the arbiter's search asks for."""

    def __init__(self, wm):
        self.wm, self.seen = wm, []
        self.privileged = False

    def score(self, ctx, plans):
        s = self.wm.score(ctx, plans)
        self.seen += list(zip(plans, s, strict=True))
        return s


def search_top_k(arb: WG3Arbiter, rec: _RecordingWM, ctx: DecisionContext, k: int) -> tuple[P.Plan, list]:
    """Run the arbiter's own (ungated) search from an accept-all incumbent; -> (its pick, top-k distinct plans by
    learned mean among the finite scores it evaluated; the pick is first)."""
    rec.seen = []
    arb.plan, conf, arb.confidence = P.accept_all(arb.regions), arb.confidence, None
    try:
        pick = arb.choose(ctx)
    finally:
        arb.confidence = conf
    uniq = {}
    for p, s in rec.seen:
        key = plan_key(p, arb.regions)
        if np.isfinite(s.mean) and (key not in uniq or s.mean < uniq[key][1]):
            uniq[key] = (p, s.mean)
    top = sorted(uniq.values(), key=lambda v: v[1])
    kp = plan_key(pick, arb.regions)
    out = [pick] + [p for p, _ in top if plan_key(p, arb.regions) != kp]
    return pick, out[:k]


# ================================================================================================ oracle scoring
class _Tap:
    """Env proxy whose copy() is remembered, so the unchanged TrueSimWM rollout can be read out per component."""

    def __init__(self, env):
        self.env, self.sims = env, []

    def copy(self):
        s = self.env.copy()
        self.sims.append(s)
        return s


def sla_components(sla0: Mapping, sla1: Mapping, ch0: int, ch1: int, lam_e: float, w_ll: float) -> dict:
    d = {k: float(sla1[k] - sla0[k]) for k in sla0}
    return {"J": objective(sla0, sla1, lam_e, w_ll), "viol": d["viol_ue_s"], "viol_LL": d["ll_viol"],
            "viol_eMBB": d["embb_viol"], "viol_BE": d["viol_ue_s"] - d["ll_viol"] - d["embb_viol"],
            "outage": d["outage_viol"], "severe": d["severe"], "rlf": d["rlf"], "energy_kwh": d["energy_j"] / 3.6e6,
            "ho": d["ho"], "pingpong": d["pingpong"], "churn": float(ch1 - ch0)}


def oracle_components(ctx: DecisionContext, plans: Sequence[P.Plan], wm: TrueSimWM | None = None) -> list[dict]:
    """Per plan: TrueSimWM's objective (its own rollout, unchanged) + the rollout's components."""
    wm = wm or TrueSimWM("accept_all")
    env = ctx.env
    sla0, ch0 = dict(env.plant.sla), int(env.stats["changes"])
    out = []
    for p in plans:
        tap = _Tap(env)
        J = wm.score(dataclasses.replace(ctx, env=tap), [p])[0].mean
        sim = tap.sims[-1]
        c = sla_components(sla0, sim.plant.sla, ch0, int(sim.stats["changes"]), ctx.lam_e, ctx.w_ll)
        c["J"] = float(J)
        out.append(c)
    return out


# ================================================================================================ metrics
def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def slot_metrics(oracle_J, pred, unidentified=None, ood=None, allowed=None) -> dict:
    """One slot, one model. ``oracle_J`` (P,) oracle objective per candidate (lower = better; index 0 = accept-all);
    ``pred`` (P,) the model's RAW predicted contrast vs accept-all (accept-all = 0); ``unidentified`` (P,) the
    support gate excludes the plan; ``ood`` (P,) context outside the training support / stale; ``allowed`` (P,)
    a calibrated gate's verdict (None = no gate). Picks (ties -> lowest index, so accept-all wins ties):
      raw = argmin pred; sup = argmin over supported plans (the ungated arbiter); gated = argmin over supported,
      in-support (not ood) and gate-allowed plans (accept-all always allowed).
    Regret = J[pick] - min J; improvement = J[accept-all] - J[pick] (> 0 = better than accept-all)."""
    J, pr = np.asarray(oracle_J, float), np.asarray(pred, float)
    n = len(J)
    un = np.zeros(n, bool) if unidentified is None else np.asarray(unidentified, bool).copy()
    oo = np.zeros(n, bool) if ood is None else np.asarray(ood, bool).copy()
    al = np.ones(n, bool) if allowed is None else np.asarray(allowed, bool).copy()
    un[0], oo[0], al[0] = False, False, True
    best = float(J.min())

    def pick(mask):
        v = np.where(mask & np.isfinite(pr), pr, np.inf)
        return int(np.argmin(v)) if np.isfinite(v).any() else 0

    i_raw, i_sup = pick(np.ones(n, bool)), pick(~un)
    i_gat = pick(~un & ~oo & al)
    sup = ~un
    return {"n_cand": n, "oracle_best": best, "oracle_best_idx": int(np.argmin(J)),
            "pick_raw": i_raw, "pick": i_sup, "pick_gated": i_gat,
            "regret_raw": float(J[i_raw] - best), "regret": float(J[i_sup] - best),
            "regret_gated": float(J[i_gat] - best), "regret_accept_all": float(J[0] - best),
            "improvement_raw": float(J[0] - J[i_raw]), "improvement": float(J[0] - J[i_sup]),
            "improvement_gated": float(J[0] - J[i_gat]),
            "rho_raw": _spearman(pr, J), "rho_supported": _spearman(np.where(sup, pr, np.nan), J),
            "deviate": bool(i_sup != 0), "deviate_gated": bool(i_gat != 0),
            "support_abstain": bool(un[i_raw]), "ood_abstain": bool(i_sup != 0 and oo[i_sup]),
            "gate_abstain": bool(i_sup != 0 and not oo[i_sup] and not al[i_sup]),
            "n_unidentified": int(un.sum()), "n_ood": int(oo.sum())}


def cluster_bootstrap(values, clusters, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean of per-row values with a percentile CI from resampling whole CLUSTERS (episodes) with replacement
    (ratio estimator: pooled mean of the resampled clusters' rows). Non-finite rows are dropped."""
    v, c = np.asarray(values, float), np.asarray(clusters)
    m = np.isfinite(v)
    v, c = v[m], c[m]
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_clusters": 0}
    ids, inv = np.unique(c, return_inverse=True)
    s = np.bincount(inv, weights=v, minlength=len(ids))
    k = np.bincount(inv, minlength=len(ids)).astype(float)
    idx = np.random.default_rng([int(seed), BOOT_KEY]).integers(len(ids), size=(n_boot, len(ids)))
    bm = s[idx].sum(1) / k[idx].sum(1)
    lo, hi = np.quantile(bm, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(v)), "n_clusters": len(ids)}


# ================================================================================================ the panel
def default_slot_times(cfg, H: int = 90, slot_s: int = 90) -> list[int]:
    """The v3 labelled slot starts: warm-up end + k * slot_s with t + H <= end of the episode."""
    t0, end = int(cfg.warmup_s), int(cfg.warmup_s + cfg.scored_s)
    return [t for t in range(t0, end + 1, slot_s) if t + H <= end]


def _as_fitted(m) -> Fitted:
    return m if isinstance(m, Fitted) else Fitted(m, {}, {})


def oracle_panel(cfg, slot_times: Sequence[int] | None = None, candidates: CandidateSpec | None = None,
                 models: Mapping | None = None, support_rule: EM.SupportRule | None = SUPPORT_RULE,
                 gates: Mapping | None = None, lam_e: float = LAM_E, w_ll: float = W_LL, D: int = 20, H: int = 90,
                 on_slot: Callable[[dict], None] | None = None, shrink: str | None = None) -> list[dict]:
    """Oracle panels of one episode (module docstring). ``models``: {name: Fitted | PolicyEffectModel} (every
    model's H must equal ``H``); ``support_rule`` None = no support gating; ``gates``: optional {name:
    ConformalGate} (calibrated on calib seeds) for the gated pick; ``shrink`` (None | "eb") is passed to every
    EffectWM (search, raw and gated scores). Returns one record per slot (also passed to ``on_slot``). PRIVILEGED (reads the true simulator); run only on diagnostic / evaluation seeds."""
    from cdd_oran.envs.e6.env import E6Env
    candidates = candidates if candidates is not None else CandidateSpec()
    models = {k: _as_fitted(v) for k, v in (models or {}).items()}
    gates = gates or {}
    env = E6Env(cfg, log=False, wg3=True)
    site = region_map(env)
    regions = sorted({int(x) for x in site})
    slots = sorted(int(t) for t in (default_slot_times(cfg, H) if slot_times is None else slot_times))
    if slots and slots[-1] + H > env.total_s:
        raise ValueError("every slot needs t + H <= total_s (the oracle rollout must fit)")
    stratum = (getattr(cfg, "scenario", "base"), cfg.load)
    wms = {}
    for name, f in models.items():
        if f.model.H != H:
            raise ValueError(f"model {name} has H={f.model.H}, panel H={H}")
        sup = f.support(stratum, support_rule) if support_rule is not None else None
        raw = EM.EffectWM(f.model, site, shrink=shrink)
        gated = EM.EffectWM(f.model, site, support=sup, shrink=shrink)
        rec = _RecordingWM(gated)
        arb = WG3Arbiter(rec, site, lam_e, w_ll, D=D, H=H, seed=cfg.seed, start_s=float(cfg.warmup_s),
                         **dict(candidates.search))
        wms[name] = (raw, gated, rec, arb, sup)
    true_wm = TrueSimWM("accept_all")
    out, pending = [], []
    while env.sec < env.total_s and (len(out) < len(slots)):
        obs = env.step_propose()
        for raw, gated, *_ in wms.values():
            raw.observe(obs)
            gated.observe(obs)
        t = int(obs["t"])
        if t in slots:
            k = slots.index(t)
            ctx = DecisionContext(obs, site, regions, H, float(D), lam_e, w_ll, dict(env.last_change), {}, env, {})
            ctx_l = dataclasses.replace(ctx, env=None)                     # learned models never see the env
            cands = fixed_candidates(regions, candidates)
            picks = {}
            for name, (_, _, rec, arb, _) in wms.items():
                pick, top = search_top_k(arb, rec, ctx_l, candidates.top_k)
                picks[name] = plan_key(pick, regions)
                cands += [(f"search:{name}", p) for p in top]
            cands += random_candidates(cfg.seed, k, regions, candidates.n_random)
            keys, plans, srcs = [], [], []
            for src, p in cands:
                key = plan_key(p, regions)
                if key in keys:
                    srcs[keys.index(key)].append(src)
                else:
                    keys.append(key)
                    plans.append(p)
                    srcs.append([src])
            comps = oracle_components(ctx, plans, true_wm)
            orc = {c: [float(x[c]) for x in comps] for c in ORACLE_COMPONENTS}
            codes = np.array(keys, int)
            rec_m = {}
            for name, (raw, gated, _, _, sup) in wms.items():
                s_raw = raw.score(ctx_l, plans)
                unid = [bool(EM.unidentified_parts(p, sup)) if sup is not None else False for p in plans]
                ood = [bool(s.ood) for s in s_raw]
                pred = [float(s.mean) for s in s_raw]
                g = gates.get(name)
                allowed = None
                if g is not None:
                    s_sup = gated.score(ctx_l, plans)
                    allowed = [bool(j == 0 or (np.isfinite(s_sup[j].mean) and g.confidence(s_sup[j], s_sup[0])))
                               for j in range(len(plans))]
                met = slot_metrics(orc["J"], pred, unid, ood, allowed)
                own, glob = raw.features(obs)
                pc = models[name].model.member_effects(own, glob, codes).mean(0).sum(1)      # (P, C) over regions
                guard = {c: orc[c][met["pick"]] - orc[c][0] for c in GUARDRAILS}
                guard_g = {c: orc[c][met["pick_gated"]] - orc[c][0] for c in GUARDRAILS}
                rec_m[name] = {"pred": pred, "std": [float(s.std) for s in s_raw], "ood": ood, "unid": unid,
                               "allowed": allowed, "pred_comp": np.round(pc, 6).tolist(),
                               "search_pick": keys.index(picks[name]), "metrics": met, "guard": guard,
                               "guard_gated": guard_g}
            row = {"type": "slot", "version": RANK_VERSION, "seed": int(cfg.seed), "scenario": stratum[0],
                   "load": stratum[1], "k": k, "t": t, "H": H, "D": D, "lam_e": lam_e, "w_ll": w_ll, "shrink": shrink,
                   "cands": [{"src": s, "codes": list(map(int, c))} for s, c in zip(srcs, keys, strict=True)],
                   "oracle": orc, "models": rec_m, "pred_components": list(EM.COMPONENTS)}
            pending.append((t, dict(env.plant.sla), int(env.stats["changes"]), row))
        env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
        for p in [p for p in pending if env.sec >= min(p[0] + H, env.total_s)]:
            pending.remove(p)
            p[3]["realized_accept_all"] = sla_components(p[1], env.plant.sla, p[2], int(env.stats["changes"]),
                                                         lam_e, w_ll)
            out.append(p[3])
            if on_slot is not None:
                on_slot(p[3])
    return out


# ================================================================================================ aggregation
METRICS = ("regret", "regret_gated", "regret_raw", "regret_accept_all", "improvement", "improvement_gated",
           "improvement_raw", "rho_raw", "rho_supported")


def summarize(rows: Sequence[Mapping], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
              by_stratum: bool = True) -> dict:
    """Slot records -> {model: {"pooled": {...}, "strata": {"scn-load": {...}}}} with episode-clustered bootstrap
    CIs of METRICS and of the guardrail deltas (pick - accept-all; > 0 = worse), abstention and deviation counts,
    and the SOL_REVIEW step-5 criterion evaluated mechanically (not a claim): gated improvement CI lower bound > 0
    and no guardrail delta with CI lower bound > 0."""
    rows = [r for r in rows if r.get("type", "slot") == "slot"]
    names = sorted({m for r in rows for m in r["models"]})

    def block(rs, name):
        ep = [r["seed"] for r in rs]
        mt = [r["models"][name]["metrics"] for r in rs]
        o = {k: cluster_bootstrap([m[k] for m in mt], ep, n_boot, alpha, seed) for k in METRICS}
        for key in ("guard", "guard_gated"):
            o[key] = {c: cluster_bootstrap([r["models"][name][key][c] for r in rs], ep, n_boot, alpha, seed)
                      for c in GUARDRAILS}
        o["n_slots"], o["n_episodes"] = len(rs), len(set(ep))
        for k in ("deviate", "deviate_gated", "support_abstain", "ood_abstain", "gate_abstain"):
            o[k] = int(sum(bool(m[k]) for m in mt))
        o["search_pick_is_panel_argmin"] = int(sum(r["models"][name]["search_pick"] == m["pick"]
                                                   for r, m in zip(rs, mt, strict=True)))
        imp = o["improvement_gated"]
        bad = [c for c, v in o["guard_gated"].items() if v["lo"] > 0]
        o["step5_criterion"] = {"improvement_gated_lo_gt_0": bool(imp["lo"] > 0), "guardrails_credibly_worse": bad,
                                "met": bool(imp["lo"] > 0 and not bad)}
        return o

    out = {}
    for name in names:
        rs = [r for r in rows if name in r["models"]]
        out[name] = {"pooled": block(rs, name)}
        if by_stratum:
            st = sorted({(r["scenario"], r["load"]) for r in rs})
            out[name]["strata"] = {f"{s}-{ld}": block([r for r in rs if (r["scenario"], r["load"]) == (s, ld)], name)
                                   for s, ld in st}
    return out


# ================================================================================================ EB re-scoring
PICKS = {"raw": ("pick_raw", "improvement_raw"), "sup": ("pick", "improvement"),
         "gated": ("pick_gated", "improvement_gated")}


def shrink_rows(rows: Sequence[Mapping], name: str, n_members: int, tau2: float | None = None,
                ddof: int = 1) -> list[dict]:
    """Offline EB shrinkage of stored (unshrunk) slot records for model ``name``: per slot, the recorded raw mean
    ``pred`` and member std ``std`` (population std over ``n_members`` members) give v = std^2 K / (K - ddof), then
    ``effect_model.eb_factors`` over the slot's candidates (the same plans EffectWM scored in one call). Returns
    copies whose ``models[name]`` has the shrunk pred / std, ``s``, ``tau2`` and recomputed metrics. The panel is
    unchanged (the arbiter's search candidates were chosen unshrunk) and the calibrated gate's verdicts are dropped
    (``allowed`` None: the gate was calibrated on unshrunk scores), so the gated pick = supported and in-support."""
    K = int(n_members)
    if K <= ddof:
        raise ValueError("n_members must exceed ddof")
    out = []
    for r in rows:
        if r.get("type", "slot") != "slot" or name not in r["models"]:
            continue
        if r.get("shrink"):
            raise ValueError("record is already shrunk")
        mr = r["models"][name]
        m, sd = np.asarray(mr["pred"], float), np.asarray(mr["std"], float)
        s, t2 = EM.eb_factors(m, sd ** 2 * K / (K - ddof), tau2)
        new = dict(mr, pred=(s * m).tolist(), std=(s * sd).tolist(), s=s.tolist(), tau2=t2, allowed=None)
        new["metrics"] = slot_metrics(r["oracle"]["J"], new["pred"], mr["unid"], mr["ood"], None)
        orc = r["oracle"]
        new["guard"] = {c: orc[c][new["metrics"]["pick"]] - orc[c][0] for c in GUARDRAILS}
        new["guard_gated"] = {c: orc[c][new["metrics"]["pick_gated"]] - orc[c][0] for c in GUARDRAILS}
        out.append(dict(r, models={**r["models"], name: new}, shrink="eb"))
    return out


def beats_zero_contrast(rows: Sequence[Mapping], name: str, pick: str = "raw", n_boot: int = 2000,
                        alpha: float = 0.05, seed: int = 0, require_split: str | None = "diag") -> dict:
    """Precondition for deploying a learned ranker: do its picks beat the zero-contrast rule (always accept-all) on
    held-out DEV slots? Paired per-slot improvement J[accept-all] - J[pick] (``pick`` in raw | sup | gated),
    episode-clustered bootstrap CI; ``beats`` iff the CI lower bound > 0 (``harms`` iff the upper bound < 0).
    ``require_split``: every slot's seed must be a DEV seed of that split (``collect.dev_stratum``; "diag" = held
    out from fitting); None skips the check. Records must carry the metrics (``oracle_panel`` / ``shrink_rows``)."""
    if pick not in PICKS:
        raise ValueError(f"pick in {tuple(PICKS)}")
    rs = [r for r in rows if r.get("type", "slot") == "slot" and name in r["models"]]
    if not rs:
        raise ValueError(f"no slot records for model {name}")
    if require_split is not None:
        def split(seed):
            try:
                return CO.dev_stratum(int(seed))["split"]
            except ValueError:
                return None

        bad = sorted({int(r["seed"]) for r in rs if split(r["seed"]) != require_split})
        if bad:
            raise ValueError(f"seeds not in DEV split {require_split!r}: {bad[:5]}")
    pk, ik = PICKS[pick]
    mt = [r["models"][name]["metrics"] for r in rs]
    ci = cluster_bootstrap([m[ik] for m in mt], [r["seed"] for r in rs], n_boot, alpha, seed)
    return {"model": name, "pick": pick, "improvement": ci, "beats": bool(ci["lo"] > 0), "harms": bool(ci["hi"] < 0),
            "deviate": int(sum(m[pk] != 0 for m in mt)), "n_slots": len(rs), "n_episodes": ci["n_clusters"]}

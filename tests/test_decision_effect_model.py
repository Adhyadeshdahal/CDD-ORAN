"""Behavioural tests of the learned policy-effect world model (cdd_oran.decision.effect_model) and the
selection-aware conformal gate (cdd_oran.decision.gate). Synthetic tables with planted effects + one short E6 run."""
from __future__ import annotations

import numpy as np
import pytest

from cdd_oran.decision import collect as CO
from cdd_oran.decision import effect_model as EM
from cdd_oran.decision import plans as P
from cdd_oran.decision.adapters.e6 import make_arbiter
from cdd_oran.decision.gate import (
    ConformalGate,
    SupportDetector,
    coverage,
    false_improvement_rate,
    gate_report,
)
from cdd_oran.decision.world_model import DecisionContext, Score
from cdd_oran.envs.e6.config import E6Config
from cdd_oran.envs.e6.env import E6Env

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

TS_REJECT = CO.encode({"mode": {"MRO": "accept", "TS": "reject", "ES": "accept", "SLICE": "accept"}, "rb": 0})
I_TS = EM.POLICY_NAMES.index("TS:reject")
DELTA = 6.0


# ------------------------------------------------------------------------------------------------ synthetic tables
def synth(n_ep, seed, E=30, R=10, spill=0.0, W=None):
    """Collector-like episodes. Context u (congestion) in [0, 1], v, g. The policy is drawn as the collector does
    but with a CONTEXT-DEPENDENT eps(u) = 0.03 + 0.9 u^3 (confounding: congested regions are perturbed more), and
    only TS acts: TS reject lowers eMBB violations by DELTA * u. Baseline 60 u^6 is outside the ridge basis."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_ep):
        u, v, g = rng.uniform(size=(E, R)), rng.normal(size=(E, R)), rng.normal(size=(E, 1))
        eps = 0.03 + 0.9 * u ** 3
        code = np.where(rng.uniform(size=(E, R)) < 1 - eps, 0, rng.integers(CO.N_CODES, size=(E, R)))
        ts = EM.policy_features(code)[..., I_TS]
        in_def = np.array([[EM.policy_class(c, ("TS",)) == EM.policy_class(0, ("TS",)) for c in r] for r in code])
        y = np.zeros((E, R, len(EM.COMPONENTS)))
        y[..., 1] = 60 * u ** 6 + 2 * v + g - DELTA * u * ts + rng.normal(0, 2, (E, R))
        if spill:
            y[..., 1] += spill * np.einsum("rs,es->er", W, ts)
        y[..., 0] = 5 * u + rng.normal(0, 1, (E, R))
        out.append({"own": np.stack([u, v], -1), "glob": g, "code": code,
                    "prop": (1 - eps) * (code == 0) + eps / CO.N_CODES, "prop_eff": (1 - eps) * in_def + eps / 8,
                    "y": y, "active": ("TS",), "H": 90})
    return out


def _model(**kw):
    kw.setdefault("selector", EM.ContextSelector("none"))
    return EM.PolicyEffectModel(90, own_names=("u", "v"), glob_names=("g",), **kw)


@pytest.fixture(scope="module")
def confounded():
    return synth(60, seed=1)


def _congested_bias(model, eps):
    d = model.stack(eps)
    M, R = d["code"].shape
    est = np.array([model.effects(d["own"][i], d["glob"][i], np.full((1, R), TS_REJECT))[0, :, 1] for i in range(M)])
    u = d["own"][..., 0]
    cong = u > 0.6
    return est[cong].mean() - (-DELTA * u[cong]).mean(), (-DELTA * u[cong]).mean()


def test_ipw_weighted_effect_recovers_planted_effect_where_naive_is_biased(confounded):
    dr = _model(n_members=3).fit(confounded)
    naive = _model(n_members=3, weighting="none").fit(confounded)
    b_dr, truth = _congested_bias(dr, confounded)
    b_naive, _ = _congested_bias(naive, confounded)
    assert truth < -4.0
    assert abs(b_dr) < 0.5                            # within ~10 % of the planted effect
    assert b_naive > 0.7 and b_naive > abs(b_dr) + 0.4  # confounding makes the naive fit see less benefit
    # inert components and inert xApps: no effect planted -> predicted contrast small
    d = dr.stack(confounded)
    es_rej = CO.encode({"mode": {"MRO": "accept", "TS": "accept", "ES": "reject", "SLICE": "accept"}, "rb": 0})
    e = dr.effects(d["own"][0], d["glob"][0], np.full((1, 10), es_rej))
    assert np.abs(e[0, :, 1]).mean() < 1.5 and np.abs(dr.effects(d["own"][0], d["glob"][0],
                                                                  np.zeros((1, 10), int))).max() == 0.0


def test_aipw_pseudo_outcomes_recover_effect(confounded):
    m = _model(n_members=2)
    a = m.aipw(confounded, TS_REJECT, n_folds=2)
    u = np.concatenate([e["own"][..., 0] for e in confounded]).ravel()
    cong = u > 0.6
    mu, se = EM.clustered_mean(a["psi"], a["ep"], cong)
    truth = (-DELTA * u[cong]).mean()
    assert se[1] > 0 and abs(mu[1] - truth) < 3 * se[1] + 0.2
    assert abs(mu[0]) < 3 * se[0] + 0.2               # LL component: no planted effect


def test_member_effects_are_paired_and_spread():
    eps = synth(12, seed=3)
    for learner in ("ridge", "hgb"):
        m = _model(n_members=4, learner=learner).fit(eps)
        codes = np.array([[TS_REJECT] * 10, [0] * 10])
        me = m.member_effects(eps[0]["own"][0], eps[0]["glob"][0], codes)
        assert me.shape == (4, 2, 10, len(EM.COMPONENTS))
        assert np.all(me[:, 1] == 0.0)                # accept-all contrast is exactly zero for every member
        assert me[:, 0, :, 1].std(0).mean() > 0       # bootstrap members disagree (spread exists)
    assert m.boot_unit == "episode"


def test_selector_neighbour_policy_captures_spillover():
    R = 6
    W = np.zeros((R, R))
    for r in range(R):
        W[r, (r - 1) % R] = W[r, (r + 1) % R] = 1.0
    eps = synth(120, seed=5, R=R, spill=4.0, W=W / 2)
    m = _model(n_members=2, selector=EM.ContextSelector("weights", weights=W)).fit(eps)
    codes = np.zeros((1, R), int)
    codes[0, 0] = TS_REJECT                            # region 0 rejects TS; its ring neighbours 1 and 5 feel it
    e = m.effects(eps[0]["own"].mean(0), eps[0]["glob"].mean(0), codes)[0, :, 1]
    assert 1.0 < e[1] < 3.0 and 1.0 < e[R - 1] < 3.0 and abs(e[3]) < 0.5   # planted spill = 4 * 1/2 = 2
    none = _model(n_members=2).fit(eps)
    assert np.all(none.effects(eps[0]["own"].mean(0), eps[0]["glob"].mean(0), codes)[0, 1:, 1] == 0.0)
    sel = EM.ContextSelector("weights", weights=W).matrix(R)
    assert np.allclose(sel.sum(1), 1.0) and np.all(np.diag(sel) == 0)


def synth_v2(n_ep, seed, E=30, R=10, fut_effect=-8.0, H=90, D=20):
    """v2-like episodes: context-free step-mixture codes (collect.step_schedule), future assignments = the region's
    own later codes, eMBB y = 5 u - DELTA u [TS reject] + fut_effect * fut[TS reject] + noise."""
    rng = np.random.default_rng(seed)
    out = []
    F = int(np.ceil(H / D)) - 1
    for i in range(n_ep):
        s = CO.step_schedule(seed * 1000 + i, E + F, list(range(R)), CO.DEFAULT_STEP_MIXTURE)
        code = s["code"][:E].astype(int)
        fc = np.stack([s["code"][j:j + E] for j in range(1, F + 1)], -1)
        fut = EM.future_features(fc, H, D)
        u, v, g = rng.uniform(size=(E, R)), rng.normal(size=(E, R)), rng.normal(size=(E, 1))
        y = np.zeros((E, R, len(EM.COMPONENTS)))
        y[..., 1] = 5 * u + 2 * v + g - DELTA * u * EM.policy_features(code)[..., I_TS] \
            + fut_effect * fut[..., I_TS] + rng.normal(0, 2, (E, R))
        out.append({"own": np.stack([u, v], -1), "glob": g, "code": code, "prop": s["prop"][:E],
                    "prop_eff": s["prop"][:E], "y": y, "active": tuple(P.XAPPS), "H": H, "version": 2, "D": D,
                    "fut": fut})
    return out


def test_future_covariates_and_continuation():
    eps = synth_v2(20, seed=7)
    on = _model(n_members=2).fit(eps)
    off = _model(n_members=2, use_future=False).fit(eps)
    ipw = _model(n_members=2, weighting="ipw").fit(eps)
    assert on.future_on and not off.future_on and on.weighting_used == "none" and ipw.weighting_used == "ipw"
    d = on.stack(eps)
    M, R = d["code"].shape
    u = d["own"][..., 0]
    truth = -DELTA * u

    def err(m):
        est = np.array([m.effects(d["own"][i], d["glob"][i], np.full((1, R), TS_REJECT))[0, :, 1] for i in range(M)])
        return np.abs(est - truth).mean()
    assert err(on) < err(off) and err(on) < 0.6
    own, glob = d["own"][0], d["glob"][0]
    codes = np.array([[TS_REJECT] * R, [0] * R])
    acc, hold = on.member_effects(own, glob, codes), on.member_effects(own, glob, codes, "hold")
    assert np.all(acc[:, 1] == 0) and np.all(hold[:, 1] == 0)          # accept-all is 0 under both continuations
    extra = EM.hold_features(TS_REJECT, 90, 20)[I_TS] * -8.0            # planted future effect of holding pi
    assert abs((hold - acc)[:, 0, :, 1].mean() - extra) < 1.5
    with pytest.raises(ValueError):
        _model(use_future=True).fit(synth(2, seed=0))


# ------------------------------------------------------------------------------------------------ gate
def _pipeline(n_ep, rng, sd, per_ep=10, n_cand=20):
    """Argmax pipeline: candidates' true improvements tau, predictions tau + episode effect + noise; the arbiter
    keeps the argmax prediction. Returns selected (pred, realized, episode) and one random candidate's pair."""
    pred, real, ep, rp, rr = [], [], [], [], []
    for e in range(n_ep):
        b = rng.normal(0, 0.5)
        for _ in range(per_ep):
            tau = rng.normal(-0.5, 1.0, n_cand)
            p = tau + b + rng.normal(0, sd, n_cand)
            j = int(np.argmax(p))
            pred.append(p[j])
            real.append(tau[j])
            ep.append(e)
            k = int(rng.integers(n_cand))
            rp.append(p[k])
            rr.append(tau[k])
    return tuple(map(np.array, (pred, real, ep, rp, rr)))


def test_selection_aware_gate_calibrates_coverage():
    rng = np.random.default_rng(0)
    pc, rc, ec, rpc, rrc = _pipeline(150, rng, 1.0)
    pt, rt, et, _, _ = _pipeline(300, rng, 1.0)
    alpha = 0.1
    gate = ConformalGate(alpha, method="pooled").calibrate(pc, rc, ec)
    rep = gate_report(gate, pt, rt, et)
    assert rep["coverage"] >= 1 - alpha - 0.03
    # calibrating single-candidate errors (not the argmax pipeline) under-covers the selected plan
    naive = ConformalGate(alpha, method="pooled").calibrate(rpc, rrc, ec)
    assert coverage(naive.lower_bound(pt), rt) < 1 - alpha - 0.05
    ep_gate = ConformalGate(alpha, method="episode_max").calibrate(pc, rc, ec)
    assert gate_report(ep_gate, pt, rt, et)["episode_all_covered"] >= 1 - alpha - 0.05
    assert ep_gate.q() > gate.q()
    fir = false_improvement_rate(gate.lower_bound(pt), rt)
    assert np.isnan(fir) or 0.0 <= fir <= 1.0


def test_stratified_gate_covers_each_stratum():
    rng = np.random.default_rng(1)
    cal = {sd: _pipeline(100, rng, sd) for sd in (0.3, 2.0)}
    pc, rc = np.concatenate([cal[sd][0] for sd in cal]), np.concatenate([cal[sd][1] for sd in cal])
    ec = np.concatenate([cal[sd][2] + 1000 * i for i, sd in enumerate(cal)])
    st = np.concatenate([[f"sd{sd}"] * len(cal[sd][0]) for sd in cal])
    gate = ConformalGate(0.1, method="pooled", min_units=50).calibrate(pc, rc, ec, stratum=st)
    for sd in cal:
        p, r = _pipeline(200, rng, sd)[:2]
        assert coverage(gate.lower_bound(p, f"sd{sd}"), r) >= 0.87
    assert gate.q("sd2.0") > gate.q("sd0.3")


def test_gate_abstains_out_of_support_and_on_small_calibration():
    det = SupportDetector().fit(np.random.default_rng(0).uniform(size=(500, 3)))
    assert not det.outside([[0.5, 0.5, 0.5]])[0] and det.outside([[0.5, 3.0, 0.5]])[0]
    assert det.outside([[np.nan, 0.5, 0.5]])[0]
    gate = ConformalGate(0.1).calibrate(np.zeros(100), np.zeros(100) - 0.1, np.arange(100))
    good = EM.EffectScore(-5.0, 0.1, (-5.0, -4.9), False)
    assert gate.confidence(good, Score(0.0))
    assert not gate.confidence(EM.EffectScore(-5.0, 0.1, (-5.0,), True), Score(0.0))     # outside support
    assert not ConformalGate(0.1).calibrate([1.0] * 5, [0.0] * 5, range(5))(good, Score(0.0))  # n too small
    agree = ConformalGate(0.1, member_agree=0.9).calibrate(np.zeros(100), np.zeros(100), np.arange(100))
    assert not agree(EM.EffectScore(-5.0, 3.0, (-12.0, 2.0), False), Score(0.0))              # members split


# ------------------------------------------------------------------------------------------------ real E6
def _cfg(seed):
    return E6Config(seed=seed, load="medium", mobility="mixed", mix="M4", warmup_s=60, scored_s=60)


def test_serve_features_match_training_features():
    """Train/serve parity: features rebuilt from the collector table + trace == features built from obs."""
    env = E6Env(_cfg(17), log=False, wg3=True, trace=True)
    col = CO.RandomizedJointPolicy(env, 0.5)
    tracker, served = EM.ReportTracker(), []
    while env.sec < env.total_s:
        obs = env.step_propose()
        tracker.observe(obs)
        n0 = len(col.rows)
        dec = col.act(obs)
        if len(col.rows) > n0:
            served.append(EM.serve_features(obs, tracker, col.site, col.regions))
        env.step_apply(dec)
        col.record(dec)
    tr = env.get_trace()
    tr.arrays.update(col.table(tr))
    tr.meta["collector"] = {"active_xapps": sorted(col.active)}
    ep = EM.episode_from_trace(tr, H=20)
    assert len(ep["t"]) == 3 and tracker.complete
    for i, t in enumerate(ep["t"]):
        k = int(np.searchsorted(tr.arrays["pol_t"], t))
        assert np.allclose(ep["own"][i], served[k][0], equal_nan=True)
        assert np.allclose(ep["glob"][i], served[k][1], equal_nan=True)
    assert ep["own"][..., EM.OWN.index("nreq_MRO"):EM.OWN.index("locked")].sum() > 0
    assert ep["y"][..., EM.COMPONENTS.index("energy_kwh")].min() > 0


@pytest.fixture(scope="module")
def e6_model():
    eps = [EM.episode_from_trace(CO.collect_episode(_cfg(s), eps=0.5), H=20) for s in (17, 18)]
    return EM.PolicyEffectModel(20, n_members=3).fit(eps)


def test_effect_wm_plugs_into_wg3_arbiter(e6_model):
    env = E6Env(_cfg(16), log=False, wg3=True)
    wm = EM.EffectWM(e6_model, env.plant.lay.cell_site)
    arb = make_arbiter(env, wm, 1.0, 1.0, D=20, H=20, n_glob=2, n_loc=1)
    s = EM.run_episode(env, arb)
    assert "svr" in s and wm.n_score > 0 and not wm.privileged
    plans, mem, ood = wm.last
    assert mem.shape == (3, len(plans)) and wm.tracker.complete
    acc = [j for j, p in enumerate(plans) if P.is_accept_all(p)]
    assert all(np.all(mem[:, j] == 0.0) and not ood[j] for j in acc)
    with pytest.raises(ValueError):                   # model horizon must match the arbiter's
        wm.score(DecisionContext({"t": 1.0, "requests": [], "new_reports": []}, wm.site, wm.regions, 90, 20.0,
                                 1.0, 1.0), [P.accept_all(wm.regions)])


def test_v2_episodes_carry_future_features_and_serve_with_declared_continuation():
    trs = [CO.collect_episode_v2(_cfg(s), label_H=40) for s in (17, 18)]
    eps = [EM.episode_from_trace(t, H=40) for t in trs]
    e = eps[0]
    assert e["version"] == 2 and e["fut"].shape == e["code"].shape + (len(EM.POLICY_NAMES),)
    fc = CO.future_assignments(trs[0], 40)["code"][: len(e["code"])]
    assert np.allclose(e["fut"], EM.future_features(fc, 40, 20))
    assert "fut" not in EM.episode_from_trace(trs[0], H=40, future=False)
    m = EM.PolicyEffectModel(40, n_members=2).fit(eps)
    assert m.future_on and m.weighting_used == "none"
    env = E6Env(_cfg(16), log=False, wg3=True)
    wms = [EM.EffectWM(m, env.plant.lay.cell_site, continuation=c) for c in ("accept_all", "hold")]
    while env.sec < 61:
        obs = env.step_propose()
        for wm in wms:
            wm.observe(obs)
        env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
    obs = env.step_propose()
    ctx = DecisionContext(obs, wms[0].site, wms[0].regions, 40, 20.0, 1.0, 1.0)
    plans = [P.accept_all(wms[0].regions), P.network(wms[0].regions, P.uniform("reject"))]
    sa, sh = (wm.score(ctx, plans) for wm in wms)
    assert sa[0].mean == 0.0 == sh[0].mean and sa[1].mean != sh[1].mean


def test_v3_fit_and_support_gate_excludes_unidentified_parts():
    cfg = [E6Config(seed=s, load="medium", mobility="mixed", mix="M4", warmup_s=60, scored_s=120) for s in (17, 18)]
    trs = [CO.collect_episode_v3(c, H=40) for c in cfg]
    eps = [EM.episode_from_trace(t, H=40) for t in trs]
    assert all(e["version"] == 3 and "fut" not in e for e in eps)
    m = EM.PolicyEffectModel(40, n_members=2).fit(eps)
    assert m.weighting_used == "none" and not m.future_on
    support = {("TS", "reject"): True, ("SLICE", "half"): True}
    env = E6Env(_cfg(16), log=False, wg3=True)
    wm = EM.EffectWM(m, env.plant.lay.cell_site, support=support)
    while env.sec < 61:
        obs = env.step_propose()
        wm.observe(obs)
        env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
    ctx = DecisionContext(env.step_propose(), wm.site, wm.regions, 40, 20.0, 1.0, 1.0)
    ts = {"mode": {"MRO": "accept", "TS": "reject", "ES": "accept", "SLICE": "half"}, "rb": 0}
    mro = {"mode": {"MRO": "half", "TS": "reject", "ES": "accept", "SLICE": "accept"}, "rb": 1}
    plans = [P.accept_all(wm.regions), P.network(wm.regions, ts), P.network(wm.regions, mro)]
    s0, s1, s2 = wm.score(ctx, plans)
    assert s0.mean == 0.0 and np.isfinite(s1.mean) and not s1.unidentified
    assert s2.mean == float("inf") and s2.ood and s2.unidentified == (("MRO", "half"), ("RB", "rb"))
    assert EM.unidentified_parts(P.accept_all(wm.regions), {}) == ()


def test_gate_default_is_episode_max():
    g = ConformalGate(0.1)
    assert g.method == "episode_max"
    g.calibrate(np.arange(60.0), np.zeros(60), np.repeat(np.arange(30), 2))
    assert g.n_units[None] == 30 and np.isfinite(g.q_pooled)                  # 30 episode units, not 60 decisions


def test_uncalibrated_gate_reproduces_accept_all_and_oracle_record_runs(e6_model):
    from cdd_oran.decision.gate import oracle_record
    runs = []
    for conf in (lambda s, s0: False, ConformalGate(0.1).calibrate([1.0] * 3, [0.0] * 3, range(3))):
        env = E6Env(_cfg(16), log=False, wg3=True)
        wm = EM.EffectWM(e6_model, env.plant.lay.cell_site)
        runs.append(EM.run_episode(env, make_arbiter(env, wm, 1.0, 1.0, D=20, H=20, n_glob=2, n_loc=1,
                                                     confidence=conf)))
    assert runs[0] == runs[1]
    env = E6Env(_cfg(16), log=False, wg3=True)
    wm = EM.EffectWM(e6_model, env.plant.lay.cell_site)
    arb = make_arbiter(env, wm, 1.0, 1.0, D=20, H=20, n_glob=2, n_loc=1)
    while env.sec < 60:
        obs = env.step_propose()
        wm.observe(obs)
        env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
    obs = env.step_propose()
    ctx = DecisionContext(obs, arb.site, arb.regions, 20, 20.0, 1.0, 1.0, env.last_change, {}, env)
    rec = oracle_record(arb, ctx)
    assert np.isfinite(rec["pred"]) and np.isfinite(rec["realized"])


# ------------------------------------------------------------------------------------------------ EB shrinkage
def test_eb_shrink_returns_exact_zero_when_member_variance_dominates():
    rng = np.random.default_rng(0)
    mem = rng.normal(0.0, 10.0, (5, 30))                         # no true contrast: all spread is member noise
    mem[:, 0] = 0.0                                              # accept-all: exact zero, no information
    out, s, tau2 = EM.eb_shrink(mem)
    assert tau2 == 0.0 and np.all(out == 0.0) and s[0] == 1.0


def test_eb_shrink_leaves_signal_dominated_contrasts_nearly_intact():
    rng = np.random.default_rng(1)
    truth = rng.normal(0.0, 50.0, 40)
    mem = truth[None, :] + rng.normal(0.0, 0.5, (5, 40))
    out, s, tau2 = EM.eb_shrink(mem)
    assert tau2 > 1000 and s.min() > 0.999
    np.testing.assert_allclose(out.mean(0), mem.mean(0), rtol=2e-3)


def test_eb_shrink_preserves_ranking_under_equal_member_variance_and_member_signs():
    rng = np.random.default_rng(2)
    base = np.linspace(-30, 30, 12)
    noise = rng.normal(0, 1.0, (5, 1)) * np.ones((1, 12))       # identical member spread for every candidate
    mem = base[None, :] + noise * 3.0
    out, s, tau2 = EM.eb_shrink(mem)
    assert 0 < s.min() < 1 and np.allclose(s, s[0])
    assert (np.argsort(out.mean(0)) == np.argsort(mem.mean(0))).all()
    assert (np.sign(out) == np.sign(mem)).all()                  # member-agreement tests are unaffected
    # a noisier candidate is shrunk harder; a fixed prior variance is used as given
    mem2 = mem.copy()
    mem2[:, 3] += rng.normal(0, 20.0, 5)
    _, s2, _ = EM.eb_shrink(mem2, tau2=100.0)
    assert s2[3] < np.delete(s2, 3).min()
    with pytest.raises(ValueError):
        EM.eb_shrink(mem, tau2=-1.0)


def test_effect_wm_shrink_option_scores_shrunk_and_accept_all_stays_zero(e6_model):
    env = E6Env(_cfg(16), log=False, wg3=True)
    site = env.plant.lay.cell_site
    raw, eb = EM.EffectWM(e6_model, site), EM.EffectWM(e6_model, site, shrink="eb")
    while env.sec < 70:
        obs = env.step_propose()
        raw.observe(obs)
        eb.observe(obs)
        env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": []})
    ctx = DecisionContext(obs, raw.site, raw.regions, 20, 20.0, 1.0, 1.0)
    plans = [P.accept_all(raw.regions), P.network(raw.regions, P.uniform("reject"))] +         [P.network(raw.regions, {"mode": {y: (m if y == x else "accept") for y in P.XAPPS}, "rb": 0})
         for x in P.XAPPS for m in ("reject", "lock")]
    r, e = raw.score(ctx, plans), eb.score(ctx, plans)
    s, tau2 = eb.last_shrink
    assert raw.last_shrink is None and tau2 >= 0 and np.all((0 <= s) & (s <= 1))
    assert e[0].mean == 0.0 == r[0].mean and e[0].std == 0.0
    np.testing.assert_allclose([x.mean for x in e], s * np.array([x.mean for x in r]), atol=1e-9)
    np.testing.assert_allclose([x.std for x in e], s * np.array([x.std for x in r]), atol=1e-9)
    with pytest.raises(ValueError):
        EM.EffectWM(e6_model, site, shrink="js")

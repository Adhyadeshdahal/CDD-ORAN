"""Behavioural tests of the E6 decision trace (cdd_oran.decision.trace) and the randomized joint-policy collector."""
from __future__ import annotations

import numpy as np
import pytest

from cdd_oran.decision import collect as CO
from cdd_oran.decision.trace import OUT_CODES, RB_CODES, WR_CODES, Trace, region_labels
from cdd_oran.envs.e6.config import E6Config
from cdd_oran.envs.e6.env import E6Env

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _cfg(seed=16, load="medium"):
    return E6Config(seed=seed, load=load, mobility="mixed", mix="M4", warmup_s=30, scored_s=120)


@pytest.fixture(scope="module")
def noarb():
    env = E6Env(_cfg(), log=True, trace=True)
    s = env.run()
    return env, s, env.get_trace()


@pytest.fixture(scope="module")
def collected():
    # eps = 1 and a tight per-epoch churn allowance exercise reject / half / lock / rollback / churn NACKs
    return CO.collect_episode(_cfg(), eps=1.0, epoch_churn_cap=3)


def _collect_score(trace_on):
    env = E6Env(_cfg(), log=False, wg3=True, trace=trace_on)
    col = CO.RandomizedJointPolicy(env, 1.0, epoch_churn_cap=3)
    while env.sec < env.total_s:
        dec = col.act(env.step_propose())
        env.step_apply(dec)
        col.record(dec)
    return env.score()


def test_trace_off_changes_nothing(noarb):
    env, s_on, _ = noarb
    off = E6Env(_cfg(), log=True)
    assert off._tr is None
    assert off.run() == s_on
    assert _collect_score(False) == _collect_score(True)
    with pytest.raises(RuntimeError):
        off.get_trace()


def test_copy_is_untraced(noarb):
    env = E6Env(_cfg(), log=False, trace=True)
    env.step_propose()
    c = env.copy()
    assert c._tr is None and env._tr is not None
    env.step_apply({"decisions": ["accept"] * len(env._pending[2]), "writes": []})
    c.step_apply({"decisions": ["accept"] * len(c._pending[2]), "writes": []})
    assert len(env.get_trace()["t"]) == 1


def test_one_row_per_second_and_config_rebuild(noarb):
    env, s, tr = noarb
    a = tr.arrays
    assert np.array_equal(a["t"], np.arange(1, env.total_s + 1))
    assert a["scored"].sum() == env.cfg.scored_s
    for k, v in tr.labels().items():
        assert len(v) == len(a["t"]), k
    for t in (5, 31, 60, 61, 100, env.total_s - 1):    # pre-action config at t+1 == post-apply config logged at t
        logged = np.array(list(env.log[t - 1]["config"].values()))
        assert np.allclose(tr.config_at(t + 1), logged)


def test_labels_sum_to_plant_counters(noarb):
    env, s, tr = noarb
    a, S = tr.arrays, env.plant.sla
    sc = a["scored"]
    assert a["lab_viol"][sc].sum() == S["viol_ue_s"]
    assert a["lab_viol"][sc][..., 0].sum() == S["ll_viol"]
    assert a["lab_viol"][sc][..., 1].sum() == S["embb_viol"]
    assert a["lab_outage"][sc].sum() == S["outage_viol"]
    assert a["lab_ue"][sc].sum() == S["ue_s"]
    assert a["lab_severe"].sum() == S["severe"]
    assert a["lab_rlf"].sum() == S["rlf"]                   # plant counts RLF in every second
    assert np.isclose(a["lab_energy_j"][sc].astype(float).sum(), S["energy_j"], rtol=1e-5)
    y = region_labels(tr, env.plant.lay.cell_site, env.cfg.warmup_s, env.total_s)
    assert len(y["region_ids"]) == 10
    assert y["viol"].sum() == S["viol_ue_s"] and np.isclose(y["energy_j"].sum(), S["energy_j"], rtol=1e-5)


def test_ack_nack_and_churn_match_stats(collected):
    a, m = collected.arrays, collected.meta
    st = dict(zip(m["stats"], a["stats"][-1], strict=True))
    out, dec = a["rq_out"], a["rq_dec"]
    OC = {c: i for i, c in enumerate(OUT_CODES)}
    ack = np.isin(out, [OC["ok"], OC["noop"]])
    assert st["req"] == len(out)
    assert st["acc"] == (ack & (dec == m["dec_codes"].index("accept"))).sum()
    assert st["mod"] == (ack & (dec == m["dec_codes"].index("modify"))).sum()
    assert st["rej"] == np.isin(out, [OC[c] for c in ("reject", "locked", "lock_set", "expired", "churn")]).sum()
    assert st["locks"] == (out == OC["lock_set"]).sum() == len(a["lk_t"]) > 0
    assert st["lock_blocked"] == (out == OC["locked"]).sum()
    assert st["rollbacks"] == (a["rb_out"] == RB_CODES.index("applied")).sum() > 0
    churn = (out == OC["churn"]).sum() + (a["rb_out"] == RB_CODES.index("churn")).sum() + \
        (a["wr_out"] == WR_CODES.index("churn")).sum()
    assert st["churn_blocked"] == churn > 0
    assert st["changes"] == (out == OC["ok"]).sum() + (a["rb_out"] == RB_CODES.index("applied")).sum()
    assert a["changes"][-1] == st["changes"] and np.all(np.diff(a["changes"]) >= 0)
    capped = a["churn_cap"] >= 0                            # every applied change respects the cap in force
    assert np.all(a["changes"][capped] <= a["churn_cap"][capped])
    assert np.all(np.diff(a["stats"], axis=0) >= 0)


def test_policy_table_consistent(collected):
    a = collected.arrays
    E, R = a["pol_code"].shape
    assert R == 10 and np.array_equal(a["pol_t"], 30 + 20 * np.arange(E))
    assert np.allclose(a["pol_prop"], np.vectorize(lambda c: CO.propensity(c, 1.0))(a["pol_code"]))
    in_epochs = (a["rq_t"] >= a["pol_t"][0]).sum()
    assert a["pol_n_req"].sum() == in_epochs == a["pol_out"].sum()
    assert a["pol_locks"].sum() == len(a["lk_t"])
    for e in range(E):                                      # decoded modes reproduce the recorded codes
        for r in range(R):
            rp = {"mode": {x: CO.P.MODES[i] for x, i in zip(CO.P.XAPPS, a["pol_modes"][e, r], strict=True)},
                  "rb": int(a["pol_rb"][e, r])}
            assert CO.encode(rp) == a["pol_code"][e, r]


def test_propensities_exact_and_empirical():
    for c in range(CO.N_CODES):
        assert CO.encode(CO.decode(c)) == c
    for eps in (0.0, 0.3, 1.0):
        assert np.isclose(sum(CO.propensity(c, eps) for c in range(CO.N_CODES)), 1.0)
        act = {"MRO", "TS", "ES"}                           # M2 mix: SLICE modes inert
        classes = {}
        for c in range(CO.N_CODES):
            rp = CO.decode(c)
            classes.setdefault((rp["rb"], *(rp["mode"][x] for x in CO.P.XAPPS if x in act)), c)
        assert np.isclose(sum(CO.propensity_eff(c, eps, act) for c in classes.values()), 1.0)
    eps, n = 0.3, 40000
    draws = [CO.draw(5, e, g, eps) for e in range(n // 10) for g in range(10)]
    codes = np.array([c for c, _ in draws])
    default = np.array([b for _, b in draws])
    assert abs(default.mean() - (1 - eps)) < 4 * np.sqrt(eps * (1 - eps) / n)
    p0 = CO.propensity(0, eps)
    assert abs((codes == 0).mean() - p0) < 4 * np.sqrt(p0 * (1 - p0) / n)
    nd = codes[~default]                                    # uniform branch covers the 512 codes evenly
    assert len(np.unique(nd)) == CO.N_CODES and np.bincount(nd, minlength=CO.N_CODES).max() < 60


def test_same_seed_same_policies(collected, tmp_path):
    again = CO.collect_episode(_cfg(), eps=1.0, epoch_churn_cap=3)
    assert np.array_equal(again["pol_code"], collected["pol_code"])
    assert np.array_equal(again["rq_out"], collected["rq_out"])
    other = CO.collect_episode(_cfg(seed=17), eps=1.0, epoch_churn_cap=3)
    assert not np.array_equal(other["pol_code"], collected["pol_code"])
    p = tmp_path / "ep.npz"
    collected.to_npz(p)
    back = Trace.from_npz(p)
    assert back.meta["schema"] == "e6-trace/1" and back.meta["collector"]["eps"] == 1.0
    assert set(back.arrays) == set(collected.arrays)
    assert np.array_equal(back["lab_pol_viol"], collected["lab_pol_viol"])


# ------------------------------------------------------------------------------------------------ v2 step collector
def _cfg2(seed=16, load="medium"):
    return E6Config(seed=seed, load=load, mobility="mixed", mix="M4", warmup_s=30, scored_s=180)


@pytest.fixture(scope="module")
def step_v2():
    return CO.collect_episode_v2(_cfg2())


def test_step_mixture_propensities_exact():
    mix = CO.DEFAULT_STEP_MIXTURE
    pv = mix.code_probs()
    assert np.isclose(pv.sum(), 1.0) and pv[0] > mix.p_accept
    mm = mix.mode_marginal()
    assert np.isclose(sum(mm.values()), 1.0)
    for x in CO.P.XAPPS:                                      # per-xApp marginal = marginalised code distribution
        for m in CO.P.MODES:
            pm = sum(pv[c] for c in range(CO.N_CODES) if CO.decode(c)["mode"][x] == m)
            assert np.isclose(pm, mm[m]), (x, m)
    with pytest.raises(ValueError):
        CO.StepMixture(0.5, 0.5, 0.1, 0.0)
    n = 20000                                                 # empirical draw frequencies match the declaration
    d = np.array([CO.draw_step(3, e, g, mix) for e in range(n // 10) for g in range(10)])
    freq = np.bincount(d[:, 1], minlength=4) / n
    for b, p in enumerate((mix.p_accept, mix.p_frac, mix.p_lock, mix.p_rb)):
        assert abs(freq[b] - p) < 4 * np.sqrt(p * (1 - p) / n)
    emp = np.bincount(d[:, 0], minlength=CO.N_CODES) / n
    assert np.all(emp[pv == 0] == 0)
    assert abs(emp[0] - pv[0]) < 4 * np.sqrt(pv[0] * (1 - pv[0]) / n)
    assert np.array_equal(d[:50], [CO.draw_step(3, e, g, mix) for e in range(5) for g in range(10)])


def test_step_schedule_quiet_epochs():
    regs = list(range(10))
    mix = CO.StepMixture(quiet_epochs=4)
    s = CO.step_schedule(9, 60, regs, mix)
    pv = mix.code_probs()
    q = CO.STEP_BRANCHES.index("quiet")
    for j in range(len(regs)):
        c, b, p = s["code"][:, j], s["branch"][:, j], s["prop"][:, j]
        e = 0
        while e < 60:
            assert b[e] != q and np.isclose(p[e], pv[c[e]])
            if c[e] != 0:
                k = slice(e + 1, min(e + 5, 60))
                assert np.all(c[k] == 0) and np.all(b[k] == q) and np.all(p[k] == 1.0)
                e += 5
            else:
                e += 1
    off = CO.step_schedule(9, 60, regs, CO.DEFAULT_STEP_MIXTURE)   # quiet off = the raw key-only draws
    raw = np.array([[CO.draw_step(9, e, g, CO.DEFAULT_STEP_MIXTURE)[0] for g in regs] for e in range(60)])
    assert np.array_equal(off["code"], raw) and not np.any(off["branch"] == q)


def test_v2_draws_independent_of_plant_and_reproducible(step_v2, noarb):
    a = step_v2.arrays
    E = len(a["pol_t"])
    s = CO.step_schedule(16, E, [int(g) for g in a["pol_region_ids"]], CO.DEFAULT_STEP_MIXTURE)
    assert np.array_equal(a["pol_code"], s["code"]) and np.allclose(a["pol_prop"], s["prop"])
    assert (a["pol_code"] != 0).any() and np.array_equal(a["pol_t"], 30 + 20 * np.arange(E))
    other = CO.collect_episode_v2(_cfg2(load="high"))         # different plant tape, same seed -> same assignments
    assert np.array_equal(other["pol_code"], a["pol_code"])
    assert not np.array_equal(other["lab_viol"], a["lab_viol"])
    ref = CO.collect_episode_v2(_cfg(), mixture=CO.ACCEPT_ALL_MIXTURE)   # paired accept-all == no arbiter
    assert np.all(ref["pol_code"] == 0) and np.all(ref["pol_prop"] == 1.0)
    s_ref, s_no = ref.meta["collector"]["score"], noarb[1]
    assert all(np.isclose(s_ref[k], s_no[k]) for k in ("svr", "energy_kwh", "changes", "rlf_per_ue_h", "req"))


def test_v2_realised_deltas_consistent_with_acks(step_v2):
    a, m = step_v2.arrays, step_v2.meta
    OC = {c: i for i, c in enumerate(OUT_CODES)}
    ok = a["rq_out"] == OC["ok"]
    assert np.all(a["rq_delta_app"][~ok] == 0)
    assert np.all(np.abs(a["rq_delta_app"][ok]) > 0)           # an ACKed change moved the knob
    assert np.allclose(a["rq_delta_app"][ok], a["rq_applied"][ok] - a["rq_cur"][ok])
    for i in np.nonzero(ok)[0][:200]:                          # the applied value is in force the next second
        if a["rq_t"][i] < a["t"][-1]:
            assert np.isclose(step_v2.config_at(a["rq_t"][i] + 1)[a["rq_knob"][i]], a["rq_applied"][i])
    inep = a["rq_epoch"] >= 0
    assert a["pol_x_n_changed"].sum() == (ok & inep).sum()
    assert np.array_equal(a["pol_x_n_ack"] + a["pol_x_n_nack"] + a["pol_x_n_rej"], a["pol_x_n_req"])
    fr = a["rq_frac"]
    assert np.all(np.isnan(fr[~inep])) and not np.isnan(fr[inep]).any()
    dc = a["rq_dec"]                                           # decisions follow the assigned fraction
    assert np.all(dc[inep & (fr == 1.0)] == m["dec_codes"].index("accept"))
    half = inep & (fr == 0.5)                                  # halve the slew: modify, or alternate accept/reject
    assert np.all(np.isin(dc[half], [m["dec_codes"].index(c) for c in ("modify", "accept", "reject")]))
    assert (dc[half] == m["dec_codes"].index("accept")).any() and (dc[half] == m["dec_codes"].index("reject")).any()
    assert np.all(np.isin(dc[inep & (fr == 0.0)], [m["dec_codes"].index(c) for c in ("reject", "lock")]))
    assert not ok[inep & (fr == 0.0)].any()                    # fraction 0 never changes the knob
    rf = a["pol_x_real_frac"][np.isfinite(a["pol_x_real_frac"])]
    assert np.all((rf >= -1e-9) & (rf <= 1 + 1e-9))


def test_v2_eligibility_and_neighbours(step_v2):
    a, meta = step_v2.arrays, step_v2.meta
    site = np.asarray(meta["cell_region"])
    regs = [int(g) for g in a["pol_region_ids"]]
    E, R, X = a["pol_x_elig"].shape
    n = np.zeros((E, R, X), int)
    for t, k, x in zip(a["rq_t"], a["rq_knob"], a["rq_xapp"], strict=True):
        e = int(np.searchsorted(a["pol_t"], t, side="right")) - 1
        if e >= 0 and t < a["pol_t"][e] + a["pol_len"][e]:
            n[e, regs.index(int(site[meta["knobs"][k][1]])), CO.P.XAPPS.index(meta["xapps"][x])] += 1
    assert np.array_equal(n, a["pol_x_n_req"]) and np.array_equal(n > 0, a["pol_x_elig"])
    assert a["pol_x_elig"].any() and not a["pol_x_elig"].all()
    assert np.all(a["pol_x_req_dabs"][~a["pol_x_elig"]] == 0)
    A = a["pol_region_adj"]
    assert np.array_equal(A, A.T) and not np.diag(A).any() and A.sum(1).min() > 0
    for e in range(E):
        for r in range(R):
            nb = np.nonzero(A[r])[0]
            assert a["pol_nbr_n_dev"][e, r] == (a["pol_code"][e, nb] != 0).sum()
            assert np.allclose(a["pol_nbr_frac"][e, r], a["pol_frac"][e, nb].mean(0))


def test_v2_future_assignments_align(step_v2):
    a = step_v2.arrays
    E, R, F = a["pol_fut_code"].shape
    assert F == 4 and int(a["pol_fut_H"]) == 90                # ceil(90 / 20) - 1
    for e in range(E):
        for j in range(1, F + 1):
            if e + j < E:
                assert np.array_equal(a["pol_fut_code"][e, :, j - 1], a["pol_code"][e + j])
                assert a["pol_fut_t"][e, j - 1] == a["pol_t"][e] + 20 * j < a["pol_t"][e] + 90
                assert np.allclose(a["pol_fut_frac"][e, :, j - 1], a["pol_frac"][e + j])
                assert np.allclose(a["pol_fut_prop"][e, :, j - 1], a["pol_prop"][e + j])
            else:
                assert np.all(a["pol_fut_code"][e, :, j - 1] == -1) and a["pol_fut_t"][e, j - 1] == -1
    f60 = CO.future_assignments(step_v2, 60)
    assert f60["F"] == 2 and np.array_equal(f60["code"], a["pol_fut_code"][..., :2])


def test_v2_propensity_columns_and_effect_model_input(step_v2, tmp_path):
    a = step_v2.arrays
    pv, mm = CO.DEFAULT_STEP_MIXTURE.code_probs(), CO.DEFAULT_STEP_MIXTURE.mode_marginal()
    assert np.allclose(a["pol_prop"], pv[a["pol_code"]])
    assert np.allclose(a["pol_prop_eff"], a["pol_prop"])       # M4: all four xApps active, no inert modes
    assert np.allclose(a["pol_prop_x"], np.array([mm[m] for m in CO.P.MODES])[a["pol_modes"]])
    assert np.allclose(a["pol_frac"], np.array([1.0, 0.0, 0.5, 0.0])[a["pol_modes"]])
    from cdd_oran.decision.effect_model import episode_from_trace
    p = tmp_path / "v2.npz"
    step_v2.to_npz(p)
    back = Trace.from_npz(p)
    assert back.meta["collector"]["mixture"] == CO.DEFAULT_STEP_MIXTURE.as_dict()
    ep = episode_from_trace(back, 90)
    assert ep["code"].shape[1] == 10 and len(ep["code"]) == (a["pol_t"] + 90 <= a["t"][-1]).sum() > 0
    assert np.allclose(ep["prop"], a["pol_prop"][: len(ep["code"])])

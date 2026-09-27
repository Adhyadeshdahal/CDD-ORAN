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

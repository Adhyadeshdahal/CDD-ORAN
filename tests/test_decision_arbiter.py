"""Behavioural tests for cdd_oran.decision: plan -> WG3 decisions, arbiter search, hooks, churn cap."""
from __future__ import annotations

import numpy as np
import pytest

from cdd_oran.decision import plans as P
from cdd_oran.decision.adapters.e6 import make_arbiter
from cdd_oran.decision.arbiter import WG3Arbiter, run_episode
from cdd_oran.decision.world_model import DecisionContext, Score, TrueSimWM
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")
SITE = np.array([0, 0, 1, 1])


def _cfg():
    return C.E6Config(seed=11, load="medium", mobility="mixed", mix="M4", warmup_s=10, scored_s=30)


def _env(**kw):
    return E6Env(_cfg(), log=False, wg3=True, **kw)


class FakeWM:
    """Non-privileged world model scoring each plan by ``f(plan)``; records every scored plan."""
    privileged = False

    def __init__(self, f):
        self.f, self.seen = f, []

    def score(self, ctx, plans):
        self.seen += [(p, self.f(p)) for p in plans]
        return [Score(self.f(p)) for p in plans]


def _n_veto(plan):                        # prefers plans that veto / modify / roll back a lot
    return -sum(m != "accept" for rp in plan.values() for m in rp["mode"].values()) - sum(rp["rb"] for rp in plan.values())


def _n_half(plan):                       # prefers modify + rollback everywhere (churn-heavy)
    return -sum(m == "half" for rp in plan.values() for m in rp["mode"].values()) - sum(rp["rb"] for rp in plan.values())


def _req(xapp, knob, cur=1.0, prop=3.0):
    return {"xapp": xapp, "knob": knob, "cur": cur, "prop": prop}


# ---------------------------------------------------------------------------------------- plan -> decisions
def test_plan_modes_map_to_wg3_decisions():
    plan = {0: {"mode": {"MRO": "accept", "TS": "reject", "ES": "half", "SLICE": "lock"}, "rb": 0},
            1: P.uniform("reject")}
    obs = {"t": 100.0, "requests": [_req("MRO", ("hys", 0)), _req("TS", ("cio", 1, 2)), _req("ES", ("sleep", 0)),
                                    _req("SLICE", ("ll_ratio", 1)), _req("MRO", ("cio", 2, 0))]}
    dec = P.decide(plan, obs, SITE, 20, {}, {}, True)
    assert dec["decisions"] == ["accept", "reject", ("modify", 2.0), ("lock", 20.0), "reject"]
    assert dec["writes"] == [] and dec["rollback"] == []


def test_rollback_only_first_second_in_window_flagged_region_and_not_our_own():
    plan = {0: P.uniform("accept", rb=1), 1: P.uniform("accept", rb=0)}
    obs = {"t": 100.0, "requests": []}
    last = {("hys", 0): 90.0, ("ttt", 1): 30.0, ("hys", 2): 95.0, ("cio", 1, 3): 80.0}
    assert P.decide(plan, obs, SITE, 20, last, {}, True)["rollback"] == [("hys", 0), ("cio", 1, 3)]
    assert P.decide(plan, obs, SITE, 20, last, {}, False)["rollback"] == []          # held plan: no rollback
    assert P.decide(plan, obs, SITE, 20, last, {("hys", 0): 90.0}, True)["rollback"] == [("cio", 1, 3)]
    assert P.decide(plan, obs, SITE, 20, last, {("hys", 0): 50.0}, True)["rollback"] == [("hys", 0), ("cio", 1, 3)]


def test_restrict_forces_accept_outside_allowed_pairs():
    plan = {0: P.uniform("lock", rb=1), 1: P.uniform("half", rb=1)}
    out = P.restrict(plan, {("TS", 0)})
    assert out[0]["mode"] == {"MRO": "accept", "TS": "lock", "ES": "accept", "SLICE": "accept"} and out[0]["rb"] == 1
    assert P.is_accept_all({1: out[1]})


# ---------------------------------------------------------------------------------------- arbiter search
def test_fixed_ranking_world_model_gets_its_top_plan():
    rank = {}                             # arbitrary but fixed score per distinct plan

    def f(p):
        return rank.setdefault(repr(sorted(p.items())), float(np.random.default_rng(len(rank)).uniform()))

    wm = FakeWM(f)
    arb = WG3Arbiter(wm, SITE, 0.0, 0.0, n_glob=4, n_loc=3, seed=3)
    best = arb.choose(DecisionContext({"t": 50.0, "requests": []}, SITE, arb.regions, 10, 20.0, 0.0, 0.0))
    assert f(best) == min(s for _, s in wm.seen)
    assert len(wm.seen) == 3 + 4 + 3 * len(arb.regions)
    # a model that ranks freeze first makes the arbiter freeze (every random alternative is worse)
    frz = P.network(arb.regions, P.uniform("reject"))
    arb = WG3Arbiter(FakeWM(lambda p: 0.0 if p == frz else 1.0), SITE, 0.0, 0.0, seed=3)
    assert arb.choose(DecisionContext({"t": 50.0, "requests": []}, SITE, arb.regions, 10, 20.0, 0.0, 0.0)) == frz


def test_confidence_always_false_is_accept_all():
    cap = E6Env(_cfg(), log=False).run()["changes"] // 2
    ref = _env(churn_cap=cap).run()
    env = _env(churn_cap=cap)
    arb = make_arbiter(env, FakeWM(_n_veto), 0.0, 0.0, D=10, confidence=lambda s, s0: False)
    assert run_episode(env, arb) == ref
    # without the gate the same model deviates (sanity: the hook is what forced accept-all)
    env = _env(churn_cap=cap)
    assert run_episode(env, make_arbiter(env, FakeWM(_n_veto), 0.0, 0.0, D=10)) != ref


def test_prune_forces_accept_for_non_pruned_pairs():
    env = _env()
    site = np.asarray(env.plant.lay.cell_site)
    allowed = {(x, g) for x in ("SLICE", "TS") for g in range(0, 10, 2)}
    wm = FakeWM(_n_veto)
    arb = make_arbiter(env, wm, 0.0, 0.0, D=10, prune=lambda obs: allowed)
    n_touch = 0
    while env.sec < env.total_s:
        obs = env.step_propose()
        dec = arb.act(obs, env.last_change, env)
        for r, d in zip(obs["requests"], dec["decisions"], strict=True):
            if (r["xapp"], int(site[r["knob"][1]])) in allowed:
                n_touch += d != "accept"
            else:
                assert d == "accept"
        assert all(int(site[k[1]]) % 2 == 0 for k in dec["rollback"])
        env.step_apply(dec)
        arb.record(dec, env.last_change)
    assert wm.seen and all((x, g) in allowed or m == "accept"
                           for p, _ in wm.seen for g, rp in p.items() for x, m in rp["mode"].items())
    assert n_touch > 0


@pytest.mark.parametrize("wm", ["true", "fake"])
def test_arbiter_never_exceeds_churn_cap(wm):
    cap = E6Env(_cfg(), log=False).run()["changes"] // 3
    env = _env(churn_cap=cap)
    m = TrueSimWM() if wm == "true" else FakeWM(lambda p: 0.0)
    s = run_episode(env, make_arbiter(env, m, 1e4, 5.0, D=10, H=6, n_glob=2, n_loc=1))
    assert s["changes"] <= cap and s["churn_blocked"] > 0
    if wm == "true":
        assert m.n_roll > 0


def test_true_sim_wm_requires_env_handle():
    ctx = DecisionContext({"t": 1.0, "requests": []}, SITE, [0, 1], 5, 20.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        TrueSimWM().score(ctx, [P.accept_all([0, 1])])

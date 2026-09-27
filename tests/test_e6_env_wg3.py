"""Behavioural tests for the E6 two-phase step, WG3 actions (lock / rollback), churn cap and the wg3 writes guard."""
from __future__ import annotations

import numpy as np
import pytest

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.baselines import freeze, priority
from cdd_oran.envs.e6.env import E6Env
from cdd_oran.envs.e6.ric import knob_get


def _env(**kw):
    return E6Env(C.E6Config(seed=11, load="medium", mobility="mixed", mix="M4", warmup_s=10, scored_s=20),
                 log=False, **kw)


def _accept_all(obs):
    return {"decisions": ["accept"] * len(obs["requests"]), "writes": []}


def _advance_to_requests(env, arbiter=_accept_all, max_s=60):
    """step until a second has pending requests; returns that second's obs (still pending)."""
    for _ in range(max_s):
        obs = env.step_propose()
        if obs["requests"]:
            return obs
        env.step_apply(arbiter(obs))
    raise AssertionError("no requests proposed")


def _same_sla(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


def test_copy_between_phases_replays_pending_requests():
    env = _env()
    for _ in range(5):
        env.step(priority)
    obs = _advance_to_requests(env, priority)
    n = len(obs["requests"])
    dec = {"decisions": [("lock", 7.0) if i == 0 else "accept" if i % 2 else "defer" for i in range(n)],
           "rollback": [obs["requests"][0]["knob"]]}
    c = env.copy()
    env.step_apply(dec)
    c.step_apply(dec)
    for _ in range(15):
        env.step(priority)
        c.step(priority)
    _same_sla(env.plant.sla, c.plant.sla)
    assert env.stats == c.stats and env.config() == c.config()


def test_lock_blocks_knob_for_duration_then_releases():
    env = _env()
    obs = _advance_to_requests(env)
    k, secs = obs["requests"][0]["knob"], 25.0
    t_lock, v0 = obs["t"], knob_get(env.plant, k)
    env.step_apply({"decisions": [("lock", secs)] + ["accept"] * (len(obs["requests"]) - 1)})
    assert env.stats["locks"] == 1
    seen_locked, released = 0, False
    for _ in range(int(secs) + 20):
        obs = env.step_propose()
        on_k = sum(r["knob"] == k for r in obs["requests"])
        before = env.stats["lock_blocked"]
        env.step_apply(_accept_all(obs))
        if obs["t"] < t_lock + secs:
            assert k in obs["locked"]
            assert knob_get(env.plant, k) == v0                  # every request on k was force-rejected
            assert env.stats["lock_blocked"] - before == on_k
            seen_locked += on_k
        else:
            assert k not in obs["locked"] and env.stats["lock_blocked"] == before
            released = True
    assert seen_locked > 0 and released


def test_rollback_restores_pre_change_value():
    env = _env()
    for _ in range(60):                                         # find a second whose accept changes a knob
        obs = env.step_propose()
        before = env.config()
        env.step_apply(_accept_all(obs))
        changed = [k for k, v in env.config().items() if v != before[k]]
        if changed:
            break
    k = changed[0]
    v_prev, v_new = before[k], knob_get(env.plant, k)
    never = next(kk for kk in env.knobs if kk not in env.prev_val)
    for _ in range(130):                                        # wait out the min interval (<= 120 s dwell) frozen
        env.step(freeze)
    assert knob_get(env.plant, k) == v_new
    obs = env.step_propose()
    cfg0 = env.config()
    env.step_apply({"decisions": ["reject"] * len(obs["requests"]), "rollback": [k, never]})
    assert knob_get(env.plant, k) == v_prev
    assert env.config()[never] == cfg0[never] and env.stats["rollbacks"] == 1


def test_churn_cap_blocks_changes_beyond_cap():
    free = _env()
    free.run()
    assert free.stats["changes"] > 5 and free.stats["churn_blocked"] == 0
    env = _env(churn_cap=5)
    s = env.run()
    assert s["changes"] == 5 and s["churn_blocked"] > 0
    assert s["rej"] >= s["churn_blocked"]                       # capped accepts are NACKed as rejections


def test_wg3_rejects_free_writes_but_allows_actions():
    env = _env(wg3=True)
    obs = env.step_propose()
    k = env.knobs[0]
    with pytest.raises(ValueError):
        env.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [(k, 3.0)]})
    env2 = _env(wg3=True)
    obs = env2.step_propose()
    env2.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [], "rollback": [k]})
    loose = _env()
    obs = loose.step_propose()
    loose.step_apply({"decisions": ["accept"] * len(obs["requests"]), "writes": [(k, 3.0)]})
    assert loose.stats["writes"] == 1


def test_phase_order_enforced():
    env = _env()
    with pytest.raises(RuntimeError):
        env.step_apply({"decisions": []})
    env.step_propose()
    with pytest.raises(RuntimeError):
        env.step_propose()

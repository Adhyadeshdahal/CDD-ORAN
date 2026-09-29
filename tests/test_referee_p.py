"""cdd_oran.decision.referee_p: knapsack budget, accept-all = zero change, honest split, ctx-only features."""
from __future__ import annotations

import numpy as np
import pytest

from cdd_oran.decision import referee_p as RP


def eff(fam, leaf, pv, e, v=0.0, rlf=0.0, lb=1.0, n=40, n_eps=10):
    d = {"pv": pv, "e": e, "v": v, "rlf": rlf}
    return RP.Effect(fam, leaf, n, n_eps, dict(d), {k: (x, x) for k, x in d.items()}, dict(d), lb,
                     supported=n >= RP.MIN_LEAF_N and n_eps >= RP.MIN_LEAF_EPS)


def test_saving_direction():
    assert RP.is_saving("carrier", -1) and not RP.is_saving("carrier", 1)
    assert RP.is_saving("sleep", 1) and not RP.is_saving("sleep", -1)
    assert RP.is_saving("ptx", -3) and not RP.is_saving("ptx", 3)
    assert not RP.is_saving("prot_min", -0.05) and not RP.is_saving("prot_min", 0.05)


@pytest.mark.parametrize("budget", [0.0, 5.0, 12.0, 30.0, np.inf])
def test_knapsack_respects_budget(budget):
    rng = np.random.default_rng(0)
    effects = [eff("carrier", f"L{i}", pv=-float(rng.uniform(1, 5)), e=float(rng.uniform(1, 8))) for i in range(6)]
    nbar = {(e.family, e.leaf): float(rng.uniform(0.5, 2)) for e in effects}
    ks = RP.knapsack(effects, nbar, budget, np.inf, np.inf)
    assert ks["e"] <= budget + 1e-9
    # optimal: no feasible subset beats it (brute force check)
    import itertools
    best = 0.0
    for r in range(len(effects) + 1):
        for c in itertools.combinations(effects, r):
            if sum(nbar[(x.family, x.leaf)] * x.shrunk["e"] for x in c) <= budget + 1e-9:
                best = max(best, -sum(nbar[(x.family, x.leaf)] * x.shrunk["pv"] for x in c))
    assert ks["obj"] == pytest.approx(best)


def test_knapsack_v_rlf_caps_and_lb_rule():
    effects = [eff("carrier", "all", pv=-5, e=1, v=10), eff("ptx", "all", pv=-3, e=1, rlf=0.5),
               eff("sleep", "all", pv=-50, e=-1, lb=-0.1)]                  # sleep: LB <= 0 -> never deferred
    nbar = {("carrier", "all"): 1.0, ("ptx", "all"): 1.0, ("sleep", "all"): 1.0}
    ks = RP.knapsack(effects, nbar, np.inf, 5.0, 1.0)
    assert ks["defer"] == [("ptx", "all")] and ks["policy"][("sleep", "all")] == "accept"
    assert RP.knapsack(effects, nbar, np.inf, np.inf, np.inf, lb_rule=False)["obj"] == pytest.approx(58)
    unsup = [eff("carrier", "all", pv=-5, e=1, n=10)]                    # unsupported leaf -> never deferred
    assert RP.knapsack(unsup, {("carrier", "all"): 1.0}, np.inf, np.inf, np.inf)["defer"] == []


def test_accept_all_table_is_zero_change():
    effects = [eff("carrier", "all", pv=-5, e=1, lb=-1.0), eff("ptx", "all", pv=2, e=1, lb=-2.0)]
    nbar = {("carrier", "all"): 2.0, ("ptx", "all"): 3.0}
    ks = RP.knapsack(effects, nbar, np.inf, np.inf, np.inf)
    assert ks["defer"] == [] and all(a == "accept" for a in ks["policy"].values())
    assert ks["obj"] == 0.0 and ks["e"] == 0.0 and ks["v"] == 0.0 and ks["rlf"] == 0.0
    # zero budget with costly deferrals: accept-all as well
    effects = [eff("carrier", "all", pv=-5, e=1)]
    assert RP.knapsack(effects, {("carrier", "all"): 1.0}, 0.0, np.inf, np.inf)["obj"] == 0.0


def synth(n_eps=24, per=12, seed=1):
    rng = np.random.default_rng(seed)
    X, y, ep = [], [], []
    for e in range(n_eps):
        for _ in range(per):
            x = rng.normal(size=len(RP.FEATURES))
            X.append(x)
            y.append(5.0 * (x[0] > 0) + rng.normal())
            ep.append(1000 + e)
    return np.array(X), np.array(y), np.array(ep)


def test_honest_split_never_uses_estimation_episodes():
    X, y, ep = synth()
    sp, es = RP.split_halves(ep, 0)
    assert sp.isdisjoint(es) and sp | es == set(ep.tolist())
    assert RP.split_halves(ep, 1) == (es, sp)
    t1 = RP.fit_honest_tree(X, y, ep, sp, es)
    assert t1.nodes and t1.nodes[""][0] == 0                                  # finds the planted split
    # scrambling the ESTIMATION half's features / outcomes cannot change the chosen splits
    est = np.isin(ep, list(es))
    X2, y2 = X.copy(), y.copy()
    rng = np.random.default_rng(5)
    X2[est] = rng.normal(size=X2[est].shape)
    y2[est] = rng.normal(size=est.sum()) * 100
    t2 = RP.fit_honest_tree(X2, y2, ep, sp, es)
    assert t2.nodes == t1.nodes or (t2.note and set(t2.nodes) < set(t1.nodes))


def test_tree_support_rules_and_fallback():
    X, y, ep = synth(n_eps=24, per=12)
    sp, es = RP.split_halves(ep, 0)
    t = RP.fit_honest_tree(X, y, ep, sp, es)
    lf = t.leaf_of(X)
    assert len(t.leaves) <= RP.MAX_LEAVES
    for h in (sp, es):
        m = np.isin(ep, list(h))
        for leaf in t.leaves:
            assert (m & (lf == leaf)).sum() >= RP.MIN_LEAF_N
            assert len(np.unique(ep[m & (lf == leaf)])) >= RP.MIN_LEAF_EPS
    Xs, ys, eps = synth(n_eps=20, per=2)                                      # 20 labels per half: no split
    sp, es = RP.split_halves(eps, 0)
    assert RP.fit_honest_tree(Xs, ys, eps, sp, es).leaves == ["all"]


def test_ctx_only_features():
    assert not any(p in f for f in RP.FEATURES for p in RP.PRIVILEGED_KEYS)
    for p in ("gt_static", "gt_labels", "lab_outcome"):
        assert p in RP.PRIVILEGED_KEYS
        with pytest.raises(ValueError):
            RP.ctx_features({}, features=(p,))
    ctx = {f: float(i) for i, f in enumerate(RP.FEATURES)}
    assert np.array_equal(RP.ctx_features(ctx), np.arange(len(RP.FEATURES), dtype=float))
    # a record's privileged content never reaches the features: only unit["ctx"] is read
    rec = {"seed": 1, "units": [{"knob": "carrier", "c": 0, "t0": 130.0, "step": -1.0, "ctx": ctx}],
           "gt_static": {"own_prb_util": 99.0}, "lab_outcome": {"own_prb_util": 99.0}}
    rows = RP.unit_table([rec])
    assert len(rows) == 1 and np.array_equal(rows[0]["x"], RP.ctx_features(ctx))


def test_label_rows_network_sign():
    delta = np.zeros((3, 6, 4))
    delta[:, 0, 1] = 2.0                                                      # accept - reject pv = +2 on cell 1
    delta[:, 1, :] = -100.0                                                   # e: accept saves 400 J network
    rec = {"seed": 7, "units": [{"knob": "carrier", "c": 0, "t0": 130.0, "step": -1.0, "ctx": {}}],
           "gt_labels": [{"i": 0, "c": 0, "knob": "carrier", "t0": 130.0, "step": -1.0, "exp": [0, 1],
                          "kpis": ["pv", "e", "v", "rlf", "load", "prb"], "delta": delta}]}
    rows = RP.label_rows([rec], lambda d: d)
    assert rows[0]["y"]["pv"] == 2.0 and rows[0]["y"]["e"] == -400.0
    tree = RP.HonestTree({}, RP.FEATURES, (), (7,))
    (ef,) = RP.effect_table(rows, {"carrier": tree}, n_boot=50)
    assert ef.defer["pv"] == -2.0 and ef.benefit == 2.0 and ef.defer["e"] == 400.0

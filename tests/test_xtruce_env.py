"""Mechanism tests for the xTRUCE plant re-implementation (cdd_oran.envs.xtruce)."""
import numpy as np
import pytest

from cdd_oran.envs.xtruce import (
    AcceptAll,
    Clipping,
    Direct,
    RejectAll,
    StaticPriority,
    Subset,
    XConfig,
    XEnv,
    conflict_groups,
    half,
    modify,
    screened_seeds,
)


def _run(env, n, arb=None):
    return [env.step(arb) for _ in range(n)]


def _same_traj(a, b):
    for ka, kb in zip(a, b, strict=True):
        for k in ("rate", "thr", "power_w", "queue_bits"):
            np.testing.assert_array_equal(ka[k], kb[k])


def test_paper_defaults():
    c = XConfig()
    assert (c.n_cells, c.n_ues, c.isd_m, c.n_rb, c.rb_hz) == (4, 20, 500.0, 12, 360e3)
    assert (c.p_max_w, c.p_cir_w, c.p_rb_w, c.noise_w, c.epoch_s) == (10.0, 50.0, 2.0, 1.15e-14, 1.0)
    assert (c.fading_rho, c.shadow_db, c.traffic_bps, c.n_prot, c.rmin_bps) == (0.349, 8.0, 6e6, 3, 2e6)
    e = XEnv(c, 0)
    assert e.prot.sum() == 3 and e.G.shape == (4, 20, 12)
    d = np.hypot(*(e.sites[1] - e.sites[0]))
    assert d == pytest.approx(500.0)
    assert (e.dist >= 35.0 - 1e-9).all()


def test_determinism_same_seed():
    a, b = XEnv(XConfig(), 3), XEnv(XConfig(), 3)
    _same_traj(_run(a, 30, AcceptAll()), _run(b, 30, AcceptAll()))
    assert a.summary() == b.summary()
    c = XEnv(XConfig(), 4)
    assert not np.array_equal(c.Gm, a.Gm)


def test_copy_without_reseed_replays_exactly():
    e = XEnv(XConfig(), 1)
    _run(e, 7, AcceptAll())
    c = e.copy()
    _same_traj(_run(e, 20, AcceptAll()), _run(c, 20, AcceptAll()))
    # copy taken between the two phases carries the pending requests
    e2 = XEnv(XConfig(), 1)
    _run(e2, 5, AcceptAll())
    reqs = e2.step_propose()
    c2 = e2.copy()
    k1, k2 = e2.step_apply(reqs), c2.step_apply(list(reqs))
    np.testing.assert_array_equal(k1["rate"], k2["rate"])
    _same_traj(_run(e2, 10), _run(c2, 10))


def test_copy_reseed_keeps_state_redraws_future():
    e = XEnv(XConfig(), 2)
    _run(e, 6, AcceptAll())
    q_snap = e.Q.copy()
    r1, r1b, r2 = e.copy(reseed=1), e.copy(reseed=1), e.copy(reseed=2)
    for c in (r1, r2):
        np.testing.assert_array_equal(c.Q, e.Q)
        np.testing.assert_array_equal(c.h, e.h)
        np.testing.assert_array_equal(c.x, e.x)
        np.testing.assert_array_equal(c.Gm, e.Gm)
    base = e.copy()
    k0, k1, k1b, k2 = base.step(AcceptAll()), r1.step(AcceptAll()), r1b.step(AcceptAll()), r2.step(AcceptAll())
    # the current channel is state (same rates this epoch); arrivals are future draws (queues differ)
    np.testing.assert_array_equal(k0["rate"], k1["rate"])
    assert not np.array_equal(k0["queue_bits"], k1["queue_bits"])
    np.testing.assert_array_equal(k1["queue_bits"], k1b["queue_bits"])        # same k -> same re-drawn future
    assert not np.array_equal(k1["queue_bits"], k2["queue_bits"])
    k0, k1 = base.step(AcceptAll()), r1.step(AcceptAll())
    assert not np.array_equal(k0["rate"], k1["rate"])                          # fading re-drawn
    # the original is untouched by its copies
    np.testing.assert_array_equal(e.Q, q_snap)
    assert e.t == 6


def test_common_random_numbers_across_arbiters():
    a, b = XEnv(XConfig(hallucination=0.5), 0), XEnv(XConfig(hallucination=0.5), 0)
    _run(a, 40, RejectAll())
    _run(b, 40, AcceptAll())
    assert a.acc["arrived_bits"] == b.acc["arrived_bits"]
    np.testing.assert_array_equal(a.h, b.h)


def _cfg_epoch(env):
    while not (env.t % env.cfg.t_cfg == 0):
        env.step(RejectAll())


def test_sleep_cuts_cell_power_and_moves_its_ues():
    e = XEnv(XConfig(), 0)
    _run(e, 1, RejectAll())
    _cfg_epoch(e)
    b = int(np.argmax(e.n_in_cell()))
    ues = np.nonzero(e.z == b)[0]
    e.step_propose()
    k = e.step_apply([{"xapp": "t", "knob": ("sleep", b), "prop": 1.0, "cur": 0.0, "t": e.t, "kind": "cfg"}])
    assert k["sleep"][b] and k["power_w"][b] == e.cfg.p_sleep_w
    assert k["power_w"][b] < k["power_w"][np.nonzero(~k["sleep"])[0]].min()
    assert (e.z[ues] != b).all() and e.alpha[e.z].all()
    assert (k["thr"] >= 0).all() and k["ptx_w"][b] == 0
    # off-cycle configuration writes are ignored; at the next configuration epoch the UEs return on wake
    e.step_propose()
    e.step_apply([{"xapp": "t", "knob": ("sleep", b), "prop": 0.0, "cur": 1.0, "t": e.t, "kind": "cfg"}])
    assert not e.alpha[b] and e.stats["cfg_offcycle"] == 1
    _cfg_epoch(e)
    e.step_propose()
    e.step_apply([{"xapp": "t", "knob": ("sleep", b), "prop": 0.0, "cur": 1.0, "t": e.t, "kind": "cfg"}])
    assert e.alpha[b] and (e.z[ues] == b).all()


def test_never_sleeps_last_cell():
    e = XEnv(XConfig(t_cfg=1), 0)
    for b in range(e.B):
        e.step_propose()
        e.step_apply([{"xapp": "t", "knob": ("sleep", b), "prop": 1.0, "cur": 0.0, "t": e.t, "kind": "cfg"}])
    assert e.alpha.sum() == 1


def test_higher_share_raises_that_ues_throughput():
    e = XEnv(XConfig(), 5)
    _run(e, 3, RejectAll())
    b = int(np.argmax(e.n_in_cell()))
    u, v = np.nonzero(e.z == b)[0][:2]
    e.step_propose()
    lo, hi = e.copy(), e.copy()
    k_lo = lo.step_apply([])
    dx = 0.5 * e.x[v]
    k_hi = hi.step_apply([modify({"xapp": "t", "knob": ("share", int(u)), "cur": e.x[u], "t": e.t, "kind": "fast",
                                  "prop": 0}, e.x[u] + dx),
                          modify({"xapp": "t", "knob": ("share", int(v)), "cur": e.x[v], "t": e.t, "kind": "fast",
                                  "prop": 0}, e.x[v] - dx)])
    assert k_hi["rate"][u] > k_lo["rate"][u] and k_hi["rate"][v] < k_lo["rate"][v]
    others = np.arange(e.U) != u
    others &= np.arange(e.U) != v
    np.testing.assert_allclose(k_hi["rate"][others], k_lo["rate"][others])  # same cell power -> no spill-over


def test_power_increase_raises_neighbour_interference_and_energy():
    e = XEnv(XConfig(), 0)
    e.step_propose()
    base = e.copy()
    k0 = base.step_apply([{"xapp": "t", "knob": ("pcap", 0), "prop": 3.0, "cur": 10.0, "t": 0, "kind": "fast"}])
    k1 = e.step_apply([])
    nb = e.z != 0
    assert k1["intf_out_w"][0] > k0["intf_out_w"][0]
    assert (e.I_meas[nb].sum(1) > base.I_meas[nb].sum(1)).all()
    assert (k1["rate"][nb] <= k0["rate"][nb] + 1e-9).all() and (k1["rate"][nb] < k0["rate"][nb]).any()
    assert k1["power_w"][0] - k0["power_w"][0] == pytest.approx(7.0)            # eq. (16): 50 + P_tx
    assert k0["power_w"][0] == pytest.approx(53.0)


def test_plant_clips_to_physical_limits():
    e = XEnv(XConfig(), 0)
    e.step_propose()
    u = int(np.nonzero(e.z == 0)[0][0])
    e.step_apply([{"xapp": "t", "knob": ("share", u), "prop": 3.0, "cur": 0, "t": 0, "kind": "fast"},
                  {"xapp": "t", "knob": ("power", u), "prop": 50.0, "cur": 0, "t": 0, "kind": "fast"}])
    c, z = e.cfg, e.z
    assert e.log[-1]["phys_viol"]
    assert (np.bincount(z, e.x_exec, minlength=e.B) <= 1 + 1e-9).all()
    assert (np.bincount(z, e.p_exec, minlength=e.B) <= c.p_max_w + 1e-9).all()
    assert (e.p_exec <= e.x_exec * e.K * c.p_rb_w + 1e-9).all()


def test_locks_drop_requests():
    e = XEnv(XConfig(), 0)
    e.lock(("pcap", 1), 3)
    for _ in range(3):
        e.step_propose()
        e.step_apply([{"xapp": "t", "knob": ("pcap", 1), "prop": 1.0, "cur": 10.0, "t": e.t, "kind": "fast"}])
        assert e.pcap[1] == 10.0
    e.step_propose()
    e.step_apply([{"xapp": "t", "knob": ("pcap", 1), "prop": 1.0, "cur": 10.0, "t": e.t, "kind": "fast"}])
    assert e.pcap[1] == 1.0 and e.stats["lock_blocked"] == 3


def test_energy_xapp_alone_meets_its_cap_and_saves():
    fr, es = XEnv(XConfig(), 0), XEnv(XConfig(), 0)
    s0, s1 = fr.run(30, RejectAll()), es.run(30, Subset(["ES"]))
    assert s1["energy_j"] < s0["energy_j"]
    assert es.log[-1]["power_w"].max() <= XConfig().es_cap_w + 1e-9


def test_qos_xapp_protects_floors():
    fr, q = XEnv(XConfig(), 0), XEnv(XConfig(), 0)
    s0, s1 = fr.run(100, RejectAll()), q.run(100, Subset(["QoS"]))
    assert s1["prot_viol_frac"] < s0["prot_viol_frac"]


def test_low_load_es_sleeps_cells():
    e = XEnv(XConfig(traffic_bps=0.5e6), 0)
    s = e.run(60, Subset(["ES"]))
    assert s["sleep_cell_s"] > 0


def test_arbiters_and_conflict_groups():
    e = XEnv(XConfig(), 0)
    e.step_propose()
    mk = lambda x, k, v: {"xapp": x, "knob": k, "prop": v, "cur": e.knob_get(k), "t": 0,  # noqa: E731
                          "kind": "cfg" if k[0] in ("sleep", "assoc") else "fast"}
    u0 = int(np.nonzero(e.z == 0)[0][0])
    reqs = [mk("ES", ("pcap", 0), 3.0), mk("QoS", ("pcap", 0), 10.0), mk("QoS", ("share", u0), 0.5),
            mk("IC", ("pcap", 2), 4.0)]
    g = conflict_groups(e, reqs)
    assert len(g) == 1 and g[0]["direct"] and g[0]["cells"] == [0] and g[0]["xapps"] == ["ES", "QoS"]
    assert conflict_groups(e, [mk("ES", ("pcap", 0), 3.0), mk("QoS", ("share", u0), 0.5)])[0]["direct"] is False
    acc = StaticPriority(["QoS", "ES", "IC"]).decide(e, reqs)
    assert [r["xapp"] for r in acc if r["knob"] == ("pcap", 0)] == ["QoS"] and len(acc) == 3
    assert {r["xapp"] for r in Subset(["ES"]).decide(e, reqs)} == {"ES"}
    assert half(reqs[0])["prop"] == pytest.approx(6.5) and half(mk("ES", ("sleep", 1), 1.0)) is None
    e.step_apply(reqs)                                                          # last writer wins
    assert e.pcap[0] == 10.0


def test_paper_benchmarks_direct_and_clipping():
    d = XEnv(XConfig(phys="none"), 0)
    d.run(5, Direct())
    assert d.acc["phys_viol_epochs"] == 5                     # Direct ignores joint feasibility
    c = XEnv(XConfig(phys="none"), 0)
    c.run(5, Clipping())
    assert c.acc["phys_viol_epochs"] == 0                     # Clipping always meets c1-c4


def test_xtruce_arbiter_keeps_floors_and_limits():
    pytest.importorskip("scipy")
    from cdd_oran.envs.xtruce.xtruce_arbiter import XTruce
    e = XEnv(XConfig(), 0)
    a = XTruce(maxiter=60)
    e.run(3, a)
    assert a.cases["uncertified"] == 0
    assert e.acc["phys_viol_epochs"] == 0 and e.acc["prot_viol_ue_s"] == 0


def test_screened_seeds_pass_the_floor_screen():
    s = screened_seeds(3, start=0)
    assert len(s) == 3 and all(XEnv(XConfig(), k).floors_reachable() for k in s)


def test_association_move_is_work_conserving():
    e = XEnv(XConfig(), 0)
    _run(e, 1, RejectAll())
    _cfg_epoch(e)
    a = int(np.argmax(e.n_in_cell()))
    u = int(np.nonzero(e.z == a)[0][0])
    b = int(np.argsort(e.Gm[:, u])[-2])
    sx, sp = np.bincount(e.z, e.x, minlength=e.B), np.bincount(e.z, e.p, minlength=e.B)
    e.step_propose()
    e.step_apply([{"xapp": "LB", "knob": ("assoc", u), "prop": float(b), "cur": float(a), "t": e.t, "kind": "cfg"}])
    assert e.z[u] == b
    np.testing.assert_allclose(np.bincount(e.z, e.x, minlength=e.B)[[a, b]], sx[[a, b]])
    np.testing.assert_allclose(np.bincount(e.z, e.p, minlength=e.B)[[a, b]], sp[[a, b]])

"""E6-P discovery knockout ground truth (cdd_oran.decision.gt_p) and the plant mechanisms it relies on: a forced pico
sleep hands its UEs to their best remaining cell, a forced macro carrier-off moves nobody, a forced ptx drop moves the
A3 border; Labeller(cells=True) per-cell contrasts; the TRUE / NULL / INDET rules on synthetic contrasts. DEV seeds."""
from __future__ import annotations

import numpy as np
import pytest

from cdd_oran.decision import collect_p as CP
from cdd_oran.decision import gt_p as G
from cdd_oran.decision import labels_p as LP
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env
from cdd_oran.envs.e6.ric import knob_set

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def cfg_p3(seed=7, warm=60.0, scored=60.0, lf=1.0):
    return C.E6Config(seed=seed, mix="ES", scenario="surge", load_factor=lf, n_ue=300, n_pico=3, mobility="ped",
                      warmup_s=warm, scored_s=scored,
                      e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "SliceGuarantee")))


@pytest.fixture(scope="module")
def env60():
    env = E6Env(cfg_p3(), log=False, wg3=True)
    while env.sec < 60:
        env.step(None)
    return env


def _ticks(plant, n):
    for _ in range(n):
        plant.tick()


# ---------------------------------------------------------------------------------------------- plant mechanisms
def test_forced_pico_sleep_moves_its_ues_to_the_best_remaining_cell(env60):
    env = env60.copy()
    p = env.plant
    picos = [c for c in range(p.nc) if not p.lay.is_macro[c]]
    pico = max(picos, key=lambda c: int((p.serv == c).sum()))
    ues = np.nonzero(p.serv == pico)[0]
    assert len(ues) > 0 and not p.asleep[pico]
    now = float(env.sec)
    off = p.asleep.copy() | (p.waking_until > now)
    off[pico] = True
    want = np.argmax(np.where(off[None, :], -np.inf, p.l3[ues]), 1)
    before = np.bincount(p.serv, minlength=p.nc)
    knob_set(p, ("sleep", pico), 1.0, now)
    assert p.asleep[pico]
    ok = p.int_until[ues] - now <= C.RLF_OUTAGE_S - 1e-9           # a handover (not a failure -> RLF)
    assert ok.any()
    assert np.array_equal(p.serv[ues][ok], want[ok])
    assert not (p.serv[ues][ok] == pico).any()
    after = np.bincount(p.serv, minlength=p.nc)
    assert after[pico] <= before[pico] - int(ok.sum())
    moved = np.bincount(want[ok], minlength=p.nc)
    assert all(after[c] >= before[c] + moved[c] - int((~ok).sum()) for c in range(p.nc) if c != pico)
    assert int(ok.sum()) + int((~ok).sum()) == len(ues)


def test_forced_macro_carrier_off_changes_no_serving_cell(env60):
    a, b = env60.copy(), env60.copy()
    pa, pb = a.plant, b.plant
    macros = [c for c in range(pa.nc) if pa.lay.is_macro[c] and pa.n_car[c] == pa.n_trx[c]]
    m = max(macros, key=lambda c: int((pa.serv == c).sum()))
    knob_set(pb, ("carrier", m), 1.0, float(b.sec))
    assert pb.n_car[m] == pa.n_car[m] - 1
    assert np.array_equal(pa.serv, pb.serv)
    assert np.array_equal(pa._gains(), pb._gains())                  # the A3 / RSRP inputs do not see carriers
    _ticks(pa, 1)
    _ticks(pb, 1)
    assert np.array_equal(pa.serv, pb.serv) and np.array_equal(pa.l3, pb.l3)
    assert pb.rho[m] >= pa.rho[m]                                     # capacity (not coverage) shrank
    for _ in range(C.TICKS_PER_CONTROL - 1):
        pa.tick()
        pb.tick()
        assert np.array_equal(pa._gains(), pb._gains())


def test_forced_ptx_drop_moves_the_a3_border_and_serving_cells(env60):
    a, b = env60.copy(), env60.copy()
    pa, pb = a.plant, b.plant
    macros = [c for c in range(pa.nc) if pa.lay.is_macro[c]]
    m = max(macros, key=lambda c: int((pa.serv == c).sum()))
    knob_set(pb, ("ptx", m), pb.ptx_off[m] - 9.0, float(b.sec))
    ga, gb = pa._gains(), pb._gains()
    assert np.allclose(gb[:, m], ga[:, m] - 9.0)
    other = np.arange(pa.nc) != m
    assert np.array_equal(ga[:, other], gb[:, other])
    best_a = (np.argmax(ga, 1) == m).sum()
    best_b = (np.argmax(gb, 1) == m).sum()
    assert best_b < best_a                                            # the RSRP border of m moved inward
    n = 5 * C.TICKS_PER_CONTROL
    _ticks(pa, n)
    _ticks(pb, n)
    sa, sb = int((pa.serv == m).sum()), int((pb.serv == m).sum())
    assert sb < sa                                                    # UEs left m (A3 handovers)
    left = np.nonzero((pa.serv == m) & (pb.serv != m))[0]
    assert len(left) > 0 and not (pb.serv[left] == m).any()


# ---------------------------------------------------------------------------------------------- Labeller(cells=True)
def test_labeller_cells_option_adds_per_cell_contrasts_and_keeps_the_default():
    H, got = 10, {}
    lab_c, lab_0 = LP.Labeller(ks=(1, 2), H=H, cells=True), LP.Labeller(ks=(1, 2), H=H)

    def hook(env, obs, snap, opened):
        for u in opened:
            if "u" in got or u["knob"] not in ("carrier", "ptx", "prot_min") or u["t0"] + H > env.total_s:
                continue
            got.update(u=u, c=lab_c.label(env, obs, snap, u, modes=G.GT_MODES),
                       d=lab_0.label(env, obs, snap, u, modes=G.GT_MODES))

    res = CP.run_collection(cfg_p3(seed=9, scored=40.0), CP.RandomizedUnitPolicy(9, tables=CP.PI0_HIGH_NO_RB),
                            labeller=hook, open_rule="feasible")
    assert "u" in got
    u, Lc, L0 = got["u"], got["c"], got["d"]
    assert "cell" not in L0 and "cell_kpis" not in L0
    for key in ("modes", "H", "ks", "raw", "d", "mean"):
        assert Lc[key] == L0[key]                                     # default output unchanged by the option
    assert Lc["cell_kpis"] == list(LP.CELL_KPIS)
    n = res["env"].plant.nc
    assert Lc["cell"]["reject"].shape == (2, len(LP.CELL_KPIS), n)
    assert not Lc["cell"]["accept"].any()
    for i in range(2):
        for j, kpi in enumerate(LP.KPIS):                             # pv, e, v: the exposure-set sums of d
            assert Lc["cell"]["reject"][i, j, u["exp"]].sum() == pytest.approx(Lc["d"]["reject"]["exp"][kpi][i])
    assert np.array_equal(G.unit_delta(Lc), -Lc["cell"]["reject"])
    with pytest.raises(ValueError):
        G.unit_delta(L0)


# ---------------------------------------------------------------------------------------------- GT rules
NC = 8
EXP = {0: [0, 1, 2, 3], 4: [2, 3, 4, 5]}          # far(0) = 4..7, far(4) = 0, 1, 6, 7


def _label(rng, c, knob, step, load_nbr, pv_nbr, noise=0.2):
    kp = list(LP.CELL_KPIS)
    D = np.zeros((3, len(kp), NC))
    nbr = [q for q in EXP[c] if q != c]
    for k in range(3):
        D[k, kp.index("load"), c] = -3 * load_nbr
        D[k, kp.index("load"), nbr] = load_nbr + rng.normal(0, noise, len(nbr))
        D[k, kp.index("pv"), nbr] = pv_nbr / len(nbr) + rng.normal(0, 0.01, len(nbr))
        D[k, kp.index("e"), c] = rng.normal(0, 50.0)                  # wide: INDET
    s = np.sign(step)
    return {"c": c, "knob": knob, "exp": EXP[c], "step": step, "kpis": kp, "delta": D * s}


def _episodes(n_ep=6, per=8, seed=0):
    rng = np.random.default_rng(seed)
    eps = []
    for e in range(n_ep):
        labs = [_label(rng, 0, "sleep", 1.0, 1.0, 2.0) for _ in range(per)]
        labs += [_label(rng, 0, "sleep", -1.0, 1.0, 2.0) for _ in range(per // 2)]   # wake: dir-aligned
        labs += [_label(rng, 4, "prot_min", 0.05, 0.0, 0.3) for _ in range(per)]    # tiny pv: under the floor
        labs += [_label(rng, 4, "ptx", -3.0, 1.0, 0.0) for _ in range(1 if e == 0 else 0)]   # unsupported
        eps.append({"seed": 1000 + e, "n_cells": NC, "gt_static": {"cand": {"0": 3}}, "gt_labels": labs})
    return eps


def test_classify_rules():
    assert G.classify(1.0, (0.5, 1.5), 0.5, 20, 5)["status"] == "TRUE"
    assert G.classify(-1.0, (-1.5, -0.5), 0.5, 20, 5)["sign"] == -1
    assert G.classify(0.3, (0.1, 0.45), 0.5, 20, 5)["status"] == "NULL"      # excludes 0 but |mean| < delta
    assert G.classify(0.0, (-0.4, 0.4), 0.5, 20, 5)["status"] == "NULL"
    assert G.classify(0.0, (-2.0, 2.0), 0.5, 20, 5)["status"] == "INDET"
    assert G.classify(0.4, (0.1, 0.7), 0.5, 20, 5)["status"] == "INDET"      # excludes 0, too small, CI over delta
    assert G.classify(9.0, (8.0, 10.0), 0.5, 5, 5)["status"] == "INDET"      # unsupported (units)
    assert G.classify(9.0, (8.0, 10.0), 0.5, 50, 2)["status"] == "INDET"     # unsupported (episodes)


def test_gt_table_true_null_indet_on_synthetic_contrasts():
    eps = _episodes()
    rows = G.unit_rows(eps)
    rel = G.relation_cells(0, EXP[0], NC)
    assert rel == {"own": [0], "nbr": [1, 2, 3], "far": [4, 5, 6, 7]}
    tab = G.gt_table(rows, "dir", n_boot=500)
    cell = {(c["family"], c["relation"], c["kpi"]): c for c in tab["cells"]}
    assert len(tab["cells"]) == len(G.FAMILIES) * len(G.RELATIONS) * len(G.GT_KPIS)
    assert cell[("sleep", "nbr", "load")]["status"] == "TRUE" and cell[("sleep", "nbr", "load")]["sign"] == 1
    assert cell[("sleep", "own", "load")]["status"] == "TRUE" and cell[("sleep", "own", "load")]["sign"] == -1
    assert cell[("sleep", "nbr", "pv")]["status"] == "TRUE" and cell[("sleep", "nbr", "pv")]["sign"] == 1
    for k in G.GT_KPIS:
        assert cell[("sleep", "far", k)]["status"] == "NULL"
    assert tab["delta"]["pv"] == 0.5                                  # floor (5 % of 2.0 = 0.1 < 0.5)
    assert cell[("prot_min", "nbr", "pv")]["status"] == "NULL"        # 0.3 UE-s: excludes 0 but under the floor
    assert cell[("sleep", "own", "e")]["status"] == "INDET"
    assert cell[("ptx", "nbr", "load")]["status"] == "INDET" and "unsupported" in cell[("ptx", "nbr", "load")]["why"]
    assert cell[("carrier", "own", "pv")]["n"] == 0
    assert sum(tab["counts"].values()) == len(tab["cells"])
    act = {(c["family"], c["relation"], c["kpi"]): c for c in G.gt_table(rows, "act", n_boot=500)["cells"]}
    assert act[("sleep", "nbr", "load")]["mean"] < cell[("sleep", "nbr", "load")]["mean"]   # wake units cancel
    again = G.gt_table(rows, "dir", n_boot=500)
    assert np.array_equal([c["ci"] for c in again["cells"]], [c["ci"] for c in tab["cells"]],
                          equal_nan=True)                             # tag-keyed, reproducible


def test_receiving_shares_top1_vs_candidate():
    eps = _episodes()
    rs = G.receiving_shares(G.unit_rows(eps), {e["seed"]: e["gt_static"] for e in eps})
    assert len(rs) == len(eps) and all(r["pico"] == 0 and r["n_units"] == 8 for r in rs)
    for r in rs:
        assert r["cand"] == 3 and r["top1"] in (1, 2, 3)
        assert sum(r["shares"].values()) == pytest.approx(1.0)
        assert r["top1_is_cand"] == (r["top1"] == 3) and r["pico_load_delta"] < 0

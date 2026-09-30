"""E6-P discovery panel (cdd_oran.decision.features.build_panel_p): the E6-P columns exist with their kinds, the panel
is obs-only (no gt_ / lab_ column, identical without the trace's lab_ arrays or site map), the record codec
round-trips, the own_sleep minority filter is judged on pico rows, the discovery kpi_owner override, and the driver's
seed / freeze guards. Short DEV-seed episode."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from cdd_oran.decision import collect_p as CP
from cdd_oran.decision import discovery as D
from cdd_oran.decision import features as F
from cdd_oran.decision.trace import Trace
from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

E6P_COLS = {"own_ptx": ("knob_own", "ptx"), "own_prot_min": ("knob_own", "prot_min"),
            "own_carrier": ("knob_own", "carrier"), "own_sleep": ("knob_own", "sleep"),
            "nbr_sleep": ("knob_nbr", "sleep"), "nbr_ptx": ("knob_nbr", "ptx"),
            "nbr_prot_min": ("knob_nbr", "prot_min"), "nbr_carrier": ("knob_nbr", "carrier"),
            "nbr_act_ue_lag": ("state_nbr", "act_ue"), "prot_viol": ("kpi", "prot_viol"),
            "prot_act_ue": ("kpi", "prot_act_ue"), "edge_sinr_p": ("kpi", "edge_sinr_p"),
            "prot_viol_lag": ("kpi_lag", "prot_viol"), "edge_sinr_p_lag": ("kpi_lag", "edge_sinr_p")}


def cfg_p3(seed=8):
    return C.E6Config(seed=seed, mix="ES", scenario="surge", load_factor=1.0, n_ue=300, n_pico=3, mobility="ped",
                      warmup_s=60.0, scored_s=120.0,
                      e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "SliceGuarantee")))


@pytest.fixture(scope="module")
def run():
    cfg = cfg_p3()
    env = E6Env(cfg, log=False, wg3=True, trace=True)
    CP.run_collection(cfg, CP.RandomizedUnitPolicy(cfg.seed, tables=CP.PI0_HIGH_NO_RB), env=env,
                      open_rule="feasible")
    tr = env.get_trace()
    return tr, F.build_panel_p(tr, step_s=10, episode=5, drop_degenerate=False)


def test_e6p_columns_exist_with_kinds_and_families(run):
    _, p = run
    for col, (kind, fam) in E6P_COLS.items():
        assert col in p.data, col
        assert p.kind[col] == kind and p.family[col] == fam
        assert p.data[col].shape == (p.n,)
    assert set(F.KNOB_FAMILIES) >= {"carrier", "sleep", "ptx", "prot_min"}
    assert set(F.KPI_FAMILIES_P) >= {"prot_viol", "prot_act_ue", "edge_sinr_p"}
    macro = p.data["is_macro"] > 0.5
    assert np.all(p.data["own_ptx"][~macro] == 0.0) and np.all(p.data["own_ptx"] <= 0.0)
    assert np.all((p.data["own_prot_min"] >= 0) & (p.data["own_prot_min"] <= 0.5))
    assert np.nanmax(p.data["own_prot_min"]) > 0                    # SliceGuarantee acted in the episode
    pv = p.data["prot_viol"]
    assert np.all(pv[np.isfinite(pv)] >= 0) and np.nansum(pv) > 0
    assert p.episode[0] == 5 and len(np.unique(p.t)) * len(p.neighbours) == p.n


def test_prot_viol_is_the_delivered_report_count(run):
    tr, p = run
    a = tr.arrays
    t0, t1 = a["kpm_fast_t0"], a["kpm_fast_t1"]
    tk = int(np.unique(p.t)[3])
    inside = np.nonzero((t0 >= tk) & (t1 <= tk + 10))[0]
    ev, fr = a["kpm_fast_prot_eval"][inside].astype(float), a["kpm_fast_prot_below_frac"][inside].astype(float)
    want = np.where(ev > 0, np.rint(np.nan_to_num(fr) * ev), 0.0).mean(0)
    got = p.data["prot_viol"][p.t == tk]
    assert np.allclose(got, want)


def test_panel_is_obs_only(run):
    tr, p = run
    assert not any(c.startswith(("lab_", "gt_")) for c in p.data)
    stripped = Trace(tr.features(), dict(tr.meta, cell_region=[0] * len(tr.meta["cell_region"])))
    assert not any(k.startswith("lab_") for k in stripped.arrays)
    q = F.build_panel_p(stripped, step_s=10, episode=5, drop_degenerate=False)
    assert list(q.data) == list(p.data)
    for c in p.data:
        assert np.array_equal(p.data[c], q.data[c], equal_nan=True), c
    rec = F.panel_to_rec(p)
    assert "cell_region" not in rec and not any(k.startswith(("gt_", "lab_")) for k in rec["data"])
    back = F.panel_from_rec(rec)
    assert list(back.data) == list(p.data) and np.array_equal(back.t, p.t) and np.array_equal(back.cell, p.cell)
    for c in p.data:
        assert np.array_equal(back.data[c], p.data[c].astype(np.float32).astype(float), equal_nan=True), c
    assert np.array_equal(back.cell_region, np.arange(len(p.neighbours)))


def test_record_codec_roundtrip():
    x = np.random.default_rng(0).normal(size=(7, 3, 5))
    e = CP.enc(x)
    assert e["shape"] == [7, 3, 5] and e["dtype"] == "float32" and isinstance(e["b64"], str)
    assert np.array_equal(CP.dec(e), x.astype(np.float32))


def _sleep_panel(pico_share):
    C_, K = 24, 400
    is_macro = np.arange(C_) < 21
    sleep = np.zeros((K, C_))
    n_on = int(round(pico_share * K))
    sleep[:n_on, 21] = 1.0
    data = {"own_sleep": sleep.reshape(-1), "is_macro": np.broadcast_to(is_macro, (K, C_)).astype(float).reshape(-1),
            "own_prot_min": np.random.default_rng(1).uniform(size=K * C_)}
    return F.Panel(data=data, kind={"own_sleep": "knob_own", "is_macro": "context", "own_prot_min": "knob_own"},
                   family={"own_sleep": "sleep", "is_macro": "is_macro", "own_prot_min": "prot_min"},
                   episode=np.zeros(K * C_, int), t=np.repeat(np.arange(K) * 10, C_), cell=np.tile(np.arange(C_), K),
                   step_s=10, cell_region=np.arange(C_), neighbours=[[] for _ in range(C_)], ownership={},
                   xapps=[])


def test_own_sleep_minority_filter_is_judged_on_pico_rows():
    p = _sleep_panel(0.15)                           # 5 % of pico rows, 0.625 % of all rows
    assert "own_sleep" in F.degenerate_columns(p.data)                # the E6 rule would drop it
    q = F.drop_degenerate_p(p)
    assert "own_sleep" in q.data and "pico rows" in q.notes["own_sleep"]
    assert "own_sleep" not in q.dropped
    r = F.drop_degenerate_p(_sleep_panel(0.005))     # 2 of 1200 pico rows: degenerate there too
    assert "own_sleep" not in r.data and "pico rows" in r.dropped["own_sleep"]


def test_kpi_owner_override_and_tuple_owners():
    assert D.DiscoveryConfig().kpi_owner is None
    edges = [{"kpi": "energy_w", "column": "own_ptx", "family": "ptx", "kind": "knob_own", "scope": "own", "p": 0.01,
              "sign": -1, "coef": -1.0, "declared": True, "weight": 1.0},
             {"kpi": "prot_viol", "column": "nbr_sleep", "family": "sleep", "kind": "knob_nbr", "scope": "neighbour",
              "p": 0.02, "sign": 1, "coef": 1.0, "declared": True, "weight": 0.9}]
    own = {"ES": {"sleep": {"req": 3, "applied": 1}, "carrier": {"req": 2, "applied": 2}},
           "PowerES": {"ptx": {"req": 5, "applied": 3}}, "SliceGuarantee": {"prot_min": {"req": 9, "applied": 9}}}
    cfg = D.DiscoveryConfig(kpi_owner=D.KPI_OWNER_P)
    g = D.TemplateGraph(edges=edges, diagnostics={}, ownership=own, xapps=["ES", "PowerES", "SliceGuarantee"],
                        kpi_owner=dict(cfg.kpi_owner), cell_region=np.arange(3), neighbours=[[], [], []], config=cfg)
    cm = D.conflict_map(g)
    assert [(c["src_xapp"], c["kpi"], c["dst_xapp"]) for c in cm] == [("ES", "prot_viol", "SliceGuarantee")]
    g.kpi_owner = dict(D.KPI_OWNER)                   # E6 owners: energy owned by ES alone -> PowerES conflicts
    assert [(c["src_xapp"], c["kpi"]) for c in D.conflict_map(g)] == [("PowerES", "energy_w")]


# ---------------------------------------------------------------------------------------------- driver guards
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratchpad", "e6_dev")


@pytest.fixture(scope="module")
def drv():
    if not os.path.exists(os.path.join(HERE, "e6p_discovery.py")) or \
            not os.path.exists(os.path.join(HERE, "e6p_state_json.py")):
        pytest.skip("scratchpad/e6_dev/e6p_discovery.py or the E6-P state is absent")
    sys.path.insert(0, HERE)
    import e6p_discovery
    return e6p_discovery


def test_driver_seed_map_and_guards(drv):
    seeds = {st: [j[2] for j in drv.jobs(st)] for st in ("dev", "eval", "gt", "placebo", "prof")}   # v1 stages only
    assert seeds["dev"] == list(range(183300, 183320)) and seeds["gt"] == list(range(183380, 183400))
    assert seeds["eval"] == list(range(183320, 183380))
    assert [j[4] for j in drv.jobs("eval")][::20] == [0, 1, 2]
    assert seeds["placebo"] == list(range(183400, 183420)) and seeds["prof"] == list(range(183420, 183444))
    allseeds = [s for v in seeds.values() for s in v]
    assert len(set(allseeds)) == len(allseeds) == 20 + 60 + 20 + 20 + 24
    for bad in (179999, 184000, 160500, 155300, 150250):
        with pytest.raises(AssertionError):
            drv.check_seed(bad)
    if drv.FROZEN_SHA256 is None:
        for st in ("eval", "gt"):
            with pytest.raises(SystemExit, match="frozen"):
                drv.run(st, "0/1", os.devnull)
    assert drv.make_policy("placebo", "", 3)[1] == "placebo" and drv.make_policy("prof", "ES", 3)[1] == "profile:ES"
    pol = drv.ProfilePolicy("PowerES")
    assert pol({"x": "PowerES"}) == ("accept", 1.0) and pol({"x": "ES"}) == ("reject", 1.0)

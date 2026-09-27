"""Template discovery (decision/features.py + decision/discovery.py): planted-edge recovery on a synthetic panel,
degenerate-column handling (MSCR BUG-1/BUG-2 guards), and an end-to-end run on a short real E6 trace."""
from __future__ import annotations

import json

import numpy as np
import pytest

from cdd_oran.decision import discovery as D
from cdd_oran.decision import features as F
from cdd_oran.discovery.mscr import MSCRConfig

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

C, K, STEP = 24, 60, 60           # cells, steps, step seconds -> 1440 rows


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    nb = [sorted({(c + d) % C for d in (-2, -1, 1, 2)}) for c in range(C)]
    hys = rng.integers(0, 11, (K, C)) * 0.5                                  # tied 0.5 dB grid
    ttt = rng.choice([40, 80, 160, 256, 320, 480, 640], (K, C)).astype(float)   # NULL knob
    car = rng.choice([0.5, 1.0], (K, C))                                     # binary, heavily tied
    nbr_car = np.stack([car[:, n].mean(1) for n in nb], 1)
    cio_out = rng.normal(0, 2, (K, C)).round()                               # NULL knob
    cio_in = (rng.uniform(size=(K, C)) < 0.03).astype(float)                 # NULL, 97 % tied at 0
    trend = np.linspace(-2, 2, K)[:, None] * np.ones((1, C))                 # strong time drift in every KPI
    z = lambda v: (v - v.mean()) / v.std()                                    # noqa: E731
    thp = 0.6 * z(hys) - 0.6 * z(nbr_car) + trend + rng.normal(0, 1, (K, C))
    energy = 0.8 * z(car) + trend + rng.normal(0, 1, (K, C))
    lag = lambda v: np.vstack([np.full((1, C), np.nan), v[:-1]])            # noqa: E731
    cols = {"own_hys": (hys, "knob_own", "hys"), "own_ttt": (ttt, "knob_own", "ttt"),
            "own_carrier": (car, "knob_own", "carrier"), "own_sleep": (np.zeros((K, C)), "knob_own", "sleep"),
            "cio_out_mean": (cio_out, "knob_own", "cio"), "cio_in_mean": (cio_in, "knob_nbr", "cio"),
            "nbr_carrier": (nbr_car, "knob_nbr", "carrier"),
            "embb_thp_p5": (thp, "kpi", "embb_thp_p5"), "energy_w": (energy, "kpi", "energy_w"),
            "embb_thp_p5_lag": (lag(thp), "kpi_lag", "embb_thp_p5"), "energy_w_lag": (lag(energy), "kpi_lag",
                                                                                     "energy_w"),
            "is_macro": (np.broadcast_to((np.arange(C) < 21).astype(float), (K, C)), "context", "is_macro")}
    own = {"MRO": {"hys": {"req": 5, "applied": 5}, "cio": {"req": 3, "applied": 1}},
           "TS": {"cio": {"req": 9, "applied": 9}}, "ES": {"carrier": {"req": 4, "applied": 2}}}
    p = F.Panel(data={k: np.asarray(v[0], float).reshape(-1) for k, v in cols.items()},
                kind={k: v[1] for k, v in cols.items()}, family={k: v[2] for k, v in cols.items()},
                episode=np.zeros(K * C, int), t=np.repeat(np.arange(K) * STEP, C), cell=np.tile(np.arange(C), K),
                step_s=STEP, cell_region=np.arange(C) // 3, neighbours=nb, ownership=own, xapps=["MRO", "TS", "ES"])
    return p.drop_degenerate()


@pytest.fixture(scope="module")
def synth_graph():
    cfg = D.DiscoveryConfig(mscr=MSCRConfig(n_perm=999), thin_s=STEP, targets=("embb_thp_p5", "energy_w"))
    return D.discover_template(_synthetic(), cfg)


def test_planted_edges_recovered_null_not(synth_graph):
    g = synth_graph
    assert all(d.status == "ok" and d.dependence_ok and d.power_floor_ok for d in g.diagnostics.values())
    hys, nbr = g.edge("own_hys", "embb_thp_p5"), g.edge("nbr_carrier", "embb_thp_p5")
    assert hys["declared"] and hys["sign"] == 1 and hys["scope"] == "own"
    assert nbr["declared"] and nbr["sign"] == -1 and nbr["scope"] == "neighbour"
    car = g.edge("own_carrier", "energy_w")
    assert car["declared"] and car["sign"] == 1
    # nulls: an untied null, a rounded-normal null, and a 97 %-tied null under a strong time trend (BUG-2 guard)
    for col in ("own_ttt", "cio_out_mean", "cio_in_mean"):
        for kpi in ("embb_thp_p5", "energy_w"):
            assert not g.edge(col, kpi)["declared"], (col, kpi)
    assert not g.edge("own_hys", "energy_w")["declared"]


def test_constant_columns_dropped_and_recorded(synth_graph):
    p = _synthetic()
    assert p.dropped == {"own_sleep": "constant"} and "own_sleep" not in p.data
    assert all(e["column"] != "own_sleep" for e in synth_graph.edges)
    bad = F.degenerate_columns({"c": np.full(50, 1.1), "near": np.r_[np.zeros(995), np.ones(5)],
                                "nan": np.full(3, np.nan), "ok": np.arange(10.0)})
    assert set(bad) == {"c", "near", "nan"}
    # degenerate TARGET (BUG-1 guard): skipped with a reason, no edges
    p.data["energy_w"] = np.full(p.n, 1.1) + 1e-15 * np.arange(p.n)
    g = D.discover_template(p, D.DiscoveryConfig(mscr=MSCRConfig(n_perm=99), thin_s=STEP, targets=("energy_w",)))
    assert g.diagnostics["energy_w"].status.startswith("skipped") and not g.edges


def test_conflict_map_and_context_prior(synth_graph):
    g = synth_graph
    cm = D.conflict_map(g, declared_only=True)
    pairs = {(r["src_xapp"], r["column"], r["kpi"], r["dst_xapp"]) for r in cm}
    assert ("MRO", "own_hys", "embb_thp_p5", "TS") in pairs          # MRO's hys reaches TS's KPI
    assert ("ES", "nbr_carrier", "embb_thp_p5", "TS") in pairs       # ES's carrier reaches TS's KPI via neighbours
    assert all(r["src_xapp"] != r["dst_xapp"] for r in D.conflict_map(g))
    assert not any(r["kpi"] == "energy_w" and r["src_xapp"] == "ES" for r in D.conflict_map(g))
    cp = D.context_mask(g, region=0)
    floor = g.config.weight_floor
    assert cp.cells == [0, 1, 2]
    assert all(floor <= w <= 1.0 for w in cp.column_weights.values())
    assert cp.column_weights["own_ttt"] >= floor                     # soft: nulls keep a floor weight
    assert cp.column_weights["own_hys"] > cp.column_weights["own_ttt"]
    assert [w for _, w in cp.ranked] == sorted((w for _, w in cp.ranked), reverse=True)
    assert cp.neighbour_regions and 0 not in dict(cp.neighbour_regions)
    regions, W = D.region_weight_matrix(g)
    assert regions == list(range(8)) and W.shape == (8, 8) and np.all(np.diag(W) == 0)
    assert np.all(W[~np.eye(8, dtype=bool)] >= floor)


def test_real_e6_trace_end_to_end():
    from cdd_oran.decision.collect import collect_episode
    from cdd_oran.envs.e6.config import E6Config

    tr = collect_episode(E6Config(seed=17, warmup_s=60, scored_s=120), eps=0.5)
    p = F.build_panel(tr, step_s=10)
    assert not any(c.startswith("lab_") for c in p.data)
    assert set(p.xapps) == set(tr.meta["xapps"]) and p.ownership.keys() == set(tr.meta["xapps"])
    assert all(r == "knob_absent_in_mix" or r.startswith(("constant", "near_constant", "all_nan"))
               for r in p.dropped.values())
    g = D.discover_template(p, D.DiscoveryConfig(mscr=MSCRConfig(n_perm=299), thin_s=10))
    assert set(g.diagnostics) == set(F.KPI_FAMILIES)
    assert any(d.status == "ok" for d in g.diagnostics.values())     # MSCR actually ran on the real panel
    for d in g.diagnostics.values():
        if d.status == "ok":
            assert set(d.n_eff) == set(d.tested) and d.n_rows >= 240
    for e in g.edges:
        assert 0 < e["p"] <= 1 and g.config.weight_floor <= e["weight"] <= 1 and e["sign"] in (-1, 0, 1)
        assert e["scope"] in ("own", "neighbour") and e["kind"] in ("knob_own", "knob_nbr", "state_nbr")
    cm = D.conflict_map(g)
    assert all(r["src_xapp"] != r["dst_xapp"] for r in cm)
    json.dumps(g.to_dict(), default=float)
    cp = D.context_mask(g, region=int(p.cell_region[0]))
    assert cp.cells and all(w >= g.config.weight_floor for w in cp.column_weights.values())

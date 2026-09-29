"""Discovery baselines (cdd_oran.decision.baselines_disc) on a tiny synthetic unit table: each recovers a strong
planted edge at its far-FPR threshold with the right sign; the threshold rules; the two-tower wrapper restores the
script; the MSCR-rowperm mapping; PACIFISTA's native severity."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cdd_oran.decision import baselines_disc as BD
from cdd_oran.decision import crt_units as CU
from cdd_oran.decision import edge_score as ES

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning", "ignore::UserWarning", "ignore::FutureWarning")

PLANT = ("sleep", "nbr", "pv")          # synthetic_records: sleep adds effect * x to nbr load and pv
OWN = ("ptx", "own", "e")               # synthetic_records: ptx adds 50 * x to own energy (the nuisance)


@pytest.fixture(scope="module")
def data():
    return CU.build_unit_data(CU.synthetic_records(np.random.default_rng(3), n_ep=6, units_per_ep=60, effect=120.0))


def _recovers(scores, h=PLANT, sign=1):
    tau = BD.tune_tau(scores, "far_fpr")
    assert np.isfinite(tau["tau"])
    d = BD.declare(scores, tau["tau"])
    assert d[h]["declared"], (h, scores[h], tau)
    assert d[h]["sign"] == sign
    assert sum(v["declared"] for k, v in d.items() if k[1] == "far") <= 1


@pytest.mark.parametrize("name", ["corr", "granger", "int"])
def test_fast_baselines_recover_the_planted_edge(data, name):
    _recovers(BD.SCORERS[name](data))


def test_shap_gbdt_recovers_the_planted_edge(data):
    sc = BD.score_shap(data)
    assert "fit_shap_importances" in sc["_note"] or "FALLBACK" in sc["_note"]
    _recovers(sc)


def test_two_tower_recovers_the_planted_edge_and_restores_the_script(data):
    import scripts.e2_baseline_gnn as G
    orig = G.TwoTower
    sc = BD.score_two_tower(data, sizes={"epochs": 300})
    assert G.TwoTower is orig
    assert "num_params" not in repr(G.TwoTower)
    _recovers(sc)


def test_qacm_recovers_an_own_cell_edge_and_never_declares_nbr(data):
    sc = BD.score_qacm(data, relations=("own", "far"), ann=False)
    tau = BD.tune_tau(sc, "far_fpr")
    d = BD.declare(sc, tau["tau"], BD.DECL_RELATIONS["qacm"])
    assert d[OWN]["declared"] and d[OWN]["sign"] == 1
    assert not any(v["declared"] for k, v in d.items() if k[1] != "own")
    ann = BD.score_qacm(data.subset(data.family == 2), relations=("own",), ann=True)     # ANN path runs
    assert np.isfinite(ann[OWN]["score"]) and ann[OWN]["model"] in ("ANN", "PR")


def test_granger_by_and_tau_rules():
    sc = {h: {"score": float("nan"), "sign": 0, "p": float("nan")} for h in ES.HYPOTHESES}
    far = [h for h in ES.HYPOTHESES if h[1] == "far"]
    for i, h in enumerate(far):
        sc[h] = {"score": float(i), "sign": 1, "p": 10.0 ** -i}
    t = BD.tune_tau(sc, "far_fpr")
    assert t["tau"] == len(far) - 2                                       # second-largest far score
    d = BD.declare(sc, t["tau"])
    assert sum(v["declared"] for v in d.values()) == 1
    few = {h: {"score": float("nan"), "sign": 0} for h in ES.HYPOTHESES}
    few[far[0]] = {"score": 1.0, "sign": 1}
    assert BD.tune_tau(few, "far_fpr")["tau"] == float("inf")
    sc[PLANT] = {"score": 100.0, "sign": 1, "p": 1e-30}
    tp = BD.tune_tau(sc, "physics_f1")
    assert BD.declare(sc, tp["tau"])[PLANT]["declared"]
    gb = BD.granger_by(sc)
    assert gb[PLANT]["declared"] and not gb[far[0]]["declared"]


def test_run_baselines_freezes_dev_tau(data):
    dev = data.subset(data.episode < 3)
    ev = data.subset(data.episode >= 3)
    out = BD.run_baselines(dev, {"pooled": ev}, methods=("corr", "granger"),
                           sup=dataclasses.replace(BD.Support(), min_episodes=2))
    assert out["corr"]["tau"]["rule"] == "far_fpr" and "granger_by" in out
    s = out["corr"]["splits"]["pooled"]
    tau = out["corr"]["tau"]["tau"]
    assert all(v["declared"] == (np.isfinite(v["score"]) and v["score"] > tau) for v in s["declared"].values())


def _fake_panel_records(rng, n_ep=2, K=72, C=24):
    from cdd_oran.decision.features import Panel, panel_to_rec
    recs = []
    for e in range(n_ep):
        n = K * C
        t = np.repeat(np.arange(K) * 10, C)
        cell = np.tile(np.arange(C), K)
        ptx = rng.choice([-6.0, -3.0, 0.0], n)
        cols = {"own_carrier": rng.uniform(0, 1, n), "own_sleep": (rng.uniform(size=n) < 0.1) * 1.0,
                "own_ptx": ptx, "own_prot_min": rng.choice([0.1, 0.2, 0.3], n),
                "nbr_sleep": rng.uniform(0, 1, n), "nbr_ptx": rng.uniform(-6, 0, n), "cio_out_mean": rng.normal(size=n),
                "energy_w": 100 + 20 * ptx + rng.normal(0, 1, n), "prot_viol": rng.poisson(1.0, n) * 1.0,
                "rlf": rng.poisson(0.5, n) * 1.0, "prb_util": rng.uniform(size=n),
                "energy_w_lag": rng.normal(size=n), "is_macro": (cell < 21) * 1.0}
        kind = {"own_carrier": "knob_own", "own_sleep": "knob_own", "own_ptx": "knob_own", "own_prot_min": "knob_own",
                "nbr_sleep": "knob_nbr", "nbr_ptx": "knob_nbr", "cio_out_mean": "knob_own", "energy_w": "kpi",
                "prot_viol": "kpi", "rlf": "kpi", "prb_util": "kpi", "energy_w_lag": "kpi_lag", "is_macro": "context"}
        fam = {"own_carrier": "carrier", "own_sleep": "sleep", "own_ptx": "ptx", "own_prot_min": "prot_min",
               "nbr_sleep": "sleep", "nbr_ptx": "ptx", "cio_out_mean": "cio", "energy_w": "energy_w",
               "prot_viol": "prot_viol", "rlf": "rlf", "prb_util": "prb_util", "energy_w_lag": "energy_w",
               "is_macro": "is_macro"}
        p = Panel(data=cols, kind=kind, family=fam, episode=np.full(n, e), t=t, cell=cell, step_s=10,
                  cell_region=np.arange(C), neighbours=[[(c + 1) % C] for c in range(C)],
                  ownership={"PowerES": {"ptx": {"req": 5, "applied": 5}}}, xapps=["PowerES"])
        recs.append({"panel": panel_to_rec(p)})
    return recs


def test_rowperm_maps_onto_the_edge_space():
    recs = _fake_panel_records(np.random.default_rng(0))
    rp = BD.mscr_rowperm(recs, n_perm=999)
    d = rp["declared"]
    assert d[("ptx", "own", "e")]["declared"] and d[("ptx", "own", "e")]["sign"] == 1
    assert all(d[h]["declared"] is None for h in ES.HYPOTHESES if h[1] == "far" or h[2] in ("v", "load"))
    assert rp["n_mapped"] + rp["n_unmapped"] == 60
    assert any(e["column"] == "cio_out_mean" for e in rp["unmapped_panel_edges"])


def test_pacifista_native_severity():
    recs = []
    for i, x in enumerate(("ES", "PowerES")):
        rs = CU.synthetic_records(np.random.default_rng(i), n_ep=1, units_per_ep=5, effect=0.0)
        for r in rs:
            r.update(sub=x, stage="prof", policy=f"profile:{x}")
        recs += rs
    out = BD.pacifista_native(recs)
    assert set(out["sigma"]) == {"ES|PowerES"} and 0.0 <= out["sigma"]["ES|PowerES"] <= 1.0
    assert set(out["per_kpi"]["ES|PowerES"]) == set(CU.KPIS)

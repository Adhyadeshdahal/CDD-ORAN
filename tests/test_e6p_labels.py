"""cdd_oran.decision.labels_p: accept label == direct accept-all rollout, contrasts, CRN reseeds, held mode, label
sampling, region probes. Short episodes and H = 20 s keep it fast (mechanisms, not values)."""
from __future__ import annotations

import numpy as np
import pytest

from cdd_oran.decision import collect_p as CP
from cdd_oran.decision import labels_p as LP
from cdd_oran.decision import units_p as UP
from cdd_oran.envs.e6 import config as C

H = 20


def cfg_p3(seed=7):
    return C.E6Config(seed=seed, mix="ES", scenario="surge", load_factor=1.0, n_ue=300, n_pico=3, mobility="ped",
                      warmup_s=60.0, scored_s=60.0,
                      e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "SliceGuarantee")))


@pytest.fixture(scope="module")
def labelled():
    """One collection episode; the first ES macro unit and the first SliceGuarantee unit get labels, direct AA
    rollouts, repeat rollouts and region probes."""
    lab = LP.Labeller(H=H)
    out = {}

    def hook(env, obs, snap, opened):
        for u in opened:
            want = ("es" if u["x"] == "ES" and u["knob"] == "carrier" else
                    "sg" if u["x"] == "SliceGuarantee" else "pico" if u["knob"] == "sleep" else None)
            if want is None or want in out:
                continue
            rec = {"unit": u, "label": lab.label(env, obs, snap, u),
                   "aa": [LP.aa_rollout(env, k, H) for k in LP.KS],
                   "cache": {k: lab.cache[(u["t0"], k)] for k in LP.KS}}
            rec["probe"] = lab.region_probe(env, obs, snap, u, env.plant.lay.cell_site, "reject", rec["label"])
            if want == "es":
                pol = UP.HoldPolicy({(u["c"], u["x"])}, "reject", u["t0"], u["t0"] + 1)
                rec["rep"] = [LP.rollout(env, obs, snap, pol, 1, H) for _ in range(2)]
                rec["rep2"] = LP.rollout(env, obs, snap, pol, 2, H)
                rec["held"] = held_decisions(env, obs, snap, u)
            out[want] = rec

    res = CP.run_collection(cfg_p3(), CP.RandomizedUnitPolicy(7, "high"), labeller=hook)
    assert {"es", "sg"} <= set(out), "config produced no ES carrier / SG unit"
    return out, res, lab


def held_decisions(env, obs, snap, u):
    """Decisions on (c, x)'s requests in a HoldPolicy(reject) rollout, by time."""
    sim = env.copy(reseed=1)
    arb = snap.fork(UP.HoldPolicy({(u["c"], u["x"])}, "reject", u["t0"], u["t0"] + 1))
    seen, o = [], obs
    for _ in range(int(UP.T_UNIT) + 15):
        d = arb(o)
        seen += [(o["t"], dd) for r, dd in zip(o["requests"], d["decisions"], strict=True)
                 if r["xapp"] == u["x"] and r["knob"][1] == u["c"]]
        others = [dd for r, dd in zip(o["requests"], d["decisions"], strict=True)
                  if not (r["xapp"] == u["x"] and r["knob"][1] == u["c"])]
        assert all(dd == "accept" for dd in others)       # every other unit accepts (AA continuation)
        sim.step_apply(d)
        if sim.sec >= sim.total_s:
            break
        o = sim.step_propose()
    return seen


def test_accept_label_equals_a_direct_accept_all_rollout(labelled):
    out, _, _ = labelled
    for rec in out.values():
        u, L = rec["unit"], rec["label"]
        for i, k in enumerate(LP.KS):
            aa, cached = rec["aa"][i], rec["cache"][k]
            assert np.array_equal(cached["net"], aa["net"]) and np.array_equal(cached["cell"], aa["cell"])
            assert L["raw"]["accept"][i]["net"] == [float(x) for x in aa["net"]]
            assert L["raw"]["accept"][i]["exp"] == [float(aa["cell"][j, u["exp"]].sum()) for j in range(3)]
            assert aa["secs"] == H
        assert all(v == [0.0, 0.0] for sc in L["d"]["accept"].values() for v in sc.values())


def test_contrasts_are_mode_minus_accept_per_reseed_and_mean_over_k(labelled):
    out, _, _ = labelled
    for rec in out.values():
        L = rec["label"]
        assert L["modes"] == list(LP.label_modes(rec["unit"]["x"])) and L["ks"] == [1, 2]
        for m in L["modes"]:
            for sc in ("exp", "net"):
                for j, kpi in enumerate(LP.KPIS):
                    per_k = [L["raw"][m][i][sc][j] - L["raw"]["accept"][i][sc][j] for i in range(2)]
                    assert L["d"][m][sc][kpi] == pytest.approx(per_k)
                    assert L["mean"][m][sc][kpi] == pytest.approx(np.mean(per_k))
        assert L["cpu_s"] > 0


def test_same_reseed_reproduces_and_different_reseeds_redraw(labelled):
    out, _, _ = labelled
    a, b, c = out["es"]["rep"][0], out["es"]["rep"][1], out["es"]["rep2"]
    assert np.array_equal(a["net"], b["net"]) and np.array_equal(a["cell"], b["cell"])
    assert not np.array_equal(a["cell"], c["cell"])
    L = out["es"]["label"]
    assert L["raw"]["reject"][0]["net"] == [float(x) for x in a["net"]]


def test_held_mode_governs_the_unit_for_T_then_accept(labelled):
    out, _, _ = labelled
    u, seen = out["es"]["unit"], out["es"]["held"]
    assert seen and seen[0][0] == u["t0"]
    for t, d in seen:
        assert d == ("reject" if t < u["t0"] + UP.T_UNIT else "accept")


def test_label_sampling_rates_and_tag():
    for x, rate in (("ES", 0.3), ("PowerES", 0.3), ("SliceGuarantee", 0.05)):
        assert LP.LABEL_RATE[x] == rate
        hits = [LP.sampled(11, {"c": c, "x": x, "x_idx": UP.x_index(x), "t0": float(t)})
                for c in range(24) for t in range(120, 720)]
        assert np.mean(hits) == pytest.approx(rate, abs=0.01)
    u = {"c": 3, "x": "ES", "x_idx": 0, "t0": 200.0}
    assert LP.sampled(11, u) == (np.random.default_rng([11, 6613, 3, 0, 200]).random() < 0.3)
    assert LP.label_modes("ES") == ("accept", "half", "reject") == LP.label_modes("PowerES")
    assert LP.label_modes("SliceGuarantee") == ("accept", "half", "reject", "accept+rb")


def test_region_probe_covers_the_units_site_and_sums_cell_contrasts(labelled):
    out, res, _ = labelled
    site = res["env"].plant.lay.cell_site
    pr = out["es"]["probe"]
    u = out["es"]["unit"]
    assert pr is not None and pr["cells"] == [q for q in range(len(site)) if site[q] == site[u["c"]]]
    assert len(pr["cells"]) == 3 and len(pr["dR"]) == 2
    for i in range(2):
        assert pr["sum_dC"][i] == pytest.approx(np.sum([pr["dC"][str(q)][i] for q in pr["cells"]], 0).tolist())
    assert pr["dC"][str(u["c"])][0] == [out["es"]["label"]["d"]["reject"]["net"][k][0] for k in LP.KPIS]
    if "pico" in out:
        assert out["pico"]["probe"] is None                # a pico region has one cell

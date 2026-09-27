"""Behavioural tests of the step-5 ranking evaluation harness (cdd_oran.decision.rank_eval): metric definitions,
episode-clustered bootstrap, abstention accounting, fit-split guards, and the oracle panel's continuation semantics
on a short real E6 episode."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from cdd_oran.decision import collect as CO
from cdd_oran.decision import rank_eval as RE
from cdd_oran.decision.trace import Trace
from cdd_oran.envs.e6.config import E6Config

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# ------------------------------------------------------------------------------------------------ pure metrics
def _panel(seed=0, n=12):
    J = np.random.default_rng(seed).normal(50.0, 10.0, n)
    J[0] = np.median(J)                                  # accept-all neither best nor worst
    return J


def test_perfect_model_has_zero_regret_and_rho_one_reversed_is_worse_than_accept_all():
    for s in range(5):
        J = _panel(s)
        perfect = RE.slot_metrics(J, J - J[0])
        assert perfect["regret"] == 0.0 == perfect["regret_gated"] and perfect["rho_raw"] == pytest.approx(1.0)
        assert perfect["improvement"] == pytest.approx(J[0] - J.min()) and perfect["deviate"]
        rev = RE.slot_metrics(J, -(J - J[0]))
        assert rev["rho_raw"] == pytest.approx(-1.0)
        assert rev["regret"] > rev["regret_accept_all"] > 0 and rev["improvement"] < 0
    flat = RE.slot_metrics(_panel(1), np.zeros(12))           # no signal -> ties -> accept-all (index 0)
    assert flat["pick"] == 0 and flat["improvement"] == 0.0 and np.isnan(flat["rho_raw"])


def test_support_ood_and_gate_abstentions_are_counted():
    J = np.array([10.0, 4.0, 6.0, 8.0, 12.0])
    pred = np.array([0.0, -6.0, -4.0, -2.0, 1.0])
    unid = np.array([False, True, False, False, False])       # the raw favourite is unidentified
    m = RE.slot_metrics(J, pred, unid)
    assert m["pick_raw"] == 1 and m["pick"] == 2 and m["support_abstain"] and m["n_unidentified"] == 1
    assert m["regret_raw"] == 0.0 and m["regret"] == 2.0 and m["regret_accept_all"] == 6.0
    ood = np.array([False, False, True, False, False])
    m = RE.slot_metrics(J, pred, unid, ood)
    assert m["pick"] == 2 and m["ood_abstain"] and m["pick_gated"] == 3 and m["improvement_gated"] == 2.0
    allowed = np.array([True, True, True, False, True])
    m = RE.slot_metrics(J, pred, unid, ood, allowed)
    assert m["pick_gated"] == 0 and not m["deviate_gated"] and m["improvement_gated"] == 0.0
    m = RE.slot_metrics(J, pred, unid, None, np.array([True, True, False, True, True]))
    assert m["gate_abstain"] and not m["ood_abstain"] and m["pick_gated"] == 3
    # accept-all can never be excluded, whatever the flags say
    m = RE.slot_metrics(J, pred, np.ones(5, bool), np.ones(5, bool), np.zeros(5, bool))
    assert m["pick"] == 0 == m["pick_gated"]


def test_bootstrap_resamples_whole_episodes():
    rng = np.random.default_rng(3)
    ep = np.repeat(np.arange(6), 40)
    v = np.repeat(rng.normal(0, 5, 6), 40) + rng.normal(0, 0.1, len(ep))  # variation is between episodes
    clus = RE.cluster_bootstrap(v, ep, n_boot=1000)
    naive = RE.cluster_bootstrap(v, np.arange(len(v)), n_boot=1000)
    assert clus["n_clusters"] == 6 and clus["n"] == 240 and clus["mean"] == pytest.approx(v.mean())
    assert (clus["hi"] - clus["lo"]) > 4 * (naive["hi"] - naive["lo"])
    means = np.array([v[ep == e].mean() for e in range(6)])
    assert means.min() - 1e-9 <= clus["lo"] <= clus["hi"] <= means.max() + 1e-9
    v2 = v.copy()
    v2[:5] = np.nan
    assert RE.cluster_bootstrap(v2, ep)["n"] == 235


def test_summarize_clusters_by_episode_and_counts_abstentions():
    rows = []
    for seed in (1, 2, 3):
        for k in range(4):
            J = _panel(seed * 10 + k, 6)
            unid = np.array([False, True, False, False, False, False])
            m = RE.slot_metrics(J, J - J[0], unid)
            g = {c: 0.0 for c in RE.GUARDRAILS}
            rows.append({"type": "slot", "seed": seed, "scenario": "base", "load": "medium",
                         "models": {"a": {"metrics": m, "guard": g, "guard_gated": g, "search_pick": m["pick"]}}})
    s = RE.summarize(rows, n_boot=200)["a"]
    p = s["pooled"]
    assert p["n_slots"] == 12 and p["n_episodes"] == 3 and p["regret"]["n_clusters"] == 3
    assert p["support_abstain"] == sum(r["models"]["a"]["metrics"]["support_abstain"] for r in rows)
    assert p["search_pick_is_panel_argmin"] == 12 and "base-medium" in s["strata"]
    assert set(p["step5_criterion"]) == {"improvement_gated_lo_gt_0", "guardrails_credibly_worse", "met"}


# ------------------------------------------------------------------------------------------------ real E6 (short)
H = 20


@pytest.fixture(scope="module")
def fit_paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("v3")
    paths = []
    for s in (17, 18):
        cfg = E6Config(seed=s, load="medium", mobility="mixed", mix="M4", warmup_s=60, scored_s=120)
        p = d / f"v3_s{s}.npz"
        CO.collect_episode_v3(cfg, H=H).to_npz(p)
        paths.append(p)
    return paths


def test_fit_from_npz_guards_the_fit_split_and_builds_selectors(fit_paths, tmp_path):
    tr = Trace.from_npz(fit_paths[0])
    with pytest.raises(ValueError):                          # seed 17 is not a registered v3 FIT seed
        RE.fit_from_npz(fit_paths, H=H)
    diag = Trace(dict(tr.arrays), copy.deepcopy(tr.meta))
    diag.meta["cfg"]["seed"] = CO.dev_seed("policy", "base", "medium", 25)
    diag.to_npz(tmp_path / "diag.npz")
    with pytest.raises(ValueError, match="FIT"):
        RE.fit_from_npz([tmp_path / "diag.npz"], H=H)
    ref = Trace(dict(tr.arrays), copy.deepcopy(tr.meta))
    ref.meta["collector"]["mixture"]["p_accept"] = 1.0
    ref.to_npz(tmp_path / "ref.npz")
    with pytest.raises(ValueError, match="reference"):
        RE.fit_from_npz([tmp_path / "ref.npz"], H=H, require_fit=False)
    f = RE.fit_from_npz(fit_paths, H=H, selector="topology", n_members=2, require_fit=False)
    W = f.model.selector.weights
    assert W.shape == (10, 10) and np.allclose(W, W.T) and W.max() <= 1.0 and np.all(np.diag(W) == 0)
    assert f.model.weighting_used == "none" and f.manifest["n_episodes"] == 2 and f.manifest["selector"] == "topology"
    assert f.manifest["n_rows"] == 2 * 2 * 10                  # ITT: every labelled slot x region
    assert ("base", "medium", "TS", "reject") in f.counts


def test_oracle_panel_scores_accept_all_as_the_realised_window_and_records_everything(fit_paths):
    fitted = {n: RE.fit_from_npz(fit_paths, H=H, selector=n, n_members=2, require_fit=False) for n in ("all", "none")}
    cfg = E6Config(seed=16, load="medium", mobility="mixed", mix="M4", warmup_s=60, scored_s=60)
    spec = RE.CandidateSpec(top_k=2, n_random=2, single_modes=("reject",))
    rows = RE.oracle_panel(cfg, [60, 90], spec, fitted, D=20, H=H, lam_e=1e4, w_ll=5.0)
    assert [r["t"] for r in rows] == [60, 90]
    for r in rows:
        orc, real = r["oracle"], r["realized_accept_all"]
        assert r["cands"][0]["src"][0] == "accept_all"
        assert orc["J"][0] == real["J"]                       # continuation semantics: exact equality
        assert all(orc[c][0] == real[c] for c in RE.ORACLE_COMPONENTS)
        assert len({tuple(c["codes"]) for c in r["cands"]}) == len(r["cands"])         # deduplicated
        for name, m in r["models"].items():
            met = m["metrics"]
            assert m["pred"][0] == 0.0 and not m["ood"][0] and not m["unid"][0]
            assert any(f"search:{name}" in s for s in (c["src"] for c in r["cands"]))
            assert f"search:{name}" in r["cands"][m["search_pick"]]["src"]
            assert met["regret_accept_all"] == pytest.approx(orc["J"][0] - min(orc["J"]))
            # 2 short fit episodes identify nothing under SupportRule(20, 20, 10): every deviation is excluded
            assert all(m["unid"][1:]) and met["pick"] == 0 and met["improvement"] == 0.0
            assert met["support_abstain"] == (met["pick_raw"] != 0)
            assert np.shape(m["pred_comp"]) == (len(r["cands"]), 6)
    s = RE.summarize(rows, n_boot=100)
    assert s["all"]["pooled"]["n_slots"] == 2 and s["all"]["pooled"]["n_episodes"] == 1
    assert s["none"]["pooled"]["support_abstain"] == sum(r["models"]["none"]["metrics"]["support_abstain"]
                                                         for r in rows)
    ungated = RE.oracle_panel(cfg, [60], spec, {"all": fitted["all"]}, support_rule=None, D=20, H=H)
    m = ungated[0]["models"]["all"]
    assert not any(m["unid"]) and m["metrics"]["pick"] == m["metrics"]["pick_raw"]
    assert ungated[0]["oracle"]["J"][0] != 0.0

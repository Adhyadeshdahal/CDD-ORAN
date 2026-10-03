"""Tests of the Study A EVAL analysis (scratchpad/xmethod/eval_analysis.py; docs/xmethod/PROTOCOL_A.md)."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import random

import numpy as np
import pytest

from cdd_oran.xmethod import api
from cdd_oran.xmethod import runner as R
from cdd_oran.xmethod.score import PLACEBO, PLACEBO_CONF
from cdd_oran.xmethod.worlds import truth_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("eval_analysis",
                                               os.path.join(ROOT, "scratchpad", "xmethod", "eval_analysis.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

FIXTURE = os.path.join(ROOT, "tests", "data", "xmethod_eval_pilot_subset.jsonl.gz")
TUNE, MEAS = list(range(3_000_000, 3_000_020)), list(range(3_000_100, 3_000_112))
PSHA, SSHA, COMMIT = "p" * 64, "s" * 64, "abc"
PKGS = {"numpy": "2.0.2"}
CI = "cdd_oran.xmethod.methods.ci:CI"

# ------------------------------------------------------------------------------------------- synthetic study
SPEC = {
    "name": "test_synth",
    "protocol_sha256": PSHA,
    "budget_cpu_s": 7200,
    "arms": {
        "pmrt_eq": {"ref": "cdd_oran.xmethod.methods.pmrt_core:PmrtCore", "config": {"covariates": "eq"},
                    "declare": "by"},
        "corr": {"ref": "cdd_oran.xmethod.methods.corr:Corr", "config": {"arm": "native"}, "declare": "by",
                 "analysis": "primary"},
        "ci_eq": {"ref": CI, "config": {"arm": "eq"}, "declare": "by", "native_partner": "ci_native"},
        "ci_native": {"ref": CI, "config": {"arm": "native"}, "declare": "by"},
        "pcorr_hac": {"ref": "cdd_oran.xmethod.methods.hac:HAC", "config": {"arm": "native"}, "declare": "by"},
        "pc_eq": {"ref": "cdd_oran.xmethod.methods.pc:PC", "config": {"arm": "eq"}, "declare": "tau"},
        "pc_native": {"ref": "cdd_oran.xmethod.methods.pc:PC", "config": {"arm": "native"}, "declare": "tau"},
    },
    "blocks": [
        {"role": "tune", "worlds": ["E1", "E4"], "regimes": ["R1", "R2", "R3"], "ns": [500, 1000], "kappas": [0.25],
         "seeds": TUNE, "e4_lams": {"R1": [], "R2": [], "R3": [1.0]}},
        {"role": "measure", "worlds": ["E1", "E4"], "regimes": ["R1", "R2", "R3"], "ns": [500, 1000],
         "kappas": [0.25], "seeds": MEAS, "e4_lams": {"R1": [], "R2": [], "R3": [1.0]}},
    ],
}
LIBERAL = {("corr", "R2"), ("ci_native", "R2")}       # design-blind failures in R2; pcorr_hac stays valid


def _record(arm, world, regime, lam, n, seed, role, liberal, commit=COMMIT):
    """One campaign-format record; nulls (incl. placebos) get p ~ U(.06, 1), or with ``liberal`` a 40 % chance of
    p ~ U(0, .01); true edges p = 1e-5. score = 1 - p; BY declarations over the primary family."""
    truth = truth_for(world, regime)
    rng = random.Random(f"{arm}|{world}|{regime}|{lam}|{n}|{seed}")
    edges = []
    for s, t in sorted(truth.edges | truth.null_edges):
        if (s, t) in truth.edges:
            p = 1e-5
        elif liberal and rng.random() < 0.4:
            p = rng.uniform(0, 0.01)
        else:
            p = rng.uniform(0.06, 1.0)
        edges.append({"source": s, "target": t, "score": 1 - p, "p": p, "sign": truth.signs.get((s, t), 1),
                      "declared": False})
    fam = [e for e in edges if not e["source"].startswith("K") and e["source"] != PLACEBO_CONF]
    for e, d in zip(fam, E.by_declare([e["p"] for e in fam]), strict=True):
        e["declared"] = d
    return {"key": R.job_key(arm, world, regime, lam, n, seed, 0.25), "method": arm, "arm": arm, "role": role,
            "status": "ok", "error": None, "job": {"world": world, "regime": regime, "lam": lam, "n": n,
                                                   "seed": seed, "kappa": 0.25},
            "edges": edges, "notes": {}, "config": {}, "cpu_s": 0.1, "peak_rss_mb": 10.0,
            "dataset_sha256": f"{world}|{regime}|{lam}|{n}|{seed}", "code": {"commit": commit, "dirty": False},
            "run_mode": {"mode": "eval", "protocol_sha256": PSHA, "spec_sha256": SSHA}, "pkgs": PKGS,
            "host": {"platform": "kaggle"}}


def synth_records(spec=SPEC):
    return [_record(u["arm"], u["world"], u["regime"], u["lam"], u["n"], u["seed"], u["role"],
                    (u["arm"], u["regime"]) in LIBERAL) for u in E.expected_units(spec).values()]


def run(recs, spec=SPEC, **kw):
    kw = {"spec_sha": SSHA, "protocol": {"ok": True}, **kw}
    return E.analyse(sorted(recs, key=lambda r: r["key"]), spec, COMMIT, **kw)


@pytest.fixture(scope="module")
def synth():
    recs = synth_records()
    return recs, run(recs)


# ------------------------------------------------------------------------------------------- unit pieces
def test_by_declare_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    for m in (1, 5, 20, 54):
        ps = list(np.concatenate([rng.uniform(0, 1e-3, m // 4), rng.uniform(0, 1, m - m // 4)]))
        assert E.by_declare(ps) == list(sm.multipletests(ps, 0.05, "fdr_by")[0])
    assert sum(E.by_declare([0.001, 0.004, 0.02, 0.5], m=40)) <= sum(E.by_declare([0.001, 0.004, 0.02, 0.5]))
    assert E.by_declare([], m=3) == []


def _res(scores):
    return api.Result("m", "v", tuple(api.EdgeResult(PLACEBO, f"K{i}", s, None, 0, False)
                                      for i, s in enumerate(scores)), 0.0, {})


def test_conformal_tau_rule_and_level():
    assert E.conformal_tau([_res(list(range(20)))]) == 19.0          # M = 20: ceil(21 * .95) = 20 -> max
    assert E.conformal_tau([_res(list(range(80)))]) == 76.0          # M = 80: ceil(81 * .95) = 77th smallest
    assert E.conformal_tau([_res(list(range(19)))]) == 18.0          # index 19 <= 19
    assert E.conformal_tau([_res([math.nan])]) == math.inf and E._conformal_tau_fallback([]) == math.inf
    rng = np.random.default_rng(3)                                   # out-of-sample level <= .05
    hits = 0
    for _ in range(4000):
        s = rng.uniform(size=41)
        hits += s[40] > E.conformal_tau([_res(list(s[:40]))])
    assert hits / 4000 <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 4000)


def test_three_way_validity():
    assert E.classify([0.051, 0.07]) == "INVALID"                    # INVALID takes precedence
    assert E.classify([0.03, 0.075]) == "VALID"
    assert E.classify([0.03, 0.08]) == "INCONCLUSIVE"
    assert E.combine(["VALID", "INCONCLUSIVE"]) == "INCONCLUSIVE"
    assert E.combine(["VALID", None, "INVALID"]) == "INVALID" and E.combine([None]) == "NO_READ"
    r = E.rate([1, 0, 2], [10, 10, 10])
    assert r["hits"] == 3 and r["n"] == 30 and r["ci"][0] <= 0.1 <= r["ci"][1] and r["validity"] == "INCONCLUSIVE"
    assert E.rate([0], [0]) is None


def test_simulation_replicates_the_analysis_bootstrap():
    rng = np.random.default_rng(0)
    for k in (12, 40):
        h = rng.integers(0, 4, k).astype(float)
        bs = (h @ E._boot_mult(k).T) / (16 * k)
        assert E.cluster_ci(h, np.full(k, 16.0)) == [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def test_fmax_simulation_properties():
    cells = [(w, "R1", 1.0 if w == "E4" else None, n) for w in ("E1", "E4") for n in (500, 1000)]
    a = E.fmax_simulate(cells, 12, ("null_raw", "plac_raw"), None, nsim=400)
    b = E.fmax_simulate(list(reversed(cells)), 12, ("null_raw", "plac_raw"), None, nsim=400)
    assert a == b and a["n_cells"] == 4 and a["p_count_le_fmax"] >= 0.95
    assert sum(a["count_dist"].values()) == 400 and 0 <= a["p_cell_invalid"] < 0.15
    big = E.fmax_simulate([("E1", "R1", None, n) for n in (500, 1000, 4000, 8000, 24000)] * 1, 40,
                          ("null_raw", "plac_raw"), None, nsim=400)
    assert big["f_max"] >= 0 and 0.5 < big["p_pooled_valid"] <= 1.0


# ------------------------------------------------------------------------------------------- synthetic study
def test_expected_units_match_campaign_expand():
    campaign = pytest.importorskip("cdd_oran.xmethod.campaign")
    assert set(E.expected_units(SPEC)) == {u.key for u in campaign.expand(SPEC)}


def test_verdicts_per_arm_and_wording(synth):
    _, out = synth
    val = {r["key"]: r["validity"] for r in out["V1_validity"]}
    assert all(v == "INVALID" for k, v in val.items() if k.startswith(("corr|", "ci_native|")) and "|R2|" in k)
    assert all(v == "VALID" for k, v in val.items() if k.startswith(("pmrt_eq|", "ci_eq|", "pcorr_hac|")))
    v = out["V4_verdicts"]
    c1 = v["C1"]
    assert c1["D"] == ["ci_native", "corr", "pcorr_hac"] and c1["verdict"] == "SUPPORTED"
    assert c1["arms"]["corr"]["verdict"] == "FAILURE" and c1["arms"]["pcorr_hac"]["verdict"] == "NOT A FAILURE"
    assert "i.i.d." in c1["wording"] and "pcorr_hac" in c1["wording"]
    assert v["C2a"]["verdict"] == "SUPPORTED" and v["C2a"]["n_cells"] == 4
    assert list(v["C2b"]["arms"]) == ["ci_eq"] and v["C2b"]["verdict"] == "SUPPORTED"
    assert v["C2_wording"].startswith("using the design restores validity")
    assert v["C3"]["verdict"] == "SUPPORTED" and set(v["C3"]["pooled"]) == {"plac_raw", "conf_raw"}
    assert set(v["C3_eq_arms"]) == {"ci_eq", "pc_eq"} and "not specific to PMRT" in v["C3"]["wording"]
    assert v["claim"] == "SUPPORTED"
    v11 = out["V11_integrity"]
    assert v11["label"] == "FINAL", v11["failed"]
    assert v11["by_recheck"]["agree"] == v11["by_recheck"]["records"]


def test_c1_needs_three_arms_and_fallback_wordings():
    spec = copy.deepcopy(SPEC)
    del spec["arms"]["pcorr_hac"]
    out = run(synth_records(spec), spec)
    assert out["V4_verdicts"]["C1"]["verdict"] == "NOT EVALUABLE"            # |D_counted| = 2 < 3
    recs = synth_records()
    for r in recs:                                                            # ci_eq liberal in R2 too
        if r["arm"] == "ci_eq" and r["job"]["regime"] == "R2" and r["role"] == "measure":
            r.update(_record("ci_eq", r["job"]["world"], "R2", r["job"]["lam"], r["job"]["n"], r["job"]["seed"],
                             "measure", True))
    v = run(recs)["V4_verdicts"]
    assert v["C2b"]["verdict"] == "NOT SUPPORTED" and v["C2a"]["verdict"] == "SUPPORTED"
    assert v["C2_wording"].startswith("design-based inference restores validity; adding design covariates")
    assert v["claim"] == "NOT SUPPORTED"


def test_same_rule_power_comparison(synth):
    _, out = synth
    rules = {(d["arm"], d["rule"]) for d in out["V3_paired"]}
    assert not any(d["arm"] == "corr" and d["regime"] == "R2" for d in out["V3_paired"])   # INVALID in R2 ...
    assert ("corr", "BY") in rules                                                          # ... compared in R1
    assert ("pc_eq", "tau") in rules and ("ci_eq", "BY") in rules
    assert all(d["rule"] == ("tau" if d["arm"].startswith("pc_") else "BY") for d in out["V3_paired"])
    assert len(out["V3_paired"]) >= len(out["V3_paired_valid_only"])          # R-39: not-INVALID gate


def test_tau_from_tune_records_only(synth):
    recs, _ = synth
    use, _ = E.screen(recs, SPEC)
    cells, _ = E.build_cells(use, SPEC)
    ck = "pc_eq|" + R.cell_key("E1", "R2", None, 1000, 0.25)
    tune = [E.to_result(r) for r in recs if r["role"] == "tune" and r["key"].startswith(ck + "|")]
    assert len(tune) == len(TUNE) and cells[ck]["tau"] == E.conformal_tau(tune)
    assert cells[ck]["n_measure"] == len(MEAS)


def test_missing_infeasible_and_under_seeded_cells_cap_verdicts():
    recs = [r for r in synth_records()
            if not (r["arm"] == "pc_native" and r["role"] == "tune" and r["job"]["regime"] == "R1")]
    bad = next(r for r in recs if r["arm"] == "corr" and r["role"] == "measure" and r["job"]["regime"] == "R2")
    bad.update(status="infeasible", edges=[], cpu_s=15000.0, reason="exceeded the safety cap")
    drop = next(r for r in recs if r["arm"] == "pmrt_eq" and r["role"] == "measure" and r["job"]["regime"] == "R1")
    recs.remove(drop)
    out = run(recs)
    st = {r["key"]: r["status"] for r in out["V1_validity"]}
    assert all(s == "untuned" for k, s in st.items() if k.startswith("pc_native|") and "|R1|" in k)
    assert st[bad["key"].rsplit("|", 1)[0]] == "infeasible"
    assert st[drop["key"].rsplit("|", 1)[0]] == "under_seeded"
    v = out["V4_verdicts"]
    c = v["C1"]["arms"]["corr"]
    assert (c["r2_counted"], c["r2_planned"], c["verdict"]) == (1, 2, "NOT EVALUABLE")   # < 90 % counted
    assert v["C1"]["D_counted"] == ["ci_native", "pcorr_hac"] and v["C1"]["verdict"] == "NOT EVALUABLE"
    assert v["C2a"]["verdict"] == "PARTIAL" and v["C2a"]["under_seeded"]
    assert out["V11_integrity"]["label"] == "PROVISIONAL"
    # an under-seeded (still counted) cell of a D arm keeps the arm's verdict but caps C1 at PARTIAL
    recs = synth_records()
    recs.remove(next(r for r in recs if r["arm"] == "corr" and r["role"] == "measure" and r["job"]["regime"] == "R1"))
    v = run(recs)["V4_verdicts"]
    assert v["C1"]["arms"]["corr"]["verdict"] == "FAILURE" and not v["C1"]["arms"]["corr"]["complete"]
    assert v["C1"]["verdict"] == "PARTIAL" and v["claim"] == "NOT SUPPORTED"


def test_t3_cells_are_shown_and_not_planned():
    spec = copy.deepcopy(SPEC)
    spec["arms"]["ci_native"].update(max_n=500, t3_cost_cpu_s={"1000": 9100.0})
    recs = synth_records(spec)
    assert not any(r["arm"] == "ci_native" and r["job"]["n"] == 1000 for r in recs)
    out = run(recs, spec)
    t3 = [r for r in out["V1_validity"] if r["status"] == "infeasible_t3"]
    assert t3 and all(r["arm"] == "ci_native" and r["n"] == 1000 and r["infeasible_cost_cpu_s"] == 9100.0
                      for r in t3)
    assert out["V4_verdicts"]["C1"]["arms"]["ci_native"]["r2_planned"] == 1
    assert any(r["t3"] for r in out["V10_cost"]) and out["V11_integrity"]["label"] == "FINAL"


def test_integrity_holes_make_output_provisional():
    base = synth_records()
    cases = {}
    recs = copy.deepcopy(base) + [_record("corr", "E1", "R1", None, 500, 3_000_150, "measure", False)]
    cases["unexpected"] = recs
    recs = copy.deepcopy(base)
    next(r for r in recs if r["role"] == "measure")["role"] = "tune"
    cases["role"] = recs
    recs = copy.deepcopy(base)
    recs[5]["edges"] = recs[5]["edges"][1:]
    cases["candidates_complete"] = recs
    recs = copy.deepcopy(base)
    recs[7]["code"]["dirty"] = None
    cases["clean"] = recs
    recs = copy.deepcopy(base)
    recs.append(copy.deepcopy(recs[3]))
    cases["duplicates"] = recs
    recs = copy.deepcopy(base)
    recs[9]["run_mode"]["protocol_sha256"] = "q" * 64
    cases["stamps"] = recs
    recs = copy.deepcopy(base)
    recs[11]["host"]["platform"] = "colab"
    cases["one_platform_per_dataset"] = recs
    recs = copy.deepcopy(base)
    recs[13]["pkgs"] = {"numpy": "2.1.0"}
    cases["pkgs_uniform"] = recs
    recs = copy.deepcopy(base)
    recs[15].update(status="error", edges=[], error="boom")
    cases["errors_listed"] = recs
    recs = copy.deepcopy(base)
    recs[17]["code"]["commit"] = "def"
    cases["commits"] = recs
    for check, rs in cases.items():
        v11 = run(rs)["V11_integrity"]
        assert v11["label"] == "PROVISIONAL" and check in v11["failed"], (check, v11["failed"])
    out = run(copy.deepcopy(base) + [_record("corr", "E1", "R1", None, 500, 3_000_150, "measure", True)])
    assert out == run(base) | {"V11_integrity": out["V11_integrity"]}        # the stray record changes no table
    # an amendment re-run and a listed persistent error are allowed (sections 11, 14)
    recs = copy.deepcopy(cases["commits"])
    am = {"amendments": [{"id": "A1", "commit": "def", "keys": [recs[17]["key"]]}],
          "persistent_errors": [{"key": cases["errors_listed"][15]["key"], "why": "OOM on every re-run"}]}
    assert "commits" not in run(recs, amendments=am)["V11_integrity"]["failed"]
    v11 = run(copy.deepcopy(cases["errors_listed"]), amendments=am)["V11_integrity"]
    assert "errors_listed" not in v11["failed"]


def test_deterministic_and_order_free(synth):
    recs, out = synth
    shuffled = list(recs)
    random.Random(7).shuffle(shuffled)
    assert json.dumps(E._clean(out), sort_keys=True) == json.dumps(E._clean(run(shuffled)), sort_keys=True)


def test_cli_writes_tables_and_guards(tmp_path):
    recs = synth_records()
    merged = tmp_path / "m.jsonl"
    merged.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    proto = tmp_path / "PROTOCOL_A.md"
    proto.write_text("# P\n\nFROZEN: yes (test)\n", encoding="utf-8")
    spec_d = {**SPEC, "protocol_sha256": E.file_sha256(str(proto))}
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(spec_d), encoding="utf-8")
    ssha = E.file_sha256(str(spec))
    for r in recs:
        r["run_mode"] = {"mode": "eval", "protocol_sha256": spec_d["protocol_sha256"], "spec_sha256": ssha}
    merged.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    common = ["--spec", str(spec), "--merged", str(merged), "--protocol", str(proto), "--dependence", ""]
    assert E.main([*common, "--out", str(tmp_path / "o"), "--freeze-commit", COMMIT]) == 0
    md = (tmp_path / "o" / "EVAL_TABLES.md").read_text(encoding="utf-8")
    assert md.startswith("# Study A EVAL tables (FINAL)") and "Claim: SUPPORTED" in md
    h1 = (tmp_path / "o" / "eval_tables.json").read_bytes()
    assert E.main([*common, "--out", str(tmp_path / "o2"), "--freeze-commit", "zzz"]) == 2
    E.main([*common, "--out", str(tmp_path / "o3"), "--freeze-commit", COMMIT])
    assert (tmp_path / "o3" / "eval_tables.json").read_bytes() == h1
    proto.write_text("# P\n\nFROZEN: no\n", encoding="utf-8")                  # unfrozen protocol -> PROVISIONAL
    assert E.main([*common, "--out", str(tmp_path / "o4"), "--freeze-commit", COMMIT]) == 2
    spec.write_text(json.dumps({**spec_d, "name": "eval_full"}), encoding="utf-8")
    for extra in (["--as-tune", "3000100"], ["--min-flag-seeds", "1"]):
        with pytest.raises(SystemExit):
            E.main([*common, "--out", str(tmp_path / "o5"), *extra])


def test_placeholder_seeds_are_reported_not_crashing():
    spec = copy.deepcopy(SPEC)
    spec["blocks"][1]["seeds"] = "TBD"
    with pytest.raises(ValueError):
        E.expected_units(spec)
    out = run(synth_records(), spec)
    assert out["V11_integrity"]["label"] == "PROVISIONAL"
    assert "placeholder" in out["V11_integrity"]["missing"][0]


def test_t1_seed_count_on_records():
    recs = synth_records()
    res = E.t1_seed_count(recs, SPEC)
    assert res["n_pairs"] > 0 and res["S"] == 40 and not res["cap_binds"]     # identical recalls -> floor
    assert all(p["cell"].split("|")[0] != "E4" for p in res["largest"])
    assert E.paired_seeds(0.3) > E.paired_seeds(0.1) and E.paired_seeds(0.0) == 2


# ------------------------------------------------------------------------------------------- DEV pilot regression
def test_pilot_subset_matches_campaign_aggregate():
    """Raw / declared counts, recall and FDP equal dev-runs' campaign.aggregate on the same pilot records
    (results/dev/pilot/agg.json at xm/dev-runs d1e6c4e)."""
    recs = E.load_records(FIXTURE)
    spec = {"name": "dev_pilot", "arms": {
        "pmrt_eq": SPEC["arms"]["pmrt_eq"], "corr": SPEC["arms"]["corr"],
        "granger_native": {"ref": "cdd_oran.xmethod.methods.granger:Granger", "config": {"arm": "native"},
                           "declare": "by", "worlds": ["E3"]}}, "blocks": []}
    cells, _ = E.build_cells(recs, spec)
    want = {  # cell: (null_decl, plac_decl, null_raw, plac_raw, recall, fdp) as (hits, n) / mean
        "pmrt_eq|E2|R2|k0.25|n1000": ((0, 64), (0, 12), (3, 64), (0, 12), 0.3125, 0.0),
        "corr|E3|R2|k0.25|n1000": ((26, 32), (7, 10), (30, 32), (7, 10), 1.0, 0.7647058823529411),
        "granger_native|E3|R2|k0.25|n1000": ((18, 32), (4, 10), (21, 32), (5, 10), 1.0, 0.6923076923076923),
        "pmrt_eq|E4|R3|lam1|k0.25|n1000": (None, (0, 2), None, (0, 2), 1.0, 0.0),
    }
    for ck, (nd, pd, nr, pr, rec, fdp) in want.items():
        p = cells[ck]["primary"]
        for k, w in (("null_decl", nd), ("plac_decl", pd), ("null_raw", nr), ("plac_raw", pr)):
            assert (None if p[k] is None else (p[k]["hits"], p[k]["n"])) == w, (ck, k)
        assert p["recall"]["mean"] == pytest.approx(rec) and p["fdp"]["mean"] == pytest.approx(fdp)
        assert p["validity"] == "few_seeds" and cells[ck]["status"] == "few_seeds"
    assert cells["pmrt_eq|E4|R3|lam1|k0.25|n1000"]["primary"]["conf_raw"]["n"] == 2
    scr = {"expected": None, "expected_error": "pilot", "duplicates": [], "unexpected": [], "role_mismatch": []}
    v11 = E.integrity(recs, scr, spec, None)
    assert v11["by_recheck"]["agree"] == len(recs) and v11["n_candidates_incomplete"] == 0


def test_as_tune_relabels_records_and_spec_blocks():
    recs = synth_records()
    rl, spec = E.relabel_tune(recs, {MEAS[0]}, SPEC)
    use, scr = E.screen(rl, spec)
    assert not scr["role_mismatch"] and not scr["unexpected"] and len(use) == len(recs)
    assert all(r["role"] == "tune" for r in rl if r["job"]["seed"] == MEAS[0])
    with pytest.raises(SystemExit):
        E.relabel_tune(recs, {MEAS[0]}, {**SPEC, "name": "eval_full"})


def test_conformal_tau_delegates_to_shared_score_placebo_tau(monkeypatch):
    calls = []

    def shared(results, max_declarations=None, alpha=0.05):
        calls.append(alpha)
        return E._conformal_tau_fallback(results, alpha)
    monkeypatch.setattr(E, "placebo_tau", shared)
    assert E.conformal_tau([_res(list(range(80)))]) == 76.0 and calls == [0.05]


def test_set_d_flag_excludes_the_unchosen_hac_variant():
    spec = copy.deepcopy(SPEC)
    spec["arms"]["pcorr_hac"]["set_D"] = "TBD-T8"
    v = run(synth_records(spec), spec)["V4_verdicts"]["C1"]
    assert v["D"] == ["ci_native", "corr"] and v["verdict"] == "NOT EVALUABLE"
    assert list(v["descriptive"]) == ["pcorr_hac"] and v["descriptive"]["pcorr_hac"]["verdict"] == "NOT A FAILURE"


def test_arm_out_of_d_is_descriptive_and_drops_its_eq_partner_from_c2b():
    spec = copy.deepcopy(SPEC)                         # Q11 / R-40: a native arm with set_D false
    spec["arms"]["ci_native"]["set_D"] = False
    out = run(synth_records(spec), spec)
    v = out["V4_verdicts"]
    assert v["C1"]["D"] == ["corr", "pcorr_hac"] and v["C1"]["verdict"] == "NOT EVALUABLE"   # |D| < 3 here
    assert v["C1"]["descriptive"]["ci_native"]["verdict"] == "FAILURE"
    assert "ci_eq" not in v["C2b"]["arms"]
    assert "ci_native (not in D, descriptive): **FAILURE**" in E.markdown(out)


def test_tau_pos_inf_declares_nothing():
    recs = synth_records()
    for r in recs:                                    # pc_eq E1 R1 n 500: no finite tune placebo score
        if r["arm"] == "pc_eq" and r["role"] == "tune" and r["job"]["regime"] == "R1" and r["job"]["n"] == 500 \
                and r["job"]["world"] == "E1":
            for e in r["edges"]:
                if e["source"] == PLACEBO:
                    e["score"] = None
    row = next(x for x in run(recs)["V1_validity"] if x["key"] == "pc_eq|" + R.cell_key("E1", "R1", None, 500, 0.25))
    assert row["tau_is_pos_inf"] and row["tau"] is None and row["status"] == "ok"
    assert row["null_decl"]["rate"] == 0.0 and row["plac_decl"]["rate"] == 0.0


def test_c2b_exclusion_and_labels_r40():
    spec = copy.deepcopy(SPEC)
    lab = "single-conditioner max statistic; cannot condition on the joint design set"
    spec["arms"]["ci_eq"].update({"c2b": False, "label": lab})
    spec["arms"]["ci_native"]["label"] = "dependence test"
    out = run(synth_records(spec), spec)
    v = out["V4_verdicts"]
    assert v["C2b"]["arms"] == {} and v["C2b"]["verdict"] == "NOT EVALUABLE"       # ci_eq was the only member
    x = v["C2b"]["excluded"]["ci_eq"]
    assert x["label"] == lab and x["native"] == "ci_native" and x["verdict"] == "SUPPORTED"
    assert v["claim"] == "NOT SUPPORTED" and "ci_eq" in v["C3_eq_arms"]               # C3 report unchanged
    assert v["arm_labels"] == {"ci_eq": lab, "ci_native": "dependence test"}
    md = E.markdown(out)
    assert "ci_eq EXCLUDED from C2b (R-40) [" + lab in md and "ci_native [dependence test]: **FAILURE**" in md


def test_eval_spec_r40():
    with open(os.path.join(ROOT, "scratchpad", "xmethod", "specs", "eval", "full.json"), encoding="utf-8") as fh:
        arms = json.load(fh)["arms"]
    assert [a for a, d in arms.items() if d.get("c2b", True) is not True] == ["mscr_eq"]
    assert arms["mscr_eq"]["label"].startswith("single-conditioner max statistic")
    assert arms["mscr_eq_min"]["label"] == arms["mscr_eq"]["label"]                      # Q11
    assert sorted(a for a, d in arms.items() if E.arm_kind(d) == "native" and d["declare"] == "by"
                  and d.get("set_D", True) is True) == ["corr", "granger_native", "mscr_native", "pcorr_hac_fb",
                                                        "pcorr_native", "rcot2_native"]
    assert [a for a, d in arms.items() if d.get("set_D", True) is not True] == ["pcorr_hac"]  # T8; pdcor gone (R-48)


def test_eval_spec_and_protocol_drop_pdcor_cmi_knn_r48_r49():
    with open(os.path.join(ROOT, "scratchpad", "xmethod", "specs", "eval", "full.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    assert len(spec["arms"]) == 25
    for gone in ("pdcor", "cmi_knn"):                                                    # R-48, R-49 revised
        assert not [a for a, d in spec["arms"].items() if gone in a or gone in d["ref"]]
        assert not [a for b in spec["blocks"] if isinstance(b.get("arms"), list) for a in b["arms"] if gone in a]
        assert gone not in spec["tbd"]["citests_arms"] and spec["tbd"][gone].startswith("dropped from Study A")
    assert not [a for a, d in spec["arms"].items() if "dependence test" in d.get("label", "")]
    assert not [a for a, d in spec["arms"].items() if d.get("budget_wall_s")]           # no GPU arm planned
    assert spec["tbd"]["large_n_citests"].startswith("OPEN (R-47")                     # T11
    with open(os.path.join(ROOT, "docs", "xmethod", "PROTOCOL_A.md"), encoding="utf-8") as fh:
        prot = fh.read()
    assert "| pdcor (" not in prot and "| cmi_knn (" not in prot                         # methods table
    assert prot.count("considered and excluded") == 2
    assert "- T11 " in prot and "OPEN" in prot.split("- T11 ")[1].split("\n## ")[0]


def test_eval_spec_cdl_placeholder_r50():
    with open(os.path.join(ROOT, "scratchpad", "xmethod", "specs", "eval", "full.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    d = spec["arms"]["cdl"]
    assert d["ref"] == "cdd_oran.xmethod.methods.cdl:TBD-R-50" and d["config"] == {"arm": "native"}
    assert d["declare"] == "tau" and d["analysis"] == "primary" and d["fixed_threshold"] == 0.16
    assert E.arm_kind(d) == "native" and E.analysis_block("cdl", spec) == "primary"
    assert [a for a, x in spec["arms"].items() if "fixed_threshold" in x] == ["cdl"]
    e4 = [b for b in spec["blocks"] if isinstance(b.get("arms"), list) and "two_tower" in b["arms"]]
    assert len(e4) == 1 and "cdl" in e4[0]["arms"]                                         # E4 R3 / R4 natives
    assert spec["tbd"]["cdl"].startswith("placeholder arm (R-50)")


def test_fixed_threshold_secondary_scoring_r50():
    res = api.Result("m", "v", tuple(api.EdgeResult(PLACEBO, f"K{i}", sc, None, 0, False)
                                     for i, sc in enumerate([0.16, 0.159, math.nan, 0.5])), 0.0, {})
    assert [e.declared for e in E.fixed_declare(res, 0.16).edges] == [True, False, False, True]   # CMI >= .16
    spec = copy.deepcopy(SPEC)
    spec["arms"]["pc_native"]["fixed_threshold"] = 0.995       # synthetic score = 1 - p: true edges only
    out = run(synth_records(spec), spec)
    v0 = out["V0_like_for_like"]
    assert {r["rule"] for r in v0 if r["arm"] == "pc_native"} == {"tau", "fixed"}
    assert {r["rule"] for r in v0 if r["arm"] == "pc_eq"} == {"tau"}
    fx = [r for r in v0 if r["arm"] == "pc_native" and r["rule"] == "fixed" and r["world"] == "E1"]
    assert fx and all(not r["placebo_is_tuning_column"] for r in fx)
    assert all(r["recall"]["mean"] == 1.0 and r["null"]["rate"] == 0.0 and r["placebo"]["rate"] == 0.0 for r in fx)
    assert all(r["state"] in ("VALID", "INCONCLUSIVE") for r in fx)                       # untuned: never INVALID here
    md = E.markdown(out)
    assert "| pc_native | fixed |" in md and "fixed threshold, fixed)" in md


def test_gpu_arm_cost_is_wall_seconds_r41():
    spec = copy.deepcopy(SPEC)
    gpu = {"budget_cpu_s": None, "budget_wall_s": 7200}
    spec["arms"]["ci_native"].update(gpu)
    spec["arms"]["ci_eq"].update(gpu, max_n=500, t3_cost_wall_s={"1000": 8000.0})
    assert E.cost_field(spec["arms"]["ci_native"]) == "wall_s" and E.cost_field(spec["arms"]["corr"]) == "cpu_s"
    recs = synth_records(spec)
    for r in recs:
        if r["arm"] == "ci_native":
            r.update(wall_s=50.0, child_wall_s=60.0)
    bad = next(r for r in recs if r["arm"] == "ci_native" and r["role"] == "measure" and r["job"]["regime"] == "R1")
    bad.update(status="infeasible", edges=[], cpu_s=300.0, wall_s=9000.0, cost=9000.0, cost_kind="wall-s",
               reason="exceeded the budget 7200 wall-s (measured 9000)")
    out = run(recs, spec)
    v1 = {r["key"]: r for r in out["V1_validity"]}
    row = v1[bad["key"].rsplit("|", 1)[0]]
    assert row["status"] == "infeasible" and row["infeasible_cost_wall_s"] == 9000.0
    assert "infeasible_cost_cpu_s" not in row
    t3 = [r for r in out["V1_validity"] if r["status"] == "infeasible_t3"]
    assert t3 and all(r["arm"] == "ci_eq" and r["infeasible_cost_wall_s"] == 8000.0 for r in t3)
    cost = out["V10_cost"]
    gpu_rows = [r for r in cost if r["arm"] == "ci_native" and not r["t3"]]
    assert gpu_rows and all(r["budget"] == "wall_s" for r in gpu_rows)
    assert all(r["wall_s_mean"] == 60.0 for r in gpu_rows if r["n_runs"])          # the child's wall clock
    assert any(r["t3"] and r["t3_dev_cost_wall_s"] == 8000.0 for r in cost if r["arm"] == "ci_eq")
    assert all(r["budget"] == "cpu_s" for r in cost if r["arm"] == "corr")
    md = E.markdown(out)
    assert "(cost 9000 GPU wall-s)" in md and "## V10 cost (GPU arms (R-41)" in md


def test_dataset_hash_equal_across_arms_and_shards_r41a():
    base = synth_records()
    gpu = [r for r in base if r["arm"] == "ci_native"]
    for r in gpu:                                       # a separate GPU shard: other node, same platform label
        r["host"] = {"platform": "kaggle", "node": "gpu-shard", "gpu": ["Tesla T4"]}
    v11 = run(base)["V11_integrity"]
    assert v11["label"] == "FINAL" and v11["checks"]["dataset_hash"] and v11["checks"]["dataset_hash_stamped"]
    # a GPU-shard record on a different dataset than the dataset's other arms
    recs = copy.deepcopy(base)
    bad = next(r for r in recs if r["arm"] == "ci_native")
    bad["dataset_sha256"] = "other"
    v11 = run(recs)["V11_integrity"]
    assert "dataset_hash" in v11["failed"] and v11["n_dataset_hash_mismatch"] == 1
    # a not-ok record is compared when stamped
    recs = copy.deepcopy(base)
    bad = next(r for r in recs if r["arm"] == "ci_native" and r["role"] == "measure")
    bad.update(status="infeasible", edges=[], cost=9000.0, cost_kind="wall-s", dataset_sha256="other")
    assert "dataset_hash" in run(recs)["V11_integrity"]["failed"]
    # an ok record without a hash fails; a not-ok one (unit not run) is only counted
    recs = copy.deepcopy(base)
    del next(r for r in recs if r["arm"] == "ci_native")["dataset_sha256"]
    v11 = run(recs)["V11_integrity"]
    assert "dataset_hash_stamped" in v11["failed"] and v11["n_dataset_hash_unstamped"] == 1
    recs = copy.deepcopy(base)
    bad = next(r for r in recs if r["arm"] == "ci_native" and r["role"] == "measure")
    bad.update(status="infeasible", edges=[], reason="infeasible at this n (T3)")
    del bad["dataset_sha256"]
    v11 = run(recs)["V11_integrity"]
    assert v11["checks"]["dataset_hash_stamped"] and v11["n_dataset_hash_unstamped_not_ok"] == 1
    assert "ok records without a dataset hash 0 (not-ok 1)" in E.markdown(run(recs))


# ------------------------------------------------------------------------------------------- R-42
def _r4_spec():
    spec = copy.deepcopy(SPEC)
    for b in spec["blocks"]:
        b["regimes"] = [*b["regimes"], "R4"]
        b["e4_lams"]["R4"] = [1.0]
    return spec


def _r4_records(spec):
    """synth records; PMRT in R4 as pmrt_core: candidates without a known design have no p, not applicable."""
    recs = synth_records(spec)
    for r in recs:
        if r["arm"] == "pmrt_eq" and r["job"]["regime"] == "R4":
            na = {}
            for e in r["edges"]:
                if e["source"] != PLACEBO and not e["source"].startswith("K"):
                    e.update(p=None, score=None, declared=False)
                    na[f"{e['source']}->{e['target']}"] = "no known design (kind none)"
            r["notes"] = {"not_applicable": na, "by_family_m": 2}
    return recs


def test_r4_pmrt_power_not_applicable_r42():
    spec = _r4_spec()
    out = run(_r4_records(spec), spec)
    ck = R.cell_key("E4", "R4", 1.0, 1000, 0.25)
    v2 = {r["key"]: r for r in out["V2_recall"]}
    assert v2[f"pmrt_eq|{ck}"]["power_not_applicable"] and v2[f"pmrt_eq|{ck}"]["recall"] is None   # never 0
    assert v2[f"pmrt_eq|{ck}"]["state"] == "VALID"                       # placebo (i.i.d. design) still read
    assert not v2[f"ci_eq|{ck}"]["power_not_applicable"] and v2[f"ci_eq|{ck}"]["recall"]["mean"] == 1.0
    r3 = v2[f"pmrt_eq|{R.cell_key('E4', 'R3', 1.0, 1000, 0.25)}"]
    assert not r3["power_not_applicable"] and r3["recall"]["mean"] == 1.0
    v6 = {r["key"]: r for r in out["V6_E4"]}
    assert v6[f"pmrt_eq|{ck}"]["recall"] is None and v6[f"pmrt_eq|{ck}"]["wrong_sign_rate"] is None
    v0 = [r for r in out["V0_like_for_like"] if r["key"] == f"pmrt_eq|{ck}"]
    assert {r["rule"] for r in v0} == {"raw_p", "tau"} and all(r["recall"] is None for r in v0)
    assert all(r["placebo"]["n"] > 0 for r in v0)
    assert not any(d["regime"] == "R4" for d in out["V3_paired"])                 # no per-seed recall -> no pair
    md = E.markdown(out)
    v2_md = md[md.index("## V2 recall"):md.index("## V3 paired")]
    line = next(x for x in v2_md.split("\n") if x.startswith("| E4 R4 l1 |"))
    assert line.split(" | ")[1] == "NA/NA"                                        # pmrt_eq column first
    assert "| pmrt_eq | raw_p | NA / " in md


def test_like_for_like_headline_r42(synth):
    recs, out = synth
    v0 = out["V0_like_for_like"]
    keys = {r["key"] for r in out["V1_validity"]}
    assert {r["key"] for r in v0} == keys
    for k in keys:
        rules = sorted(r["rule"] for r in v0 if r["key"] == k)
        assert rules == (["tau"] if k.startswith("pc_") else ["raw_p", "tau"])
    # raw_p: declared iff p <= .05 per edge (no multiplicity), scored like the tau arms
    ck = "corr|" + R.cell_key("E1", "R2", None, 1000, 0.25)
    row = next(r for r in v0 if r["key"] == ck and r["rule"] == "raw_p")
    truth = truth_for("E1", "R2")
    nulls = {f"{s}->{t}" for s, t in truth.null_edges if not s.startswith("K") and s != PLACEBO}
    meas = [r for r in recs if r["key"].startswith(ck + "|") and r["role"] == "measure"]
    hits = sum(float(e["p"]) <= 0.05 for r in meas for e in r["edges"] if f"{e['source']}->{e['target']}" in nulls)
    n = sum(f"{e['source']}->{e['target']}" in nulls for r in meas for e in r["edges"])
    assert row["null"]["rate"] == pytest.approx(hits / n) and row["null"]["validity"] == "INVALID"
    assert row["state"] == "INVALID" and row["recall"]["mean"] == 1.0              # shown next to its rates
    n_plac = sum(s == PLACEBO for s, _ in truth.null_edges)
    assert row["placebo"]["n"] == n_plac * len(meas) and not row["placebo_is_tuning_column"]
    tau = next(r for r in v0 if r["key"] == ck and r["rule"] == "tau")
    assert tau["placebo_is_tuning_column"] and tau["recall"] is not None
    pc = next(r for r in v0 if r["key"].startswith("pc_eq|") and r["rule"] == "tau")
    assert pc["declare"] == "tau" and pc["null"] is not None
    md = E.markdown(out)
    assert md.index("## V0 like-for-like (headline, R-42)") < md.index("## V4 claim verdicts")
    assert "| pmrt_eq | raw_p | 1.00 / " in md and "| pc_eq | tau | " in md
    # an untuned tau scoring of a p arm is shown as such
    recs2 = [r for r in recs if not (r["arm"] == "ci_eq" and r["role"] == "tune")]
    rows = [r for r in run(recs2)["V0_like_for_like"] if r["arm"] == "ci_eq"]
    assert all(r["state"] == "untuned" and r["recall"] is None for r in rows if r["rule"] == "tau")
    assert all(r["recall"] is not None for r in rows if r["rule"] == "raw_p")


def test_focal_pmrt_arm_from_spec_r42():
    spec = copy.deepcopy(SPEC)
    spec["arms"] = {"pmrt_nl_eq": {**SPEC["arms"]["pmrt_eq"], "analysis": "primary"}, **spec["arms"]}
    spec["arms"]["pmrt_eq"] = {**spec["arms"]["pmrt_eq"], "analysis": "secondary", "label": "linear statistic"}
    spec["focal"] = "pmrt_nl_eq"
    assert E.focal_arm(spec) == "pmrt_nl_eq" and E.focal_arm(SPEC) == "pmrt_eq"
    recs = synth_records(spec)
    for r in recs:                                      # the linear arm is liberal in R2: secondary, no effect
        if r["arm"] == "pmrt_eq" and r["job"]["regime"] == "R2":
            r.update(_record("pmrt_eq", r["job"]["world"], "R2", r["job"]["lam"], r["job"]["n"], r["job"]["seed"],
                             r["role"], True))
    out = run(recs, spec)
    v = out["V4_verdicts"]
    assert out["focal"] == v["focal"] == "pmrt_nl_eq"
    assert v["C2a"]["verdict"] == "SUPPORTED" and v["C3"]["verdict"] == "SUPPORTED" and v["claim"] == "SUPPORTED"
    assert v["pmrt_secondary"]["pmrt_eq"]["C2a"]["verdict"] == "NOT SUPPORTED"
    assert "pmrt_eq" not in v["C3_eq_arms"] and "pmrt_eq" not in v["C3"]["wording"]
    assert {d["arm"] for d in out["V3_paired"]} >= {"pmrt_eq", "corr"}            # focal minus every other arm
    md = E.markdown(out)
    assert "## V3 paired recall difference pmrt_nl_eq - arm" in md and "C2a pmrt_nl_eq valid in R1 / R2" in md
    assert "pmrt_eq [linear statistic] (secondary PMRT arm" in md
    assert E.t1_seed_count(recs, spec)["focal"] == "pmrt_nl_eq"
    assert E.t1_seed_count(recs, spec, focal="pmrt_eq")["focal"] == "pmrt_eq"


def test_eval_spec_pmrt_nl_placeholder_r42():
    with open(os.path.join(ROOT, "scratchpad", "xmethod", "specs", "eval", "full.json"), encoding="utf-8") as fh:
        spec = json.load(fh)
    arms = spec["arms"]
    assert spec["focal"] == "pmrt_nl_eq" and E.focal_arm(spec) == "pmrt_nl_eq"
    nl = arms["pmrt_nl_eq"]
    assert E.arm_kind(nl) == "pmrt" and nl["declare"] == "by" and E.analysis_block("pmrt_nl_eq", spec) == "primary"
    assert nl["config"] == {"covariates": "eq", "statistic": "TBD-R-42"} and "pmrt_nl_eq" in spec["tbd"]
    assert E.analysis_block("pmrt_eq", spec) == "secondary" and "secondary PMRT arm" in arms["pmrt_eq"]["label"]
    e4 = [b for b in spec["blocks"] if b["seeds"] == "TBD_E4"]
    assert len(e4) == 1 and {"pmrt_nl_eq", "pmrt_eq", "pmrt_r3"} <= set(e4[0]["arms"])    # C3 readers (R-39)
    assert sorted(a for a, d in arms.items() if E.arm_kind(d) == "pmrt") == ["pmrt_eq", "pmrt_nl_eq", "pmrt_r3"]
    assert all(E.power_not_applicable(arms[a], "R4") and not E.power_not_applicable(arms[a], "R3")
               for a in ("pmrt_eq", "pmrt_nl_eq", "pmrt_r3"))
    assert not E.power_not_applicable(arms["pcorr_eq"], "R4")


def test_r4_real_actions_have_no_design():
    """NO_DESIGN_REGIMES matches the generator: in R4 every real action has design kind 'none'."""
    from cdd_oran.xmethod.worlds import generate
    ds, _ = generate.generate_dataset("E4", "R4", 500, 3_000_000, lam=1.0)
    real = [d.kind for a, d in zip(ds.action_names, ds.designs, strict=True) if a not in (PLACEBO, PLACEBO_CONF)]
    assert real and set(real) == {"none"} and E.NO_DESIGN_REGIMES == ("R4",)
    for regime in ("R1", "R2", "R3"):
        ds, _ = generate.generate_dataset("E4", regime, 500, 3_000_000, lam=1.0)
        assert "none" not in {d.kind for a, d in zip(ds.action_names, ds.designs, strict=True) if a == "P0"}

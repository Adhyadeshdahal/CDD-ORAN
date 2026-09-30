"""E6-P discovery v4 (docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md): collection guards (PI0_V4, seed block, freeze,
registry, policies) and the v4 analyzer's pure pieces (slices, criteria, verdict precedence, artifact load / verify,
null-outcome shifts). Synthetic / small only (no simulation, no Kaggle data)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E6DEV = os.path.join(ROOT, "scratchpad", "e6_dev")
for _p in (ROOT, E6DEV):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e6p_disc_analyze_v4 as A  # noqa: E402
import e6p_discovery as D  # noqa: E402

from cdd_oran.decision import collect_p as PM  # noqa: E402
from cdd_oran.decision.edge_score import HYPOTHESES  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# ============================================================================================ collection
def test_pi0_v4_table_and_draws():
    assert set(PM.PI0_V4) == {"ES", "PowerES", "SliceGuarantee"}
    assert all(t == {"accept": 0.5, "reject": 0.5} for t in PM.PI0_V4.values())
    assert PM.PI0["v4"] is PM.PI0_V4 and "PI0_V4" in PM.__all__
    assert PM.PI0_HIGH_NO_RB["ES"] == {"accept": 0.5, "half": 0.2, "reject": 0.3}          # old table untouched
    pol = PM.RandomizedUnitPolicy(7, tables=PM.PI0_V4)
    modes = [pol({"c": c, "x": "ES", "x_idx": 0, "t0": t})[0] for c in range(5) for t in range(200)]
    assert set(modes) == {"accept", "reject"}
    assert abs(np.mean([m == "accept" for m in modes]) - 0.5) < 0.05
    pl = PM.PlaceboPolicy(7, tables=PM.PI0_V4)
    u = {"c": 1, "x": "ES", "x_idx": 0, "t0": 3}
    assert pl(dict(u)) == ("accept", 1.0)
    uu = dict(u)
    pl(uu)
    assert (uu["pi0_mode"], uu["pi0_p"]) == pol(dict(u))                                  # same draw as pi0


def test_seed_guard_v4():
    assert D.check_seed(188000, v4=True) == 188000 and D.check_seed(189999, v4=True) == 189999
    for bad in (187999, 190000, 183300, 186500):
        with pytest.raises(AssertionError):
            D.check_seed(bad, v4=True)
    with pytest.raises(AssertionError):
        D.check_seed(188000)                       # v1-v3 stages never accept the v4 block
    assert D.check_seed(183300) == 183300          # v1 check unchanged
    for lo, hi in D.V4_FORBIDDEN:                  # the block is disjoint from every forbidden block
        assert D.V4_BLOCK[1] < lo or D.V4_BLOCK[0] > hi


def test_jobs_v4_seed_map():
    exp = {"dev_v4": ("dev", 188000, 20), "eval_v4": ("eval", 188100, 600), "placebo_v4": ("placebo", 188700, 40),
           "gt_v4": ("gt", 188800, 40)}
    seen = set()
    for st, (rs, base, n) in exp.items():
        J = D.jobs(st)
        assert [x[2] for x in J] == list(range(base, base + n))
        assert all(x[0] == rs and x[1] == "v4" for x in J)
        assert [x[3] for x in J] == list(range(n))
        assert [x[4] for x in J] == ([j // 120 for j in range(n)] if rs == "eval" else [None] * n)
        assert not seen & {x[2] for x in J}
        seen |= {x[2] for x in J}
    assert D.jobs("eval_v3")[0][2] == 182000 and len(D.jobs("eval_v3")) == 1200          # v3 unchanged
    assert set(D.FROZEN_STAGES) >= {"eval_v4", "gt_v4", "placebo_v4"} and "dev_v4" not in D.FROZEN_STAGES
    assert {"eval", "eval_v2", "eval_v3", "gt", "gt_ext"} <= set(D.FROZEN_STAGES)


def test_make_policy_v4():
    p, name, tab = D.make_policy("eval", "v4", 5)
    assert isinstance(p, PM.RandomizedUnitPolicy) and not isinstance(p, PM.PlaceboPolicy)
    assert name == "pi0" and tab is PM.PI0_V4 and p.tables is PM.PI0_V4
    p, name, tab = D.make_policy("placebo", "v4", 5)
    assert isinstance(p, PM.PlaceboPolicy) and name == "placebo" and p.tables is PM.PI0_V4
    p, name, tab = D.make_policy("eval", "v2", 5)                                         # old stages: old table
    assert tab is PM.PI0_HIGH_NO_RB


def test_registry_v4():
    reg = D.registry_check("eval_v4")
    assert reg is not None and all(reg.values()) and reg["block_188000_189999"]
    assert "block_188000_189999" not in D.registry_check("eval_v3")
    d = json.load(open(os.path.join(ROOT, "docs", "benchmark", "SEED_REGISTRY.json")))
    assert d["E6"]["dev_reserved"]["e6p_discovery_v4_episodes"] == [[188000, 189999]]
    rngs = []

    def walk(o, path):
        if isinstance(o, list) and len(o) == 2 and all(isinstance(v, int) for v in o):
            rngs.append((path, o))
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + "." + k)
    walk(d["E6"], "E6")
    walk(d["XTRUCE"], "XTRUCE")
    hits = [p for p, (lo, hi) in rngs if not (hi < 188000 or lo > 189999)]
    assert hits == ["E6.dev_reserved.e6p_discovery_v4_episodes"]


def test_freeze_guard_v4(monkeypatch, tmp_path):
    doc = os.path.join(ROOT, D.PROTOCOL_DOC_V4)
    sha = D._sha_lf(doc)
    monkeypatch.setattr(D, "FROZEN_SHA256_V4", None)
    fz = D.freeze_status("eval_v4")
    assert fz["doc"] == D.PROTOCOL_DOC_V4 and not fz["frozen"]
    for st in ("eval_v4", "placebo_v4", "gt_v4"):
        out = tmp_path / f"{st}.jsonl"
        with pytest.raises(SystemExit, match="FROZEN_SHA256_V4"):
            D.run(st, "0/1", str(out))
        assert not out.exists()                               # refused before anything is written
    monkeypatch.setattr(D, "FROZEN_SHA256_V4", "0" * 64)
    assert not D.freeze_status("gt_v4")["frozen"]
    monkeypatch.setattr(D, "FROZEN_SHA256_V4", sha)
    assert D.freeze_status("placebo_v4")["frozen"]
    assert D.freeze_status("eval_v3")["doc"] == D.PROTOCOL_DOC                            # v1-v3: old doc


def test_driver_constant_is_none_until_freeze():
    st = A.protocol_status()
    assert st["exists"] and st["line"].startswith("FROZEN:")
    assert st["driver_sha256"] == D.FROZEN_SHA256_V4
    if D.FROZEN_SHA256_V4 is None:
        assert not st["frozen"]


# ============================================================================================ analyzer: slices
def test_slice_plan_full():
    seeds = list(range(188100, 188700))[::-1]
    plan = A.slice_plan(seeds)
    kinds = [p["kind"] for p in plan]
    assert kinds.count("pooled") == 1 and kinds.count("60") == 10 and kinds.count("120") == 5 and kinds.count("300") == 2
    pooled = plan[0]
    assert pooled["split"] == 0 and pooled["n_target"] == 600 and len(pooled["seeds"]) == 600
    for kind, n, base_split in (("60", 60, 20), ("120", 120, 40), ("300", 300, 60)):
        sl = [p for p in plan if p["kind"] == kind]
        assert [p["split"] for p in sl] == [base_split + i for i in range(len(sl))]
        assert all(p["n_target"] == n and len(p["seeds"]) == n for p in sl)
        assert sorted(s for p in sl for s in p["seeds"]) == list(range(188100, 188700))
        for i, p in enumerate(sl):
            assert p["seeds"] == list(range(188100 + i * n, 188100 + (i + 1) * n))
    # a missing episode keeps the slice identity (seed - 188100), the slice is just short
    plan2 = A.slice_plan([s for s in range(188100, 188700) if s != 188161])
    s60_1 = next(p for p in plan2 if p["name"] == "s60_1")
    assert 188161 not in s60_1["seeds"] and len(s60_1["seeds"]) == 59 and s60_1["n_target"] == 60


def test_slice_plan_pseudo():
    plan = A.slice_plan([183300 + j for j in range(20)], pseudo=True, sizes=(4, 10, 20))
    assert [p["kind"] for p in plan].count("60") == 5 and [p["kind"] for p in plan].count("120") == 2
    assert next(p for p in plan if p["name"] == "s120_1")["seeds"] == list(range(183310, 183320))


# ============================================================================================ analyzer: criteria
def _decl(ps, declared=()):
    out = {}
    for i, h in enumerate(HYPOTHESES):
        p = ps[i] if i < len(ps) else float("nan")
        out[h] = {"p": p, "declared": h in declared, "sign": 1,
                  "status": "undetermined" if not np.isfinite(p) else ("declared" if h in declared else "nd")}
    return out


def test_k0_check():
    ok = A.k0_check(_decl([0.5] * 57 + [0.04, 0.03, 0.01]))
    assert ok["pass"] and ok["m"] == 60 and ok["n_reject"] == 3
    many = A.k0_check(_decl([0.5] * 50 + [0.01] * 10))                   # P(Binom(60, .05) >= 10) < .01
    assert not many["pass"] and many["binom_tail"] < 0.01
    two = A.k0_check(_decl([0.5] * 60, declared=HYPOTHESES[:2]))
    assert not two["pass"] and two["n_declared"] == 2
    one = A.k0_check(_decl([0.5] * 60, declared=HYPOTHESES[:1]))
    assert one["pass"]
    assert not A.k0_check(_decl([]))["pass"]                               # nothing tested -> fail


def test_k0n_check():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)
    fams = np.array(["carrier", "sleep", "ptx", "prot_min"] * 500)
    good = A.k0n_check(p, fams, [0] * 38 + [1, 1])
    assert good["pass"] and good["n_variants"] == 40
    bad_fam = p.copy()
    bad_fam[fams == "sleep"] = np.where(rng.uniform(size=500) < 0.15, 0.01, 0.5)
    r = A.k0n_check(bad_fam, fams, [0] * 40)
    assert r["rate05_family"]["sleep"] > 0.10 and not r["pass"]
    assert not A.k0n_check(p, fams, [1] * 6 + [0] * 34)["pass"]             # 6 of 40 variants declare
    assert A.k0n_check(p, fams, [1] * 5 + [0] * 35)["pass"]
    assert not A.k0n_check(np.full(100, 0.04), ["sleep"] * 100, [0])["pass"]


def test_k1_g_checks():
    sl = {f"s120_{k}": {"sleep_units": 80, "sleep_rejects": 20} for k in range(5)}
    assert A.k1_check(sl)["pass"]
    sl["s120_3"] = {"sleep_units": 80, "sleep_rejects": 14}
    assert not A.k1_check(sl)["pass"]
    sl["s120_3"] = {"sleep_units": 59, "sleep_rejects": 30}
    assert not A.k1_check(sl)["pass"]
    cell = {"family": "sleep", "relation": "nbr", "kpi": "pv", "status": "TRUE", "sign": 1}
    assert A.g_check([cell])["pass"]
    assert not A.g_check([dict(cell, sign=-1)])["pass"]
    assert not A.g_check([dict(cell, status="INDET")])["pass"]
    assert not A.g_check([])["pass"]


def _m(prem, hits, true=4, f1=0.5, sign=1.0):
    return {"premise_hit": float(prem), "chain_hits": hits, "chain_true": true, "ind_f1": f1, "sign_acc": sign}


def test_p1_check():
    ok = [_m(1, 3), _m(1, 4), _m(1, 3), _m(0, 4), _m(1, 2)]
    assert A.p1_check(ok)["pass"] and A.p1_check(ok)["n_ok"] == 3
    assert not A.p1_check([_m(1, 3), _m(1, 4), _m(0, 3), _m(0, 4), _m(1, 2)])["pass"]
    small = [_m(1, 2, true=2)] * 3 + [_m(0, 0, true=2)] * 2                # |C*| = 2 -> need 2
    assert A.p1_check(small)["pass"]


def test_p2_check_ties_and_nan():
    nan = float("nan")
    f60 = [0.5] * 6 + [0.1] * 4
    f120 = [0.5] * 3 + [0.1] * 2
    base60 = {"corr": [0.5] * 10, "granger": [0.4] * 10, "granger_by": [nan] * 10}      # ties count for PMRT
    base120 = {"corr": [0.5] * 5, "granger": [0.4] * 5, "granger_by": [nan] * 5}
    r = A.p2_check(f60, f120, base60, base120)
    assert r["pass"] and r["per_baseline"]["corr"]["wins60"] == 6 and r["per_baseline"]["granger_by"]["wins60"] == 10
    f60b = [nan] + f60[1:]                                                           # undefined PMRT F1 = loss
    assert not A.p2_check(f60b, f120, base60, base120)["per_baseline"]["corr"]["ok"]
    base120["granger"] = [0.6] * 5
    r = A.p2_check(f60, f120, base60, base120)
    assert not r["pass"] and not r["per_baseline"]["granger"]["ok"] and r["per_baseline"]["corr"]["ok"]


def _ref():
    ref = {h: {"status": "NULL", "sign": 0} for h in HYPOTHESES}
    for f, r, k, s in (("sleep", "nbr", "load", 1), ("sleep", "nbr", "pv", 1), ("sleep", "nbr", "v", 1),
                       ("ptx", "nbr", "load", -1), ("carrier", "own", "e", 1)):
        ref[(f, r, k)] = {"status": "TRUE", "sign": s}
    return ref


def test_p3_and_s_checks():
    ref = _ref()
    d = {h: {"declared": False, "sign": 0} for h in HYPOTHESES}
    for h in (("sleep", "nbr", "load"), ("sleep", "nbr", "pv"), ("sleep", "nbr", "v"), ("carrier", "own", "e")):
        d[h] = {"declared": True, "sign": 1}
    r = A.p3_check(d, ref)
    assert r["pass"] and r["values"]["chain_hits"] == 3 and r["values"]["overall_precision"] == 1.0
    d2 = dict(d)
    d2[("sleep", "nbr", "pv")] = {"declared": True, "sign": -1}             # premise with the wrong sign
    r2 = A.p3_check(d2, ref)
    assert not r2["pass"] and not r2["parts"]["premise_declared_plus"]
    d3 = dict(d)
    d3[("prot_min", "far", "e")] = {"declared": True, "sign": 1}            # one FP: precision 4/5 = .8 still ok
    assert A.p3_check(d3, ref)["pass"]
    d3[("prot_min", "far", "v")] = {"declared": True, "sign": 1}            # 4/6 < .8
    assert not A.p3_check(d3, ref)["parts"]["overall_precision"]
    assert A.s_check(1.0, [1.0, 0.9, float("nan"), 0.95, 0.9])["pass"]
    assert not A.s_check(0.89, [1.0] * 5)["pass"]
    assert not A.s_check(1.0, [0.8, 0.9, 0.9, 0.9, 0.9])["pass"]
    assert not A.s_check(1.0, [float("nan")] * 5)["pass"]


def test_verdict_precedence():
    T, F = {"pass": True}, {"pass": False}
    args = dict(k0=T, k0n=T, g=T, k1=T, p1=T, p2=T, p3=T, s=T)

    def v(**kw):
        return A.verdict(**dict(args, **kw), mode="full", frozen=True)["verdict"]
    assert v() == "PASS"
    assert v(k0=F, g=F, k1=F) == "INVALID" and v(k0n=F, g=F) == "INVALID" and v(k0n=None) == "INVALID"
    assert v(g=F, k1=F, p3=F) == "NO-CHAIN"
    assert v(k1=F, p1=F) == "UNDERPOWERED"
    assert v(p1=F) == "PARTIAL" and v(p2=F) == "PARTIAL" and v(p1=F, p2=F) == "PARTIAL"
    assert v(p3=F) == "KILL" and v(s=F) == "KILL" and v(p1=F, s=F) == "KILL"
    assert A.verdict(**args, mode="full", frozen=False)["label"] == "PASS (PROVISIONAL: protocol not frozen)"
    assert A.verdict(**args, mode="dry-run", frozen=True)["label"].startswith("NOT A VERDICT")


# ============================================================================================ analyzer: artifact
def test_artifact_load_and_verify(tmp_path):
    import pmrt_bench as PB
    import pmrt_artifacts as PAR

    from cdd_oran.decision import fdr_layer as FL
    path = os.path.join(ROOT, A.ARTIFACT)
    st = A.artifact_status(path)
    assert st["sha_ok"] and st["schema"] == PAR.ARTIFACT_SCHEMA
    assert st["code_ok"], {k: v for k, v in st["code"].items() if not v["ok"]}
    side = open(path + ".sha256").read().split()[0]
    assert side == A.ARTIFACT_SHA256
    params, priors = PAR.load_artifact(path)
    assert len(params["hyp"]) == 60 and set(priors) == set(PB.STATS)
    dirs = FL.prior_directions(priors["loadsp_c"], PB.THR)
    assert sum(1 for d in dirs.values() if d) == 33                       # protocol section 3: 33 of 60 one-sided
    cfg = PAR.pmrt_config(params, 19)
    assert cfg.B == 19 and cfg.min_units == 30 and cfg.min_reject == 5
    bad = tmp_path / "tampered.json"
    raw = open(path, "rb").read()
    open(bad, "wb").write(raw.replace(b'"n_prior":480', b'"n_prior":481', 1))
    assert not A.artifact_status(str(bad))["sha_ok"]


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, ".tmp", "mscr_plus", "infra", "cache", "placebo1.npz")),
                    reason="small local placebo cache not present")
def test_artifact_statistic_runs_on_small_cache():
    import pmrt_bench as PB
    import pmrt_artifacts as PAR

    from cdd_oran.decision import pmrt as PM
    params, priors = PAR.load_artifact(os.path.join(ROOT, A.ARTIFACT))
    pd = PM.pmrt_data(PM.load_pmrt_pool([os.path.join(ROOT, ".tmp", "mscr_plus", "infra", "cache", "placebo1.npz")]))
    run = PB.run_integrated(pd, params, PAR.pmrt_config(params, 19), A.SPLIT["placebo"])
    decl = PB.declare_all(run, priors, 20)[A.PRIMARY]
    k0 = A.k0_check(decl)
    assert k0["m"] > 0 and all(v["procedure"] == "wby1s" for v in decl.values())
    tab = A.hyp_table(run, decl)
    assert len(tab) == 60


# ============================================================================================ analyzer: K0n variants
def test_shift_variants():
    recs = [{"seed": i, "units": [{"t0": i}], "lab_series": {"id": i}} for i in range(6)]
    vs = A.shift_variants(recs, 4)
    assert [n for n, _ in vs] == ["shift1", "shift2", "shift3", "shift4"]
    for k, (_, rr) in enumerate(vs, start=1):
        assert [r["lab_series"]["id"] for r in rr] == [(i + k) % 6 for i in range(6)]
        assert [r["units"] for r in rr] == [r["units"] for r in recs]               # real skeleton / modes kept
        assert all(r["lab_series"]["id"] != r["seed"] for r in rr)                 # never the own series
    assert [n for n, _ in A.shift_variants(recs[:2], 4)] == ["shift1", "shift3"]    # k % G == 0 skipped


def test_data_status_requires_complete_v4():
    class P:
        def __init__(self, seeds, sub="v4", smoke=False):
            self.eps = {"seed": np.array(seeds), "sub": np.array([sub] * len(seeds)),
                        "smoke": np.array([smoke] * len(seeds))}
    ok = A.data_status({"eval": P(range(188100, 188700)), "placebo": P(range(188700, 188740)),
                        "dev": P(range(188000, 188020))}, list(range(188800, 188840)), dry=False, allow_smoke=False)
    assert ok["all_ok"]
    short = A.data_status({"eval": P(range(188100, 188699)), "placebo": P(range(188700, 188740)),
                           "dev": P(range(188000, 188020))}, list(range(188800, 188840)), dry=False, allow_smoke=False)
    assert not short["all_ok"] and not short["eval"]["ok"]
    wrong_sub = A.data_status({"eval": P(range(188100, 188700), sub="v3"), "placebo": P(range(188700, 188740)),
                               "dev": P(range(188000, 188020))}, list(range(188800, 188840)), dry=False,
                              allow_smoke=False)
    assert not wrong_sub["all_ok"]
    assert not A.data_status({"eval": P(range(188100, 188700)), "placebo": P(range(188700, 188740)),
                              "dev": P(range(188000, 188020))}, list(range(188800, 188840)), dry=True,
                             allow_smoke=False)["all_ok"]


def test_cloud_bundles_v4_docs():
    """cloud.py's file list for an e6p_disc* script includes the V4 protocol and the artifacts (no bundle built)."""
    src = open(os.path.join(E6DEV, "cloud.py"), encoding="utf-8").read()
    assert "docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md" in src
    assert '"docs", "benchmark", "artifacts", "*"' in src

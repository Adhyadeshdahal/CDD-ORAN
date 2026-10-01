"""Follow-up study 2b, the certified-safe referee (docs/benchmark/E6P_CERTSAFE_PROTOCOL.md): tau derivation, bounds,
the three-way edge classification, the gate's decisions (and bit-identity of default MapGateV2), the estimators, the
driver's seed / freeze guards and registration, the artifact round trip and the analyzer's criteria
(cdd_oran/decision/certsafe.py, scratchpad/e6_dev/e6p_certsafe.py, e6p_certsafe_analyze.py)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys

import numpy as np
import pytest

from cdd_oran.decision import certsafe as CS
from cdd_oran.decision import mapgate as MG

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E6DEV = os.path.join(ROOT, "scratchpad", "e6_dev")
if E6DEV not in sys.path:
    sys.path.insert(0, E6DEV)
import e6p_certsafe as X  # noqa: E402
import e6p_certsafe_analyze as XA  # noqa: E402
import e6p_conf as C  # noqa: E402
import e6p_conf_analyze as AN  # noqa: E402

DISC_JSON = os.path.join(ROOT, "scratchpad", "e6_dev", "decision", "conf_disc_conf.json")
MAPGATE_SHA_OPTA = "64ad9d1beb336681d3f384b5d9b1871615bc317f78c0eaa6ca0d9b9dcd426dd9"   # pinned in E6P_CONF_MAPS.json


def _maps():
    return X.maps_from_artifact()


# ---------------------------------------------------------------------------------------------- tau / bounds
def test_tau_derivation_moves_the_guard_ratio_by_rho_minus_one():
    G = {"rlf": 43.15, "v": 20318.0}
    N = {("ptx", 1): 13.15, ("sleep", 1): 3.8, ("sleep", -1): 0.4}
    tau = CS.tau_table(G, N)
    assert set(tau) == {(f, d, k) for f in MG.FAMILIES for d in (1, -1) for k in CS.GUARD_KPIS}
    assert tau[("ptx", 1, "rlf")] == pytest.approx(.05 * 43.15 / (13.15 * 3))
    # every one of the N units deferred with harm tau on each of the 3 relations -> ratio (G + .05 G) / G = 1.05
    for (f, d), n in N.items():
        for k in CS.GUARD_KPIS:
            if n >= 1:
                assert (G[k] + n * 3 * tau[(f, d, k)]) / G[k] == pytest.approx(CS.RHO)
    assert tau[("sleep", -1, "rlf")] == pytest.approx(.05 * 43.15 / 3)          # N floored at 1
    assert tau[("carrier", 1, "v")] == pytest.approx(.05 * 20318.0 / 3)         # absent class -> floor
    assert CS.RHO == 1.05 and CS.Z90 == pytest.approx(1.2815515655446004)
    from scipy.stats import norm
    assert CS.Z90 == pytest.approx(norm.ppf(.9), abs=1e-12)


def test_bound_ub_slope_ci_and_missing():
    b = {"kind": "slope", "beta": -.06, "se": .02}
    assert CS.bound_ub(b, -1) == pytest.approx(.06 + CS.Z90 * .02)               # deferral of d = +1: h = -1
    assert CS.bound_ub(b, 1) == pytest.approx(-.06 + CS.Z90 * .02)
    c = {"kind": "ci", "lo": -.02, "hi": .03}
    assert CS.bound_ub(c, 1) == .03 and CS.bound_ub(c, -1) == .02
    for bad in (None, {}, {"kind": "slope", "beta": 1.0, "se": None}, {"kind": "slope", "beta": float("nan"),
                                                                        "se": 1.0}):
        assert CS.bound_ub(bad, 1) == math.inf
    with pytest.raises(ValueError):
        CS.bound_ub({"kind": "?"}, 1)


def test_three_way_edge_classification():
    M = {("ptx", "own", "rlf"): .03, ("ptx", "far", "v"): None}               # None = declared, no effect
    bnd = {("ptx", "nbr", "rlf"): {"kind": "slope", "beta": -.06, "se": .02},
           ("ptx", "far", "rlf"): {"kind": "slope", "beta": .001, "se": .01}}
    assert CS.classify_edge(M, bnd, "ptx", "own", "rlf", 1, .05)[0] == CS.DECLARED
    assert CS.classify_edge(M, bnd, "ptx", "far", "v", 1, .05)[0] == CS.DECLARED
    st, ub = CS.classify_edge(M, bnd, "ptx", "nbr", "rlf", 1, .05)             # deferring ptx-up raises nbr rlf
    assert st == CS.UNRESOLVED and ub == pytest.approx(.06 + CS.Z90 * .02)
    assert CS.classify_edge(M, bnd, "ptx", "nbr", "rlf", 1, .2)[0] == CS.CERTIFIED   # same bound, larger tau
    assert CS.classify_edge(M, bnd, "ptx", "nbr", "rlf", -1, .05)[0] == CS.CERTIFIED  # deferring ptx-down helps
    assert CS.classify_edge(M, bnd, "ptx", "far", "rlf", 1, .05)[0] == CS.CERTIFIED
    assert CS.classify_edge(M, bnd, "ptx", "own", "v", 1, 1e9)[0] == CS.UNRESOLVED   # no bound: never certified
    assert CS.classify_edge(M, None, "ptx", "nbr", "rlf", 1, 1e9)[0] == CS.UNRESOLVED


def test_certify_on_the_frozen_maps():
    maps = _maps()
    tau = CS.tau_table({"rlf": 43.15, "v": 20318.0}, {(f, d): 10.0 for f in MG.FAMILIES for d in (1, -1)})
    tight = {(f, r, k): {"kind": "slope", "beta": 0.0, "se": 0.0} for f in MG.FAMILIES for r in MG.RELS_V2
             for k in CS.GUARD_KPIS}
    pm = maps["MG:PMRT"]
    c0 = CS.certify(pm, tight, tau)
    assert c0["uncertified"] == []                                             # zero-width bounds certify all
    assert {k for k, v in c0["classes"].items() if v["can_defer"]} == {"carrier|-1", "sleep|1", "ptx|1",
                                                                       "prot_min|-1"}
    assert c0["classes"]["carrier|1"]["edges"] == []                           # never deferred: nothing checked
    wide = {**tight, ("ptx", "nbr", "rlf"): {"kind": "slope", "beta": -.10, "se": .05}}
    c1 = CS.certify(pm, wide, tau)
    assert c1["uncertified"] == [["ptx", 1]]                                   # the option (a) failure edge
    e = {(r, k): s for r, k, s, _, _ in c1["classes"]["ptx|1"]["edges"]}
    assert e[("nbr", "rlf")] == CS.UNRESOLVED and e[("own", "rlf")] == CS.DECLARED
    assert CS.certify(pm, None, tau)["uncertified"] == [["carrier", -1], ["sleep", 1], ["ptx", 1],
                                                       ["prot_min", -1]]
    # same rule for every arm: a map without any bound (blanket2) becomes all-accept
    bl = maps["blanket2"]
    unc = [tuple(x) for x in CS.certify(bl, None, tau)["uncertified"]]
    assert CS.certsafe_signature(bl, unc) == MG.all_accept_signature()
    # the GT bounds come from the frozen GT cells (95 % CI)
    gt = CS.gt_ci_bounds(json.load(open(DISC_JSON))["gt"]["cells"])
    assert len(gt) == 4 * 3 * 2 and gt[("ptx", "nbr", "rlf")]["hi"] < 0


# ---------------------------------------------------------------------------------------------- gate
def _ctx(knob, step, own=np.nan, nbr=np.nan):
    return {"knob": knob, "step": float(step), MG.PRESSURE["own"]: own, MG.PRESSURE["nbr"]: nbr}


def _units(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return [{"c": int(rng.integers(3)), "ctx": _ctx(str(rng.choice(MG.FAMILIES)), float(rng.choice([-1, 1, 0])),
                                                    float(rng.uniform(0, .2)), float(rng.uniform(0, .2)))}
            for _ in range(n)]


def _run(g, units):
    return [g(u)[0] for u in units]


def test_default_mapgate_v2_is_bit_identical():
    """mapgate.py is untouched (its sha256 is the one the frozen maps artifact pins), and the certsafe gate with an
    empty uncertified set takes exactly MapGateV2's decisions (stateful: duty bound) on every frozen map."""
    assert C.sha_lf(os.path.join(ROOT, "cdd_oran", "decision", "mapgate.py")) == MAPGATE_SHA_OPTA
    assert json.load(open(os.path.join(ROOT, C.MAPS_DOC)))["mapgate"]["sha256"] == MAPGATE_SHA_OPTA
    units = _units()
    for arm, M in _maps().items():
        a, b = MG.MapGateV2(M), CS.CertSafeMapGateV2(M, ())
        assert _run(a, units) == _run(b, units), arm
        assert {k: v for k, v in b.n.items() if not k.startswith(("dir_", "uncert_"))} == a.n
        assert CS.certsafe_signature(M, ()) == MG.decision_signature(M)


def test_gate_withholds_only_uncertified_deferrals_and_signature_is_the_policy():
    M = _maps()["MG:PMRT"]
    units = _units(seed=1)
    base = _run(MG.MapGateV2(M), units)
    g = CS.CertSafeMapGateV2(M, [("ptx", 1)])
    got = _run(g, units)
    for u, x, y in zip(units, base, got, strict=True):
        if u["ctx"]["knob"] == "ptx" and u["ctx"]["step"] > 0:
            assert y == "accept"
        elif u["ctx"]["knob"] != "ptx":
            assert x == y                                    # other knobs: own duty-bound state, unchanged
    assert g.n["uncert_ptx"] == sum(x == "reject" for u, x in zip(units, base, strict=True)
                                    if u["ctx"]["knob"] == "ptx" and u["ctx"]["step"] > 0) > 0
    assert sum(g.n[f"dir_{f}_{d:+d}"] for f in MG.FAMILIES for d in (1, -1, 0) if f"dir_{f}_{d:+d}" in g.n) == len(units)
    # equal certsafe signatures <=> identical policies
    M2 = {k: 2.5 * v for k, v in M.items()}
    assert CS.certsafe_signature(M, [("ptx", 1)]) == CS.certsafe_signature(M2, [("ptx", 1)])
    assert _run(CS.CertSafeMapGateV2(M2, [("ptx", 1)]), units) == got
    assert CS.certsafe_signature(M, [("ptx", 1)]) != CS.certsafe_signature(M, [])
    allu = [(f, d) for f in MG.FAMILIES for d in (1, -1)]
    assert set(_run(CS.CertSafeMapGateV2(M, allu), units)) == {"accept"}
    assert CS.certsafe_signature(M, allu) == MG.all_accept_signature()
    arb = CS.certsafe_arbiter(M, [("ptx", 1)])
    assert isinstance(arb, MG.DirectionalUnitArbiter) and arb.policy.uncertified == {("ptx", 1)}


# ---------------------------------------------------------------------------------------------- estimators
def _ud(seed=0, beta=2.0, n_ep=40, per=100):
    from tests.test_opta_study import _ud as ud_opta
    return ud_opta(n_ep=n_ep, per=per, beta=beta, seed=seed)


def test_design_slope_se_is_the_map_beta_with_calibrated_se_and_naive_is_confounded():
    import e6p_opta_kb_analyze as KBA
    bs, ses, cover = [], [], 0
    for sd in range(12):
        ud = _ud(seed=sd)
        r = CS.design_slope_se(ud, "sleep", "nbr", "pv")
        assert r["beta"] == AN.design_slope(ud, "sleep", "nbr", "pv")["beta"]       # exactly the option (a) map beta
        bs.append(r["beta"])
        ses.append(r["se"])
        cover += 2.0 <= r["beta"] + CS.Z90 * r["se"]                                  # one-sided 90 % UB covers
        o = CS.naive_slope_se(ud, "sleep", "nbr", "pv")
        assert o["beta"] == pytest.approx(KBA.naive_slope(ud, "sleep", "nbr", "pv"), rel=1e-12)
        assert o["beta"] < 0 and o["se"] > 0                                         # confounded, yet "precise"
    assert abs(np.mean(bs) - 2.0) < .3
    assert .5 < np.std(bs) / np.mean(ses) < 2.0                                       # SE matches the spread
    assert cover >= 9
    tb = XA.bounds_tables(_ud(seed=3), {"MG:PMRT": {("sleep", "nbr", "pv"): AN.design_slope(_ud(seed=3), "sleep",
                                                                                              "nbr", "pv")["beta"]}})
    assert tb["repro"]["ok"] and len(tb["pmrt"]) == len(tb["ols"]) == 24
    assert not XA.bounds_tables(_ud(seed=3), {"MG:PMRT": {("sleep", "nbr", "pv"): 9.0}})["repro"]["ok"]


# ---------------------------------------------------------------------------------------------- seeds / registry
def _registered_blocks():
    reg = json.load(open(os.path.join(ROOT, X.REGISTRY_DOC)))
    out = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + (k,))
        elif isinstance(o, list) and o and all(isinstance(x, list) and len(x) == 2 for x in o):
            out.extend((path, int(a), int(b)) for a, b in o)
    for w in ("E6", "XTRUCE"):
        walk(reg[w], (w,))
    return out


def test_seed_layout_registration_and_guards():
    assert X.stage_seeds("dev") == list(range(191000, 191040))
    assert X.stage_seeds("eval") == list(range(191100, 191260))
    J = X.jobs("dev")
    assert len(J) == 40 + 2 and J[:3] == [(191000, X.CAL_ARM), (191000, "noarb"), (191001, X.CAL_ARM)]
    for st, bad in (("dev", 190999), ("dev", 191040), ("eval", 191099), ("eval", 191260), ("eval", 187000),
                    ("dev", 186080)):
        with pytest.raises(AssertionError):
            X.check_seed(bad, st)
    seeds = set(X.stage_seeds("dev")) | set(X.stage_seeds("eval"))
    for path, lo, hi in _registered_blocks():
        if path[-1] == "e6p_certsafe_episodes":
            assert (lo, hi) == X.SEED_BLOCK
            continue
        assert not any(lo <= s <= hi for s in seeds), (path, lo, hi)
    assert not seeds & set(C.stage_seeds("eval")) and min(seeds) < 960000              # never TEST / option (a)
    reg = X.registry_check()
    assert reg is not None and all(reg.values())
    assert C.registry_check() is not None and all(C.registry_check().values())         # option (a) still registered


def _root(tmp_path, frozen_line="FROZEN: no"):
    for rel in (X.PROTOCOL_DOC, X.REGISTRY_DOC, C.MAPS_DOC):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), dst)
    p = tmp_path / X.PROTOCOL_DOC
    p.write_bytes(re.sub(r"^FROZEN:.*$", frozen_line, p.read_text(encoding="utf-8"), count=1,
                         flags=re.M).encode("utf-8"))
    return str(tmp_path)


def test_freeze_guards_refuse_until_frozen(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(X, "FROZEN_SHA256_CS", None)
    monkeypatch.setattr(X, "CERTSAFE_SHA256", None)
    for st in X.STAGES:
        with pytest.raises(SystemExit):
            X.guard_run(st, smoke=False, root=root)                     # protocol not frozen (dev too)
        assert X.guard_run(st, smoke=True, root=root)["freeze"]["frozen"] is False
    monkeypatch.setattr(X, "FROZEN_SHA256_CS", X.freeze_status(root)["sha256"])
    with pytest.raises(SystemExit):
        X.guard_run("dev", smoke=False, root=root)                      # sha matches but "FROZEN: no"
    root2 = _root(tmp_path / "f", "FROZEN: yes")
    monkeypatch.setattr(X, "FROZEN_SHA256_CS", X.freeze_status(root2)["sha256"])
    assert X.guard_run("dev", smoke=False, root=root2)["freeze"]["frozen"]
    with pytest.raises(SystemExit):
        X.guard_run("eval", smoke=False, root=root2)                    # no certsafe artifact / CERTSAFE_SHA256
    ap = tmp_path / "a.json"
    ap.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        X.guard_run("eval", smoke=False, artifact_path=str(ap), root=root2)
    monkeypatch.setattr(X, "CERTSAFE_SHA256", hashlib.sha256(b"{}").hexdigest())
    assert X.guard_run("eval", smoke=False, artifact_path=str(ap), root=root2)["artifact"]["ok"]
    mp = tmp_path / "f" / C.MAPS_DOC                                    # option (a) maps artifact changed -> refuse
    mp.write_bytes(mp.read_bytes() + b" ")
    with pytest.raises(SystemExit):
        X.guard_run("dev", smoke=False, root=root2)
    shutil.copy(os.path.join(ROOT, C.MAPS_DOC), mp)
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(SystemExit):
        X.guard_run("dev", smoke=False, root=root2)                     # Linux numerics only


def test_driver_sha_ignores_the_artifact_pin(tmp_path):
    d = tmp_path / XA.DRIVER
    d.parent.mkdir(parents=True, exist_ok=True)
    src = open(os.path.join(ROOT, XA.DRIVER), encoding="utf-8").read()
    d.write_text(src, encoding="utf-8")
    h0 = X.driver_sha(str(tmp_path))
    d.write_text(re.sub(r"^CERTSAFE_SHA256 = .*$", 'CERTSAFE_SHA256 = "' + "b" * 64 + '"', src, count=1,
                        flags=re.M), encoding="utf-8")
    assert X.driver_sha(str(tmp_path)) == h0
    d.write_text(src + "\n# changed\n", encoding="utf-8")
    assert X.driver_sha(str(tmp_path)) != h0


# ---------------------------------------------------------------------------------------------- calib / artifact
def _dev_records(n=40, rlf=40.0):
    recs = []
    for s in X.stage_seeds("dev")[:n]:
        pol = {f"dir_{f}_{d:+d}": (10 if f == "ptx" else 2) for f in MG.FAMILIES for d in (1, -1)}
        base = {f: 1.0 for f in C.D.SUM_FIELDS}
        recs.append({"kind": "job", "schema": X.SCHEMA_EVAL, "sub": X.SUB, "cs_stage": "dev", "seed": s,
                     "arm": X.CAL_ARM, "smoke": False, **base, "rlf": rlf, "viol_ue_s": 2e4,
                     "policy_counts": {"policy": pol}})
        if s in X.REPRO_SEEDS:
            recs.append(dict(recs[-1], arm="noarb", policy_counts={}))
    return recs


def test_calib_and_artifact_roundtrip(tmp_path, monkeypatch):
    f = tmp_path / "dev.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in _dev_records()) + "\n", encoding="utf-8")
    R, _ = XA.load_jobs([str(f)], "dev")
    cb = XA.calib_from(R)
    assert cb["n"] == 40 and cb["G"] == {"v": 2e4, "rlf": 40.0} and cb["repro"]["ok"] and cb["noarb_alias"]
    assert dict(((a, b), v) for a, b, v in cb["N"])[("ptx", 1)] == 10.0
    R[191001]["noarb"] = dict(R[191001]["noarb"], rlf=41.0)
    assert not XA.calib_from(R)["repro"]["ok"]
    calib = {"schema": XA.SCHEMA_CALIB, "mode": "full", **cb, "platform": {}}
    zero = CS.bounds_to_json({(f_, r, k): {"kind": "slope", "beta": 0.0, "se": 0.01} for f_ in MG.FAMILIES
                              for r in MG.RELS_V2 for k in CS.GUARD_KPIS})
    bounds = {"schema": XA.SCHEMA_BOUNDS, "mode": "full", "pmrt": zero, "ols": None, "beta_repro": {"ok": True},
              "sha256": {"maps_artifact": C.maps_status()["sha256"]}, "platform": {}}
    disc = json.load(open(DISC_JSON))
    sha = {"disc_conf.json": C.sha_lf(DISC_JSON), "certsafe_bounds.json": "x", "certsafe_calib.json": "y"}
    monkeypatch.setattr(X, "freeze_status", lambda root=None: {"frozen": False})  # independent of the real freeze
    with pytest.raises(SystemExit):                                             # protocol not frozen
        XA.build_artifact(bounds, calib, disc, os.path.join(ROOT, C.MAPS_DOC), sha)
    monkeypatch.setattr(X, "freeze_status", lambda root=None: {"frozen": True})
    art = XA.build_artifact(bounds, calib, disc, os.path.join(ROOT, C.MAPS_DOC), sha)
    assert sorted(art["arms"]) == sorted(X.CS_ARMS) and art["noarb_alias"]
    assert art["arms"]["CS:PMRT"]["uncertified"] == []                          # tight PMRT bounds
    assert art["arms"]["CS:blanket2"]["alias_of"] == "noarb"                     # no bound -> all-accept -> noarb
    for a in X.CS_ASSOC:                                                        # ols bounds absent here
        for key, v in art["arms"][a]["certification"].items():
            f_, d_ = key.split("|")
            if any(e[2] != CS.DECLARED for e in v["edges"]):
                assert [f_, int(d_)] in art["arms"][a]["uncertified"]
    p = tmp_path / "E6P_CERTSAFE.json"
    p.write_text(json.dumps(art, indent=1), encoding="utf-8")
    ld = X.load_artifact(str(p))
    assert ld["alias"]["jobs"] == art["jobs"] and "MG:PMRT" in ld["maps"]
    al = ld["alias"]
    J = X.jobs("eval", al)
    arms = X.eval_arms(al)
    assert len(J) == 160 * len(arms) and J[0] == (191100, "freeze") and J[len(arms)][0] == 191101
    assert set(X.ANCHORS) | set(X.REFS) <= set(arms) and X.CAL_ARM not in arms
    bad = json.loads(p.read_text(encoding="utf-8"))
    bad["arms"]["CS:PMRT"]["uncertified"] = [["ptx", 1]]                        # tampered: signature mismatch
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SystemExit):
        X.load_artifact(str(p))
    with pytest.raises(SystemExit):                                             # refusals
        XA.build_artifact(dict(bounds, mode="partial"), calib, disc, os.path.join(ROOT, C.MAPS_DOC), sha)
    with pytest.raises(SystemExit):
        XA.build_artifact(dict(bounds, beta_repro={"ok": False}), calib, disc, os.path.join(ROOT, C.MAPS_DOC), sha)
    with pytest.raises(SystemExit):
        XA.build_artifact(bounds, calib, disc, os.path.join(ROOT, C.MAPS_DOC), dict(sha, **{"disc_conf.json": "z"}))
    sm = X.smoke_artifact()
    assert sm["sha256"] is None and set(sm["tables"]) == set(X.CS_ARMS)


# ---------------------------------------------------------------------------------------------- criteria
def _R(n=40, seed=0):
    rng = np.random.default_rng(seed)
    base = {"prot_viol": 100.0, "prot_ue_s": 3600.0, "energy_j": 1e6, "viol_ue_s": 50.0, "ue_s": 1e4,
            "nonprot_embb_viol": 10.0, "ll_viol": 5.0, "rlf": 2.0}
    arm = {"noarb": (1.0, 1.0, 1.0), "freeze": (1.3, 1.0, 1.0), "sub:ES+PowerES": (1.0, .8, 1.0),
           "sub:ES": (.9, .9, 1.0), "sub:PowerES": (.95, .9, 1.0), "sub:SliceGuarantee": (.6, 1.0, 1.0),
           "never_sleep": (.85, .8, 1.0), "CS:PMRT": (.75, .8, 1.04), "tie": (.85, .8, 1.0)}
    R = {}
    for s in range(n):
        R[s] = {}
        for a, (v, e, rl) in arm.items():
            r = {k: x * (1 + .05 * rng.normal()) for k, x in base.items()}
            r["prot_viol"] *= v
            r["energy_j"] *= e
            r["rlf"] *= rl
            R[s][a] = r
        R[s]["tie"] = dict(R[s]["never_sleep"])
    return R


def test_d3_s1_and_verdict():
    st = AN.arm_stats_multi(_R(), ["never_sleep", "CS:PMRT", "tie"], n_boot=1000)
    P, B = st["point"], st["boot"]
    d3 = XA.d3_check(P, B)
    assert P["CS:PMRT"]["eligible"] and d3["pass"] and d3["delta3"] > 0 and d3["lb90"] > 0
    assert not XA.d3_check(P, B, primary="tie")["pass"]                         # a tie with never_sleep fails D3
    s1 = XA.s1_check(B)
    assert s1["pass"] and 1.0 < s1["rlf_ratio_ub90"] <= 1.20
    assert not XA.s1_check(B, cap=1.0)["pass"]
    V = XA.eval_verdict
    assert V(False, True, True, True, True, True, True)["verdict"] == "INVALID"
    assert V(True, True, False, True, True, True, True)["verdict"] == "NOT ELIGIBLE"
    assert V(True, True, True, True, True, True, True)["label"] == "PASS"
    assert V(True, True, True, True, True, False, True)["verdict"] == "PARTIAL"
    assert V(True, True, True, False, False, True, True)["verdict"] == "PARTIAL"
    assert V(True, True, True, False, False, False, True)["verdict"] == "FAIL"
    v = V(True, True, True, True, True, True, False, ["smoke records"])
    assert v["verdict"] == "PASS" and v["label"].startswith("NOT A VERDICT")

"""Option (a) full study (docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md): the driver's seed / freeze / maps guards, the
directional collection arbiter, per-unit rows through the unit table, map building (design slope, n-free granger tau,
signatures / aliasing, the maps artifact) and the criteria logic of the analyzer (K1 / X / label, D1 with per-resample
re-selection, D2 Holm, verdict precedence) - scratchpad/e6_dev/e6p_conf.py and e6p_conf_analyze.py."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np
import pytest

from cdd_oran.decision import collect_p as CP
from cdd_oran.decision import crt_units as CU
from cdd_oran.decision import disc_bench as DB
from cdd_oran.decision import mapgate as MG
from cdd_oran.envs.e6 import config as EC

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E6DEV = os.path.join(ROOT, "scratchpad", "e6_dev")
if E6DEV not in sys.path:
    sys.path.insert(0, E6DEV)
import e6p_conf as C  # noqa: E402
import e6p_conf_analyze as AN  # noqa: E402


# ---------------------------------------------------------------------------------------------- seeds / registry
def test_seed_layout_guards_and_jobs():
    assert C.stage_seeds("disc")[0] == 186100 and len(C.stage_seeds("disc")) == 600
    J = C.jobs("dev_conf")
    assert len(J) == 20 + 2 * len(C.REPRO_ARMS)
    assert J[:4] == [("coll", 186080, 0), ("arm", 186080, "incumbent"), ("arm", 186080, "MG:allaccept"),
                     ("arm", 186080, "noarb")]
    assert [len(C.jobs(s)) for s in ("disc", "placebo", "gt")] == [600, 200, 40]
    assert {x[1] for x in C.jobs("gt")} == set(range(186900, 186940))
    for st, bad in (("disc", 186099), ("disc", 186700), ("placebo", 186699), ("gt", 186940), ("dev_conf", 186079),
                    ("eval", 186999), ("eval", 187160), ("dev_conf", 186000)):
        with pytest.raises(AssertionError):
            C.check_seed(bad, st)
    for st in C.STAGES:
        for sd in (C.stage_seeds(st)[0], C.stage_seeds(st)[-1]):
            assert C.check_seed(sd, st) == sd
    all_seeds = [s for st in C.STAGES for s in C.stage_seeds(st)]
    assert len(all_seeds) == len(set(all_seeds)) == 1020
    assert not set(all_seeds) & set(range(186000, 186080))                     # K-B never re-used
    reg = C.registry_check()
    assert reg is not None and all(reg.values())


def test_eval_jobs_from_alias_table_are_seed_major_and_distinct():
    al = C.alias_table(C.smoke_maps())
    J = C.jobs("eval", al)
    arms = C.eval_arms(al)
    assert len(J) == 160 * len(arms) and J[0] == ("arm", 187000, "freeze") and J[len(arms)][1] == 187001
    assert len(set(arms)) == len(arms) and set(C.ANCHORS) | set(C.REFS) <= set(arms)
    assert set(al["alias_of"]) == set(C.MAP_ARMS)
    assert set(al["jobs"]) == {a for a in C.MAP_ARMS if al["alias_of"][a] == a}


# ---------------------------------------------------------------------------------------------- freeze / maps guards
def _root(tmp_path, frozen_line="FROZEN: no"):
    for rel in (C.PROTOCOL_DOC, C.REGISTRY_DOC):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), dst)
    p = tmp_path / C.PROTOCOL_DOC
    txt = re.sub(r"^FROZEN:.*$", frozen_line, p.read_text(encoding="utf-8"), count=1, flags=re.M)
    p.write_bytes(txt.encode("utf-8"))
    return str(tmp_path)


def test_freeze_guards_refuse_until_frozen(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(C, "FROZEN_SHA256_CONF", None)             # independent of the real freeze state
    for st in C.FROZEN_STAGES:
        with pytest.raises(SystemExit):
            C.guard_run(st, smoke=False, root=root)                # FROZEN_SHA256_CONF is None
        assert C.guard_run(st, smoke=True, root=root)["freeze"]["frozen"] is False   # smoke is exempt
    assert C.guard_run("dev_conf", smoke=False, root=root)["registry"]["layout"]     # dev_conf: pre-freeze OK
    sha_no = C.freeze_status(root)["sha256"]
    monkeypatch.setattr(C, "FROZEN_SHA256_CONF", sha_no)
    with pytest.raises(SystemExit):
        C.guard_run("disc", smoke=False, root=root)               # sha matches but the doc says "FROZEN: no"
    root2 = _root(tmp_path / "f", "FROZEN: yes")
    monkeypatch.setattr(C, "FROZEN_SHA256_CONF", C.freeze_status(root2)["sha256"])
    for st in ("disc", "placebo", "gt"):
        assert C.guard_run(st, smoke=False, root=root2)["freeze"]["frozen"]
    with pytest.raises(SystemExit):
        C.guard_run("eval", smoke=False, root=root2)              # no maps artifact / MAPS_SHA256 None
    mp = tmp_path / "maps.json"
    mp.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        C.guard_run("eval", smoke=False, maps_path=str(mp), root=root2)
    monkeypatch.setattr(C, "MAPS_SHA256", hashlib.sha256(b"{}").hexdigest())
    assert C.guard_run("eval", smoke=False, maps_path=str(mp), root=root2)["maps"]["ok"]
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(SystemExit):
        C.guard_run("disc", smoke=False, root=root2)              # Linux numerics only for the frozen stages


def test_load_maps_refuses_an_artifact_whose_alias_table_does_not_reproduce(tmp_path):
    maps = C.smoke_maps()
    al = C.alias_table(maps)
    art = {"arms": {a: {"map": MG.map_to_json(M)} for a, M in maps.items()}, "mapgate": {"theta": C.THETA},
           "noarb_alias": False, "jobs": al["jobs"], "alias_of": al["alias_of"]}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(art), encoding="utf-8")
    assert C.load_maps(str(p))["alias"]["jobs"] == al["jobs"]
    art["jobs"] = art["jobs"][:-1]
    p.write_text(json.dumps(art), encoding="utf-8")
    with pytest.raises(SystemExit):
        C.load_maps(str(p))
    with pytest.raises(SystemExit):
        C.load_maps(str(tmp_path / "absent.json"), smoke=False)
    assert C.load_maps(str(tmp_path / "absent.json"), smoke=True)["sha256"] is None


# ---------------------------------------------------------------------------------------------- directional arbiter
def _obs(t, reqs):
    return {"t": float(t), "config": {("sleep", 1): 0.0}, "new_reports": [], "requests": reqs,
            "static": {"cells": 2, "is_macro": [True, False], "neighbours": [[1], [0]]}}


def _req(cur, prop):
    return {"xapp": "ES", "knob": ("sleep", 1), "cur": float(cur), "prop": float(prop)}


def test_directional_arbiter_defers_only_the_opening_direction_and_forks_as_directional():
    def rej(u):
        return "reject", 1.0
    for cls, want in ((MG.DirectionalUnitArbiter, ["reject", "accept", "reject"]),
                      (C.UP.UnitArbiter, ["reject", "reject", "reject"])):
        arb = cls(rej, T=60.0, warmup_s=0.0, open_rule="feasible")
        out = [arb(_obs(t, [r]))["decisions"][0] for t, r in ((0, _req(0, 1)), (1, _req(1, 0)), (2, _req(0, 1)))]
        assert out == want
        assert len(arb.units) == 1
    arb = MG.DirectionalUnitArbiter(rej, T=60.0, warmup_s=0.0, open_rule="feasible")
    arb(_obs(0, [_req(0, 1)]))
    assert arb.passed == 0
    arb(_obs(1, [_req(1, 0)]))
    assert arb.passed == 1
    fk = arb.fork(lambda u: ("accept", 1.0))
    assert isinstance(fk, MG.DirectionalUnitArbiter) and fk.active == {}     # GT rollouts keep the semantics


def _cfg(seed=6):
    return EC.E6Config(seed=seed, mix="ES", scenario="surge", load_factor=1.0, n_ue=300, n_pico=3, mobility="ped",
                       warmup_s=60.0, scored_s=40.0,
                       e6p=EC.E6PConfig(ptx_on=True, prot_on=True, xapps=("PowerES", "SliceGuarantee")))


def test_directional_incumbent_collection_logs_rows_and_placebo_relabels():
    import e6p_opta_kb as KB
    cfg = _cfg()
    res = CP.run_collection(cfg, CP.IncumbentPolicy(cfg.seed), open_rule="feasible", arb_warmup_s=0.0,
                            count_all=True, arbiter_cls=MG.DirectionalUnitArbiter)
    assert isinstance(res["arb"], MG.DirectionalUnitArbiter) and res["units"]
    urecs = [KB.unit_record(i, u, 0) for i, u in enumerate(res["units"])]
    rec = {"units": urecs, "pi0_table": None, "directional": True}
    assert AN.check_episode(rec)["ok"]
    pl = CP.run_collection(cfg, CP.PlaceboIncumbent(cfg.seed), open_rule="feasible", arb_warmup_s=0.0,
                           count_all=True, arbiter_cls=MG.DirectionalUnitArbiter)
    assert pl["arb"].passed == 0                                               # accept applied: nothing to pass
    pr = [KB.unit_record(i, u, 0) for i, u in enumerate(pl["units"])]
    assert all(u["applied_mode"] == "accept" for u in pr) and {u["mode"] for u in pr} <= {"accept", "reject"}
    assert AN.check_episode({"units": pr, "pi0_table": None, "directional": True})["ok"]
    bad = [dict(u) for u in pr]
    bad[0].pop("probs")
    c = AN.check_episode({"units": bad, "pi0_table": None})
    assert not c["ok"] and c["bad"] == {"missing": 1}
    assert not AN.check_episode({"units": pr, "pi0_table": {"ES": {"accept": .5}}})["ok"]


def test_random_sized_map_default_key_unchanged():
    import e6p_opta_ka as KA
    M = MG.random_sized_map(KA.M_GT, KA._cells(), 6623)
    items = sorted([[*k, v] for k, v in M.items()])
    assert hashlib.sha256(json.dumps(items).encode()).hexdigest() == \
        "00618ffc25252af123bab2db6db6e40e97d4a9b71ff52ed4e26cd99c3d5fe7bb"        # K-A2 GT2_rand (header of e6p-optaka2-2)
    assert MG.random_sized_map(KA.M_GT, KA._cells(), 6623, 2) == M
    M3 = MG.random_sized_map(KA.M_GT, KA._cells(), 6623, 3)
    assert M3 != M and len(M3) == len(M)


# ---------------------------------------------------------------------------------------------- records -> unit table
def _incumbent_records(n_ep=4, units=60, seed=3, drop_row=False):
    rng = np.random.default_rng(seed)
    inc = CP.IncumbentPolicy(0)
    out = []
    for r in CU.synthetic_records(rng, n_ep=n_ep, units_per_ep=units):
        us = []
        for u in r["units"]:
            u = dict(u)
            s = float(rng.uniform())
            pa = inc.accept_prob({"knob": u["knob"], "step": u["step"], "own_prb_util": s,
                                  "nbr_max_prb_util": np.nan, "own_prot_below_frac": 0.0})[0]
            m = "accept" if rng.uniform() < pa else "reject"
            u.update(mode=m, applied_mode=m, p=pa if m == "accept" else 1 - pa, probs={"accept": pa, "reject": 1 - pa})
            us.append(u)
        if drop_row:
            us[0].pop("probs")
        out.append(dict(r, units=us, pi0_table=None, sub=C.SUB, conf_stage="disc", stage="eval", smoke=False,
                        kind="episode", key=["eval", C.SUB, r["seed"]], directional=True))
    return out


def test_incumbent_records_give_per_unit_rows_and_scan_catches_a_missing_row(tmp_path):
    recs = _incumbent_records()
    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(json.dumps(r, default=float) for r in recs) + "\n", encoding="utf-8")
    sc = AN.scan_files([str(f)], "disc", keep=True, allow_smoke=False)
    assert sc["check"]["ok"] and sc["check"]["episodes"] == 4 and len(sc["records"]) == 4
    DB.build_cache([str(f)], str(tmp_path / "c.npz"), stages={"eval"})
    ud = DB.load_pool([str(tmp_path / "c.npz")], H=90, H_pre=90, stages={"eval"}).unit_data()
    assert ud.n > 0 and ud.meta["p_mismatch"] == 0
    assert set(np.round(ud.probs[:, 0], 6)) <= {.15, .3, .35, .85} and np.allclose(ud.probs.sum(1), 1.0)
    assert np.allclose(ud.p, ud.probs[np.arange(ud.n), ud.mode])
    f2 = tmp_path / "bad.jsonl"
    f2.write_text("\n".join(json.dumps(r, default=float) for r in _incumbent_records(drop_row=True)) + "\n",
                  encoding="utf-8")
    c2 = AN.scan_files([str(f2)], "disc", keep=False, allow_smoke=False)["check"]
    assert not c2["ok"] and c2["bad_units"] == {"missing": 4}


# ---------------------------------------------------------------------------------------------- maps
def _ud(n_ep=40, per=100, beta=2.0, gamma=10.0, seed=0):
    """Synthetic confounded sleep units: s ~ U(0, 1) drives P(accept) (.85 / .15 at s >= .6, the incumbent's
    saving row) AND the outcome (+gamma at s >= .6); true effect beta of accepting (+1 step)."""
    rng = np.random.default_rng(seed)
    n = n_ep * per
    s = rng.uniform(size=n)
    pa = np.where(s >= .6, .15, .85)
    A = (rng.uniform(size=n) < pa).astype(int)
    mode = np.where(A == 1, CU.MODES.index("accept"), CU.MODES.index("reject"))
    probs = np.zeros((n, len(CU.MODES)))
    probs[:, CU.MODES.index("accept")] = pa
    probs[:, CU.MODES.index("reject")] = 1 - pa
    y = beta * A + gamma * (s >= .6) + rng.normal(0, 1, n)
    ep = np.repeat(np.arange(n_ep), per)
    t0 = np.tile(np.arange(per) * 5 + 100, n_ep)
    one = np.ones(n)
    Y = {(r, k): (y if (r, k) == ("nbr", "pv") else np.zeros(n)) for r in CU.RELATIONS for k in CU.KPIS}
    return CU.UnitData(episode=ep, seed=ep, fold=-one.astype(int), family=np.full(n, CU.FAMILIES.index("sleep")),
                       xapp=np.array(["ES"] * n, object), c=np.zeros(n, int), t0=t0, mode=mode, applied=mode,
                       p=probs[np.arange(n), mode], probs=probs, step=one, sgn=one, y=Y,
                       pre={k: rng.normal(0, 1, n) for k in Y}, ycell={}, rel_mask={}, z={}, prev=-one.astype(int),
                       n_cells=1, H=90, H_pre=90, relations=CU.RELATIONS, kpis=CU.KPIS, meta={})


def test_design_slope_is_unbiased_where_the_naive_slope_is_confounded():
    import e6p_opta_kb_analyze as KBA
    bs, ns = [], []
    for sd in range(4):
        ud = _ud(seed=sd)
        bs.append(AN.design_slope(ud, "sleep", "nbr", "pv")["beta"])
        ns.append(KBA.naive_slope(ud, "sleep", "nbr", "pv"))
    assert abs(np.mean(bs) - 2.0) < 0.25
    assert np.mean(ns) < -3.0                                         # confounding flips the naive sign
    ud = _ud(seed=9)
    decl = {("sleep", "nbr", "pv"): {"declared": True, "sign": 1}, ("sleep", "own", "pv"): {"declared": True,
                                                                                           "sign": 0},
            ("sleep", "far", "pv"): {"declared": False, "sign": 1}}
    M, dis = AN.pmrt_map(decl, ud)
    assert list(M) == [("sleep", "nbr", "pv")] and 1.5 < M[("sleep", "nbr", "pv")] < 2.5 and dis == []
    M2, dis2 = AN.pmrt_map({("sleep", "nbr", "pv"): {"declared": True, "sign": -1}}, ud)
    assert M2[("sleep", "nbr", "pv")] < 0 and len(dis2) == 1                  # PMRT sign kept, disagreement counted


def test_granger_n_free_placebo_transfer():
    plc = {("sleep", "nbr", "pv"): {"r2": .02, "sign": 1, "F": 4.1, "df": 200},
           ("sleep", "own", "pv"): {"r2": .01, "sign": 1, "F": 2.0, "df": 200},
           ("ptx", "own", "e"): {"r2": .005, "sign": -1, "F": 1.0, "df": 200}}
    disc = {("sleep", "nbr", "pv"): {"r2": .012, "sign": 1, "F": 24.3, "df": 2000},      # -log10 p grew with n
            ("sleep", "own", "pv"): {"r2": .009, "sign": -1, "F": 18.2, "df": 2000},    # p ~ 2e-5, not declared
            ("ptx", "own", "e"): {"r2": .3, "sign": -1, "F": 857.0, "df": 2000}}
    d, tau = AN.granger_plc_transfer(plc, disc)
    assert tau["tau_r2"] == .01                                                      # <= 1 placebo declaration
    assert {h for h, v in d.items() if v["declared"]} == {("sleep", "nbr", "pv"), ("ptx", "own", "e")}
    assert d[("ptx", "own", "e")]["sign"] == -1 and d[("sleep", "own", "pv")]["sign"] == -1
    dp, _ = AN.granger_plc_transfer(plc, plc)
    assert sum(v["declared"] for v in dp.values()) == 1
    d0, t0 = AN.granger_plc_transfer({}, disc)
    assert t0["tau_r2"] == float("-inf") and sum(v["declared"] for v in d0.values()) == 3
    ud = _ud(seed=1)
    gs = AN.granger_stats(ud)
    h = ("sleep", "nbr", "pv")
    assert gs[h]["df"] == ud.n - 3 and gs[h]["r2"] == pytest.approx(gs[h]["F"] / (gs[h]["F"] + gs[h]["df"]))


def _ctx(knob, step, own=np.nan, nbr=np.nan):
    return {"knob": knob, "step": float(step), MG.PRESSURE["own"]: own, MG.PRESSURE["nbr"]: nbr}


def _decisions(M, ctxs):
    g = MG.MapGateV2(M)
    return [g({"c": c, "ctx": x})[0] for c, x in ctxs]


def test_signature_equal_iff_identical_policies_and_alias_table():
    rng = np.random.default_rng(0)
    ctxs = [(int(rng.integers(3)), _ctx(str(rng.choice(MG.FAMILIES)), float(rng.choice([-1, 1])),
                                        float(rng.uniform(0, .2)), float(rng.uniform(0, .2)))) for _ in range(3000)]
    M1 = {("sleep", "nbr", "pv"): 9.0, ("sleep", "own", "e"): -100.0, ("carrier", "own", "pv"): -6.8,
          ("carrier", "own", "v"): -116.0, ("carrier", "own", "e"): 6000.0, ("ptx", "own", "pv"): 2.0,
          ("ptx", "own", "e"): 4800.0, ("ptx", "nbr", "v"): -31.0}
    M2 = {k: 3.7 * v for k, v in M1.items()}                                          # same signs / ratios
    M3 = {**M1, ("ptx", "nbr", "v"): 0.0}                                       # loses the ptx guard conflict
    s = {n: MG.decision_signature(M) for n, M in (("1", M1), ("2", M2), ("3", M3))}
    assert s["1"] == s["2"] and s["1"] != s["3"]
    assert _decisions(M1, ctxs) == _decisions(M2, ctxs)
    assert _decisions(M1, ctxs) != _decisions(M3, ctxs)
    empty = {("carrier", "own", "e"): 5.0}                                            # no harm anywhere
    assert MG.decision_signature(empty) == MG.all_accept_signature() == MG.decision_signature({})
    assert set(_decisions(empty, ctxs)) == {"accept"} and set(_decisions({}, ctxs)) == {"accept"}
    maps = {"MG:PMRT": M1, "MG:GT": M2, "MG:rand": M3, "blanket2": {}, "MG:corr@dev": {}, "MG:corr@plc": M1}
    al = C.alias_table(maps)
    assert al["alias_of"] == {"MG:PMRT": "MG:PMRT", "MG:GT": "MG:PMRT", "MG:rand": "MG:rand",
                              "blanket2": "blanket2", "MG:corr@dev": "blanket2", "MG:corr@plc": "MG:PMRT"}
    assert al["jobs"] == ["MG:PMRT", "MG:rand", "blanket2"]
    al2 = C.alias_table(maps, noarb_alias=True)
    assert al2["alias_of"]["blanket2"] == "noarb" and al2["jobs"] == ["MG:PMRT", "MG:rand"]


def _disc_json(maps, label="DISC-PASS", mode="full", missing=()):
    return {"schema": AN.SCHEMA_DISC, "label": {"label": label}, "mode": mode, "missing_arms": list(missing),
            "p_mismatch": {"disc": 0, "placebo": 0, "dev": 0},
            "record_checks": {k: {"ok": True} for k in ("dev_conf", "disc", "placebo")},
            "maps": {a: MG.map_to_json(M) for a, M in maps.items()}, "provenance": {a: {"m": a} for a in maps},
            "repro": {"noarb_alias": False}, "protocol": {"frozen": False}, "K0": {"pass": True},
            "K0n": {"pass": True}, "artifact": {"sha256": AN.V4_ARTIFACT_SHA256}, "code_sha256": {},
            "platform": {"keys": []}}


def test_maps_artifact_build_refusals_and_verify_roundtrip(tmp_path):
    maps = C.smoke_maps()
    d = _disc_json(maps)
    art = AN.build_artifact(d, "x" * 64)
    assert set(art["arms"]) == set(C.MAP_ARMS) and art["jobs"] == C.alias_table(maps)["jobs"]
    assert art["mapgate"]["sha256"] == AN.A4.sha_lf(os.path.join(ROOT, AN.MAPGATE))
    for bad in (_disc_json(maps, label="INVALID"), _disc_json(maps, mode="dry-run"),
                _disc_json(maps, missing=["MG:qacm@plc"]), dict(d, p_mismatch={"disc": 1})):
        with pytest.raises(SystemExit):
            AN.build_artifact(bad, "x" * 64)
    assert AN.build_artifact(_disc_json(maps, mode="dry-run"), "x" * 64, allow_dry=True)["dry"]
    dj = tmp_path / "disc_conf.json"
    dj.write_text(json.dumps(d), encoding="utf-8")
    out = tmp_path / "E6P_CONF_MAPS.json"

    class A:
        disc_json, allow_dry = str(dj), False
    A.out = str(out)
    AN.cmd_build(A)
    v = AN.verify_artifact(str(out), str(dj))
    assert v["ok"] and v["sha256"] == AN.A4.sha_lf(str(out)) and not v["driver_matches"]   # MAPS_SHA256 still None
    a = json.loads(out.read_text(encoding="utf-8"))
    a["arms"]["MG:GT"]["map"] = []
    out.write_text(json.dumps(a), encoding="utf-8")
    assert not AN.verify_artifact(str(out), str(dj))["ok"]


# ---------------------------------------------------------------------------------------------- criteria (disc)
def test_x_checks_k1_and_discovery_label_precedence():
    ref = {("sleep", "nbr", "pv"): {"status": "TRUE", "sign": 1}, ("sleep", "nbr", "load"): {"status": "TRUE",
                                                                                              "sign": 1},
           ("sleep", "nbr", "v"): {"status": "TRUE", "sign": 1}, ("ptx", "nbr", "load"): {"status": "TRUE",
                                                                                           "sign": -1},
           ("ptx", "own", "e"): {"status": "TRUE", "sign": 1}, ("carrier", "own", "pv"): {"status": "NULL", "sign": 0},
           ("carrier", "nbr", "v"): {"status": "INDET", "sign": 0}}
    dec = {h: {"declared": True, "sign": s} for h, s in ((("sleep", "nbr", "pv"), 1), (("sleep", "nbr", "load"), 1),
                                                        (("sleep", "nbr", "v"), 1), (("ptx", "own", "e"), 1),
                                                        (("carrier", "nbr", "v"), -1))}
    x = AN.x_checks(dec, ref)
    assert x["X1"]["pass"] and x["X3"]["pass"] and x["X3"]["chain_hits"] == 3
    assert x["X2"]["precision"] == 1.0 and x["X2"]["n_scored"] == 3 and x["X2"]["pass"]   # load / INDET excluded
    dec2 = {**dec, ("carrier", "own", "pv"): {"declared": True, "sign": 1}}
    x2 = AN.x_checks(dec2, ref)
    assert x2["X2"]["precision"] == .75 and not x2["X2"]["pass"]
    dec3 = {**dec, ("ptx", "own", "e"): {"declared": True, "sign": -1}}
    assert AN.x_checks(dec3, ref)["X2"]["sign_accuracy"] == pytest.approx(2 / 3)
    ok, no = {"pass": True}, {"pass": False}
    k1 = AN.k1_check({"sleep_units": 500, "sleep_rejects": 75})
    assert k1["pass"] and not AN.k1_check({"sleep_units": 900, "sleep_rejects": 74})["pass"]
    lab = lambda *a: AN.disc_label(*a)["label"]                                        # noqa: E731
    assert lab(no, ok, no, no, x2) == "INVALID" and lab(ok, None, ok, ok, x) == "INVALID"
    assert lab(ok, ok, no, no, x) == "NO-CHAIN" and lab(ok, ok, ok, no, x) == "UNDERPOWERED"
    assert lab(ok, ok, ok, ok, x) == "DISC-PASS" and lab(ok, ok, ok, ok, x2) == "DISC-FAIL"
    xp = {"X1": no, "X2": ok, "X3": ok}
    assert lab(ok, ok, ok, ok, xp) == "DISC-PARTIAL"


# ---------------------------------------------------------------------------------------------- criteria (eval)
def _R(n=40, seed=0):
    rng = np.random.default_rng(seed)
    base = {"prot_viol": 100.0, "prot_ue_s": 3600.0, "energy_j": 1e6, "viol_ue_s": 50.0, "ue_s": 1e4,
            "nonprot_embb_viol": 10.0, "ll_viol": 5.0, "rlf": 2.0}
    arm = {"noarb": (1.0, 1.0), "freeze": (1.3, 1.0), "sub:ES+PowerES": (1.0, .8), "sub:ES": (.9, .9),
           "sub:PowerES": (.95, .9), "sub:SliceGuarantee": (.6, 1.0), "good": (.8, .8), "costly": (.7, 1.0),
           "guard": (.8, .8)}
    R = {}
    for s in range(n):
        R[s] = {}
        for a, (v, e) in arm.items():
            r = {k: x * (1 + .05 * rng.normal()) for k, x in base.items()}
            r["prot_viol"] *= v
            r["energy_j"] *= e
            if a == "guard":
                r["rlf"] *= 1.5
            R[s][a] = r
    return R


def test_arm_stats_multi_eligibility_and_rstar():
    st = AN.arm_stats_multi(_R(), ["good", "costly", "guard"], n_boot=500)
    P, B = st["point"], st["boot"]
    assert st["n"] == 40 and st["ref_arm"] == "sub:SliceGuarantee"
    assert P["noarb"]["R"] == pytest.approx(0.0) and P["sub:SliceGuarantee"]["R"] == pytest.approx(1.0)
    assert P["good"]["eligible"] and P["good"]["Rstar"] == P["good"]["R"] > 0
    assert not P["costly"]["matched"] and P["costly"]["Rstar"] == 0.0 < P["costly"]["R"]
    assert not P["guard"]["guard_ok"] and P["guard"]["Rstar"] == 0.0
    assert np.all(B["costly"]["Rstar"] <= 0) and B["good"]["Rstar"].shape == (500,)
    st2 = AN.arm_stats_multi(_R(), ["good", "costly", "guard"], n_boot=500)
    assert np.array_equal(st2["boot"]["good"]["R"], B["good"]["R"])                    # fixed key [6624, 20, n]


def test_d1_reselects_the_best_map_per_resample_and_d2_holm():
    rng = np.random.default_rng(1)
    nb = 4000
    P = {"p": {"Rstar": .50}, "b1": {"Rstar": .30}, "b2": {"Rstar": .20}}
    B = {"p": {"Rstar": .5 + .01 * rng.normal(size=nb)}, "b1": {"Rstar": .3 + .01 * rng.normal(size=nb)},
         "b2": {"Rstar": np.where(rng.uniform(size=nb) < .5, .6, .0)}}
    d1 = AN.d1_check(P, B, "p", ["b1", "b2"])
    assert d1["best_point"] == "b1" and d1["delta1"] == pytest.approx(.2)
    assert d1["lb90"] < 0 and not d1["pass"]                                            # b2 wins half the resamples
    assert .4 < d1["best_selected_share"]["b2"] < .6
    naive = np.quantile(B["p"]["Rstar"] - B["b1"]["Rstar"], .05)
    assert naive > 0                                                                     # a fixed-best LB would pass
    B["b2"]["Rstar"] = .1 + .01 * rng.normal(size=nb)
    assert AN.d1_check(P, B, "p", ["b1", "b2"])["pass"]
    P["b1"]["Rstar"] = .45
    assert not AN.d1_check(P, B, "p", ["b1", "b2"])["pass"]                             # delta1 < .10
    assert AN.holm({"a": .01, "b": .02, "c": .04}) == {"a": True, "b": True, "c": True}
    assert AN.holm({"a": .01, "b": .03, "c": .04}) == {"a": True, "b": False, "c": False}
    assert AN.holm({"a": .06}) == {"a": False}
    sig = {"p": "S0", "b1": "S1", "b2": "S2", "b3": "S2"}
    B = {"p": {"Rstar": .5 + .01 * rng.normal(size=nb)}, "b1": {"Rstar": .01 * rng.normal(size=nb)},
         "b2": {"Rstar": .02 * rng.normal(size=nb)}}
    B["b3"] = B["b2"]
    d2 = AN.d2_check(B, "p", ["b1", "b2", "b3"], sig)
    assert d2["pass"] and d2["n_hypotheses"] == 2 and d2["hypotheses"]["b2"]["members"] == ["b2", "b3"]
    B["b4"] = B["p"]
    d2b = AN.d2_check(B, "p", ["b1", "b2", "b3", "b4"], dict(sig, b4="S0"))
    assert not d2b["pass"] and d2b["hypotheses"]["b4"]["p"] == 1.0 and d2b["hypotheses"]["b4"][
        "same_signature_as_primary"]


def test_eval_verdict_precedence_and_not_a_verdict_label():
    V = AN.eval_verdict
    assert V(False, True, True, True, True, True)["verdict"] == "INVALID"
    assert V(True, False, True, True, True, True)["verdict"] == "INVALID"                 # K0n is required
    assert V(True, True, False, True, True, True)["verdict"] == "NOT ELIGIBLE"
    assert V(True, True, True, True, True, True)["label"] == "PASS"
    assert V(True, True, True, True, False, True)["verdict"] == "PARTIAL"
    assert V(True, True, True, False, True, True)["verdict"] == "PARTIAL"
    assert V(True, True, True, False, False, True)["verdict"] == "FAIL"
    v = V(True, True, True, True, True, False, ["smoke records"])
    assert v["verdict"] == "PASS" and v["label"].startswith("NOT A VERDICT")


def test_load_eval_expands_aliases(tmp_path):
    recs = [{"kind": "job", "schema": C.SCHEMA_EVAL, "sub": C.SUB, "conf_stage": "eval", "seed": 187000,
             "arm": a, "prot_viol": 1.0, "smoke": False} for a in ("MG:PMRT", "blanket2")]
    f = tmp_path / "e.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    R, _ = AN.load_eval([str(f)], {"MG:PMRT": "MG:PMRT", "MG:GT": "MG:PMRT", "blanket2": "blanket2",
                                   "MG:corr@dev": "noarb"})
    assert R[187000]["MG:GT"]["alias_of"] == "MG:PMRT" and "MG:corr@dev" not in R[187000]


def test_driver_sha_ignores_maps_pin(tmp_path):
    """freeze 2 is circular (the maps artifact pins the driver, the driver pins the maps artifact): the driver's hash
    must not depend on the MAPS_SHA256 line, and must still depend on everything else."""
    d = tmp_path / AN.DRIVER
    d.parent.mkdir(parents=True, exist_ok=True)
    src = open(os.path.join(ROOT, AN.DRIVER), encoding="utf-8").read()
    d.write_text(re.sub(r"^MAPS_SHA256 = .*$", "MAPS_SHA256 = None", src, count=1, flags=re.M), encoding="utf-8")
    h0 = AN.driver_sha(str(tmp_path))
    d.write_text(re.sub(r"^MAPS_SHA256 = .*$", 'MAPS_SHA256 = "' + "a" * 64 + '"', src, count=1, flags=re.M),
                 encoding="utf-8")
    assert AN.driver_sha(str(tmp_path)) == h0
    d.write_text(src + "\n# changed\n", encoding="utf-8")
    assert AN.driver_sha(str(tmp_path)) != h0

"""Tests of the Experiment C runtime table (scratchpad/xmethod/exp_c_runtime.py; PROTOCOL_A s.7, T3; R-13, R-51)."""
from __future__ import annotations

import gzip
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("exp_c_runtime",
                                               os.path.join(ROOT, "scratchpad", "xmethod", "exp_c_runtime.py"))
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)

REF = X.REF_HOST
LIGHT = "lightning|Intel(R) Xeon(R) Platinum 8488C"


def rec(arm, n, seed, cpu, host=REF, world="E2", regime="R2", lam=None, kappa=.25, status="ok", rss=300.0,
        wall=None, **kw):
    plat, cpu_model = host.split("|", 1)
    key = "|".join([arm, world, regime, *([f"lam{lam:g}"] if lam is not None else []), f"k{kappa:g}", f"n{n}",
                    f"s{seed}"])
    r = {"key": key, "arm": arm, "method": arm.split("_")[0], "status": status, "role": "measure",
         "job": {"world": world, "regime": regime, "lam": lam, "n": n, "seed": seed, "kappa": kappa},
         "cpu_s": cpu, "wall_s": cpu if wall is None else wall, "peak_rss_mb": rss, "rss_scope": "job",
         "gen_cpu_s": .01, "host": {"platform": plat, "cpu_model": cpu_model}, "spec_name": "t",
         "code": {"commit": "abc"}}
    r.update(kw)
    return r


def write(tmp_path, name, recs, gz=False):
    p = tmp_path / name
    op = gzip.open if gz else open
    with op(p, "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return str(p)


def test_stats_and_rss_reference_only(tmp_path):
    p = write(tmp_path, "a.jsonl.gz", [rec("pcorr_eq", 500, s, c, rss=r) for s, c, r in
                                       ((1, 1.0, 300), (2, 2.0, 310), (3, 6.0, 320))], gz=True)
    recs, info = X.load([p])
    out = X.build(recs, info)
    row = out["table"][0]
    assert row["cost_ref"] == {"mean": 3.0, "median": 2.0, "min": 1.0, "max": 6.0}
    assert row["peak_rss_mb"]["median"] == 310 and row["peak_rss_mb"]["max"] == 320
    assert row["share_non_ref"] == 0 and out["factors"] == []
    assert row["t3"]["status"] == "feasible"
    assert out["feasibility"]["pcorr_eq"]["by_n"]["1000"]["status"] == "no DEV cost"


def test_observational_factor_converts_other_host():
    # same cell: ref seeds cost 4, Lightning seeds cost 2 -> f = 2; Lightning units count as 4 ref-s
    rs = [X.slim(rec("rcot2_eq", 1000, s, 4.0)) for s in (1, 2)] + \
         [X.slim(rec("rcot2_eq", 1000, s, 2.0, host=LIGHT)) for s in (3, 4)]
    out = X.build(rs, {"files": [], "lines": 4, "duplicates": 0, "conflicts": 0})
    (f,) = out["factors"]
    assert f["host"] == LIGHT and f["f"] == pytest.approx(2.0) and f["source"] == "observational"
    row = out["table"][0]
    assert row["cost_ref"]["max"] == pytest.approx(4.0) and row["cost_raw"]["max"] == pytest.approx(4.0)
    assert row["cost_ref"]["mean"] == pytest.approx(4.0) and row["n_unconverted"] == 0
    assert row["factor_sources"] == ["observational"]


def test_no_factor_leaves_unit_unconverted_and_provisional():
    rs = [X.slim(rec("mscr_eq", 500, 1, 10.0, host=LIGHT))]
    out = X.build(rs, {"files": [], "lines": 1, "duplicates": 0, "conflicts": 0})
    row = out["table"][0]
    assert row["n_unconverted"] == 1 and row["cost_ref"]["max"] == 10.0
    assert out["feasibility"]["mscr_eq"]["by_n"]["500"]["provisional"] is True
    out2 = X.build(rs, {"files": [], "lines": 1, "duplicates": 0, "conflicts": 0}, observational=False)
    assert out2["factors"] == []


def test_explicit_factor_beats_observational_and_wildcard():
    rs = [X.slim(rec("cdl", 4000, 1, 100.0)), X.slim(rec("cdl", 4000, 2, 50.0, host=LIGHT))]
    info = {"files": [], "lines": 2, "duplicates": 0, "conflicts": 0}
    out = X.build(rs, info, factors={("*", LIGHT): {"f": 3.0, "source": "t3_calibration"}})
    assert out["factors"][0]["f"] == 3.0 and out["table"][0]["cost_ref"]["max"] == 150.0
    out = X.build(rs, info, factors={("cdl", LIGHT): {"f": 1.5, "source": "t3_calibration"},
                                     ("*", LIGHT): {"f": 3.0, "source": "t3_calibration"}})
    assert out["table"][0]["cost_ref"]["max"] == 100.0       # 50 x 1.5 < the ref unit's 100


def test_t3_first_infeasible_n_propagates_to_larger_n():
    rs = [X.slim(rec("pdc", n, 1, c)) for n, c in ((500, 10.0), (1000, 8000.0), (4000, 20.0))]
    out = X.build(rs, {"files": [], "lines": 3, "duplicates": 0, "conflicts": 0})
    fe = out["feasibility"]["pdc"]
    assert fe["first_infeasible_n"] == 1000 and fe["first_infeasible_cost"] == 8000.0
    st = {n: e["status"] for n, e in fe["by_n"].items()}
    assert st["500"] == "feasible" and st["1000"] == "infeasible"
    assert st["4000"] == st["24000"] == "infeasible (smaller n infeasible)"
    assert out["spec_t3_fields"] == {"pdc": {"max_n": 500, "t3_cost_cpu_s": {"1000": 8000.0}}}


def test_t3_budget_uses_converted_cost():
    # 4000 raw-s on a host 2x faster than Kaggle = 8000 ref-s > 7200: infeasible
    rs = [X.slim(rec("a", 500, s, 2.0)) for s in (1, 2)] + [X.slim(rec("a", 500, 3, 1.0, host=LIGHT)),
                                                             X.slim(rec("a", 1000, 4, 4000.0, host=LIGHT))]
    out = X.build(rs, {"files": [], "lines": 4, "duplicates": 0, "conflicts": 0})
    assert out["feasibility"]["a"]["first_infeasible_n"] == 1000
    assert out["feasibility"]["a"]["first_infeasible_cost"] == pytest.approx(8000.0)


def test_measured_infeasible_unit_and_errors():
    rs = [X.slim(rec("b", 500, 1, 5.0)),
          X.slim(rec("b", 1000, 1, None, status="infeasible", cost=7300.0, cost_kind="CPU-s")),
          X.slim(rec("b", 1000, 2, None, status="error"))]
    out = X.build(rs, {"files": [], "lines": 3, "duplicates": 0, "conflicts": 0})
    row = next(r for r in out["table"] if r["n"] == 1000)
    assert row["n_infeasible"] == 1 and row["n_errors"] == 1
    assert out["feasibility"]["b"]["first_infeasible_n"] == 1000


def test_gpu_arm_reads_wall_seconds():
    rs = [X.slim(rec("g", 500, 1, 10.0, wall=50.0, child_wall_s=60.0))]
    out = X.build(rs, {"files": [], "lines": 1, "duplicates": 0, "conflicts": 0},
                  arm_specs={"g": {"budget_wall_s": 7200}})
    row = out["table"][0]
    assert row["cost_field"] == "wall_s" and row["cost_ref"]["max"] == 60.0


def test_contention_and_multithread_flags():
    rs = [X.slim(rec("c", 500, 1, 10.0, wall=25.0)), X.slim(rec("c", 500, 2, 10.0, wall=4.0))]
    row = X.build(rs, {"files": [], "lines": 2, "duplicates": 0, "conflicts": 0})["table"][0]
    assert row["n_contended"] == 1 and row["n_multi_thread"] == 1


def test_load_dedupes_and_keeps_cross_host_copies(tmp_path):
    a = write(tmp_path, "a.jsonl", [rec("p", 500, 1, 4.0), rec("p", 500, 2, None, status="error")])
    b = write(tmp_path, "b.jsonl", [rec("p", 500, 1, 2.0, host=LIGHT), rec("p", 500, 2, 3.0)])
    recs, info = X.load([a, b])
    assert len(recs) == 2 and info["duplicates"] == 2 and info["conflicts"] == 1
    by = {r["seed"]: r for r in recs}
    assert by[1]["host"] == REF and by[2]["status"] == "ok"       # ok replaces the error copy
    assert set(info["copies"]) == {by[1]["key"], by[2]["key"]}


def test_plan_and_calib_round_trip(tmp_path):
    rs = []
    for w, rg, c in (("E1", "R1", 1.0), ("E2", "R2", 9.0), ("E3", "R2", 5.0), ("E5", "R1", 7.0)):
        rs += [rec("q", 4000, s, c, world=w, regime=rg) for s in X.CALIB_SEEDS]
        rs.append(rec("q", 500, 3_000_005, c / 10, world=w, regime=rg, host=LIGHT))
    recs, _ = X.load([write(tmp_path, "r.jsonl", rs)])
    plan = X.calibration_plan([X.slim(r) for r in rs], REF)
    d = plan["arms"]["q"]
    assert d["n"] == 4000 and d["hosts_needing_factor"] == [LIGHT]
    assert [(u["world"], u["seed"]) for u in d["units"][::3]] == [("E2", 3_000_000), ("E5", 3_000_000),
                                                                  ("E3", 3_000_000)]
    assert all(u["in_records"] for u in d["units"])
    # the plan's units re-run on Lightning, 2x faster
    host = [X.slim(rec("q", 4000, u["seed"], cost, world=u["world"], regime=u["regime"], host=LIGHT))
            for u, cost in zip(d["units"], [4.5] * 3 + [3.5] * 3 + [2.5] * 3)]
    cal = X.calibrate([X.slim(r) for r in rs] + host, REF, plan)
    f = cal["factors"]["q"][LIGHT]
    assert f["source"] == "t3_calibration" and f["n_units"] == 9 and f["f"] == pytest.approx(2.0)
    assert f["f_lo"] == pytest.approx(2.0) and f["f_hi"] == pytest.approx(2.0)
    assert cal["factors"]["*"][LIGHT]["arms"] == ["q"]
    assert cal["missing"] == []
    p = tmp_path / "f.json"
    p.write_text(json.dumps(cal))
    assert X.load_factor_file(str(p))[("q", LIGHT)]["f"] == pytest.approx(2.0)


def test_cli_table_writes_json_and_md(tmp_path):
    p = write(tmp_path, "m.jsonl", [rec("corr", 500, 1, .1), rec("corr", 500, 2, .2, host=LIGHT)])
    assert X.main(["table", "--records", p, "--no-observational", "--out-dir", str(tmp_path / "o")]) == 0
    out = json.load(open(tmp_path / "o" / "exp_c_runtime.json"))
    assert out["version"] == X.VERSION and out["table"][0]["n_unconverted"] == 1
    md = (tmp_path / "o" / "EXP_C_RUNTIME.md").read_text(encoding="utf-8")
    assert "| corr |" in md and "unconverted" in md


def test_factor_error_counts_as_over_budget():
    # central 7000 ref-s is under 7200, but f_hi 1.1 puts it at 7700: infeasible within the factor's error (R-55)
    rs = [X.slim(rec("e", 4000, 1, 3500.0, host=LIGHT))]
    out = X.build(rs, {"files": [], "lines": 1, "duplicates": 0, "conflicts": 0},
                  factors={("e", LIGHT): {"f": 2.0, "f_hi": 2.2, "source": "r55_calibration_block"}})
    row = out["table"][0]
    assert row["cost_ref"]["max"] == pytest.approx(7000.0) and row["cost_ref_hi_max"] == pytest.approx(7700.0)
    assert out["feasibility"]["e"]["first_infeasible_n"] == 4000
    assert row["t3"]["status"].startswith("infeasible (within the speed factor's error")


def test_calib_block_pooled_anchors_and_dataset_hash(tmp_path):
    rs = []
    for arm, ratio in (("a1", 1.5), ("a2", 2.5)):
        for n in (500, 1000):
            for sd in X.CALIB_SEEDS:
                rs.append(X.slim(rec(arm, n, sd, 10.0, spec_name="exp_c_calib", dataset_sha256=f"h{n}{sd}")))
                rs.append(X.slim(rec(arm, n, sd, 10.0 / ratio, host=LIGHT, spec_name="exp_c_calib",
                                     dataset_sha256=f"h{n}{sd}" if arm == "a1" else "other")))
    cal = X.calibrate(rs, REF)
    assert cal["source"] == "r55_calibration_block"
    assert cal["factors"]["a1"][LIGHT]["n_same_dataset"] == 6
    assert cal["factors"]["a2"][LIGHT]["n_different_dataset"] == 6
    pooled = cal["factors"]["*"][LIGHT]
    assert pooled["f_lo"] == pytest.approx(1.5) and pooled["f_hi"] == pytest.approx(2.5)
    assert pooled["f"] == pytest.approx(120 / (60 / 1.5 + 60 / 2.5))
    # a non-anchor arm takes the pooled factor and its spread
    p = tmp_path / "f.json"
    p.write_text(json.dumps(cal))
    out = X.build([X.slim(rec("other", 500, 1, 100.0, host=LIGHT))],
                  {"files": [], "lines": 1, "duplicates": 0, "conflicts": 0}, factors=X.load_factor_file(str(p)))
    assert out["table"][0]["cost_ref_hi_max"] == pytest.approx(250.0)


def test_calib_two_host_sessions_pooled_with_spread():
    rs = []
    for n in (500, 1000):
        rs.append(X.slim(rec("a", n, 1, 10.0, spec_name="exp_c_calib")))
        for node, c in (("s1", 5.0), ("s2", 10.0 / 3)):         # session f 2.0 and 3.0
            r = rec("a", n, 1, c, host=LIGHT, spec_name="exp_c_calib")
            r["host"]["node"] = node
            rs.append(X.slim(r))
    d = X.calibrate(rs, REF)["factors"]["a"][LIGHT]
    assert d["f"] == pytest.approx(20 / (2 * (5.0 + 10.0 / 3) / 2))   # host cost = mean over sessions
    assert d["f_by_session"] == pytest.approx({"s1": 2.0, "s2": 3.0})
    assert d["f_lo"] == pytest.approx(2.0) and d["f_hi"] == pytest.approx(3.0) and d["n_units"] == 2


def test_procs_per_vcpu_from_load_fields():
    r = rec("l", 500, 1, 1.0, load={"start": {"campaign_procs": 8, "cpu_quota": 4}})
    t = rec("l", 500, 2, 1.0, load={"start": {"timing_procs": 2, "cpu_quota": 2}})
    out = X.build([X.slim(r), X.slim(t)], {"files": [], "lines": 2, "duplicates": 0, "conflicts": 0})
    assert out["table"][0]["procs_per_vcpu"] == {"median": 1.5, "max": 2.0}


# ------------------------------------------------------------------------------------------- timing driver
_tspec = importlib.util.spec_from_file_location("exp_c_timing",
                                                os.path.join(ROOT, "scratchpad", "xmethod", "exp_c_timing.py"))
T = importlib.util.module_from_spec(_tspec)
_tspec.loader.exec_module(T)


def test_timing_spec_blocks():
    sp = T.load_spec()
    cal = T.units(sp, "calib")
    assert {u["arm"] for u in cal} == set(T.CALIB_ANCHORS)
    assert all((u["world"], u["regime"], u["kappa"]) == ("E2", "R2", .25) for u in cal)
    assert {u["seed"] for u in cal} == set(T.SEEDS) and max(u["n"] for u in cal) == 4000
    assert max(u["n"] for u in cal if u["arm"] == "mscr_eq_min") == 1000           # R-54
    assert len(cal) == 5 * 3 * 3 + 2 * 3
    pap = T.units(sp, "paper")
    assert not any(u["arm"].startswith(("pdcor", "cmi_knn")) for u in pap)         # R-48, R-49
    assert {"pmrt_nl_eq", "cdl"} <= {u["arm"] for u in pap}
    assert {u["world"] for u in pap if u["arm"].startswith("granger")} == {"E3"}
    assert len({u["key"] for u in pap}) == len(pap)


def test_timing_assign_balances_and_keeps_every_unit():
    us = [{"key": str(i), "est_cost_s": c} for i, c in enumerate([10, 9, 8, 1, 1, 1, 1])]
    bins = T.assign(us, 2)
    assert sorted(u["key"] for b in bins for u in b) == sorted(u["key"] for u in us)
    loads = sorted(sum(u["est_cost_s"] for u in b) for b in bins)
    assert loads == [14, 17] and loads[1] - loads[0] <= 10       # LPT: gap at most the largest unit


def test_cpu_quota_reads_cgroup_v2(tmp_path):
    (tmp_path / "cpu.max").write_text("200000 100000")
    assert T.cpu_quota(str(tmp_path)) == min(2, T.cpu_quota(str(tmp_path / "none")))


def test_cpu_quota_reads_own_scope_cgroup(tmp_path, monkeypatch):
    # systemd-run --scope -p CPUQuota=700% puts cpu.max on the scope, not on the root (VPS lane)
    scope = tmp_path / "system.slice" / "cdd-xm-x.scope"
    scope.mkdir(parents=True)
    (scope / "cpu.max").write_text("700000 100000")
    monkeypatch.setattr(T, "_own_cgroup_dirs", lambda cg: [str(scope), str(tmp_path / "system.slice"), cg])
    assert T.cpu_quota(str(tmp_path)) == min(7, T.cpu_quota(str(tmp_path / "none")))

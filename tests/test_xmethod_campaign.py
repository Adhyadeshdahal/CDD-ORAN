"""Tests of the DEV campaign driver (cdd_oran/xmethod/campaign.py): units, partition, run / resume, budget,
merge (missing / duplicate / broken shards) and aggregate (tau from tune seeds only, rates, CIs)."""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pytest

from cdd_oran.xmethod import campaign as C
from cdd_oran.xmethod.worlds import truth_for

DUMMY = "cdd_oran.xmethod.runner:DummyMethod"


def _spec(**kw):
    s = {"name": "t", "budget_cpu_s": None,
         "arms": {"d_by": {"ref": DUMMY, "declare": "by"},
                  "d_tau": {"ref": DUMMY, "declare": "tau"},
                  "d_e3": {"ref": DUMMY, "declare": "by", "worlds": ["E3"], "max_n": 300}},
         "blocks": [{"role": "tune", "worlds": ["E1", "E4"], "regimes": ["R1", "R3"], "ns": [300, 400],
                     "kappas": [0.25], "seeds": [3000000, 3000002]},
                    {"role": "measure", "worlds": ["E1", "E4"], "regimes": ["R1", "R3"], "ns": [300, 400],
                     "kappas": [0.25], "seeds": [3000100, 3000103]}]}
    s.update(kw)
    return s


# --------------------------------------------------------------------------------------------- units
def test_expand_counts_restrictions_and_lams():
    units = C.expand(_spec())
    # E1: R1 only (R3 undefined) ; E4: R1 lam 1 (inert), R3 four lams -> 1 + 1 + 4 = 6 (world, regime, lam) cells
    # x 2 n x (3 tune + 4 measure seeds) x 2 arms (d_e3 is E3-only)
    assert len(units) == 6 * 2 * 7 * 2
    assert len({u.key for u in units}) == len(units)
    assert {u.lam for u in units if u.world == "E4" and u.regime == "R3"} == {0.0, 0.5, 1.0, 1.5}
    assert {u.lam for u in units if u.world == "E4" and u.regime == "R1"} == {1.0}
    assert {u.lam for u in units if u.world == "E1"} == {None}
    assert not [u for u in units if u.arm == "d_e3"]
    s = _spec()
    s["blocks"][1]["worlds"] = ["E3"]
    e3 = [u for u in C.expand(s) if u.arm == "d_e3"]
    assert e3 and {u.n for u in e3} == {300}                      # max_n
    assert {u.role for u in units if u.seed < 3000100} == {"tune"}


def test_expand_dedups_across_blocks_and_validates():
    s = _spec()
    s["blocks"].append(dict(s["blocks"][1]))
    assert len(C.expand(s)) == len(C.expand(_spec()))
    bad = _spec()
    bad["blocks"][1]["seeds"] = [3000002, 3000003]                 # overlaps tune
    with pytest.raises(ValueError, match="overlap"):
        C.expand(bad)
    ev = _spec()
    ev["blocks"][1]["seeds"] = [3100000]
    with pytest.raises(ValueError, match="EVAL seeds .* refused"):
        C.expand(ev)
    nk = _spec()
    del nk["blocks"][0]["kappas"]
    with pytest.raises(ValueError, match="kappas"):
        C.expand(nk)
    nd = _spec()
    nd["arms"]["d_by"].pop("declare")
    with pytest.raises(ValueError, match="declare"):
        C.expand(nd)


def test_partition_whole_datasets_exact_cover_deterministic():
    units = C.expand(_spec())
    parts = C.partition(units, 5)
    flat = [u.key for p in parts for u in p]
    assert sorted(flat) == sorted(u.key for u in units)
    owner = {}
    for i, p in enumerate(parts):
        for u in p:
            assert owner.setdefault(u.dataset, i) == i
    assert [[u.key for u in p] for p in C.partition(units, 5)] == [[u.key for u in p] for p in parts]
    ct = {f"d_by|E1|R1|n{n}": 100.0 for n in (300, 400)}
    loads = [sum(C.unit_cost(u, ct) for u in p) for p in C.partition(units, 3, ct)]
    assert max(loads) - min(loads) <= 100.0 + 0.4 * 2          # LPT balance within one group's cost


# --------------------------------------------------------------------------------------------- run
def _small_spec():
    return _spec(blocks=[{"role": "tune", "worlds": ["E4"], "regimes": ["R1"], "ns": [300], "kappas": [0.25],
                          "seeds": [3000000, 3000001]},
                         {"role": "measure", "worlds": ["E4"], "regimes": ["R1"], "ns": [300, 400],
                          "kappas": [0.25], "seeds": [3000100, 3000101]}])


def test_run_resumable_and_provenance(tmp_path):
    spec = _small_spec()
    out = tmp_path / "r.jsonl"
    n_units = len(C.expand(spec))
    assert C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None) == n_units
    assert C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None) == 0
    recs = C.load_jsonl(str(out))
    assert len(recs) == n_units
    r = recs[0]
    assert r["status"] == "ok" and r["arm"] in spec["arms"] and r["role"] in C.ROLES
    assert "commit" in r["code"] and r["pkgs"]["numpy"] == np.__version__ and r["host"]["platform"]
    assert r["key"].startswith(r["arm"] + "|E4|R1|lam1|k0.25|n")
    # an interrupted last line is tolerated and the unit is redone
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][: len(lines[-1]) // 2])
    assert C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None) == 1


def test_budget_marks_infeasible_and_cascades(tmp_path, monkeypatch):
    spec = _small_spec()
    out = tmp_path / "b.jsonl"
    real = C.R.run_one
    monkeypatch.setattr(C.R, "run_one", lambda *a, **k: {**real(*a, **k), "cpu_s": 5.0})   # every unit "costs" 5 s
    C.run_part(spec, 0, 1, str(out), budget=1.0, isolate=False, log=lambda *_: None)
    recs = {r["key"]: r for r in C.load_jsonl(str(out))}
    assert all(r["status"] == "infeasible" for r in recs.values())
    reasons = [r["reason"] for r in recs.values()]
    assert any(x.startswith("exceeded the budget") for x in reasons)
    assert any(x.startswith("not run: n 300") for x in reasons)     # larger n skipped, recorded
    assert C.run_part(spec, 0, 1, str(out), budget=1.0, isolate=False, log=lambda *_: None) == 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork + RLIMIT_CPU isolation is Linux-only")
def test_isolated_run_matches_in_process(tmp_path):
    spec = _small_spec()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    C.run_part(spec, 0, 1, str(a), isolate=False, log=lambda *_: None)
    C.run_part(spec, 0, 1, str(b), isolate=True, log=lambda *_: None)
    ra, rb = {r["key"]: r for r in C.load_jsonl(str(a))}, {r["key"]: r for r in C.load_jsonl(str(b))}
    assert ra.keys() == rb.keys()
    for k in ra:
        assert ra[k]["edges"] == rb[k]["edges"] and rb[k]["child_cpu_s"] >= 0


# --------------------------------------------------------------------------------------------- merge
def test_merge_missing_duplicate_broken(tmp_path):
    spec = _small_spec()
    full = tmp_path / "full.jsonl"
    C.run_part(spec, 0, 1, str(full), isolate=False, log=lambda *_: None)
    recs = C.load_jsonl(str(full))
    p1, p2 = tmp_path / "p1.jsonl", tmp_path / "p2.jsonl"
    err = dict(recs[0], status="error", error="boom", edges=[])
    inf = dict(recs[1], status="infeasible", reason="x", edges=[])
    p1.write_text("\n".join(json.dumps(r) for r in [err, inf] + recs[2:5]) + "\n{broken\n")
    p2.write_text("\n".join(json.dumps(r) for r in recs[:4]) + "\n")   # duplicates; recs[5:] missing
    s = C.merge(spec, [str(p1), str(p2)], str(tmp_path / "m.jsonl"))
    merged = {r["key"]: r for r in C.load_jsonl(str(tmp_path / "m.jsonl"))}
    assert s["n_records"] == 5 and s["bad_lines"] == 1 and s["duplicates"] == 4
    assert merged[recs[0]["key"]]["status"] == "ok"                 # ok beats error
    assert merged[recs[1]["key"]]["status"] == "ok"                 # ok beats infeasible
    assert set(s["missing"]) == {r["key"] for r in recs[5:]}
    assert not s["errors"] and not s["conflicts"]
    # infeasible beats error; conflicting declarations on the same dataset are reported
    flip = json.loads(json.dumps(recs[2]))
    flip["edges"][0]["declared"] = not flip["edges"][0]["declared"]
    p3 = tmp_path / "p3.jsonl"
    p3.write_text("\n".join(json.dumps(r) for r in [inf, err, flip]) + "\n")
    best, st = C.merge_records([str(p1), str(p3)])
    assert best[recs[1]["key"]]["status"] == "infeasible" and st["conflicts"] == [recs[2]["key"]]


# --------------------------------------------------------------------------------------------- aggregate
def test_cluster_ci_and_wilson():
    lo, hi = C.cluster_ci(np.array([1, 0, 2, 1]), np.array([20, 20, 20, 20]))
    assert lo <= 4 / 80 <= hi
    assert C.cluster_ci(np.zeros(5), np.full(5, 10)) == [0.0, 0.0]
    w = C.wilson_deff(np.zeros(5), np.full(5, 10))
    assert w[0] == 0.0 and 0 < w[1] < 0.1
    r = C._rate(np.full(10, 5), np.full(10, 10))
    assert r["rate"] == 0.5 and r["above"]
    assert not C._rate(np.ones(10), np.full(10, 20))["above"]


def test_aggregate_tau_from_tune_seeds_only(tmp_path):
    spec = _small_spec()
    out = tmp_path / "r.jsonl"
    C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None)
    recs = C.load_jsonl(str(out))
    agg = C.aggregate(recs, spec)
    c = agg["cells"]["d_tau|E4|R1|lam1|k0.25|n300"]
    tune = [C.to_result(r) for r in recs if r["arm"] == "d_tau" and r["role"] == "tune" and r["job"]["n"] == 300]
    from cdd_oran.xmethod.score import placebo_tau
    assert c["tau"] == placebo_tau(tune) and c["n_tune"] == 2 and c["n_measure"] == 2
    # declarations of the tau arm on measurement seeds = score > tau, scored against truth
    truth = truth_for("E4", "R1")
    m = [r for r in recs if r["arm"] == "d_tau" and r["role"] == "measure" and r["job"]["n"] == 300]
    exp_plac = sum(1 for r in m for e in r["edges"] if e["source"] == "P_placebo" and e["score"] > c["tau"])
    assert c["primary"]["plac_decl"]["hits"] == exp_plac
    assert c["primary"]["plac_decl"]["n"] == 2 * sum(1 for e in truth.null_edges if e[0] == "P_placebo")
    # n 400 has no tune seeds: the tau arm is untuned there, never tuned on measurement seeds
    u = agg["cells"]["d_tau|E4|R1|lam1|k0.25|n400"]
    assert u["tau"] is None and u["primary"] is None and "untuned" in u["note"]
    # BY arm: raw-p rates and the record's own declarations
    b = agg["cells"]["d_by|E4|R1|lam1|k0.25|n300"]
    raw = sum(1 for r in recs if r["arm"] == "d_by" and r["role"] == "measure" and r["job"]["n"] == 300
              for e in r["edges"] if (e["source"], e["target"]) in truth.null_edges
              and e["source"] not in ("P_placebo",) and not e["source"].startswith("K") and e["p"] <= .05)
    assert b["primary"]["null_raw05"]["hits"] == raw
    assert set(b["primary"]["per_seed"]) == {"3000100", "3000101"}
    assert b["primary"]["valid_flag"] == "few_seeds"              # < MIN_FLAG_SEEDS: no flag from 2 clusters
    assert "secondary_tau" in b
    assert agg["cost"]["d_by|E4|R1|n300"]["n_runs"] == 4


def test_project_counts_and_caps(tmp_path):
    spec = _small_spec()
    out = tmp_path / "r.jsonl"
    C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None)
    agg = C.aggregate(C.load_jsonl(str(out)), spec)
    pr = C.project(agg, spec)
    assert pr["n_units"] == len(C.expand(spec)) and pr["cpu_h_total"] >= 0 and not pr["arms_without_cost"]
    for v in agg["cost"].values():
        v["cpu_s_mean"] = 10.0
    capped = C.project(agg, {**spec, "budget_cpu_s": 1.0})
    assert sum(capped["units_capped_at_budget"].values()) == pr["n_units"]
    assert math.isclose(capped["cpu_h_total"] - capped["cpu_h_generation"], pr["n_units"] / 3600)


# --------------------------------------------------------------------------------------------- power (step 4)
def test_paired_seeds_and_null_precision():
    from cdd_oran.xmethod import dev_power as P
    assert P.paired_seeds(0.0) == 2
    s = P.paired_seeds(0.3)                                       # normal approx (2.80 * 2)^2 = 31.4 -> t: 34
    assert 32 <= s <= 36
    assert P.paired_seeds(0.6) > s
    assert P.null_rate_seeds(1, 1.0) == 457 and P.null_rate_seeds(16, 1.0) == 29
    assert P.null_rate_seeds(0) is None
    assert P.design_effect([0, 0, 0], [5, 5, 5]) == 1.0
    assert P.design_effect([5, 0, 5, 0], [5, 5, 5, 5]) > 1.0


def test_recall_power_uses_valid_paired_seeds():
    from cdd_oran.xmethod import dev_power as P

    def cell(arm, recalls, flag="ok"):
        return {"arm": arm, "primary": {"valid_flag": flag, "per_seed": {str(i): {"recall": r} for i, r in
                                                                         enumerate(recalls)}}}
    agg = {"cells": {"pmrt_eq|E1|R1|k0.25|n500": cell("pmrt_eq", [1, .5, 1, .75]),
                     "corr|E1|R1|k0.25|n500": cell("corr", [.5, .5, .75, .25]),
                     "bad|E1|R1|k0.25|n500": cell("bad", [0, 0, 0, 0], "ABOVE")}}
    out = P.recall_power(agg)
    c = out["cells"]["E1|R1|k0.25|n500"]
    assert c["invalid_arms"] == ["bad"] and len(c["pairs"]) == 1
    pr = c["pairs"][0]
    assert math.isclose(abs(pr["mean_d"]), 0.3125) and pr["seeds_needed"] == P.paired_seeds(pr["sd_d"])
    assert out["max_over_cells_focal"] == pr["seeds_needed"]


def test_post_hoc_tau_equals_method_tune_per_arm(tmp_path):
    """The driver never calls Method.tune(); its post-hoc tau per (arm, cell) must equal
    ClassicBase.tune(dev, config={'arm': arm}) on the same tune seeds (fix-classic2: tune respects the arm)."""
    from cdd_oran.xmethod.methods.pc import PC
    from cdd_oran.xmethod.worlds import generate_dataset
    ref = "cdd_oran.xmethod.methods.pc:PC"
    spec = _spec(arms={"pc_eq": {"ref": ref, "config": {"arm": "eq"}, "declare": "tau"},
                       "pc_native": {"ref": ref, "config": {"arm": "native"}, "declare": "tau"}},
                 blocks=[{"role": "tune", "worlds": ["E4"], "regimes": ["R2"], "ns": [300], "kappas": [0.25],
                          "seeds": [3000000, 3000002]},
                         {"role": "measure", "worlds": ["E4"], "regimes": ["R2"], "ns": [300], "kappas": [0.25],
                          "seeds": [3000100]}])
    out = tmp_path / "r.jsonl"
    C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None)
    agg = C.aggregate(C.load_jsonl(str(out)), spec)
    dev = [generate_dataset("E4", "R2", 300, s, lam=1.0, kappa=0.25)[0] for s in (3000000, 3000001, 3000002)]
    for arm in ("eq", "native"):
        want = PC().tune(dev, config={"arm": arm})
        assert want["arm"] == arm
        assert agg["cells"][f"pc_{arm}|E4|R2|lam1|k0.25|n300"]["tau"] == want["tau"]


# --------------------------------------------------------------------------------------------- EVAL guard
def _eval_spec(sha=None):
    s = _small_spec()
    s["blocks"][1]["seeds"] = [3100000, 3100001]                  # EVAL block (never generated in these tests)
    if sha is not None:
        s["protocol_sha256"] = sha
    return s


def _protocol(tmp_path, monkeypatch, frozen: bool, crlf: bool = False):
    text = "# Protocol\n\n" + ("FROZEN: yes (2026-10-04)\n" if frozen else "FROZEN: no (DRAFT)\n") + "body\n"
    p = tmp_path / "PROTOCOL_A.md"
    p.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode())
    monkeypatch.setattr(C, "PROTOCOL_PATH", str(p))
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()               # LF-normalised hash


def test_eval_guard_accepts_frozen_matching_protocol(tmp_path, monkeypatch):
    sha = _protocol(tmp_path, monkeypatch, frozen=True, crlf=True)  # CRLF on disk, LF hash: accepted
    assert C.protocol_sha256() == sha
    units = C.expand(_eval_spec(sha))
    assert {u.seed for u in units if u.role == "measure"} == {3100000, 3100001}
    assert C.eval_authorised(_eval_spec(sha.upper()))[0]


@pytest.mark.parametrize("case", ["no_sha", "wrong_sha", "not_frozen", "missing_file"])
def test_eval_guard_refuses(tmp_path, monkeypatch, case):
    sha = _protocol(tmp_path, monkeypatch, frozen=case != "not_frozen")
    if case == "missing_file":
        monkeypatch.setattr(C, "PROTOCOL_PATH", str(tmp_path / "nope.md"))
    spec = _eval_spec(None if case == "no_sha" else "0" * 64 if case == "wrong_sha" else sha)
    with pytest.raises(ValueError, match="EVAL seeds .* refused"):
        C.expand(spec)
    with pytest.raises(ValueError, match="refused"):
        C.run_part(spec, 0, 1, str(tmp_path / "x.jsonl"), isolate=False, log=lambda *_: None)
    assert not (tmp_path / "x.jsonl").exists()


def test_eval_guard_real_draft_protocol_refused_and_dev_unchanged():
    import hashlib
    import os
    if not os.path.exists(C.PROTOCOL_PATH):
        pytest.skip("docs/xmethod/PROTOCOL_A.md not in this bundle")
    sha = hashlib.sha256(open(C.PROTOCOL_PATH, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    ok, why = C.eval_authorised(_eval_spec(sha))
    assert not ok and "not frozen" in why                           # PROTOCOL_A is a DRAFT today
    assert C.expand(_small_spec()) == C.expand({**_small_spec(), "protocol_sha256": "0" * 64})  # DEV: no guard
    bad = _small_spec()
    bad["blocks"][1]["seeds"] = [3000200]                            # between DEV and EVAL: always refused
    with pytest.raises(ValueError, match="DEV block"):
        C.expand(bad)


# --------------------------------------------------------------------------------------------- denominators
def test_rate_denominators_exclude_not_testable_like_eval_analysis(tmp_path):
    import importlib.util
    import os
    spec = _spec(blocks=[{"role": "measure", "worlds": ["E1"], "regimes": ["R1"], "ns": [300], "kappas": [0.25],
                          "seeds": [3000100]}])
    out = tmp_path / "r.jsonl"
    C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None)
    r = next(x for x in C.load_jsonl(str(out)) if x["arm"] == "d_by")
    truth = truth_for("E1", "R1")
    null = sorted(e for e in truth.null_edges if not e[0].startswith(("K", "P_")))[0]
    plac = sorted(e for e in truth.null_edges if e[0] == "P_placebo")[0]
    nt = [f"{s}->{t}" for s, t in (null, plac)]
    for e in r["edges"]:                                             # mark both not testable (R-22), declared
        if f"{e['source']}->{e['target']}" in nt:
            e.update(score=None, p=None, declared=True)
    r["notes"] = {"not_testable_edges": nt}
    row = C.seed_row(r, C.to_result(r), truth)
    n_null = sum(1 for e in truth.null_edges if not e[0].startswith(("K", "P_")))
    n_plac = sum(1 for e in truth.null_edges if e[0] == "P_placebo")
    assert row["n_null"] == n_null - 1 and row["n_plac"] == n_plac - 1
    assert row["raw_null"][1] == n_null - 1 and row["nt_null"] == 1
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(root, "scratchpad", "xmethod", "eval_analysis.py")):
        pytest.skip("eval_analysis.py not in this bundle")
    sp = importlib.util.spec_from_file_location("eval_analysis_x",
                                                os.path.join(root, "scratchpad", "xmethod", "eval_analysis.py"))
    E = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(E)
    ref = E.seed_row(r, E.to_result(r), truth)["cnt"]
    assert [row["null_fp"], row["n_null"]] == ref["null_decl"] and row["raw_null"] == ref["null_raw"]
    assert [row["plac_decl"], row["n_plac"]] == ref["plac_decl"] and row["raw_plac"] == ref["plac_raw"]


# --------------------------------------------------------------------------------------------- R-35 integrity
def _frozen_text(tag="v1"):
    return f"# Protocol {tag}\n\nFROZEN: yes (2026-10-04)\nbody\n".encode()


def _sha(b):
    import hashlib
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def _setup_amended(tmp_path, monkeypatch, copy: bytes | None):
    """Live protocol = frozen text + an appended amendment (sha changed); optional bundled frozen copy."""
    frozen = _frozen_text()
    live = tmp_path / "PROTOCOL_A.md"
    live.write_bytes(frozen + b"\n## Amendment 1 (2026-10-10)\nbug-fix re-run\n")
    cp = tmp_path / "frozen_copy.md"
    if copy is not None:
        cp.write_bytes(copy)
    elif cp.exists():
        cp.unlink()
    monkeypatch.setattr(C, "PROTOCOL_PATH", str(live))
    monkeypatch.setattr(C, "FROZEN_COPY_PATH", str(cp))
    return _sha(frozen)


def test_guard_amended_protocol_verified_by_frozen_copy(tmp_path, monkeypatch):
    sha = _setup_amended(tmp_path, monkeypatch, _frozen_text())
    ok, why = C.eval_authorised(_eval_spec(sha))
    assert ok and "bundled frozen copy" in why
    sha = _setup_amended(tmp_path, monkeypatch, None)                       # no copy, no git: constant unverifiable
    ok, why = C.eval_authorised(_eval_spec(sha))
    assert not ok and "matches no frozen protocol text" in why
    sha = _setup_amended(tmp_path, monkeypatch, _frozen_text().replace(b"FROZEN: yes", b"FROZEN: no"))
    assert not C.eval_authorised(_eval_spec(_sha(_frozen_text().replace(b"FROZEN: yes", b"FROZEN: no"))))[0]
    assert not C.eval_authorised(_eval_spec("abc"))[0]                         # malformed constant


def test_guard_verified_by_git_blob_at_freeze_commit(tmp_path, monkeypatch):
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    (repo / "docs" / "xmethod").mkdir(parents=True)
    f = repo / "docs" / "xmethod" / "PROTOCOL_A.md"
    f.write_bytes(_frozen_text())
    g = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false"]
    subprocess.run(g[:3] + ["init", "-q"], check=True)
    subprocess.run(g + ["add", "."], check=True)
    subprocess.run(g + ["commit", "-qm", "freeze"], check=True)
    head = subprocess.run(g[:3] + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    f.write_bytes(_frozen_text() + b"\n## Amendment\n")
    monkeypatch.setattr(C, "GIT_ROOT", str(repo))
    monkeypatch.setattr(C, "PROTOCOL_PATH", str(f))
    monkeypatch.setattr(C, "FROZEN_COPY_PATH", str(tmp_path / "none.md"))
    sha = _sha(_frozen_text())
    ok, why = C.eval_authorised({**_eval_spec(sha), "freeze_commit": head})
    assert ok and "git blob" in why
    assert not C.eval_authorised({**_eval_spec(sha), "freeze_commit": "0" * 40})[0]
    f.write_bytes(b"# draft\nFROZEN: no\n")                               # live file unfrozen again: refused
    assert not C.eval_authorised({**_eval_spec(sha), "freeze_commit": head})[0]


def test_eval_seed_block_rules(tmp_path, monkeypatch):
    sha = _protocol(tmp_path, monkeypatch, frozen=True)
    s = _eval_spec(sha)
    s["blocks"][1]["seeds"] = [3000150, 3100000, 3100001]                          # measure must be EVAL only
    with pytest.raises(ValueError, match="measure blocks must use EVAL seeds"):
        C.expand(s)
    s = _eval_spec(sha)
    s["blocks"][0]["seeds"] = [3100005]                                     # tune must be DEV only
    with pytest.raises(ValueError, match="tune blocks must use DEV seeds"):
        C.expand(s)
    s = _eval_spec(sha)
    s["blocks"][1]["seeds"] = "TBD"
    with pytest.raises(ValueError, match="TBD"):
        C.expand(s)


def test_dirty_paths_scope():
    st = "\n".join([" M cdd_oran/xmethod/runner.py", "?? cdd_oran/xmethod/new.py", "!! cdd_oran/x/__pycache__/a.pyc",
                    "!! scratchpad/xmethod/specs/eval/full.json", "!! scratchpad/xmethod/eval_analysis2.py",
                    "!! scratchpad/xmethod/results/dev/agg.json", "!! scratchpad/xmethod/_bundle/requirements.lock.txt",
                    "?? notes.txt", " M README.md", "?? cdd_oran/newpkg/", "!! cdd_oran/x/__pycache__/"])
    assert C.dirty_paths(st) == ["cdd_oran/xmethod/runner.py", "cdd_oran/xmethod/new.py",
                                 "scratchpad/xmethod/specs/eval/full.json", "scratchpad/xmethod/eval_analysis2.py",
                                 "README.md", "cdd_oran/newpkg/"]
    assert C.dirty_paths("") == []


def test_records_stamp_commit_protocol_spec_sha(tmp_path, monkeypatch):
    monkeypatch.setenv("XM_CODE_COMMIT", "abc123")
    monkeypatch.setenv("XM_CODE_DIRTY", "0")
    spec = _small_spec()
    out = tmp_path / "r.jsonl"
    C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None, spec_file_sha256="f" * 64)
    r = C.load_jsonl(str(out))[0]
    it = r["integrity"]
    assert it["code_commit"] == "abc123" and it["code_dirty"] is False and it["spec_sha256"] == C.spec_sha256(spec)
    assert it["spec_file_sha256"] == "f" * 64 and it["python"] and "protocol_sha256" in it
    assert r["run_mode"]["mode"] == "dev" and r["code"]["source"].startswith("env")


def test_pin_check_against_lock(tmp_path):
    import importlib.metadata as md
    lock = tmp_path / "req.txt"
    lock.write_text(f"numpy=={md.version('numpy')} \\\n    --hash=sha256:00\n"
                    "scipy==0.0.1 ; python_version >= '3'\n"
                    "torch==9.9.9 ; sys_platform == 'nonexistent'\n# comment\n")
    pc = C.pin_check(str(lock))
    assert pc["n_pins"] == 2 and set(pc["mismatches"]) == {"scipy"}
    assert pc["mismatches"]["scipy"][0] == "0.0.1"
    assert C.pin_check(str(tmp_path / "missing.txt"))["mismatches"] is None


def test_eval_run_refused_before_any_data(tmp_path, monkeypatch):
    sha = _protocol(tmp_path, monkeypatch, frozen=True)
    spec = _eval_spec(sha)
    good = {"commit": "abc", "dirty": False}
    monkeypatch.setattr(C.platform, "python_version", lambda: C.EVAL_PYTHON)
    with pytest.raises(ValueError, match="not clean"):
        C.check_eval_preconditions(spec, {"commit": "abc", "dirty": None}, {"mismatches": {}})
    with pytest.raises(ValueError, match="no uv.lock export"):
        C.check_eval_preconditions(spec, good, {"mismatches": None})
    with pytest.raises(ValueError, match="differ from uv.lock"):
        C.check_eval_preconditions(spec, good, {"mismatches": {"numpy": ["2.4.2", "2.0.2"]}})
    for other in ("3.13.15", "3.12.13", "3.12.15"):                     # Q7 / R-57: exactly 3.12.14
        monkeypatch.setattr(C.platform, "python_version", lambda v=other: v)
        with pytest.raises(ValueError, match="requires exactly 3.12.14"):
            C.check_eval_preconditions(spec, good, {"mismatches": {}})
    monkeypatch.setattr(C.platform, "python_version", lambda: "3.12.14")
    C.check_eval_preconditions(spec, good, {"mismatches": {}})
    monkeypatch.setattr(C, "pin_check", lambda *a, **k: {"mismatches": {"numpy": ["2.4.2", "2.0.2"]}})
    monkeypatch.setattr(C, "generate_dataset", lambda *a, **k: pytest.fail("EVAL data generated"))
    out = tmp_path / "eval.jsonl"
    with pytest.raises(ValueError, match="EVAL refused"):
        C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None)
    assert not out.exists()


def test_classify_kill():
    assert C.classify_kill(24, 3.0, 100.0) == "cpu_limit"                   # SIGXCPU
    assert C.classify_kill(9, 110.0, 100.0) == "cpu_limit"                  # SIGKILL at the hard CPU limit
    assert C.classify_kill(9, 12.0, 100.0) == "killed"                      # OOM killer: error, not infeasible
    assert C.classify_kill(11, 12.0, 100.0) == "killed"


def test_eval_rules_cap_is_error_no_propagation_and_t3(tmp_path, monkeypatch):
    real = C.R.run_one
    monkeypatch.setattr(C.R, "run_one", lambda *a, **k: {**real(*a, **k), "cpu_s": 5.0})
    spec = _small_spec()
    units = C.expand(spec)
    out = tmp_path / "e.jsonl"
    C.run_units(spec, units, str(out), budget=2.0, isolate=False, log=lambda *_: None, eval_rules=True)
    recs = C.load_jsonl(str(out))
    assert len(recs) == len(units) and all(r["status"] == "error" for r in recs)   # cap 4 s < 5 s, every unit run
    assert all("safety cap 4" in r["error"] for r in recs)
    out2 = tmp_path / "e2.jsonl"
    C.run_units(spec, units, str(out2), budget=3.0, isolate=False, log=lambda *_: None, eval_rules=True)
    assert all(r["status"] == "ok" for r in C.load_jsonl(str(out2)))     # 5 s < cap 6 s: fine in EVAL
    spec_t3 = _small_spec()
    spec_t3["arms"]["d_by"]["infeasible_n"] = {"400": 9100.0}
    out3 = tmp_path / "t3.jsonl"
    C.run_part(spec_t3, 0, 1, str(out3), isolate=False, log=lambda *_: None)
    t3 = [r for r in C.load_jsonl(str(out3)) if r["arm"] == "d_by" and r["job"]["n"] == 400]
    assert t3 and all(r["status"] == "infeasible" and "DEV measured cost 9100" in r["reason"] for r in t3)


def test_dev_and_eval_outputs_never_mix(tmp_path):
    spec = _small_spec()
    out = tmp_path / "r.jsonl"
    out.write_text(json.dumps({"key": "x", "run_mode": {"mode": "eval", "protocol_sha256": "a" * 64}}) + "\n")
    with pytest.raises(ValueError, match="another run mode"):
        C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None)
    good = tmp_path / "g.jsonl"
    C.run_part(spec, 0, 1, str(good), isolate=False, log=lambda *_: None)
    recs = C.load_jsonl(str(good))
    ev = dict(recs[0], run_mode={"mode": "eval", "protocol_sha256": "a" * 64})
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(json.dumps(ev) + "\n")
    s = C.merge(spec, [str(good), str(mixed)], str(tmp_path / "m.jsonl"))
    assert s["foreign"] == 1 and s["n_records"] == len(recs) and s["mode"] == "dev"


def test_merge_one_platform_per_dataset(tmp_path):
    spec = _small_spec()
    out = tmp_path / "r.jsonl"
    C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None)
    recs = C.load_jsonl(str(out))
    ds0 = C._dataset_of(recs[0])
    same = [r for r in recs if C._dataset_of(r) == ds0]
    assert len(same) == 2                                                   # two arms on the first dataset
    a = [dict(r, host={"platform": "kaggle"}) for r in recs]
    b = [dict(same[0], host={"platform": "colab"})]                         # one arm re-run elsewhere, later
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    pa.write_text("\n".join(json.dumps(r) for r in a) + "\n")
    pb.write_text("\n".join(json.dumps(r) for r in b) + "\n")
    best, st = C.merge_records([str(pa), str(pb)])
    assert {C._platform(best[r["key"]]) for r in same} == {"kaggle"} and st["platform_switched_datasets"] == 1
    a2 = [r for r in a if r["key"] != same[0]["key"]]                       # kaggle lacks that arm: unresolvable
    pa.write_text("\n".join(json.dumps(r) for r in a2) + "\n")
    best, st = C.merge_records([str(pa), str(pb)])
    assert len(st["mixed_platform_datasets"]) == 1


def test_relaunch_reruns_whole_incomplete_datasets(tmp_path):
    spec = _small_spec()
    units = C.expand(spec)
    out = tmp_path / "r.jsonl"
    C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None)
    recs = C.load_jsonl(str(out))
    drop = recs[0]["key"]
    part = tmp_path / "partial.jsonl"
    part.write_text("\n".join(json.dumps(r) for r in recs if r["key"] != drop) + "\n")
    skip = C.complete_dataset_keys(units, [str(part)])
    ds = C._dataset_of(recs[0])
    assert not {u.key for u in units if u.dataset == ds} & skip                # whole dataset re-run
    assert skip == {u.key for u in units if u.dataset != ds}
    new = tmp_path / "new.jsonl"
    n = C.run_part(spec, 0, 1, str(new), isolate=False, log=lambda *_: None, skip_complete_from=[str(part)])
    assert n == sum(1 for u in units if u.dataset == ds)


def test_pin_check_torch_build_exception(tmp_path, monkeypatch):
    """R-35 amendment: torch must match the lock's public version (2.10.0); its build (+cpu / +cu128) is free;
    triton / nvidia-*-cu12 belong to the build and are not required; everything else stays exact."""
    import types
    lock = tmp_path / "req.txt"
    cont = " " + chr(92) + "\n"                                            # uv export line continuation
    lock.write_text(f"torch==2.10.0+cu128{cont}    --hash=sha256:00\ntriton==3.6.0\nnvidia-cudnn-cu12==9.10.2.21\n"
                    "nvidia-nccl-cu13==2.32.3\nnumpy==2.4.2\n")

    def fake(installed):
        dists = [types.SimpleNamespace(metadata={"Name": n}, version=v) for n, v in installed.items()]
        monkeypatch.setattr(C, "md", types.SimpleNamespace(distributions=lambda: dists))

    fake({"torch": "2.10.0+cpu", "nvidia-nccl-cu13": "2.32.3", "numpy": "2.4.2"})
    pc = C.pin_check(str(lock))
    assert pc["mismatches"] == {} and pc["torch_build"] == "2.10.0+cpu"
    assert pc["torch_build_exempt"] == {"triton": None, "nvidia-cudnn-cu12": None}
    fake({"torch": "2.9.1+cpu", "nvidia-nccl-cu13": "2.32.3", "numpy": "2.4.2"})
    assert set(C.pin_check(str(lock))["mismatches"]) == {"torch"}
    fake({"torch": "2.10.0+cu128", "numpy": "2.0.2"})                       # xgboost's nccl-cu13 and numpy exact
    assert set(C.pin_check(str(lock))["mismatches"]) == {"nvidia-nccl-cu13", "numpy"}
    fake({"torch": "2.10.0+cpu", "nvidia-nccl-cu13": "2.32.3", "numpy": "2.4.2"})   # R-57: exact EVAL interpreter
    for py, bad in (("3.12.14", {}), ("3.12.15", {"python": ["3.12.14", "3.12.15"]}),
                    ("3.12.13", {"python": ["3.12.14", "3.12.13"]})):
        monkeypatch.setattr(C.platform, "python_version", lambda v=py: v)
        pc = C.pin_check(str(lock), python=C.EVAL_PYTHON)
        assert pc["mismatches"] == bad and pc["python"] == py and pc["python_required"] == "3.12.14"
        assert C.pin_check(str(lock))["mismatches"] == {}                  # DEV: no interpreter pin
    assert C.EVAL_PYTHON == "3.12.14"


def test_install_lock_drops_only_the_torch_build():
    c = " " + chr(92) + "\n"                                               # uv export line continuation
    text = (f"# header\nnumpy==2.4.2{c}    --hash=sha256:aa\ntorch==2.10.0+cu128 ; sys_platform == 'linux'{c}"
            f"    --hash=sha256:bb\n    --hash=sha256:cc\ntriton==3.6.0 ; sys_platform == 'linux'{c}"
            f"    --hash=sha256:dd\nnvidia-cublas-cu12==12.8.4.1{c}    --hash=sha256:ee\n"
            f"nvidia-nccl-cu13==2.32.3{c}    --hash=sha256:ff\n")
    blocks = C._req_blocks(text)
    assert "".join(blocks) == text and len(blocks) == 6
    from packaging.utils import canonicalize_name
    kept = "".join(b for b in blocks if not C.is_torch_build(canonicalize_name(b.split(";")[0].split("==")[0].strip())))
    assert "numpy==2.4.2" in kept and "nvidia-nccl-cu13" in kept and "--hash=sha256:ff" in kept
    assert "torch" not in kept and "triton" not in kept and "cublas" not in kept and "sha256:bb" not in kept


def test_eval_python_312_venv_setup():
    """Q7: EVAL kernels install Python 3.12 with uv and run everything in a fresh venv; DEV keeps the image's."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pv = os.path.join(root, ".python-version")
    if os.path.exists(pv):                                   # the local .python-version (3.12) admits it
        assert (C.EVAL_PYTHON + ".").startswith(open(pv).read().strip() + ".")
    ev = C._setup_cmd("$OUT", False, "2.10.0", C.EVAL_PYTHON)
    assert "uv python install 3.12.14 " in ev and "uv venv --python 3.12.14 " in ev     # R-57: exact patch
    assert "python='3.12.14'" in ev and "python='" not in C._setup_cmd("$OUT", False, "2.10.0", "3.12")
    order = [ev.index(x) for x in ("uv python install 3.12.14", "uv venv --python 3.12.14 /tmp/xm_venv",
                                   "source /tmp/xm_venv/bin/activate", "unset UV_SYSTEM_PYTHON",
                                   "uv pip install --python /tmp/xm_venv/bin/python --require-hashes --no-deps",
                                   "torch==2.10.0", "pin_check(")]
    assert order == sorted(order) and "--system" not in ev
    assert "pip install pytest" not in ev.replace("uv pip install --python /tmp/xm_venv/bin/python pytest", "")
    dev = C._setup_cmd("$OUT", False, "2.10.0")
    assert "xm_venv" not in dev and "uv pip install --system --require-hashes --no-deps" in dev


# --------------------------------------------------------------------------------------------- CI pilot: budgets
def _budget_spec():
    s = _small_spec()
    s["arms"]["d_by"]["budget_cpu_s"] = 1.0                            # CPU arm over budget (cost patched to 5)
    s["arms"]["d_tau"].update(budget_cpu_s=None, budget_wall_s=10.0)  # GPU-style arm: wall budget only (R-41)
    return s


def test_arm_budgets():
    s = _budget_spec()
    assert C.arm_budgets(s, "d_by", 7200.0) == (1.0, None)
    assert C.arm_budgets(s, "d_tau", 7200.0) == (None, 10.0)
    assert C.arm_budgets(s, "d_e3", 7200.0) == (7200.0, None)


def test_registry_shared_across_processes_and_wall_budget(tmp_path, monkeypatch):
    real = C.R.run_one
    monkeypatch.setattr(C.R, "run_one", lambda *a, **k: {**real(*a, **k), "cpu_s": 5.0, "wall_s": 50.0})
    spec = _budget_spec()
    units = [u for u in C.expand(spec) if u.role == "measure"]
    reg = tmp_path / "registry.jsonl"
    first = [u for u in units if u.seed == 3000100 and u.n == 300]       # "process 1"
    rest = [u for u in units if u not in first]                          # "process 2"
    C.run_units(spec, first, str(tmp_path / "p1.jsonl"), budget=7200, isolate=False, log=lambda *_: None,
                registry=str(reg))
    p1 = C.load_jsonl(str(tmp_path / "p1.jsonl"))
    assert all(r["status"] == "infeasible" for r in p1)
    assert {r["cost_kind"] for r in p1} == {"CPU-s", "wall-s"}
    assert {r["arm"]: r["cost"] for r in p1} == {"d_by": 5.0, "d_tau": 50.0}
    assert C.read_registry(str(reg)) == {"d_by": (300, 5.0, "CPU-s"), "d_tau": (300, 50.0, "wall-s")}
    C.run_units(spec, rest, str(tmp_path / "p2.jsonl"), budget=7200, isolate=False, log=lambda *_: None,
                registry=str(reg))
    p2 = C.load_jsonl(str(tmp_path / "p2.jsonl"))
    assert len(p2) == len(rest) and all(r["status"] == "infeasible" for r in p2)
    assert all(r["reason"].startswith("not run: arm infeasible at n 300 in this session") for r in p2)
    agg = C.aggregate(p1 + p2, spec)
    mi = C.measured_infeasible(agg)
    assert mi == {"d_by": (300, 5.0, "CPU-s"), "d_tau": (300, 50.0, "wall-s")}   # unrun ones are not "measured"
    pr = C.project(agg, spec)
    assert sum(pr["units_infeasible_not_run"].values()) == len(C.expand(spec))
    assert pr["cpu_h_total"] == pr["cpu_h_generation"] and pr["gpu_wall_h_total"] == 0


def test_project_splits_gpu_wall_from_cpu(tmp_path, monkeypatch):
    real = C.R.run_one
    monkeypatch.setattr(C.R, "run_one", lambda *a, **k: {**real(*a, **k), "cpu_s": 2.0, "wall_s": 3.0})
    spec = _small_spec()
    spec["arms"]["d_tau"].update(budget_cpu_s=None, budget_wall_s=7200.0)
    out = tmp_path / "r.jsonl"
    C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None)
    agg = C.aggregate(C.load_jsonl(str(out)), spec)
    pr = C.project(agg, spec)
    n_tau = sum(u.arm == "d_tau" for u in C.expand(spec))
    n_by = sum(u.arm == "d_by" for u in C.expand(spec))
    assert math.isclose(pr["gpu_wall_h_by_arm"]["d_tau"], 3.0 * n_tau / 3600)
    assert math.isclose(pr["cpu_h_by_arm"]["d_by"], 2.0 * n_by / 3600) and "d_tau" not in pr["cpu_h_by_arm"]


def test_gpu_cloud_and_setup_commands():
    code = {"commit": "abc", "dirty": False}
    cmd = C._cloud_cmd("spec.json", [0, 1], 2, "$OUT", "kaggle", None, 3600, code, registry=True, gpu=True)
    assert "NG=$(nvidia-smi -L" in cmd and "CUDA_VISIBLE_DEVICES=$(( 1 % NG ))" in cmd
    assert cmd.count("--registry $OUT/registry.jsonl") == 2 and "PYTORCH_NVML_BASED_CUDA_CHECK=1" in cmd
    cpu = C._cloud_cmd("spec.json", [0], 1, "$OUT", "kaggle", None, 3600, code)
    assert "CUDA_VISIBLE_DEVICES" not in cpu and "--registry" not in cpu
    rel = C._cloud_cmd("spec.json", [0, 1], 2, "$OUT", "colab", None, 3600, code, skip_from=["d/l1_*.jsonl"])
    assert rel.count("--skip-complete-from 'd/l1_*.jsonl'") == 2      # relaunch elsewhere: quoted glob, every part
    assert "whl/cu128" in C._setup_cmd("$OUT", False, "2.10.0", gpu=True)
    assert "whl/cpu" in C._setup_cmd("$OUT", False, "2.10.0")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="fork isolation is Linux-only")
def test_isolated_wall_limit_kills_child(tmp_path, monkeypatch):
    import time as _t
    monkeypatch.setattr(C.R, "run_one", lambda *a, **k: _t.sleep(30) or {})
    spec = _small_spec()
    spec["arms"]["d_tau"].update(budget_cpu_s=None, budget_wall_s=1.0)
    units = [u for u in C.expand(spec) if u.arm == "d_tau"][:1]
    t0 = _t.perf_counter()
    C.run_units(spec, units, str(tmp_path / "w.jsonl"), budget=7200, isolate=True, log=lambda *_: None)
    r = C.load_jsonl(str(tmp_path / "w.jsonl"))[0]
    assert _t.perf_counter() - t0 < 15 and r["status"] == "infeasible" and r["cost_kind"] == "wall-s"


def test_method_load_failure_is_an_error_record_not_a_crash(tmp_path):
    spec = _small_spec()
    spec["arms"]["d_by"]["ref"] = "cdd_oran.xmethod.no_such_module:Nope"
    out = tmp_path / "r.jsonl"
    C.run_part(spec, 0, 1, str(out), isolate=False, log=lambda *_: None)
    recs = C.load_jsonl(str(out))
    bad = [r for r in recs if r["arm"] == "d_by"]
    assert len(recs) == len(C.expand(spec)) and bad and all(r["status"] == "error" for r in bad)
    assert all("method load failed" in r["error"] for r in bad)
    assert all(r["status"] == "ok" for r in recs if r["arm"] == "d_tau")


def test_streaming_merge_and_aggregate_equal_in_memory(tmp_path):
    spec = _small_spec()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    C.run_part(spec, 0, 2, str(a), isolate=False, log=lambda *_: None)
    C.run_part(spec, 1, 2, str(b), isolate=False, log=lambda *_: None)
    s = C.merge(spec, [str(a), str(b)], str(tmp_path / "m.jsonl.gz"))
    assert s["n_records"] == s["n_expected"] == len(C.expand(spec)) and not s["missing"]
    assert (tmp_path / "m.jsonl.summary.json").exists()
    recs = list(C.iter_jsonl(str(tmp_path / "m.jsonl.gz")))
    assert [r["key"] for r in recs] == sorted(r["key"] for r in recs)
    streamed = C.aggregate(C.iter_jsonl(str(tmp_path / "m.jsonl.gz")), spec, stream_sorted=True)
    assert json.dumps(streamed, sort_keys=True, default=str) == json.dumps(C.aggregate(recs, spec), sort_keys=True,
                                                                           default=str)
    with pytest.raises(ValueError, match="not sorted"):
        C.aggregate(recs[1:] + recs[:1], spec, stream_sorted=True)          # first cell split in two


def test_cloud_bundle_imports_every_method(tmp_path):
    """A cloud bundle = cdd_oran/**/*.py + BUNDLE_DATA; every method module must import from it alone
    (CI pilot: pcorr -> cdd_oran.config reads configs/*.yaml; two_tower imports scripts/)."""
    import glob
    import os
    import shutil
    import subprocess
    for f in glob.glob(os.path.join(C.R.ROOT, "cdd_oran", "**", "*.py"), recursive=True):
        rel = os.path.relpath(f, C.R.ROOT)
        os.makedirs(tmp_path / os.path.dirname(rel), exist_ok=True)
        shutil.copy(f, tmp_path / rel)
    for d in C.BUNDLE_DATA:
        shutil.copytree(os.path.join(C.R.ROOT, d), tmp_path / d)
    mods = sorted(os.path.basename(m)[:-3] for m in glob.glob(os.path.join(C.R.ROOT, "cdd_oran", "xmethod", "methods",
                                                                           "[a-z]*.py")))
    def bad(cwd, env):                               # modules that do not import, one fresh process each
        return {m for m in mods if subprocess.run([sys.executable, "-c", f"import cdd_oran.xmethod.methods.{m}"],
                                                  cwd=cwd, env=env, capture_output=True).returncode != 0}
    in_bundle = bad(tmp_path, {**os.environ, "PYTHONPATH": str(tmp_path)})
    assert in_bundle <= bad(C.R.ROOT, dict(os.environ)), in_bundle   # only local-platform failures (shap_dag DLL)


def test_lightning_and_colab_gpu_launch_commands(capsys, monkeypatch):
    """R-46: `campaign lightning` = the same setup + parts commands, run by xm_lightning.py on a studio (Py 3.12
    venv by default); `campaign colab --gpu` = GPU per process + torch CUDA fallback."""
    import os
    monkeypatch.setattr(C, "write_lock", lambda: C.LOCK_PATH)
    monkeypatch.setattr(C, "code_info", lambda: {"commit": "abc", "dirty": False})
    sp = os.path.join(C.R.ROOT, "scratchpad", "xmethod", "specs", "dev", "smoke.json")   # a repo spec (relpath)
    if not os.path.exists(sp):
        pytest.skip("cloud bundle without the smoke spec")
    C.main(["lightning", "--spec", str(sp), "--name", "xm-t", "--parts", "4", "--part-set", "2,3", "--dry-run"])
    out = capsys.readouterr().out
    assert "xm_lightning.py" in out and "launch xm-t --machine CPU --out-dir xm_out" in out
    assert "XM_PLATFORM=lightning" in out and "uv python install 3.12" in out
    assert "--part 2/4" in out and "--part 3/4" in out and "--part 0/4" not in out
    assert "--registry xm_out/registry.jsonl" in out and "configs" in out and "scripts" in out
    C.main(["colab", "--spec", str(sp), "--name", "xm-g", "--parts", "1", "--gpu", "--dry-run"])
    out = capsys.readouterr().out
    assert "CUDA_VISIBLE_DEVICES" in out and "whl/cu128" in out and "XM_PLATFORM=colab" in out


def test_r55_cpu_quota_procs_and_load(tmp_path):
    """R-55: vCPU quota from the cgroup (v2 / v1 / none), top-level campaign processes (forked unit children not
    counted), host load fields."""
    import os
    cap = os.cpu_count() or 1
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "cpu.max").write_text("400000 100000\n")
    assert C.cpu_quota(str(v2)) == min(4, cap)
    (v2 / "cpu.max").write_text("150000 100000\n")
    assert C.cpu_quota(str(v2)) == min(2, cap)                     # 1.5 CPUs -> 2 processes
    (v2 / "cpu.max").write_text("max 100000\n")
    assert C.cpu_quota(str(v2)) >= 1
    v1 = tmp_path / "v1" / "cpu"
    v1.mkdir(parents=True)
    (v1 / "cpu.cfs_quota_us").write_text("200000")
    (v1 / "cpu.cfs_period_us").write_text("100000")
    assert C.cpu_quota(str(tmp_path / "v1")) == min(2, cap)
    host = tmp_path / "host"                                         # VPS: quota on the systemd scope, not the root
    sc = host / "system.slice" / "cdd-xm-j.scope"
    sc.mkdir(parents=True)
    (host / "cpu.max").write_text("max 100000\n")
    (sc / "cpu.max").write_text("700000 100000\n")
    me = tmp_path / "self_cgroup"
    me.write_text("0::/system.slice/cdd-xm-j.scope\n")
    assert C.cpu_quota(str(host), str(me)) == min(7, cap)
    (host / "system.slice" / "cpu.max").write_text("300000 100000\n")   # a tighter parent wins
    assert C.cpu_quota(str(host), str(me)) == min(3, cap)
    me.write_text("0::/\n")                                          # container: own cgroup = root
    assert C.cpu_quota(str(host), str(me)) == cap
    proc = tmp_path / "proc"
    for pid, ppid, cmd in ((9, 1, "timeout 99 python -u -m cdd_oran.xmethod.campaign run --part 0/2 --out o/r_0.jsonl"),
                           (10, 9, "python -u -m cdd_oran.xmethod.campaign run --part 0/2 --out o/r_0.jsonl"),
                           (11, 10, "python -u -m cdd_oran.xmethod.campaign run --part 0/2 --out o/r_0.jsonl"),  # fork
                           (12, 1, "python -u -m cdd_oran.xmethod.campaign run --part 1/2 --out o/r_1.jsonl"),
                           (13, 1, "bash -c sleep 9"),
                           (14, 1, "python -m cdd_oran.xmethod.campaign merge --spec s --out m.jsonl")):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
        (d / "stat").write_text(f"{pid} (py thon) S {ppid} 1 1 0")
    (proc / "loadavg").write_text("3.50 2.25 1.00 4/180 999\n")
    assert C.campaign_procs(str(proc)) == 2
    li = C.load_info(str(proc))
    assert li["loadavg"] == [3.5, 2.25, 1.0] and (li["procs_running"], li["procs_total"]) == (4, 180)
    assert li["campaign_procs"] == 2 and li["proc_visible"] == 6 and li["cpu_quota"] >= 1
    assert C.campaign_procs(str(tmp_path / "none")) is None
    C.check_one_per_cpu(4, 4)
    C.check_one_per_cpu(None, 1)
    with pytest.raises(ValueError, match="R-55"):
        C.check_one_per_cpu(5, 4)


def test_r55_records_log_load_and_eval_refuses_oversubscription(tmp_path, monkeypatch):
    spec = _small_spec()
    units = C.expand(spec)[:2]
    out = tmp_path / "d.jsonl"
    C.run_units(spec, units, str(out), isolate=False, log=lambda *_: None)
    for r in C.load_jsonl(str(out)):
        assert set(r["load"]) == {"start", "end"} and "campaign_procs" in r["load"]["start"]
        assert "cpu_model" in r["host"] and "cpu_quota" in r["load"]["end"]
    monkeypatch.setattr(C, "is_eval", lambda s: True)
    monkeypatch.setattr(C, "check_eval_preconditions", lambda *a, **k: None)
    monkeypatch.setattr(C, "campaign_procs", lambda *a, **k: 9)
    monkeypatch.setattr(C, "cpu_quota", lambda *a, **k: 4)
    with pytest.raises(ValueError, match="9 campaign processes on 4 vCPU"):
        C.run_units(spec, units, str(tmp_path / "e.jsonl"), isolate=False, log=lambda *_: None)
    assert not (tmp_path / "e.jsonl").exists()                         # refused before any data


def test_r55_eval_cloud_cmd_caps_processes_at_quota():
    code = {"commit": "abc", "dirty": False}
    ev = C._cloud_cmd("spec.json", [0, 1, 2, 3, 4, 5], 6, "$OUT", "kaggle", None, 3600, code, "eval_res",
                      one_per_cpu=True)
    assert 'NP=$(python -c "from cdd_oran.xmethod.campaign import cpu_quota' in ev
    assert ev.count("while [ $(jobs -rp | wc -l) -ge $NP ]; do wait -n; done;") == 6
    assert ev.index("NP=$(") < ev.index("--part 0/6")
    dev = C._cloud_cmd("spec.json", [0, 1], 2, "$OUT", "kaggle", None, 3600, code)
    assert "wait -n" not in dev and "NP=" not in dev


def test_vps_launch_steps(capsys, monkeypatch):
    """VPS lane (vps_run.py CLI v1): push tracked code + generated lock exports, a venv on Python 3.12.14 with the
    exact-interpreter pin check, launch inside the capped scope with <= 7 processes ({OUT} placeholder)."""
    import os
    sp = os.path.join(C.R.ROOT, "scratchpad", "xmethod", "specs", "dev", "smoke.json")
    if not os.path.exists(sp):
        pytest.skip("cloud bundle without the smoke spec")
    monkeypatch.setattr(C, "write_lock", lambda: C.LOCK_PATH)
    monkeypatch.setattr(C, "code_info", lambda: {"commit": "abc", "dirty": False})
    C.main(["vps", "--spec", sp, "--name", "xm-t-v1", "--parts", "14", "--part-set", "0,1,2,3,4,5,6", "--dry-run"])
    out = capsys.readouterr().out.splitlines()
    push, venv, launch = (next(ln for ln in out if f" {k} xm-t-v1" in ln) for k in ("push", "venv", "launch"))
    assert "cdd_oran" in push and "configs" in push and "scratchpad/xmethod/specs/dev/smoke.json" in push
    assert f"={C.LOCK_REL}" in push and f"={C.LOCK_INSTALL_REL}" in push
    assert "pin_check(" in venv and "3.12.14" in venv and C.LOCK_INSTALL_REL in venv
    assert "--procs 7" in launch and "XM_PLATFORM=vps" in launch and "{OUT}/res_6.jsonl" in launch
    with pytest.raises(SystemExit, match="one process per part"):
        C.main(["vps", "--spec", sp, "--name", "xm-t-v2", "--parts", "14", "--part-set", "0,1,2,3,4,5,6,7",
                "--dry-run"])

"""R-60 extras wrapper (cdd_oran/xmethod/extras.py): X2 reproduction cell bit-for-bit, design overrides, X3 told
designs, the XMETHOD_EXTRAS seed guard, campaign hooks (restored), records, declared specs, frozen files intact."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from cdd_oran.xmethod import campaign as C
from cdd_oran.xmethod import extras as X
from cdd_oran.xmethod import runner as R
from cdd_oran.xmethod.worlds import e4_logged
from cdd_oran.xmethod.worlds import generate as G

S0 = 3_200_020


def _same(a, b) -> bool:
    return G.dataset_hash(a) == G.dataset_hash(b)


@pytest.mark.parametrize("world,regime", [("E1", "R2"), ("E2", "R2"), ("E3", "R2"), ("E5", "R2"), ("E1", "R1"),
                                          ("E4", "R3")])
def test_reproduction_cell_matches_frozen_generator_bit_for_bit(world, regime):
    for seed, n in ((S0, 200), (3_200_119, 1000)):
        ref, tr = G.generate_dataset(world, regime, n, seed, kappa=0.25)
        got, tg = X.generate(world, regime, n, seed, kappa=0.25, design={"delta": 0.10, "n_blocks": 20})
        assert _same(ref, got) and tr == tg
        for a, b in ((ref.X_action, got.X_action), (ref.X_kpi_lag, got.X_kpi_lag), (ref.Y, got.Y)):
            assert a.tobytes() == b.tobytes()
    assert (G.DITHER_DELTA, G.DITHER_BLOCKS) == (0.10, 20)


def test_design_override_changes_only_the_dither_and_is_restored():
    ref, _ = G.generate_dataset("E1", "R2", 400, S0, kappa=0.25)
    half, _ = X.generate("E1", "R2", 400, S0, kappa=0.25, design={"delta": 0.05, "n_blocks": 20})
    for d0, d1 in zip(ref.designs, half.designs, strict=True):        # same uniform draws, half the amplitude
        assert np.allclose(d1.random_part, 0.5 * d0.random_part, rtol=0, atol=1e-12)
        assert np.array_equal(d1.fixed_part, d0.fixed_part)
        assert d1.dist["hi"] == pytest.approx(0.5 * d0.dist["hi"])
    assert half.meta["dither"]["delta"] == 0.05 and not _same(ref, half)
    b5, _ = X.generate("E1", "R2", 400, S0, kappa=0.25, design={"delta": 0.10, "n_blocks": 5})
    assert b5.meta["dither"]["n_blocks"] == 5 and len(np.unique(b5.meta["block"])) == 5
    assert all(len(np.unique(d.fixed_part)) == 5 for d in b5.designs)
    assert (G.DITHER_DELTA, G.DITHER_BLOCKS) == (0.10, 20)
    with pytest.raises(ZeroDivisionError):
        with X.dither_constants(0.1, 0):
            raise ZeroDivisionError
    assert (G.DITHER_DELTA, G.DITHER_BLOCKS) == (0.10, 20)
    for bad in ({"delta": 0.25, "n_blocks": 20}, {"delta": 0.0, "n_blocks": 20}, {"delta": 0.1, "n_blocks": 0}):
        with pytest.raises(ValueError):
            X.check_design(bad)


def test_told_width_and_shift():
    ds, _ = G.generate_dataset("E2", "R2", 1000, S0, kappa=0.25)
    assert X.shift_rows(0.02, 1000, 20) == 1 and X.shift_rows(0.05, 1000, 20) == 3
    assert _same(X.tell(ds, {"width": 1.0}), ds)                      # identity told = the frozen dataset
    w = X.tell(ds, {"width": 2.0})
    for d0, d1 in zip(ds.designs, w.designs, strict=True):
        assert d1.dist["lo"] == 2.0 * d0.dist["lo"] and d1.dist["hi"] == 2.0 * d0.dist["hi"]
        assert np.array_equal(d1.random_part, d0.random_part) and np.array_equal(d1.fixed_part, d0.fixed_part)
    assert np.array_equal(w.X_action, ds.X_action) and np.array_equal(w.Y, ds.Y)
    sh = X.tell(ds, {"shift": 0.05})
    for j, (d0, d1) in enumerate(zip(ds.designs, sh.designs, strict=True)):
        assert np.array_equal(d1.fixed_part[3:], d0.fixed_part[:-3]) and np.all(d1.fixed_part[:3] == d0.fixed_part[0])
        same = d1.fixed_part == d0.fixed_part
        assert np.array_equal(d1.random_part[same], d0.random_part[same])
        assert np.allclose(d1.fixed_part + d1.random_part, ds.X_action[:, j], rtol=0, atol=1e-12)
        assert (~same).sum() == 3 * (len(np.unique(ds.meta["block"])) - 1)   # 3 rows after each switch
        assert d1.dist == d0.dist
    with pytest.raises(ValueError):
        X.tell(ds, {"lam": 0.5})                                       # no logged design in R2
    with pytest.raises(ValueError):
        X.check_told({"width": 0.5, "shift": 0.02})


def test_told_lambda_e4_r3():
    ds, _ = G.generate_dataset("E4", "R3", 500, S0, kappa=0.25)
    assert _same(X.tell(ds, {"lam": 1.0}), ds)
    t = X.tell(ds, {"lam": 2.0})
    z = ds.context[:, 0]
    for d0, d1 in zip(ds.designs, t.designs, strict=True):
        if d0.kind == "logged":
            assert d1.dist["lam"] == 2.0 and np.array_equal(d1.propensity, e4_logged.policy_probs(z, 2.0))
        else:
            assert d1 == d0                                           # the i.i.d. P_placebo design untouched
    with pytest.raises(ValueError):
        X.tell(ds, {"width": 2.0})


def _spec(experiment="X2", seeds=(S0, S0 + 1), arms=None, design=None):
    fz = X._frozen_arms()
    arms = arms or {"corr": X._arm(fz, "corr")}
    ex = {"experiment": experiment, "design": design, "protocol": X.PROTOCOL_REL}
    if experiment == "X2" and design is None:
        ex["design"] = {"delta": 0.05, "n_blocks": 20}
    return {"name": "t_extras", "budget_cpu_s": 7200, "extras": ex, "arms": arms,
            "blocks": [{"role": "measure", "worlds": ["E1"], "regimes": ["R2"], "ns": [200], "kappas": [0.25],
                        "seeds": list(seeds)}]}


def test_seed_guard_and_hooks_restored():
    spec = _spec()
    with pytest.raises(ValueError, match="DEV block|refused"):          # frozen guard, untouched outside the hooks
        C.expand(spec)
    with X.installed(spec):
        units = C.expand(spec)
        assert len(units) == 2 and not C.is_eval(spec)
        assert C.is_eval({"blocks": [{"seeds": [3_100_000, 3_100_001]}]})   # non-extras specs keep the frozen rule
    assert C.is_eval is X.C.is_eval and C.run_units.__module__ == "cdd_oran.xmethod.campaign"
    assert C.generate_dataset is G.generate_dataset and R.run_one.__module__ == "cdd_oran.xmethod.runner"
    for bad in ([3_100_000, 3_100_001], [3_000_000, 3_000_001], [3_200_199, 3_200_200], [3_199_999]):
        with pytest.raises(ValueError, match="XMETHOD_EXTRAS"):
            X.validate(_spec(seeds=bad))
    with pytest.raises(ValueError):
        X.validate({**_spec(), "extras": {"experiment": "X3", "design": {"delta": 0.1, "n_blocks": 20}}})


def test_cloud_command_runs_extras():
    spec = _spec()
    with X.installed(spec):
        cmd = C._cloud_cmd("scratchpad/xmethod/specs/extras/x2_d05.json", [0, 1], 2, "$JOB_OUT", "kaggle", None,
                           3600, {"commit": "abc", "dirty": False})
    assert "-m cdd_oran.xmethod.extras run" in cmd and "-m cdd_oran.xmethod.campaign run" not in cmd


def test_records_x2_design_and_x3_told(tmp_path):
    fz = X._frozen_arms()
    s2 = _spec()
    out2 = str(tmp_path / "x2.jsonl")
    with X.installed(s2):
        C.run_units(s2, C.expand(s2), out2, isolate=False, log=lambda *_: None)
    recs = [json.loads(x) for x in open(out2, encoding="utf-8")]
    assert len(recs) == 2 and all(r["status"] == "ok" and r["run_mode"]["mode"] == "dev" for r in recs)
    for r in recs:
        ext = r["integrity"]["extras"]
        assert ext["experiment"] == "X2" and ext["design"] == {"delta": 0.05, "n_blocks": 20}
        seed = r["job"]["seed"]
        want = X.generate("E1", "R2", 200, seed, kappa=0.25, design={"delta": 0.05, "n_blocks": 20})[0]
        assert r["dataset_sha256"] == G.dataset_hash(want)
        assert r["dataset_sha256"] != G.dataset_hash(G.generate_dataset("E1", "R2", 200, seed, kappa=0.25)[0])
    s3 = _spec("X3", seeds=[S0, S0 + 1], arms={"pmrt_eq": X._arm(fz, "pmrt_eq"),
                                               "pmrt_eq.w200": X._arm(fz, "pmrt_eq", {"width": 2.0})})
    s3["blocks"][0]["seeds"] = [S0]
    out3 = str(tmp_path / "x3.jsonl")
    with X.installed(s3):
        C.run_units(s3, C.expand(s3), out3, isolate=False, log=lambda *_: None)
    r3 = {r["arm"]: r for r in map(json.loads, open(out3, encoding="utf-8"))}
    frozen = G.dataset_hash(G.generate_dataset("E1", "R2", 200, S0, kappa=0.25)[0])
    assert r3["pmrt_eq"]["dataset_sha256"] == frozen and "told" not in r3["pmrt_eq"]
    t = r3["pmrt_eq.w200"]
    assert t["status"] == "ok" and t["told"] == {"width": 2.0} and t["dataset_sha256_generated"] == frozen
    assert t["dataset_sha256"] != frozen and t["config"] == r3["pmrt_eq"]["config"]
    assert [e["p"] for e in t["edges"]] != [e["p"] for e in r3["pmrt_eq"]["edges"]]
    # merge keeps only records of the same spec (designs share unit keys)
    with X.installed(s2):
        acc = C.accept_rule(s2)
    assert all(acc(r) for r in recs) and not any(acc(r) for r in r3.values())


def test_declared_specs_match_protocol_and_frozen_arms():
    specs = X.build_specs()
    fz = X._frozen_arms()
    assert set(specs) == {*X.X2_DESIGNS, "x3_told"}
    for name, spec in specs.items():
        X.validate(spec)
        on_disk = json.load(open(X._path(f"{X.SPEC_DIR_REL}/{name}.json"), encoding="utf-8"))
        assert on_disk == spec, f"{name}.json differs from build_specs(): run `extras specs`"
        for a, d in spec["arms"].items():
            base = d.get("base", a)
            for k in ("ref", "config", "declare"):
                assert d[k] == fz[base][k]
    seeds = {b["role"]: X.seed_list(b["seeds"]) for b in specs["x2_d10"]["blocks"]}
    assert len(seeds["tune"]) == 20 and len(seeds["measure"]) == 40 and not set(seeds["tune"]) & set(seeds["measure"])
    x3 = [X.seed_list(b["seeds"]) for b in specs["x3_told"]["blocks"]]
    assert all(len(s) == 60 and not set(s) & (set(seeds["tune"]) | set(seeds["measure"])) for s in x3)
    assert specs["x2_d10"]["extras"]["design"] == {"delta": G.DITHER_DELTA, "n_blocks": G.DITHER_BLOCKS}
    reg = json.load(open(X._path("docs/benchmark/SEED_REGISTRY.json"), encoding="utf-8"))
    for w in ("E1", "E2", "E3", "E4"):
        assert reg[w]["claimed_by"]["XMETHOD_EXTRAS"] == [[3_200_000, 3_200_199]]


def test_freeze_manifest_still_valid():
    man = X._path("scratchpad/xmethod/freeze/FREEZE_MANIFEST.sha256")
    if not os.path.exists(man):
        pytest.skip("no freeze manifest in this checkout")
    bad = []
    for line in open(man, encoding="utf-8"):
        want, rel = line.split(maxsplit=1)
        p = X._path(rel.strip())
        if not os.path.exists(p) or C._sha_lf(open(p, "rb").read()) != want:
            bad.append(rel.strip())
    assert not bad, f"frozen files changed (LF rule): {bad[:5]}"


@pytest.mark.skipif(not (hasattr(os, "fork") and os.sys.platform.startswith("linux")), reason="fork isolation: Linux")
def test_isolated_children_inherit_the_hooks(tmp_path):
    fz = X._frozen_arms()
    s2 = _spec(seeds=[S0])
    s3 = _spec("X3", seeds=[S0], arms={"pmrt_eq.s05": X._arm(fz, "pmrt_eq", {"shift": 0.05})})
    for spec in (s2, s3):
        out = str(tmp_path / f"{spec['extras']['experiment']}.jsonl")
        with X.installed(spec):
            C.run_units(spec, C.expand(spec), out, isolate=True, log=lambda *_: None)
        (r,) = [json.loads(x) for x in open(out, encoding="utf-8")]
        assert r["status"] == "ok" and r["integrity"]["isolation"] == "fork+rlimit"
        frozen = G.dataset_hash(G.generate_dataset("E1", "R2", 200, S0, kappa=0.25)[0])
        if spec is s2:
            assert r["dataset_sha256"] == G.dataset_hash(
                X.generate("E1", "R2", 200, S0, kappa=0.25, design=spec["extras"]["design"])[0]) != frozen
        else:
            assert r["told"] == {"shift": 0.05} and r["dataset_sha256_generated"] == frozen != r["dataset_sha256"]


def test_analyse_end_to_end_provenance_and_summaries(tmp_path):
    fz = X._frozen_arms()
    specs = {"d10": _spec(design={"delta": 0.10, "n_blocks": 20}), "d05": _spec(design={"delta": 0.05, "n_blocks": 20}),
             "x3": _spec("X3", arms={"pmrt_eq": X._arm(fz, "pmrt_eq"),
                                     "pmrt_eq.w200": X._arm(fz, "pmrt_eq", {"width": 2.0})})}
    pairs = []
    for k, spec in specs.items():
        spec["name"] = f"t_{k}"
        sp, out = str(tmp_path / f"{k}.json"), str(tmp_path / f"{k}.jsonl")
        json.dump(spec, open(sp, "w", encoding="utf-8"))
        with X.installed(spec):
            C.run_units(spec, C.expand(spec), out, isolate=False, log=lambda *_: None)
        pairs.append((sp, out))
    res = X.analyse(pairs)
    prov = {k: v["provenance"] for k, v in res["checks"].items()}
    assert prov["t_d10"] == {"ok": True, "datasets": 2, "equal_to_frozen": 2, "differ_from_frozen": 0,
                             "expected": "equal"}
    assert prov["t_d05"]["ok"] and prov["t_d05"]["differ_from_frozen"] == 2
    assert prov["t_x3"]["ok"] and prov["t_x3"]["equal_to_frozen"] == 2
    assert {(p["spec"], p["arm"]) for p in res["pooled"]} == {("t_d10", "corr"), ("t_d05", "corr"),
                                                              ("t_x3", "pmrt_eq"), ("t_x3", "pmrt_eq.w200")}
    assert all(t["trend"].startswith("not reported") for t in res["x2_trend"])      # 2 of the 4 deltas only
    assert {c["family"] for c in res["x3_curve"]} == {"width wider"}
    md = X.analyse_md(res)
    assert res["wording"] in md and "## X3 degradation curve" in md
    assert X.trend_word([0.1, 0.1, 0.2, 0.3]) == "rises" and X.trend_word([0.3, 0.2, 0.2, 0.1]) == "falls"
    assert X.trend_word([0.1, 0.3, 0.2, 0.3]) == "is not monotone" and X.trend_word([0.1] * 4) == "does not change"

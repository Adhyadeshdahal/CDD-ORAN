"""E6-P follow-up study 2b ANALYZER: the certified-safe referee (protocol docs/benchmark/E6P_CERTSAFE_PROTOCOL.md;
driver scratchpad/e6_dev/e6p_certsafe.py; gate cdd_oran/decision/certsafe.py). Reuses the option (a) analyzer
(scratchpad/e6_dev/e6p_conf_analyze.py: arm_stats_multi, d1_check, d2_check, holm) and the e6p_step2_dev
definitions unchanged.

  python scratchpad/e6_dev/e6p_certsafe_analyze.py bounds --disc SPEC --disc-json disc_conf.json [--out DIR]
         [--maps docs/benchmark/artifacts/E6P_CONF_MAPS.json] [--cache-dir DIR] [--workers 4] [--stage eval]
         [--allow-partial]
  python scratchpad/e6_dev/e6p_certsafe_analyze.py calib --records SPEC [--out DIR] [--allow-smoke]
  python scratchpad/e6_dev/e6p_certsafe_analyze.py build --bounds B.json --calib C.json --disc-json disc_conf.json
         [--maps ...E6P_CONF_MAPS.json] [--out docs/benchmark/artifacts/E6P_CERTSAFE.json] [--allow-partial]
  python scratchpad/e6_dev/e6p_certsafe_analyze.py verify [--artifact ...E6P_CERTSAFE.json]
  python scratchpad/e6_dev/e6p_certsafe_analyze.py eval --records SPEC --disc-json disc_conf.json
         [--artifact ...E6P_CERTSAFE.json] [--out DIR] [--n-boot 10000] [--allow-smoke]

bounds (Kaggle job; reads the option (a) DISC records 186100-186699, no new data): the DISC unit table exactly as the
  option (a) maps (disc_bench.load_pool H 90 / H_pre 90, stage "eval"); checks that the design slope reproduces
  every MG:PMRT map |beta| and the naive slope every associational map |beta| (rel. tol 1e-9) and that p_mismatch
  is 0; then for every (family, rel, k in GUARD_KPIS): PMRT design slope + martingale SE (certsafe.design_slope_se)
  and naive OLS slope + iid SE (certsafe.naive_slope_se). Output OUT/certsafe_bounds.json.
calib (local; DEV 191000-191039): G_k = mean per-episode scored guard total of CAL:allaccept (rlf: "rlf", v:
  "viol_ue_s"); N(f, d) = mean per-episode "dir_<f>_<d>" unit count of CAL:allaccept; repro CAL:allaccept vs noarb
  on 191000-191001 (bit identity -> noarb alias). Output OUT/certsafe_calib.json.
build (freeze 2): tau = certsafe.tau_table(G, N); bounds per method (pmrt, ols from bounds; gt = the 95 % CI of the
  option (a) GT cells in disc_conf.json); per CS arm: certify -> uncertified classes, certsafe signature, decision
  table; alias table (noarb alias from calib). Writes the artifact + .sha256. Refuses a partial bounds / calib, a
  changed maps artifact (MAPS_SHA256) or a disc_conf.json that is not the one the maps artifact pins.
eval: Gate A stats of every arm on ONE paired bootstrap matrix (N 10000, default_rng([6624, 20, n_seeds]), as option
  (a)); E (CS:PMRT eligible, point), D1 / D2 over the 13 CS associational arms (definitions unchanged), D3
  (R*(CS:PMRT) - R*(never_sleep) > 0 with paired LB90 > 0), S1 (RLF ratio upper 90 % CI <= 1.20, secondary),
  descriptive contrasts, verdict INVALID > NOT ELIGIBLE > PASS (D1, D2, D3) > PARTIAL (some) > FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e6p_certsafe as X  # noqa: E402
import e6p_conf as C  # noqa: E402
import e6p_conf_analyze as AN  # noqa: E402
import e6p_disc_analyze_v4 as A4  # noqa: E402
import numpy as np  # noqa: E402

from cdd_oran.decision import certsafe as CS  # noqa: E402
from cdd_oran.decision import mapgate as MG  # noqa: E402

SCHEMA_BOUNDS = "e6p-certsafe-bounds/1"
SCHEMA_CALIB = "e6p-certsafe-calib/1"
SCHEMA_ART = "e6p-certsafe/1"
SCHEMA_EVAL = "e6p-certsafe-eval/1"
ANALYZER = "scratchpad/e6_dev/e6p_certsafe_analyze.py"
DRIVER = "scratchpad/e6_dev/e6p_certsafe.py"
CERTSAFE = "cdd_oran/decision/certsafe.py"
MAPGATE = "cdd_oran/decision/mapgate.py"
N_BOOT = AN.N_BOOT
RLF_CI_MAX = 1.20                                  # S1 (reviewer B): upper 90 % CI of the RLF ratio
SECONDARY = ("noarb", "incumbent", "never_sleep", "MG:PMRT", "CS:GT", "CS:rand", "CS:blanket2")
log = A4.log


# ============================================================================================ bounds (DISC)
def bounds_tables(ud, maps: dict) -> dict:
    """{"pmrt": {(f, rel, k): bound}, "ols": {...}, "repro": {...}} on the DISC UnitData ``ud``; ``maps`` {arm: M}
    of the option (a) maps artifact (for the beta reproduction checks)."""
    pm, ols = {}, {}
    for f in MG.FAMILIES:
        for rel in MG.RELS_V2:
            for k in CS.GUARD_KPIS:
                pm[(f, rel, k)] = CS.design_slope_se(ud, f, rel, k)
                ols[(f, rel, k)] = CS.naive_slope_se(ud, f, rel, k)
    rep = {"pmrt": [], "ols": []}
    for h, b in maps.get("MG:PMRT", {}).items():
        if b is None:
            continue
        got = abs(CS.design_slope_se(ud, *h)["beta"])
        rep["pmrt"].append([*h, abs(b), got, bool(np.isclose(got, abs(b), rtol=1e-9, atol=0.0))])
    for arm in C.ASSOC_ARMS:
        for h, b in maps.get(arm, {}).items():
            if b is None:
                continue
            got = abs(CS.naive_slope_se(ud, *h)["beta"])
            rep["ols"].append([arm, *h, abs(b), got, bool(np.isclose(got, abs(b), rtol=1e-9, atol=0.0))])
    ok = all(r[-1] for r in rep["pmrt"]) and all(r[-1] for r in rep["ols"]) and bool(rep["pmrt"])
    return {"pmrt": pm, "ols": ols, "repro": dict(rep, ok=ok)}


def cmd_bounds(a) -> dict:
    from cdd_oran.decision import disc_bench as DB
    t_all = time.time()
    out = a.out or os.environ.get("JOB_OUT") or "."
    os.makedirs(out, exist_ok=True)
    maps_st = C.maps_status(a.maps)
    maps = X.maps_from_artifact(a.maps)
    disc = json.load(open(a.disc_json))
    art = json.load(open(a.maps))
    files = A4.expand(a.disc)
    caches = A4.build_caches(files, a.stage, os.path.join(a.cache_dir or os.path.join(out, "cache"), "disc"), "disc",
                             a.workers)
    pool = DB.load_pool(caches, H=AN.H, H_pre=AN.H_PRE, stages={a.stage})
    ud = pool.unit_data()
    seeds = sorted({int(s) for s in np.unique(ud.seed)})
    log(f"DISC unit table: {ud.n} units, {len(seeds)} episodes, p_mismatch {ud.meta.get('p_mismatch')}")
    tb = bounds_tables(ud, maps)
    full = (X.freeze_status(ROOT)["frozen"] and seeds == C.stage_seeds("disc") and a.stage == "eval"
            and ud.meta.get("p_mismatch") == 0
            and maps_st["ok"] and A4.sha_lf(a.disc_json) == art["sha256"]["disc_conf.json"])
    mode = "full" if full else "partial"
    if not full and not a.allow_partial:
        raise SystemExit(f"bounds need the frozen protocol, the 600 DISC episodes, the frozen maps artifact and its disc_conf.json "
                         f"(--allow-partial for a smoke): seeds {len(seeds)}, maps {maps_st['ok']}")
    try:
        import cloud
        plat = cloud.numeric_env()
    except Exception as e:                                                # noqa: BLE001
        plat = {"error": repr(e)}
    z_cmp = []
    for (f, rel, k), b in tb["pmrt"].items():
        hy = disc["disc_hyp"].get(f"{f}|{rel}|{k}", {})
        z_cmp.append([f, rel, k, b.get("z"), (hy.get("plain_c") or {}).get("z"), (hy.get("loadsp_c") or {}).get("z")])
    rep = {"schema": SCHEMA_BOUNDS, "mode": mode, "argv": sys.argv[1:], "n_units": int(ud.n), "n_episodes": len(seeds),
           "seeds": [seeds[0], seeds[-1]] if seeds else None, "p_mismatch": ud.meta.get("p_mismatch"),
           "pmrt": CS.bounds_to_json(tb["pmrt"]), "ols": CS.bounds_to_json(tb["ols"]), "beta_repro": tb["repro"],
           "z_compare": {"cols": ["family", "rel", "kpi", "z_design_slope", "z_plain_c", "z_loadsp_c"],
                         "rows": z_cmp},
           "sha256": {"maps_artifact": maps_st["sha256"], "disc_conf.json": A4.sha_lf(a.disc_json),
                      CERTSAFE: A4.sha_lf(os.path.join(ROOT, CERTSAFE)), ANALYZER: A4.sha_lf(os.path.join(ROOT, ANALYZER))},
           "platform": plat, "timing_s": round(time.time() - t_all, 1)}
    log(f"beta reproduction ok {tb['repro']['ok']}; mode {mode}")
    for f, rel, k, z1, z2, _z3 in z_cmp:
        if k == "rlf":
            b = tb["pmrt"][(f, rel, k)]
            log(f"  {f:8s} {rel:3s} rlf: beta {b['beta']:+.4f} se {b['se'] or float('nan'):.4f} z {z1 if z1 is not None else float('nan'):+.2f} "
                f"(plain_c z {z2 if z2 is not None else float('nan'):+.2f})")
    A4.dump(rep, os.path.join(out, "certsafe_bounds.json"))
    return rep


# ============================================================================================ calibration (DEV)
def load_jobs(files: list, stage: str, allow_smoke: bool = False) -> tuple[dict, list]:
    """({seed: {arm: record}}, headers) of the certsafe driver's job records of ``stage``."""
    R, heads = {}, []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("kind") == "header" and r.get("driver") == "e6p_certsafe" and r.get("cs_stage") == stage:
                    heads.append(r)
                if (r.get("kind") != "job" or r.get("schema") != X.SCHEMA_EVAL or r.get("sub") != X.SUB
                        or r.get("cs_stage") != stage or (r.get("smoke") and not allow_smoke)):
                    continue
                R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    return R, heads


def calib_from(R: dict) -> dict:
    """G_k, N(f, d) and the repro check from DEV job records {seed: {arm: rec}}."""
    seeds = sorted(s for s in R if X.CAL_ARM in R[s])
    cal = [R[s][X.CAL_ARM] for s in seeds]
    G = {k: float(np.mean([float(r[CS.GUARD_FIELD[k]]) for r in cal])) for k in CS.GUARD_KPIS} if cal else {}
    N = {}
    for f in MG.FAMILIES:
        for d in CS.DIRS:
            key = f"dir_{f}_{d:+d}"
            N[(f, d)] = float(np.mean([float(((r.get("policy_counts") or {}).get("policy") or {}).get(key, 0))
                                       for r in cal])) if cal else 0.0
    rep = {}
    for s in X.REPRO_SEEDS:
        if s in R and X.CAL_ARM in R[s] and "noarb" in R[s]:
            rep[s] = max(abs(float(R[s][X.CAL_ARM][f]) - float(R[s]["noarb"][f])) for f in C.D.SUM_FIELDS)
    ok = len(rep) == len(X.REPRO_SEEDS) and all(v == 0 for v in rep.values())
    return {"seeds": seeds, "n": len(seeds), "G": G, "N": [[f, d, v] for (f, d), v in sorted(N.items())],
            "repro": {"per_seed": rep, "ok": ok}, "noarb_alias": bool(ok)}


def cmd_calib(a) -> dict:
    out = a.out or os.environ.get("JOB_OUT") or "."
    os.makedirs(out, exist_ok=True)
    R, heads = load_jobs(A4.expand(a.records), "dev", a.allow_smoke)
    cb = calib_from(R)
    smoke = any(r.get("smoke") for s in R for r in R[s].values())
    plats = {AN.platform_key(h.get("platform")) for h in heads if not h.get("smoke")}
    full = cb["seeds"] == X.stage_seeds("dev") and not smoke and cb["repro"]["ok"] and len(plats) == 1
    rep = {"schema": SCHEMA_CALIB, "mode": "full" if full else "partial", "argv": sys.argv[1:], **cb,
           "platform": {"keys": sorted(map(list, plats), key=str), "equal": len(plats) == 1},
           "tau_preview": CS.tau_to_json(CS.tau_table(cb["G"], {(f, d): v for f, d, v in cb["N"]})) if cb["G"] else None}
    log(f"calib: {cb['n']} DEV seeds, G {cb['G']}, repro {cb['repro']}, mode {rep['mode']}")
    log(f"  N(f, d): {cb['N']}")
    A4.dump(rep, os.path.join(out, "certsafe_calib.json"))
    return rep


# ============================================================================================ artifact (freeze 2)
def build_artifact(bounds: dict, calib: dict, disc: dict, maps_path: str, sha: dict, allow_partial: bool = False):
    if bounds.get("schema") != SCHEMA_BOUNDS or calib.get("schema") != SCHEMA_CALIB:
        raise SystemExit("not a certsafe_bounds.json / certsafe_calib.json")
    if not bounds["beta_repro"]["ok"] and not allow_partial:
        raise SystemExit("bounds: the DISC unit table does not reproduce the frozen map betas")
    if (bounds["mode"] != "full" or calib["mode"] != "full") and not allow_partial:
        raise SystemExit(f"partial bounds ({bounds['mode']}) / calib ({calib['mode']}): only full ones can be frozen")
    mst = C.maps_status(maps_path)
    art_maps = json.load(open(maps_path))
    if not allow_partial and (not mst["ok"] or sha["disc_conf.json"] != art_maps["sha256"]["disc_conf.json"]
                              or bounds["sha256"]["maps_artifact"] != mst["sha256"]):
        raise SystemExit("maps artifact / disc_conf.json differ from the frozen option (a) ones")
    if disc["label"]["label"] == "INVALID":
        raise SystemExit("discovery INVALID")
    if not X.freeze_status(ROOT)["frozen"] and not allow_partial:
        raise SystemExit(f"the protocol {X.PROTOCOL_DOC} is not frozen (FROZEN_SHA256_CS)")
    maps = X.maps_from_artifact(maps_path)
    tau = CS.tau_table(calib["G"], {(f, int(d)): v for f, d, v in calib["N"]})
    bset = {"pmrt": CS.bounds_from_json(bounds["pmrt"]), "ols": CS.bounds_from_json(bounds["ols"]),
            "gt": CS.gt_ci_bounds(disc["gt"]["cells"])}
    tables = X.cs_tables(maps, bset, tau)
    al = X.cs_alias_table(tables, X.THETA, bool(calib["noarb_alias"]))
    arms = {}
    for arm, t in tables.items():
        arms[arm] = {"source": t["source"], "map": MG.map_to_json(t["map"]), "bound_method": t["bound_method"],
                     "certification": t["cert"]["classes"], "uncertified": [list(x) for x in t["uncertified"]],
                     "signature": t["signature"], "alias_of": al["alias_of"][arm], "decision_table": t["decision_table"]}
    return {"schema": SCHEMA_ART, "mode": "full" if bounds["mode"] == calib["mode"] == "full" else "partial",
            "protocol": X.freeze_status(ROOT), "maps_artifact": {"path": X.MAPS_DOC, "sha256": mst["sha256"]},
            "constants": {"rho": CS.RHO, "z90": CS.Z90, "guard_kpis": list(CS.GUARD_KPIS), "rels": list(MG.RELS_V2),
                          "guard_field": CS.GUARD_FIELD, "harm_direction": "deferral: h = -d",
                          "tau_rule": "tau(f, d, k) = (rho - 1) * G_k / (max(N(f, d), 1) * 3)"},
            "calib": {"G": calib["G"], "N": calib["N"], "seeds": [calib["seeds"][0], calib["seeds"][-1]]
                      if calib["seeds"] else None, "repro": calib["repro"]},
            "tau": CS.tau_to_json(tau), "bounds": {k: CS.bounds_to_json(v) for k, v in bset.items()},
            "arms": arms, "jobs": al["jobs"], "alias_of": al["alias_of"], "noarb_alias": al["noarb_alias"],
            "all_accept_signature": al["all_accept"],
            "mapgate": {"theta": X.THETA, "k_conf": X.K_CONF, "T": X.T, "open_rule": X.OPEN_RULE, "warmup_s": 0.0,
                        "directional": True, "sha256": A4.sha_lf(os.path.join(ROOT, MAPGATE))},
            "sha256": dict(sha, **{CERTSAFE: A4.sha_lf(os.path.join(ROOT, CERTSAFE)),
                                   ANALYZER: A4.sha_lf(os.path.join(ROOT, ANALYZER)), DRIVER: X.driver_sha(ROOT)}),
            "platform": {"bounds": bounds.get("platform"), "calib": calib.get("platform")}}


def cmd_build(a):
    bounds, calib, disc = (json.load(open(p)) for p in (a.bounds, a.calib, a.disc_json))
    sha = {"certsafe_bounds.json": A4.sha_lf(a.bounds), "certsafe_calib.json": A4.sha_lf(a.calib),
           "disc_conf.json": A4.sha_lf(a.disc_json)}
    art = build_artifact(bounds, calib, disc, a.maps, sha, a.allow_partial)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="\n") as fh:
        json.dump(art, fh, indent=1, default=A4._js)
    s = A4.sha_lf(a.out)
    open(a.out + ".sha256", "w", newline="\n").write(f"{s}  {os.path.basename(a.out)}\n")
    print(json.dumps({"out": a.out, "sha256": s, "mode": art["mode"], "jobs": art["jobs"],
                      "uncertified": {k: v["uncertified"] for k, v in art["arms"].items()}}, indent=1))
    return art


def verify_artifact(path: str) -> dict:
    a = json.load(open(path))
    sha = A4.sha_lf(path)
    loaded = X.load_artifact(path)                     # signatures and alias table reproduce (or SystemExit)
    tau = CS.tau_from_json(a["tau"])
    bset = {k: CS.bounds_from_json(v) for k, v in a["bounds"].items()}
    maps = {v["source"]: MG.map_from_json(v["map"]) for v in a["arms"].values()}
    re_t = X.cs_tables(maps, bset, tau)
    chk = {"schema": a.get("schema") == SCHEMA_ART,
           "uncertified": all(sorted(map(list, re_t[x]["uncertified"])) == sorted(a["arms"][x]["uncertified"])
                              for x in a["arms"]),
           "tau_rule": CS.tau_to_json(CS.tau_table(a["calib"]["G"], {(f, int(d)): v for f, d, v in a["calib"]["N"]}))
                       == a["tau"],
           "arms_complete": sorted(a["arms"]) == sorted(X.CS_ARMS),
           "maps_artifact": a["maps_artifact"]["sha256"] == C.MAPS_SHA256 == C.maps_status()["sha256"],
           "maps_equal": all(maps[k] == m for k, m in X.maps_from_artifact().items() if k in maps),
           "mapgate_sha": a["mapgate"]["sha256"] == A4.sha_lf(os.path.join(ROOT, MAPGATE)),
           "certsafe_sha": a["sha256"][CERTSAFE] == A4.sha_lf(os.path.join(ROOT, CERTSAFE)),
           "analyzer_sha": a["sha256"][ANALYZER] == A4.sha_lf(os.path.join(ROOT, ANALYZER)),
           "driver_sha": a["sha256"][DRIVER] == X.driver_sha(ROOT),
           "mode_full": a["mode"] == "full", "jobs": loaded["alias"]["jobs"] == a["jobs"]}
    return {"path": path, "sha256": sha, "checks": chk, "ok": all(chk.values()),
            "driver_CERTSAFE_SHA256": X.CERTSAFE_SHA256, "driver_matches": X.CERTSAFE_SHA256 == sha}


def cmd_verify(a):
    v = verify_artifact(a.artifact)
    print(json.dumps(v, indent=1))
    print(f'CERTSAFE_SHA256 = "{v["sha256"]}"   # e6p_certsafe.py (after the freeze-2 commit)')
    if not v["ok"]:
        raise SystemExit("certsafe artifact verification FAILED")
    return v


# ============================================================================================ eval
def d3_check(point: dict, boot: dict, primary: str = X.PRIMARY_ARM, ref: str = "never_sleep") -> dict:
    """D3: R*(primary) - R*(never_sleep) > 0 with paired LB90 (5th percentile) > 0."""
    d = float(point[primary]["Rstar"] - point[ref]["Rstar"])
    diff = boot[primary]["Rstar"] - boot[ref]["Rstar"]
    fin = diff[np.isfinite(diff)]
    lb = float(np.quantile(fin, 0.05)) if fin.size else float("nan")
    return {"pass": bool(np.isfinite(d) and d > 0 and np.isfinite(lb) and lb > 0), "delta3": d, "lb90": lb,
            "ci90": AN._ci(diff), "ref": ref}


def s1_check(boot: dict, primary: str = X.PRIMARY_ARM, cap: float = RLF_CI_MAX) -> dict:
    """S1 (secondary): the upper end of the 90 % CI (95th percentile) of the primary's RLF ratio <= cap."""
    r = boot[primary]["guard"]["rlf"]
    fin = r[np.isfinite(r)]
    ub = float(np.quantile(fin, 0.95)) if fin.size else float("nan")
    return {"pass": bool(np.isfinite(ub) and ub <= cap), "rlf_ratio_ub90": ub, "cap": cap}


def eval_verdict(k0: bool, k0n: bool, e: bool, d1: bool, d2: bool, d3: bool, complete: bool, why=()) -> dict:
    """INVALID (K0 / K0n of option (a)) > NOT ELIGIBLE (E) > PASS (D1, D2 and D3) > PARTIAL (one or two) > FAIL."""
    if not (k0 and k0n):
        v = "INVALID"
    elif not e:
        v = "NOT ELIGIBLE"
    elif d1 and d2 and d3:
        v = "PASS"
    elif d1 or d2 or d3:
        v = "PARTIAL"
    else:
        v = "FAIL"
    return {"verdict": v, "label": v if complete else f"NOT A VERDICT (would-be: {v}; {'; '.join(why)})",
            "complete": complete, "precedence": "INVALID > NOT ELIGIBLE (E) > PASS (D1, D2, D3) > PARTIAL > FAIL"}


def cmd_eval(a) -> dict:
    t_all = time.time()
    out = a.out or os.environ.get("JOB_OUT") or "."
    os.makedirs(out, exist_ok=True)
    disc = json.load(open(a.disc_json))
    av = verify_artifact(a.artifact)
    art = json.load(open(a.artifact))
    R, heads = load_jobs(A4.expand(a.records), "eval", a.allow_smoke)
    for s in R:                                                    # aliased CS arms from their simulated target
        for arm, tgt in art["alias_of"].items():
            if tgt != arm and tgt in R[s] and arm not in R[s]:
                R[s][arm] = dict(R[s][tgt], arm=arm, alias_of=tgt)
    arms = [x for x in X.ALL_ARMS if any(x in R[s] for s in R)]
    st = AN.arm_stats_multi(R, arms, a.n_boot)
    if not st["n"]:
        raise SystemExit("no EVAL seed with every anchor and arm")
    P, B = st["point"], st["boot"]
    rep = {"schema": SCHEMA_EVAL, "argv": sys.argv[1:], "artifact_verify": av, "n_seeds": st["n"],
           "arms_present": arms, "missing_arms": [x for x in X.ALL_ARMS if x not in arms],
           "gate_a": {k: st[k] for k in ("V_AA", "V_ref", "ref_arm", "den", "n")}}
    rep["arms"] = {x: dict(P[x], R_ci90=AN._ci(B[x]["R"]), Rstar_ci90=AN._ci(B[x]["Rstar"]),
                           retention_ci90=AN._ci(B[x]["retention"]),
                           guard_ratio_ci90={k: AN._ci(v) for k, v in B[x]["guard"].items()},
                           counts={g: AN._sum_counts(R, st["seeds"], x, g) for g in ("defer", "defer_units", "units",
                                                                                    "policy", "passed")},
                           alias_of=R[st["seeds"][0]][x].get("alias_of")) for x in P}
    A = [x for x in X.CS_ASSOC if x in P]
    sig = {arm: v["signature"] for arm, v in art["arms"].items()}
    pa = X.PRIMARY_ARM
    E = {"pass": bool(P[pa]["eligible"]) if pa in P else False,
         "retention_ci90": rep["arms"].get(pa, {}).get("retention_ci90"),
         "guard_ratio_ci90": rep["arms"].get(pa, {}).get("guard_ratio_ci90")}
    D1 = AN.d1_check(P, B, pa, A) if pa in P and A else {"pass": False, "note": "arms missing"}
    D2 = AN.d2_check(B, pa, A, sig) if pa in P and A else {"pass": False, "note": "arms missing"}
    D3 = d3_check(P, B) if pa in P and "never_sleep" in P else {"pass": False, "note": "arms missing"}
    S1 = s1_check(B) if pa in P else {"pass": False, "note": "arm missing"}
    sec = {c: {"dR": P[pa]["R"] - P[c]["R"], "ci90": AN._ci(B[pa]["R"] - B[c]["R"])}
           for c in SECONDARY if c in P and pa in P}
    why = []
    if st["seeds"] != X.stage_seeds("eval"):
        why.append(f"eval seeds with every arm: {st['n']} / 160")
    if rep["missing_arms"]:
        why.append(f"missing arms {rep['missing_arms']}")
    if any(R[s][x].get("smoke") for s in R for x in R[s]):
        why.append("smoke records")
    if not av["ok"] or not av["driver_matches"]:
        why.append("certsafe artifact sha / checks (CERTSAFE_SHA256)")
    if not X.freeze_status(ROOT)["frozen"]:
        why.append("protocol not frozen (FROZEN_SHA256_CS)")
    plats = {AN.platform_key(h.get("platform")) for h in heads if not h.get("smoke")}
    if len(plats) != 1:
        why.append(f"platform fingerprints differ across eval shards: {sorted(map(str, plats))}")
    if {h.get("certsafe_sha256") for h in heads} - {art["sha256"][CERTSAFE]}:
        why.append("certsafe.py sha256 of the eval runs differs from the artifact")
    k0, k0n = bool(disc["K0"]["pass"]), bool((disc.get("K0n") or {}).get("pass"))
    V = eval_verdict(k0, k0n, E["pass"], D1["pass"], D2["pass"], D3["pass"], not why, why)
    rep.update(K0=k0, K0n=k0n, E=E, D1=D1, D2=D2, D3=D3, S1=S1, secondary=sec, verdict=V,
               cpu_h_eval=float(sum(R[s][x].get("cpu_s", 0.0) for s in R for x in R[s] if not R[s][x].get("alias_of"))
                                / 3600), timing_s=round(time.time() - t_all, 1))
    f = C.D._f
    log(f"EVAL: {st['n']} seeds; den {st['den']:.2f}")
    for x in X.ALL_ARMS:
        if x in P:
            p, ci = P[x], rep["arms"][x]["R_ci90"]
            log(f"  {x:20s} R {f(p['R'])} [{f(ci[0])},{f(ci[1])}] R* {f(p['Rstar'])} ret {p['retention']:.3f} "
                + " ".join(f"{p['guard_ratio'][k]:.2f}" for k in C.D.GUARD_KEYS) + f" {'Y' if p['eligible'] else 'n'}")
    log(f"E {E['pass']} D1 {D1['pass']} D2 {D2['pass']} D3 {D3['pass']} (delta3 {D3.get('delta3')}, LB90 "
        f"{D3.get('lb90')}) S1 {S1['pass']} (RLF UB90 {S1.get('rlf_ratio_ub90')})")
    log(f"VERDICT: {V['label']}")
    A4.dump(rep, os.path.join(out, "eval_certsafe.json"))
    A4.dump({k: rep[k] for k in ("verdict", "K0", "K0n", "E", "D1", "D2", "D3", "S1", "n_seeds")},
            os.path.join(out, "verdict_certsafe.json"))
    return rep


# ============================================================================================ cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    maps_default = os.path.join(ROOT, C.MAPS_DOC)
    art_default = os.path.join(ROOT, X.ARTIFACT_DOC)
    b = sp.add_parser("bounds")
    b.add_argument("--disc", required=True)
    b.add_argument("--disc-json", dest="disc_json", required=True)
    b.add_argument("--maps", default=maps_default)
    b.add_argument("--out", default=None)
    b.add_argument("--cache-dir", dest="cache_dir", default=None)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--stage", default="eval")
    b.add_argument("--allow-partial", dest="allow_partial", action="store_true")
    c = sp.add_parser("calib")
    c.add_argument("--records", required=True)
    c.add_argument("--out", default=None)
    c.add_argument("--allow-smoke", dest="allow_smoke", action="store_true")
    u = sp.add_parser("build")
    u.add_argument("--bounds", required=True)
    u.add_argument("--calib", required=True)
    u.add_argument("--disc-json", dest="disc_json", required=True)
    u.add_argument("--maps", default=maps_default)
    u.add_argument("--out", default=art_default)
    u.add_argument("--allow-partial", dest="allow_partial", action="store_true")
    v = sp.add_parser("verify")
    v.add_argument("--artifact", default=art_default)
    e = sp.add_parser("eval")
    e.add_argument("--records", required=True)
    e.add_argument("--disc-json", dest="disc_json", required=True)
    e.add_argument("--artifact", default=art_default)
    e.add_argument("--out", default=None)
    e.add_argument("--n-boot", dest="n_boot", type=int, default=N_BOOT)
    e.add_argument("--allow-smoke", dest="allow_smoke", action="store_true")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    return {"bounds": cmd_bounds, "calib": cmd_calib, "build": cmd_build, "verify": cmd_verify,
            "eval": cmd_eval}[a.op](a)


if __name__ == "__main__":
    main()

"""E6-P MSCR causal discovery, step 1 RE-TEST (discovery v2): LOCAL ANALYSIS of MSCR-CRT v2
(cdd_oran/decision/crt_units_v2.py) on the fresh v2 data (protocol docs/benchmark/E6P_DISCOVERY_PROTOCOL_V2.md).

  python scratchpad/e6_dev/e6p_disc_analyze_v2.py analyze --dev F --eval F --gt F --placebo F [--placebo-dev F]
         [--prof F] [--json OUT] [--B 9999] [--methods shap_gbdt,corr,...] [--no-baselines] [--no-ann] [--with-v1]
         [--allow-smoke]
Each F may be a comma-separated list of files.
  --dev          DEV records (stage "dev"): baseline thresholds are tuned here (v1 code path, far-FPR rule).
  --eval         EVAL records (stage "eval"; v2 records sub "v2", fold 0-3). If the files carry stage "dev" instead,
                 the run is a PIPELINE DRY RUN (EVAL := DEV, pseudo-folds = episode index % 4): NOT A VERDICT.
  --gt           knockout GT records (stage "gt"; fresh ones sub "ext"); gt_p.summarize, frozen rule, "dir".
  --placebo      PLACEBO records (stage "placebo"; fresh ones sub "ext"): K0 of MSCR-CRT v2 + the validity column.
  --placebo-dev  optional second placebo set (e.g. the step-1 placebo-1): descriptive K0 / validity only.
  --with-v1      also run MSCR-CRT v1 (crt_units.run_crt_units, no audit) on EVAL: comparison only (slow).
Verdict (precedence top-down, as v1): INVALID (K0) > NO-CHAIN (G) > UNDERPOWERED (K1) > PASS (P1v2 and P2v2) /
PARTIAL (P1v2 only) / KILL. Labelled PROVISIONAL unless the v2 protocol doc has a line "FROZEN: yes".
  P1v2  chain set C = {sleep->nbr load +, sleep->nbr pv +, sleep->nbr v +, ptx->nbr load -} restricted to members
        GT-TRUE with that sign. PASS iff sleep->nbr pv declared with sign +1 AND >= min(3, |C_true|) members of C_true
        declared with the correct sign AND overall precision (GT TRUE u NULL) >= 0.80 AND overall sign accuracy
        >= 0.90. Far declarations are reported, not a criterion.
  P2v2  MSCR-CRT v2 indirect (nbr) F1 >= the best DEV-tuned baseline's, pooled AND in >= 3 of 4 folds.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e6p_disc_analyze as A1  # noqa: E402  (v1 analyzer: k1_support, g_premise, gt_episode, verdict, json helpers)

from cdd_oran.decision import baselines_disc as BD  # noqa: E402
from cdd_oran.decision import crt_units as CU  # noqa: E402
from cdd_oran.decision import crt_units_v2 as V2  # noqa: E402
from cdd_oran.decision import edge_score as ES  # noqa: E402
from cdd_oran.decision import gt_p as G  # noqa: E402

SCHEMA = "e6p-disc-rec/1"
PROTOCOL_DOC_V2 = "docs/benchmark/E6P_DISCOVERY_PROTOCOL_V2.md"
K0_RULE = {"binom_level": 0.01, "max_by": 1}
N_FOLDS = 4
FOLDS = tuple(f"fold{k}" for k in range(N_FOLDS))
P2_MIN_FOLDS = 3
CHAIN = (("sleep", "nbr", "load", 1), ("sleep", "nbr", "pv", 1), ("sleep", "nbr", "v", 1), ("ptx", "nbr", "load", -1))
PREMISE_EDGE = ("sleep", "nbr", "pv")
P1V2_RULES = {"chain_min": 3, "precision": 0.80, "sign_acc": 0.90}
log = A1.log


# ---------------------------------------------------------------------------------------------- io
def read_records(spec, stages, what, allow_smoke=False) -> tuple[list, str | None]:
    """Episode records of the files in ``spec`` whose stage is in ``stages`` (one stage per option). Returns (records
    sorted by (seed, sub), the stage)."""
    if not spec:
        return [], None
    recs = []
    for path in [p for p in spec.split(",") if p]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("kind") == "episode" and r.get("schema") == SCHEMA:
                    recs.append(r)
    st = sorted({r["stage"] for r in recs})
    if len(st) != 1 or st[0] not in stages:
        raise SystemExit(f"--{what}: stages {st}, expected one of {sorted(stages)}")
    smoke = [r for r in recs if r.get("smoke")]
    if smoke and not allow_smoke:
        raise SystemExit(f"--{what}: {len(smoke)} smoke records (pass --allow-smoke for a pipeline dry run)")
    uniq = {}
    for r in recs:
        uniq[tuple(r["key"]) + (bool(r.get("smoke")),)] = r
    return sorted(uniq.values(), key=lambda r: (r["seed"], r.get("sub", ""))), st[0]


def protocol_status() -> dict:
    path = os.path.join(ROOT, PROTOCOL_DOC_V2)
    if not os.path.exists(path):
        return {"doc": PROTOCOL_DOC_V2, "exists": False, "frozen": False, "sha256": None, "line": None}
    raw = open(path, "rb").read()
    txt = raw.decode("utf-8", "replace")
    line = next((ln.strip() for ln in txt.splitlines() if ln.strip().startswith("FROZEN:")), "FROZEN: ?")
    return {"doc": PROTOCOL_DOC_V2, "exists": True, "frozen": line.lower().startswith("frozen: yes"), "line": line,
            "sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


def peak_rss_mb() -> float | None:
    """Peak resident set of this process (Windows: PeakWorkingSetSize; POSIX: ru_maxrss)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return round(c.PeakWorkingSetSize / 2 ** 20, 1)
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:                                                  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------------- pieces
def crt_summary(run: dict) -> dict:
    res = run["results"]
    return {"n_declared": run["n_declared"], "n_tested": run["n_tested"], "power_floor": run["power_floor"],
            "wall_s": run.get("wall_s"),
            "declared": [(r["family"], r["relation"], r["kpi"], r["sign"], r["p"], round(r["beta"], 3),
                          round(r["z_approx"], 2)) for r in res if r["status"] == "declared"],
            "smallest_p": sorted((float(r["p"]), r["family"], r["relation"], r["kpi"]) for r in res
                                 if r["status"] != "undetermined")[:10],
            "undetermined": sorted({(r["family"], r["reason"]) for r in res if r["status"] == "undetermined"}),
            "per_hypothesis": [{"h": (r["family"], r["relation"], r["kpi"]), "status": r["status"], "p": r["p"],
                                "beta": r["beta"], "z_approx": r["z_approx"], "sign": r["sign"], "n": r["n"],
                                "resid": r["resid"]} for r in res]}


def p1v2_check(decl: dict, ref: dict, score: dict) -> dict:
    """P1v2 on MSCR-CRT v2 pooled EVAL (module docstring)."""
    members = []
    for f, rel, k, s in CHAIN:
        g = ref.get((f, rel, k), {"status": "INDET", "sign": 0})
        d = decl.get((f, rel, k)) or {}
        members.append({"h": (f, rel, k), "expected_sign": s, "gt_status": g["status"], "gt_sign": g["sign"],
                        "gt_true_expected": g["status"] == "TRUE" and g["sign"] == s,
                        "declared": bool(d.get("declared")), "declared_sign": int(d.get("sign", 0)),
                        "hit": bool(d.get("declared")) and int(d.get("sign", 0)) == s})
    c_true = [m for m in members if m["gt_true_expected"]]
    need = min(P1V2_RULES["chain_min"], len(c_true))
    n_hit = sum(m["hit"] for m in c_true)
    prem = decl.get(PREMISE_EDGE) or {}
    ov = score["overall"]

    def ge(v, t):
        return bool(np.isfinite(v) and v >= t - 1e-12)
    parts = {"premise_edge_declared_plus": bool(prem.get("declared")) and int(prem.get("sign", 0)) == 1,
             "chain_hits": n_hit >= need,
             "overall_precision": ge(ov["precision"], P1V2_RULES["precision"]),
             "sign_accuracy": ge(ov["sign_acc"], P1V2_RULES["sign_acc"])}
    return {"pass": all(parts.values()), "parts": parts, "rules": dict(P1V2_RULES), "members": members,
            "not_gt_true_expected": [m["h"] for m in members if not m["gt_true_expected"]],
            "values": {"chain_true": len(c_true), "chain_need": need, "chain_hits": n_hit,
                       "overall_precision": ov["precision"], "sign_accuracy": ov["sign_acc"],
                       "far_declared (reported, not a criterion)": score["far_declared"],
                       "indirect_recall_all_gt_true_nbr": score["indirect"]["recall"]}}


def n_declared(decl: dict) -> int:
    return sum(1 for v in decl.values() if v and v.get("declared"))


# ---------------------------------------------------------------------------------------------- analyze
def analyze(a) -> dict:
    t_all = time.time()
    timing = {}
    allow = a.get("allow_smoke", False)
    dev_recs, _ = read_records(a.get("dev"), {"dev"}, "dev", allow)
    if not dev_recs:
        raise SystemExit("--dev is required (baseline thresholds are tuned on DEV)")
    ev_recs, ev_stage = read_records(a.get("eval"), {"eval", "dev"}, "eval", allow)
    gt_recs, _ = read_records(a.get("gt"), {"gt"}, "gt", allow)
    pl_recs, _ = read_records(a.get("placebo"), {"placebo"}, "placebo", allow)
    pld_recs, _ = read_records(a.get("placebo_dev"), {"placebo"}, "placebo-dev", allow)
    prof_recs, _ = read_records(a.get("prof"), {"prof"}, "prof", allow)
    if not pl_recs:
        raise SystemExit("--placebo is required (K0)")
    dry = ev_stage != "eval"
    mode = "dry-run" if dry else ("full" if gt_recs else "partial")
    smoke = any(r.get("smoke") for rs in (dev_recs, ev_recs, gt_recs, pl_recs) for r in rs)
    if smoke and mode == "full":
        mode = "smoke"
    ps = protocol_status()
    cfg = V2.UnitCRTConfigV2(B=int(a.get("B", 9999)))
    if mode != "full":                     # as v1: pipeline exercise (never a verdict)
        cfg = dataclasses.replace(cfg, min_episodes=1)
    n_eps = {"dev": len(dev_recs), "eval": len(ev_recs) if not dry else len(dev_recs), "gt": len(gt_recs),
             "placebo": len(pl_recs), "placebo_dev": len(pld_recs), "prof": len(prof_recs)}
    rep = {"mode": mode, "banner": "FULL ANALYSIS (MSCR-CRT v2)" if mode == "full" else
           "PIPELINE DRY RUN (MSCR-CRT v2) -- NOT A VERDICT", "protocol": ps, "config": dataclasses.asdict(cfg),
           "version": V2.CRT_UNITS_V2_VERSION, "episodes": n_eps, "smoke": smoke,
           "placebo_subs": sorted({r.get("sub", "") for r in pl_recs}),
           "eval_subs": sorted({r.get("sub", "") for r in ev_recs}), "gt_subs": sorted({r.get("sub", "") for r in gt_recs})}
    if dry:
        rep["dry_run_note"] = "EVAL := DEV (in-sample), pseudo-folds = DEV episode index % 4"
    log(f"== {rep['banner']} (mode {mode}); episodes {n_eps}; protocol {ps.get('line')}")

    # K0 (v2 on the placebo) + descriptive second placebo
    t = time.time()
    pdat = CU.build_unit_data(pl_recs)
    del pl_recs
    k0 = V2.placebo_rejection_units_v2(pdat, cfg, **K0_RULE)
    k0.pop("run", None)
    timing["k0_placebo"] = round(time.time() - t, 1)
    rep["K0"] = k0
    log(f"K0 placebo (v2): units {pdat.n}, tested {k0['n_tested']}/{k0['n_hypotheses']}, rate {k0['rate']:.3f} "
        f"({k0['n_reject']} <= alpha, k_max {k0['k_max']}), BY {k0['n_by_declared']} {k0['declared']} -> "
        f"{'PASS' if k0['pass'] else 'FAIL'} [{timing['k0_placebo']} s]")
    pddat = None
    if pld_recs:
        pddat = CU.build_unit_data(pld_recs)
        del pld_recs
        k0d = V2.placebo_rejection_units_v2(pddat, cfg, **K0_RULE)
        k0d.pop("run", None)
        rep["K0_placebo_dev_descriptive"] = k0d
        log(f"K0 placebo-dev (descriptive): rate {k0d['rate']:.3f}, BY {k0d['n_by_declared']}")

    # DEV / EVAL tables
    t = time.time()
    dev = CU.build_unit_data(dev_recs)
    del dev_recs
    if not dry:
        ev = CU.build_unit_data(ev_recs)
        del ev_recs
        folds = {f"fold{k}": ev.subset(ev.fold == k) for k in range(N_FOLDS)}
        rep["fold_units"] = {f: int(d.n) for f, d in folds.items()}
        if int((ev.fold < 0).sum()) or int((ev.fold >= N_FOLDS).sum()):
            rep["fold_warning"] = f"{int(((ev.fold < 0) | (ev.fold >= N_FOLDS)).sum())} units outside folds 0-3"
    else:
        ev = dev
        folds = {f"fold{k}": ev.subset(ev.episode % N_FOLDS == k) for k in range(N_FOLDS)}
        rep["pseudo_folds"] = "DEV episode index % 4 (dry run)"
    timing["unit_tables"] = round(time.time() - t, 1)
    rep["units"] = {"dev": dev.meta["n_units"], "eval": ev.meta["n_units"], "dropped_eval": ev.meta["dropped"],
                    "p_mismatch_eval": ev.meta["p_mismatch"],
                    "by_family_eval": {f: int(len(ev.rows_of(f))) for f in CU.FAMILIES}}
    k1 = A1.k1_support(ev, n_eps["eval"])
    rep["K1"] = k1
    log(f"K1 support: sleep units {k1['sleep_units']} rejects {k1['sleep_rejects']} -> "
        f"{'PASS' if k1['pass'] else 'FAIL'}; units {rep['units']['by_family_eval']}")

    # MSCR-CRT v2
    t = time.time()
    crt = {"pooled": V2.run_crt_units_v2(ev, cfg, CU.SPLIT_POOLED)}
    timing["crt_v2_pooled"] = round(time.time() - t, 1)
    for k, f in enumerate(FOLDS):
        t = time.time()
        crt[f] = V2.run_crt_units_v2(folds[f], cfg, 1 + k)
        timing[f"crt_v2_{f}"] = round(time.time() - t, 1)
    rep["crt_v2"] = {s: crt_summary(r) for s, r in crt.items()}
    log(f"MSCR-CRT v2 pooled: declared {crt['pooled']['n_declared']}/{crt['pooled']['n_tested']} "
        f"(floor ok {crt['pooled']['power_floor']['rank1_ok']}) [{timing['crt_v2_pooled']} s]; folds "
        f"{[crt[f]['n_declared'] for f in FOLDS]}")
    for d in rep["crt_v2"]["pooled"]["declared"]:
        log(f"   declared {d}")

    # optional v1 comparison
    v1run = None
    if a.get("with_v1"):
        t = time.time()
        c1 = CU.UnitCRTConfig(B=cfg.B, min_episodes=cfg.min_episodes)
        v1run = CU.run_crt_units(ev, c1, CU.SPLIT_POOLED, audit=False)
        timing["crt_v1_pooled"] = round(time.time() - t, 1)
        log(f"MSCR-CRT v1 (comparison) pooled: declared {v1run['n_declared']}/{v1run['n_tested']} "
            f"[{timing['crt_v1_pooled']} s]")

    # baselines: exactly the v1 code path (DEV-tuned tau, far-FPR rule), + the placebo(s) as extra splits
    bl = {}
    if not a.get("no_baselines"):
        t = time.time()
        methods = tuple(a["methods"].split(",")) if a.get("methods") else BD.TUNED
        splits = {"pooled": ev, **folds, "placebo": pdat}
        if pddat is not None:
            splits["placebo_dev"] = pddat
        sup = BD.Support() if mode == "full" else BD.Support(min_episodes=1)
        bl = BD.run_baselines(dev, splits, methods, sup=sup, log=log, ann=not a.get("no_ann", False))
        timing["baselines"] = round(time.time() - t, 1)
    if prof_recs:                                                      # PACIFISTA native severity (as v1)
        rep["pacifista_native"] = BD.pacifista_native(prof_recs)
        log(f"PACIFISTA native sigma {rep['pacifista_native']['sigma']}")
        del prof_recs

    # reference
    g = None
    if gt_recs:
        gs = G.summarize([A1.gt_episode(r) for r in gt_recs])
        table = gs["tables"][G.PRIMARY_ORIENT]
        ref = ES.gt_reference(table)
        g = A1.g_premise(table)
        rep["gt"] = {"n_labels": gs["n_labels"], "labels_per_family": gs["labels_per_family"],
                     "counts": table["counts"], "delta": table["delta"], "cells": table["cells"],
                     "receiving": ES.gt_receiving_summary(gs["receiving"]), "G": g,
                     "act_counts": gs["tables"]["act"]["counts"],
                     "nbr_true": sorted((c["family"], c["kpi"], c["sign"]) for c in table["cells"]
                                        if c["relation"] == "nbr" and c["status"] == "TRUE")}
        log(f"GT ({table['orient']}): {table['counts']}; G premise {'PASS' if g['pass'] else 'FAIL'} "
            f"(sleep->nbr pv mean {g['cell']['mean']:.2f} CI {np.round(g['cell']['ci'], 2).tolist()}); nbr TRUE "
            f"{rep['gt']['nbr_true']}")
        rep["reference"] = "knockout GT (dir, frozen gt_p rule)"
        rep["reference_robust"] = "skipped: no 'robust' GT rule is defined in the repo (descriptive, optional)"
        del gt_recs
    else:
        ref = ES.physics_reference()
        rep["reference"] = "PHYSICS PROXY (edge_score.PHYSICS_PRIOR) -- dry run only, not ground truth"

    # scoring
    decl = {"mscr_crt_v2": {s: V2.edges_from_crt_v2(r) for s, r in crt.items()}}
    if v1run is not None:
        decl["mscr_crt_v1"] = {"pooled": CU.edges_from_crt(v1run)}
    for m, r in bl.items():
        decl[m] = {s: v["declared"] for s, v in r["splits"].items() if not s.startswith("placebo")}
    scores = {m: {s: ES.score_method(d, ref) for s, d in dd.items()} for m, dd in decl.items()}
    validity = {"mscr_crt_v2": {"placebo": k0["n_by_declared"]}}
    if "K0_placebo_dev_descriptive" in rep:
        validity["mscr_crt_v2"]["placebo_dev"] = rep["K0_placebo_dev_descriptive"]["n_by_declared"]
    for m, r in bl.items():
        validity[m] = {s: n_declared(v["declared"]) for s, v in r["splits"].items() if s.startswith("placebo")}
    rep["validity_placebo_declarations"] = validity
    p1 = p1v2_check(decl["mscr_crt_v2"]["pooled"], ref, scores["mscr_crt_v2"]["pooled"])
    mscr_f1 = {s: scores["mscr_crt_v2"][s]["indirect"]["f1"] for s in ("pooled", *FOLDS)}
    base_f1 = {m: {s: scores[m][s]["indirect"]["f1"] for s in ("pooled", *FOLDS)} for m in BD.TUNED if m in scores}
    if base_f1:
        p2 = ES.p2_check(mscr_f1, base_f1, FOLDS, min_folds=P2_MIN_FOLDS)
    else:
        p2 = {"pass": False, "folds_ok": None, "per_split": {}, "note": "no baselines run (--no-baselines)"}
    rep["scores"] = scores
    rep["baselines"] = {m: {"tau": r["tau"], "tau_physics": r.get("tau_physics"), "note": r.get("note"),
                            "decl_relations": r.get("decl_relations"), "cpu_s": r.get("cpu_s"),
                            "wall_s": r.get("wall_s")} for m, r in bl.items()}
    rep["declared"] = {m: {s: sorted(("|".join(h), v["sign"]) for h, v in d.items() if v.get("declared"))
                           for s, d in dd.items()} for m, dd in decl.items()}
    rep["P1v2"], rep["P2v2"] = p1, p2
    rep["verdict"] = A1.verdict(k0, g, k1, p1, p2, mode, ps["frozen"])
    if a.get("no_baselines") and mode == "full":
        rep["verdict"]["label"] += " [P2v2 not evaluated: --no-baselines]"
    timing["total"] = round(time.time() - t_all, 1)
    rep["timing_s"] = timing
    rep["peak_rss_mb"] = peak_rss_mb()

    # print
    log("")
    log(f"{'method':>14} {'split':>7} {'ind P':>6} {'ind R':>6} {'ind F1':>6} {'ov P':>6} {'F1':>6} {'sign':>6} "
        f"{'far':>4} {'#dec':>5} {'plc':>4}")
    for m, ss in scores.items():
        for s, v in ss.items():
            ind, ov = v["indirect"], v["overall"]
            plc = validity.get(m, {}).get("placebo", "") if s == "pooled" else ""
            log(f"{m:>14} {s:>7} {ind['precision']:6.2f} {ind['recall']:6.2f} {ind['f1']:6.2f} {ov['precision']:6.2f} "
                f"{ov['f1']:6.2f} {ov['sign_acc']:6.2f} {v['far_declared']:4d} {v['n_declared']:5d} {plc!s:>4}")
    log("chain set C (pooled MSCR-CRT v2):")
    e2 = decl["mscr_crt_v2"]["pooled"]
    for mbr in p1["members"]:
        h = mbr["h"]
        log(f"   {'|'.join(h):18s} exp {mbr['expected_sign']:+d} GT {mbr['gt_status']}({mbr['gt_sign']:+d}) "
            f"declared {mbr['declared']} sign {mbr['declared_sign']:+d} p {e2[h]['p']:.4g} beta {e2[h]['beta']:.4g} "
            f"z {e2[h]['z_approx']:.2f}")
    log(f"P1v2 {'PASS' if p1['pass'] else 'FAIL'} {p1['parts']} values {p1['values']}")
    if base_f1:
        log(f"P2v2 {'PASS' if p2['pass'] else 'FAIL'} (folds ok {p2['folds_ok']}/{N_FOLDS}); pooled MSCR v2 "
            f"{p2['per_split']['pooled']['mscr']} vs best {p2['per_split']['pooled']['best_baseline']} "
            f"({p2['per_split']['pooled']['best_method']})")
    else:
        log("P2v2 not evaluated (--no-baselines)")
    log(f"placebo declarations (validity): {validity}")
    log(f"VERDICT: {rep['verdict']['label']}")
    log(f"timing (s): {timing}; peak RSS {rep['peak_rss_mb']} MB")
    if a.get("json"):
        with open(a["json"], "w", newline="\n") as fh:
            json.dump(A1._keys(rep), fh, indent=1, default=A1._js)
        log(f"wrote {a['json']}")
    return rep


def cli(argv):
    if not argv or argv[0] != "analyze":
        raise SystemExit(__doc__)
    it = iter(argv[1:])
    a = {}
    flags = {"--allow-smoke": "allow_smoke", "--no-baselines": "no_baselines", "--no-ann": "no_ann",
             "--with-v1": "with_v1"}
    vals = {"--dev": "dev", "--eval": "eval", "--gt": "gt", "--placebo": "placebo", "--placebo-dev": "placebo_dev",
            "--prof": "prof", "--json": "json", "--B": "B", "--methods": "methods"}
    for x in it:
        if x in flags:
            a[flags[x]] = True
        elif x in vals:
            a[vals[x]] = next(it)
        else:
            raise SystemExit(f"unknown argument {x}\n{__doc__}")
    if "B" in a:
        a["B"] = int(a["B"])
    return analyze(a)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])

"""E6-P MSCR causal discovery, step 1: LOCAL ANALYSIS (scratchpad/e6_dev/decision/STEP1_MSCR_PLAN.md; protocol
docs/benchmark/E6P_DISCOVERY_PROTOCOL.md). Reads the JSONL records of scratchpad/e6_dev/e6p_discovery.py (schema
"e6p-disc-rec/1") and computes K0, K1, G, P1, P2 and the verdict.

  python scratchpad/e6_dev/e6p_disc_analyze.py analyze --dev F --eval F --gt F --placebo F --prof F [--json OUT]
         [--allow-smoke] [--B 9999] [--audit-B 999] [--no-audit] [--no-rowperm] [--rowperm-perm N]
         [--methods shap_gbdt,corr,...] [--no-ann]
Each F may be a comma-separated list of files. Modes:
  full      DEV + EVAL + GT + PLACEBO (+ PROF): the verdict (PASS / PARTIAL / KILL / INVALID / UNDERPOWERED /
            NO-CHAIN); labelled PROVISIONAL while the protocol doc says "FROZEN: no".
  dry-run   DEV + PLACEBO only (EVAL / GT absent): PIPELINE DRY RUN, NOT A VERDICT. DEV stands in for EVAL (baselines
            tuned AND scored on DEV: in-sample), pseudo-folds = episode index % 3, the reference is the declared
            physics PROXY (edge_score.PHYSICS_PRIOR), never the GT.
Smoke records (smoke = True) are refused unless --allow-smoke, and force NOT A VERDICT.

Verdict (precedence top-down): INVALID (K0 fails) > NO-CHAIN (G fails) > UNDERPOWERED (K1 fails; one extension
to 120 EVAL episodes allowed first) > PASS (P1 and P2) / PARTIAL (P1 only) / KILL (P1 fails).
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cdd_oran.decision import baselines_disc as BD  # noqa: E402
from cdd_oran.decision import crt_units as CU  # noqa: E402
from cdd_oran.decision import edge_score as ES  # noqa: E402
from cdd_oran.decision import gt_p as G  # noqa: E402
from cdd_oran.decision.collect_p import dec  # noqa: E402

SCHEMA = "e6p-disc-rec/1"
PROTOCOL_DOC = "docs/benchmark/E6P_DISCOVERY_PROTOCOL.md"
K0_RULE = {"binom_level": 0.01, "max_by": 1}
K1_RULE = {"min_sleep_units": 60, "min_sleep_rejects": 15, "extension_episodes": 120}
FOLDS = ("fold0", "fold1", "fold2")
VERDICTS = ("PASS", "PARTIAL", "KILL", "INVALID", "UNDERPOWERED", "NO-CHAIN")


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------- io
def read_records(spec, stage, allow_smoke=False) -> list:
    if not spec:
        return []
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
    bad = [r["key"] for r in recs if r["stage"] != stage]
    if bad:
        raise SystemExit(f"--{stage} files carry other stages: {bad[:3]}")
    smoke = [r for r in recs if r.get("smoke")]
    if smoke and not allow_smoke:
        raise SystemExit(f"--{stage}: {len(smoke)} smoke records (pass --allow-smoke for a pipeline dry run)")
    uniq = {}
    for r in recs:
        uniq[tuple(r["key"]) + (bool(r.get("smoke")),)] = r
    return sorted(uniq.values(), key=lambda r: (r["seed"], r.get("sub", "")))


def protocol_status() -> dict:
    path = os.path.join(ROOT, PROTOCOL_DOC)
    if not os.path.exists(path):
        return {"doc": PROTOCOL_DOC, "exists": False, "frozen": False, "sha256": None}
    raw = open(path, "rb").read()
    txt = raw.decode("utf-8", "replace")
    line = next((ln.strip() for ln in txt.splitlines() if ln.strip().startswith("FROZEN:")), "FROZEN: ?")
    return {"doc": PROTOCOL_DOC, "exists": True, "frozen": line.lower().startswith("frozen: yes"), "line": line,
            "sha256": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


def gt_episode(rec) -> dict:
    return {"seed": rec["seed"], "n_cells": rec["n_cells"], "gt_static": rec["gt_static"],
            "gt_labels": [dict(L, delta=dec(L["delta"])) for L in rec["gt_labels"]]}


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _keys(d):
    """tuple keys -> "a|b|c" (JSON)."""
    if isinstance(d, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k): _keys(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_keys(v) for v in d]
    return d


# ---------------------------------------------------------------------------------------------- pieces
def k1_support(data: CU.UnitData, n_episodes: int) -> dict:
    rows = data.rows_of("sleep")
    n_rej = int((data.mode[rows] == CU.MODES.index("reject")).sum())
    ok = len(rows) >= K1_RULE["min_sleep_units"] and n_rej >= K1_RULE["min_sleep_rejects"]
    return {"sleep_units": int(len(rows)), "sleep_rejects": n_rej, "pass": bool(ok), "rule": K1_RULE,
            "episodes": n_episodes, "extension_allowed": (not ok) and n_episodes < K1_RULE["extension_episodes"],
            "note": "counted on the CRT unit table (units whose H / pre windows fit in the episode)"}


def g_premise(table: dict) -> dict:
    c = next(c for c in table["cells"] if (c["family"], c["relation"], c["kpi"]) == ("sleep", "nbr", "pv"))
    return {"pass": c["status"] == "TRUE" and c["sign"] == 1, "cell": c,
            "rule": "GT (dir) shows sleep -> nbr pv TRUE(+)"}


def crt_summary(run: dict) -> dict:
    return {"n_declared": run["n_declared"], "n_tested": run["n_tested"], "power_floor": run["power_floor"],
            "max_replica_err": run["max_replica_err"], "max_joint_err": run["max_joint_err"],
            "declared": [(r["family"], r["relation"], r["kpi"], r["sign"], r["p_mscr"]) for r in run["results"]
                         if r["status"] == "declared"],
            "smallest_p": sorted((float(r["p_mscr"]), r["family"], r["relation"], r["kpi"]) for r in run["results"]
                                 if r["status"] != "undetermined")[:10],
            "undetermined": sorted({(r["family"], r["reason"]) for r in run["results"]
                                    if r["status"] == "undetermined"})}


def verdict(k0, g, k1, p1, p2, mode, frozen) -> dict:
    if k0 is None or not k0["pass"]:
        v = "INVALID"
    elif g is not None and not g["pass"]:
        v = "NO-CHAIN"
    elif not k1["pass"]:
        v = "UNDERPOWERED"
    elif p1["pass"] and p2["pass"]:
        v = "PASS"
    elif p1["pass"]:
        v = "PARTIAL"
    else:
        v = "KILL"
    label = v
    if mode != "full":
        label = f"NOT A VERDICT ({mode}; would-be: {v})"
    elif not frozen:
        label = f"{v} (PROVISIONAL: protocol not frozen)"
    return {"verdict": v, "label": label, "precedence": "INVALID > NO-CHAIN > UNDERPOWERED > PASS/PARTIAL/KILL",
            "consequence": {"PASS": "step 2 may use the MSCR map",
                            "PARTIAL": "step 3 must include the SHAP-map arm",
                            "KILL": "step 2 uses physics only, as a declared privilege",
                            "INVALID": "the CRT is not calibrated on the placebo: no discovery claim",
                            "UNDERPOWERED": "one extension to 120 EVAL episodes, then UNDERPOWERED stands",
                            "NO-CHAIN": "the indirect chain is absent in the GT: stop"}[v]}


# ---------------------------------------------------------------------------------------------- analyze
def analyze(a) -> dict:
    t_all = time.time()
    timing = {}
    allow = a.get("allow_smoke", False)
    R = {s: read_records(a.get(s), s, allow) for s in ("dev", "eval", "gt", "placebo", "prof")}
    if not R["dev"]:
        raise SystemExit("--dev is required (baseline thresholds are tuned on DEV)")
    if not R["placebo"]:
        raise SystemExit("--placebo is required (K0)")
    mode = "full" if (R["eval"] and R["gt"]) else ("dry-run" if not R["eval"] and not R["gt"] else "partial")
    smoke = any(r.get("smoke") for rs in R.values() for r in rs)
    if smoke and mode == "full":
        mode = "smoke"
    ps = protocol_status()
    cfg = CU.UnitCRTConfig(B=int(a.get("B", 9999)), audit_B=int(a.get("audit_B", 999)))
    if mode != "full":                     # pipeline exercise on a handful of episodes (never a verdict)
        cfg = dataclasses.replace(cfg, min_episodes=1)
    rep = {"mode": mode, "banner": "FULL ANALYSIS" if mode == "full" else "PIPELINE DRY RUN -- NOT A VERDICT",
           "protocol": ps, "config": dataclasses.asdict(cfg),
           "episodes": {s: len(v) for s, v in R.items()}, "smoke": smoke}
    log(f"== {rep['banner']} (mode {mode}); episodes {rep['episodes']}; protocol {ps.get('line')}")

    # K0
    t = time.time()
    pdat = CU.build_unit_data(R["placebo"])
    k0 = CU.placebo_rejection_units(pdat, cfg, **K0_RULE)
    k0.pop("run", None)
    timing["k0_placebo"] = round(time.time() - t, 1)
    rep["K0"] = k0
    log(f"K0 placebo: units {pdat.n}, tested {k0['n_tested']}/{k0['n_hypotheses']}, rate {k0['rate_mscr']:.3f} "
        f"(signed {k0['rate_signed']:.3f}), BY {k0['n_by_declared']} -> {'PASS' if k0['pass'] else 'FAIL'} "
        f"[{timing['k0_placebo']} s]")

    # DEV / EVAL tables
    dev = CU.build_unit_data(R["dev"])
    if R["eval"]:
        ev_recs = R["eval"]
        ev = CU.build_unit_data(ev_recs)
        folds = {f"fold{k}": ev.subset(ev.fold == k) for k in range(3)}
    else:
        ev_recs = R["dev"]
        ev = dev
        folds = {f"fold{k}": ev.subset(ev.episode % 3 == k) for k in range(3)}
        rep["pseudo_folds"] = "DEV episode index % 3 (dry run)"
    rep["units"] = {"dev": dev.meta["n_units"], "eval": ev.meta["n_units"], "dropped_eval": ev.meta["dropped"],
                    "p_mismatch_eval": ev.meta["p_mismatch"],
                    "by_family_eval": {f: int(len(ev.rows_of(f))) for f in CU.FAMILIES}}
    k1 = k1_support(ev, len(ev_recs))
    rep["K1"] = k1
    log(f"K1 support: sleep units {k1['sleep_units']} rejects {k1['sleep_rejects']} -> "
        f"{'PASS' if k1['pass'] else 'FAIL'}; units {rep['units']['by_family_eval']}")

    # MSCR-CRT
    t = time.time()
    crt = {"pooled": CU.run_crt_units(ev, cfg, CU.SPLIT_POOLED, audit=not a.get("no_audit", False))}
    timing["crt_pooled"] = round(time.time() - t, 1)
    for k, f in enumerate(FOLDS):
        t = time.time()
        crt[f] = CU.run_crt_units(folds[f], cfg, 1 + k, audit=False)
        timing[f"crt_{f}"] = round(time.time() - t, 1)
    rep["crt"] = {s: crt_summary(r) for s, r in crt.items()}
    if "audit" in crt["pooled"]:
        au = crt["pooled"]["audit"]
        rep["crt"]["audit"] = {x: au[x] for x in ("B", "n_declared", "n_tested", "rate_p_le_alpha", "power_floor",
                                                    "smallest", "declared", "wording")}
    log(f"MSCR-CRT pooled: declared {crt['pooled']['n_declared']}/{crt['pooled']['n_tested']} "
        f"(floor ok {crt['pooled']['power_floor']['rank1_ok']}) [{timing['crt_pooled']} s]; folds "
        f"{[crt[f]['n_declared'] for f in FOLDS]}")
    for d in rep["crt"]["pooled"]["declared"]:
        log(f"   declared {d}")

    # baselines
    t = time.time()
    methods = tuple(a["methods"].split(",")) if a.get("methods") else BD.TUNED
    splits = {"pooled": ev, **folds}
    sup = BD.Support() if mode == "full" else BD.Support(min_episodes=1)
    bl = BD.run_baselines(dev, splits, methods, sup=sup, log=log, ann=not a.get("no_ann", False))
    timing["baselines"] = round(time.time() - t, 1)

    # rowperm
    rp = None
    if not a.get("no_rowperm", False):
        t = time.time()
        try:
            rp = BD.mscr_rowperm(ev_recs, n_perm=a.get("rowperm_perm"))
        except Exception as e:                                        # noqa: BLE001  (reported, not fatal)
            rp = {"error": repr(e)}
        timing["rowperm"] = round(time.time() - t, 1)
        log(f"MSCR-rowperm: {'error ' + rp['error'] if 'error' in rp else str(rp['n_mapped']) + ' mapped'} "
            f"[{timing['rowperm']} s]")

    # PACIFISTA native
    if R["prof"]:
        rep["pacifista_native"] = BD.pacifista_native(R["prof"])
        log(f"PACIFISTA native sigma {rep['pacifista_native']['sigma']}")

    # reference
    g = None
    if R["gt"]:
        gs = G.summarize([gt_episode(r) for r in R["gt"]])
        table = gs["tables"][G.PRIMARY_ORIENT]
        ref = ES.gt_reference(table)
        g = g_premise(table)
        rep["gt"] = {"n_labels": gs["n_labels"], "labels_per_family": gs["labels_per_family"],
                     "counts": table["counts"], "delta": table["delta"], "cells": table["cells"],
                     "receiving": ES.gt_receiving_summary(gs["receiving"]), "G": g,
                     "act_counts": gs["tables"]["act"]["counts"]}
        log(f"GT ({table['orient']}): {table['counts']}; G premise {'PASS' if g['pass'] else 'FAIL'}; receiving "
            f"{rep['gt']['receiving']}")
        rep["reference"] = "knockout GT (dir)"
    else:
        ref = ES.physics_reference()
        rep["reference"] = "PHYSICS PROXY (edge_score.PHYSICS_PRIOR) -- dry run only, not ground truth"

    # scoring
    decl = {"mscr_crt": {s: CU.edges_from_crt(r) for s, r in crt.items()}}
    if rp is not None and "declared" in rp:
        decl["mscr_rowperm"] = {"pooled": rp["declared"]}
    for m, r in bl.items():
        decl[m] = {s: v["declared"] for s, v in r["splits"].items()}
    scores = {m: {s: ES.score_method(d, ref) for s, d in dd.items()} for m, dd in decl.items()}
    sens = {m: {s: ES.score_method(v["declared_physics_tau"], ref) for s, v in r["splits"].items()}
            for m, r in bl.items() if m in BD.TUNED}
    loc = CU.localise_receivers(ev)
    sleep_nbr_load = decl["mscr_crt"]["pooled"][("sleep", "nbr", "load")]["declared"]
    gt_static = {int(r["seed"]): r.get("gt_static") or {} for r in ev_recs}       # privileged: scorer only
    top1 = ES.top1_accuracy(loc, gt_static)
    top1["gated_by_declared_sleep_nbr_load"] = bool(sleep_nbr_load)
    if not sleep_nbr_load:
        top1 = dict(top1, acc_effective=0.0, note="MSCR-CRT did not declare sleep -> nbr load: no localisation claim")
    rep["top1_receiving"] = {"mscr_crt": top1, "others": "not supported (no per-cell localisation output)"}
    p1 = ES.p1_check(scores["mscr_crt"]["pooled"])
    mscr_f1 = {s: scores["mscr_crt"][s]["indirect"]["f1"] for s in ("pooled", *FOLDS)}
    base_f1 = {m: {s: scores[m][s]["indirect"]["f1"] for s in ("pooled", *FOLDS)} for m in BD.TUNED if m in scores}
    p2 = ES.p2_check(mscr_f1, base_f1, FOLDS)
    rep["scores"] = scores
    rep["scores_physics_tau_sensitivity"] = sens
    rep["baselines"] = {m: {"tau": r["tau"], "tau_physics": r.get("tau_physics"), "note": r.get("note"),
                            "decl_relations": r.get("decl_relations"), "cpu_s": r.get("cpu_s"),
                            "wall_s": r.get("wall_s")}
                        for m, r in bl.items()}
    rep["declared"] = {m: {s: sorted(("|".join(h), v["sign"]) for h, v in d.items() if v.get("declared"))
                           for s, d in dd.items()} for m, dd in decl.items()}
    if rp is not None:
        rep["rowperm"] = {k: v for k, v in rp.items() if k != "declared"}
    rep["P1"], rep["P2"] = p1, p2
    rep["verdict"] = verdict(k0, g, k1, p1, p2, mode, ps["frozen"])
    timing["total"] = round(time.time() - t_all, 1)
    rep["timing_s"] = timing

    log("")
    log(f"{'method':>14} {'split':>7} {'ind P':>6} {'ind R':>6} {'ind F1':>6} {'F1':>6} {'sign':>6} {'far':>4} "
        f"{'#dec':>5}")
    for m, ss in scores.items():
        for s, v in ss.items():
            ind, ov = v["indirect"], v["overall"]
            log(f"{m:>14} {s:>7} {ind['precision']:6.2f} {ind['recall']:6.2f} {ind['f1']:6.2f} {ov['f1']:6.2f} "
                f"{ov['sign_acc']:6.2f} {v['far_declared']:4d} {v['n_declared']:5d}")
    log(f"P1 {'PASS' if p1['pass'] else 'FAIL'} {p1['parts']}")
    log(f"P2 {'PASS' if p2['pass'] else 'FAIL'} (folds ok {p2['folds_ok']}); pooled MSCR "
        f"{p2['per_split']['pooled']['mscr']} vs best {p2['per_split']['pooled']['best_baseline']} "
        f"({p2['per_split']['pooled']['best_method']})")
    log(f"top-1 receiving (MSCR-CRT): {top1}")
    log(f"VERDICT: {rep['verdict']['label']}")
    log(f"timing (s): {timing}")
    if a.get("json"):
        with open(a["json"], "w", newline="\n") as fh:
            json.dump(_keys(rep), fh, indent=1, default=_js)
        log(f"wrote {a['json']}")
    return rep


def cli(argv):
    if not argv or argv[0] != "analyze":
        raise SystemExit(__doc__)
    it = iter(argv[1:])
    a = {}
    flags = {"--allow-smoke": "allow_smoke", "--no-audit": "no_audit", "--no-rowperm": "no_rowperm",
             "--no-ann": "no_ann"}
    vals = {"--dev": "dev", "--eval": "eval", "--gt": "gt", "--placebo": "placebo", "--prof": "prof",
            "--json": "json", "--B": "B", "--audit-B": "audit_B", "--rowperm-perm": "rowperm_perm",
            "--methods": "methods"}
    for x in it:
        if x in flags:
            a[flags[x]] = True
        elif x in vals:
            a[vals[x]] = next(it)
        else:
            raise SystemExit(f"unknown argument {x}\n{__doc__}")
    for k in ("B", "audit_B", "rowperm_perm"):
        if k in a:
            a[k] = int(a[k])
    return analyze(a)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])

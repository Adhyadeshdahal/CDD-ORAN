"""Study A EVAL report generator: merged campaign records -> REPORT.md + report.json + figure-ready CSVs, with every
pre-registered output of PROTOCOL_A sections 8-12 (claim and C1 / C2a / C2b / C3 with the R-56 sensitivity lines,
the unfiltered eq-arm table and the pre-registered named-arm sentence, V0 headline, power among cells not INVALID
with the R4 "not applicable" rows, E4, kappa sweep, cost, integrity). All numbers come from eval_analysis.py (same
definitions, nothing tuned); this module only drives it arm by arm and lays the outputs out.

  EVAL: uv run python scratchpad/xmethod/eval_report.py --spec scratchpad/xmethod/specs/eval/full.json \\
            --merged <EVAL merged files> --freeze-commit <sha> [--amendments A] [--dev-merged <DEV files>] \\
            --out scratchpad/xmethod/results/eval
  DEV:  uv run python scratchpad/xmethod/eval_report.py --dev --dev-spec <specs/dev/full.json> <specs/dev/ci_c.json> \\
            --merged <results/dev/full/merged.jsonl.gz> <results/dev/ci_c/merged.jsonl.gz> \\
            --out scratchpad/xmethod/results/eval_dev

Memory: records are split by arm into a temporary directory in one streaming pass, and each arm is screened and
summarised on its own (eval_analysis.build_cells is per arm); the verdicts, tables and V11 run once on the merged
cells. V11 = per-arm integrity (candidates, BY re-check, tune reproducibility) + the cross-arm checks (dataset hash,
one platform per dataset, packages, missing / unexpected keys) on slim records. The result equals
eval_analysis.analyse on the same records (tests/test_xmethod_eval_report.py).

DEV mode (development only, watermarked everywhere): arms and blocks of the DEV specs, analysis fields (labels, set_D,
c2b, native partners, c1_sensitivity_drop, fixed_threshold, analysis block) from the EVAL spec. If the EVAL focal
arm has no DEV arm, the PROTOCOL_A T10 fallback is applied FOR THIS RENDERING ONLY (pmrt_eq focal, primary). DEV
output is PROVISIONAL by construction (no frozen protocol stamps) and is never an EVAL result.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_analysis as E  # noqa: E402

REPORT_VERSION = "xm-eval-report/1"
DEV_MARK = ("DEV DATA, NOT EVAL: development seeds 3_000_000-3_000_199 on the DEV specs; a rehearsal of the report, "
            "not the pre-registered result")
SLIM_DROP = ("edges", "notes", "scores", "versions", "result_config", "config")


# ================================================================================================ inputs
def iter_lines(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield line


def arm_of(r: dict) -> str:
    return r.get("arm") or str(r.get("key", "")).split("|")[0]


def split_by_arm(paths: list[str], tmp: str, tag: str) -> tuple[dict[str, str], dict[str, int]]:
    """One streaming pass: every record line to <tmp>/<tag>_<arm>.jsonl (unparseable lines are skipped, as
    eval_analysis.load_records does)."""
    files, handles, n = {}, {}, defaultdict(int)
    try:
        for p in paths:
            for line in iter_lines(p):
                try:
                    a = arm_of(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if a not in handles:
                    files[a] = os.path.join(tmp, f"{tag}_{hashlib.sha1(a.encode()).hexdigest()[:12]}.jsonl")
                    handles[a] = open(files[a], "w", encoding="utf-8", newline="\n")
                handles[a].write(line if line.endswith("\n") else line + "\n")
                n[a] += 1
    finally:
        for h in handles.values():
            h.close()
    return files, dict(n)


def load_sorted(path: str | None) -> list[dict]:
    """An arm's records in eval_analysis._load_many order."""
    return [] if path is None else sorted(E.load_records(path),
                                          key=lambda r: (r.get("key", ""), json.dumps(r, sort_keys=True)))


_SHARED: dict[str, object] = {}


def slim(r: dict) -> dict:
    """Record without edges / notes / scores (V10 and the cross-arm V11 checks); repeated sub-dicts shared."""
    out = {}
    for k, v in r.items():
        if k in SLIM_DROP:
            continue
        if k == "integrity" and isinstance(v, dict):                     # V11 stamps read only the spec file sha
            v = {"spec_file_sha256": v.get("spec_file_sha256")}
        if isinstance(v, dict) and k in ("code", "run_mode", "integrity", "pkgs", "host"):
            v = _SHARED.setdefault(f"{k}:{json.dumps(v, sort_keys=True)}", v)
        out[k] = v
    return out


# ================================================================================================ specs
def sub_spec(spec: dict, arm: str) -> dict:
    """The spec restricted to one arm (focal and analysis fields kept, so blocks and labels match)."""
    blocks = []
    for b in spec["blocks"]:
        sel = b.get("arms", "all")
        if sel == "all" or arm in sel:
            blocks.append({**b, "arms": [arm]})
    return {**spec, "arms": {arm: spec["arms"][arm]}, "blocks": blocks}


def dev_spec(eval_spec: dict, dev_specs: list[dict]) -> tuple[dict, list[str]]:
    """DEV rendering spec: DEV arms and blocks, EVAL analysis fields (section docstring)."""
    notes, arms, blocks = [], {}, []
    for s in dev_specs:
        for a, d in s["arms"].items():
            if a not in eval_spec["arms"]:
                notes.append(f"DEV arm {a} is not in the EVAL spec: left out")
                continue
            ev = eval_spec["arms"][a]
            if (d.get("ref"), d.get("config")) != (ev.get("ref"), ev.get("config")):
                notes.append(f"DEV arm {a}: ref / config differ from the EVAL spec (DEV {d.get('ref')} "
                             f"{d.get('config')}); EVAL analysis fields used")
            arms[a] = {**ev, **{k: d[k] for k in ("ref", "config", "declare", "worlds") if k in d}}
        for b in s["blocks"]:
            sel = b.get("arms", "all")
            names = [a for a in (list(s["arms"]) if sel == "all" else sel) if a in arms]
            if names:
                blocks.append({**b, "arms": names})
    focal = eval_spec.get("focal")
    out = {"name": "DEV:" + "+".join(str(s.get("name")) for s in dev_specs), "arms": arms, "blocks": blocks,
           "budget_cpu_s": eval_spec.get("budget_cpu_s"), "fmax_dependence_sha256": eval_spec.get("fmax_dependence_sha256"),
           "protocol_sha256": eval_spec.get("protocol_sha256"), "focal": focal,
           "not_in_grid": (eval_spec.get("not_in_grid") or []) + [   # last: gaps of the DEV runs' own scope
               {"ruling": "DEV scope", "arms": list(arms), "regimes": None, "ns": None, "kappas": None,
                "reason": "the DEV specs ran a subset of the EVAL grid (DEV rendering only)"}]}
    if focal not in arms:
        if "pmrt_eq" not in arms:
            raise SystemExit(f"DEV specs have neither the EVAL focal arm {focal} nor pmrt_eq")
        out["focal"] = "pmrt_eq"
        arms["pmrt_eq"] = {k: v for k, v in arms["pmrt_eq"].items() if k != "label"} | {"analysis": "primary"}
        notes.append(f"{focal} has no DEV records: PROTOCOL_A T10 fallback applied for this DEV rendering only "
                     "(pmrt_eq focal and primary, label dropped)")
    missing = sorted(set(eval_spec["arms"]) - set(arms))
    if missing:
        notes.append("EVAL arms without DEV records (absent from this rendering): " + ", ".join(missing))
    return out, notes


# ================================================================================================ chunked analysis
def _merge_integrity(per_arm: list[dict], glob: dict) -> dict:
    """V11: cross-arm checks from ``glob`` (eval_analysis.integrity on slim records); the record-content checks
    (candidate completeness, BY re-check, tune reproducibility) summed over the per-arm runs."""
    v = dict(glob)
    inc = [k for x in per_arm for k in x["candidates_incomplete"]]
    v["n_candidates_incomplete"] = sum(x["n_candidates_incomplete"] for x in per_arm)
    v["candidates_incomplete"] = inc[:20]
    bad = [k for x in per_arm for k in x["by_recheck"]["disagree_first"]]
    v["by_recheck"] = {"records": sum(x["by_recheck"]["records"] for x in per_arm),
                       "agree": sum(x["by_recheck"]["agree"] for x in per_arm), "disagree_first": bad[:20]}
    reps = [x["tune_reproducibility_vs_dev"] for x in per_arm if x["tune_reproducibility_vs_dev"] is not None]
    v["tune_reproducibility_vs_dev"] = ({k: sum(r[k] for r in reps) for k in ("compared", "dataset_sha_equal",
                                                                             "declarations_equal")} if reps else None)
    checks = dict(v["checks"])
    checks["candidates_complete"] = v["n_candidates_incomplete"] == 0
    v["checks"] = checks
    v["failed"] = sorted(k for k, ok in checks.items() if not ok)
    v["label"] = "FINAL" if all(checks.values()) else "PROVISIONAL"
    return v


def chunked_analyse(paths: list[str], spec: dict, freeze_commit: str | None = None, *, dep: dict | None = None,
                    spec_sha: str | None = None, protocol: dict | None = None, amendments: dict | None = None,
                    dev_paths: list[str] | None = None, tmp_dir: str | None = None, log=print) -> tuple[dict, dict]:
    """eval_analysis.analyse, arm by arm. Returns (out, input record counts by arm)."""
    tmp = tempfile.mkdtemp(prefix="xm_eval_report_", dir=tmp_dir)
    try:
        files, n_in = split_by_arm(paths, tmp, "rec")
        dev_files = split_by_arm(dev_paths, tmp, "dev")[0] if dev_paths else {}
        cells, rows, slim_use, per_integ = {}, {}, [], []
        scr = {"expected": {}, "expected_error": None, "duplicates": [], "unexpected": [], "role_mismatch": []}
        for a in sorted(set(files) - set(spec["arms"])):                     # records of arms not in the spec
            scr["unexpected"] += [r.get("key") for r in load_sorted(files[a])]
        for a in sorted(spec["arms"]):
            t0 = time.time()
            ss = sub_spec(spec, a)
            recs = load_sorted(files.get(a))
            use, s = E.screen(recs, ss)
            c, r = E.build_cells(use, ss)
            cells.update(c)
            rows.update(r)
            per_integ.append(E.integrity(use, s, ss, freeze_commit, spec_sha=spec_sha, protocol=protocol,
                                         amendments=amendments, dep=dep,
                                         dev_records=load_sorted(dev_files.get(a)) if dev_paths else None))
            if s["expected"] is None:
                scr["expected"] = None
                scr["expected_error"] = s["expected_error"]
            elif scr["expected"] is not None:
                scr["expected"].update(s["expected"])
            for k in ("duplicates", "unexpected", "role_mismatch"):
                scr[k] += s[k]
            slim_use += [slim(x) for x in use]
            log(f"[report] {a}: {len(recs)} records, {len(use)} used, {len(c)} cells ({time.time() - t0:.0f} s)")
            del recs, use
        for k in ("duplicates", "unexpected", "role_mismatch"):
            scr[k] = sorted(scr[k])
        glob = E.integrity(slim_use, scr, spec, freeze_commit, spec_sha=spec_sha, protocol=protocol,
                           amendments=amendments, dep=dep)
        v11 = _merge_integrity(per_integ, glob)
        return E.assemble(cells, rows, slim_use, spec, dep, v11), n_in
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ================================================================================================ CSVs
def _rate_cols(prefix: str, x: dict | None) -> dict:
    x = x or {}
    ci = x.get("ci") or [None, None]
    return {f"{prefix}_rate": x.get("rate"), f"{prefix}_lo": ci[0], f"{prefix}_hi": ci[1],
            f"{prefix}_validity": x.get("validity")}


def _mean_cols(prefix: str, x: dict | None) -> dict:
    x = x or {}
    ci = x.get("ci") or [None, None]
    return {f"{prefix}_mean": x.get("mean"), f"{prefix}_sd": x.get("sd"), f"{prefix}_lo": ci[0], f"{prefix}_hi": ci[1]}


CELL = ("arm", "world", "regime", "lam", "n", "kappa")


def csv_tables(out: dict) -> dict[str, list[dict]]:
    """Figure-ready long tables (one row per arm x cell [x rule])."""
    t: dict[str, list[dict]] = {}
    t["validity_cells"] = [{**{f: r.get(f) for f in (*CELL, "block", "declare", "status", "validity", "n_seeds")},
                            **{k: v for rk in E.RATE_KEYS for k, v in _rate_cols(rk, r.get(rk)).items()}}
                           for r in out["V1_validity"]]
    t["recall_vs_n"] = [{**{f: r.get(f) for f in (*CELL[:5], "block", "state", "power_not_applicable")},
                         "powered": r["recall"] is not None, **_mean_cols("recall", r["recall"]),
                         **_mean_cols("fdp", r["fdp"]), **_rate_cols("sign_acc", r["sign_acc"])}
                        for r in out["V2_recall"]]
    t["like_for_like"] = [{**{f: r.get(f) for f in (*CELL, "rule", "declare", "state", "n_seeds",
                                                     "power_not_applicable", "placebo_is_tuning_column")},
                           **_mean_cols("recall", r["recall"]), **_rate_cols("null", r["null"]),
                           **_rate_cols("placebo", r["placebo"]), **_rate_cols("placebo_conf", r["placebo_conf"])}
                          for r in out["V0_like_for_like"]]
    t["paired_diff"] = [{**{f: r.get(f) for f in (*CELL, "rule", "block", "descriptive_only", "n_seeds", "mean_d",
                                                   "sd_d")}, "lo": r["ci"][0], "hi": r["ci"][1]}
                        for r in out["V3_paired"]]
    t["information_levels_R2"] = [{**{f: r.get(f) for f in (*CELL, "block", "status", "validity")},
                                   **_mean_cols("recall", r.get("recall")), **_rate_cols("null_raw", r.get("null_raw")),
                                   **_rate_cols("null_decl", r.get("null_decl"))} for r in out["V5_information_R2"]]
    t["e4_by_lambda"] = [{**{f: r.get(f) for f in (*CELL, "block", "status", "validity", "power_not_applicable",
                                                    "wrong_sign_rate")},
                          **_mean_cols("recall", r.get("recall")), **_rate_cols("plac_raw", r.get("plac_raw")),
                          **_rate_cols("conf_raw", r.get("conf_raw")), **_rate_cols("conf_decl", r.get("conf_decl"))}
                         for r in out["V6_E4"]]
    t["kappa_sweep"] = [{**{f: r.get(f) for f in (*CELL, "status", "validity")}, **_mean_cols("recall", r["recall"])}
                        for r in out["V9_kappa_sweep"]]
    t["cost"] = [{f: r.get(f) for f in ("arm", "world", "regime", "n", "budget", "n_runs", "cpu_s_mean", "cpu_s_max",
                                        "wall_s_mean", "wall_s_max", "peak_rss_mb_max", "gpu_s_mean", "n_infeasible",
                                        "n_errors", "t3")} for r in out["V10_cost"]]
    t["not_in_grid"] = [{f: c.get(f) for f in (*CELL, "ruling")} for c in (out.get("not_in_grid") or {}).get("cells", [])]
    v4 = out["V4_verdicts"]
    t["eq_arms"] = [{"arm": a, **{k: x.get(k) for k in ("membership", "native", "native_c1", "verdict", "n_cells",
                                                        "n_invalid", "f_max", "r2_invalid", "r2_planned")},
                     **_rate_cols("r1_null_raw", (x.get("pooled_r1") or {}).get("null_raw")),
                     **_rate_cols("r2_null_raw", (x.get("pooled_r2") or {}).get("null_raw"))}
                    for a, x in v4.get("eq_arms_table", {}).items()]
    return t


def write_csvs(tables: dict[str, list[dict]], d: str, mode: str) -> dict[str, str]:
    """One CSV per table; the first column ``data_mode`` ('DEV-NOT-EVAL' or 'EVAL') watermarks every row."""
    os.makedirs(d, exist_ok=True)
    shas = {}
    tagv = "DEV-NOT-EVAL" if mode == "DEV" else "EVAL"
    for name, rows in tables.items():
        p = os.path.join(d, f"{name}.csv")
        cols = ["data_mode", *dict.fromkeys(k for r in rows for k in r)]
        with open(p, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({"data_mode": tagv, **{k: ("" if v is None else v) for k, v in r.items()}})
        shas[f"csv/{name}.csv"] = E.file_sha256(p)
    return shas


# ================================================================================================ REPORT.md
def _rt(x: dict | None) -> str:
    if not x or x.get("rate") is None:
        return "-"
    ci = x.get("ci") or [None, None]
    return f"{E._f(x['rate'], 3)} [{E._f(ci[0], 3)}, {E._f(ci[1], 3)}] {x.get('validity') or ''}".strip()


def report_md(out: dict, meta: dict) -> str:
    dev = meta["mode"] == "DEV"
    mark = [f"> **{DEV_MARK}**", ""] if dev else []
    v4, v11 = out["V4_verdicts"], out["V11_integrity"]
    lab = v4.get("arm_labels", {})

    def tag(a: str) -> str:
        return f" [{lab[a]}]" if a in lab else ""
    title = "# [DEV, NOT EVAL] Study A report rehearsal" if dev else "# Study A EVAL report"
    pr = v11["protocol"]
    ptxt = ("frozen, sha matches" if pr.get("ok") else
            f"NOT verified: frozen {pr.get('frozen')}, spec protocol_sha256 {str(pr.get('spec_protocol_sha256'))[:16]}"
            f" vs file {str(pr.get('file_sha256'))[:16]}" + (f" ({pr['why']})" if pr.get("why") else ""))
    L = [title, "", *mark,
         f"Generator `{REPORT_VERSION}` on `{out['analysis']}`; spec `{out['spec_name']}` (sha256 "
         f"{(meta['spec_sha256'] or '-')[:16]}); protocol docs/xmethod/PROTOCOL_A.md "
         f"({ptxt}); "
         f"integrity **{v11['label']}**" + (f" (failed: {', '.join(v11['failed'])})" if v11["failed"] else "") + ".",
         f"Primary PMRT arm (focal): {out['focal']}; primary kappa {out['primary_kappa']}; generated {meta['utc']}.", "",
         "Inputs:", ""]
    L += [f"- `{p}`: sha256 {s[:16]}" for p, s in meta["inputs"]]
    L += [f"- records by arm: {', '.join(f'{a} {n}' for a, n in sorted(meta['records_by_arm'].items()))}"]
    L += [f"- note: {x}" for x in meta.get("notes", [])]
    # 1. claim
    comp = v4["components"]
    L += ["", "## 1. Claim (PROTOCOL_A s.10)", "", *mark, f"**Claim: {v4['claim']}**", "",
          "| component | verdict | wording |", "|---|---|---|",
          f"| C1 design-blind tests invalid on R2 | {comp['C1']} | {v4['C1']['wording']} |",
          f"| C2a design-based test valid (R1 + R2) | {comp['C2a']} | {v4['C2_wording']} |",
          f"| C2b design covariates restore validity | {comp['C2b']} | (C2 wording above) |",
          f"| C3 valid under the logged confounded policy | {comp['C3']} | {v4['C3'].get('wording', '-')} |"]
    # 2. C1
    c1 = v4["C1"]
    L += ["", "## 2. C1: design-blind arms (set D)", "", *mark,
          f"**{c1['verdict']}**: {c1['n_failures']} of {len(c1['D_counted'])} assessable arms are FAILURES "
          f"(|D| = {len(c1['D'])}; >= 3 assessable needed).", "",
          "| arm | in D | verdict | R2 INVALID / planned (counted) | pooled R2 truth-null raw | pooled R2 placebo raw | "
          "R1 INVALID / counted (F_max) |", "|---|---|---|---|---|---|---|"]
    for a, x in [*((a, x) for a, x in c1["arms"].items()), *((a, x) for a, x in c1.get("descriptive", {}).items())]:
        L.append(f"| {a}{tag(a)} | {'yes' if a in c1['arms'] else 'no (descriptive)'} | {x['verdict']} | "
                 f"{x['r2_invalid']}/{x['r2_planned']} ({x['r2_counted']}) | {_rt(x['pooled_r2'].get('null_raw'))} | "
                 f"{_rt(x['pooled_r2'].get('plac_raw'))} | {x['r1_invalid']}/{x['r1_counted']} ({x['r1_f_max']}) |")
    sw = c1.get("sensitivity_without")
    L += ["", "C1 sensitivity (R-56, no effect): " + (
        f"without {', '.join(sw['dropped'])}: **{sw['verdict']}** ({sw['n_failures']} of {len(sw['D_counted'])} "
        "assessable arms fail)." if sw else "no arm flagged `c1_sensitivity_drop` in this spec.")]

    def comp_row(name: str, x: dict) -> str:
        return (f"| {name} | {x['verdict']} | {x.get('n_cells', '-')}/{x.get('n_planned', '-')} | "
                f"{x.get('n_invalid', '-')} ({x.get('f_max', '-')}) | "
                + "; ".join(f"{k} {_rt(p)}" for k, p in (x.get("pooled") or {}).items()) + " |")
    head = ["| arm | verdict | cells counted / planned | INVALID (F_max) | pooled rates |", "|---|---|---|---|---|"]
    # 3. C2a
    L += ["", "## 3. C2a: the design-based test (R1 + R2)", "", *mark, *head,
          comp_row(f"{out['focal']} (primary)", v4["C2a"])]
    L += [comp_row(f"{a}{tag(a)} (secondary PMRT arm, no effect)", x["C2a"]) for a, x in v4["pmrt_secondary"].items()]
    # 4. C2b
    c2b = v4["C2b"]
    w2 = c2b.get("with_r1_failing_partners") or {}
    L += ["", "## 4. C2b: design-covariate adjustment (eq arms whose native partner is a C1 FAILURE)", "", *mark,
          f"**{c2b['verdict']}** (membership rule as frozen); with the eq arms whose native partner is INVALID IN R1 "
          f"but meets C1 legs (i)-(ii) added (R-56 sensitivity, no effect): **{w2.get('verdict', '-')}** (added: "
          f"{', '.join(w2.get('added', [])) or 'none'}).", "", *head]
    L += [comp_row(f"{a}{tag(a)} (native {x['native']})", x) for a, x in c2b["arms"].items()]
    L += [comp_row(f"{a}{tag(a)} EXCLUDED (R-40), no effect", x) for a, x in c2b.get("excluded", {}).items()]
    L += [comp_row(f"{a} (R-56 sensitivity member)", x) for a, x in w2.get("arms", {}).items()]
    L += ["", f"C2 wording: {v4['C2_wording']}.", "", "Pre-registered disclosure (R-56; named eq arms):"]
    L += [f"- {x['sentence']}." for x in v4.get("C2_named", [])] or ["- none (no eq arm meets the naming rule)."]
    L += ["", "Every eq arm, same rule on R1 + R2, unfiltered (R-56):", "",
          "| eq arm | C2b membership | native C1 | verdict | INVALID / cells (F_max) | R2 INVALID / planned | "
          "pooled R1 truth-null raw | pooled R2 truth-null raw |", "|---|---|---|---|---|---|---|---|"]
    for a, x in v4.get("eq_arms_table", {}).items():
        L.append(f"| {a}{tag(a)} | {x['membership']} | {x['native_c1'] or '-'} | {x['verdict']} | "
                 f"{x['n_invalid']}/{x['n_cells']} ({x['f_max']}) | "
                 f"{'-' if x['r2_invalid'] is None else str(x['r2_invalid']) + '/' + str(x['r2_planned'])} | "
                 f"{_rt((x['pooled_r1'] or {}).get('null_raw'))} | {_rt((x['pooled_r2'] or {}).get('null_raw'))} |")
    # 5. C3
    L += ["", "## 5. C3: logged confounded policy (E4 R3; C3 readers at S_E4)", "", *mark, *head,
          comp_row(f"{out['focal']} (primary)", v4["C3"])]
    L += [comp_row(f"{a}{tag(a)} (eq arm, no claim effect)", x) for a, x in v4["C3_eq_arms"].items()]
    L += [comp_row(f"{a}{tag(a)} (secondary PMRT arm, no effect)", x["C3"]) for a, x in v4["pmrt_secondary"].items()]
    # 6. power, R4 NA
    na = [r for r in out["V2_recall"] if r["power_not_applicable"]]
    pw = [r for r in out["V2_recall"] if r["recall"] is not None]
    L += ["", "## 6. Power among cells not INVALID (V2) and the R4 'not applicable' rows", "", *mark,
          f"{len(pw)} (arm, cell) with recall (counted, not INVALID; R-39); the grid is in the appendix (V2) and "
          "`csv/recall_vs_n.csv`. PMRT arms in R4 (no known design): power NOT APPLICABLE, never recall 0 (R-42); "
          "their placebo rates are still read:", "", "| arm | cell | state | placebo raw (P_placebo) |", "|---|---|---|---|"]
    v1 = {r["key"]: r for r in out["V1_validity"]}
    L += [f"| {r['arm']} | {r['key'].split('|', 1)[1].replace('|', ' ')} | NA ({r['state']}) | {_rt((v1.get(r['key']) or {}).get('plac_raw'))} |"
          for r in na]
    # 7. kappa sweep
    L += ["", "## 7. kappa sweep (R2, n 1000; V9)", "", *mark, "| arm | world | kappa | status | validity | recall |",
          "|---|---|---|---|---|---|"]
    L += [f"| {r['arm']} | {r['world']} | {r['kappa']:g} | {r['status']} | {r['validity']} | "
          f"{E._f((r['recall'] or {}).get('mean'))} |" for r in sorted(out["V9_kappa_sweep"],
                                                                     key=lambda r: (r["arm"], r["world"], r["kappa"]))]
    # 8. not in grid
    ng = out.get("not_in_grid") or {}
    L += ["", "## 8. Not in grid (pre-registered grid choices, R-54 / R-58)", "", *mark,
          "Cells outside an arm's grid are not planned: they never count as missing, never cap a verdict at PARTIAL "
          "and never make a component NOT EVALUABLE; every table labels them `nig`.", "",
          "| ruling | arms | where | cells | reason |", "|---|---|---|---|---|"]
    L += [f"| {r['ruling']} | {', '.join(r['arms'])} | "
          f"{'; '.join(f'{k} {v}' for k, v in r['where'].items() if v) or 'all'} | {r['n_cells']} | {r['reason']} |"
          for r in ng.get("by_rule", [])]
    L += ["", f"{ng.get('n_cells', 0)} cells in total; unexplained gaps: "
          + (", ".join(ng["unexplained"][:10]) + (" ..." if len(ng["unexplained"]) > 10 else "")
             if ng.get("unexplained") else "none") + "."]
    # 9. files
    L += ["", "## 9. Figure-ready CSVs", "", *mark]
    L += [f"- `{p}` (sha256 {s[:16]})" for p, s in sorted(meta["csv"].items())]
    L += ["", "## Appendix: every eval_analysis table (V0-V11)", "", *mark]
    body = E.markdown(out).split("\n", 3)[3] if "\n" in E.markdown(out) else ""
    L += [body.replace("\n## ", "\n### ")]
    if dev:
        L += ["", f"> **{DEV_MARK}**"]
    return "\n".join(L).rstrip("\n") + "\n"


# ================================================================================================ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval_report")
    ap.add_argument("--spec", default=os.path.join(HERE, "specs", "eval", "full.json"), help="EVAL spec")
    ap.add_argument("--merged", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", action="store_true", help="DEV rehearsal (watermarked); needs --dev-spec")
    ap.add_argument("--dev-spec", nargs="+", default=None)
    ap.add_argument("--freeze-commit", default=None)
    ap.add_argument("--amendments", default=None)
    ap.add_argument("--protocol", default=E.PROTOCOL_PATH)
    ap.add_argument("--dependence", default=E.DEP_PATH)
    ap.add_argument("--dev-merged", nargs="+", default=None, help="EVAL: DEV records for the tune reproducibility check")
    ap.add_argument("--tmp-dir", default=None)
    a = ap.parse_args(argv)
    eval_spec = json.load(open(a.spec, encoding="utf-8"))
    notes: list[str] = []
    if a.dev:
        if not a.dev_spec:
            raise SystemExit("--dev needs --dev-spec")
        spec, notes = dev_spec(eval_spec, [json.load(open(p, encoding="utf-8")) for p in a.dev_spec])
        spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        if a.dev_merged:
            raise SystemExit("--dev-merged is the EVAL tune reproducibility check; not in --dev")
    else:
        if a.dev_spec:
            raise SystemExit("--dev-spec needs --dev")
        if any(int(x) < 3_100_000 for b in eval_spec["blocks"] if b["role"] == "measure"
               for x in (b["seeds"] if isinstance(b["seeds"], list) else [])):
            raise SystemExit("EVAL mode needs an EVAL spec (measure seeds >= 3_100_000); use --dev for DEV data")
        spec, spec_sha = eval_spec, E.file_sha256(a.spec)
    out, n_in = chunked_analyse(a.merged, spec, a.freeze_commit, dep=E.load_dependence(a.dependence),
                                spec_sha=spec_sha, protocol=E.protocol_check(spec, a.protocol),
                                amendments=E.load_amendments(a.amendments), dev_paths=a.dev_merged, tmp_dir=a.tmp_dir)
    mode = "DEV" if a.dev else "EVAL"
    os.makedirs(a.out, exist_ok=True)
    csv_sha = write_csvs(csv_tables(out), os.path.join(a.out, "csv"), mode)
    meta = {"mode": mode, "watermark": DEV_MARK if a.dev else None, "generator": REPORT_VERSION,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spec_sha256": spec_sha,
            "inputs": [(p.replace("\\", "/"), E.file_sha256(p)) for p in [a.spec, *(a.dev_spec or []), *a.merged]],
            "records_by_arm": n_in, "notes": notes, "csv": csv_sha}
    js = json.dumps(E._clean({"meta": meta, "spec": spec if a.dev else None, "tables": out}), indent=1,
                    sort_keys=True, allow_nan=False) + "\n"
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(js)
    with open(os.path.join(a.out, "REPORT.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report_md(out, meta))
    print(f"[report] {mode} {out['V11_integrity']['label']}: claim {out['V4_verdicts']['claim']} "
          f"{out['V4_verdicts']['components']} -> {a.out}")
    return 0 if (not a.dev and out["V11_integrity"]["label"] == "FINAL") else 2


if __name__ == "__main__":
    sys.exit(main())

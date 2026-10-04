"""DEV report (brief dev-runs step 5): REPORT.md (< 150 lines) from the PROTOCOL_A cells of the DEV run.

    uv run python scratchpad/xmethod/dev_report.py --dir results/dev/full

Inputs in --dir: dev_cells.json + t1.json (scratchpad/xmethod/dev_t1.py: eval_analysis.py definitions, R-29 tau,
R-30 validity, rule T1), agg.json (campaign aggregate: cost), merged.jsonl.summary.json (run integrity).
Descriptive only: nothing is tuned on these numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cdd_oran.xmethod.dev_power import null_rate_seeds  # noqa: E402

ARM_ORDER = ["pmrt_eq", "pmrt_r3", "pc_eq", "pc_native", "granger_eq", "granger_native", "corr", "notears",
             "shap_dag", "two_tower", "mscr_eq", "mscr_native", "mscr_eq_min", "pcorr_eq", "pcorr_native",
             "pcorr_eq_min", "pcorr_hac", "pcorr_hac_fb", "pcorr_hac_eq_min", "pcorr_hac_fb_eq_min", "pdcor_eq",
             "pdcor_native", "pdcor_eq_min", "rcot2_eq", "rcot2_native", "rcot2_eq_min", "cmi_knn_eq",
             "cmi_knn_native", "cmi_knn_eq_min"]
LETTER = {"VALID": "V", "INVALID": "I", "INCONCLUSIVE": "c", "NO_READ": "-"}
RATES = ("null_raw", "plac_raw", "null_decl", "plac_decl", "conf_raw", "conf_decl")


def _f(x, d=2):
    if x is None:
        return "-"
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("0.") else s.replace("-0.", "-.")


def _wr(e):
    return f"{e['world']} {e['regime']}" + ("" if e["lam"] is None else f" l{e['lam']:g}")


def _arms(cells):
    present = {e["arm"] for e in cells.values()}
    return [a for a in ARM_ORDER if a in present] + sorted(present - set(ARM_ORDER))


def _cell(cells, arm, wr, n, kappa=0.25):
    return next((e for e in cells.values() if e["arm"] == arm and _wr(e) == wr and e["n"] == n
                 and e["kappa"] == kappa), None)


def report(d: str, title: str, arms_keep: list[str] | None = None, notes: list[str] | None = None,
           seeds: str | None = None, cost_where: str | None = None, aggs: list[str] | None = None,
           runs: list[tuple[str, str]] | None = None) -> int:
    """``arms_keep``: report only these arms (T1 still counts every pair of dev_cells.json); ``notes`` replaces the
    10-arm run's section-5 notes; ``seeds`` / ``cost_where`` replace the seed line / the cost platform text;
    ``aggs`` (agg.json files, cost dicts united) and ``runs`` ((spec, merge summary) per run) replace the files of
    ``d`` (several DEV runs in one report)."""
    cells = json.load(open(os.path.join(d, "dev_cells.json"), encoding="utf-8"))["cells"]
    if arms_keep:
        cells = {k: e for k, e in cells.items() if e["arm"] in arms_keep}
    t1 = json.load(open(os.path.join(d, "t1.json"), encoding="utf-8"))
    agg = {}
    for p in aggs or [os.path.join(d, "agg.json")]:
        x = json.load(open(p, encoding="utf-8"))
        agg = {**x, "cost": {**agg.get("cost", {}), **x["cost"]}}
    sha = dict(map(tuple, t1["inputs"]["spec"]))
    runs = runs or [(t1["inputs"]["spec"][0][0], os.path.join(d, "merged.jsonl.summary.json"))]
    head = []
    for sp, sm in runs:
        summ = json.load(open(sm, encoding="utf-8"))
        head.append(f"DEV run, spec {sp} (sha256 {sha.get(sp, '?')[:12]}); {summ['n_records']} / "
                    f"{summ['n_expected']} units, {summ['n_ok']} ok, {len(summ['errors'])} errors, "
                    f"{len(summ['infeasible'])} infeasible, {len(summ['missing'])} missing; commits "
                    f"{[c[:7] for c in summ['commits']]}; platforms {summ['platforms']}.")
    arms = _arms(cells)
    wr = sorted({_wr(e) for e in cells.values()})
    val = Counter(e["primary"]["validity"] for e in cells.values() if e.get("primary"))
    L = [f"# {title}", ""] + head + [
         (seeds or "Seeds: tune 3_000_000-019; measure 3_000_100-119, + 3_000_120-159 in the R-21 cells (R2 all "
                    "worlds, E4 R3; n <= 4000).") + " Definitions: eval_analysis.py (PROTOCOL_A sec 8; R-29 conformal tau from the tune seeds, "
         "R-30 three-way validity: INVALID = CI lower bound > .05, VALID = upper <= .075, else INCONCLUSIVE; p arms: "
         "raw p <= .05 and BY declarations; tau arms: truth-null and confounded-placebo declarations). "
         "Descriptive only: nothing is tuned on these numbers. Files: dev_cells.json, t1.json, t1_pairs.json, agg.json.",
         "", f"Cells: {len(cells)}; validity {dict(val)} (NO_READ = no truth-null / confounded-placebo candidate: "
         "tau arms in E4 R1 / R2).", ""]
    # ------------------------------------------------------------------ validity
    L += ["## 1. Validity (R-30) per world-regime: INVALID / INCONCLUSIVE / VALID cells over n, lam, kappa", "",
          "| world regime | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    cnt = defaultdict(Counter)
    for e in cells.values():
        if e.get("primary"):
            cnt[(_wr(e), e["arm"])][e["primary"]["validity"]] += 1
    for w in wr:
        row = []
        for a in arms:
            c = cnt[(w, a)]
            row.append("-" if not c else ("nr" if set(c) == {"NO_READ"} else f"{c['INVALID']}/{c['INCONCLUSIVE']}/{c['VALID']}"))
        L.append(f"| {w} | " + " | ".join(row) + " |")
    L += ["", "Key reads, R2 (setpoint + dither) at n 1000, kappa .25: truth-null raw-p rate (p arms) or declaration "
          "rate (tau arms) [95 % seed-cluster CI]; P_placebo raw-p rate (p arms).", "",
          "| arm | " + " | ".join(f"E{i} R2" for i in (1, 2, 3, 5)) + " | P_placebo (E2 R2) |", "|---|---|---|---|---|---|"]
    for a in arms:
        row = []
        for w in ("E1 R2", "E2 R2", "E3 R2", "E5 R2"):
            e = _cell(cells, a, w, 1000)
            p = (e or {}).get("primary") or {}
            r = p.get("null_raw") or p.get("null_decl")
            row.append("-" if not r else f"{_f(r['rate'], 3)} [{_f(r['ci'][0], 3)}, {_f(r['ci'][1], 3)}]")
        e = _cell(cells, a, "E2 R2", 1000)
        pr = ((e or {}).get("primary") or {}).get("plac_raw")
        row.append("-" if not pr else f"{_f(pr['rate'], 3)} [{_f(pr['ci'][0], 3)}, {_f(pr['ci'][1], 3)}]")
        L.append(f"| {a} | " + " | ".join(row) + " |")
    # ------------------------------------------------------------------ recall
    NS = (1000, 4000, 24000)
    L += ["", "## 2. Recall (mean over measurement seeds), kappa .25, n " + " / ".join(map(str, NS)),
          "* = INVALID cell (not compared, R-39); ~ = INCONCLUSIVE / NO_READ; - = no cell.", "",
          "| world regime | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for w in wr:
        row = []
        for a in arms:
            v = []
            for n in NS:
                e = _cell(cells, a, w, n)
                p = (e or {}).get("primary")
                if not p or not p.get("recall"):
                    v.append("-")
                else:
                    v.append(_f(p["recall"]["mean"]) + {"INVALID": "*", "INCONCLUSIVE": "~", "NO_READ": "~"}.get(
                        p["validity"], ""))
            row.append("-" if all(x == "-" for x in v) else "/".join(v))
        L.append(f"| {w} | " + " | ".join(row) + " |")
    # ------------------------------------------------------------------ cost
    ns = sorted({int(k.rsplit("|n", 1)[1]) for k in agg["cost"]})
    L += ["", "## 3. Cost per (method, dataset): CPU-s mean / max over worlds, regimes, seeds (1 thread, "
          + (cost_where or "Kaggle Xeon 2.2 GHz") + ")", "", f"Budget R-13: {agg.get('budget_cpu_s')} CPU-s.", "",
          "| arm | " + " | ".join(f"n {n}" for n in ns) + " | peak RSS MB |", "|---|" + "---|" * (len(ns) + 1)]
    infeas = []
    for a in arms:
        row, rss = [], 0.0
        for n in ns:
            v = [c for k, c in agg["cost"].items() if k.startswith(a + "|") and k.endswith(f"|n{n}")]
            m = [c["cpu_s_mean"] for c in v if c["cpu_s_mean"] is not None]
            x = [c["cpu_s_max"] for c in v if c["cpu_s_max"] is not None]
            rss = max([rss] + [c["peak_rss_mb_max"] or 0 for c in v])
            row.append(f"{sum(m) / len(m):.1f} / {max(x):.{0 if max(x) >= 10 else 1}f}" if m else "-")
            infeas += [(a, n, i) for c in v for i in c["infeasible"]]
        L.append(f"| {a} | " + " | ".join(row) + f" | {rss:.0f} |")
    L += ["", "Infeasible (method, dataset): " + ("none (no unit near the budget)." if not infeas else "")]
    L += [f"- {a} n {n}: infeasible at this n (measured cost {i.get('cost') or i.get('cpu_s')}; {i['reason']})"
          for a, n, i in infeas[:10]]
    # ------------------------------------------------------------------ seeds
    L += ["", "## 4. Seed-count proposal (R-12, PROTOCOL_A rule T1, R-34)", "",
          f"- T1 (eval_analysis.t1_seed_count): kappa .25 cells at n 500 / 1000 / 4000 (E4 excluded), {t1.get('focal', 'pmrt_eq')} vs "
          f"each primary-block arm with the same declaration rule, neither INVALID; paired recall gap .15, alpha .05 "
          f"two-sided, power .8: {t1['n_pairs']} pairs, S_power = **{t1['S_power']}** (max over cells), "
          f"S = min(cap {t1['cap']}, max(40, S_power up to a multiple of 10)) = **{t1['S']}** "
          f"(floor binds: {t1['S'] == 40 and (t1['S_power'] or 0) <= 40}; cap binds: {t1['cap_binds']}). Minimum "
          f"detectable gap at S: max {_f(t1['max_mdg_at_S'], 3)}"
          + (f"; minimum power at S over the pairs {_f(t1['min_power_at_S'], 3)}." if t1.get("min_power_at_S")
             is not None else "."),
          "- Largest per-pair requirements: " + "; ".join(
              f"{p['arm']} {p['cell']} {p['seeds_needed']} (sd_d {_f(p['sd_d'])})" for p in t1["largest"][:6]) + ".",
          "- E4 (R3 / R4): S_E4 by rule T9 (R-34 / R-39; not computed here)."]
    pairs = json.load(open(os.path.join(d, "t1_pairs.json"), encoding="utf-8"))
    per_cell: dict[str, int] = {}
    for p in pairs:
        per_cell[p["cell"]] = max(per_cell.get(p["cell"], 0), p["seeds_needed"])
    L += ["- Per cell (max over its pairs; all pairs in t1_pairs.json): " + "; ".join(
        f"{c} {v}" for c, v in sorted(per_cell.items(), key=lambda t: (-t[1], t[0]))[:12]) + "; others smaller."]
    need = defaultdict(list)
    for k, e in cells.items():
        p = e.get("primary") or {}
        for r in ("null_raw", "plac_raw", "null_decl"):
            x = p.get(r)
            if x and x.get("n_clusters"):
                s = null_rate_seeds(x["n"] / x["n_clusters"], x.get("deff") or 1.0)
                if s:
                    need[r].append((s, k))
    L += ["- Null-rate precision (95 % CI half-width <= .02 at a true rate .05, DEV seed-cluster design effect): "
          "seeds needed, median / max over cells:"]
    for r, v in sorted(need.items()):
        v.sort()
        L.append(f"  {r}: {v[len(v) // 2][0]} / {v[-1][0]} (max: {v[-1][1]})")
    inv = Counter(e["regime"] for e in cells.values() if (e.get("primary") or {}).get("validity") == "INVALID")
    inv_arm = Counter(e["arm"] for e in cells.values() if (e.get("primary") or {}).get("validity") == "INVALID")
    pm = sorted(k for k, e in cells.items() if e["arm"].startswith("pmrt") and (e.get("primary") or {}).get(
        "validity") == "INVALID")
    L += ["", "## 5. Notes", ""]
    if notes is not None:
        L += [f"- INVALID cells by regime {dict(inv)}; by arm {dict(inv_arm.most_common())}. Reported, not tuned on "
              "(R-21)."] + notes
        open(os.path.join(d, "REPORT.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
        return len(L)
    L += [f"- INVALID cells by regime {dict(inv)}; by arm {dict(inv_arm.most_common())}. Reported, not tuned on (R-21).",
          f"- PMRT INVALID cells: {pm or 'none'}. All are E2 R2 n 1000; the binding rate is the truth-null raw p <= .05 "
          "(pmrt_eq .070 / .061 / .084 at kappa .125 / .25 / .5; .042 at n 500, .049 at n 4000); P_placebo raw and "
          "every BY declaration rate are VALID. Rejections are diffuse over the 32 null edges (max 14 / 100 per edge; "
          "sources P6 / P7 highest, .078 / .090). Flagged for the orchestrator (C2a; R-36 estimand check).",
          "- Runtime: Kaggle CPU, Python 3.12.13, numpy 2.0.2, scipy 1.16.3 (--pin off at launch a7b6e36; versions in "
          "every record). The CI-test arms (citests2 / hac2) are a separate pilot (status file)."]
    open(os.path.join(d, "REPORT.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    return len(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--title", default="DEV runs, Study A: validity, power, cost (10 arms)")
    ap.add_argument("--arms", default=None, help="comma list: report only these arms")
    ap.add_argument("--notes", default=None, help="text file: section-5 note lines (replaces the 10-arm notes)")
    ap.add_argument("--seeds", default=None, help="seed-line text")
    ap.add_argument("--cost-where", default=None, help="platform text of the cost section")
    ap.add_argument("--agg", nargs="+", default=None, help="agg.json files (default: --dir/agg.json)")
    ap.add_argument("--run", nargs="+", default=None, metavar="SPEC=SUMMARY",
                    help="spec path = its merge summary, one per run (default: --dir's merge summary)")
    a = ap.parse_args()
    notes = ([ln.rstrip("\n") for ln in open(a.notes, encoding="utf-8") if ln.strip()] if a.notes else None)
    n = report(a.dir, a.title, a.arms.split(",") if a.arms else None, notes, a.seeds, a.cost_where, a.agg,
               [tuple(x.split("=", 1)) for x in a.run] if a.run else None)
    print(f"REPORT.md {n} lines -> {a.dir}")


if __name__ == "__main__":
    main()

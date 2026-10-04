"""PROTOCOL_A definitions on the DEV campaign, streaming (brief dev-runs steps 4-5; low memory).

    uv run python scratchpad/xmethod/dev_t1.py --spec specs/dev/full.json [more DEV specs] \
        --merged results/dev/full/merged.jsonl.gz [more] --eval-spec specs/eval/full.json --out results/dev/full

eval_analysis.py (feat/v2) owns the definitions: per-cell rates, R-29 conformal tau, R-30 three-way validity,
cell status, and rule T1 (``t1_seed_count``). Its CLI loads every record into memory; here the key-sorted merged
file(s) are streamed one cell at a time through its own ``build_cells`` (its planned grid restricted to that
cell), and ``t1_seed_count`` then runs unchanged on the collected cells. Screening (one record per expected key,
expected role) is done here with eval_analysis.expected_units. Analysis-only arm fields (analysis, native_partner,
label, set_D, c2b) come from the EVAL spec for arms of the same name; nothing else is changed. Descriptive only.

Outputs: dev_cells.json (protocol cells, per-seed recall), t1.json (rule T1 result + screen counts + input shas).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from cdd_oran.xmethod.campaign import iter_jsonl  # noqa: E402

ANALYSIS_FIELDS = ("analysis", "native_partner", "label", "set_D", "c2b")


def load_eval_analysis(path: str | None = None):
    """eval_analysis.py of this tree, or ``path`` (e.g. feat/v2's current copy, exported with git show)."""
    sp = importlib.util.spec_from_file_location("eval_analysis", path or os.path.join(ROOT, "scratchpad", "xmethod",
                                                                                      "eval_analysis.py"))
    E = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(E)
    return E


def united_spec(paths: list[str], eval_spec: str | None) -> dict:
    """The DEV specs united as eval_analysis' t1 CLI does, plus the EVAL spec's analysis-only arm fields."""
    specs = [json.load(open(x, encoding="utf-8")) for x in paths]
    spec = {"name": "+".join(str(x.get("name")) for x in specs),
            "arms": {k: dict(v) for x in specs for k, v in x["arms"].items()},
            "blocks": [{**b, "arms": b.get("arms", "all") if b.get("arms", "all") != "all" else list(x["arms"])}
                       for x in specs for b in x["blocks"]]}
    if eval_spec:
        ev = json.load(open(eval_spec, encoding="utf-8"))["arms"]
        for a, d in spec["arms"].items():
            d.update({f: ev[a][f] for f in ANALYSIS_FIELDS if a in ev and f in ev[a]})
    return spec


def stream_cells(E, merged: list[str], spec: dict) -> tuple[dict, dict, dict]:
    """(cells, rows, screen) with eval_analysis' own build_cells, one cell at a time."""
    want = E.expected_units(spec)
    plan_all = E._grid_cells(spec)
    real_grid = E._grid_cells
    cells, rows = {}, {}
    scr = {"unexpected": [], "role_mismatch": [], "duplicates": [], "n_used": 0}
    seen: set[str] = set()

    def flush(ck, group):
        E._grid_cells = lambda _spec: ({ck: plan_all[ck]} if ck in plan_all else {})
        try:
            c, r = E.build_cells(group, spec)
        finally:
            E._grid_cells = real_grid
        cells.update(c)
        rows.update(r)

    for path in merged:
        cur, group = None, []
        for rec in iter_jsonl(path):
            k = rec.get("key")
            if k in seen:
                scr["duplicates"].append(k)
                continue
            seen.add(k)
            if k not in want:
                scr["unexpected"].append(k)
                continue
            u = E.unit_of(rec)
            if u["role"] != want[k]["role"]:
                scr["role_mismatch"].append(k)
                continue
            ck = f"{u['arm']}|{E.cell_of(u['world'], u['regime'], u['lam'], u['n'], u['kappa'])}"
            if ck != cur:
                if cur is not None:
                    if cur in cells:
                        raise ValueError(f"{path}: not sorted by cell ({cur} again)")
                    flush(cur, group)
                cur, group = ck, []
            group.append(rec)
            scr["n_used"] += 1
        if cur is not None:
            flush(cur, group)
    for ck in sorted(set(plan_all) - set(cells)):         # planned cells without any record: 'missing'
        flush(ck, [])
    return cells, rows, scr


def t1_pairs(E, cells: dict, spec: dict, focal: str | None = None) -> list[dict]:
    """Every (cell, arm) pair of rule T1 with its paired-seed count: the same filters as
    eval_analysis.t1_seed_count (which returns only the largest), for the per-cell report (brief step 4)."""
    focal = focal or E.FOCAL
    out = []
    for k, e in sorted(cells.items()):
        if (e["arm"] == focal or e["block"] != "primary" or e["kappa"] != E.PRIMARY_KAPPA or e["n"] not in E.T1_NS
                or e["world"] == "E4" or not E.counts(e) or e["primary"]["validity"] == "INVALID"):
            continue
        f = cells.get(f"{focal}|{k.split('|', 1)[1]}")
        if f is None or not E.counts(f) or f["primary"]["validity"] == "INVALID":
            continue
        fp = f["primary"] if e["declare"] == "by" else f.get("secondary_tau")
        if not fp or fp["validity"] in ("INVALID", "few_seeds"):
            continue
        d = E._paired(fp["per_seed_recall"], e["primary"]["per_seed_recall"])
        if len(d) < 2:
            continue
        sd = float(np.std(d, ddof=1))
        out.append({"cell": k.split("|", 1)[1], "arm": e["arm"], "n_seeds": len(d), "mean_d": float(np.mean(d)),
                    "sd_d": sd, "seeds_needed": E.paired_seeds(sd)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--merged", nargs="+", required=True)
    ap.add_argument("--eval-spec", default=os.path.join(ROOT, "scratchpad", "xmethod", "specs", "eval", "full.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--eval-analysis", default=None, help="eval_analysis.py to use (default: this tree's)")
    ap.add_argument("--focal", default=None, help="PMRT arm of the T1 pairs (R-43: pmrt_nl_eq; default FOCAL)")
    a = ap.parse_args()
    E = load_eval_analysis(a.eval_analysis)
    spec = united_spec(a.spec, a.eval_spec)
    cells, rows, scr = stream_cells(E, a.merged, spec)
    real_screen, real_build = E.screen, E.build_cells
    E.screen = lambda recs, sp: (recs, {})
    E.build_cells = lambda use, sp: (cells, rows)
    try:
        t1 = (E.t1_seed_count([], spec, a.cap or E.S_CAP, focal=a.focal) if a.focal
              else E.t1_seed_count([], spec, a.cap or E.S_CAP))
    finally:
        E.screen, E.build_cells = real_screen, real_build
    pairs = t1_pairs(E, cells, spec, a.focal)                     # every pair (t1_seed_count keeps the 15 largest)
    if t1["S_power"] is not None and max(p["seeds_needed"] for p in pairs) != t1["S_power"]:
        raise AssertionError("per-pair reconstruction disagrees with t1_seed_count")
    json.dump(E._clean(pairs), open(os.path.join(a.out, "t1_pairs.json"), "w", encoding="utf-8"), indent=1)
    t1["inputs"] = {"spec": [[x, E.file_sha256(x)] for x in a.spec], "merged": [[m, E.file_sha256(m)] for m in a.merged],
                    "eval_spec_fields": a.eval_spec, "eval_analysis": E.ANALYSIS_VERSION,
                    "eval_analysis_file": [a.eval_analysis or "scratchpad/xmethod/eval_analysis.py",
                                           E.file_sha256(a.eval_analysis or os.path.join(ROOT, "scratchpad", "xmethod",
                                                                                         "eval_analysis.py"))]}
    t1["screen"] = {k: (len(v) if isinstance(v, list) else v) for k, v in scr.items()}
    os.makedirs(a.out, exist_ok=True)
    json.dump(E._clean(t1), open(os.path.join(a.out, "t1.json"), "w", encoding="utf-8"), indent=1)
    json.dump(E._clean({"spec": spec["name"], "eval_analysis": E.ANALYSIS_VERSION, "cells": cells}),
              open(os.path.join(a.out, "dev_cells.json"), "w", encoding="utf-8"), indent=1)
    st: dict[str, int] = {}
    for e in cells.values():
        st[e["status"]] = st.get(e["status"], 0) + 1
    print(json.dumps({"cells": len(cells), "status": st, "S": t1["S"], "S_power": t1["S_power"],
                      "n_pairs": t1["n_pairs"], "screen": t1["screen"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

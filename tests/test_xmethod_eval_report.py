"""Tests of the Study A report generator (scratchpad/xmethod/eval_report.py) on the synthetic study of
tests/test_xmethod_eval_analysis.py."""
from __future__ import annotations

import copy
import csv
import gzip
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str, path: str):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


T = _load("t_eval_analysis", os.path.join(ROOT, "tests", "test_xmethod_eval_analysis.py"))
RP = _load("eval_report", os.path.join(ROOT, "scratchpad", "xmethod", "eval_report.py"))
E = RP.E


def _canon(x) -> str:
    return json.dumps(E._clean(x), sort_keys=True)


def _write(recs, path, gz=False):
    op = gzip.open if gz else open
    with op(path, "wt", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def recs():
    return T.synth_records()


def test_chunked_equals_analyse(recs, tmp_path):
    half = len(recs) // 2                                   # two shards, one gzipped, records interleaved
    p1 = _write(recs[::2], tmp_path / "a.jsonl.gz", gz=True)
    p2 = _write(recs[1::2] + recs[:3], tmp_path / "b.jsonl")      # + 3 duplicates
    assert half
    kw = {"spec_sha": T.SSHA, "protocol": {"ok": True}}
    ref = E.analyse(E._load_many([p1, p2]), T.SPEC, T.COMMIT, **kw)
    got, n_in = RP.chunked_analyse([p1, p2], T.SPEC, T.COMMIT, tmp_dir=str(tmp_path), log=lambda *_: None, **kw)
    assert _canon(got) == _canon(ref)
    assert sum(n_in.values()) == len(recs) + 3 and got["V11_integrity"]["n_duplicates"] == 3
    assert not [p for p in os.listdir(tmp_path) if p.startswith("xm_eval_report_")]       # temp dir removed


def test_chunked_equals_analyse_with_defects(recs, tmp_path):
    rs = copy.deepcopy(recs)
    rs[5]["edges"] = rs[5]["edges"][:-1]                     # candidate set incomplete
    rs[7]["code"] = {"commit": "other", "dirty": True}        # commit / clean violations
    rs[9]["dataset_sha256"] = "x"                             # dataset hash mismatch across arms
    rs = [r for i, r in enumerate(rs) if i != 11]             # a missing unit
    rs.append({**copy.deepcopy(recs[0]), "key": "ghost|" + recs[0]["key"].split("|", 1)[1], "arm": "ghost"})
    p = _write(rs, tmp_path / "d.jsonl")
    kw = {"spec_sha": T.SSHA, "protocol": {"ok": True}}
    ref = E.analyse(E._load_many([p]), T.SPEC, T.COMMIT, **kw)
    got, _ = RP.chunked_analyse([p], T.SPEC, T.COMMIT, tmp_dir=str(tmp_path), log=lambda *_: None, **kw)
    assert _canon(got) == _canon(ref)
    v = got["V11_integrity"]
    assert v["label"] == "PROVISIONAL" and {"candidates_complete", "commits", "clean", "dataset_hash", "missing",
                                             "unexpected"} <= set(v["failed"])


def test_dev_spec_fields_and_t10_fallback():
    ev = json.load(open(os.path.join(ROOT, "scratchpad", "xmethod", "specs", "eval", "full.json"), encoding="utf-8"))
    dev = {"name": "dev_x", "arms": {a: {k: ev["arms"][a][k] for k in ("ref", "config", "declare")}
                                     for a in ("pmrt_eq", "corr", "mscr_native", "mscr_eq", "pcorr_eq")},
           "blocks": [{"role": "measure", "worlds": ["E1"], "regimes": ["R1"], "ns": [500], "kappas": [0.25],
                       "seeds": [3000100, 3000119]}]}
    dev["arms"]["pdcor_eq"] = {"ref": "x:Y", "config": {}, "declare": "by"}
    spec, notes = RP.dev_spec(ev, [dev])
    assert spec["focal"] == "pmrt_eq" and spec["arms"]["pmrt_eq"]["analysis"] == "primary"
    assert "label" not in spec["arms"]["pmrt_eq"] and "pdcor_eq" not in spec["arms"]
    assert spec["arms"]["mscr_native"]["c1_sensitivity_drop"] and spec["arms"]["mscr_eq"]["c2b"] is False
    assert spec["blocks"][0]["arms"] == ["pmrt_eq", "corr", "mscr_native", "mscr_eq", "pcorr_eq"]
    assert any("T10 fallback" in x for x in notes) and any("pdcor_eq" in x for x in notes)
    dev["arms"]["pmrt_nl_eq"] = {k: ev["arms"]["pmrt_nl_eq"][k] for k in ("ref", "config", "declare")}
    spec2, notes2 = RP.dev_spec(ev, [dev])
    assert spec2["focal"] == "pmrt_nl_eq" and not any("T10 fallback" in x for x in notes2)
    assert RP.sub_spec(spec2, "corr")["arms"].keys() == {"corr"} and RP.sub_spec(spec2, "corr")["focal"] == "pmrt_nl_eq"


def test_main_dev_report_watermarked(recs, tmp_path):
    ev = copy.deepcopy(T.SPEC)
    ev["arms"]["ci_native"]["c1_sensitivity_drop"] = True
    dev = {"name": "dev_synth", "arms": {a: {k: d[k] for k in ("ref", "config", "declare")}
                                         for a, d in T.SPEC["arms"].items()}, "blocks": T.SPEC["blocks"]}
    pe, pd = tmp_path / "eval.json", tmp_path / "dev.json"
    pe.write_text(json.dumps(ev), encoding="utf-8")
    pd.write_text(json.dumps(dev), encoding="utf-8")
    pm = _write(recs, tmp_path / "m.jsonl.gz", gz=True)
    out = tmp_path / "rep"
    rc = RP.main(["--dev", "--spec", str(pe), "--dev-spec", str(pd), "--merged", pm, "--out", str(out),
                  "--tmp-dir", str(tmp_path)])
    assert rc == 2                                             # DEV is never FINAL
    md = (out / "REPORT.md").read_text(encoding="utf-8")
    assert md.startswith("# [DEV, NOT EVAL]") and md.rstrip().endswith(f"> **{RP.DEV_MARK}**")
    secs = md.split("\n## ")[1:]
    assert len(secs) >= 9 and all(RP.DEV_MARK in s for s in secs)                  # watermark in every section
    for h in ("1. Claim", "2. C1", "3. C2a", "4. C2b", "5. C3", "6. Power", "7. kappa sweep", "8. Not in grid",
              "9. Figure-ready"):
        assert f"\n## {h}" in md
    assert "C1 sensitivity (R-56, no effect): without ci_native" in md
    assert "Every eq arm, same rule on R1 + R2, unfiltered (R-56)" in md and "Pre-registered disclosure" in md
    assert "### V0 like-for-like" in md and "### V11 integrity" in md
    js = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert js["meta"]["mode"] == "DEV" and js["meta"]["watermark"] == RP.DEV_MARK
    assert js["tables"]["V4_verdicts"]["C1"]["sensitivity_without"]["dropped"] == ["ci_native"]
    names = {"validity_cells", "recall_vs_n", "like_for_like", "paired_diff", "information_levels_R2",
             "e4_by_lambda", "kappa_sweep", "cost", "eq_arms", "not_in_grid"}
    assert {p[:-4] for p in os.listdir(out / "csv")} == names
    with open(out / "csv" / "validity_cells.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(js["tables"]["V1_validity"]) and {"arm", "null_raw_rate", "null_raw_lo"} <= set(rows[0])
    assert list(rows[0])[0] == "data_mode" and {r["data_mode"] for r in rows} == {"DEV-NOT-EVAL"}   # watermark
    with open(out / "csv" / "eq_arms.csv", encoding="utf-8") as fh:
        assert {r["arm"] for r in csv.DictReader(fh)} == {"ci_eq", "pc_eq"}


def test_r4_not_applicable_rows_listed(tmp_path):
    spec = copy.deepcopy(T.SPEC)
    for b in spec["blocks"]:
        b.update(worlds=["E4"], regimes=["R4"], ns=[500], e4_lams={"R4": [1.0]})
    rs = [T._record(u["arm"], u["world"], u["regime"], u["lam"], u["n"], u["seed"], u["role"], False)
          for u in E.expected_units(spec).values()]
    out, _ = RP.chunked_analyse([_write(rs, tmp_path / "r4.jsonl")], spec, T.COMMIT, spec_sha=T.SSHA,
                                protocol={"ok": True}, tmp_dir=str(tmp_path), log=lambda *_: None)
    meta = {"mode": "EVAL", "spec_sha256": T.SSHA, "utc": "t", "inputs": [], "records_by_arm": {}, "csv": {}}
    md = RP.report_md(out, meta)
    sec = md.split("\n## 6. ")[1].split("\n## ")[0]
    assert "| pmrt_eq | E4 R4 lam1 k0.25 n500 | NA (" in sec                      # PMRT in R4: never recall 0
    assert not [r for r in out["V2_recall"] if r["arm"] == "pmrt_eq" and r["recall"] is not None]
    assert "DEV" not in md.split("\n", 1)[0] and RP.DEV_MARK not in md                # EVAL rendering: no watermark


def test_report_not_in_grid_section(recs, tmp_path):
    spec = copy.deepcopy(T.SPEC)
    spec["blocks"] = [{**b, "arms": [a for a in spec["arms"] if a != "ci_native"]} for b in T.SPEC["blocks"]] + \
        [{**b, "ns": [500], "arms": ["ci_native"]} for b in T.SPEC["blocks"]]
    spec["not_in_grid"] = [{"ruling": "R-x", "arms": ["ci_native"], "regimes": None, "ns": [1000], "kappas": None,
                            "reason": "test trim"}]
    rs = [r for r in recs if not (r["arm"] == "ci_native" and r["job"]["n"] == 1000)]
    out, _ = RP.chunked_analyse([_write(rs, tmp_path / "g.jsonl")], spec, T.COMMIT, spec_sha=T.SSHA,
                                protocol={"ok": True}, tmp_dir=str(tmp_path), log=lambda *_: None)
    meta = {"mode": "EVAL", "spec_sha256": T.SSHA, "utc": "t", "inputs": [], "records_by_arm": {}, "csv": {}}
    md = RP.report_md(out, meta)
    sec = md.split("\n## 8. Not in grid")[1].split("\n## ")[0]
    assert "| R-x | ci_native | ns [1000] | " in sec and "test trim" in sec and "unexplained gaps: none" in sec
    assert out["V4_verdicts"]["C1"]["arms"]["ci_native"]["complete"] and out["V11_integrity"]["checks"]["grid_gaps_ruled"]
    nig = RP.csv_tables(out)["not_in_grid"]
    assert nig and {r["arm"] for r in nig} == {"ci_native"} and {r["n"] for r in nig} == {1000}


def test_eval_mode_refuses_dev_seeds(recs, tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(T.SPEC), encoding="utf-8")
    with pytest.raises(SystemExit, match="EVAL spec"):
        RP.main(["--spec", str(p), "--merged", str(_write(recs[:5], tmp_path / "m.jsonl")), "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="--dev-spec needs --dev"):
        RP.main(["--spec", str(p), "--dev-spec", str(p), "--merged", "x", "--out", str(tmp_path)])

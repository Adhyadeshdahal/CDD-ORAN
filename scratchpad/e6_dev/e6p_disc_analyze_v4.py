"""E6-P discovery v4 ANALYZER: frozen PMRT on small fresh slices (protocol docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md,
sections 3-6). Runs on Kaggle (scratchpad/e6_dev/kaggle_job.py, --sources = the v4 collection kernels); a local DRY RUN
on small DEV data is possible (never a verdict).

  python scratchpad/e6_dev/e6p_disc_analyze_v4.py analyze --eval SPEC --placebo SPEC --gt SPEC --dev SPEC
         [--artifact docs/benchmark/artifacts/E6P_MSCRPLUS_V4_FROZEN.json | E6P_PMRT_V4.json] [--out DIR]
         [--cache-dir DIR]
         [--B 9999] [--B-null 999] [--null-group 60] [--null-shifts 4] [--workers 4]
         [--desc-methods shap_gbdt,int,qacm,two_tower] [--desc-splits pooled,300,120] [--no-v2] [--no-null]
         [--dry-run] [--slice-sizes 60,120,300] [--allow-smoke] [--allow-artifact-mismatch]
SPEC = comma list of JSONL files / directories (recursive res_*.jsonl or all.jsonl, /bundle/ skipped) / globs. --gt may
also be a disc_bench gtref .json (DB.save_ref). Each JSONL input is streamed into a disc_bench cache npz
(disc_bench.build_cache, Hs 30 / 60 / 90 / 150; stage filter per option) in --cache-dir (default OUT/cache); the GT
reference is disc_bench.gt_reference_files over every gt_v4 episode (gt_p.summarize, frozen rule, orientation "dir").
Output: OUT/analysis_v4.json (everything) + OUT/verdict_v4.json (criteria + verdict) + the log on stdout.

METHOD (frozen; section 3). The artifact is the protocol's frozen E6P_MSCRPLUS_V4_FROZEN.json (``ARTIFACT_SHA256``;
method label "MSCR+") or its label-only successor E6P_PMRT_V4.json (``PMRT_ARTIFACT_SHA256``; same learned content,
docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4_ADDENDUM_PMRT.md); ``artifact_status`` checks its sha256 and its embedded code
sha256s against the running files (for the frozen artifact under the renamed code: via the PMRT artifact's
``supersedes`` / ``legacy_code_sha256`` / ``renamed_from`` chain, ``code_via`` "supersession"). It is loaded by
pmrt_artifacts.load_artifact; the statistic is pmrt_bench.run_integrated
(arms plain_c / loadsp_c / max, B conditional re-draws, RNG default_rng([0, 6616, 3, family_idx, split])), the layer
fdr_layer.declare with the artifact's ev2 prior (q .05, floor .2, thr 3, cap 5) and n_target = the slice size. PRIMARY
= loadsp_c + wby1s; the other 8 combinations are descriptive. Unit window rule H = 90, H_pre = 90 (t0 >= 90).
Slices (section 4; ``slice_plan``): eval_v4 j = seed - 188100; pooled (split 0, n_target 600), slice60 i = j // 60
(split 20 + i, n_target 60), slice120 k = j // 120 (split 40 + k, n_target 120); descriptive slice300 m = j // 300
(split 60 + m, n_target 300; split fixed here, not in the protocol). Placebo: split 9 (as the ev3 bench), n_target 40.
K0n variants: split 0, n_target = the group size (60).

CRITERIA (section 5; PRIMARY combination only; C = chain set, C* = members TRUE with the expected sign in gt_v4):
  K0   placebo_v4 pooled: P(Binom(m, .05) >= n_reject) >= .01 over the m finite p's the layer uses AND <= 1 declaration
  K0n  eval_v4 null outcomes: groups = the 10 slice60 groups, cyclic shifts k = 1..4 of the OTHER episodes' lab_series
       inside the group (the .tmp/mscr_plus/V/validity_check.variants recipe, re-implemented here), B 999 -> 40 variants:
       pooled rate(used p <= .05) <= .075 AND every family's rate <= .10 AND variants with >= 1 declaration <= 5
  K1   every 120-slice: >= 60 tested sleep units (H 90 / H_pre 90 unit table) with >= 15 logged rejects
  G    gt_v4 sleep -> nbr pv TRUE with sign +1
  P1   in >= 3 of the 5 120-slices: premise (sleep -> nbr pv) declared + AND >= min(3, |C*|) members of C* declared
       with the GT sign
  P2   indirect (nbr) F1 of PMRT >= the baseline's in >= 6 of 10 60-slices AND >= 3 of 5 120-slices, for EACH of corr,
       granger (DEV-v4 far-FPR tau) and granger_by; ties count for PMRT; an undefined PMRT F1 is a loss (an undefined
       baseline F1 counts as 0, as edge_score.p2_check)
  P3   pooled 600: premise declared +, >= min(3, |C*|) hits, overall precision >= .80, overall sign accuracy >= .90
  S    pooled sign accuracy >= .90 AND the mean over the 120-slices with a defined sign accuracy (>= 1 TP) >= .90
       (no slice defined -> fail)
Verdict precedence: INVALID (K0 or K0n) > NO-CHAIN (G) > UNDERPOWERED (K1) > PASS (P1 and P2 and P3 and S) / PARTIAL
(P3 and S, P1 or P2 fails) / KILL. The label is "NOT A VERDICT" unless mode == "full" (complete non-smoke v4 data: eval
600 = 188100-188699, placebo 40 = 188700-188739, gt 40 = 188800-188839, dev 20 = 188000-188019, all sub "v4"; K0n run;
artifact / code sha match) and "(PROVISIONAL: protocol not frozen)" unless the V4 doc says "FROZEN: yes" and its
LF sha256 equals e6p_discovery.FROZEN_SHA256_V4.
DESCRIPTIVE: the other 8 combinations, MSCR-CRT v2 + BY, the two 300-slices, recall over every GT-TRUE nbr edge
(ind_recall), per-hypothesis z / p (every split), the baselines' placebo declarations with their DEV tau, the descriptive
baselines (shap_gbdt, int, qacm, two_tower; DEV far-FPR tau; errors recorded, e.g. no torch / shap), unit counts and the
tested units with t0 in [90, 120).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from cdd_oran.decision import disc_bench as DB  # noqa: E402
from cdd_oran.decision.crt_units import FAMILIES, MODES  # noqa: E402

PROTOCOL_DOC_V4 = "docs/benchmark/E6P_DISCOVERY_PROTOCOL_V4.md"
DRIVER = "scratchpad/e6_dev/e6p_discovery.py"
ARTIFACT = "docs/benchmark/artifacts/E6P_MSCRPLUS_V4_FROZEN.json"          # the protocol's frozen artifact ("MSCR+")
ARTIFACT_SHA256 = "4735a85a1975edc6ddea412972be2f972a4ade9015844f153ae45d82f47d14ba"
PMRT_ARTIFACT = "docs/benchmark/artifacts/E6P_PMRT_V4.json"                    # label-only successor (PMRT)
PMRT_ARTIFACT_SHA256 = "TBD"
KNOWN_ARTIFACTS = {ARTIFACT_SHA256: ARTIFACT, PMRT_ARTIFACT_SHA256: PMRT_ARTIFACT}
PRIMARY = ("loadsp_c", "wby1s")
SUB = "v4"
SEEDS = {"dev": (188000, 20), "eval": (188100, 600), "placebo": (188700, 40), "gt": (188800, 40)}
N60, N120, N300 = 60, 120, 300
SPLIT = {"pooled": 0, "60": 20, "120": 40, "300": 60, "placebo": 9, "null": 0}
H, H_PRE = 90, 90
BASELINES = ("corr", "granger", "granger_by")
DESC_METHODS = ("shap_gbdt", "int", "qacm", "two_tower")
K0_RULE = {"alpha": 0.05, "binom_level": 0.01, "max_decl": 1}
K0N_RULE = {"rate": 0.075, "family_rate": 0.10, "max_var_decl": 5, "alpha": 0.05}
K1_RULE = {"min_sleep_units": 60, "min_sleep_rejects": 15}
P1_RULE = {"min_slices": 3, "chain_min": 3}
P2_RULE = {"min_60": 6, "min_120": 3}
P3_RULE = {"chain_min": 3, "precision": 0.80, "sign_acc": 0.90}
S_RULE = {"sign_acc": 0.90}
KEEP = ("kind", "schema", "key", "stage", "sub", "seed", "j", "fold", "smoke", "policy", "pi0_table", "n_cells", "units",
        "lab_series")


def log(s=""):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def sha_lf(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def _js(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _keys(d):
    if isinstance(d, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k): _keys(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_keys(v) for v in d]
    return d


def dump(obj, path):
    with open(path, "w", newline="\n") as fh:
        json.dump(_keys(obj), fh, indent=1, default=_js)


def peak_rss_mb():
    try:
        from e6p_disc_analyze_v2 import peak_rss_mb as f
        return f()
    except Exception:                                                  # noqa: BLE001
        return None


# ============================================================================================ status: protocol / artifact
def protocol_status(root: str = ROOT) -> dict:
    """V4 doc: FROZEN line, LF sha256 and the driver's FROZEN_SHA256_V4 (parsed from the source, no import)."""
    path = os.path.join(root, PROTOCOL_DOC_V4)
    drv = os.path.join(root, DRIVER)
    want = None
    if os.path.exists(drv):
        m = re.search(r'^FROZEN_SHA256_V4 = (None|"([0-9a-f]{64})")', open(drv, encoding="utf-8").read(), re.M)
        want = m.group(2) if m else None
    if not os.path.exists(path):
        return {"doc": PROTOCOL_DOC_V4, "exists": False, "frozen": False, "sha256": None, "driver_sha256": want}
    txt = open(path, "rb").read().decode("utf-8", "replace")
    line = next((ln.strip() for ln in txt.splitlines() if ln.strip().startswith("FROZEN:")), "FROZEN: ?")
    sha = sha_lf(path)
    return {"doc": PROTOCOL_DOC_V4, "exists": True, "line": line, "sha256": sha, "driver_sha256": want,
            "frozen": bool(line.lower().startswith("frozen: yes") and want is not None and sha == want)}


def _code_direct(want: dict, root: str) -> dict:
    code = {}
    for rel, w in (want or {}).items():
        p = os.path.join(root, *rel.split("/"))
        have = sha_lf(p) if os.path.exists(p) else None
        code[rel] = {"want": w, "have": have, "ok": have == w}
    return code


def _code_via_pmrt(a: dict, sha: str, root: str) -> dict | None:
    """The frozen (legacy-label) artifact's code check under the renamed code: the PMRT artifact (sha
    PMRT_ARTIFACT_SHA256) must supersede exactly this artifact, its ``legacy_code_sha256`` must equal this artifact's
    code_sha256, and every running (renamed) file must match the PMRT artifact's code_sha256. Keys = the OLD paths."""
    pm = os.path.join(root, *PMRT_ARTIFACT.split("/"))
    if not os.path.exists(pm) or sha_lf(pm) != PMRT_ARTIFACT_SHA256:
        return None
    b = json.load(open(pm))
    old = a.get("code_sha256") or {}
    ren = b.get("renamed_from") or {}
    if ((b.get("supersedes") or {}).get("sha256") != sha or b.get("legacy_code_sha256") != old
            or sorted(ren.get(n, n) for n in b.get("code_sha256", {})) != sorted(old)):
        return None
    code = {}
    for new, w in b["code_sha256"].items():
        rel = ren.get(new, new)
        p = os.path.join(root, *new.split("/"))
        have = sha_lf(p) if os.path.exists(p) else None
        code[rel] = {"want": old[rel], "have": have, "running": new, "want_renamed": w, "ok": have == w}
    return code


def artifact_status(path: str, root: str = ROOT) -> dict:
    """sha256 of the artifact vs the known v4 artifacts (``KNOWN_ARTIFACTS``: the frozen one and its PMRT successor)
    and its embedded code sha256s vs the running files; for the frozen artifact under the renamed code, the check goes
    through the PMRT artifact's supersession record (``_code_via_pmrt``; ``code_via`` "supersession")."""
    a = json.load(open(path))
    sha = sha_lf(path)
    code, via = _code_direct(a.get("code_sha256"), root), "direct"
    if sha == ARTIFACT_SHA256 and not (code and all(v["ok"] for v in code.values())):
        chained = _code_via_pmrt(a, sha, root)
        if chained is not None:
            code, via = chained, "supersession"
    return {"path": path, "sha256": sha, "expected": sorted(KNOWN_ARTIFACTS), "sha_ok": sha in KNOWN_ARTIFACTS,
            "artifact": KNOWN_ARTIFACTS.get(sha), "schema": a.get("schema"), "version": a.get("version"),
            "legacy_version": a.get("legacy_version"), "code": code, "code_via": via,
            "code_ok": bool(code) and all(v["ok"] for v in code.values()), "source": a.get("source")}


# ============================================================================================ inputs
def expand(spec: str, patterns=("res_*.jsonl", "all.jsonl")) -> list:
    out = []
    for p in [x for x in (spec or "").split(",") if x]:
        if os.path.isdir(p):
            hits = []
            for pat in patterns:
                hits += glob.glob(os.path.join(p, "**", pat), recursive=True)
            res = [f for f in hits if os.path.basename(f).startswith("res_")]
            hits = res if res else hits                      # a pulled run dir has both: prefer the shard files
            out += sorted(f for f in hits if "/bundle/" not in f.replace("\\", "/"))
        elif any(ch in p for ch in "*?["):
            out += sorted(glob.glob(p, recursive=True))
        else:
            out.append(p)
    seen, uniq = set(), []
    for f in out:
        a = os.path.abspath(f)
        if a not in seen:
            seen.add(a)
            uniq.append(f)
    return uniq


def _cache_one(args):
    path, stage, dst = args
    if not os.path.exists(dst):
        DB.build_cache([path], dst, stages={stage}, src=os.path.basename(path))
    return dst


def build_caches(files: list, stage: str, cache_dir: str, tag: str, workers: int = 1) -> list:
    """npz inputs pass through; each JSONL -> one cache npz (episodes of ``stage`` only; files without such episodes
    are skipped)."""
    os.makedirs(cache_dir, exist_ok=True)
    npz = [f for f in files if f.endswith(".npz")]
    jobs = []
    for i, f in enumerate(f for f in files if not f.endswith(".npz")):
        h = hashlib.sha256(os.path.abspath(f).encode()).hexdigest()[:10]
        jobs.append((f, stage, os.path.join(cache_dir, f"{tag}_{i:03d}_{h}.npz")))
    done = []
    if workers > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(min(workers, len(jobs))) as pool:
            for r in pool.imap_unordered(_safe_cache, jobs):
                if r:
                    done.append(r)
    else:
        for j in jobs:
            r = _safe_cache(j)
            if r:
                done.append(r)
    return npz + sorted(done)


def _safe_cache(job):
    try:
        return _cache_one(job)
    except ValueError as e:                     # no episode of this stage in the file (e.g. a header-only shard)
        print(f"[cache] skip {job[0]}: {e}", flush=True)
        return None


def load_records_light(files: list, stage: str, allow_smoke: bool = False) -> list:
    """Episode records of ``stage`` with only the KEEP keys (lab_kpi dropped), deduped by (seed, sub), sorted."""
    recs = {}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip() or '"episode"' not in line[:200]:
                    continue
                r = json.loads(line)
                if r.get("kind") != "episode" or r.get("stage") != stage:
                    continue
                if r.get("smoke") and not allow_smoke:
                    continue
                for u in r["units"]:
                    u.pop("lab_kpi", None)
                recs[(int(r["seed"]), str(r.get("sub") or ""))] = {k: r[k] for k in KEEP if k in r}
    return [recs[k] for k in sorted(recs)]


# ============================================================================================ slices
def slice_plan(seeds, base: int = SEEDS["eval"][0], pseudo: bool = False, sizes=(N60, N120, N300)) -> list:
    """[{"name", "kind", "index", "split", "n_target", "seeds"}] for the pooled set and the 60 / 120 / 300 slices
    (section 4). ``pseudo`` (dry run): j = the rank of the seed instead of seed - base."""
    seeds = sorted(int(s) for s in seeds)
    j = {s: (r if pseudo else s - base) for r, s in enumerate(seeds)}
    n60, n120, n300 = sizes
    out = [{"name": "pooled", "kind": "pooled", "index": 0, "split": SPLIT["pooled"], "n_target": len(seeds),
            "seeds": list(seeds)}]
    n_total = (max(j.values()) + 1) if seeds else 0
    if not pseudo:
        n_total = SEEDS["eval"][1]
    for kind, n in (("60", n60), ("120", n120), ("300", n300)):
        for i in range(n_total // n):
            mem = [s for s in seeds if j[s] // n == i]
            out.append({"name": f"s{kind}_{i}", "kind": kind, "index": i, "split": SPLIT[kind] + i, "n_target": n,
                        "seeds": mem})
    return out


def episodes_of(pool, seeds) -> np.ndarray:
    want = set(int(s) for s in seeds)
    return np.array([e for e in range(pool.n_eps) if int(pool.eps["seed"][e]) in want], int)


def data_status(pools: dict, gt_seeds: list, dry: bool, allow_smoke: bool) -> dict:
    """Completeness of the v4 data (mode "full" needs every check True)."""
    st = {}
    for name, pool in pools.items():
        base, n = SEEDS[name]
        if pool is None:
            st[name] = {"ok": False, "n": 0, "note": "missing"}
            continue
        seeds = sorted(int(s) for s in pool.eps["seed"])
        subs = sorted(set(str(s) for s in pool.eps["sub"]))
        smoke = bool(np.any(pool.eps["smoke"]))
        ok = (seeds == list(range(base, base + n)) and subs == [SUB] and not smoke)
        st[name] = {"ok": bool(ok), "n": len(seeds), "subs": subs, "smoke": smoke,
                    "seed_range": [seeds[0], seeds[-1]] if seeds else None, "expected": [base, base + n - 1]}
    base, n = SEEDS["gt"]
    st["gt"] = {"ok": sorted(gt_seeds) == list(range(base, base + n)), "n": len(gt_seeds),
                "expected": [base, base + n - 1]}
    st["all_ok"] = all(v["ok"] for v in st.values() if isinstance(v, dict)) and not dry
    return st


# ============================================================================================ criteria (pure)
def k0_check(decl: dict, rule=K0_RULE) -> dict:
    from scipy.stats import binom
    ps = [float(v["p"]) for v in decl.values() if v.get("status") != "undetermined" and np.isfinite(v.get("p", np.nan))]
    m, k = len(ps), int(sum(p <= rule["alpha"] for p in ps))
    tail = float(binom.sf(k - 1, m, rule["alpha"])) if m else 1.0
    nd = int(sum(bool(v.get("declared")) for v in decl.values()))
    return {"pass": bool(m > 0 and tail >= rule["binom_level"] and nd <= rule["max_decl"]), "m": m, "n_reject": k,
            "rate": k / m if m else float("nan"), "binom_tail": tail, "n_declared": nd,
            "declared": sorted("|".join(h) for h, v in decl.items() if v.get("declared")), "rule": dict(rule),
            "min_p": min(ps) if ps else float("nan")}


def k0n_check(pvals, fams, var_decl: list, rule=K0N_RULE) -> dict:
    """``pvals`` / ``fams``: every used p over every variant; ``var_decl``: #declarations per variant."""
    P, F = np.asarray(pvals, float), np.asarray(fams, object)
    rate = float(np.mean(P <= rule["alpha"])) if len(P) else float("nan")
    fr = {f: float(np.mean(P[F == f] <= rule["alpha"])) for f in sorted(set(F.tolist()))}
    nvd = int(sum(int(d) > 0 for d in var_decl))
    ok = bool(len(P) and len(var_decl) and rate <= rule["rate"] + 1e-12
              and all(v <= rule["family_rate"] + 1e-12 for v in fr.values()) and nvd <= rule["max_var_decl"])
    return {"pass": ok, "n_variants": len(var_decl), "n_p": int(len(P)), "rate05": rate, "rate05_family": fr,
            "n_variants_with_decl": nvd, "n_decl": int(sum(var_decl)), "rule": dict(rule)}


def k1_check(per_slice: dict, rule=K1_RULE) -> dict:
    """``per_slice`` {name: {"sleep_units", "sleep_rejects"}} over the 120-slices."""
    rows = {k: dict(v, ok=bool(v["sleep_units"] >= rule["min_sleep_units"]
                                 and v["sleep_rejects"] >= rule["min_sleep_rejects"])) for k, v in per_slice.items()}
    return {"pass": bool(rows) and all(v["ok"] for v in rows.values()), "slices": rows, "rule": dict(rule)}


def g_check(cells: list) -> dict:
    c = next((c for c in cells if (c["family"], c["relation"], c["kpi"]) == DB.PREMISE), None)
    if c is None:
        return {"pass": False, "cell": None, "rule": "gt_v4 sleep -> nbr pv TRUE(+)"}
    return {"pass": bool(c["status"] == "TRUE" and int(c["sign"]) == 1), "cell": c,
            "rule": "gt_v4 sleep -> nbr pv TRUE(+) (dir, frozen gt_p rule)"}


def p1_check(m120: list, rule=P1_RULE) -> dict:
    """``m120``: disc_bench.subset_metrics of the primary combination on each 120-slice."""
    per = []
    for m in m120:
        need = min(rule["chain_min"], int(m["chain_true"]))
        ok = bool(m["premise_hit"] == 1.0 and m["chain_hits"] >= need)
        per.append({"name": m.get("name"), "premise": bool(m["premise_hit"] == 1.0), "chain_hits": int(m["chain_hits"]),
                    "chain_true": int(m["chain_true"]), "need": need, "ok": ok})
    n_ok = sum(p["ok"] for p in per)
    return {"pass": bool(n_ok >= rule["min_slices"]), "n_ok": n_ok, "n_slices": len(per), "slices": per,
            "rule": dict(rule)}


def _win(m, b) -> bool:
    m = float(m) if m is not None else float("nan")
    b = float(b) if b is not None else float("nan")
    b = 0.0 if not np.isfinite(b) else b
    return bool(np.isfinite(m) and m >= b - 1e-12)


def p2_check(f1_60: list, f1_120: list, base60: dict, base120: dict, rule=P2_RULE) -> dict:
    """``f1_*``: PMRT indirect F1 per slice (slice order); ``base*`` {baseline: [F1 per slice]}."""
    per = {}
    for b in base60:
        w60 = [_win(m, x) for m, x in zip(f1_60, base60[b], strict=True)]
        w120 = [_win(m, x) for m, x in zip(f1_120, base120[b], strict=True)]
        per[b] = {"wins60": int(sum(w60)), "n60": len(w60), "wins120": int(sum(w120)), "n120": len(w120),
                  "win60": w60, "win120": w120,
                  "ok": bool(sum(w60) >= rule["min_60"] and sum(w120) >= rule["min_120"])}
    return {"pass": bool(per) and all(v["ok"] for v in per.values()), "per_baseline": per, "rule": dict(rule),
            "ties": "count for PMRT; undefined PMRT F1 = loss; undefined baseline F1 = 0"}


def p3_check(decl: dict, ref: dict, rule=P3_RULE) -> dict:
    from cdd_oran.decision.edge_score import score_method
    m = DB.subset_metrics(decl, ref)
    sc = score_method(decl, ref)
    need = min(rule["chain_min"], int(m["chain_true"]))

    def ge(v, t):
        return bool(v is not None and np.isfinite(v) and v >= t - 1e-12)
    parts = {"premise_declared_plus": bool(m["premise_hit"] == 1.0), "chain_hits": bool(m["chain_hits"] >= need),
             "overall_precision": ge(sc["overall"]["precision"], rule["precision"]),
             "sign_accuracy": ge(sc["overall"]["sign_acc"], rule["sign_acc"])}
    return {"pass": all(parts.values()), "parts": parts, "rule": dict(rule),
            "values": {"chain_true": int(m["chain_true"]), "chain_need": need, "chain_hits": int(m["chain_hits"]),
                       "overall_precision": sc["overall"]["precision"], "sign_accuracy": sc["overall"]["sign_acc"],
                       "far_declared": sc["far_declared"], "ind_recall": sc["indirect"]["recall"]},
            "chain": m["chain"]}


def s_check(pooled_sign: float, signs120: list, rule=S_RULE) -> dict:
    f = [float(s) for s in signs120 if s is not None and np.isfinite(float(s))]
    mean = float(np.mean(f)) if f else float("nan")
    ok_p = bool(pooled_sign is not None and np.isfinite(pooled_sign) and pooled_sign >= rule["sign_acc"] - 1e-12)
    ok_s = bool(f and mean >= rule["sign_acc"] - 1e-12)
    return {"pass": ok_p and ok_s, "pooled": pooled_sign, "mean120": mean, "n_defined120": len(f),
            "pooled_ok": ok_p, "mean120_ok": ok_s, "rule": dict(rule)}


def verdict(k0, k0n, g, k1, p1, p2, p3, s, mode: str, frozen: bool) -> dict:
    if not (k0 and k0["pass"]) or not (k0n and k0n["pass"]):
        v = "INVALID"
    elif not (g and g["pass"]):
        v = "NO-CHAIN"
    elif not (k1 and k1["pass"]):
        v = "UNDERPOWERED"
    elif p1["pass"] and p2["pass"] and p3["pass"] and s["pass"]:
        v = "PASS"
    elif p3["pass"] and s["pass"]:
        v = "PARTIAL"
    else:
        v = "KILL"
    label = v
    if mode != "full":
        label = f"NOT A VERDICT ({mode}; would-be: {v})"
    elif not frozen:
        label = f"{v} (PROVISIONAL: protocol not frozen)"
    return {"verdict": v, "label": label,
            "precedence": "INVALID (K0 or K0n) > NO-CHAIN (G) > UNDERPOWERED (K1) > PASS (P1 P2 P3 S) / PARTIAL "
                          "(P3 S) / KILL"}


# ============================================================================================ methods
def hyp_table(run: dict, decl: dict) -> dict:
    """Per-hypothesis z / p of the three statistic arms + what the primary layer used."""
    out = {}
    for h, o in run.items():
        d = decl.get(tuple(h.split("|")), {})
        row = {"status": o.get("status")}
        if o.get("status") == "tested":
            for s in ("plain_c", "loadsp_c", "max"):
                row[s] = {k: o[s].get(k) for k in ("z", "p2", "p_plus", "p_minus", "sign")}
        row["primary"] = {k: d.get(k) for k in ("declared", "sign", "p", "w", "one_sided", "status")}
        out[h] = row
    return out


def desc_method(name: str, dev, seed: int = 0):
    """A descriptive baseline (BD.SCORERS name) with the DEV far-FPR tau, as baselines_disc.run_baselines."""
    from cdd_oran.decision import baselines_disc as BD
    fn = BD.SCORERS[name]
    kw = {"seed": seed} if name in ("shap_gbdt", "two_tower", "qacm") else {}
    dev_kw = dict(kw, relations=("own", "far")) if name == "qacm" else kw
    tau = BD.tune_tau(fn(dev, **dev_kw), "far_fpr")
    rels = BD.DECL_RELATIONS.get(name, BD.RELATIONS)

    def m(data, split):
        return BD.declare(fn(data, **kw), tau["tau"], rels)
    m.label, m.tau = name, tau
    return m


def unit_counts(ud) -> dict:
    fam = {f: int(len(ud.rows_of(f))) for f in FAMILIES}
    rej = MODES.index("reject")
    early = {}
    for f in FAMILIES:
        r = ud.rows_of(f)
        early[f] = int(((ud.t0[r] >= 90) & (ud.t0[r] < 120)).sum())
    sl = ud.rows_of("sleep")
    return {"units": int(ud.n), "by_family": fam, "t0_90_120_by_family": early,
            "sleep_units": int(len(sl)), "sleep_rejects": int((ud.mode[sl] == rej).sum()),
            "n_episodes": int(len(np.unique(ud.episode)))}


# ============================================================================================ K0n (null outcomes)
def shift_variants(recs: list, shifts: int) -> list:
    """[(name, records)]: cyclic shift k = 1..shifts of the lab_series inside the group ``recs`` (episode i gets the
    series of episode (i + k) mod G); the same recipe as .tmp/mscr_plus/V/validity_check.variants(group=0)."""
    n = len(recs)
    out = []
    for k in range(1, shifts + 1):
        if n == 0 or k % n == 0:
            continue
        out.append((f"shift{k}", [dict(r, lab_series=recs[(i + k) % n]["lab_series"]) for i, r in enumerate(recs)]))
    return out


def run_k0n(recs: list, groups: list, params, priors, cfg, shifts: int, log_=log) -> dict:
    import pmrt_bench as PB

    from cdd_oran.decision import pmrt as PM
    tmp = tempfile.mkdtemp(prefix="k0n_v4_")
    per = {PB.cname(s, lay): {"p": [], "fam": [], "var_decl": []} for s, lay in PB.COMBOS}
    names = []
    for gi, idx in enumerate(groups):
        sub = [recs[i] for i in idx]
        for name, rr in shift_variants(sub, shifts):
            t = time.time()
            pd = PM.pmrt_data_from_records(rr, tmp)
            run = PB.run_integrated(pd, params, cfg, SPLIT["null"])
            for (s, lay), d in PB.declare_all(run, priors, len(sub)).items():
                o = per[PB.cname(s, lay)]
                o["var_decl"].append(int(sum(bool(v["declared"]) for v in d.values())))
                for h, v in d.items():
                    if v.get("status") != "undetermined" and np.isfinite(v.get("p", np.nan)):
                        o["p"].append(float(v["p"]))
                        o["fam"].append(h[0])
            names.append(f"g{gi}:{name}")
            pc = per[PB.cname(*PRIMARY)]
            log_(f"  K0n g{gi} {name}: units {pd.n}, primary decl {pc['var_decl'][-1]} ({time.time() - t:.0f} s)")
            del pd, rr
    res = {k: k0n_check(o["p"], o["fam"], o["var_decl"]) for k, o in per.items()}
    return {"variants": names, "combos": res, "primary": res[PB.cname(*PRIMARY)]}


# ============================================================================================ analyze
def analyze(a) -> dict:
    import pmrt_bench as PB
    import pmrt_artifacts as PAR

    from cdd_oran.decision import pmrt as PM
    t_all = time.time()
    out = a.out or os.environ.get("JOB_OUT") or "."
    os.makedirs(out, exist_ok=True)
    cache_dir = a.cache_dir or os.path.join(out, "cache")
    timing = {}
    rep = {"schema": "e6p-disc-v4-analysis/1", "protocol": protocol_status(), "argv": sys.argv[1:],
           "primary": "+".join(PRIMARY)}
    # ---- artifact
    ast = artifact_status(a.artifact)
    rep["artifact"] = ast
    log(f"artifact {a.artifact}: sha {ast['sha256']} ({ast['artifact']}) ok {ast['sha_ok']}; code ok "
        f"{ast['code_ok']} via {ast['code_via']} {[k for k, v in ast['code'].items() if not v['ok']]}")
    if not (ast["sha_ok"] and ast["code_ok"]) and not a.allow_artifact_mismatch:
        raise SystemExit("artifact sha256 / code sha256 mismatch (pass --allow-artifact-mismatch for a dry run)")
    params, priors = PAR.load_artifact(a.artifact)
    cfg = PAR.pmrt_config(params, a.B)
    cfg_null = PAR.pmrt_config(params, a.B_null)
    sizes = tuple(int(x) for x in a.slice_sizes.split(","))
    # ---- caches
    t = time.time()
    files = {k: expand(getattr(a, k)) for k in ("eval", "placebo", "dev")}
    caches = {k: build_caches(files[k], k, os.path.join(cache_dir, k), k, a.workers) for k in files}
    timing["caches"] = round(time.time() - t, 1)
    log(f"caches: { {k: len(v) for k, v in caches.items()} } [{timing['caches']} s]")
    ev_stage = "eval"
    if not caches["eval"] and a.dry_run:                     # dry run on DEV data: EVAL := DEV
        caches["eval"], ev_stage = caches["dev"], "dev"
    if not caches["eval"] or not caches["placebo"] or not caches["dev"]:
        raise SystemExit(f"missing inputs: { {k: len(v) for k, v in caches.items()} }")
    pool = PM.load_pmrt_pool(caches["eval"], stages={ev_stage})
    ppool = PM.load_pmrt_pool(caches["placebo"], stages={"placebo"})
    dpool = DB.load_pool(caches["dev"], H=H, H_pre=H_PRE, stages={"dev"})
    if not a.allow_smoke and any(bool(np.any(p.eps["smoke"])) for p in (pool, ppool, dpool)):
        raise SystemExit("smoke records in the inputs (pass --allow-smoke for a dry run)")
    # ---- GT reference
    t = time.time()
    if a.gt.endswith(".json") and os.path.isfile(a.gt):
        g = DB.load_ref(a.gt)
    else:
        g = DB.gt_reference_files(expand(a.gt))
    ref = g["ref"]
    timing["gt"] = round(time.time() - t, 1)
    G = g_check(g["cells"])
    rep["gt"] = {k: g[k] for k in ("n_episodes", "n_labels", "labels_per_family", "counts", "nbr_true", "seeds",
                                   "delta", "cells") if k in g}
    rep["G"] = G
    log(f"GT: {g['n_episodes']} eps, {g['n_labels']} labels, counts {g['counts']}, nbr TRUE {g['nbr_true']}; "
        f"G {'PASS' if G['pass'] else 'FAIL'} [{timing['gt']} s]")
    # ---- mode
    dry = bool(a.dry_run) or ev_stage != "eval"
    dstat = data_status({"eval": pool if ev_stage == "eval" else None, "placebo": ppool, "dev": dpool},
                        g.get("seeds", []), dry, a.allow_smoke)
    rep["data"] = dstat
    # ---- slices
    ev_seeds = sorted(int(s) for s in pool.eps["seed"])
    e0, en = SEEDS["eval"]
    in_range = bool(ev_seeds) and all(e0 <= x < e0 + en for x in ev_seeds)
    plan = slice_plan(ev_seeds, pseudo=dry or not in_range, sizes=sizes)
    for sl in plan:
        sl["episodes"] = episodes_of(pool, sl["seeds"])
        sl["complete"] = len(sl["episodes"]) == sl["n_target"]
    rep["slices"] = [{k: sl[k] for k in ("name", "kind", "index", "split", "n_target", "complete")} |
                     {"seed_range": [min(sl["seeds"]), max(sl["seeds"])] if sl["seeds"] else None,
                      "n": len(sl["episodes"])} for sl in plan]
    log("slices: " + ", ".join(f"{sl['name']}:{len(sl['episodes'])}" for sl in plan))
    # ---- methods
    dev = dpool.unit_data()
    bmeth = {b: DB.method_baseline(b, dev) for b in BASELINES}
    rep["baseline_tau"] = {b: getattr(m, "tau", None) for b, m in bmeth.items()}
    dmeth, derr = {}, {}
    for nm in [x for x in (a.desc_methods or "").split(",") if x]:
        try:
            t = time.time()
            dmeth[nm] = desc_method(nm, dev)
            log(f"desc {nm}: tau {dmeth[nm].tau.get('tau')} [{time.time() - t:.0f} s]")
        except Exception as e:                                              # noqa: BLE001
            derr[nm] = repr(e)[:400]
            log(f"desc {nm}: unavailable ({derr[nm][:120]})")
    rep["desc_tau"] = {k: m.tau for k, m in dmeth.items()}
    rep["desc_errors"] = derr
    desc_kinds = set(x for x in (a.desc_splits or "").split(",") if x)
    v2cfg = None
    if not a.no_v2:
        from cdd_oran.decision import crt_units_v2 as V2
        v2cfg = V2.UnitCRTConfigV2(B=a.B)
    # ---- K0 (placebo)
    t = time.time()
    pdp = PM.pmrt_data(ppool)
    run_p = PB.run_integrated(pdp, params, cfg, SPLIT["placebo"])
    decl_p = PB.declare_all(run_p, priors, int(ppool.n_eps))
    K0 = k0_check(decl_p[PRIMARY])
    plc = {PB.cname(*k): k0_check(d) for k, d in decl_p.items()}
    for b, m in list(bmeth.items()) + list(dmeth.items()):
        try:
            plc[b] = DB.placebo_check(m, pdp.ud, SPLIT["placebo"])
        except Exception as e:                                              # noqa: BLE001
            plc[b] = {"error": repr(e)[:300]}
    if v2cfg is not None:
        e2 = V2.edges_from_crt_v2(V2.run_crt_units_v2(pdp.ud, v2cfg, SPLIT["placebo"]))
        plc["v2+by"] = {"n_declared": int(sum(bool(v["declared"]) for v in e2.values())),
                        "rate05": float(np.mean([v["p"] <= .05 for v in e2.values() if np.isfinite(v["p"])] or [np.nan]))}
    timing["k0"] = round(time.time() - t, 1)
    rep["K0"], rep["placebo_descriptive"] = K0, plc
    rep["placebo_units"] = unit_counts(pdp.ud)
    rep["placebo_hyp"] = hyp_table(run_p, decl_p[PRIMARY])
    log(f"K0 {'PASS' if K0['pass'] else 'FAIL'}: m {K0['m']} reject {K0['n_reject']} tail {K0['binom_tail']:.3g} "
        f"declared {K0['n_declared']} {K0['declared']} [{timing['k0']} s]")
    del pdp
    # ---- slices
    res, pooled_decl = {}, None
    for sl in plan:
        if not len(sl["episodes"]):
            continue
        t = time.time()
        pd = PM.pmrt_data(pool, sl["episodes"])
        run = PB.run_integrated(pd, params, cfg, sl["split"])
        decls = {PB.cname(*k): d for k, d in PB.declare_all(run, priors, sl["n_target"]).items()}
        for b, m in bmeth.items():
            decls[b] = m(pd.ud, sl["split"])
        if v2cfg is not None:
            decls["v2+by"] = V2.edges_from_crt_v2(V2.run_crt_units_v2(pd.ud, v2cfg, sl["split"]))
        if sl["kind"] in desc_kinds:
            for b, m in dmeth.items():
                try:
                    decls[b] = m(pd.ud, sl["split"])
                except Exception as e:                                      # noqa: BLE001
                    derr[f"{b}@{sl['name']}"] = repr(e)[:300]
        mets = {}
        for k, d in decls.items():
            mm = DB.subset_metrics(d, ref)
            mm["declared"] = sorted(("|".join(h), int(v.get("sign", 0))) for h, v in d.items()
                                    if v and v.get("declared"))
            mets[k] = mm
        res[sl["name"]] = {"kind": sl["kind"], "index": sl["index"], "split": sl["split"], "n_target": sl["n_target"],
                           "n_episodes": int(len(sl["episodes"])), "units": unit_counts(pd.ud), "metrics": mets,
                           "hyp": hyp_table(run, decls[PB.cname(*PRIMARY)]), "wall_s": round(time.time() - t, 1)}
        pm = mets[PB.cname(*PRIMARY)]
        if sl["kind"] == "pooled":
            pooled_decl = decls[PB.cname(*PRIMARY)]
        log(f"{sl['name']:>9} ({len(sl['episodes'])} eps, {pd.n} units, {res[sl['name']]['wall_s']} s): primary chain "
            f"{pm['chain_hits']}/{pm['chain_true']} premise {pm['premise_hit']:.0f} indF1 {pm['ind_f1']:.2f} ovP "
            f"{pm['ov_precision']:.2f} sign {pm['sign_acc']:.2f} #dec {pm['n_declared']} | "
            + " ".join(f"{b}:{mets[b]['ind_f1']:.2f}" for b in BASELINES))
        del pd
        if sl["kind"] == "pooled":
            dump(dict(rep, slices_result=res), os.path.join(out, "analysis_v4.partial.json"))
    rep["slices_result"] = res
    timing["slices"] = round(sum(r["wall_s"] for r in res.values()), 1)
    # ---- K0n
    K0n = None
    if not a.no_null:
        t = time.time()
        recs = load_records_light(files["eval"] if ev_stage == "eval" else files["dev"], ev_stage, a.allow_smoke)
        seeds = [int(r["seed"]) for r in recs]
        if dry or not in_range:
            order = list(range(len(recs)))
            groups = [order[i:i + a.null_group] for i in range(0, len(order) - a.null_group + 1, a.null_group)]
        else:
            groups = [[i for i, s in enumerate(seeds) if (s - SEEDS["eval"][0]) // a.null_group == gi]
                      for gi in range(SEEDS["eval"][1] // a.null_group)]
        log(f"K0n: {len(recs)} records, {len(groups)} groups of {a.null_group}, {a.null_shifts} shifts, B {a.B_null}")
        kn = run_k0n(recs, groups, params, priors, cfg_null, a.null_shifts)
        del recs
        K0n = kn["primary"]
        rep["K0n"], rep["K0n_descriptive"] = K0n, {"variants": kn["variants"], "combos": kn["combos"]}
        timing["k0n"] = round(time.time() - t, 1)
        log(f"K0n {'PASS' if K0n['pass'] else 'FAIL'}: {K0n['n_variants']} variants, rate {K0n['rate05']:.4f} "
            f"families {K0n['rate05_family']} variants with decl {K0n['n_variants_with_decl']} [{timing['k0n']} s]")
    else:
        rep["K0n"] = None
    # ---- criteria
    P = PB.cname(*PRIMARY)
    s60 = [res[s["name"]] for s in plan if s["kind"] == "60" and s["name"] in res]
    s120 = [res[s["name"]] for s in plan if s["kind"] == "120" and s["name"] in res]
    K1 = k1_check({f"s120_{r['index']}": {"sleep_units": r["units"]["sleep_units"],
                                          "sleep_rejects": r["units"]["sleep_rejects"]} for r in s120})
    P1 = p1_check([dict(r["metrics"][P], name=f"s120_{r['index']}") for r in s120])
    P2 = p2_check([r["metrics"][P]["ind_f1"] for r in s60], [r["metrics"][P]["ind_f1"] for r in s120],
                  {b: [r["metrics"][b]["ind_f1"] for r in s60] for b in BASELINES},
                  {b: [r["metrics"][b]["ind_f1"] for r in s120] for b in BASELINES})
    pooled = res["pooled"]
    P3 = p3_check(pooled_decl, ref)
    S = s_check(pooled["metrics"][P]["sign_acc"], [r["metrics"][P]["sign_acc"] for r in s120])
    mode = "full" if (dstat["all_ok"] and K0n is not None and ast["sha_ok"] and ast["code_ok"]
                      and all(s["complete"] for s in plan)) else ("dry-run" if dry else "partial")
    V = verdict(K0, K0n, G, K1, P1, P2, P3, S, mode, rep["protocol"]["frozen"])
    rep.update(mode=mode, K1=K1, P1=P1, P2=P2, P3=P3, S=S, verdict=V)
    rep["desc_errors"] = derr
    timing["total"] = round(time.time() - t_all, 1)
    rep["timing_s"], rep["peak_rss_mb"] = timing, peak_rss_mb()
    # ---- print + write
    log("")
    log(f"{'split':>9} " + " ".join(f"{k:>16}" for k in [P, "corr", "granger", "granger_by"]) + "   (ind F1)")
    for name, r in res.items():
        log(f"{name:>9} " + " ".join(f"{r['metrics'][k]['ind_f1']:16.3f}" for k in [P, "corr", "granger", "granger_by"]))
    for nm, c in (("K0", K0), ("K0n", K0n), ("G", G), ("K1", K1), ("P1", P1), ("P2", P2), ("P3", P3), ("S", S)):
        log(f"{nm:>4}: {'PASS' if c and c['pass'] else 'FAIL'}"
            + (f"  {c.get('parts', '')}" if c and "parts" in c else ""))
    log(f"P2 per baseline: { {b: (v['wins60'], v['wins120']) for b, v in P2['per_baseline'].items()} }")
    log(f"mode {mode}; data {({k: v['ok'] for k, v in dstat.items() if isinstance(v, dict)})}; protocol "
        f"{rep['protocol'].get('line')} (frozen {rep['protocol']['frozen']})")
    log(f"VERDICT: {V['label']}")
    log(f"timing {timing}; peak RSS {rep['peak_rss_mb']} MB")
    dump(rep, os.path.join(out, "analysis_v4.json"))
    dump({k: rep[k] for k in ("mode", "verdict", "K0", "K0n", "G", "K1", "P1", "P2", "P3", "S", "protocol", "artifact",
                              "data")}, os.path.join(out, "verdict_v4.json"))
    part = os.path.join(out, "analysis_v4.partial.json")
    if os.path.exists(part):
        os.remove(part)
    log(f"wrote {os.path.join(out, 'analysis_v4.json')}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    b = sp.add_parser("analyze")
    b.add_argument("--eval", default="")
    b.add_argument("--placebo", required=True)
    b.add_argument("--gt", required=True)
    b.add_argument("--dev", required=True)
    b.add_argument("--artifact", default=os.path.join(ROOT, ARTIFACT))
    b.add_argument("--out", default=None)
    b.add_argument("--cache-dir", default=None)
    b.add_argument("--B", type=int, default=9999)
    b.add_argument("--B-null", dest="B_null", type=int, default=999)
    b.add_argument("--null-group", dest="null_group", type=int, default=60)
    b.add_argument("--null-shifts", dest="null_shifts", type=int, default=4)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--desc-methods", dest="desc_methods", default=",".join(DESC_METHODS))
    b.add_argument("--desc-splits", dest="desc_splits", default="pooled,300,120",
                   help="slice kinds on which the descriptive baselines run (pooled, 300, 120, 60)")
    b.add_argument("--slice-sizes", dest="slice_sizes", default=f"{N60},{N120},{N300}",
                   help="dry run only (the protocol fixes 60,120,300)")
    b.add_argument("--no-v2", dest="no_v2", action="store_true")
    b.add_argument("--no-null", dest="no_null", action="store_true")
    b.add_argument("--dry-run", dest="dry_run", action="store_true")
    b.add_argument("--allow-smoke", dest="allow_smoke", action="store_true")
    b.add_argument("--allow-artifact-mismatch", dest="allow_artifact_mismatch", action="store_true")
    a = ap.parse_args(argv)
    if a.slice_sizes != f"{N60},{N120},{N300}" and not a.dry_run:
        raise SystemExit("--slice-sizes is for --dry-run only")
    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    return analyze(a)


if __name__ == "__main__":
    main()

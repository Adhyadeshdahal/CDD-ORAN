"""E6-P option (a) kill test K-B, ANALYSIS ("does confounding bite?"; scratchpad/e6_dev/decision/OPTION_A_PLAN.md
section 6; DEV kill test, NOT frozen). Runs as a Kaggle job (scratchpad/e6_dev/kaggle_job.py; SHAP / two-tower / QACM
need more RAM than the laptop may use). Input: the K-B collection records (scratchpad/e6_dev/e6p_opta_kb.py, schema
"e6p-disc-rec/1", policies "placebo_incumbent" / "incumbent") and the step-1 DEV records (runs/e6p-disc-dev-1, pi0,
unconfounded) for the baselines' DEV thresholds.

  python scratchpad/e6_dev/e6p_opta_kb_analyze.py analyze --records SPEC --dev SPEC [--json OUT] [--B 9999]
         [--methods shap_gbdt,corr,...] [--no-ann] [--H 90] [--H-pre 60] [--n-boot 2000] [--allow-smoke]
SPEC = comma-separated files / directories (every *.jsonl below, e.g. $JOB_SRC/<kernel slug>) / globs. The K-B
records are split by their ``kb_stage`` (placebo | applied); ``panel`` is dropped on load (unused).

Steps
  unit tables  crt_units.build_unit_data(H = 90, H_pre = 60): H_pre = 60 so units opening at t0 in [60, 120) (inside
               the warm-up, where the picos fall asleep) survive; t0 < 60 is dropped (counted). Every K-B unit carries
               its own logged row ``probs`` (IncumbentPolicy), which build_unit_data prefers -> MSCR-CRT v2 re-draws
               each unit from its OWN context-dependent row. The DEV table uses the same H / H_pre.
  K0           MSCR-CRT v2 (crt_units_v2.placebo_rejection_units_v2) on the confounded PLACEBO: binomial count rule
               (level .01) and <= 1 BY declaration (frozen v1 / v2 rule).
  placebo      each baseline (baselines_disc.run_baselines: shap_gbdt, corr, granger, two_tower, int, qacm, +
               granger_by untuned) declares on the placebo with (a) its DEV tau (far-FPR rule on runs/e6p-disc-dev-1,
               exactly the v1 / v2 code path) and (b) a PLACEBO-CALIBRATED tau = the (MAX_FP + 1)-th largest finite
               placebo score over its declarable relations (MAX_FP = 1 = K0's BY allowance): <= 1 placebo
               declaration by construction. Every placebo declaration is false (sharp null).
  applied      each method's declared map on the APPLIED set: MSCR-CRT v2 (BY q .05; beta = its design-centred slope)
               and each baseline under (a) and (b) (beta = the method's sign x |naive OLS slope of y on x| over the
               family's applied units: the associational effect size). Edges vs the knockout GT (gt_ext, dir):
               TRUE-right-sign / TRUE-wrong-sign / NULL / INDET.
  replay       OFFLINE MapGateV2 (cdd_oran/decision/mapgate.py, theta .05, k_conf 1) replay: for every map (the GT
               map = TRUE edges of gt_ext, ``e6p_opta_ka.M_GT``, and each method's map) a fresh MapGateV2 per episode
               decides every logged unit of the APPLIED episodes in opening order (reads the unit's ctx and cell
               only; the duty-bound run is the map's own). flips(map) per episode = #units whose decision differs
               from the GT map's, split by family / direction and by type (GT defer -> map accept, GT accept -> map
               defer); mean per episode + 90 % bootstrap CI over episodes (default_rng([6624, 1, map_idx])).
               Offline = on the incumbent's logged contexts (not a MapGate-driven trajectory): it measures how
               often the map would CHANGE a referee decision, not the KPI consequence.
               Three flip counts per method map M (pairs): "vs GT" = flips(M, GT map) (wrong AND missing edges);
               "~clean" = flips(M, M restricted to its GT-TRUE right-sign edges) (decisions changed by M's WRONG
               edges only); "GT+plc" = flips(GT map + the method's PLACEBO-declared edges (placebo naive-slope
               magnitudes), GT map) (decisions changed by pure-confounding edges; the plan's "do baseline placebo
               edges flip >= 1 MapGate decision per episode?").
Kill rule (plan section 6), two labels:
  label (AS SPECIFIED)  KILL iff MSCR K0 fails, or no baseline map (DEV tau, the v1 / v2 code path) flips >= 1
                        MapGate decision per episode (mean) vs the GT map; else CONTINUE. CAVEAT: missing edges also
                        flip decisions, so an under-powered (even empty) map passes this rule.
  label_attributable    the same with max("~clean", "GT+plc") in place of "vs GT": flips caused by wrong edges only.
Reported alongside (not in either rule): the placebo-calibrated tau variants and MSCR's own three flip counts.
"""
from __future__ import annotations

import dataclasses
import glob
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

from cdd_oran.decision import baselines_disc as BD  # noqa: E402
from cdd_oran.decision import crt_units as CU  # noqa: E402
from cdd_oran.decision import crt_units_v2 as V2  # noqa: E402
from cdd_oran.decision import mapgate as MG  # noqa: E402

SCHEMA = "e6p-disc-rec/1"
PLAN = "scratchpad/e6_dev/decision/OPTION_A_PLAN.md section 6 K-B (2026-09-30, not frozen)"
K0_RULE = {"binom_level": 0.01, "max_by": 1}
H_DEFAULT, H_PRE_DEFAULT = 90, 60
MAX_FP = 1
FLIP_MIN = 1.0                          # kill rule: >= 1 flipped MapGate decision per episode
BOOT_TAG = 6624
GT_JSON = os.path.join(HERE, "decision", "step1_v3_full.json")


def log(*a):
    print(*a, flush=True)


def _js(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _keys(d):
    if isinstance(d, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else k): _keys(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_keys(v) for v in d]
    return d


def peak_rss_mb():
    try:
        import e6p_disc_analyze_v2 as A2
        return A2.peak_rss_mb()
    except Exception:                                                 # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------------- io
def expand(spec) -> list:
    """Files of a comma-separated spec (files, directories -> every *.jsonl below, globs), sorted, unique."""
    out = []
    for s in [x for x in (spec or "").split(",") if x]:
        if os.path.isdir(s):
            out += glob.glob(os.path.join(s, "**", "*.jsonl"), recursive=True)
        elif any(ch in s for ch in "*?["):
            out += glob.glob(s, recursive=True)
        elif os.path.exists(s):
            out.append(s)
        else:
            raise SystemExit(f"not found: {s}")
    return sorted(set(out))


def read_episodes(files, keep=None, allow_smoke=False) -> list:
    """Episode records (schema e6p-disc-rec/1) of ``files`` passing ``keep(rec)``; ``panel`` dropped; deduplicated by
    (key, smoke); sorted by seed."""
    uniq = {}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip() or '"episode"' not in line[:200]:
                    continue
                r = json.loads(line)
                if r.get("kind") != "episode" or r.get("schema") != SCHEMA:
                    continue
                if r.get("smoke") and not allow_smoke:
                    continue
                if keep is not None and not keep(r):
                    continue
                r.pop("panel", None)
                uniq[tuple(r["key"]) + (bool(r.get("smoke")),)] = r
    return sorted(uniq.values(), key=lambda r: (r["seed"], r.get("sub", "")))


# ---------------------------------------------------------------------------------------------- GT
def gt_cells() -> list:
    """gt_ext "dir" cells (e6p_opta_ka.GT_EXT_CELLS, = step1_v3_full.json["gt"]["cells"], checked when present)."""
    import e6p_opta_ka as KA
    cells = KA._cells()
    if os.path.exists(GT_JSON):
        js = json.load(open(GT_JSON))["gt"]["cells"]
        if [(c["family"], c["relation"], c["kpi"], c["status"], c["mean"]) for c in js] != list(KA.GT_EXT_CELLS):
            raise SystemExit(f"GT table mismatch: {GT_JSON} vs e6p_opta_ka.GT_EXT_CELLS")
    return cells


def gt_map(cells) -> dict:
    return MG.map_from_gt(cells, true_only=True)


def edge_quality(M: dict, cells) -> dict:
    """Map edges vs the GT cells: TRUE right sign / TRUE wrong sign / NULL / INDET / outside the GT universe."""
    st = {(c["family"], c["relation"], c["kpi"]): c for c in cells}
    q = {"true_right": [], "true_wrong": [], "null": [], "indet": [], "unknown": []}
    for h, b in sorted(M.items()):
        c = st.get(h)
        s = int(np.sign(b)) if b is not None else 0
        if c is None:
            q["unknown"].append(h)
        elif c["status"] == "TRUE":
            q["true_right" if s == int(np.sign(c["mean"])) else "true_wrong"].append(h)
        elif c["status"] == "NULL":
            q["null"].append(h)
        else:
            q["indet"].append(h)
    gt_true = {h for h, c in st.items() if c["status"] == "TRUE"}
    n_true = len(q["true_right"])
    return {"n_edges": len(M), **{f"n_{k}": len(v) for k, v in q.items()}, "edges": q,
            "precision_true_right": n_true / len(M) if M else float("nan"),
            "recall_true_right": n_true / len(gt_true) if gt_true else float("nan")}


# ---------------------------------------------------------------------------------------------- maps
def naive_slope(data: CU.UnitData, family: str, rel: str, kpi: str) -> float:
    """OLS slope of y(rel, kpi) on x = level * sgn over the family's units (associational effect size)."""
    rows = data.rows_of(family)
    if len(rows) < 3:
        return 0.0
    x, y = data.x[rows], data.y[(rel, kpi)][rows]
    xc = x - x.mean()
    den = float(xc @ xc)
    return float(xc @ (y - y.mean()) / den) if den > 0 else 0.0


def map_from_declared(decl: dict, data: CU.UnitData) -> tuple[dict, int]:
    """{(f, rel, kpi): sign * |naive slope|} over the declared hypotheses with a nonzero sign; returns (map, number of
    declared edges dropped for sign 0)."""
    M, drop = {}, 0
    for h, v in decl.items():
        h = tuple(h.split("|")) if isinstance(h, str) else tuple(h)
        if not v.get("declared"):
            continue
        s = int(v.get("sign", 0))
        if s == 0:
            drop += 1
            continue
        M[h] = float(s * abs(naive_slope(data, *h)))
    return M, drop


def map_from_mscr(run: dict) -> dict:
    """MSCR-CRT v2 declared edges -> {h: beta} (design-centred slope, "dir" orientation)."""
    return {h: float(v["beta"]) for h, v in V2.edges_from_crt_v2(run).items() if v["declared"] and v["sign"] != 0}


def placebo_tau(scores: dict, relations, max_fp: int = MAX_FP) -> dict:
    """tau = the (max_fp + 1)-th largest finite placebo score over ``relations`` (declare iff score > tau)."""
    s = sorted((v["score"] for h, v in scores.items() if (h.split("|") if isinstance(h, str) else h)[1] in relations
                and np.isfinite(v.get("score", np.nan))), reverse=True)
    if len(s) <= max_fp:
        return {"tau": float("-inf"), "max_fp": max_fp, "n_scorable": len(s),
                "note": "<= max_fp scorable placebo hypotheses: declares every scorable one"}
    return {"tau": float(s[max_fp]), "max_fp": max_fp, "n_scorable": len(s)}


def _tuple_scores(sc: dict) -> dict:
    return {tuple(k.split("|")) if isinstance(k, str) else k: v for k, v in sc.items()}


# ---------------------------------------------------------------------------------------------- replay
def replay(episodes, maps: dict, ref: str = "GT", pairs: dict | None = None, theta: float = MG.THETA,
           k_conf: int = MG.K_CONF, warm: float = 120.0) -> dict:
    """Offline MapGateV2 replay (module docstring). ``episodes``: records (units with c, knob, t0, ctx); ``maps``:
    {name: map}; ``pairs``: {comparison name: (map a, reference map b)} (default: every map vs ``ref``). A flip = a
    unit whose decision under a differs from b's; "by" keys = "<family><+|-|0>:refdefer->accept" (b defers, a
    accepts) or ":refaccept->defer". Returns {"n_units": [per episode], "decisions": {map: [defers per episode]},
    "defer_rate": {map}, "flips": {pair: {"per_episode", "pre_warm", "post_warm", "by": {key: [per episode]}}}}."""
    names = list(maps)
    if pairs is None:
        if ref not in names:
            raise ValueError(f"reference map {ref!r} missing")
        pairs = {m: (m, ref) for m in names if m != ref}
    for a, b in pairs.values():
        if a not in maps or b not in maps:
            raise ValueError(f"pair ({a}, {b}) names a missing map")
    out = {"n_units": [], "decisions": {m: [] for m in names}, "defer_rate": {},
           "flips": {p: {"per_episode": [], "by": {}, "pre_warm": [], "post_warm": []} for p in pairs}}
    for e, rec in enumerate(episodes):
        gates = {m: MG.MapGateV2(maps[m], theta, k_conf) for m in names}
        units = [u for u in rec["units"] if u["knob"] in MG.FAMILIES]
        dec = {m: np.zeros(len(units), bool) for m in names}
        for i, u in enumerate(units):
            view = {"c": int(u["c"]), "ctx": u["ctx"]}
            for m in names:
                dec[m][i] = gates[m](dict(view))[0] == "reject"
        out["n_units"].append(len(units))
        for m in names:
            out["decisions"][m].append(int(dec[m].sum()))
        pre = np.array([float(u["t0"]) < warm for u in units], bool)
        keys = [f"{u['knob']}{'+' if float(u['ctx']['step']) > 0 else '-' if float(u['ctx']['step']) < 0 else '0'}"
                for u in units]
        for p, (a, b) in pairs.items():
            diff = dec[a] != dec[b]
            fl = out["flips"][p]
            fl["per_episode"].append(int(diff.sum()))
            fl["pre_warm"].append(int((diff & pre).sum()))
            fl["post_warm"].append(int((diff & ~pre).sum()))
            for i in np.nonzero(diff)[0]:
                k = f"{keys[i]}:{'refdefer->accept' if dec[b][i] else 'refaccept->defer'}"
                fl["by"].setdefault(k, np.zeros(len(episodes), int))
                fl["by"][k][e] += 1
    for m in names:
        out["defer_rate"][m] = float(sum(out["decisions"][m]) / max(sum(out["n_units"]), 1))
    for fl in out["flips"].values():
        fl["by"] = {k: v.tolist() for k, v in sorted(fl["by"].items())}
    return out


def clean_map(M: dict, cells) -> dict:
    """M restricted to its GT-TRUE right-sign edges (the method's own magnitudes)."""
    keep = set(edge_quality(M, cells)["edges"]["true_right"])
    return {h: b for h, b in M.items() if h in keep}


def inject(M_ref: dict, M_add: dict) -> dict:
    """M_ref with M_add's edges added (a key in both takes M_add's value)."""
    return {**M_ref, **M_add}


def boot_ci(v, idx: int, n_boot: int = 2000, level: float = 0.90) -> list:
    v = np.asarray(v, float)
    if len(v) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng([BOOT_TAG, 1, int(idx)])
    m = v[rng.integers(0, len(v), (int(n_boot), len(v)))].mean(1)
    a = (1 - level) / 2
    return [float(np.quantile(m, a)), float(np.quantile(m, 1 - a))]


def flip_summary(rep: dict, n_boot: int) -> dict:
    out = {}
    for j, (m, fl) in enumerate(sorted(rep["flips"].items())):
        pe = np.asarray(fl["per_episode"], float)
        out[m] = {"mean_per_episode": float(pe.mean()) if len(pe) else float("nan"), "ci90": boot_ci(pe, j, n_boot),
                  "median": float(np.median(pe)) if len(pe) else float("nan"),
                  "share_units": float(pe.sum() / max(sum(rep["n_units"]), 1)),
                  "pre_warm_mean": float(np.mean(fl["pre_warm"])) if fl["pre_warm"] else float("nan"),
                  "post_warm_mean": float(np.mean(fl["post_warm"])) if fl["post_warm"] else float("nan"),
                  "by_mean": {k: float(np.mean(v)) for k, v in fl["by"].items()}}
    return out


def verdict(k0_pass: bool, flips: dict, dev_arms, pl_arms) -> dict:
    """Two labels. ``label`` = the rule AS SPECIFIED (flips of the method's applied map vs the GT map);
    ``label_attributable`` = the same with the flips a map's WRONG edges cause: max(false_edge = flips(M vs M restricted
    to its GT-TRUE right-sign edges), placebo_inject = flips(GT + the method's placebo edges vs GT)). The first also
    counts decisions lost by MISSING edges (power), so an empty map passes it; the second isolates confounding."""
    def best(arms, fmt):
        c = [(flips[fmt.format(a)]["mean_per_episode"], a) for a in arms
             if fmt.format(a) in flips and np.isfinite(flips[fmt.format(a)]["mean_per_episode"])]
        return max(c) if c else (float("nan"), None)

    def attributable(arms):
        c = [max(((flips[k]["mean_per_episode"], a) for k in (f"{a}~clean", f"GT+plc:{a}") if k in flips
                  and np.isfinite(flips[k]["mean_per_episode"])), default=(float("nan"), a)) for a in arms]
        c = [x for x in c if np.isfinite(x[0])]
        return max(c) if c else (float("nan"), None)

    def ok(b):
        return bool(np.isfinite(b[0]) and b[0] >= FLIP_MIN)
    b_dev, b_pl = best(dev_arms, "{}"), best(pl_arms, "{}")
    a_dev, a_pl = attributable(dev_arms), attributable(pl_arms)
    why, why_a = [], []
    if not k0_pass:
        why.append("MSCR K0 failed on the confounded placebo")
        why_a.append("MSCR K0 failed on the confounded placebo")
    if not ok(b_dev):
        why.append(f"no baseline map (DEV tau) flips >= {FLIP_MIN:g} MapGate decision / episode vs the GT map "
                   f"(best {b_dev[1]} {b_dev[0]:.2f})")
    if not ok(a_dev):
        why_a.append(f"no baseline's wrong edges (DEV tau) flip >= {FLIP_MIN:g} decision / episode (best {a_dev[1]} "
                     f"{a_dev[0]:.2f})")
    return {"label": "CONTINUE" if (k0_pass and ok(b_dev)) else "KILL", "why": why,
            "label_attributable": "CONTINUE" if (k0_pass and ok(a_dev)) else "KILL", "why_attributable": why_a,
            "k0_pass": bool(k0_pass),
            "best_dev_tau_vs_gt": {"arm": b_dev[1], "flips_per_episode": b_dev[0]},
            "best_placebo_tau_vs_gt": {"arm": b_pl[1], "flips_per_episode": b_pl[0]},
            "best_dev_tau_attributable": {"arm": a_dev[1], "flips_per_episode": a_dev[0]},
            "best_placebo_tau_attributable": {"arm": a_pl[1], "flips_per_episode": a_pl[0]},
            "rule": f"KILL iff MSCR K0 fails or max over baselines (DEV tau) of mean flips / episode < {FLIP_MIN:g} "
                    f"(label: vs the GT map, as specified; label_attributable: wrong-edge / placebo-edge flips)"}


# ---------------------------------------------------------------------------------------------- analyze
def analyze(a) -> dict:
    t_all = time.time()
    timing = {}
    allow = bool(a.get("allow_smoke"))
    H, H_pre = int(a.get("H", H_DEFAULT)), int(a.get("H_pre", H_PRE_DEFAULT))
    n_boot = int(a.get("n_boot", 2000))
    files = expand(a.get("records"))
    pl = read_episodes(files, lambda r: r.get("kb_stage") == "placebo", allow)
    ap = read_episodes(files, lambda r: r.get("kb_stage") == "applied", allow)
    dev_recs = read_episodes(expand(a.get("dev")), lambda r: r.get("stage") == "dev", allow)
    if not pl or not ap or not dev_recs:
        raise SystemExit(f"need placebo, applied and DEV records: {len(pl)} / {len(ap)} / {len(dev_recs)}")
    smoke = any(r.get("smoke") for rs in (pl, ap, dev_recs) for r in rs)
    cfg = V2.UnitCRTConfigV2(B=int(a.get("B", 9999)))
    sup = BD.Support()
    if smoke:                                    # plumbing only: tiny support so every code path runs
        cfg = dataclasses.replace(cfg, min_units=8, min_accept=2, min_reject=2, min_episodes=1)
        sup = BD.Support(min_units=8, min_accept=2, min_reject=2, min_episodes=1)
    rep = {"plan": PLAN, "banner": "K-B ANALYSIS" + (" -- SMOKE (NOT A VERDICT)" if smoke else ""),
           "episodes": {"placebo": len(pl), "applied": len(ap), "dev": len(dev_recs)}, "smoke": smoke,
           "H": H, "H_pre": H_pre, "config": dataclasses.asdict(cfg), "files": files,
           "policies": sorted({r.get("policy") for r in pl + ap})}
    log(f"== {rep['banner']}: episodes {rep['episodes']}; H {H} H_pre {H_pre}; B {cfg.B}")

    # unit tables
    t = time.time()
    pdat = CU.build_unit_data(pl, H=H, H_pre=H_pre)
    adat = CU.build_unit_data(ap, H=H, H_pre=H_pre)
    dev = CU.build_unit_data(dev_recs, H=H, H_pre=H_pre)
    del dev_recs
    timing["unit_tables"] = round(time.time() - t, 1)
    rep["units"] = {n: {"n": d.n, "probs_rows": d.meta["n_probs_rows"], "p_mismatch": d.meta["p_mismatch"],
                        "dropped": d.meta["dropped"],
                        "by_family": {f: int(len(d.rows_of(f))) for f in CU.FAMILIES},
                        "accept_share_by_family": {f: float((d.mode[d.rows_of(f)] == 0).mean())
                                                   if len(d.rows_of(f)) else None for f in CU.FAMILIES},
                        "warmup_units": int((d.t0 < 120).sum())}
                    for n, d in (("placebo", pdat), ("applied", adat), ("dev", dev))}
    for n in ("placebo", "applied"):
        u = rep["units"][n]
        if u["probs_rows"] != u["n"] or u["p_mismatch"]:
            raise SystemExit(f"{n}: {u['probs_rows']}/{u['n']} units with a logged row, p_mismatch {u['p_mismatch']}")
    log(f"units {rep['units']}")

    # K0 on the confounded placebo
    t = time.time()
    k0 = V2.placebo_rejection_units_v2(pdat, cfg, **K0_RULE)
    k0.pop("run", None)
    timing["k0"] = round(time.time() - t, 1)
    rep["K0"] = k0
    log(f"K0 (MSCR-CRT v2, confounded placebo): tested {k0['n_tested']}/{k0['n_hypotheses']}, rate {k0['rate']:.3f} "
        f"({k0['n_reject']} <= alpha, k_max {k0['k_max']}), BY {k0['n_by_declared']} {k0['declared']} -> "
        f"{'PASS' if k0['pass'] else 'FAIL'} [{timing['k0']} s]")

    # MSCR on the applied set
    t = time.time()
    mrun = V2.run_crt_units_v2(adat, cfg, CU.SPLIT_POOLED)
    timing["mscr_applied"] = round(time.time() - t, 1)
    rep["mscr_applied"] = {"n_declared": mrun["n_declared"], "n_tested": mrun["n_tested"],
                           "declared": [(r["family"], r["relation"], r["kpi"], r["sign"], r["p"], r["beta"],
                                         r["z_approx"]) for r in mrun["results"] if r["status"] == "declared"]}
    log(f"MSCR-CRT v2 applied: declared {mrun['n_declared']}/{mrun['n_tested']} [{timing['mscr_applied']} s]")

    # baselines: DEV tau (v1 / v2 path) + placebo-calibrated tau
    t = time.time()
    methods = tuple(a["methods"].split(",")) if a.get("methods") else BD.TUNED
    bl = BD.run_baselines(dev, {"placebo": pdat, "applied": adat}, methods, sup=sup, log=log,
                          ann=not a.get("no_ann", False))
    timing["baselines"] = round(time.time() - t, 1)

    cells = gt_cells()
    maps = {"GT": gt_map(cells), "MSCR": map_from_mscr(mrun)}
    plc_maps = {"MSCR": map_from_declared({tuple(x[:3]): {"declared": True, "sign": x[3]} for x in k0["declared"]},
                                          pdat)[0]}
    decl_counts, bl_rep, dev_arms, pl_arms = {"MSCR": {"placebo": k0["n_by_declared"],
                                                       "applied": mrun["n_declared"]}}, {}, [], []
    for m, r in bl.items():
        rels = r.get("decl_relations") or list(CU.RELATIONS)
        pd_ = r["splits"]["placebo"]["declared"]
        ad_ = r["splits"]["applied"]["declared"]
        entry = {"tau_dev": r.get("tau"), "note": r.get("note"), "cpu_s": r.get("cpu_s"),
                 "placebo_declared_dev_tau": sorted(("|".join(h), v["sign"]) for h, v in pd_.items() if v["declared"])}
        name = m if m == "granger_by" else f"{m}@dev"
        maps[name], entry["dropped_sign0_dev"] = map_from_declared(ad_, adat)
        plc_maps[name] = map_from_declared(pd_, pdat)[0]
        dev_arms.append(name)
        decl_counts[name] = {"placebo": len(entry["placebo_declared_dev_tau"]),
                             "applied": sum(v["declared"] for v in ad_.values())}
        if m != "granger_by":
            tp = placebo_tau(r["splits"]["placebo"]["scores"], rels)
            asc = _tuple_scores(r["splits"]["applied"]["scores"])
            psc = _tuple_scores(r["splits"]["placebo"]["scores"])
            ad_p = BD.declare(asc, tp["tau"], rels)
            pd_p = BD.declare(psc, tp["tau"], rels)
            entry["tau_placebo"] = tp
            nm = f"{m}@placebo"
            maps[nm], entry["dropped_sign0_placebo"] = map_from_declared(ad_p, adat)
            plc_maps[nm] = map_from_declared(pd_p, pdat)[0]
            pl_arms.append(nm)
            decl_counts[nm] = {"placebo": sum(v["declared"] for v in pd_p.values()),
                               "applied": sum(v["declared"] for v in ad_p.values())}
        bl_rep[m] = entry
    arms = [n for n in maps if n != "GT"]
    rep["baselines"] = bl_rep
    rep["declarations"] = decl_counts
    rep["maps"] = {n: sorted([*h, b] for h, b in M.items()) for n, M in maps.items()}
    rep["placebo_maps"] = {n: sorted([*h, b] for h, b in M.items()) for n, M in plc_maps.items()}
    rep["edge_quality"] = {n: edge_quality(M, cells) for n, M in maps.items()}
    rep["decision_tables"] = {n: {f"{k[0]}{'+' if k[1] > 0 else '-'}": v
                                  for k, v in MG.decision_table_v2(M, MG.THETA).items()} for n, M in maps.items()}
    gt_tab = rep["decision_tables"]["GT"]
    rep["table_diff_vs_GT"] = {n: {k: v for k, v in tab.items() if v != gt_tab[k]}
                               for n, tab in rep["decision_tables"].items() if n != "GT"}
    log("declarations (every placebo one is false):", decl_counts)

    # offline MapGateV2 replay on the applied episodes: vs GT, wrong-edge flips, placebo-edge injection
    t = time.time()
    all_maps, pairs = dict(maps), {}
    for n in arms:
        all_maps[f"{n}|clean"] = clean_map(maps[n], cells)
        all_maps[f"GT+plc:{n}"] = inject(maps["GT"], plc_maps.get(n, {}))
        pairs[n] = (n, "GT")
        pairs[f"{n}~clean"] = (n, f"{n}|clean")
        pairs[f"GT+plc:{n}"] = (f"GT+plc:{n}", "GT")
    rp = replay(ap, all_maps, pairs=pairs)
    timing["replay"] = round(time.time() - t, 1)
    fs = flip_summary(rp, n_boot)
    rep["replay"] = {"n_units_per_episode": float(np.mean(rp["n_units"])), "defer_rate": rp["defer_rate"],
                     "defers_per_episode": {m: float(np.mean(v)) for m, v in rp["decisions"].items()},
                     "pairs": pairs, "flips": fs, "raw": rp,
                     "note": "vs GT: decisions changed w.r.t. the GT map (wrong AND missing edges); ~clean: changed "
                             "by the map's wrong edges only; GT+plc: changed by the method's PLACEBO (pure "
                             "confounding) edges injected into the GT map"}
    rep["verdict"] = verdict(k0["pass"], fs, dev_arms, pl_arms)
    timing["total"] = round(time.time() - t_all, 1)
    rep["timing_s"] = timing
    rep["peak_rss_mb"] = peak_rss_mb()

    # print
    log("")
    log(f"GT map: {len(maps['GT'])} edges; decision table {gt_tab}")
    log(f"{'map':>18} {'#edg':>4} {'T+':>3} {'T-':>3} {'nul':>3} {'ind':>3} {'plc':>4} {'vs GT/ep':>8} {'90% CI':>15} "
        f"{'wrong/ep':>8} {'plcinj/ep':>9} {'defer/ep':>8}  table diff vs GT")

    def fm(k):
        f = fs.get(k)
        return f"{f['mean_per_episode']:8.2f}" if f else f"{'-':>8}"
    for n in maps:
        q = rep["edge_quality"][n]
        f = fs.get(n)
        ci = f"[{f['ci90'][0]:6.2f},{f['ci90'][1]:6.2f}]" if f else ""
        log(f"{n:>18} {q['n_edges']:4d} {q['n_true_right']:3d} {q['n_true_wrong']:3d} {q['n_null']:3d} "
            f"{q['n_indet']:3d} {decl_counts.get(n, {}).get('placebo', '-')!s:>4} {fm(n)} {ci:>15} "
            f"{fm(n + '~clean')} {fm('GT+plc:' + n):>9} {rep['replay']['defers_per_episode'][n]:8.2f}  "
            f"{rep['table_diff_vs_GT'].get(n, '')}")
    v = rep["verdict"]
    log(f"VERDICT (as specified: flips vs GT map): {v['label']}  {v['why']}")
    log(f"VERDICT (attributable: wrong / placebo-edge flips): {v['label_attributable']}  {v['why_attributable']}")
    log(f"   best DEV-tau vs GT {v['best_dev_tau_vs_gt']}; attributable {v['best_dev_tau_attributable']}; "
        f"placebo-tau vs GT {v['best_placebo_tau_vs_gt']}, attributable {v['best_placebo_tau_attributable']}")
    log(f"   MSCR: vs GT {fs['MSCR']['mean_per_episode']:.2f}/ep, wrong-edge {fs['MSCR~clean']['mean_per_episode']:.2f}"
        f"/ep, placebo-edge {fs['GT+plc:MSCR']['mean_per_episode']:.2f}/ep")
    log(f"timing (s): {timing}; peak RSS {rep['peak_rss_mb']} MB")
    if a.get("json"):
        os.makedirs(os.path.dirname(os.path.abspath(a["json"])), exist_ok=True)
        with open(a["json"], "w", newline="\n") as fh:
            json.dump(_keys(rep), fh, indent=1, default=_js)
        log(f"wrote {a['json']}")
    return rep


def cli(argv):
    if not argv or argv[0] != "analyze":
        raise SystemExit(__doc__)
    it = iter(argv[1:])
    a = {}
    flags = {"--allow-smoke": "allow_smoke", "--no-ann": "no_ann"}
    vals = {"--records": "records", "--dev": "dev", "--json": "json", "--B": "B", "--methods": "methods", "--H": "H",
            "--H-pre": "H_pre", "--n-boot": "n_boot"}
    for x in it:
        if x in flags:
            a[flags[x]] = True
        elif x in vals:
            a[vals[x]] = next(it)
        else:
            raise SystemExit(f"unknown argument {x}\n{__doc__}")
    return analyze(a)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])

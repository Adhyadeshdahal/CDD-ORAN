"""E6-P option (a) study ANALYZER (protocol docs/benchmark/E6P_CONFOUNDED_PROTOCOL.md sections 4, 7, 9, 12; driver
scratchpad/e6_dev/e6p_conf.py). ``disc`` and ``eval`` run as Kaggle jobs (scratchpad/e6_dev/kaggle_job.py); ``build``
and ``verify`` (the maps artifact, freeze 2) are light and run locally.

  python scratchpad/e6_dev/e6p_conf_analyze.py disc --dev SPEC --disc SPEC --placebo SPEC --gt SPEC [--out DIR]
         [--artifact docs/benchmark/artifacts/E6P_PMRT_V4.json] [--cache-dir DIR] [--B 9999] [--B-null 999]
         [--null-group 60] [--null-shifts 4] [--workers 4] [--methods shap_gbdt,corr,granger,two_tower,int,qacm]
         [--no-ann] [--no-v2] [--no-null] [--dev-step1 SPEC] [--dry-run] [--allow-smoke] [--allow-artifact-mismatch]
  python scratchpad/e6_dev/e6p_conf_analyze.py build --disc-json OUT/disc_conf.json
         [--out docs/benchmark/artifacts/E6P_CONF_MAPS.json] [--allow-dry]
  python scratchpad/e6_dev/e6p_conf_analyze.py verify [--artifact docs/benchmark/artifacts/E6P_CONF_MAPS.json]
         [--disc-json OUT/disc_conf.json] [--allow-dry]
  python scratchpad/e6_dev/e6p_conf_analyze.py eval --records SPEC --disc-json disc_conf.json
         [--maps docs/benchmark/artifacts/E6P_CONF_MAPS.json] [--out DIR] [--n-boot 10000] [--allow-smoke]
SPEC = comma list of JSONL files / directories (recursive res_*.jsonl or all.jsonl; /bundle/ skipped) / globs.

disc (section 9 step 4):
  checks     the v4 artifact (sha256 + embedded code sha256s, e6p_disc_analyze_v4.artifact_status; the params pickle
             that ``pmrt_artifacts.py verify`` needs is not in any bundle, so the sha checks stand in for it, as in
             the v4 analyzer); every dev_conf / disc / placebo record: every unit of crt_units.FAMILIES carries its own
             ``probs`` row (keys accept / reject, sums to 1, entries in [.15, .85], p = probs[mode]), pi0_table null,
             directional collection; UnitData.meta["p_mismatch"] == 0 on every pool (section 4.1 (a), (b)). Any
             failure stops the analysis (no maps).
  PMRT       v4 artifact (E6P_PMRT_V4.json), PRIMARY loadsp_c + wby1s; pooled DISC (split 0, n_target = #DISC episodes), placebo K0
             (split 9, n_target = #placebo episodes), K0n = the v4 recipe on DISC (groups of 60 consecutive episodes
             j // 60, 4 cyclic series shifts, B 999). K1 (>= 500 tested sleep units, >= 75 rejects), G, X1, X2 (kpi in
             pv / v / rlf / e, INDET excluded: precision >= .80, sign accuracy >= .90), X3 (chain >= min(3, |C*|)),
             discovery label (section 7.1 precedence).
  baselines  baselines_disc on the same unit table (H 90 / H_pre 90): @dev tau = far-FPR on dev_conf; @plc tau = the
             (MAX_FP + 1)-th largest finite placebo score over the declarable relations (K-B placebo_tau), except
             granger: n-free transfer (partial r^2 = F / (F + df); tau_r2 = the (MAX_FP + 1)-th largest placebo r^2,
             declare at DISC iff r^2 > tau_r2, sign unchanged); granger_by untuned.
  maps       (section 4.3) MG:PMRT = declared edges, beta = design-based slope sum v r / sum v^2 (v = PMRT design-
             centred treatment with the unit's own row, r = y - predictable running centre, eprocess_units.
             predictable_residuals; sign disagreements keep the PMRT sign, counted); MG:<b> = sign x |naive OLS
             slope| (K-B map_from_declared); MG:GT = TRUE edges of the fresh gt table ("dir"); MG:rand =
             random_sized_map(M_PMRT, gt cells, 6623, key 3); blanket2 = blanket_saving_map(). Signatures and the alias
             table (e6p_conf.alias_table; noarb alias iff the dev_conf repro check shows all-accept MapGateV2 == noarb
             bit for bit). Offline replay on the DISC logged contexts (K-B pairs vs GT, ~clean, GT+plc; bootstrap
             default_rng([6624, 10, pair_idx])). Descriptive: the 8 other PMRT combinations, MSCR-CRT v2 + BY,
             IPW-Wald and Granger+ctx (both BY q .05; discovery only), the @dev maps with the step-1 DEV tau
             (--dev-step1, offline replay only).
  output     OUT/disc_conf.json (everything, including every map and the discovery label). INVALID -> build refuses.
build: the maps artifact (schema e6p-conf-maps/1) from disc_conf.json: every EVAL map arm (map, provenance,
  decision_table_v2, signature, alias target), jobs, MapGateV2 constants + mapgate.py sha256, sha256 of disc_conf.json,
  of this analyzer, of the driver and of the v4 artifact; + a .sha256 file. verify: re-derives signatures / aliases,
  checks every sha256 and prints the MAPS_SHA256 line for e6p_conf.py.
eval (section 7.2): Gate A stats (e6p_step2_dev definitions) of every arm on ONE paired bootstrap index matrix
  (N_BOOT 10000, default_rng([6624, 20, n_seeds])) over the seeds with every arm (aliased arms expanded from their
  simulated target); eligibility held at its point value; R* = R if eligible else min(R, 0). E, D1 (Delta1 >= .10 and
  LB90 > 0, the best associational map RE-SELECTED per resample), D2 (one-sided p per distinct associational
  signature, Holm alpha .05), never_sleep report (R(PMRT) - R(never_sleep), 90 % CI, non-inferiority LB90 > -.10),
  secondary contrasts, verdict precedence INVALID > NOT ELIGIBLE > PASS / PARTIAL / FAIL; "NOT A VERDICT" unless the
  data are complete and every sha256 matches. Output OUT/eval_conf.json + OUT/verdict_conf.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e6p_conf as C  # noqa: E402
import e6p_disc_analyze_v4 as A4  # noqa: E402
import e6p_opta_kb_analyze as KBA  # noqa: E402
import numpy as np  # noqa: E402

from cdd_oran.decision import baselines_disc as BD  # noqa: E402
from cdd_oran.decision import disc_bench as DB  # noqa: E402
from cdd_oran.decision import mapgate as MG  # noqa: E402
from cdd_oran.decision.crt_units import FAMILIES, KPIS, RELATIONS  # noqa: E402

SCHEMA_DISC = "e6p-conf-disc/1"
SCHEMA_MAPS = "e6p-conf-maps/1"
SCHEMA_EVAL = "e6p-conf-eval/1"
ANALYZER = "scratchpad/e6_dev/e6p_conf_analyze.py"
DRIVER = "scratchpad/e6_dev/e6p_conf.py"
MAPGATE = "cdd_oran/decision/mapgate.py"
V4_ARTIFACT = A4.PMRT_ARTIFACT                 # PMRT v4 artifact (label-only successor of the frozen v4 one)
V4_ARTIFACT_SHA256 = A4.PMRT_ARTIFACT_SHA256
PRIMARY = A4.PRIMARY
H = H_PRE = 90
SPLIT = {"pooled": 0, "placebo": 9, "null": 0}
MAX_FP = 1
INC_FLOOR = 0.15
K0_RULE, K0N_RULE = A4.K0_RULE, A4.K0N_RULE
K1_RULE = {"min_sleep_units": 500, "min_sleep_rejects": 75}      # fixed at freeze 1 (implementation note 9; dev_conf projection 2340 / 780)
X2_RULE = {"kpis": ("pv", "v", "rlf", "e"), "precision": 0.80, "sign_acc": 0.90}
X3_RULE = {"chain_min": 3}
DISC_BOOT_KEY = (6624, 10)
EVAL_BOOT_TAG, EVAL_BOOT_KEY = 6624, 20
N_BOOT = 10_000
D1_MARGIN = 0.10
ALPHA_D2 = 0.05
NONINF = -0.10
SECONDARY = ("noarb", "incumbent", "blanket2", "never_sleep", "B2", "MG:GT", "MG:rand")
PRIMARY_ARM = "MG:PMRT"
TUNED = BD.TUNED
log = A4.log


# ============================================================================================ record checks (sec. 4.1)
def check_unit_row(u: dict) -> str | None:
    """None if the unit carries a valid logged incumbent row, else the reason."""
    pr = u.get("probs")
    if pr is None:
        return "missing"
    if set(pr) - {"accept", "reject"}:
        return "keys"
    v = [float(x) for x in pr.values()]
    if abs(sum(v) - 1.0) > 1e-9:
        return "sum"
    if min(v) < INC_FLOOR - 1e-9 or max(v) > 1.0 - INC_FLOOR + 1e-9:
        return "floor"
    if u.get("mode") not in pr or abs(float(pr[u["mode"]]) - float(u["p"])) > 1e-9:
        return "p"
    return None


def check_episode(rec: dict) -> dict:
    """Section 4.1 (a) on one collection record: per-unit rows on every unit of FAMILIES, pi0_table null."""
    bad = {}
    n = 0
    for u in rec["units"]:
        if u["knob"] not in FAMILIES:
            continue
        n += 1
        why = check_unit_row(u)
        if why is not None:
            bad[why] = bad.get(why, 0) + 1
    return {"units": n, "bad": bad, "pi0_table_null": rec.get("pi0_table") is None,
            "directional": bool(rec.get("directional")), "ok": not bad and rec.get("pi0_table") is None}


KEEP = ("kind", "schema", "key", "stage", "sub", "conf_stage", "seed", "j", "fold", "smoke", "policy", "pi0_table",
        "n_cells", "units", "lab_series")


def scan_files(files: list, conf_stage: str, keep: bool, allow_smoke: bool) -> dict:
    """Stream ``files``: every collection record of ``conf_stage`` (sub conf) is checked; ``keep`` keeps a light copy
    (KEEP keys, lab_kpi dropped). Also returns the headers' numeric env and the dev_repro / eval job records."""
    recs, chk, headers, jobs = {}, {}, [], []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if '"episode"' not in line[:200]:
                    if '"header"' in line[:200] or '"job"' in line[:200]:
                        r = json.loads(line)
                        if r.get("kind") == "header" and r.get("driver") == "e6p_conf":
                            headers.append({"conf_stage": r.get("conf_stage"), "smoke": r.get("smoke"),
                                            "platform": r.get("platform"), "freeze": r.get("freeze"),
                                            "mapgate_sha256": r.get("mapgate_sha256"), "maps": r.get("maps")})
                        elif (r.get("kind") == "job" and r.get("sub") == C.SUB
                              and (allow_smoke or not r.get("smoke"))):
                            jobs.append(r)
                    continue
                r = json.loads(line)
                if (r.get("kind") != "episode" or r.get("sub") != C.SUB or r.get("conf_stage") != conf_stage
                        or (r.get("smoke") and not allow_smoke)):
                    continue
                key = (int(r["seed"]), bool(r.get("smoke")))
                chk[key] = check_episode(r)
                if keep:
                    for u in r["units"]:
                        u.pop("lab_kpi", None)
                    recs[key] = {k: r[k] for k in KEEP if k in r}
    rows = [chk[k] for k in sorted(chk)]
    bad = {}
    for c in rows:
        for w, v in c["bad"].items():
            bad[w] = bad.get(w, 0) + v
    return {"records": [recs[k] for k in sorted(recs)], "headers": headers, "jobs": jobs,
            "check": {"episodes": len(rows), "units": int(sum(c["units"] for c in rows)), "bad_units": bad,
                      "pi0_table_nonnull": int(sum(not c["pi0_table_null"] for c in rows)),
                      "not_directional": int(sum(not c["directional"] for c in rows)),
                      "seeds": sorted({k[0] for k in chk}), "smoke": any(k[1] for k in chk),
                      "ok": bool(rows) and all(c["ok"] for c in rows)}}


def platform_key(env: dict | None) -> tuple:
    """Numerics-relevant part of a header's numeric env: numpy, scipy, OS / glibc, machine, SIMD."""
    env = env or {}
    plat = str(env.get("platform", ""))
    return (env.get("numpy"), env.get("scipy"), plat.split("-")[0], plat.split("-with-")[-1] if "-with-" in plat
            else None, env.get("machine"), tuple(env.get("simd") or ()))


# ============================================================================================ maps (sec. 4.3)
def design_slope(ud, f: str, rel: str, kpi: str) -> dict:
    """PMRT map beta: sum v r / sum v^2 over the family's tested units in information order; v = sgn (L(mode) -
    probs . L) with the unit's own row, r = y(rel, kpi) - its predictable running centre (the PMRT "pred" residual,
    eprocess_units.predictable_residuals). Design-unbiased for a linear level effect (E[v c] = 0 for predictable c)."""
    from cdd_oran.decision.eprocess_units import EProcConfig, predictable_residuals, unit_order
    from cdd_oran.decision.pmrt import v_design
    rows = unit_order(ud, ud.rows_of(f))
    if len(rows) < 2:
        return {"beta": 0.0, "n": int(len(rows)), "vv": 0.0}
    v = v_design(ud.mode[rows], ud.probs[rows], ud.sgn[rows])
    y, pre = ud.y[(rel, kpi)][rows], ud.pre[(rel, kpi)][rows]
    r, _ = predictable_residuals(y, pre if np.all(np.isfinite(pre)) else None, ud.episode[rows], ud.sgn[rows],
                                 EProcConfig())
    vv = float(v @ v)
    return {"beta": float(v @ r / vv) if vv > 0 else 0.0, "n": int(len(rows)), "vv": vv}


def pmrt_map(decl: dict, ud) -> tuple[dict, list]:
    """Declared (nonzero-sign) edges of the primary layer -> {h: sign * |design slope|}; the list of sign
    disagreements (h, declared sign, slope)."""
    M, dis = {}, []
    for h, v in sorted(decl.items()):
        if not v.get("declared") or int(v.get("sign", 0)) == 0:
            continue
        s = int(v["sign"])
        b = design_slope(ud, *h)["beta"]
        if b != 0 and int(np.sign(b)) != s:
            dis.append([*h, s, b])
        M[tuple(h)] = float(s * abs(b))
    return M, dis


def granger_stats(ud, sup=BD.SUPPORT) -> dict:
    """baselines_disc.granger_pvalues' regression (post = a + b pre + c x) with F, df and partial r^2 =
    F / (F + df) of c (the n-free transfer statistic of section 4.2)."""
    from scipy.stats import f as fdist
    out = {}
    for f in FAMILIES:
        if not BD.supported(ud, f, sup):
            continue
        rows = ud.rows_of(f)
        for r in RELATIONS:
            for k in KPIS:
                pre = ud.pre[(r, k)][rows]
                post = ud.y[(r, k)][rows] + pre
                x = ud.x[rows]
                n = len(rows)
                A = np.column_stack([np.ones(n), pre, x])
                if np.std(post) == 0 or np.linalg.matrix_rank(A) < 3:
                    continue
                beta, *_ = np.linalg.lstsq(A, post, rcond=None)
                res = post - A @ beta
                s2 = float(res @ res) / (n - 3)
                cov = s2 * np.linalg.pinv(A.T @ A)
                se = float(np.sqrt(max(cov[2, 2], 0.0)))
                if se <= 0:
                    continue
                F = float((beta[2] / se) ** 2)
                df = n - 3
                out[(f, r, k)] = {"F": F, "df": df, "r2": F / (F + df), "sign": int(np.sign(beta[2])),
                                  "p": float(fdist.sf(F, 1, df))}
    return out


def granger_plc_transfer(plc: dict, disc: dict, relations=RELATIONS, max_fp: int = MAX_FP) -> tuple[dict, dict]:
    """n-free placebo tau for granger (section 4.2): tau_r2 = the (max_fp + 1)-th largest placebo partial r^2 over
    ``relations``; DISC declares iff r^2 > tau_r2 (sign unchanged). Returns (decl over HYPOTHESES, tau info)."""
    from cdd_oran.decision.edge_score import HYPOTHESES
    s = sorted((v["r2"] for h, v in plc.items() if h[1] in relations and np.isfinite(v["r2"])), reverse=True)
    if len(s) <= max_fp:
        tau = {"tau_r2": float("-inf"), "max_fp": max_fp, "n_scorable": len(s),
               "note": "<= max_fp scorable placebo hypotheses: declares every scorable one"}
    else:
        tau = {"tau_r2": float(s[max_fp]), "max_fp": max_fp, "n_scorable": len(s)}
    decl = {}
    for h in HYPOTHESES:
        v = disc.get(h)
        ok = v is not None and h[1] in relations and np.isfinite(v["r2"]) and v["r2"] > tau["tau_r2"]
        decl[h] = {"declared": bool(ok), "sign": int(v["sign"]) if v else 0, "score": v["r2"] if v else float("nan")}
    return decl, tau


def n_declared(decl: dict) -> int:
    return int(sum(bool(v.get("declared")) for v in decl.values() if v))


def decl_list(decl: dict) -> list:
    return sorted(["|".join(h), int(v.get("sign", 0))] for h, v in decl.items() if v and v.get("declared"))


# ============================================================================================ descriptive competitors
def _by(p, q=0.05):
    from cdd_oran.discovery.mscr import by_declare
    return by_declare(list(p), q) if len(p) else []


def ipw_wald(ud, sup=BD.SUPPORT, q: float = 0.05) -> dict:
    """Descriptive IPW-Wald test with the logged propensities (section 7.4): per (f, rel, kpi), psi_u = sgn_u y_u
    (A_u / m_u - (1 - A_u) / (1 - m_u)) (A = LEVEL_V2 of the logged mode, m = probs . LEVEL_V2), estimate = mean psi,
    episode-cluster SE, two-sided normal p; BY at q over the tested hypotheses."""
    from scipy.stats import norm

    from cdd_oran.decision.crt_units_v2 import LEVEL_V2_ARR
    from cdd_oran.decision.edge_score import HYPOTHESES
    out, keys, ps = {h: {"declared": False, "sign": 0, "p": float("nan")} for h in HYPOTHESES}, [], []
    for f in FAMILIES:
        if not BD.supported(ud, f, sup):
            continue
        rows = ud.rows_of(f)
        m = ud.probs[rows] @ LEVEL_V2_ARR
        A = LEVEL_V2_ARR[ud.mode[rows]]
        ok = (m > 0) & (m < 1)
        w = np.where(ok, A / np.where(ok, m, 1) - (1 - A) / np.where(ok, 1 - m, 1), 0.0)
        ep = ud.episode[rows]
        eps = np.unique(ep)
        for r in RELATIONS:
            for k in KPIS:
                psi = ud.sgn[rows] * ud.y[(r, k)][rows] * w
                if not np.all(np.isfinite(psi)):
                    continue
                est = float(psi.mean())
                cl = np.array([np.sum(psi[ep == e] - est) for e in eps])
                se = float(np.sqrt(np.sum(cl ** 2)) / len(psi))
                if se <= 0:
                    continue
                z = est / se
                p = float(2 * norm.sf(abs(z)))
                out[(f, r, k)] = {"declared": False, "sign": int(np.sign(est)), "p": p, "est": est, "z": z}
                keys.append((f, r, k))
                ps.append(p)
    for h, d in zip(keys, _by(ps, q), strict=True):
        out[h]["declared"] = bool(d)
    return out


def granger_ctx(ud, sup=BD.SUPPORT, q: float = 0.05) -> dict:
    """Descriptive Granger with ctx conditioners (section 7.4): post = a + b pre + c x + G z_ctx (every numeric obs-only
    ctx column of the unit table, standardised, NaN -> 0 + a missing indicator), F test of c, BY at q."""
    from scipy.stats import f as fdist

    from cdd_oran.decision.edge_score import HYPOTHESES
    out, keys, ps = {h: {"declared": False, "sign": 0, "p": float("nan")} for h in HYPOTHESES}, [], []
    ctx_cols = [c for c in ud.z if c.startswith("ctx_") and c not in ("ctx_step", "ctx_cur", "ctx_prop")]
    for f in FAMILIES:
        if not BD.supported(ud, f, sup):
            continue
        rows = ud.rows_of(f)
        Z = []
        for c in ctx_cols:
            v = np.asarray(ud.z[c][rows], float)
            fin = np.isfinite(v)
            if fin.sum() < 2 or np.nanstd(v) == 0:
                continue
            vz = np.where(fin, (v - np.nanmean(v)) / np.nanstd(v), 0.0)
            Z.append(vz)
            if not fin.all():
                Z.append((~fin).astype(float))
        Z = np.column_stack(Z) if Z else np.zeros((len(rows), 0))
        for r in RELATIONS:
            for k in KPIS:
                pre = ud.pre[(r, k)][rows]
                post = ud.y[(r, k)][rows] + pre
                A = np.column_stack([np.ones(len(rows)), pre, ud.x[rows], Z])
                n, p_ = A.shape
                if n - p_ < 5 or np.std(post) == 0:
                    continue
                beta, *_ = np.linalg.lstsq(A, post, rcond=None)
                res = post - A @ beta
                s2 = float(res @ res) / (n - p_)
                se = float(np.sqrt(max((s2 * np.linalg.pinv(A.T @ A))[2, 2], 0.0)))
                if se <= 0:
                    continue
                F = float((beta[2] / se) ** 2)
                p = float(fdist.sf(F, 1, n - p_))
                out[(f, r, k)] = {"declared": False, "sign": int(np.sign(beta[2])), "p": p, "F": F}
                keys.append((f, r, k))
                ps.append(p)
    for h, d in zip(keys, _by(ps, q), strict=True):
        out[h]["declared"] = bool(d)
    return out


# ============================================================================================ criteria (sec. 7.1)
def x_checks(decl: dict, ref: dict) -> dict:
    """X1 (premise declared +), X2 (referee-relevant precision / sign accuracy, INDET excluded), X3 (chain)."""
    prem = decl.get(DB.PREMISE) or {}
    x1 = bool(prem.get("declared")) and int(prem.get("sign", 0)) == 1
    tp = fp = sign_ok = n_true = 0
    rows = []
    for h, v in sorted(decl.items()):
        if not v or not v.get("declared") or h[2] not in X2_RULE["kpis"]:
            continue
        g = ref.get(tuple(h), {"status": "INDET", "sign": 0})
        s = int(v.get("sign", 0))
        rows.append(["|".join(h), s, g["status"], int(g["sign"])])
        if g["status"] == "INDET":
            continue
        if g["status"] == "TRUE":
            n_true += 1
            sign_ok += int(s == int(g["sign"]))
            if s == int(g["sign"]):
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    sacc = sign_ok / n_true if n_true else float("nan")
    x2 = bool(np.isfinite(prec) and prec >= X2_RULE["precision"] - 1e-12 and np.isfinite(sacc)
              and sacc >= X2_RULE["sign_acc"] - 1e-12)
    m = DB.subset_metrics(decl, ref)
    need = min(X3_RULE["chain_min"], int(m["chain_true"]))
    x3 = bool(m["chain_hits"] >= need)
    return {"X1": {"pass": x1, "premise": {k: prem.get(k) for k in ("declared", "sign", "p")}},
            "X2": {"pass": x2, "precision": prec, "sign_accuracy": sacc, "n_scored": tp + fp, "n_true": n_true,
                   "edges": rows, "rule": {k: list(v) if isinstance(v, tuple) else v for k, v in X2_RULE.items()}},
            "X3": {"pass": x3, "chain_hits": int(m["chain_hits"]), "chain_true": int(m["chain_true"]), "need": need,
                   "chain": m["chain"]}}


def k1_check(units: dict, rule=K1_RULE) -> dict:
    ok = units["sleep_units"] >= rule["min_sleep_units"] and units["sleep_rejects"] >= rule["min_sleep_rejects"]
    return {"pass": bool(ok), "sleep_units": units["sleep_units"], "sleep_rejects": units["sleep_rejects"],
            "rule": dict(rule), "note": "provisional thresholds (protocol section 7.1 / 13.5)"}


def disc_label(k0, k0n, g, k1, x) -> dict:
    """Section 7.1 precedence: INVALID > NO-CHAIN > UNDERPOWERED > DISC-PASS / DISC-PARTIAL / DISC-FAIL."""
    if not (k0 and k0["pass"]) or not (k0n and k0n["pass"]):
        lab = "INVALID"
    elif not (g and g["pass"]):
        lab = "NO-CHAIN"
    elif not (k1 and k1["pass"]):
        lab = "UNDERPOWERED"
    elif x["X1"]["pass"] and x["X2"]["pass"] and x["X3"]["pass"]:
        lab = "DISC-PASS"
    elif x["X2"]["pass"]:
        lab = "DISC-PARTIAL"
    else:
        lab = "DISC-FAIL"
    return {"label": lab, "stops_before_eval": lab == "INVALID",
            "precedence": "INVALID (K0 or K0n) > NO-CHAIN (G) > UNDERPOWERED (K1) > DISC-PASS (X1 X2 X3) / "
                          "DISC-PARTIAL (X2) / DISC-FAIL"}


# ============================================================================================ replay summary
def flip_summary(rep: dict, n_boot: int = 2000) -> dict:
    """K-B flip_summary with the study's bootstrap key default_rng([6624, 10, pair_idx])."""
    out = {}
    for j, (m, fl) in enumerate(sorted(rep["flips"].items())):
        pe = np.asarray(fl["per_episode"], float)
        ci = [float("nan"), float("nan")]
        if len(pe) >= 2:
            rng = np.random.default_rng([*DISC_BOOT_KEY, j])
            mm = pe[rng.integers(0, len(pe), (int(n_boot), len(pe)))].mean(1)
            ci = [float(np.quantile(mm, .05)), float(np.quantile(mm, .95))]
        out[m] = {"mean_per_episode": float(pe.mean()) if len(pe) else float("nan"), "ci90": ci,
                  "pre_warm_mean": float(np.mean(fl["pre_warm"])) if fl["pre_warm"] else float("nan"),
                  "post_warm_mean": float(np.mean(fl["post_warm"])) if fl["post_warm"] else float("nan"),
                  "by_mean": {k: float(np.mean(v)) for k, v in fl["by"].items()}}
    return out


# ============================================================================================ disc
def data_status(checks: dict, gt_seeds: list, dry: bool) -> dict:
    exp = {"dev_conf": "dev_conf", "disc": "disc", "placebo": "placebo"}
    st = {}
    for name, stage in exp.items():
        c = checks.get(name)
        want = C.stage_seeds(stage)
        st[name] = {"ok": bool(c and c["seeds"] == want and not c["smoke"] and c["ok"]), "n": len(c["seeds"]) if c
                    else 0, "expected": [want[0], want[-1]], "smoke": bool(c and c["smoke"])}
    want = C.stage_seeds("gt")
    st["gt"] = {"ok": sorted(int(s) for s in gt_seeds) == want, "n": len(gt_seeds), "expected": [want[0], want[-1]]}
    st["all_ok"] = all(v["ok"] for v in st.values() if isinstance(v, dict)) and not dry
    return st


def cmd_disc(a) -> dict:
    import pmrt_artifacts as PAR
    import pmrt_bench as PB

    from cdd_oran.decision import pmrt as PM
    t_all = time.time()
    out = a.out or os.environ.get("JOB_OUT") or "."
    os.makedirs(out, exist_ok=True)
    cache_dir = a.cache_dir or os.path.join(out, "cache")
    timing = {}
    rep = {"schema": SCHEMA_DISC, "argv": sys.argv[1:], "protocol": C.freeze_status(ROOT), "primary": "+".join(PRIMARY),
           "code_sha256": {p: A4.sha_lf(os.path.join(ROOT, p)) for p in (ANALYZER, DRIVER, MAPGATE)}}
    # ---- artifact
    ast = A4.artifact_status(a.artifact)
    rep["artifact"] = ast
    log(f"v4 artifact sha ok {ast['sha_ok']}, code ok {ast['code_ok']} {[k for k, v in ast['code'].items() if not v['ok']]}")
    if not (ast["sha_ok"] and ast["code_ok"]) and not a.allow_artifact_mismatch:
        raise SystemExit("v4 artifact sha256 / code sha256 mismatch (--allow-artifact-mismatch for a dry run only)")
    params, priors = PAR.load_artifact(a.artifact)
    cfg, cfg_null = PAR.pmrt_config(params, a.B), PAR.pmrt_config(params, a.B_null)
    # ---- records: section 4.1 (a) checks + light DISC records (K0n, replay)
    t = time.time()
    files = {k: A4.expand(getattr(a, k)) for k in ("dev", "disc", "placebo", "gt")}
    scans = {"dev_conf": scan_files(files["dev"], "dev_conf", False, a.allow_smoke),
             "disc": scan_files(files["disc"], "disc", True, a.allow_smoke),
             "placebo": scan_files(files["placebo"], "placebo", False, a.allow_smoke)}
    rep["record_checks"] = {k: v["check"] for k, v in scans.items()}
    timing["scan"] = round(time.time() - t, 1)
    log(f"record checks: { {k: (v['episodes'], v['bad_units'], v['pi0_table_nonnull'], v['ok']) for k, v in rep['record_checks'].items()} }")
    bad = {k: v for k, v in rep["record_checks"].items() if not v["ok"]}
    if bad:
        raise SystemExit(f"section 4.1 (a): records without a valid per-unit row / with a pi0_table: {bad}")
    recs = scans["disc"]["records"]
    plats = {platform_key(h["platform"]) for s in scans.values() for h in s["headers"] if not h["smoke"]}
    rep["platform"] = {"keys": sorted(map(list, plats), key=str), "equal": len(plats) == 1,
                       "linux": all(p[2] == "Linux" for p in plats) and bool(plats)}
    # ---- repro (dev_conf repro jobs) -> noarb alias
    arms_r = {}
    for r in scans["dev_conf"]["jobs"]:
        if r.get("conf_stage") == "dev_repro":
            arms_r.setdefault(int(r["seed"]), {})[r["arm"]] = r
    coll = dev_outcomes(files["dev"], a.allow_smoke)
    rep["repro"] = C.repro_check(coll, arms_r)
    noarb_alias = bool(rep["repro"]["noarb_alias"])
    log(f"repro: {json.dumps(rep['repro'], default=A4._js)}")
    # ---- caches / pools
    t = time.time()
    stage_of = {"dev": "dev", "disc": "eval", "placebo": "placebo"}
    caches = {k: A4.build_caches(files[k], stage_of[k], os.path.join(cache_dir, k), k, a.workers)
              for k in ("dev", "disc", "placebo")}
    timing["caches"] = round(time.time() - t, 1)
    pool = PM.load_pmrt_pool(caches["disc"], stages={"eval"})
    ppool = PM.load_pmrt_pool(caches["placebo"], stages={"placebo"})
    dpool = DB.load_pool(caches["dev"], H=H, H_pre=H_PRE, stages={"dev"})
    for nm, p in (("disc", pool), ("placebo", ppool), ("dev", dpool)):
        subs = sorted(set(str(s) for s in p.eps["sub"]))
        if subs != [C.SUB]:
            raise SystemExit(f"{nm}: episodes of subs {subs} (only '{C.SUB}' allowed)")
        if not a.allow_smoke and bool(np.any(p.eps["smoke"])):
            raise SystemExit(f"{nm}: smoke records (pass --allow-smoke for a dry run)")
    pdisc, pplc = PM.pmrt_data(pool), PM.pmrt_data(ppool)
    dud = dpool.unit_data()
    rep["units"] = {"disc": A4.unit_counts(pdisc.ud), "placebo": A4.unit_counts(pplc.ud), "dev": A4.unit_counts(dud)}
    pm = {"disc": pdisc.ud.meta["p_mismatch"], "placebo": pplc.ud.meta["p_mismatch"], "dev": dud.meta["p_mismatch"]}
    rep["p_mismatch"] = pm
    if any(pm.values()):
        raise SystemExit(f"section 4.1 (b): UnitData p_mismatch != 0: {pm}")
    log(f"units {rep['units']['disc']['units']} disc / {rep['units']['placebo']['units']} placebo / "
        f"{rep['units']['dev']['units']} dev; p_mismatch {pm} [{timing['caches']} s]")
    # ---- GT
    t = time.time()
    g = DB.gt_reference_files(files["gt"])
    ref, cells = g["ref"], g["cells"]
    G = A4.g_check(cells)
    rep["gt"] = {k: g[k] for k in ("n_episodes", "n_labels", "labels_per_family", "counts", "nbr_true", "seeds",
                                   "delta", "cells") if k in g}
    rep["G"] = G
    timing["gt"] = round(time.time() - t, 1)
    dry = bool(a.dry_run)
    rep["data"] = data_status(rep["record_checks"], g.get("seeds", []), dry)
    log(f"GT {g['n_episodes']} eps, {g['n_labels']} labels; G {G['pass']}; data {rep['data']}")
    # ---- PMRT placebo (K0)
    t = time.time()
    n_plc, n_disc = int(ppool.n_eps), int(pool.n_eps)
    run_p = PB.run_integrated(pplc, params, cfg, SPLIT["placebo"])
    decl_p_all = PB.declare_all(run_p, priors, n_plc)
    K0 = A4.k0_check(decl_p_all[PRIMARY])
    rep["K0"] = K0
    rep["placebo_hyp"] = A4.hyp_table(run_p, decl_p_all[PRIMARY])
    rep["placebo_combos"] = {PB.cname(*k): A4.k0_check(d) for k, d in decl_p_all.items()}
    timing["k0"] = round(time.time() - t, 1)
    log(f"K0 {K0['pass']}: m {K0['m']} reject {K0['n_reject']} tail {K0['binom_tail']:.3g} declared "
        f"{K0['declared']} [{timing['k0']} s]")
    # ---- PMRT pooled DISC
    t = time.time()
    run = PB.run_integrated(pdisc, params, cfg, SPLIT["pooled"])
    decls_all = PB.declare_all(run, priors, n_disc)
    decl = decls_all[PRIMARY]
    rep["disc_hyp"] = A4.hyp_table(run, decl)
    rep["pmrt_combos"] = {PB.cname(*k): {"metrics": DB.subset_metrics(d, ref), "declared": decl_list(d)}
                          for k, d in decls_all.items()}
    timing["pmrt_disc"] = round(time.time() - t, 1)
    log(f"PMRT DISC ({n_disc} eps): declared {decl_list(decl)} [{timing['pmrt_disc']} s]")
    # ---- K0n
    K0n = None
    if not a.no_null:
        t = time.time()
        seeds = [int(r["seed"]) for r in recs]
        b0 = C.LAYOUT["disc"][0]
        if dry or not all(C.LAYOUT["disc"][0] <= s < b0 + C.LAYOUT["disc"][1] for s in seeds):
            order = list(range(len(recs)))
            groups = [order[i:i + a.null_group] for i in range(0, len(order) - a.null_group + 1, a.null_group)]
        else:
            groups = [[i for i, s in enumerate(seeds) if (s - b0) // a.null_group == gi]
                      for gi in range(C.LAYOUT["disc"][1] // a.null_group)]
        kn = A4.run_k0n(recs, groups, params, priors, cfg_null, a.null_shifts)
        K0n = kn["primary"]
        rep["K0n_descriptive"] = {"variants": kn["variants"], "combos": kn["combos"], "groups": len(groups)}
        timing["k0n"] = round(time.time() - t, 1)
        log(f"K0n {K0n['pass']}: {K0n['n_variants']} variants, rate {K0n['rate05']} [{timing['k0n']} s]")
    rep["K0n"] = K0n
    # ---- K1 / X / label
    K1 = k1_check(rep["units"]["disc"])
    X = x_checks(decl, ref)
    rep.update(K1=K1, X=X, label=disc_label(K0, K0n, G, K1, X))
    log(f"K1 {K1['pass']}  X1 {X['X1']['pass']}  X2 {X['X2']['pass']} (prec {X['X2']['precision']}, sign "
        f"{X['X2']['sign_accuracy']})  X3 {X['X3']['pass']}  -> discovery label {rep['label']['label']}")
    # ---- baselines
    t = time.time()
    methods = tuple(a.methods.split(",")) if a.methods else TUNED
    sup = BD.Support()
    splits = {"placebo": pplc.ud, "disc": pdisc.ud}
    bl, bl_err = {}, {}
    for m in methods:
        try:
            bl.update(BD.run_baselines(dud, splits, (m,), sup=sup, log=log, ann=not a.no_ann))
        except Exception as e:                                                 # noqa: BLE001
            bl_err[m] = repr(e)[:400]
            log(f"baseline {m}: FAILED {bl_err[m][:160]}")
    timing["baselines"] = round(time.time() - t, 1)
    # ---- maps
    t = time.time()
    maps, prov, plc_maps, bl_rep = {}, {}, {}, {}
    M_pmrt, disagree = pmrt_map(decl, pdisc.ud)
    maps["MG:PMRT"] = M_pmrt
    prov["MG:PMRT"] = {"method": "PMRT " + "+".join(PRIMARY), "declared": decl_list(decl),
                        "beta": "design-based slope (section 4.3)", "sign_disagreements": disagree}
    plc_maps["MG:PMRT"] = KBA.map_from_declared(decl_p_all[PRIMARY], pplc.ud)[0]
    maps["MG:GT"] = MG.map_from_gt(cells, true_only=True)
    prov["MG:GT"] = {"method": "knockout GT, TRUE edges ('dir'), beta = the edge mean", "gt_seeds": g.get("seeds")}
    maps["MG:rand"] = MG.random_sized_map(M_pmrt, cells, C.RAND_TAG, C.RAND_KEY)
    prov["MG:rand"] = {"method": f"random_sized_map(M_PMRT, gt cells, tag {C.RAND_TAG}, key {C.RAND_KEY})",
                       "n_edges": len(M_pmrt)}
    maps["blanket2"] = MG.blanket_saving_map()
    prov["blanket2"] = {"method": "blanket_saving_map() (no-map control)"}
    gp = gd = None
    for m, r in bl.items():
        if m == "granger_by":
            d = r["splits"]["disc"]["declared"]
            maps["MG:granger_by"], drop = KBA.map_from_declared(d, pdisc.ud)
            pdcl = r["splits"]["placebo"]["declared"]
            plc_maps["MG:granger_by"] = KBA.map_from_declared(pdcl, pplc.ud)[0]
            prov["MG:granger_by"] = {"method": "granger, BY q .05 (untuned)", "declared": decl_list(d),
                                     "dropped_sign0": drop, "placebo_declared": decl_list(pdcl)}
            continue
        rels = r.get("decl_relations") or list(RELATIONS)
        d_dev = r["splits"]["disc"]["declared"]
        p_dev = r["splits"]["placebo"]["declared"]
        arm = f"MG:{m}@dev"
        maps[arm], drop = KBA.map_from_declared(d_dev, pdisc.ud)
        plc_maps[arm] = KBA.map_from_declared(p_dev, pplc.ud)[0]
        prov[arm] = {"method": f"{m}, far-FPR tau on dev_conf", "tau": r["tau"], "declared": decl_list(d_dev),
                     "dropped_sign0": drop, "placebo_declared": decl_list(p_dev), "note": r.get("note")}
        arm = f"MG:{m}@plc"
        if m == "granger":
            if gp is None:
                gp, gd = granger_stats(pplc.ud, sup), granger_stats(pdisc.ud, sup)
            d_plc, tau = granger_plc_transfer(gp, gd, rels)
            p_plc, _ = granger_plc_transfer(gp, gp, rels)
            tau["transfer"] = "n-free: partial r^2 = F / (F + df)"
        else:
            tau = KBA.placebo_tau(r["splits"]["placebo"]["scores"], rels)
            d_plc = BD.declare(KBA._tuple_scores(r["splits"]["disc"]["scores"]), tau["tau"], rels)
            p_plc = BD.declare(KBA._tuple_scores(r["splits"]["placebo"]["scores"]), tau["tau"], rels)
        maps[arm], drop = KBA.map_from_declared(d_plc, pdisc.ud)
        plc_maps[arm] = KBA.map_from_declared(p_plc, pplc.ud)[0]
        prov[arm] = {"method": f"{m}, placebo-calibrated tau (MAX_FP {MAX_FP})", "tau": tau, "declared":
                     decl_list(d_plc), "dropped_sign0": drop, "placebo_declared": decl_list(p_plc)}
        bl_rep[m] = {"tau_dev": r["tau"], "tau_plc": tau, "cpu_s": r.get("cpu_s"), "wall_s": r.get("wall_s"),
                     "n_placebo_declared": {"dev": n_declared(p_dev), "plc": n_declared(p_plc)},
                     "n_disc_declared": {"dev": n_declared(d_dev), "plc": n_declared(d_plc)}}
    rep["baselines"] = bl_rep
    rep["baseline_errors"] = bl_err
    rep["missing_arms"] = [x for x in C.MAP_ARMS if x not in maps]
    al = C.alias_table(maps, C.THETA, noarb_alias)
    rep["maps"] = {arm: MG.map_to_json(M) for arm, M in maps.items()}
    rep["placebo_maps"] = {arm: MG.map_to_json(M) for arm, M in plc_maps.items()}
    rep["provenance"] = prov
    rep["alias"] = al
    rep["decision_tables"] = {arm: {f"{k[0]}{'+' if k[1] > 0 else '-'}": v
                                    for k, v in MG.decision_table_v2(M, C.THETA).items()} for arm, M in maps.items()}
    rep["edge_quality"] = {arm: KBA.edge_quality(M, cells) for arm, M in maps.items()}
    log(f"maps: { {k: len(v) for k, v in maps.items()} }; jobs {al['jobs']}; missing {rep['missing_arms']}")
    # ---- offline replay on the DISC logged contexts
    all_maps, pairs = {"GT": maps["MG:GT"]}, {}
    for arm, M in maps.items():
        if arm == "MG:GT":
            continue
        all_maps[arm] = M
        all_maps[f"{arm}|clean"] = KBA.clean_map(M, cells)
        pairs[arm] = (arm, "GT")
        pairs[f"{arm}~clean"] = (arm, f"{arm}|clean")
        if arm in plc_maps:
            all_maps[f"GT+plc:{arm}"] = KBA.inject(maps["MG:GT"], plc_maps[arm])
            pairs[f"GT+plc:{arm}"] = (f"GT+plc:{arm}", "GT")
    if a.dev_step1:                                  # descriptive: @dev maps with the step-1 unconfounded DEV tau
        s1 = step1_maps(a, bl, pdisc.ud, cache_dir)
        rep["step1_dev_tau"] = s1["report"]
        for arm, M in s1["maps"].items():
            all_maps[arm] = M
            pairs[arm] = (arm, "GT")
    rp = KBA.replay(recs, all_maps, pairs=pairs)
    rep["replay"] = {"n_units_per_episode": float(np.mean(rp["n_units"])) if rp["n_units"] else float("nan"),
                     "defer_rate": rp["defer_rate"],
                     "defers_per_episode": {m: float(np.mean(v)) for m, v in rp["decisions"].items()},
                     "pairs": pairs, "flips": flip_summary(rp)}
    timing["maps_replay"] = round(time.time() - t, 1)
    # ---- descriptive competitors
    t = time.time()
    desc = {}
    for nm, fn in (("ipw_wald", ipw_wald), ("granger_ctx", granger_ctx)):
        try:
            dd, dp = fn(pdisc.ud), fn(pplc.ud)
            desc[nm] = {"declared": decl_list(dd), "metrics": DB.subset_metrics(dd, ref),
                        "placebo_declared": decl_list(dp), "edge_quality": KBA.edge_quality(
                            KBA.map_from_declared(dd, pdisc.ud)[0], cells)}
        except Exception as e:                                                 # noqa: BLE001
            desc[nm] = {"error": repr(e)[:300]}
    if not a.no_v2:
        try:
            from cdd_oran.decision import crt_units_v2 as V2
            e2 = V2.edges_from_crt_v2(V2.run_crt_units_v2(pdisc.ud, V2.UnitCRTConfigV2(B=a.B), SPLIT["pooled"]))
            desc["v2+by"] = {"declared": decl_list(e2), "metrics": DB.subset_metrics(e2, ref)}
        except Exception as e:                                                 # noqa: BLE001
            desc["v2+by"] = {"error": repr(e)[:300]}
    rep["descriptive"] = desc
    timing["descriptive"] = round(time.time() - t, 1)
    # ---- mode
    full = bool(rep["data"]["all_ok"] and K0n is not None and ast["sha_ok"] and ast["code_ok"]
                and rep["protocol"]["frozen"] and rep["platform"]["equal"] and rep["platform"]["linux"]
                and not rep["missing_arms"])
    rep["mode"] = "full" if full else ("dry-run" if dry else "partial")
    timing["total"] = round(time.time() - t_all, 1)
    rep["timing_s"], rep["peak_rss_mb"] = timing, A4.peak_rss_mb()
    A4.dump(rep, os.path.join(out, "disc_conf.json"))
    log(f"mode {rep['mode']}; label {rep['label']['label']}; wrote {os.path.join(out, 'disc_conf.json')}")
    return rep


def dev_outcomes(files: list, allow_smoke: bool) -> dict:
    """{seed: {"lab_outcome"}} of the dev_conf collection records on REPRO_SEEDS."""
    out = {}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if '"episode"' not in line[:200]:
                    continue
                r = json.loads(line)
                if (r.get("kind") == "episode" and r.get("sub") == C.SUB and r.get("conf_stage") == "dev_conf"
                        and int(r["seed"]) in C.REPRO_SEEDS and (allow_smoke or not r.get("smoke"))
                        and r.get("lab_outcome") is not None):
                    out[int(r["seed"])] = {"lab_outcome": r["lab_outcome"]}
    return out


def step1_maps(a, bl: dict, ud_disc, cache_dir: str) -> dict:
    """Descriptive (section 7.4): the @dev maps with K-B's calibration (the step-1 unconfounded DEV tau)."""
    files = A4.expand(a.dev_step1)
    cz = A4.build_caches(files, "dev", os.path.join(cache_dir, "dev_step1"), "dev1", a.workers)
    d1 = DB.load_pool(cz, H=H, H_pre=H_PRE, stages={"dev"}).unit_data()
    maps, rep = {}, {}
    for m, r in bl.items():
        if m == "granger_by":
            continue
        fn = BD.SCORERS[m]
        kw = {"seed": 0} if m in ("shap_gbdt", "two_tower", "qacm") else {}
        if m == "qacm":
            kw["ann"] = not a.no_ann
        dkw = dict(kw, relations=("own", "far")) if m == "qacm" else kw
        tau = BD.tune_tau(fn(d1, **dkw), "far_fpr")
        rels = r.get("decl_relations") or list(RELATIONS)
        d = BD.declare(KBA._tuple_scores(r["splits"]["disc"]["scores"]), tau["tau"], rels)
        maps[f"MG:{m}@dev1"] = KBA.map_from_declared(d, ud_disc)[0]
        rep[m] = {"tau": tau, "declared": decl_list(d)}
    return {"maps": maps, "report": rep}


# ============================================================================================ build / verify (freeze 2)
def driver_sha(root: str = ROOT) -> str:
    """LF sha256 of the driver with its ``MAPS_SHA256`` line normalised to None: the maps artifact pins the driver,
    and the driver pins the maps artifact (freeze 2), so the pin itself must not enter the driver's hash."""
    txt = open(os.path.join(root, DRIVER), "rb").read().decode("utf-8").replace("\r\n", "\n")
    txt = re.sub(r"^MAPS_SHA256 = .*$", "MAPS_SHA256 = None", txt, count=1, flags=re.M)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def build_artifact(disc: dict, disc_sha: str, allow_dry: bool = False) -> dict:
    """The maps artifact (section 9 step 5) from a disc_conf.json dict. Refuses INVALID, failed record checks, a
    non-full analysis (unless ``allow_dry``) and missing arms (unless ``allow_dry``)."""
    if disc.get("schema") != SCHEMA_DISC:
        raise SystemExit(f"not a disc_conf.json (schema {disc.get('schema')})")
    if disc["label"]["label"] == "INVALID":
        raise SystemExit("discovery label INVALID: the study stops before EVAL (no maps artifact)")
    if any(disc["p_mismatch"].values()) or not all(v["ok"] for v in disc["record_checks"].values()):
        raise SystemExit("record checks failed (probs / p_mismatch)")
    if disc["mode"] != "full" and not allow_dry:
        raise SystemExit(f"disc analysis mode {disc['mode']!r}: only a full analysis can be frozen (--allow-dry)")
    if disc["missing_arms"] and not allow_dry:
        raise SystemExit(f"missing map arms {disc['missing_arms']}")
    maps = {arm: MG.map_from_json(v) for arm, v in disc["maps"].items()}
    noarb_alias = bool(disc["repro"]["noarb_alias"])
    al = C.alias_table(maps, C.THETA, noarb_alias)
    arms = {}
    for arm in [x for x in C.MAP_ARMS if x in maps]:
        arms[arm] = {"map": MG.map_to_json(maps[arm]), "provenance": disc["provenance"].get(arm),
                     "signature": al["signature"][arm], "alias_of": al["alias_of"][arm],
                     "decision_table": {f"{k[0]}{'+' if k[1] > 0 else '-'}": v
                                        for k, v in MG.decision_table_v2(maps[arm], C.THETA).items()}}
    return {"schema": SCHEMA_MAPS, "mode": disc["mode"], "dry": disc["mode"] != "full",
            "protocol": disc["protocol"], "discovery_label": disc["label"]["label"],
            "K0": disc["K0"]["pass"], "K0n": (disc["K0n"] or {}).get("pass"),
            "arms": arms, "jobs": al["jobs"], "alias_of": al["alias_of"], "noarb_alias": noarb_alias,
            "repro": disc["repro"], "all_accept_signature": al["all_accept"],
            "mapgate": {"theta": C.THETA, "k_conf": C.K_CONF, "T": C.T, "open_rule": C.OPEN_RULE, "warmup_s": 0.0,
                        "priority": list(MG.PRIORITY_V2), "guards": list(MG.GUARDS_V2), "rels": list(MG.RELS_V2),
                        "directional": True, "sha256": A4.sha_lf(os.path.join(ROOT, MAPGATE))},
            "sha256": {"disc_conf.json": disc_sha, ANALYZER: A4.sha_lf(os.path.join(ROOT, ANALYZER)),
                       DRIVER: driver_sha(),
                       "v4_artifact": disc["artifact"]["sha256"], "disc_code": disc["code_sha256"]},
            "platform": disc["platform"]}


def cmd_build(a):
    disc = json.load(open(a.disc_json))
    art = build_artifact(disc, A4.sha_lf(a.disc_json), a.allow_dry)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", newline="\n") as fh:
        json.dump(art, fh, indent=1, default=A4._js)
    sha = A4.sha_lf(a.out)
    open(a.out + ".sha256", "w", newline="\n").write(f"{sha}  {os.path.basename(a.out)}\n")
    print(json.dumps({"out": a.out, "sha256": sha, "mode": art["mode"], "jobs": art["jobs"],
                      "alias_of": art["alias_of"]}, indent=1))
    return art


def verify_artifact(path: str, disc_json: str | None = None, allow_dry: bool = False) -> dict:
    a = json.load(open(path))
    sha = A4.sha_lf(path)
    maps = {arm: MG.map_from_json(v["map"]) for arm, v in a["arms"].items()}
    al = C.alias_table(maps, float(a["mapgate"]["theta"]), bool(a["noarb_alias"]))
    chk = {"schema": a.get("schema") == SCHEMA_MAPS,
           "signatures": all(al["signature"][x] == a["arms"][x]["signature"] for x in maps),
           "alias": al["alias_of"] == a["alias_of"] and al["jobs"] == a["jobs"],
           "arms_complete": sorted(maps) == sorted(C.MAP_ARMS),
           "mapgate_sha": a["mapgate"]["sha256"] == A4.sha_lf(os.path.join(ROOT, MAPGATE)),
           "mapgate_consts": (a["mapgate"]["theta"], a["mapgate"]["k_conf"]) == (C.THETA, C.K_CONF),
           "analyzer_sha": a["sha256"][ANALYZER] == A4.sha_lf(os.path.join(ROOT, ANALYZER)),
           "driver_sha": a["sha256"][DRIVER] == driver_sha(),
           "v4_artifact_sha": a["sha256"]["v4_artifact"] == V4_ARTIFACT_SHA256,
           "mode_full": a["mode"] == "full" or allow_dry, "not_invalid": a["discovery_label"] != "INVALID"}
    if disc_json:
        d = json.load(open(disc_json))
        chk["disc_sha"] = a["sha256"]["disc_conf.json"] == A4.sha_lf(disc_json)
        chk["disc_maps"] = all(MG.map_from_json(d["maps"][x]) == maps[x] for x in maps)
    return {"path": path, "sha256": sha, "checks": chk, "ok": all(chk.values()),
            "driver_MAPS_SHA256": C.MAPS_SHA256, "driver_matches": C.MAPS_SHA256 == sha}


def cmd_verify(a):
    v = verify_artifact(a.artifact, a.disc_json, a.allow_dry)
    print(json.dumps(v, indent=1))
    if json.load(open(a.artifact)).get("dry"):
        print("DRY / non-full artifact: plumbing only, NEVER freeze it (no MAPS_SHA256 line)")
    else:
        print(f'MAPS_SHA256 = "{v["sha256"]}"   # e6p_conf.py (after the freeze-2 commit)')
    if not v["ok"]:
        raise SystemExit("maps artifact verification FAILED")
    return v


# ============================================================================================ eval (sec. 7.2)
def load_eval(files: list, alias_of: dict, allow_smoke: bool = False) -> tuple[dict, list]:
    """({seed: {arm: record}} with aliased map arms filled from their simulated target, headers)."""
    R, heads = {}, []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("kind") == "header" and r.get("driver") == "e6p_conf" and r.get("conf_stage") == "eval":
                    heads.append(r)
                if (r.get("kind") != "job" or r.get("schema") != C.SCHEMA_EVAL or r.get("sub") != C.SUB
                        or r.get("conf_stage") != "eval" or (r.get("smoke") and not allow_smoke)):
                    continue
                R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    for s in R:
        for arm, tgt in alias_of.items():
            if tgt != arm and tgt in R[s] and arm not in R[s]:
                R[s][arm] = dict(R[s][tgt], arm=arm, alias_of=tgt)
    return R, heads


def arm_stats_multi(R: dict, arms, n_boot: int = N_BOOT, key=(EVAL_BOOT_TAG, EVAL_BOOT_KEY)) -> dict:
    """Gate A stats (e6p_step2_dev definitions) of every arm on the seeds that have every anchor and every arm in
    ``arms``, all on ONE bootstrap index matrix default_rng([*key, n_seeds]). Returns {"seeds", "n", "point":
    {arm: {...}}, "boot": {arm: {"R", "Rstar", "retention", "guard": {k}}}, "den", ...}."""
    D = C.D
    need = list(dict.fromkeys(list(D.REF_ARMS) + list(arms)))
    seeds = sorted(s for s in R if set(need) <= set(R[s]))
    if not seeds:
        return {"seeds": [], "n": 0}
    n = len(seeds)
    A = {a: D._arrays(R, seeds, a) for a in need}
    P = {a: D._pooled(A[a]) for a in need}
    V_AA = P["noarb"]["V"]
    ref = min(D.SINGLES, key=lambda x: P[x]["V"])
    den = V_AA - P[ref]["V"]
    Ef, EA = P["freeze"]["E"], P[D.A_ARM]["E"]
    point = {}
    for a in need:
        p = P[a]
        Rv = float((V_AA - p["V"]) / den) if abs(den) > 1e-12 else float("nan")
        ret = float((Ef - p["E"]) / (Ef - EA)) if abs(Ef - EA) > 1e-12 else float("nan")
        matched = bool(Ef - p["E"] >= D.X_MATCH * (Ef - EA) - 1e-9)
        g = {k: float(D._ratio(np.float64(p[k]), np.float64(P["noarb"][k]))) for k in D.GUARD_KEYS}
        guard_ok = bool(all(p[k] <= D.GUARD * P["noarb"][k] + 1e-9 for k in D.GUARD_KEYS))
        el = matched and guard_ok
        point[a] = {"V": float(p["V"]), "R": Rv, "Rstar": Rv if el else min(Rv, 0.0), "retention": ret,
                    "matched": matched, "guard_ratio": g, "guard_ok": guard_ok, "eligible": el,
                    "E_kj_per_ep": float(p["E"] / n / 1e3)}
    out = {"seeds": seeds, "n": n, "V_AA": float(V_AA), "V_ref": float(P[ref]["V"]), "ref_arm": ref,
           "den": float(den), "point": point, "boot": {}}
    if n_boot:
        rng = np.random.default_rng([*key, n])
        idx = rng.integers(0, n, (int(n_boot), n))
        Pb = {a: D._pooled(A[a], idx) for a in need}
        den_b = Pb["noarb"]["V"] - np.min([Pb[x]["V"] for x in D.SINGLES], 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            for a in need:
                Rb = (Pb["noarb"]["V"] - Pb[a]["V"]) / den_b
                out["boot"][a] = {"R": Rb, "Rstar": Rb if point[a]["eligible"] else np.minimum(Rb, 0.0),
                                  "retention": (Pb["freeze"]["E"] - Pb[a]["E"]) / (Pb["freeze"]["E"] - Pb[D.A_ARM]["E"]),
                                  "guard": {k: D._ratio(Pb[a][k], Pb["noarb"][k]) for k in D.GUARD_KEYS}}
    return out


def _ci(x):
    return C.D._ci(x)


def d1_check(point: dict, boot: dict, primary: str, assoc: list, margin: float = D1_MARGIN) -> dict:
    """D1: Delta1 = R*(primary) - max_b R*(b) >= margin AND LB90 > 0, LB90 = 5th percentile over resamples of
    R*_boot(primary) - max_b R*_boot(b) (the best map re-selected in every resample)."""
    best = max(assoc, key=lambda b: point[b]["Rstar"])
    d1 = float(point[primary]["Rstar"] - point[best]["Rstar"])
    diff = boot[primary]["Rstar"] - np.max(np.stack([boot[b]["Rstar"] for b in assoc]), 0)
    fin = diff[np.isfinite(diff)]
    lb = float(np.quantile(fin, 0.05)) if fin.size else float("nan")
    sel = np.argmax(np.stack([boot[b]["Rstar"] for b in assoc]), 0)
    freq = {assoc[i]: float(np.mean(sel == i)) for i in range(len(assoc))}
    ok = bool(np.isfinite(d1) and d1 >= margin - 1e-12 and np.isfinite(lb) and lb > 0)
    return {"pass": ok, "delta1": d1, "best_point": best, "lb90": lb, "ci90": _ci(diff),
            "n_nonfinite": int(diff.size - fin.size), "best_selected_share": freq, "margin": margin}


def holm(pvals: dict, alpha: float = ALPHA_D2) -> dict:
    """Holm step-down: {name: rejected}."""
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    m = len(order)
    rej, stop = {}, False
    for i, k in enumerate(order):
        if not stop and pvals[k] <= alpha / (m - i) + 1e-15:
            rej[k] = True
        else:
            stop = True
            rej[k] = False
    return rej


def d2_check(boot: dict, primary: str, assoc: list, signature: dict, alpha: float = ALPHA_D2) -> dict:
    """D2: for every distinct signature of ``assoc`` one hypothesis (aliased maps have identical outcomes): one-sided
    paired bootstrap p = (1 + #{R*_boot(primary) - R*_boot(b) <= 0}) / (N + 1); Holm at alpha; pass iff all rejected."""
    groups = {}
    for b in assoc:
        groups.setdefault(signature[b], []).append(b)
    pv, rows = {}, {}
    for sig, bs in groups.items():
        b = bs[0]
        diff = boot[primary]["Rstar"] - boot[b]["Rstar"]
        nb = len(diff)
        p = float((1 + np.count_nonzero(~(diff > 0))) / (nb + 1))       # NaN counts against the primary
        pv[b] = p
        rows[b] = {"members": bs, "p": p, "same_signature_as_primary": sig == signature.get(primary)}
    rej = holm(pv, alpha)
    for b in rows:
        rows[b]["rejected"] = rej[b]
    return {"pass": bool(rows) and all(rej.values()), "n_hypotheses": len(rows), "hypotheses": rows,
            "alpha": alpha, "procedure": "Holm over the distinct signatures of A, one-sided paired bootstrap"}


def eval_verdict(k0: bool, k0n: bool, e: bool, d1: bool, d2: bool, complete: bool, why_incomplete=()) -> dict:
    """Section 7.2 precedence: INVALID > NOT ELIGIBLE > PASS / PARTIAL / FAIL."""
    if not (k0 and k0n):
        v = "INVALID"
    elif not e:
        v = "NOT ELIGIBLE"
    elif d1 and d2:
        v = "PASS"
    elif d1 or d2:
        v = "PARTIAL"
    else:
        v = "FAIL"
    label = v if complete else f"NOT A VERDICT (would-be: {v}; {'; '.join(why_incomplete)})"
    return {"verdict": v, "label": label, "complete": complete,
            "precedence": "INVALID (K0 or K0n) > NOT ELIGIBLE (E) > PASS (D1 and D2) / PARTIAL (one) / FAIL"}


def cmd_eval(a) -> dict:
    t_all = time.time()
    out = a.out or os.environ.get("JOB_OUT") or "."
    os.makedirs(out, exist_ok=True)
    disc = json.load(open(a.disc_json))
    mv = verify_artifact(a.maps, None, allow_dry=True)
    art = json.load(open(a.maps))
    sig = {arm: v["signature"] for arm, v in art["arms"].items()}
    R, heads = load_eval(A4.expand(a.records), art["alias_of"], a.allow_smoke)
    arms = [x for x in C.ALL_ARMS if any(x in R[s] for s in R)]
    st = arm_stats_multi(R, arms, a.n_boot)
    rep = {"schema": SCHEMA_EVAL, "argv": sys.argv[1:], "maps_verify": mv, "n_seeds_any": len(R),
           "n_seeds": st["n"], "arms_present": arms, "missing_arms": [x for x in C.ALL_ARMS if x not in arms]}
    if not st["n"]:
        raise SystemExit("no EVAL seed with every anchor and arm")
    P, B = st["point"], st["boot"]
    A = [x for x in C.ASSOC_ARMS if x in P]
    rep["gate_a"] = {k: st[k] for k in ("V_AA", "V_ref", "ref_arm", "den", "n")}
    rep["arms"] = {x: dict(P[x], R_ci90=_ci(B[x]["R"]), Rstar_ci90=_ci(B[x]["Rstar"]),
                           retention_ci90=_ci(B[x]["retention"]),
                           guard_ratio_ci90={k: _ci(v) for k, v in B[x]["guard"].items()},
                           counts={g: _sum_counts(R, st["seeds"], x, g) for g in ("defer", "defer_units", "units",
                                                                                 "collateral", "policy", "passed")},
                           cpu_s_mean=float(np.mean([R[s][x].get("cpu_s", np.nan) for s in st["seeds"]])),
                           alias_of=R[st["seeds"][0]][x].get("alias_of"))
                   for x in P}
    k0 = bool(disc["K0"]["pass"])
    k0n = bool((disc.get("K0n") or {}).get("pass"))
    E = {"pass": bool(P[PRIMARY_ARM]["eligible"]) if PRIMARY_ARM in P else False,
         "retention_ci90": rep["arms"].get(PRIMARY_ARM, {}).get("retention_ci90"),
         "guard_ratio_ci90": rep["arms"].get(PRIMARY_ARM, {}).get("guard_ratio_ci90")}
    D1 = d1_check(P, B, PRIMARY_ARM, A) if PRIMARY_ARM in P and A else {"pass": False, "note": "arms missing"}
    D2 = d2_check(B, PRIMARY_ARM, A, sig) if PRIMARY_ARM in P and A else {"pass": False, "note": "arms missing"}
    sec = {}
    for c in SECONDARY:
        if c in P and PRIMARY_ARM in P:
            d = B[PRIMARY_ARM]["R"] - B[c]["R"]
            sec[c] = {"dR": P[PRIMARY_ARM]["R"] - P[c]["R"], "ci90": _ci(d)}
    ns = sec.get("never_sleep")
    if ns:
        ns = dict(ns, noninferior_lb90_gt=NONINF, noninferior=bool(ns["ci90"][0] is not None
                                                                   and ns["ci90"][0] > NONINF),
                  statement="pre-registered: the study does NOT expect and will NOT claim that the PMRT map beats "
                            "the best static rule; reported whatever it shows")
    eff = None
    if PRIMARY_ARM in P and "MG:GT" in P:
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = {"ratio": P[PRIMARY_ARM]["R"] / P["MG:GT"]["R"] if P["MG:GT"]["R"] else float("nan"),
                   "ci90": _ci(B[PRIMARY_ARM]["R"] / B["MG:GT"]["R"])}
    # ---- completeness / sha
    why = []
    dd = disc.get("data", {})
    if disc.get("mode") != "full":
        why.append(f"disc analysis mode {disc.get('mode')}")
    if not dd.get("all_ok"):
        why.append("discovery data incomplete")
    want = C.stage_seeds("eval")
    if st["seeds"] != want:
        why.append(f"eval seeds with every arm: {st['n']} / {len(want)}")
    if rep["missing_arms"]:
        why.append(f"missing arms {rep['missing_arms']}")
    if any(R[s][x].get("smoke") for s in R for x in R[s]):
        why.append("smoke records")
    if not mv["ok"] or not mv["driver_matches"]:
        why.append("maps artifact sha / checks (MAPS_SHA256)")
    fz = C.freeze_status(ROOT)
    if not fz["frozen"]:
        why.append("protocol not frozen (FROZEN_SHA256_CONF)")
    if not (disc["artifact"]["sha_ok"] and disc["artifact"]["code_ok"]):
        why.append("v4 artifact / code sha256")
    plats = {platform_key(h.get("platform")) for h in heads if not h.get("smoke")}
    dplat = {tuple(tuple(y) if isinstance(y, list) else y for y in x) for x in disc.get("platform", {}).get("keys", [])}
    if len(plats | dplat) != 1:
        why.append(f"platform fingerprints differ across stages: {sorted(map(str, plats | dplat))}")
    mg_heads = {h.get("mapgate_sha256") for h in heads}
    if mg_heads and mg_heads != {art["mapgate"]["sha256"]}:
        why.append("mapgate.py sha256 of the eval runs differs from the maps artifact")
    V = eval_verdict(k0, k0n, E["pass"], D1["pass"], D2["pass"], not why, why)
    rep.update(K0=k0, K0n=k0n, E=E, D1=D1, D2=D2, never_sleep=ns, secondary=sec, map_efficiency=eff, verdict=V,
               discovery_label=disc["label"]["label"])
    rep["cpu_h_eval"] = float(sum(R[s][x].get("cpu_s", 0.0) for s in R for x in R[s] if not R[s][x].get("alias_of"))
                              / 3600)
    rep["timing_s"] = round(time.time() - t_all, 1)
    # ---- print
    log(f"EVAL: {st['n']} seeds; V_AA {st['V_AA']:.2f} V_ref {st['V_ref']:.2f} ({st['ref_arm']}) den {st['den']:.2f}")
    f = C.D._f
    for x in C.ALL_ARMS:
        if x not in P:
            continue
        p, ci = P[x], rep["arms"][x]["R_ci90"]
        log(f"  {x:18s} R {f(p['R'])} [{f(ci[0])},{f(ci[1])}] R* {f(p['Rstar'])} ret {p['retention']:.3f} "
            + " ".join(f"{p['guard_ratio'][k]:.2f}" for k in C.D.GUARD_KEYS) + f" {'Y' if p['eligible'] else 'n'}"
            + (f"  (= {rep['arms'][x]['alias_of']})" if rep["arms"][x]["alias_of"] else ""))
    log(f"E {E['pass']}  D1 {D1['pass']} (delta1 {D1.get('delta1')}, LB90 {D1.get('lb90')}, best {D1.get('best_point')})"
        f"  D2 {D2['pass']}")
    if ns:
        log(f"R(PMRT) - R(never_sleep) = {ns['dR']:+.3f} [{f(ns['ci90'][0])},{f(ns['ci90'][1])}] noninferior "
            f"{ns['noninferior']}")
    log(f"VERDICT: {V['label']}")
    A4.dump(rep, os.path.join(out, "eval_conf.json"))
    A4.dump({k: rep[k] for k in ("verdict", "K0", "K0n", "E", "D1", "D2", "never_sleep", "discovery_label",
                                 "maps_verify", "n_seeds")}, os.path.join(out, "verdict_conf.json"))
    return rep


def _sum_counts(R, seeds, arm, grp) -> dict:
    tot = {}
    for s in seeds:
        for k, v in ((R[s][arm].get("policy_counts") or {}).get(grp) or {}).items():
            if isinstance(v, (int, float)):
                tot[k] = tot.get(k, 0) + v
    return dict(sorted(tot.items()))


# ============================================================================================ cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    d = sp.add_parser("disc")
    for k in ("dev", "disc", "placebo", "gt"):
        d.add_argument(f"--{k}", required=True)
    d.add_argument("--artifact", default=os.path.join(ROOT, V4_ARTIFACT))
    d.add_argument("--out", default=None)
    d.add_argument("--cache-dir", dest="cache_dir", default=None)
    d.add_argument("--B", type=int, default=9999)
    d.add_argument("--B-null", dest="B_null", type=int, default=999)
    d.add_argument("--null-group", dest="null_group", type=int, default=60)
    d.add_argument("--null-shifts", dest="null_shifts", type=int, default=4)
    d.add_argument("--workers", type=int, default=1)
    d.add_argument("--methods", default=",".join(TUNED))
    d.add_argument("--no-ann", dest="no_ann", action="store_true")
    d.add_argument("--no-v2", dest="no_v2", action="store_true")
    d.add_argument("--no-null", dest="no_null", action="store_true")
    d.add_argument("--dev-step1", dest="dev_step1", default="")
    d.add_argument("--dry-run", dest="dry_run", action="store_true")
    d.add_argument("--allow-smoke", dest="allow_smoke", action="store_true")
    d.add_argument("--allow-artifact-mismatch", dest="allow_artifact_mismatch", action="store_true")
    b = sp.add_parser("build")
    b.add_argument("--disc-json", dest="disc_json", required=True)
    b.add_argument("--out", default=os.path.join(ROOT, C.MAPS_DOC))
    b.add_argument("--allow-dry", dest="allow_dry", action="store_true")
    v = sp.add_parser("verify")
    v.add_argument("--artifact", default=os.path.join(ROOT, C.MAPS_DOC))
    v.add_argument("--disc-json", dest="disc_json", default=None)
    v.add_argument("--allow-dry", dest="allow_dry", action="store_true")
    e = sp.add_parser("eval")
    e.add_argument("--records", required=True)
    e.add_argument("--disc-json", dest="disc_json", required=True)
    e.add_argument("--maps", default=os.path.join(ROOT, C.MAPS_DOC))
    e.add_argument("--out", default=None)
    e.add_argument("--n-boot", dest="n_boot", type=int, default=N_BOOT)
    e.add_argument("--allow-smoke", dest="allow_smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.op in ("disc", "eval"):
        warnings.simplefilter("ignore")
        np.seterr(all="ignore")
    return {"disc": cmd_disc, "build": cmd_build, "verify": cmd_verify, "eval": cmd_eval}[a.op](a)


if __name__ == "__main__":
    main()

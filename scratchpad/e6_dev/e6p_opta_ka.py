"""E6-P option (a) kill test K-A (scratchpad/e6_dev/decision/OPTION_A_PLAN.md sections 3 and 6; DEV only, NOT frozen).

Question: on the UNCHANGED P3 surge-L40 plant, does a referee that acts from t = 0 with the knockout-GT causal map
(MapGate, cdd_oran/decision/mapgate.py) beat the same referee with a wrong map (own-only, carrier sign-flipped) by a
margin that would make a discovery comparison meaningful?

Plant and references: exactly e6p_step2_dev (make_cfg("P3", 3, seed, lf), lf = e6p_state L40 = 1.75546875, 120 s
warm-up + 600 s scored; the warm-up stays UNSCORED, the referee may act in it). Seeds: DEV 184200-184239 (registered
for step-2 DEV, block 184000-185999). The reference arms (freeze, sub:ES+PowerES, noarb, sub:ES, sub:PowerES,
sub:SliceGuarantee) and the step-2 context arms (B1, B2, M1, M2) are REUSED from runs/e6p-s2dev-1/all.jsonl (same
seeds, same config; ``summary`` checks the headers and the ``noarb_repro`` jobs, which re-run noarb on REPRO_SEEDS in
THIS driver and must reproduce the reused records bit for bit).

New arms (WG3 decisions only; every policy reads only obs):
  GT          UnitArbiter(MapGate(M_GT, .05)), warmup_s = 0, T = 60, open_rule "feasible"
  GT_own      same with own_only(M_GT) (nbr / far edges dropped)
  GT_flip     same with sign_flip(M_GT, carrier) (every carrier edge negated)
  never_sleep static: reject every pico sleep request (prop > cur) from t = 0, accept everything else
  GT_th0      GT with theta = 0     (sensitivity)
  GT_th20     GT with theta = .2    (sensitivity)
  noarb_repro accept all (= noarb) on REPRO_SEEDS only: plumbing / platform check against the reused noarb records
M_GT = TRUE edges of the knockout GT "dir" table gt_ext (step1_v2_full.json["gt"] == step1_v3_full.json["gt"]; 20
fresh GT episodes), beta = the edge mean (GT_EXT_CELLS below; ``check_gt_map`` compares it with the JSON when present).

CLI (repo root, PYTHONPATH=.; in the cloud bundle the same file is e6dev/e6p_opta_ka.py):
  python scratchpad/e6_dev/e6p_opta_ka.py run --part i/k --out F.jsonl [--smoke] [--short S] [--arms a,b]
  python scratchpad/e6_dev/e6p_opta_ka_run.py run --part i/k --out F.jsonl          # cloud.py wrapper
  python scratchpad/e6_dev/e6p_opta_ka.py summary --in F.jsonl [--ref REF.jsonl] [--json OUT] [--allow-smoke]
  python scratchpad/e6_dev/e6p_opta_ka.py list
--smoke: seed -> seed % 31, only the shard's first seed, --short S scored seconds (warm-up stays 120 s).

RECORD (schema "e6p-optaka-rec/1"): header / job / close as e6p_step2_dev, plus per job ``defer``: request-level
rejects by the arbiter per "<family>_<pre|post>" (t < 120 vs >= 120), ``defer_units`` (units opened with reject, same
keys), ``units`` (units opened per family), ``collateral`` (rejected requests whose own direction differs from the
direction of the unit's opening request, per family).

Summary: Gate A quantities as e6p_step2_dev.arm_stats (V_ref = min pooled V of the three singles, re-minimised per
bootstrap resample; paired seed bootstrap N_BOOT = 10000, tag 6618 = the step-2 DEV bootstrap tag, same seeds), plus
dR(GT - arm) with its 90 % CI for every arm. KILL if GT is ineligible, or R(GT) - R(GT_own) < .10, or
R(GT) - R(GT_flip) < .10 (point estimates; CIs printed).
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import e6p_screen as S  # noqa: E402  (sets ROOT / BUNDLE and sys.path)
import e6p_step2_dev as D  # noqa: E402

from cdd_oran.decision.mapgate import (  # noqa: E402
    THETA,
    decision_table,
    map_from_gt,
    mapgate_arbiter,
    own_only,
    sign_flip,
)

PLAN = "scratchpad/e6_dev/decision/OPTION_A_PLAN.md sec. 3 + 6 (K-A); DEV only, 2026-09-30, not frozen"
SCHEMA = "e6p-optaka-rec/1"
PAIR, STRATUM = D.PAIR, D.STRATUM
SEEDS = D.SEEDS
REPRO_SEEDS = (184200, 184201, 184202)
WARM = 120.0                                                     # scored-window start (config warm-up), for counts

REF_ARMS, SINGLES, A_ARM = D.REF_ARMS, D.SINGLES, D.A_ARM
CTX_ARMS = D.POLICY_ARMS                                         # B1 B2 M1 M2 (reused, context only)
NEW_ARMS = ("GT", "GT_own", "GT_flip", "never_sleep", "GT_th0", "GT_th20")
ARMS = NEW_ARMS + ("noarb_repro",)
KILL_MARGIN = 0.10
REF_DEFAULT = os.path.join(HERE, "runs", "e6p-s2dev-1", "all.jsonl")
REF_COMMIT = "a1384a3"                                           # git head of the reused runs
N_BOOT, BOOT_TAG = D.N_BOOT, D.BOOT_TAG

# gt_ext "dir" table (step1_v2_full.json["gt"]["cells"]): (family, relation, kpi, status, mean)
GT_EXT_CELLS = (
    ("carrier", "own", "pv", "TRUE", -6.783080260303688), ("carrier", "own", "v", "TRUE", -116.27548806941432),
    ("carrier", "own", "e", "TRUE", 6581.903035313515), ("carrier", "own", "rlf", "INDET", 0.005061460592913955),
    ("carrier", "own", "load", "TRUE", -55.611713665943604), ("carrier", "nbr", "pv", "TRUE", 0.8250180766449747),
    ("carrier", "nbr", "v", "INDET", 14.370932754880695), ("carrier", "nbr", "e", "INDET", 301.05112627527734),
    ("carrier", "nbr", "rlf", "INDET", 0.03470715835140997), ("carrier", "nbr", "load", "NULL", 36.4887924801157),
    ("carrier", "far", "pv", "INDET", 0.2566883586406363), ("carrier", "far", "v", "NULL", 6.36948662328272),
    ("carrier", "far", "e", "NULL", 117.91223646646007), ("carrier", "far", "rlf", "INDET", 0.014461315979754157),
    ("carrier", "far", "load", "NULL", 19.122921185827913),
    ("sleep", "own", "pv", "TRUE", -0.6933333333333335), ("sleep", "own", "v", "INDET", -12.68),
    ("sleep", "own", "e", "NULL", -190.07756876627607), ("sleep", "own", "rlf", "TRUE", -0.11555555555555555),
    ("sleep", "own", "load", "TRUE", -995.9377777777779), ("sleep", "nbr", "pv", "TRUE", 9.066666666666668),
    ("sleep", "nbr", "v", "TRUE", 399.1333333333334), ("sleep", "nbr", "e", "TRUE", 4695.624190710959),
    ("sleep", "nbr", "rlf", "INDET", 0.05333333333333334), ("sleep", "nbr", "load", "TRUE", 921.2622222222221),
    ("sleep", "far", "pv", "TRUE", 2.417777777777778), ("sleep", "far", "v", "TRUE", 33.711111111111116),
    ("sleep", "far", "e", "TRUE", 1367.366021516699), ("sleep", "far", "rlf", "TRUE", 0.07555555555555554),
    ("sleep", "far", "load", "TRUE", 74.67555555555555),
    ("ptx", "own", "pv", "TRUE", 2.0338461538461536), ("ptx", "own", "v", "TRUE", 41.83794871794872),
    ("ptx", "own", "e", "TRUE", 4811.618351655618), ("ptx", "own", "rlf", "INDET", -0.16102564102564101),
    ("ptx", "own", "load", "TRUE", 319.1569230769231), ("ptx", "nbr", "pv", "INDET", -0.34974358974358966),
    ("ptx", "nbr", "v", "TRUE", -31.317948717948713), ("ptx", "nbr", "e", "TRUE", -503.11579152657976),
    ("ptx", "nbr", "rlf", "INDET", -0.2348717948717949), ("ptx", "nbr", "load", "TRUE", -212.9876923076923),
    ("ptx", "far", "pv", "INDET", -0.5394871794871796), ("ptx", "far", "v", "TRUE", -20.45641025641026),
    ("ptx", "far", "e", "TRUE", -334.2005622661469), ("ptx", "far", "rlf", "INDET", -0.18974358974358976),
    ("ptx", "far", "load", "TRUE", -106.16923076923077),
    ("prot_min", "own", "pv", "TRUE", -5.126126126126126), ("prot_min", "own", "v", "NULL", 0.15894465894465903),
    ("prot_min", "own", "e", "NULL", -2.2252273244725624), ("prot_min", "own", "rlf", "NULL", 0.0),
    ("prot_min", "own", "load", "NULL", -0.10489060489060488), ("prot_min", "nbr", "pv", "NULL", -0.006435006435006429),
    ("prot_min", "nbr", "v", "NULL", -0.258043758043758), ("prot_min", "nbr", "e", "NULL", -10.443604805554294),
    ("prot_min", "nbr", "rlf", "NULL", -0.002574002574002574), ("prot_min", "nbr", "load", "NULL", -0.05984555984555983),
    ("prot_min", "far", "pv", "NULL", 0.0341055341055341), ("prot_min", "far", "v", "NULL", -0.05019305019305022),
    ("prot_min", "far", "e", "NULL", -9.389110710423813), ("prot_min", "far", "rlf", "NULL", -0.0006435006435006435),
    ("prot_min", "far", "load", "NULL", 0.1647361647361647),
)
GT_JSON = os.path.join(HERE, "decision", "step1_v2_full.json")


def _cells(rows=GT_EXT_CELLS):
    return [{"family": f, "relation": r, "kpi": k, "status": s, "mean": m} for f, r, k, s, m in rows]


M_GT = map_from_gt(_cells(), true_only=True)
MAPS = {"GT": M_GT, "GT_own": own_only(M_GT), "GT_flip": sign_flip(M_GT, ("carrier",))}
THETAS = {"GT": THETA, "GT_own": THETA, "GT_flip": THETA, "GT_th0": 0.0, "GT_th20": 0.2}
MAP_OF = {"GT": "GT", "GT_own": "GT_own", "GT_flip": "GT_flip", "GT_th0": "GT", "GT_th20": "GT"}


def check_gt_map(path=GT_JSON):
    """True / False: the embedded table equals the JSON's gt cells (family, relation, kpi, status, mean); None if the
    JSON is absent (cloud bundle)."""
    if not os.path.exists(path):
        return None
    cells = json.load(open(path))["gt"]["cells"]
    return [(c["family"], c["relation"], c["kpi"], c["status"], c["mean"]) for c in cells] == list(GT_EXT_CELLS)


# ---------------------------------------------------------------------------------------------- arbiters
def direction(r) -> str:
    return D.direction(r)


def _fam_dir(r):
    return r["knob"][0], (r["prop"] > r["cur"]) - (r["prop"] < r["cur"])


class Counting:
    """Wraps an obs-only arbiter; tallies its rejects per family and phase (t < WARM: pre, else post). For a
    UnitArbiter it also tallies units opened / deferred and the collateral rejects (a request rejected by a unit whose
    opening request had the other direction). Adds nothing to the decisions."""

    def __init__(self, inner, is_unit: bool):
        self.inner, self.is_unit = inner, is_unit
        self.n = {"defer": {}, "defer_units": {}, "units": {}, "collateral": {}}

    def _inc(self, grp, key):
        self.n[grp][key] = self.n[grp].get(key, 0) + 1

    def __call__(self, obs):
        out = self.inner(obs)
        t = obs["t"]
        ph = "pre" if t < WARM else "post"
        if self.is_unit:
            for u in self.inner.opened:
                self._inc("units", f"{u['knob']}_{ph}")
                if u["mode"] == "reject":
                    self._inc("defer_units", f"{u['knob']}_{ph}")
        for r, d in zip(obs["requests"], out["decisions"], strict=True):
            if d != "reject":
                continue
            f, sd = _fam_dir(r)
            self._inc("defer", f"{f}_{ph}")
            if self.is_unit:
                u = self.inner.active.get((int(r["knob"][1]), r["xapp"]))
                if u is not None and sd != (u["ctx"]["step"] > 0) - (u["ctx"]["step"] < 0):
                    self._inc("collateral", f)
        return out


def make_arbiter(arm: str):
    if arm in MAP_OF:
        return Counting(mapgate_arbiter(MAPS[MAP_OF[arm]], THETAS[arm]), is_unit=True)
    if arm == "never_sleep":
        return Counting(D.BlanketGate({"sleep"}, warmup_s=-1.0), is_unit=False)   # t > -1: active from t = 0
    if arm == "noarb_repro":
        return None
    raise KeyError(arm)


# ---------------------------------------------------------------------------------------------- jobs
def jobs() -> list:
    J = [(D.check_seed(sd), arm) for sd in SEEDS for arm in NEW_ARMS]
    J += [(D.check_seed(sd), "noarb_repro") for sd in REPRO_SEEDS]
    return J


def run_job(seed: int, arm: str, lf: float, smoke: bool = False, short=None) -> dict:
    D.check_seed(seed)
    if arm not in ARMS:
        raise KeyError(arm)
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg(PAIR, STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    arb = make_arbiter(arm)
    t_wall, t_cpu = time.time(), time.process_time()
    env, _ = S.run_env(cfg, arb)
    cpu = time.process_time() - t_cpu
    pc = {}
    if arb is not None:
        pc = dict(arb.n)
        pol = getattr(getattr(arb.inner, "policy", None), "n", None) or getattr(arb.inner, "n", None)
        pc["policy"] = dict(pol or {})
    return {"kind": "job", "schema": SCHEMA, "key": [seed, arm], "seed": seed, "cfg_seed": sd, "arm": arm,
            "smoke": smoke, "short": short, "episode_s": env.total_s, "warmup_s": cfg.warmup_s, "load_factor": lf,
            **S.outcome(env), "policy_counts": pc, "cpu_s": round(cpu, 2), "secs": round(time.time() - t_wall, 1),
            "rss_mb": D.peak_rss_mb()}


def run(part, out, smoke=False, short=None, arms=None):
    if short and not smoke:
        raise SystemExit("--short is smoke-only")
    reg = D.registry_check()
    if not smoke and reg is not None and not all(reg.values()) and not os.environ.get("E6P_S2_ALLOW_UNREGISTERED"):
        raise SystemExit(f"seed block {D.REG_BLOCK} not registered in {D.REGISTRY_DOC}: {reg}")
    if arms:
        bad = [a for a in arms if a not in ARMS]
        if bad:
            raise SystemExit(f"unknown arms {bad}; arms: {ARMS}")
    state = S.load_state()
    lf = S.lf_of(state, STRATUM)
    i, k = map(int, part.split("/"))
    head = S.header("opta_ka", part, smoke, state)
    head.update(kind="header", schema=SCHEMA, plan=PLAN, driver="e6p_opta_ka", registry=reg, gt_map_check=check_gt_map(),
                consts={"pair": PAIR, "stratum": STRATUM, "load_factor": lf, "seeds": [SEEDS[0], SEEDS[-1]],
                        "repro_seeds": REPRO_SEEDS, "arms": ARMS, "thetas": THETAS, "T": 60.0, "warmup_s_arbiter": 0.0,
                        "open_rule": "feasible", "maps": {a: [[*kk, v] for kk, v in sorted(m.items())]
                                                          for a, m in MAPS.items()},
                        "smoke": {"short": short} if smoke else None, "arms_filter": arms})
    S._append(out, head)
    done = {tuple(r["key"]) for r in S._read(out) if r.get("kind") == "job" and r.get("smoke") == smoke}
    t0, n, first_seed = time.time(), 0, None
    J = jobs()
    for u, (seed, arm) in enumerate(J):
        if u % k != i or (seed, arm) in done or (arms and arm not in arms):
            continue
        if smoke:
            first_seed = seed if first_seed is None else first_seed
            if seed != first_seed:
                break
        rec = run_job(seed, arm, lf, smoke, short)
        S._append(out, rec)
        n += 1
        print(json.dumps({x: rec.get(x) for x in ("seed", "cfg_seed", "arm", "psvr", "energy_j", "policy_counts",
                                                  "cpu_s", "secs", "rss_mb")}), flush=True)
    S._append(out, {"kind": "close", "stage": "opta_ka", "part": part, "n_jobs": n, "n_total": len(J),
                    "secs": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------------------------- summary
def load_all(paths, ref_path, allow_smoke=False):
    """({seed: {arm: rec}}, new headers, ref headers). Reference file: only REF_ARMS + CTX_ARMS records of schema
    e6p-s2dev-rec/1; new files: schema e6p-optaka-rec/1."""
    R, hn, hr = {}, [], []
    for r in S._read(ref_path):
        if r.get("kind") == "header" and r.get("schema") == D.SCHEMA:
            hr.append(r)
        if (r.get("kind") == "job" and r.get("schema") == D.SCHEMA and not r.get("smoke")
                and r["arm"] in REF_ARMS + CTX_ARMS):
            R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    for p in paths:
        for r in S._read(p):
            if r.get("kind") == "header" and r.get("schema") == SCHEMA:
                hn.append(r)
            if r.get("kind") != "job" or r.get("schema") != SCHEMA or (r.get("smoke") and not allow_smoke):
                continue
            R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    return R, hn, hr


def config_match(R, hn, hr) -> dict:
    """Checks that the reused reference runs and the new runs share plant, config, code and numeric platform."""
    def one(hs, f):
        return sorted({json.dumps(f(h), sort_keys=True, default=str) for h in hs})

    fields = {"load_factor": lambda h: h["consts"]["load_factor"], "pair_stratum": lambda h: [h["consts"]["pair"],
                                                                                              h["consts"]["stratum"]],
              "scenario_version": lambda h: h["scenario_version"], "e6p_spec_commit": lambda h: h["e6p_spec_commit"],
              "frozen_sha256": lambda h: h["frozen_sha256"], "state_load_factor": lambda h: h["state"]["load_factor"],
              "numpy_scipy_platform": lambda h: [h["numeric_env"].get(x) for x in ("numpy", "scipy", "platform")]}
    out = {}
    for k, f in fields.items():
        a, b = one(hr, f), one(hn, f)
        out[k] = {"ref": a, "new": b, "ok": len(a) == 1 and a == b} if hn else {"ref": a, "new": None, "ok": None}
    env_diff = None
    try:
        r = subprocess.run(["git", "diff", "--stat", REF_COMMIT, "--", "cdd_oran/envs/e6", "scratchpad/e6_dev/e6p_screen.py",
                            "scratchpad/e6_dev/e6p_state.json", "scratchpad/e6_dev/e6p_state_json.py"],
                           cwd=S.ROOT, capture_output=True, text=True, timeout=30)
        env_diff = r.stdout.strip() if r.returncode == 0 else f"git error {r.returncode}"
    except Exception as e:  # noqa: BLE001
        env_diff = f"unavailable: {e!r}"
    out["plant_code_diff_vs_ref_commit"] = {"diff": env_diff, "ok": env_diff == ""}
    rep = {}
    for s in REPRO_SEEDS:
        if s in R and "noarb" in R[s] and "noarb_repro" in R[s]:
            a, b = R[s]["noarb"], R[s]["noarb_repro"]
            rep[s] = max(abs(float(a[f]) - float(b[f])) for f in D.SUM_FIELDS)
    out["noarb_repro_max_abs_diff"] = {"per_seed": rep, "ok": (all(v == 0 for v in rep.values()) if rep else None)}
    job_lf = {float(R[s][a]["load_factor"]) for s in R for a in R[s]}
    job_ep = {(float(R[s][a]["warmup_s"]), float(R[s][a]["episode_s"])) for s in R for a in R[s] if not R[s][a].get("smoke")}
    out["jobs"] = {"load_factor": sorted(job_lf), "warmup_episode_s": sorted(job_ep),
                   "ok": len(job_lf) == 1 and len(job_ep) == 1}
    return out


def arm_stats(R, arm, comps=(), n_boot=N_BOOT):
    """Gate A R / retention / guards / eligibility of ``arm`` on the seeds that have every reference arm, the arm and
    every arm of ``comps``; dR = R(c) - R(arm) for c in comps (paired, same resample). None if no such seed."""
    need = set(REF_ARMS) | {arm} | set(comps)
    seeds = sorted(s for s in R if need <= set(R[s]))
    if not seeds:
        return None
    n = len(seeds)
    A = {a: D._arrays(R, seeds, a) for a in need}
    P = {a: D._pooled(A[a]) for a in need}
    V_AA = P["noarb"]["V"]
    ref = min(SINGLES, key=lambda a: P[a]["V"])
    den = V_AA - P[ref]["V"]
    Ef, EA = P["freeze"]["E"], P[A_ARM]["E"]
    p = P[arm]
    g = {kk: float(D._ratio(np.float64(p[kk]), np.float64(P["noarb"][kk]))) for kk in D.GUARD_KEYS}
    matched = bool(Ef - p["E"] >= D.X_MATCH * (Ef - EA) - 1e-9)
    guard_ok = bool(all(p[kk] <= D.GUARD * P["noarb"][kk] + 1e-9 for kk in D.GUARD_KEYS))
    out = {"arm": arm, "n_seeds": n, "V": float(p["V"]), "V_AA": float(V_AA), "V_ref": float(P[ref]["V"]),
           "ref_arm": ref, "den": float(den), "R": float((V_AA - p["V"]) / den), "retention": float((Ef - p["E"]) / (Ef - EA)),
           "matched": matched, "guard_ratio": g, "guard_ok": guard_ok, "eligible": bool(matched and guard_ok),
           "E_kj_per_ep": float(p["E"] / n / 1e3),
           "dR": {c: float((P[arm]["V"] - P[c]["V"]) / den) for c in comps}}
    if n_boot:
        rng = np.random.default_rng([BOOT_TAG, n])
        idx = rng.integers(0, n, (n_boot, n))
        Pb = {a: D._pooled(A[a], idx) for a in need}
        den_b = Pb["noarb"]["V"] - np.min([Pb[a]["V"] for a in SINGLES], 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["R_ci90"] = D._ci((Pb["noarb"]["V"] - Pb[arm]["V"]) / den_b)
            out["retention_ci90"] = D._ci((Pb["freeze"]["E"] - Pb[arm]["E"]) / (Pb["freeze"]["E"] - Pb[A_ARM]["E"]))
            out["guard_ratio_ci90"] = {kk: D._ci(D._ratio(Pb[arm][kk], Pb["noarb"][kk])) for kk in D.GUARD_KEYS}
            out["dR_ci90"] = {c: D._ci((Pb[arm]["V"] - Pb[c]["V"]) / den_b) for c in comps}
    return out


def _sum_counts(R, arm, grp):
    tot = {}
    for s in R:
        rec = R[s].get(arm)
        if rec is None:
            continue
        for kk, v in (rec.get("policy_counts") or {}).get(grp, {}).items():
            tot[kk] = tot.get(kk, 0) + v
    return dict(sorted(tot.items()))


def summary(paths, ref_path=REF_DEFAULT, json_out=None, allow_smoke=False, n_boot=N_BOOT):
    R, hn, hr = load_all(paths, ref_path, allow_smoke)
    cm = config_match(R, hn, hr)
    f = D._f
    print(f"== E6-P option (a) K-A (P3 surge-L40, DEV {SEEDS[0]}-{SEEDS[-1]}): {len(R)} seeds; reference file {ref_path}")
    print("   config match:", {k: v["ok"] for k, v in cm.items()})
    report = {"plan": PLAN, "schema": SCHEMA, "config_match": cm, "gt_map_check": check_gt_map(),
              "decision_tables": {a: {f"{k[0]}{'+' if k[1] > 0 else '-'}": v for k, v in decision_table(
                  MAPS[MAP_OF[a]], THETAS[a]).items()} for a in MAP_OF}, "arms": {}}
    rows = [a for a in NEW_ARMS + REF_ARMS + CTX_ARMS if any(a in R[s] for s in R)]
    for arm in rows:
        comps = tuple(c for c in (("GT",) if arm != "GT" else ()) if any(c in R[s] for s in R))
        st = arm_stats(R, arm, comps, n_boot)
        if st is None:
            continue
        st["defer"] = _sum_counts(R, arm, "defer")
        st["defer_units"] = _sum_counts(R, arm, "defer_units")
        st["units"] = _sum_counts(R, arm, "units")
        st["collateral"] = _sum_counts(R, arm, "collateral")
        st["cpu_s_mean"] = float(np.mean([R[s][arm]["cpu_s"] for s in R if arm in R[s]]))
        report["arms"][arm] = st
    A = report["arms"]
    if A:
        r0 = next(iter(A.values()))
        print(f"   V_AA {r0['V_AA']:.2f}  V_ref {r0['V_ref']:.2f} ({r0['ref_arm']})  den {r0['den']:.2f}  n {r0['n_seeds']}")
        print(f"   {'arm':20s} {'n':>3} {'V':>7} {'R':>7} {'R 90% CI':>17} {'GT-arm':>7} {'90% CI':>17} {'ret':>6} "
              f"{'svr':>5} {'nonp':>5} {'ll':>5} {'rlf':>5} elig  cpu_s")
        for arm, st in A.items():
            ci = st.get("R_ci90", [None, None])
            dR = st["dR"].get("GT")
            dci = st.get("dR_ci90", {}).get("GT", [None, None])
            print(f"   {arm:20s} {st['n_seeds']:>3} {st['V']:7.2f} {f(st['R']):>7} [{f(ci[0])},{f(ci[1])}] "
                  f"{f(dR):>7} [{f(dci[0])},{f(dci[1])}] {st['retention']:6.3f} "
                  + " ".join(f"{st['guard_ratio'][k]:5.2f}" for k in D.GUARD_KEYS)
                  + f"  {'Y' if st['eligible'] else 'n'}{'' if st['matched'] else 'E'}{'' if st['guard_ok'] else 'G'}"
                  f"  {st['cpu_s_mean']:6.1f}")
        print("   (GT-arm = R(GT) - R(arm); eligible = retention >= .90 and every guard ratio <= 1.10; E energy / G guard fail)")
        for arm in NEW_ARMS:
            if arm in A:
                print(f"   {arm:12s} request rejects {A[arm]['defer']}  deferred units {A[arm]['defer_units']}  "
                      f"collateral {A[arm]['collateral']}")
        kill, why = None, []
        if "GT" in A:
            kill = False
            if not A["GT"]["eligible"]:
                kill = True
                why.append("GT ineligible")
            for c in ("GT_own", "GT_flip"):
                if c in A:
                    d = A[c]["dR"]["GT"]                              # = R(GT) - R(c)
                    ci = A[c].get("dR_ci90", {}).get("GT", [None, None])
                    print(f"   R(GT) - R({c}) = {d:+.3f}  90% CI [{f(ci[0])},{f(ci[1])}]  (kill if < {KILL_MARGIN})")
                    if d < KILL_MARGIN:
                        kill = True
                        why.append(f"R(GT) - R({c}) < {KILL_MARGIN}")
                else:
                    why.append(f"{c} missing")
                    kill = None if kill is False else kill
        report["kill"] = {"kill": kill, "why": why}
        print(f"   K-A verdict: {'KILL' if kill else ('PASS (continue)' if kill is False else 'INCOMPLETE')}"
              + (f"  ({'; '.join(why)})" if why else ""))
        cpu_new = [R[s][a]["cpu_s"] for s in R for a in R[s] if a in ARMS]
        report["cpu_h_new"] = float(sum(cpu_new) / 3600)
        print(f"   CPU (new arms) {report['cpu_h_new']:.2f} h total, {np.mean(cpu_new) if cpu_new else float('nan'):.1f} s/job")
    if json_out:
        json.dump(report, open(json_out, "w", newline="\n"), indent=1, default=S._js)
    return report


def list_jobs():
    J = jobs()
    print("plan:", PLAN)
    print("registry:", D.registry_check(), " gt_map_check:", check_gt_map())
    print(f"{len(J)} jobs = {len(SEEDS)} seeds x {len(NEW_ARMS)} arms {NEW_ARMS} + noarb_repro on {REPRO_SEEDS}")
    for a in MAP_OF:
        print(f"  {a:8s} theta {THETAS[a]:.2f}  map edges (own/nbr read):",
              {f"{k[0]}.{k[1]}.{k[2]}": round(v, 3) for k, v in sorted(MAPS[MAP_OF[a]].items())
               if k[1] in ("own", "nbr") and k[2] in ("pv", "v", "e")})
        print("           decisions:", {f"{k[0]}{'+' if k[1] > 0 else '-'}": v
                                       for k, v in decision_table(MAPS[MAP_OF[a]], THETAS[a]).items()})


def cli(argv):
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    flags = {x for x in rest if x in ("--smoke", "--allow-smoke")}
    rest = [x for x in rest if x not in flags]
    if cmd == "run":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        run(a["--part"], a["--out"], smoke="--smoke" in flags,
            short=float(a["--short"]) if a.get("--short") else None,
            arms=[x for x in a["--arms"].split(",") if x] if a.get("--arms") else None)
    elif cmd == "summary":
        files, js, ref = [], None, REF_DEFAULT
        it = iter(rest)
        for x in it:
            if x == "--json":
                js = next(it)
            elif x == "--ref":
                ref = next(it)
            elif x != "--in":
                files.extend(y for y in x.split(",") if y)
        summary(files, ref_path=ref, json_out=js, allow_smoke="--allow-smoke" in flags)
    elif cmd == "list":
        list_jobs()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])

"""E6-P step 2, DEV on-policy arms (scratchpad/e6_dev/decision/STEP2_REFEREE_PLAN.md + the step-2 debate,
.tmp/step2debate/{F,G}; DEV only, NOT frozen). Paired-by-seed comparison of two blanket static gates (F) and two
map-informed gates (G) against the Gate A references on the cell P3 surge-L40.

Plant: ``e6p_screen.make_cfg("P3", 3, seed, lf)`` (frozen E6P config, 120 s warm-up + 600 s scored = 720 s; lf = the
calibrated L40 load factor from e6p_state, 1.75546875), ``e6p_screen.run_env`` (E6Env(wg3=True, log=False)).
Freeze runs with ``make_cfg("*", ...)`` (mix none, no xApps) exactly as Gate A stage 1 does (plant identical).

Seeds: DEV 184200-184239 (40 seeds; block 184000-185999 registered as E6 "e6p_referee_episodes" in
docs/benchmark/SEED_REGISTRY.json); every job seed is asserted in [184200, 184239] and outside the sealed blocks.
One job = one (seed, arm); jobs are seed-major, dealt to shards by index (job u -> shard u % k); resumable (keys already
in --out with the same smoke flag are skipped).

Arms (all WG3 decisions only; the four policy arms read ONLY ``obs``):
  freeze              reject everything (Ef, energy-retention anchor)                           [Gate A arm 1]
  sub:ES+PowerES      the energy side alone (EA, energy-retention anchor = Gate A "A" arm)      [Gate A arm 4]
  noarb               accept all = AA                                                          [Gate A arm 3]
  sub:ES | sub:PowerES | sub:SliceGuarantee   single-xApp arms; V_ref = min pooled V of the three [Gate A arm 2]
  B1  F blanket gate "no_ptxup": reject every PowerES ptx-up request (prop >= cur) once t > warm-up.
  B2  F blanket gate "no_ptxup_sleep_caroff" (F's best, R .30 on 8 local seeds): B1 + reject pico sleep (prop > cur)
      + reject macro carrier-off (prop < cur), once t > warm-up.
  M1  G's "GP": once t >= warm-up, reject a ptx-up request (prop > cur) iff the requesting cell is saturated: its own
      prb_util (mean of its last W_REPORTS = 5 delivered fast KPM reports, units_p.UnitArbiter.mediators) >= .999.
  M2  M1 + an LL guard. G described it but never implemented it (not in .tmp/step2debate/G), so it is defined here:
      region of cell c = its obs-only exposure set N(c) (c + in/out neighbours of obs["static"]["neighbours"]; the
      site map is privileged and not used). Keep the last 2 x LL_W delivered fast reports' ll_delay_p95 (per cell);
      the region's LL delay is RISING iff nanmean over (last LL_W reports x N(c)) > nanmean over (the LL_W reports
      before them x N(c)) (both finite; fewer than 2 x LL_W reports -> not rising). While rising, M1's ptx-up
      rejection in that cell is released (accept); LL_W = 10 (about 10 s vs the previous 10 s).
Warm-up edges copied from each author: F gates t > warmup_s, G gates t >= warmup_s (differs only at t = 120).

CLI (repo root, PYTHONPATH=.; in the cloud bundle the same file is e6dev/e6p_step2_dev.py):
  python scratchpad/e6_dev/e6p_step2_dev.py run --stage dev --part i/k --out F.jsonl [--smoke] [--short S] [--arms a,b]
  python scratchpad/e6_dev/e6p_step2_dev_run.py run --part i/k --out F.jsonl          # cloud.py wrapper (stage dev)
  python scratchpad/e6_dev/e6p_step2_dev.py summary --in F.jsonl [F2.jsonl ...] [--json OUT] [--allow-smoke]
  python scratchpad/e6_dev/e6p_step2_dev.py list
--smoke: DEV plumbing only; seed -> seed % 31 (E6 DEV seeds 0-30); runs only the jobs of the shard's FIRST seed
(every arm of it, or --arms); --short S sets the scored seconds (smoke only; warm-up stays 120 s); records smoke=True
(summary ignores them unless --allow-smoke).

RECORD (schema "e6p-s2dev-rec/1", JSONL): "header" (per run call), "job" (one per (seed, arm)), "close".
job: key [seed, arm], seed, cfg_seed, arm, smoke, short, episode_s, warmup_s, load_factor, e6p_screen.outcome fields
(prot_viol, prot_ue_s, psvr, energy_j, viol_ue_s, ue_s, svr, nonprot_embb_viol, ll_viol, rlf, ... st_*),
policy_counts (rejects / releases per direction), cpu_s (process CPU of the episode), secs (wall), rss_mb (peak).

Summary (Gate A definitions, e6p_screen.analyse_pair_stratum, pooled over the seeds that have every reference arm and
the arm): V = 3600 sum prot_viol / sum prot_ue_s; V_ref = min V of the three singles; R = (V_AA - V) / (V_AA - V_ref);
retention = (Ef - E) / (Ef - EA); guard ratio_k = pooled k / pooled AA k for k in svr (3600 sum viol_ue_s / sum ue_s),
nonprot_embb_viol, ll_viol, rlf; eligible = (Ef - E >= .90 (Ef - EA)) and every ratio <= 1.10. Paired seed bootstrap
(N_BOOT, default_rng([BOOT_TAG, n_seeds])), V_ref re-minimised per resample; 90 % CI = 5th-95th percentiles, of R and
of R(arm) - R(B1) (same resample). R_fixed = (V_AA - V) / 45.82 (Gate A v2 P3/3 denominator, .tmp/step2/kl0.json) is
printed for reference only.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import e6p_screen as S  # noqa: E402  (sets ROOT / BUNDLE and sys.path)

from cdd_oran.decision.units_p import REPORT_MEDIATORS, UnitArbiter, accept_all_policy  # noqa: E402

PLAN = ("scratchpad/e6_dev/decision/STEP2_REFEREE_PLAN.md + step-2 debate F/G (.tmp/step2debate); DEV only, "
        "2026-09-30, not frozen")
SCHEMA = "e6p-s2dev-rec/1"
REGISTRY_DOC = "docs/benchmark/SEED_REGISTRY.json"
PAIR, STRATUM = "P3", 3
SEED_LO, SEED_HI = 184200, 184239
REG_BLOCK = (184000, 185999)
FORBIDDEN = ((150200, 150399), (155200, 155399), (160000, 179999))
SEEDS = tuple(range(SEED_LO, SEED_HI + 1))
STAGES = ("dev",)

SINGLES = tuple("sub:" + x for x in S.deployed(PAIR))              # sub:ES, sub:PowerES, sub:SliceGuarantee
A_ARM = S.a_arm(PAIR)                                               # sub:ES+PowerES
REF_ARMS = ("freeze", A_ARM, "noarb") + SINGLES
POLICY_ARMS = ("B1", "B2", "M1", "M2")
ARMS = REF_ARMS + POLICY_ARMS
BASE_ARM = "B1"

B_GATES = {"B1": frozenset({"ptx_up"}), "B2": frozenset({"ptx_up", "sleep", "car_off"})}
SAT_UTIL = 0.999                                                    # G: own prb_util >= .999 = saturated
J_PRB = REPORT_MEDIATORS.index("prb_util")
LL_W = 10                                                           # M2: fast reports per window (recent vs previous)

GUARD_KEYS = ("svr", "nonprot_embb_viol", "ll_viol", "rlf")
X_MATCH, GUARD = S.X_MATCH, S.GUARD                                 # .90, 1.10
N_BOOT, BOOT_TAG = 10_000, 6618                                     # 6618: first tag of the step-2 reserved 6618-6621
DEN_GATE_A = 45.822393822393835                                     # Gate A v2 P3/3 V_AA - V_ref (kl0.json constants)
SUM_FIELDS = ("prot_viol", "prot_ue_s", "energy_j", "viol_ue_s", "ue_s", "nonprot_embb_viol", "ll_viol", "rlf")


# ---------------------------------------------------------------------------------------------- seeds / jobs
def check_seed(seed: int) -> int:
    seed = int(seed)
    assert SEED_LO <= seed <= SEED_HI, f"seed {seed} outside the step-2 DEV block [{SEED_LO}, {SEED_HI}]"
    for lo, hi in FORBIDDEN:
        assert not lo <= seed <= hi, f"seed {seed} inside a forbidden block [{lo}, {hi}]"
    return seed


def jobs(stage: str = "dev") -> list:
    """[(seed, arm)] in shard order (seed-major)."""
    if stage not in STAGES:
        raise SystemExit(f"--stage in {STAGES}")
    return [(check_seed(sd), arm) for sd in SEEDS for arm in ARMS]


def registry_check():
    """{"block": bool} if SEED_REGISTRY.json is present (repo checkout), else None (cloud bundle)."""
    path = os.path.join(S.ROOT, REGISTRY_DOC)
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    rngs = []

    def walk(o):
        if isinstance(o, list) and len(o) == 2 and all(isinstance(v, int) for v in o):
            rngs.append(o)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
    walk(d.get("E6", {}))
    return {"block_184000_185999": any(lo <= REG_BLOCK[0] and hi >= REG_BLOCK[1] for lo, hi in rngs)}


# ---------------------------------------------------------------------------------------------- policies (obs only)
def direction(r) -> str:
    """F's request direction (instr.py): from the request's knob / cur / prop only."""
    typ, cur, prop = r["knob"][0], r["cur"], r["prop"]
    if typ == "carrier":
        return "car_off" if prop < cur else "car_on"
    if typ == "sleep":
        return "sleep" if prop > cur else "wake"
    if typ == "ptx":
        return "ptx_down" if prop < cur else "ptx_up"
    if typ == "prot_min":
        return "pm_up" if prop > cur else "pm_down"
    return typ


class BlanketGate:
    """F's ``gate:<spec>`` (instr.py make_inner): reject every request whose direction is in ``rej`` once
    obs["t"] > warmup_s (a config constant); accept everything else."""

    def __init__(self, rej, warmup_s: float):
        self.rej, self.warmup_s = frozenset(rej), float(warmup_s)
        self.n = {"rej_" + d: 0 for d in sorted(self.rej)}

    def __call__(self, obs):
        t = obs["t"]
        dec = []
        for r in obs["requests"]:
            d = direction(r)
            if t > self.warmup_s and d in self.rej:
                dec.append("reject")
                self.n["rej_" + d] += 1
            else:
                dec.append("accept")
        return {"decisions": dec, "writes": [], "rollback": []}


class SatPtxGate:
    """G's GP (g_sim.py GRef, arm "GP"): once obs["t"] >= warmup_s, reject a ptx-up request (prop > cur) iff the
    requesting cell's own prb_util (UnitArbiter observation state: mean of its last 5 delivered fast reports, NaN -> 0)
    is >= SAT_UTIL. ``ll_guard=True`` (M2) releases the rejection while the cell's region LL delay is rising (module
    docstring). Reads only obs."""

    def __init__(self, warmup_s: float, ll_guard: bool = False, sat: float = SAT_UTIL, ll_w: int = LL_W):
        self.warmup_s, self.ll_guard, self.sat, self.ll_w = float(warmup_s), bool(ll_guard), float(sat), int(ll_w)
        self.ua = UnitArbiter(accept_all_policy, warmup_s=1e9, record=False)   # observation state only
        self.ll = deque(maxlen=2 * self.ll_w)
        self._rise_t, self._rise = None, {}
        self.n = {"rej_ptx_up": 0, "sat_ptx_up": 0, "ll_release": 0}

    def _observe_ll(self, obs):
        for rep in obs["new_reports"]:
            if rep.get("gran") == "fast" and "ll_delay_p95" in rep:
                self.ll.append(np.asarray(rep["ll_delay_p95"], float))
                self._rise_t = None                      # window changed: drop this second's cache

    def ll_rising(self, c: int, t) -> bool:
        if self._rise_t != t:
            self._rise_t, self._rise = t, {}
        if c not in self._rise:
            ok = False
            if len(self.ll) >= 2 * self.ll_w:
                L = np.stack(list(self.ll))
                cells = list(self.ua.exp[c])
                old, new = L[:self.ll_w][:, cells], L[self.ll_w:][:, cells]
                if np.isfinite(old).any() and np.isfinite(new).any():
                    ok = bool(np.nanmean(new) > np.nanmean(old))
            self._rise[c] = ok
        return self._rise[c]

    def __call__(self, obs):
        self.ua(obs)                                     # config history / fast-report window; returns accept-all
        self._observe_ll(obs)
        reqs, now = obs["requests"], obs["t"]
        if now < self.warmup_s:
            return {"decisions": ["accept"] * len(reqs), "writes": [], "rollback": []}
        M = self.ua.mediators(obs)
        dec = []
        for r in reqs:
            k = r["knob"]
            d = "accept"
            if k[0] == "ptx" and r["prop"] > r["cur"] and np.nan_to_num(M[int(k[1]), J_PRB]) >= self.sat:
                self.n["sat_ptx_up"] += 1
                if self.ll_guard and self.ll_rising(int(k[1]), now):
                    self.n["ll_release"] += 1
                else:
                    d = "reject"
                    self.n["rej_ptx_up"] += 1
            dec.append(d)
        return {"decisions": dec, "writes": [], "rollback": []}


def make_arbiter(arm: str, warmup_s: float):
    if arm in B_GATES:
        return BlanketGate(B_GATES[arm], warmup_s)
    if arm == "M1":
        return SatPtxGate(warmup_s, ll_guard=False)
    if arm == "M2":
        return SatPtxGate(warmup_s, ll_guard=True)
    return S.make_arbiter(arm)                           # freeze | noarb (None) | sub:...


# ---------------------------------------------------------------------------------------------- one job
def peak_rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: KiB
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.PeakWorkingSetSize / 2 ** 20
    except Exception:  # noqa: BLE001
        pass
    return None


def run_job(seed: int, arm: str, lf: float, smoke: bool = False, short=None) -> dict:
    check_seed(seed)
    if arm not in ARMS:
        raise KeyError(arm)
    sd = S._seed(seed, smoke)
    cfg = S.make_cfg("*" if arm == "freeze" else PAIR, STRATUM, sd, lf)
    if smoke and short:
        cfg = dataclasses.replace(cfg, scored_s=float(short))
    arb = make_arbiter(arm, cfg.warmup_s)
    t_wall, t_cpu = time.time(), time.process_time()
    env, _ = S.run_env(cfg, arb)
    cpu = time.process_time() - t_cpu
    return {"kind": "job", "schema": SCHEMA, "key": [seed, arm], "seed": seed, "cfg_seed": sd, "arm": arm,
            "smoke": smoke, "short": short, "episode_s": env.total_s, "warmup_s": cfg.warmup_s, "load_factor": lf,
            **S.outcome(env), "policy_counts": dict(getattr(arb, "n", {}) or {}), "cpu_s": round(cpu, 2),
            "secs": round(time.time() - t_wall, 1), "rss_mb": peak_rss_mb()}


# ---------------------------------------------------------------------------------------------- run
def run(stage, part, out, smoke=False, short=None, arms=None):
    if stage not in STAGES:
        raise SystemExit(f"--stage in {STAGES}")
    if short and not smoke:
        raise SystemExit("--short is smoke-only")
    reg = registry_check()
    if not smoke and reg is not None and not all(reg.values()) and not os.environ.get("E6P_S2_ALLOW_UNREGISTERED"):
        raise SystemExit(f"seed block {REG_BLOCK} not registered in {REGISTRY_DOC}: {reg}")
    if arms:
        bad = [a for a in arms if a not in ARMS]
        if bad:
            raise SystemExit(f"unknown arms {bad}; arms: {ARMS}")
    state = S.load_state()
    lf = S.lf_of(state, STRATUM)
    i, k = map(int, part.split("/"))
    head = S.header(stage, part, smoke, state)
    head.update(kind="header", schema=SCHEMA, plan=PLAN, driver="e6p_step2_dev", registry=reg,
                consts={"pair": PAIR, "stratum": STRATUM, "load_factor": lf, "seeds": [SEED_LO, SEED_HI],
                        "arms": ARMS, "b_gates": {a: sorted(v) for a, v in B_GATES.items()}, "sat_util": SAT_UTIL,
                        "ll_w": LL_W, "smoke": {"short": short} if smoke else None, "arms_filter": arms})
    S._append(out, head)
    done = {tuple(r["key"]) for r in S._read(out) if r.get("kind") == "job" and r.get("smoke") == smoke}
    t0, n, first_seed = time.time(), 0, None
    J = jobs(stage)
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
    S._append(out, {"kind": "close", "stage": stage, "part": part, "n_jobs": n, "n_total": len(J),
                    "secs": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------------------------- summary
def load_records(paths, allow_smoke=False) -> dict:
    """{seed: {arm: record}} (last record wins per (seed, arm, smoke))."""
    R = {}
    for p in paths:
        for r in S._read(p):
            if r.get("kind") != "job" or r.get("schema") != SCHEMA or (r.get("smoke") and not allow_smoke):
                continue
            R.setdefault(int(r["seed"]), {})[r["arm"]] = r
    return R


def _arrays(R, seeds, arm):
    return {f: np.array([float(R[s][arm][f]) for s in seeds]) for f in SUM_FIELDS}


def _pooled(A, idx=None):
    """Pooled Gate A quantities of one arm; idx (B, n) resamples -> arrays (B,), None -> scalars."""
    if idx is None:
        s = {f: v.sum() for f, v in A.items()}
    else:
        s = {f: v[idx].sum(1) for f, v in A.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"V": 3600.0 * s["prot_viol"] / np.maximum(s["prot_ue_s"], 1e-9), "E": s["energy_j"],
                "svr": 3600.0 * s["viol_ue_s"] / np.maximum(s["ue_s"], 1e-9),
                "nonprot_embb_viol": s["nonprot_embb_viol"], "ll_viol": s["ll_viol"], "rlf": s["rlf"]}


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.where(a > 0, np.inf, 1.0))


def _ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95))] if x.size else [None, None]


def arm_stats(R, arm, n_boot=N_BOOT, base=BASE_ARM):
    """Gate A R / retention / guard ratios / eligibility of ``arm`` on the seeds that have every reference arm, the arm
    and (for the difference) ``base``; paired seed bootstrap. None if no such seed."""
    need = set(REF_ARMS) | {arm} | ({base} if base in {a for s in R for a in R[s]} else set())
    seeds = sorted(s for s in R if need <= set(R[s]))
    if not seeds:
        return None
    n = len(seeds)
    arms_used = sorted(need)
    A = {a: _arrays(R, seeds, a) for a in arms_used}
    P = {a: _pooled(A[a]) for a in arms_used}
    V_AA = P["noarb"]["V"]
    ref = min(SINGLES, key=lambda a: P[a]["V"])
    V_ref = P[ref]["V"]
    den = V_AA - V_ref
    Ef, EA = P["freeze"]["E"], P[A_ARM]["E"]
    p = P[arm]
    Rv = (V_AA - p["V"]) / den if abs(den) > 1e-12 else float("nan")
    ret = (Ef - p["E"]) / (Ef - EA) if abs(Ef - EA) > 1e-12 else float("nan")
    matched = bool(Ef - p["E"] >= X_MATCH * (Ef - EA) - 1e-9)
    g = {kk: float(_ratio(np.float64(p[kk]), np.float64(P["noarb"][kk]))) for kk in GUARD_KEYS}
    guard_ok = bool(all(p[kk] <= GUARD * P["noarb"][kk] + 1e-9 for kk in GUARD_KEYS))
    out = {"arm": arm, "n_seeds": n, "seeds": seeds, "V": float(p["V"]), "V_AA": float(V_AA), "V_ref": float(V_ref),
           "ref_arm": ref, "den": float(den), "R": float(Rv), "R_fixed_den": float((V_AA - p["V"]) / DEN_GATE_A),
           "retention": float(ret), "matched": matched, "guard_ratio": g, "guard_ok": guard_ok,
           "eligible": bool(matched and guard_ok), "E_kj_per_ep": float(p["E"] / n / 1e3)}
    if n_boot:
        rng = np.random.default_rng([BOOT_TAG, n])
        idx = rng.integers(0, n, (n_boot, n))
        Pb = {a: _pooled(A[a], idx) for a in arms_used}
        Vref_b = np.min([Pb[a]["V"] for a in SINGLES], 0)
        den_b = Pb["noarb"]["V"] - Vref_b
        with np.errstate(divide="ignore", invalid="ignore"):
            Rb = (Pb["noarb"]["V"] - Pb[arm]["V"]) / den_b
            retb = (Pb["freeze"]["E"] - Pb[arm]["E"]) / (Pb["freeze"]["E"] - Pb[A_ARM]["E"])
        out.update(R_ci90=_ci(Rb), retention_ci90=_ci(retb),
                   guard_ratio_ci90={kk: _ci(_ratio(Pb[arm][kk], Pb["noarb"][kk])) for kk in GUARD_KEYS})
        if base in Pb:
            with np.errstate(divide="ignore", invalid="ignore"):
                dRb = (Pb[base]["V"] - Pb[arm]["V"]) / den_b
            out.update(dR_vs_base=float((P[base]["V"] - p["V"]) / den) if abs(den) > 1e-12 else float("nan"),
                       dR_vs_base_ci90=_ci(dRb), base=base)
    return out


def _f(v):
    return f"{v:+.3f}" if v is not None and np.isfinite(v) else "   nan"


def summary(paths, json_out=None, allow_smoke=False, n_boot=N_BOOT):
    R = load_records(paths, allow_smoke)
    present = sorted({a for s in R for a in R[s]}, key=lambda a: ARMS.index(a) if a in ARMS else 99)
    complete = sorted(s for s in R if set(ARMS) <= set(R[s]))
    report = {"plan": PLAN, "schema": SCHEMA, "n_seeds_any": len(R), "n_seeds_complete": len(complete),
              "missing": {a: sorted(s for s in R if a not in R[s]) for a in ARMS if any(a not in R[s] for s in R)},
              "arms": {}}
    print(f"== E6-P step-2 DEV (P3 surge-L40): {len(R)} seeds with records, {len(complete)} complete over {len(ARMS)} arms")
    if report["missing"]:
        print("   missing (arm: seeds):", {a: len(v) for a, v in report["missing"].items()})
    for arm in present:
        st = arm_stats(R, arm, n_boot)
        if st is None:
            continue
        report["arms"][arm] = st
    if report["arms"]:
        r0 = next(iter(report["arms"].values()))
        print(f"   V_AA {r0['V_AA']:.2f}  V_ref {r0['V_ref']:.2f} ({r0['ref_arm']})  den {r0['den']:.2f} psvr "
              f"(Gate A v2 den {DEN_GATE_A:.2f})  n {r0['n_seeds']}")
        print(f"   {'arm':22s} {'n':>3} {'V':>7} {'R':>7} {'R 90% CI':>17} {'R-R(B1)':>8} {'90% CI':>17} "
              f"{'Rfix':>6} {'ret':>6} {'svr':>5} {'nonp':>5} {'ll':>5} {'rlf':>5} elig  cpu_s")
        for arm, st in report["arms"].items():
            ci = st.get("R_ci90", [None, None])
            dci = st.get("dR_vs_base_ci90", [None, None])
            cpu = np.mean([R[s][arm]["cpu_s"] for s in st["seeds"]])
            print(f"   {arm:22s} {st['n_seeds']:>3} {st['V']:7.2f} {_f(st['R']):>7} [{_f(ci[0])},{_f(ci[1])}] "
                  f"{_f(st.get('dR_vs_base')):>8} [{_f(dci[0])},{_f(dci[1])}] {_f(st['R_fixed_den']):>6} "
                  f"{st['retention']:6.3f} " + " ".join(f"{st['guard_ratio'][k]:5.2f}" for k in GUARD_KEYS)
                  + f"  {'Y' if st['eligible'] else 'n'}{'' if st['matched'] else 'E'}{'' if st['guard_ok'] else 'G'}"
                  f"  {cpu:6.1f}")
        cpu_all = [R[s][a]["cpu_s"] for s in R for a in R[s]]
        report["cpu_h_total"] = float(sum(cpu_all) / 3600)
        report["cpu_s_per_job"] = float(np.mean(cpu_all))
        print(f"   eligible = retention >= {X_MATCH} and every guard ratio <= {GUARD} (E = energy fail, G = guard fail);"
              f" CPU {report['cpu_h_total']:.2f} h total, {report['cpu_s_per_job']:.1f} s/job")
    if json_out:
        json.dump(report, open(json_out, "w", newline="\n"), indent=1, default=S._js)
    return report


def list_jobs():
    J = jobs("dev")
    print("plan:", PLAN)
    print("registry:", registry_check())
    print(f"dev: {len(J)} jobs = {len(SEEDS)} seeds ({SEED_LO}..{SEED_HI}) x {len(ARMS)} arms {ARMS}")


def cli(argv, stage=None):
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    flags = {x for x in rest if x in ("--smoke", "--allow-smoke")}
    rest = [x for x in rest if x not in flags]
    if cmd == "run":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        run(a.get("--stage") or stage or "dev", a["--part"], a["--out"], smoke="--smoke" in flags,
            short=float(a["--short"]) if a.get("--short") else None,
            arms=[x for x in a["--arms"].split(",") if x] if a.get("--arms") else None)
    elif cmd == "summary":
        files, js = [], None
        it = iter(rest)
        for x in it:
            if x == "--json":
                js = next(it)
            elif x != "--in":
                files.extend(f for f in x.split(",") if f)
        summary(files, json_out=js, allow_smoke="--allow-smoke" in flags)
    elif cmd == "list":
        list_jobs()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])

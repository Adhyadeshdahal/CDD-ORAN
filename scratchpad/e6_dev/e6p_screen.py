"""E6-P conflict screen (Gate A v2) driver. Implements docs/benchmark/E6P_SCREEN_PROTOCOL.md (frozen together with
docs/benchmark/E6P_SPEC.md at E6P_SPEC_COMMIT). Nothing here chooses a protocol value; every constant below is copied
from those documents and checked against them (``freeze-check``) and against the plant config (``check_frozen``).

CLI (repo root, PYTHONPATH=.; in the Kaggle bundle the same file is e6dev/e6p_screen.py):
  python scratchpad/e6_dev/e6p_screen.py list
  python scratchpad/e6_dev/e6p_screen.py run --stage {0a,0b,0c,1,2,3} --part i/k --out FILE.jsonl [--smoke]
  python scratchpad/e6_dev/e6p_stage1.py run --part i/k --out FILE.jsonl     # one wrapper per stage (cloud.py only
                                                                              # passes `run --part --out`)
  python scratchpad/e6_dev/e6p_screen.py summary --in FILES... [--write-state] [--json OUT] [--allow-smoke]
  python scratchpad/e6_dev/e6p_screen.py estimate [--arms freeze,sub,noarb,prio,cell,lock,qacm,oracle] [--scored S]
  python scratchpad/e6_dev/e6p_screen.py m5a [--seconds N] [--seed DEV]      # local only (needs git)
  python scratchpad/e6_dev/e6p_screen.py freeze-check

Stage selection: ``--stage``, else env E6P_STAGE, else the wrapper's name (e6p_stage<S>.py).
State between stages (load factors from 0a, the (pair, stratum) gates for stages 2 / 3, inert-xApp pairs from 0b) is
written by ``summary --write-state`` to scratchpad/e6_dev/e6p_state.json AND to the mirror e6p_state_json.py (cloud.py
bundles only *.py). Stages read it; nothing is hard-coded.

--smoke: DEV plumbing test only. Protocol seeds are remapped to DEV seeds (seed % 31), episodes are shortened, and every
record is flagged smoke=True (summary ignores them unless --allow-smoke). Never use protocol seeds locally.

Declared implementation choices where the protocol is silent (reported, not tuned): see IMPLEMENTATION_NOTES.
"""
from __future__ import annotations

import copy
import dataclasses
import glob
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(os.path.dirname(HERE), "MANIFEST.json")):      # Kaggle bundle: <root>/e6dev/
    ROOT, BUNDLE = os.path.dirname(HERE), True
else:
    ROOT, BUNDLE = os.path.dirname(os.path.dirname(HERE)), False
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cdd_oran.envs.e6 import config as C  # noqa: E402
from cdd_oran.envs.e6 import sim  # noqa: E402
from cdd_oran.envs.e6.baselines import (  # noqa: E402
    CellPriorityLock,
    KnobLock,
    all_subsets,
    freeze,
    priority,
    region_hindsight,
    region_subset,
    subset,
)
from cdd_oran.envs.e6.env import E6Env  # noqa: E402
from cdd_oran.envs.e6.ric import knob_set  # noqa: E402

IMPLEMENTATION_NOTES = [
    "QACM (arm 8) needs a KPI predictor fitted on logs; the protocol names no fit seeds. It is fitted per (pair, stratum)"
    " on accept-all logs of E6 DEV seeds QACM_FIT_SEEDS (registered E6 DEV range 0-30), same scenario and load_factor.",
    "QACM E6-P mapping (published.MANIFEST_P / QOS_P): SliceGuarantee KPI = per-cell protected below-floor share, QoS"
    " 0.05 (sg_viol_hi); Coverage KPI = per-cell edge-SINR p5, QoS -6 dB (cov_sinr_low_db); energy has no QoS.",
    "Oracle 'half' mode = cdd_oran.decision.plans slew rule (half the step in actuator quanta; single-quantum steps"
    " alternate accept/reject) with ptx 1 dB / prot_min 0.05 quanta, not wg3_oracle.py's legacy cur+(prop-cur)/2"
    " (which the actuator grid rounds to a no-op or a full step for carrier / sleep / prot_min).",
    "Oracle rollouts hold the plan for the whole horizon (wg3_oracle.py semantics): first second + H-1 seconds.",
    "Oracle stage 2 of the search only accepts ADMISSIBLE trials (energy constraint); an admissible trial beats an"
    " inadmissible incumbent; if the final plan is inadmissible accept-all is used.",
    "rho_sign: advantage = lexicographic sign of (AA - plan) on (protected violated UE-s, all-UE violated UE-s) over"
    " the rollout; re-drawn copy = env.copy(reseed=int(t)); a tie (sign 0) kept on both counts as kept.",
    "Bootstrap RNG default_rng([6611, pair_idx, stratum]) uses pair_idx = 1, 2, 3 for P1, P2, P3; the lower bound is"
    " the 10th percentile of V_AA - V_ref over resamples, V_ref re-minimised per resample; arm eligibility is fixed at"
    " the point estimate; reported intervals are 5th-95th percentiles.",
    "Non-protected eMBB violated UE-s = embb_viol - prot_viol (exact: prot_floor_bps == EMBB_THP_TARGET_BPS and the"
    " same 0.2 s activity gate / outage rule; asserted).",
    "M1(b) is evaluated per mechanism seed (each of the 4 seeds must reach 90 %); M4 is run per pair deployment (the"
    " xApp variant draw depends on its index in the deployment), 7 (pair, xApp) arms + freeze per seed.",
    "M5(a) (E6PConfig() default bit-identical to E6-scn-v1) is a local command `m5a` comparing against the pre-E6-P"
    " commit 1284869 on a DEV seed (M5 has no protocol seed); M5(b) runs in stage 0b on 150020.",
    "Stage 0c times the oracle for P1 base-L40 (the heaviest load of the primary pair); scores are not recorded.",
    "Freeze is run once per (seed, stratum) with mix 'none' and no xApps (plant identical: xApps never touch the plant"
    " under freeze) and shared by all pairs.",
]

# ---------------------------------------------------------------------------------------------- frozen record
PROTOCOL = "E6P_SCREEN_PROTOCOL.md Gate A v2 (E6P-v1)"
E6P_SPEC_COMMIT = "2afbc1c8da381375891c438efea6500f7f5e5467"
FROZEN_SHA256 = {"docs/benchmark/E6P_SPEC.md": "5db7ef17b6a5fa3f53b49dac3e28e58fecca25190a2248f4cca6f34f6b4b21a9",
                 "docs/benchmark/E6P_SCREEN_PROTOCOL.md":
                     "9705653c8e797af159bf3b74424b49d85818dd7dc38bf1727797142399be90d0",
                 "docs/benchmark/SEED_REGISTRY.json": "5ab935fe27816dd3c3d9e257f7a6af5d66d4911fb992772321df49144200f0f5"}
FROZEN_E6P = dict(ptx_scope="macro", ptx_range_db=(-9.0, 3.0), ptx_init_db=0.0, ptx_grid_db=1.0, ptx_max_step_db=3.0,
                  ptx_min_interval_s=10.0, prot_frac=0.2, prot_floor_bps=2e6, prot_min_backlog_s=0.2,
                  prot_min_range=(0.0, 0.5), prot_min_init=0.0, prot_min_grid=0.05, prot_min_max_step=0.10,
                  prot_min_interval_s=5.0, edge_pct=5.0, pico_sleep_ho=True, es_pico=True,
                  pes_cadence_s=10.0, pes_u_low=0.30, pes_u_high=0.80, pes_hold_s=60.0, pes_step_db=3.0,
                  cov_cadence_s=10.0, cov_sinr_low_db=-6.0, cov_sinr_ok_db=-3.0, cov_margin_db=3.0,
                  cov_floor_frac=0.05, cov_step_db=3.0, sg_cadence_s=5.0, sg_viol_hi=0.05, sg_viol_lo=0.01,
                  sg_slack=0.5, sg_hold_s=10.0, sg_step=0.05)
E6_FROZEN = dict(n_ue=300, n_pico=3, mobility="ped", kpm="nominal", update=False, warmup_s=120.0, scored_s=600.0)

PAIRS = {"P1": dict(idx=1, mix="ES", p_xapps=("SliceGuarantee",), A=("ES",), B=("SliceGuarantee",)),
         "P2": dict(idx=2, mix="none", p_xapps=("PowerES", "Coverage"), A=("PowerES",), B=("Coverage",)),
         "P3": dict(idx=3, mix="ES", p_xapps=("PowerES", "SliceGuarantee"), A=("ES", "PowerES"),
                    B=("SliceGuarantee",))}
STRATA = (("base", "L10"), ("base", "L40"), ("surge", "L10"), ("surge", "L40"))    # stratum = 2*scenario + load
TARGET_L = {"L10": 0.10, "L40": 0.40}
LOADCAL = dict(lo=0.05, hi=3.0, max_iter=12, tol=0.01, fit=tuple(range(150000, 150010)),
               check=tuple(range(150010, 150020)))
MECH_SEEDS = tuple(range(150020, 150024))
PILOT_SEED = 150060
N_SCREEN, N_M4 = 8, 4
X_MATCH, SCREEN_FLOOR, GUARD = 0.90, 0.01, 1.10
M4_MATERIAL = {"psvr": 36.0, "lowsinr": 0.01}   # 8a: same 1 % of UE-time floor as criterion 1
LAMBDA_MIN, MATERIAL, R_OR_MIN, HEADROOM, RHO_MIN = 0.15, 36.0, 0.50, 0.10, 0.70
N_BOOT, BOOT_TAG, LB_Q = 10_000, 6611, 0.10
ORACLE = dict(D=20, H=90, n_glob=6, n_loc=2)
LOCK_S = 60.0
QACM_FIT_SEEDS = (20, 21, 22, 23)            # [OURS] E6 DEV seeds (protocol silent, see IMPLEMENTATION_NOTES)
DEV_EST_SEED = 3
STAGES = ("0a", "0b", "0c", "1", "2", "3")
_SD = os.environ.get("E6P_STATE_DIR", HERE)          # override only for DEV smoke runs (keeps the real state clean)
STATE_JSON = os.path.join(_SD, "e6p_state.json")
STATE_PY = os.path.join(_SD, "e6p_state_json.py")
OWN_KPI = {"ES": "energy", "PowerES": "energy", "SliceGuarantee": "psvr", "Coverage": "lowsinr"}


def m4_seed(s, j):
    return 150020 + 10 * s + j


def screen_seed(s, j):
    return 150100 + 10 * s + j


def deployed(pair):
    p = PAIRS[pair]
    return (("ES",) if p["mix"] == "ES" else ()) + p["p_xapps"]


def a_arm(pair):
    return "sub:" + "+".join(PAIRS[pair]["A"])


# ---------------------------------------------------------------------------------------------- state / config
def _sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def docs_check():
    """{file: ok} for the frozen documents when they exist (repo checkout); None in the Kaggle bundle."""
    if BUNDLE:
        return None
    return {f: os.path.exists(os.path.join(ROOT, f)) and _sha_lf(os.path.join(ROOT, f)) == h
            for f, h in FROZEN_SHA256.items()}


def load_state():
    js = json.load(open(STATE_JSON)) if os.path.exists(STATE_JSON) else None
    py = None
    if os.path.exists(STATE_PY):
        ns = {}
        exec(open(STATE_PY).read(), ns)                                     # noqa: S102 (our own data mirror)
        py = json.loads(ns["STATE"])
    if js is not None and py is not None and js != py:
        raise SystemExit("e6p_state.json and e6p_state_json.py differ: re-run summary --write-state")
    return js if js is not None else (py or {})


def write_state(state):
    s = json.dumps(state, indent=1, sort_keys=True)
    open(STATE_JSON, "w", newline="\n").write(s + "\n")
    open(STATE_PY, "w", newline="\n").write('"""Mirror of e6p_state.json (cloud.py bundles only *.py). Written by '
                                            'e6p_screen.py summary --write-state; do not edit."""\n'
                                            f"STATE = r'''{s}'''\n")


def lf_of(state, stratum):
    lfs = state.get("load_factor", {})
    key = STRATA[stratum][1]
    if key not in lfs or lfs[key] is None:
        raise SystemExit(f"no calibrated load_factor for {key}: run stage 0a and summary --write-state first")
    return float(lfs[key])


def strata_available(state):
    return [s for s in range(4) if state.get("load_factor", {}).get(STRATA[s][1]) is not None]


def make_cfg(pair, stratum, seed, lf, smoke=False):
    if pair == "*":
        mix, px = "none", ()
    else:
        mix, px = PAIRS[pair]["mix"], PAIRS[pair]["p_xapps"]
    kw = dict(E6_FROZEN)
    if smoke:
        kw.update(warmup_s=20.0, scored_s=20.0)
    cfg = C.E6Config(seed=int(seed), mix=mix, scenario=STRATA[stratum][0], load_factor=float(lf),
                     e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=px), **kw)
    check_frozen(cfg, smoke)
    return cfg


def check_frozen(cfg, smoke=False):
    P = cfg.e6p
    bad = {k: (getattr(P, k), v) for k, v in FROZEN_E6P.items() if getattr(P, k) != v}
    for k, v in E6_FROZEN.items():
        if smoke and k in ("warmup_s", "scored_s"):
            continue
        if getattr(cfg, k) != v:
            bad[k] = (getattr(cfg, k), v)
    ref = C.E6Config()
    for f in ("surge_mult", "surge_radius_m", "surge_loc", "surge_onset_frac", "surge_ramp_frac", "surge_hold_frac",
              "surge_speed_mps", "frac_ll", "frac_embb", "embb_files_per_s", "be_files_per_s", "ramp", "m_tau_s",
              "m_sigma", "kpm_delay_s", "kpm_drop", "pl_offset_db", "xapp_threshold_scale"):
        if getattr(cfg, f) != getattr(ref, f):
            bad[f] = (getattr(cfg, f), getattr(ref, f))
    if not (P.ptx_on and P.prot_on):
        bad["ptx_on/prot_on"] = (P.ptx_on, P.prot_on)
    if P.prot_floor_bps != C.EMBB_THP_TARGET_BPS or P.prot_min_backlog_s != 0.2:
        bad["nonprot_embb_identity"] = (P.prot_floor_bps, C.EMBB_THP_TARGET_BPS)
    if bad:
        raise SystemExit(f"config differs from the frozen E6P values: {bad}")


# ---------------------------------------------------------------------------------------------- episodes
def outcome(env):
    S, st = env.plant.sla, env.stats
    o = {k: float(S[k]) for k in ("viol_ue_s", "ue_s", "ll_viol", "embb_viol", "outage_viol", "severe", "energy_j",
                                  "rlf", "ho", "pingpong", "prot_viol", "prot_ue_s", "lowsinr_ue_s")}
    o.update(psvr=3600.0 * S["prot_viol"] / max(S["prot_ue_s"], 1e-9),
             svr=3600.0 * S["viol_ue_s"] / max(S["ue_s"], 1e-9), energy_kwh=S["energy_j"] / 3.6e6,
             nonprot_embb_viol=float(S["embb_viol"] - S["prot_viol"]))
    o.update({"st_" + k: v for k, v in st.items()})
    return o


def run_env(cfg, arb, wg3=True, instrument=None, energy_trace=False, log=False, churn_cap=None):
    env = E6Env(cfg, log=log, wg3=wg3, churn_cap=churn_cap)
    if instrument is not None:
        instrument(env)
    E = [0.0] if energy_trace else None
    while env.sec < env.total_s:
        env.step(arb)
        if E is not None:
            E.append(float(env.plant.sla["energy_j"]))
    return env, E


def stage1_arms(pair):
    dep = deployed(pair)
    arms = ["sub:" + x for x in dep] + ["noarb"]
    arms += ["sub:" + "+".join(s) for s in all_subsets(dep) if 1 < len(s) < len(dep)]
    perms = list(itertools.permutations(dep))
    arms += ["prio:" + ">".join(o) for o in perms] + ["cell:" + ">".join(o) for o in perms]
    return arms + ["lock", "qacm"]


def arm_group(arm):
    """Protocol sec. 5 arm number."""
    if arm == "freeze":
        return 1
    if arm == "noarb":
        return 3
    if arm.startswith("sub:"):
        return 2 if "+" not in arm else 4
    return {"prio": 5, "cell": 6, "lock": 7, "qacm": 8, "hind": 9, "oracle": 10}[arm.split(":")[0]]


def make_arbiter(arm):
    if arm == "freeze":
        return freeze
    if arm == "noarb":
        return None
    if arm.startswith("sub:"):
        return subset(arm[4:].split("+"))
    if arm.startswith("prio:"):
        order = tuple(arm[5:].split(">"))
        return lambda obs: priority(obs, order)
    if arm.startswith("cell:"):
        return CellPriorityLock(tuple(arm[5:].split(">")), hold_s=LOCK_S)
    if arm == "lock":
        return KnobLock(lease_s=LOCK_S)
    raise KeyError(arm)


# ---------------------------------------------------------------------------------------------- QACM (arm 8)
_QACM = {}


def qacm_predictor(pair, stratum, lf, smoke):
    from cdd_oran.envs.e6.published import MANIFEST_P, KPIPredictor
    key = (pair, stratum, float(lf), smoke)
    if key not in _QACM:
        t = time.time()
        kpis = tuple(dict.fromkeys(MANIFEST_P[x]["kpi"] for x in deployed(pair)))
        logs = []
        for seed in QACM_FIT_SEEDS[:1] if smoke else QACM_FIT_SEEDS:
            env, _ = run_env(make_cfg(pair, stratum, seed, lf, smoke), None, log=True)
            logs.append(env.log)
        pred = KPIPredictor(kpis=kpis, seed=0).fit(logs)
        _QACM[key] = (pred, {"fit_seeds": list(QACM_FIT_SEEDS[:1] if smoke else QACM_FIT_SEEDS), "kpis": list(kpis),
                             "report": {f"{a}|{b}": v for (a, b), v in pred.report.items()},
                             "fit_secs": round(time.time() - t, 1)})
    return _QACM[key]


def qacm_arbiter(pred):
    from cdd_oran.envs.e6.published import MANIFEST_P, QACM, QOS_P
    return QACM(pred, manifest=MANIFEST_P, qos=QOS_P)


# ---------------------------------------------------------------------------------------------- oracle (arm 10)
MODES = ("accept", "reject", "half", "lock")
RB_WINDOW = 60.0


def _quantum():
    from cdd_oran.decision.plans import QUANTUM
    P = C.E6PConfig()
    return dict(QUANTUM, ptx=P.ptx_grid_db, prot_min=P.prot_min_grid)


QUANTUM_P = _quantum()


def half_step(knob, cur, prop, hs):
    """cdd_oran.decision.plans.half_step (slew rule) with the E6-P quanta (ptx 1 dB, prot_min 0.05)."""
    q = QUANTUM_P.get(knob[0])
    n = abs(prop - cur) / q if q else 0.0
    if n >= 2 - 1e-6:
        k = int(np.ceil(round(n / 2, 6)))
        return ("modify", float(cur + np.sign(prop - cur) * k * q))
    c = hs.get(knob, 0)
    hs[knob] = c + 1
    phase = sum(int(x) for x in knob[1:]) % 2
    return "accept" if (c + phase) % 2 == 0 else "reject"


def uniform(xapps, mode, rb=0):
    return {"mode": {x: mode for x in xapps}, "rb": rb}


def is_accept_all(plan):
    return all(not rp["rb"] and all(m == "accept" for m in rp["mode"].values()) for rp in plan.values())


def decide(plan, obs, site, D, last_change, rb_at, first, hs):
    dec = []
    for r in obs["requests"]:
        m = plan[int(site[r["knob"][1]])]["mode"][r["xapp"]]
        dec.append(half_step(r["knob"], r["cur"], r["prop"], hs) if m == "half" else
                   "accept" if m == "accept" else "reject" if m == "reject" else ("lock", float(D)))
    rb = []
    if first:
        now = obs["t"]
        rb = [k for k, t in last_change.items()
              if now - t <= RB_WINDOW and plan[int(site[k[1]])]["rb"] and rb_at.get(k) != t]
    return {"decisions": dec, "writes": [], "rollback": rb}


def lex_better(a, b):
    return a[0] < b[0] - 1e-9 or (abs(a[0] - b[0]) <= 1e-9 and a[1] < b[1] - 1e-9)


def lex_sign(aa, pl):
    for d in (aa[0] - pl[0], aa[1] - pl[1]):
        if abs(d) > 1e-9:
            return int(np.sign(d))
    return 0


class Oracle:
    """Budgeted WG3 oracle (protocol sec. 5 arm 10) on wg3_oracle.py mechanics: every D s, stage 1 = best of
    accept-all, freeze, incumbent and n_glob random network-wide plans; stage 2 = one coordinate-descent pass over the
    regions in random order with n_loc random local alternatives each. Objective over the H-s rollout on the TRUE tape:
    (protected violated UE-s, all-UE violated UE-s), lexicographic. A plan is admissible iff the rollout's cumulative
    scored energy at the end of the horizon <= Ef(t) - X (Ef(t) - EA(t)) (freeze / A-alone traces on the same tape);
    no admissible plan -> accept-all. Churn parity via the env's churn_cap. Privileged, not deployable.
    ``bound=None`` (timing only) makes every plan admissible."""

    def __init__(self, env, xapps, Ef, EA, D=20, H=90, n_glob=6, n_loc=2, rho=True):
        self.env, self.xapps = env, tuple(xapps)
        self.Ef, self.EA = Ef, EA
        self.D, self.H, self.n_glob, self.n_loc, self.rho = D, H, n_glob, n_loc, rho
        self.site = np.asarray(env.plant.lay.cell_site)
        self.regions = sorted({int(x) for x in self.site})
        self.plan = {g: uniform(self.xapps, "accept") for g in self.regions}
        self.until, self.rb_at, self.hs = -1.0, {}, {}
        self.n_roll, self.picks, self.deviations, self.epochs = 0, {}, [], []

    def aa(self):
        return {g: uniform(self.xapps, "accept") for g in self.regions}

    def random_region(self, r):
        return {"mode": {x: MODES[int(r.integers(len(MODES)))] for x in self.xapps}, "rb": int(r.integers(2))}

    def bound(self, sec):
        if self.Ef is None:
            return np.inf
        sec = min(sec, len(self.Ef) - 1)
        return self.Ef[sec] - X_MATCH * (self.Ef[sec] - self.EA[sec])

    def roll(self, plan, obs, reseed=None):
        sim_ = self.env.copy(reseed=reseed)
        s0 = dict(sim_.plant.sla)
        hs = dict(self.hs)
        sim_.step_apply(decide(plan, obs, self.site, self.D, sim_.last_change, self.rb_at, True, hs))
        for _ in range(self.H - 1):
            if sim_.sec >= sim_.total_s:
                break
            o = sim_.step_propose()
            sim_.step_apply(decide(plan, o, self.site, self.D, sim_.last_change, self.rb_at, False, hs))
        self.n_roll += 1
        S = sim_.plant.sla
        return (S["prot_viol"] - s0["prot_viol"], S["viol_ue_s"] - s0["viol_ue_s"],
                bool(S["energy_j"] <= self.bound(sim_.sec) + 1e-6))

    def choose(self, obs):
        t = obs["t"]
        r = np.random.default_rng([self.env.cfg.seed, int(t), 13])
        cands = [self.aa(), {g: uniform(self.xapps, "reject") for g in self.regions}, dict(self.plan)]
        for _ in range(self.n_glob):
            p = self.random_region(r)
            cands.append({g: p for g in self.regions})
        sc = [self.roll(c, obs) for c in cands]
        i_best = None
        for i, s in enumerate(sc):
            if s[2] and (i_best is None or lex_better(s, sc[i_best])):
                i_best = i
        best, s_best, ok = (dict(cands[i_best]), sc[i_best], True) if i_best is not None else (cands[0], sc[0], False)
        changed = 0
        for g in r.permutation(self.regions):
            for _ in range(self.n_loc):
                trial = dict(best)
                trial[int(g)] = self.random_region(r)
                s = self.roll(trial, obs)
                if s[2] and (not ok or lex_better(s, s_best)):
                    best, s_best, ok, changed = trial, s, True, changed + 1
        if not ok:
            best, s_best = cands[0], sc[0]
        key = (f"stage1={['accept', 'freeze', 'incumbent'][i_best] if i_best is not None and i_best < 3 else 'random' if i_best is not None else 'none_admissible'}"
               f"|local={min(changed, 5)}|final_ok={int(ok)}")
        self.picks[key] = self.picks.get(key, 0) + 1
        if self.rho and not is_accept_all(best):
            sa, sp = self.roll(cands[0], obs, reseed=int(t)), self.roll(best, obs, reseed=int(t))
            self.deviations.append({"t": t, "aa_true": list(sc[0]), "plan_true": list(s_best),
                                    "aa_redraw": list(sa), "plan_redraw": list(sp),
                                    "sign_true": lex_sign(sc[0], s_best), "sign_redraw": lex_sign(sa, sp)})
        return best

    def act(self, obs):
        first = False
        if obs["t"] >= self.env.cfg.warmup_s and obs["t"] >= self.until:
            t0 = time.time()
            self.plan, self.until, first = self.choose(obs), obs["t"] + self.D, True
            self.epochs.append(round(time.time() - t0, 2))
        return decide(self.plan, obs, self.site, self.D, self.env.last_change, self.rb_at, first, self.hs)

    def run(self):
        env = self.env
        while env.sec < env.total_s:
            obs = env.step_propose()
            dec = self.act(obs)
            env.step_apply(dec)
            for k in dec["rollback"]:
                if k in env.last_change:
                    self.rb_at[k] = env.last_change[k]


def oracle_job(pair, stratum, seed, lf, smoke, record_scores=True):
    cfg = make_cfg(pair, stratum, seed, lf, smoke)
    t = time.time()
    ef_env, Ef = run_env(cfg, freeze, energy_trace=True)
    ea_env, EA = run_env(cfg, make_arbiter(a_arm(pair)), energy_trace=True)
    aa_env, _ = run_env(cfg, None)
    cap = aa_env.stats["changes"]
    t_ref = time.time() - t
    env = E6Env(cfg, log=False, wg3=True, churn_cap=cap)
    orc = Oracle(env, deployed(pair), Ef, EA, **ORACLE)
    t = time.time()
    orc.run()
    rec = {"churn_cap": cap, "n_roll": orc.n_roll, "picks": orc.picks, "n_epochs": len(orc.epochs),
           "epoch_secs": orc.epochs, "ref_secs": round(t_ref, 1), "oracle_secs": round(time.time() - t, 1)}
    if record_scores:
        rec.update(outcome(env), deviations=orc.deviations,
                   ref_check={"freeze_prot_viol": ef_env.plant.sla["prot_viol"],
                              "A_prot_viol": ea_env.plant.sla["prot_viol"],
                              "AA_prot_viol": aa_env.plant.sla["prot_viol"],
                              "AA_energy_j": aa_env.plant.sla["energy_j"]})
    else:
        rec["n_deviations"] = len(orc.deviations)
    return rec


# ---------------------------------------------------------------------------------------------- arm 9
def energy_ok(o, of, oa):
    return (of["energy_j"] - o["energy_j"]) >= X_MATCH * (of["energy_j"] - oa["energy_j"]) - 1e-9


def guard_ok(o, aa):
    return all(o[k] <= GUARD * aa[k] + 1e-9 for k in ("svr", "nonprot_embb_viol", "ll_viol", "rlf"))


def hind_job(pair, stratum, seed, lf, smoke):
    cfg = make_cfg(pair, stratum, seed, lf, smoke)
    of = outcome(run_env(cfg, freeze)[0])
    oa = outcome(run_env(cfg, make_arbiter(a_arm(pair)))[0])
    site = np.asarray(E6Env(cfg, log=False).plant.lay.cell_site)
    regions = sorted({int(x) for x in site})
    search = regions[:2] if smoke else regions            # smoke: plumbing only (2 regions searched)
    outs, aa = {}, {}

    def evaluate(kb):
        o = outcome(run_env(cfg, region_subset(kb, site))[0])
        if not aa:
            aa.update(o)                                   # first evaluation = every region keeps every xApp = AA
        outs[json.dumps(sorted((g, list(v)) for g, v in kb.items()))] = o
        return o["psvr"], energy_ok(o, of, oa) and guard_ok(o, aa)

    res = region_hindsight(evaluate, search, deployed(pair), all_regions=regions)
    best = outs[json.dumps(sorted((g, list(v)) for g, v in res["best"].items()))]
    full = tuple(deployed(pair))
    return {**best, "best_keep": {str(g): list(v) for g, v in res["best"].items() if tuple(v) != full},
            "best_ok": res["best_ok"], "n_eval": res["n_eval"],
            "path": [{"diff": {str(g): list(v) for g, v in p["keep"].items() if tuple(v) != full},
                      "psvr": p["val"], "ok": p["ok"]} for p in res["path"]]}


# ---------------------------------------------------------------------------------------------- stage 0a
def measure_L(args):
    """Network mean over the 24 cells (equal weight) and all scored seconds of used PRBs / N_PRB, freeze arm, base.
    Reads ONLY the prb_used / prb_cap counters (no SLA, protected or energy quantity is computed or returned)."""
    seed, lf, smoke = args
    cfg = make_cfg("*", 0, seed, lf, smoke)
    env = E6Env(cfg, log=False, wg3=True)
    p, per_s = env.plant, []
    orig = p.take_counters

    def take():
        c = orig()
        per_s.append((c["prb_used"].sum(1) / max(c["ticks"], 1) / C.N_PRB, c["prb_cap"].copy()))
        return c

    p.take_counters = take
    while env.sec < env.total_s:
        env.step(freeze)
    w = int(cfg.warmup_s)
    return float(np.mean([u.mean() for u, _ in per_s[w:]]))                 # seconds w+1..total (window >= warmup)


def _pool_map(fn, args, workers):
    if workers <= 1 or len(args) <= 1:
        return [fn(a) for a in args]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, args)


def run_0a(i, k, out, smoke):
    mine = [t for n, t in enumerate(("L10", "L40")) if n % k == i]
    workers = int(os.environ.get("E6P_WORKERS", max(1, (os.cpu_count() or 1) // min(k, 2))))
    prior = [r for r in _read(out) if r.get("stage") == "0a"]
    fit = LOADCAL["fit"][:2] if smoke else LOADCAL["fit"]
    chk = LOADCAL["check"][:2] if smoke else LOADCAL["check"]
    iters_max = 2 if smoke else LOADCAL["max_iter"]
    for tgt in mine:
        if any(r["kind"] == "0a_result" and r["target"] == tgt for r in prior):
            continue
        T, lo, hi = TARGET_L[tgt], LOADCAL["lo"], LOADCAL["hi"]
        hist = sorted((r for r in prior if r["kind"] == "0a_iter" and r["target"] == tgt), key=lambda r: r["iter"])
        final, trace = None, []
        for it in range(iters_max):
            lf = (lo + hi) / 2
            if it < len(hist):
                if abs(hist[it]["lf"] - lf) > 1e-12:
                    raise SystemExit("0a resume: recorded bisection path differs")
                L = hist[it]["L_mean"]
            else:
                t0 = time.time()
                Ls = _pool_map(measure_L, [(_seed(s, smoke), lf, smoke) for s in fit], workers)
                L = float(np.mean(Ls))
                _append(out, {"kind": "0a_iter", "stage": "0a", "target": tgt, "iter": it, "lf": lf, "L_seeds": Ls,
                              "L_mean": L, "secs": round(time.time() - t0, 1), "smoke": smoke})
            trace.append({"iter": it, "lf": lf, "L_mean": L})
            if abs(L - T) <= LOADCAL["tol"]:
                final = lf
                break
            lo, hi = (lf, hi) if L < T else (lo, lf)
        res = {"kind": "0a_result", "stage": "0a", "target": tgt, "target_L": T, "load_factor": final,
               "feasible": final is not None, "iters": trace, "smoke": smoke}
        if final is not None:
            Lc = _pool_map(measure_L, [(_seed(s, smoke), final, smoke) for s in chk], workers)
            res.update(check_L_mean=float(np.mean(Lc)), check_L_min=float(np.min(Lc)), check_L_max=float(np.max(Lc)),
                       check_L_seeds=Lc)
        _append(out, res)
        print(json.dumps({x: res.get(x) for x in ("target", "load_factor", "feasible", "check_L_mean")}), flush=True)


# ---------------------------------------------------------------------------------------------- stage 0b
def _copy_plant(p):
    return copy.deepcopy(p, {id(p.gm): p.gm, id(p.lay): p.lay})


def m1_job(seed, lf, smoke):
    cfg = make_cfg("*", 0, seed, lf, smoke)
    env = E6Env(cfg, log=False, wg3=True)
    while env.sec < int(cfg.warmup_s):
        env.step(freeze)
    base, now = env.plant, float(env.sec)
    occ0, serv0, iu0 = base.occ.copy(), base.serv.copy(), base.int_until.copy()
    A = _copy_plant(base)
    A.tick()
    gA = A._gains()
    idx = np.arange(base.n)
    pw = 10 ** (gA / 10) / 1000.0
    load = pw * occ0[None, :]
    i_int = load.sum(1) - load[idx, serv0]
    rows = []
    for c in np.nonzero(base.lay.is_macro)[0]:
        c = int(c)
        own = serv0 == c
        edge = ~own & (gA[:, c] >= gA[idx, serv0] - 6.0)
        row = {"cell": c, "occ_c": float(occ0[c])}               # occ 0 -> c causes no interference (diagnostic)
        for off in (-3.0, 3.0):
            B = _copy_plant(base)
            knob_set(B, ("ptx", c), off, now)
            B.tick()
            valid = (A.serv == serv0) & (B.serv == serv0) & (A.int_until == iu0) & (B.int_until == iu0)
            real = (B.sinr_ewma - A.sinr_ewma) / 0.2
            Ip = np.where(own, i_int, i_int + load[:, c] * (10 ** (off / 10) - 1))
            ana = 10 * np.log10((i_int + sim.NOISE_W) / (Ip + sim.NOISE_W)) + off * own
            e = edge & valid
            inp = (B.l3[:, c] - A.l3[:, c]) / sim.A_L3
            car, rho = int(B.n_car[c]), float(B.rho[c])
            PB = B.ctr["energy_j"][c] / C.TICK_S
            got = PB - (car * (C.MACRO_P0_W + C.MACRO_DP * rho * C.MACRO_PMAX_W) +
                        (C.MACRO_NTRX - car) * C.MACRO_SLEEP_W / C.MACRO_NTRX)
            want = car * C.MACRO_DP * rho * C.MACRO_PMAX_W * (10 ** (off / 10) - 1)
            row[str(int(off))] = {"a_err_db": float(np.max(np.abs(real - ana)[valid])) if valid.any() else None,
                                  "n_valid": int(valid.sum()), "n_edge": int(e.sum()),
                                  "edge_mean_db": float(real[e].mean()) if e.any() else None,
                                  "c_err_db": float(np.max(np.abs(inp - off))), "d_err_w": float(abs(got - want))}
        rows.append(row)
    a_ok = all(r[o]["a_err_db"] is not None and r[o]["a_err_db"] <= 0.01 for r in rows for o in ("-3", "3"))
    with_edge = [r for r in rows if r["-3"]["n_edge"] and r["3"]["n_edge"]]
    b_frac = (sum(r["-3"]["edge_mean_db"] > 0 and r["3"]["edge_mean_db"] < 0 for r in with_edge) / len(with_edge)
              if with_edge else float("nan"))
    c_ok = all(r[o]["c_err_db"] <= 1e-9 for r in rows for o in ("-3", "3"))
    d_ok = all(r[o]["d_err_w"] <= 1e-9 for r in rows for o in ("-3", "3"))
    loaded = [r for r in with_edge if r["occ_c"] > 0]
    b_loaded = (sum(r["-3"]["edge_mean_db"] > 0 and r["3"]["edge_mean_db"] < 0 for r in loaded) / len(loaded)
                if loaded else float("nan"))
    return {"M1": {"a": a_ok, "b_frac": b_frac, "b_frac_loaded_macros_diag": b_loaded,
                   "n_idle_macros_with_edge_ues": len(with_edge) - len(loaded), "b": bool(b_loaded >= 0.9), "c": c_ok, "d": d_ok,
                   "pass": bool(a_ok and b_loaded >= 0.9 and c_ok and d_ok), "rows": rows}}   # 8a: loaded macros


def m2_job(seed, lf, smoke):
    cfg = dataclasses.replace(make_cfg("*", 0, seed, lf, smoke), frac_ll=0.0, embb_files_per_s=0.0,
                              be_files_per_s=0.0)                          # scheduler unit test: no arrivals

    def plant(prot_min, prot_q, other_q, ll=0.0):
        p = sim.Plant(cfg)
        for _ in range(5):
            p.tick()
        ok = p.int_until <= p.t * C.TICK_S
        cnt = np.bincount(p.serv[p.prot & ok], minlength=p.nc)
        oth = np.bincount(p.serv[~p.prot & ok], minlength=p.nc)
        c = int(np.argmax(np.minimum(cnt, oth) * p.lay.is_macro))
        p.prot_min[:] = prot_min
        p.ll_ratio[:] = ll
        p.q[:] = np.where(p.prot, prot_q, other_q)
        p.ctr = p._new_counters()
        p.tick()
        return p, c, int(cnt[c]), int(oth[c])

    p0, c, n_p, n_o = plant(0.0, 0.0, 1e9)
    if n_p < 1 or n_o < 1:
        return {"M2": {"pass": False, "setup": f"no macro with protected and other UEs (prot {n_p}, other {n_o})"}}
    p1, _, _, _ = plant(0.5, 0.0, 1e9)
    keys = ("prb_used", "prb_rsv", "prb_cap", "bits", "energy_j")
    a = bool(np.array_equal(p1.q, p0.q) and all(np.array_equal(p1.ctr[k], p0.ctr[k]) for k in keys))
    util = [(q.ctr["prb_used"].sum(1) + q.ctr["prb_rsv"]) / np.maximum(q.ctr["prb_cap"], 1e-9) for q in (p0, p1)]
    d = bool(np.array_equal(util[0], util[1]) and np.array_equal(p0.ctr["prb_rsv"], p1.ctr["prb_rsv"]))
    pb, _, _, _ = plant(0.5, 2e3, 1e9)
    cap = pb.ctr["prb_cap"][c]
    dem = pb.ctr["prot_dem"][c] * cap
    b = bool(0 < dem < 0.5 * cap and abs(pb.ctr["prot_used"][c] - dem) <= 1e-9 * max(1.0, dem)
             and abs(pb.ctr["prb_used"][c].sum() - cap) <= 1e-9 * cap)
    pc, _, _, _ = plant(0.5, 1e9, 1e9)
    c_ = bool(pc.ctr["prot_used"][c] >= 0.5 * pc.ctr["prb_cap"][c] - 1e-9)
    pe, _, _, _ = plant(0.5, 1e9, 1e9, ll=0.5)
    e = bool(np.all(pe.ctr["prot_min_prb"] <= pe.ctr["prb_cap"] * (1 - 0.5) + 1e-9))
    return {"M2": {"a": a, "b": b, "c": c_, "d": d, "e": e, "pass": a and b and c_ and d and e, "cell": c,
                   "n_prot": n_p, "n_other": n_o}}


def m3_job(seed, lf, smoke):
    p = sim.Plant(make_cfg("*", 0, seed, lf, smoke))
    c = int(np.nonzero(p.lay.is_macro)[0][0])
    pc = int(np.nonzero(~p.lay.is_macro)[0][0])

    def fresh():
        p.ctr = p._new_counters()

    def ticks(n):
        for _ in range(n):
            p.tick()

    def earth(car):
        rho, lin = p.rho[c], 10 ** (p.ptx_off[c] / 10)
        return car * (C.MACRO_P0_W + C.MACRO_DP * rho * C.MACRO_PMAX_W * lin) + \
            (C.MACRO_NTRX - car) * C.MACRO_SLEEP_W / C.MACRO_NTRX

    out = {}
    fresh()
    ticks(1)
    out["cap_on"] = p.ctr["prb_cap"][c] == C.N_PRB and abs(p.ctr["energy_j"][c] / C.TICK_S - earth(2)) <= 1e-9
    knob_set(p, ("carrier", c), 1, p.t * C.TICK_S)
    fresh()
    ticks(1)
    out["cap_off"] = p.ctr["prb_cap"][c] == C.N_PRB / 2 and abs(p.ctr["energy_j"][c] / C.TICK_S - earth(1)) <= 1e-9
    knob_set(p, ("carrier", c), 2, p.t * C.TICK_S)
    fresh()
    n = int(round(C.CARRIER_ON_S / C.TICK_S)) - 1
    ticks(n)
    out["reactivating"] = abs(p.ctr["prb_cap"][c] - C.N_PRB / 2 * n) <= 1e-9
    ticks(3)
    fresh()
    ticks(1)
    out["cap_back"] = p.ctr["prb_cap"][c] == C.N_PRB
    knob_set(p, ("sleep", pc), 1.0, p.t * C.TICK_S)
    fresh()
    ticks(1)
    out["pico_sleep"] = p.ctr["prb_cap"][pc] == 0 and abs(p.ctr["energy_j"][pc] - C.PICO_SLEEP_W * C.TICK_S) <= 1e-12
    knob_set(p, ("sleep", pc), 0.0, p.t * C.TICK_S)
    fresh()
    ticks(int(round(C.PICO_WAKE_S / C.TICK_S)) - 1)
    out["waking"] = p.ctr["prb_cap"][pc] == 0
    ticks(3)
    fresh()
    ticks(1)
    out["pico_back"] = p.ctr["prb_cap"][pc] == C.N_PRB
    out = {k: bool(v) for k, v in out.items()}
    return {"M3": dict(out, **{"pass": all(out.values())})}


def m5b_job(seed, lf, smoke):
    cfg = make_cfg("*", 0, seed, lf, smoke)
    e_on = E6Env(cfg, log=False, wg3=True)
    e_off = E6Env(dataclasses.replace(cfg, e6p=C.E6PConfig()), log=False, wg3=True)
    ok, first_bad = True, None
    while e_on.sec < e_on.total_s:
        e_on.step(None)
        e_off.step(None)
        same = (np.array_equal(e_on.plant.q, e_off.plant.q) and np.array_equal(e_on.plant.rho, e_off.plant.rho)
                and e_on.plant.sla["energy_j"] == e_off.plant.sla["energy_j"]
                and np.array_equal(e_on.plant.serv, e_off.plant.serv))
        if not same:
            ok, first_bad = False, e_on.sec
            break
    return {"M5b": {"pass": ok, "first_diff_sec": first_bad, "seconds": int(e_on.sec)}}


def instrument_m6(env, store):
    p = env.plant
    store.update(sleeps=[], rlf=[])
    orig_sh, orig_rlf = p.sleep_handover, p._rlf

    def sh(c, now):
        g = p._gains()
        pw = 10 ** (g / 10) / 1000.0
        off = p.asleep | (p.waking_until > now)
        for u in np.nonzero(p.serv == c)[0]:
            tgt = int(np.argmax(np.where(off, -np.inf, p.l3[u])))
            i = (pw[u] * p.occ).sum() - pw[u, tgt] * p.occ[tgt]
            s = 10 * np.log10(pw[u, tgt] / (i + sim.NOISE_W) + 1e-30)
            store["sleeps"].append((int(u), float(now), bool(s > C.QOUT_DB)))
        return orig_sh(c, now)

    def rlf(u, now, ho_fail=False):
        store["rlf"].append((int(u), float(now)))
        return orig_rlf(u, now, ho_fail)

    p.sleep_handover, p._rlf = sh, rlf


def m6_count(store):
    n = 0
    for u, t, elig in store["sleeps"]:
        if elig:
            n += sum(1 for v, tr in store["rlf"] if v == u and t <= tr <= t + 2.0)
    return {"n_sleep_ue_events": len(store["sleeps"]), "n_eligible": sum(e for _, _, e in store["sleeps"]),
            "rlf_within_2s": n}


# ---------------------------------------------------------------------------------------------- jobs
def _seed(seed, smoke):
    return seed % 31 if smoke else seed


def units_for(stage, state):
    """List of units (each a list of job keys (stage, pair, stratum, seed, arm)); unit u goes to shard u % k."""
    U = []
    if stage == "0b":
        U += [[("0b", "*", 0, s, "M1")] for s in MECH_SEEDS]
        U += [[("0b", "*", 0, MECH_SEEDS[0], m)] for m in ("M2", "M3", "M5b")]
        for s in strata_available(state):
            for j in range(N_M4):
                seed = m4_seed(s, j)
                U.append([("0b", "*", s, seed, "freeze")])
                U += [[("0b", pair, s, seed, "sub:" + x)] for pair in PAIRS for x in deployed(pair)]
    elif stage == "0c":
        U = [[("0c", "P1", 1, PILOT_SEED, "oracle_pilot")]]
    elif stage == "1":
        ok = strata_available(state)
        live = [p for p in PAIRS if p not in state.get("not_screenable_pairs", [])]   # M4 amendment: skip inert pairs
        U += [[("1", pair, s, screen_seed(s, j), "qacm") for j in range(N_SCREEN)] for pair in live for s in ok]
        for s in ok:
            for j in range(N_SCREEN):
                seed = screen_seed(s, j)
                U.append([("1", "*", s, seed, "freeze")])
                U += [[("1", pair, s, seed, a)] for pair in live for a in stage1_arms(pair) if a != "qacm"]
    elif stage == "2":
        U = [[("2", pair, s, screen_seed(s, j), "oracle")] for pair, s in state.get("stage1_crit1", [])
             for j in range(N_SCREEN)]
    elif stage == "3":
        U = [[("3", pair, s, screen_seed(s, j), "hind")] for pair, s in state.get("stage2_crit12", [])
             for j in range(N_SCREEN)]
    for u in U:
        for key in u:
            assert 150000 <= key[3] < 150200, key                      # never the reserved confirmation range
    return U


def run_job(key, state, smoke):
    stage, pair, s, seed, arm = key
    lf = lf_of(state, s)
    sd = _seed(seed, smoke)
    t = time.time()
    rec = {"kind": "job", "key": list(key), "stage": stage, "pair": pair, "stratum": s, "seed": seed, "arm": arm,
           "load_factor": lf, "smoke": smoke}
    if stage == "0b" and arm == "M1":
        rec.update(m1_job(sd, lf, smoke))
    elif stage == "0b" and arm == "M2":
        rec.update(m2_job(sd, lf, smoke))
    elif stage == "0b" and arm == "M3":
        rec.update(m3_job(sd, lf, smoke))
    elif stage == "0b" and arm == "M5b":
        rec.update(m5b_job(sd, lf, smoke))
    elif stage in ("0b", "1") and arm not in ("qacm",):
        store = {}
        inst = (lambda env: instrument_m6(env, store)) if (stage == "0b" and arm == "sub:ES") else None
        env, _ = run_env(make_cfg(pair, s, sd, lf, smoke), make_arbiter(arm), instrument=inst)
        rec.update(outcome(env))
        if store:
            rec["M6"] = m6_count(store)
    elif arm == "qacm":
        pred, info = qacm_predictor(pair, s, lf, smoke)
        arb = qacm_arbiter(pred)
        env, _ = run_env(make_cfg(pair, s, sd, lf, smoke), arb)
        rec.update(outcome(env), qacm=info, qacm_stats=arb.stats)
    elif arm == "oracle":
        rec.update(oracle_job(pair, s, sd, lf, smoke))
    elif arm == "oracle_pilot":
        rec.update(oracle_job(pair, s, sd, lf, smoke, record_scores=False))
    elif arm == "hind":
        rec.update(hind_job(pair, s, sd, lf, smoke))
    else:
        raise KeyError(key)
    rec["secs"] = round(time.time() - t, 1)
    return rec


def _read(path):
    out = []
    for f in glob.glob(path) if any(ch in path for ch in "*?[") else [path]:
        if not os.path.exists(f):
            continue
        for line in open(f):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _append(out, rec):
    with open(out, "a", newline="\n") as f:
        f.write(json.dumps(rec, default=_js) + "\n")


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def header(stage, part, smoke, state):
    man = {}
    mp = os.path.join(ROOT, "MANIFEST.json")
    if BUNDLE and os.path.exists(mp):
        m = json.load(open(mp))
        man = {"git_head": m.get("git_head"), "e6_dirty": m.get("e6_dirty")}
    else:
        try:
            man = {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                              text=True, timeout=20).stdout.strip()}
        except Exception:  # noqa: BLE001
            man = {}
    try:
        import cloud
        nenv = cloud.numeric_env()
    except Exception as e:  # noqa: BLE001
        nenv = {"error": repr(e)}
    return {"kind": "header", "stage": stage, "part": part, "smoke": smoke, "protocol": PROTOCOL,
            "e6p_spec_commit": E6P_SPEC_COMMIT, "frozen_sha256": FROZEN_SHA256, "docs_check": docs_check(),
            "code": man, "numeric_env": nenv, "scenario_version": C.SCENARIO_VERSION, "state": state,
            "cpus": os.cpu_count(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run(stage, part, out, smoke=False):
    if stage not in STAGES:
        raise SystemExit(f"--stage in {STAGES}")
    dc = docs_check()
    if dc is not None and not all(dc.values()) and not smoke:
        raise SystemExit(f"frozen documents changed since {E6P_SPEC_COMMIT[:7]}: {dc}")
    i, k = map(int, part.split("/"))
    state = load_state()
    _append(out, header(stage, part, smoke, state))
    t0 = time.time()
    if stage == "0a":
        run_0a(i, k, out, smoke)
        _append(out, {"kind": "close", "stage": stage, "part": part, "secs": round(time.time() - t0, 1)})
        return
    done = {tuple(r["key"]) for r in _read(out) if r.get("kind") == "job" and r.get("smoke") == smoke}
    n = 0
    units = units_for(stage, state)
    for u, unit in enumerate(units):
        if u % k != i:
            continue
        for key in unit:
            if key in done:
                continue
            rec = run_job(key, state, smoke)
            _append(out, rec)
            n += 1
            print(json.dumps({x: rec.get(x) for x in ("stage", "pair", "stratum", "seed", "arm", "secs")}), flush=True)
            if smoke and stage in ("1", "0b") and n >= 3:
                break
        if smoke and n >= 3:
            break
    _append(out, {"kind": "close", "stage": stage, "part": part, "n_jobs": n, "n_units": len(units),
                  "secs": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------------------------- summary
def _pool(recs):
    pv, pu = sum(r["prot_viol"] for r in recs), sum(r["prot_ue_s"] for r in recs)
    return {"V": 3600.0 * pv / max(pu, 1e-9), "E": sum(r["energy_j"] for r in recs),
            "svr": 3600.0 * sum(r["viol_ue_s"] for r in recs) / max(sum(r["ue_s"] for r in recs), 1e-9),
            "nonprot_embb_viol": sum(r["nonprot_embb_viol"] for r in recs),
            "ll_viol": sum(r["ll_viol"] for r in recs), "rlf": sum(r["rlf"] for r in recs),
            "lowsinr": sum(r["lowsinr_ue_s"] for r in recs) / max(sum(r["ue_s"] for r in recs), 1e-9),
            "changes": float(np.mean([r["st_changes"] for r in recs]))}


def _vboot(recs, idx):
    pv = np.array([r["prot_viol"] for r in recs])
    pu = np.array([r["prot_ue_s"] for r in recs])
    return 3600.0 * pv[idx].sum(1) / np.maximum(pu[idx].sum(1), 1e-9)


def summarise_0a(recs):
    res = {}
    for r in recs:
        if r.get("kind") == "0a_result":
            res[r["target"]] = r
    return res


def summarise_0b(R, strata):
    out = {}
    for m in ("M1", "M2", "M3", "M5b"):
        rs = [r for k, r in R.items() if k[0] == "0b" and k[4] == m]
        if rs:
            out[m] = {"pass": all(r[m]["pass"] for r in rs), "n": len(rs),
                      "detail": [{x: v for x, v in r[m].items() if x != "rows"} for r in rs]}
    m4, inert_pairs = {}, set()
    for pair in PAIRS:
        for x in deployed(pair):
            per = {}
            for s in strata:
                seeds = [m4_seed(s, j) for j in range(N_M4)]
                al = [R.get(("0b", pair, s, sd, "sub:" + x)) for sd in seeds]
                fr = [R.get(("0b", "*", s, sd, "freeze")) for sd in seeds]
                if not all(al) or not all(fr):
                    continue
                kpi = OWN_KPI[x]

                def val(rs, kpi=kpi):
                    p = _pool(rs)
                    return {"energy": p["E"], "psvr": p["V"], "lowsinr": p["lowsinr"]}[kpi]
                acts = float(np.mean([a["st_changes"] for a in al])) >= 1.0
                imp = val(al) < val(fr)
                n_imp = sum(val([a]) < val([f]) for a, f in zip(al, fr, strict=True))
                material = kpi == "energy" or val(fr) >= M4_MATERIAL[kpi]      # protocol 8a (M4 amendment)
                per[s] = {"acts": acts, "material": bool(material),
                          "mean_changes": float(np.mean([a["st_changes"] for a in al])),
                          "own_alone": val(al), "own_freeze": val(fr), "pooled_improves": imp, "n_seeds_improve": n_imp,
                          "ok": (not acts) or (not material) or (imp and n_imp >= 3)}
            if not per:
                continue
            n_act = sum(v["acts"] and v["material"] for v in per.values())
            inert = bool(per) and n_act < 2
            ok = bool(per) and all(v["ok"] for v in per.values()) and not inert
            m4[f"{pair}:{x}"] = {"strata": per, "inert": inert, "pass": ok}
            if not ok:
                inert_pairs.add(pair)            # failing or inert xApp -> pair not screenable (reported, not fixed)
    if m4:
        out["M4"] = {"pass": any(p not in inert_pairs for p in PAIRS), "detail": m4, "inert_pairs": sorted(inert_pairs),
                     "rule": "8a: evaluated only in strata with material freeze own-KPI; per-xApp; pairs with a "
                             "failing/inert xApp are not screenable; M4 passes if >= 1 pair remains screenable"}
    m6 = [r["M6"] for k, r in R.items() if k[0] == "0b" and k[1] == "P1" and k[2] == 0 and "M6" in r]
    if m6:
        out["M6"] = {"pass": sum(x["rlf_within_2s"] for x in m6) == 0, "n_seeds": len(m6),
                     "rlf_within_2s": sum(x["rlf_within_2s"] for x in m6),
                     "n_eligible": sum(x["n_eligible"] for x in m6)}
    return out, sorted(inert_pairs)


def analyse_pair_stratum(R, pair, s):
    seeds = [screen_seed(s, j) for j in range(N_SCREEN)]
    fr = [R.get(("1", "*", s, sd, "freeze")) for sd in seeds]
    if not all(fr):
        return None
    arms = ["freeze"] + stage1_arms(pair) + ["hind", "oracle"]
    per = {"freeze": fr}
    for a in arms[1:]:
        st = {"hind": "3", "oracle": "2"}.get(a, "1")
        rs = [R.get((st, pair, s, sd, a)) for sd in seeds]
        if all(rs):
            per[a] = rs
    need = [a for a in stage1_arms(pair)]
    if not all(a in per for a in need):
        return {"complete": False, "missing": [a for a in need if a not in per]}
    P = {a: _pool(rs) for a, rs in per.items()}
    Ef, EA, AA = P["freeze"]["E"], P[a_arm(pair)]["E"], P["noarb"]
    screenable = Ef - EA >= SCREEN_FLOOR * Ef

    def matched(a):
        return Ef - P[a]["E"] >= X_MATCH * (Ef - EA) - 1e-9

    def guard(a):
        return all(P[a][k] <= GUARD * AA[k] + 1e-9 for k in ("svr", "nonprot_embb_viol", "ll_viol", "rlf"))

    singles = ["sub:" + x for x in deployed(pair)]
    ref = min(singles, key=lambda a: P[a]["V"])
    V_AA, V_ref = AA["V"], P[ref]["V"]
    den = V_AA - V_ref
    Lam = den / V_ref if V_ref > 0 else (np.inf if den > 0 else np.nan)
    table = {}
    for a in P:
        R_a = (V_AA - P[a]["V"]) / den if abs(den) > 1e-12 else np.nan
        table[a] = {"group": arm_group(a), "V": P[a]["V"], "E_kwh": P[a]["E"] / 3.6e6,
                    "retention": (Ef - P[a]["E"]) / (Ef - EA) if abs(Ef - EA) > 1e-12 else np.nan,
                    "matched": matched(a), "guard": guard(a), "eligible": matched(a) and guard(a), "R": R_a,
                    "changes": P[a]["changes"]}
    static_arms = [a for a in table if 4 <= table[a]["group"] <= 9 and table[a]["eligible"]]
    R_static = max([table[a]["R"] for a in static_arms], default=-np.inf)
    has_or, has_hind = "oracle" in P, "hind" in P
    R_or = (table["oracle"]["R"] if table["oracle"]["eligible"] else 0.0) if has_or else None
    # paired-seed bootstrap
    rng = np.random.default_rng([BOOT_TAG, PAIRS[pair]["idx"], s])
    idx = rng.integers(0, N_SCREEN, (N_BOOT, N_SCREEN))
    Vb = {a: _vboot(rs, idx) for a, rs in per.items()}
    Vref_b = np.min([Vb[a] for a in singles], 0)
    d_b = Vb["noarb"] - Vref_b
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_b = d_b / Vref_b
        Rb = {a: (Vb["noarb"] - Vb[a]) / d_b for a in Vb}
    lb = float(np.quantile(d_b, LB_Q))
    c1 = bool(screenable and matched("noarb") and Lam >= LAMBDA_MIN and den >= MATERIAL and lb > 0)
    out = {"complete": True, "screenable": bool(screenable), "E_saving_A_frac": (Ef - EA) / Ef,
           "AA_matched": matched("noarb"), "AA_energy_shortfall": 1 - table["noarb"]["retention"],
           "V_AA": V_AA, "V_ref": V_ref, "ref_arm": ref, "Lambda": Lam, "V_AA_minus_V_ref": den, "lb90_diff": lb,
           "Lambda_ci90": [float(np.nanquantile(lam_b, 0.05)), float(np.nanquantile(lam_b, 0.95))],
           "R_static": R_static if static_arms else None, "R_static_arm": max(static_arms, key=lambda a: table[a]["R"])
           if static_arms else None, "c1": c1, "arms": table}
    if has_or:
        Ror_b = Rb["oracle"] if table["oracle"]["eligible"] else np.zeros(N_BOOT)
        out.update(R_or=R_or, R_or_ci90=[float(np.nanquantile(Ror_b, 0.05)), float(np.nanquantile(Ror_b, 0.95))],
                   c2=bool(c1 and R_or >= R_OR_MIN))
        devs = [d for r in per["oracle"] for d in r.get("deviations", [])]
        kept = sum(d["sign_true"] == d["sign_redraw"] for d in devs)
        rho = kept / len(devs) if devs else float("nan")
        out.update(rho_sign=rho, n_deviations=len(devs), c4=bool(devs) and rho >= RHO_MIN)
        if has_hind:
            Rs_b = np.max([Rb[a] for a in static_arms], 0) if static_arms else np.full(N_BOOT, -np.inf)
            h_b = Ror_b - Rs_b
            out.update(headroom=R_or - R_static,
                       headroom_ci90=[float(np.nanquantile(h_b, 0.05)), float(np.nanquantile(h_b, 0.95))],
                       c3=bool(out["c2"] and R_or - R_static >= HEADROOM))
    return out


def verdict(rows, inert):
    if inert:
        return "NOT SCREENABLE (inert xApp, M4)"
    S1 = [r for r in rows.values() if r and r.get("complete") and r["c1"]]
    if any(r is None or not r.get("complete") for r in rows.values()):
        pend = "stage 1 incomplete"
    else:
        pend = None
    if not S1:
        return f"INCOMPLETE ({pend})" if pend else "DEAD (no loss)"
    if any("c2" not in r for r in S1):
        return "PENDING stage 2 (oracle)"
    S12 = [r for r in S1 if r["c2"]]
    if not S12:
        return "DEAD (not recoverable)"
    if any("c3" not in r for r in S12):
        return "PENDING stage 3 (hindsight static)"
    S123 = [r for r in S12 if r["c3"]]
    if any(r["c4"] for r in S123):
        return "PASS"
    if S123:
        return "NOISE STOP"
    return "NO-EDGE STOP"


def summary(paths, write=False, json_out=None, allow_smoke=False):
    recs = [r for p in paths for r in _read(p)]
    recs = [r for r in recs if allow_smoke or not r.get("smoke")]
    state = load_state()
    new_state = dict(state)
    report = {"protocol": PROTOCOL, "e6p_spec_commit": E6P_SPEC_COMMIT, "notes": IMPLEMENTATION_NOTES}
    heads = [r for r in recs if r.get("kind") == "header"]
    report["platforms"] = sorted({json.dumps({k: (h.get("numeric_env") or {}).get(k) for k in ("platform", "numpy")})
                                  for h in heads})
    a = summarise_0a(recs)
    if a:
        report["stage0a"] = {t: {x: r.get(x) for x in ("load_factor", "feasible", "check_L_mean", "check_L_min",
                                                       "check_L_max", "iters")} for t, r in a.items()}
        new_state.setdefault("load_factor", {})
        for t, r in a.items():
            new_state["load_factor"][t] = r["load_factor"]
        new_state["l_feasible"] = {t: r["feasible"] for t, r in a.items()}
        for t, r in a.items():
            print(f"0a {t}: load_factor={r['load_factor']} feasible={r['feasible']} "
                  f"check L mean={r.get('check_L_mean')} range=[{r.get('check_L_min')}, {r.get('check_L_max')}]")
    R = {}
    for r in recs:
        if r.get("kind") == "job":
            R[tuple(r["key"])] = r
    strata = strata_available(new_state) if new_state.get("load_factor") else list(range(4))
    ob, inert = summarise_0b(R, strata)
    if ob:
        report["stage0b"] = ob
        allm = all(v["pass"] for v in ob.values())
        print("0b:", {m: v["pass"] for m, v in ob.items()}, "ALL PASS" if allm and len(ob) >= 6 else "")
        if "M4" in ob:
            for nm, v in ob["M4"]["detail"].items():
                print(f"  M4 {nm}: pass={v['pass']} inert={v['inert']} " + "; ".join(
                    f"s{s}: acts={d['acts']} {d['own_alone']:.4g} vs {d['own_freeze']:.4g} ({d['n_seeds_improve']}/4)"
                    for s, d in v["strata"].items()))
        new_state["not_screenable_pairs"] = inert
        new_state["stage0b_all_pass"] = bool(allm and len(ob) >= 6)
    pilot = [r for r in R.values() if r["arm"] == "oracle_pilot"]
    if pilot:
        p = pilot[0]
        report["stage0c"] = {x: p.get(x) for x in ("secs", "ref_secs", "oracle_secs", "n_roll", "n_epochs",
                                                   "n_deviations")}
        print("0c pilot:", report["stage0c"])
    rows_all, crit1, crit12 = {}, [], []
    for pair in PAIRS:
        rows = {s: analyse_pair_stratum(R, pair, s) for s in strata}
        if all(v is None for v in rows.values()):
            continue
        rows_all[pair] = {"rows": rows, "verdict": verdict(rows, pair in inert)}
        print(f"\n== {pair} ({'+'.join(PAIRS[pair]['A'])} vs {'+'.join(PAIRS[pair]['B'])}): "
              f"{rows_all[pair]['verdict']}")
        for s, r in rows.items():
            if r is None or not r.get("complete"):
                print(f"  stratum {s} {STRATA[s]}: incomplete {r.get('missing') if r else ''}")
                continue
            print(f"  stratum {s} {STRATA[s]}: screenable={r['screenable']} (A saves {r['E_saving_A_frac']:.3%}) "
                  f"AA matched={r['AA_matched']} shortfall={r['AA_energy_shortfall']:.3f} V_AA={r['V_AA']:.1f} "
                  f"V_ref={r['V_ref']:.1f} ({r['ref_arm']}) Lambda={r['Lambda']:.3f} CI{r['Lambda_ci90']} "
                  f"LB90={r['lb90_diff']:.1f} c1={r['c1']} R_static={r['R_static']} ({r['R_static_arm']})"
                  + (f" R_or={r['R_or']:.3f} CI{r['R_or_ci90']} c2={r['c2']} rho={r['rho_sign']:.3f} "
                     f"(n={r['n_deviations']}) c4={r['c4']}" if "R_or" in r else "")
                  + (f" headroom={r['headroom']:.3f} CI{r['headroom_ci90']} c3={r['c3']}" if "c3" in r else ""))
            for a, t in r["arms"].items():
                print(f"    [{t['group']:>2}] {a:40s} V={t['V']:8.2f} R={t['R']:7.3f} ret={t['retention']:7.3f} "
                      f"matched={int(t['matched'])} guard={int(t['guard'])} elig={int(t['eligible'])} "
                      f"churn={t['changes']:.0f}")
            if pair not in inert and r["c1"]:
                crit1.append([pair, s])
                if r.get("c2"):
                    crit12.append([pair, s])
    report["pairs"] = rows_all
    stage1_done = rows_all and all(r is not None and r.get("complete") for pair, v in rows_all.items()
                                   if pair not in inert for r in v["rows"].values())   # inert pairs are never run
    if stage1_done:
        new_state["stage1_crit1"] = crit1
    if stage1_done and all(("R_or" in v["rows"][s]) for pair, s in crit1 for v in [rows_all[pair]]):
        new_state["stage2_crit12"] = crit12
    if json_out:
        json.dump(report, open(json_out, "w", newline="\n"), indent=1, default=_js)
    if write:
        new_state["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        new_state["sources"] = sorted({os.path.abspath(f) for p in paths for f in glob.glob(p)})
        write_state(new_state)
        print("state ->", STATE_JSON, {k: new_state.get(k) for k in ("load_factor", "stage1_crit1", "stage2_crit12",
                                                                        "not_screenable_pairs")})
    return report


# ---------------------------------------------------------------------------------------------- estimate / m5a
def estimate(arms=None, scored=None):
    """Wall time of one episode per arm type and one oracle decision on DEV seed 3 (P3 deployment, base, the L40
    load_factor if calibrated else 1.0 [timing placeholder]). Scores are not printed."""
    state = load_state()
    lf = state.get("load_factor", {}).get("L40") or 1.0
    arms = arms or ["freeze", "sub", "noarb", "prio", "cell", "lock", "qacm", "oracle"]
    pair = "P3"
    kw = dict(E6_FROZEN)
    if scored:
        kw["scored_s"] = float(scored)
    cfg = C.E6Config(seed=DEV_EST_SEED, mix="ES", scenario="base", load_factor=float(lf),
                     e6p=C.E6PConfig(ptx_on=True, prot_on=True, xapps=PAIRS[pair]["p_xapps"]), **kw)
    scale = (E6_FROZEN["warmup_s"] + E6_FROZEN["scored_s"]) / (cfg.warmup_s + cfg.scored_s)
    dep = deployed(pair)
    names = {"freeze": "freeze", "sub": "sub:" + dep[0], "noarb": "noarb", "prio": "prio:" + ">".join(dep),
             "cell": "cell:" + ">".join(dep), "lock": "lock"}
    out = {"load_factor": lf, "seed": DEV_EST_SEED, "episode_s": cfg.warmup_s + cfg.scored_s}
    for a in arms:
        t = time.time()
        if a in names:
            run_env(cfg, make_arbiter(names[a]))
            out[a] = round((time.time() - t) * scale, 2)
        elif a == "qacm":
            from cdd_oran.envs.e6.published import MANIFEST_P, KPIPredictor
            env, _ = run_env(cfg, None, log=True)
            t_log = time.time() - t
            t1 = time.time()
            pred = KPIPredictor(kpis=tuple(dict.fromkeys(MANIFEST_P[x]["kpi"] for x in dep)), seed=0).fit([env.log])
            t_fit1 = time.time() - t1
            t2 = time.time()
            run_env(cfg, qacm_arbiter(pred))
            out["qacm_log_episode"] = round(t_log * scale, 2)
            out["qacm_fit_1log"] = round(t_fit1, 2)
            out["qacm"] = round((time.time() - t2) * scale, 2)
        elif a == "oracle_roll":                            # cheap proxy: 2 rollouts x the fixed 29 per decision
            c2 = dataclasses.replace(cfg, scored_s=E6_FROZEN["scored_s"])
            env = E6Env(c2, log=False, wg3=True)
            while env.sec < int(c2.warmup_s) - 1:
                env.step(None)
            obs = env.step_propose()
            orc = Oracle(env, dep, None, None, **ORACLE)
            t = time.time()
            orc.roll(orc.aa(), obs)
            orc.roll(orc.aa(), obs, reseed=1)
            out["oracle_rollout"] = round((time.time() - t) / 2, 2)
            n_dec = 3 + ORACLE["n_glob"] + 10 * ORACLE["n_loc"]
            out["oracle_decision"] = round(n_dec * out["oracle_rollout"], 2)
            out["oracle_decision_rollouts"] = n_dec
            out["oracle_decision_is_proxy"] = True
        elif a == "oracle":
            c2 = dataclasses.replace(cfg, scored_s=E6_FROZEN["scored_s"])
            env = E6Env(c2, log=False, wg3=True)
            while env.sec < int(c2.warmup_s) - 1:
                env.step(None)
            obs = env.step_propose()
            orc = Oracle(env, dep, None, None, **ORACLE)
            t = time.time()
            orc.choose(obs)
            out["oracle_decision"] = round(time.time() - t, 2)
            out["oracle_decision_rollouts"] = orc.n_roll
            t = time.time()
            orc.roll(orc.aa(), obs, reseed=1)
            out["oracle_rollout"] = round(time.time() - t, 2)
        print(json.dumps({a: out.get(a)}), flush=True)
    ep = np.mean([out[a] for a in ("freeze", "sub", "noarb", "prio", "cell", "lock") if a in out]) if any(
        a in out for a in names) else None
    if ep is not None:
        n1 = 32 * (1 + sum(len(stage1_arms(p)) for p in PAIRS))           # incl. the 12 x 8 QACM episodes
        proj = {"0a_max": 240 * ep, "0b_M4": 16 * 8 * ep + 4 * ep + 2 * ep, "1": n1 * ep}
        if "qacm" in out:
            proj["1"] += 12 * (4 * out["qacm_log_episode"] + 4 * out["qacm_fit_1log"]) + 12 * 8 * (out["qacm"] - ep)
        if "oracle_decision" in out:
            n_ep = E6_FROZEN["scored_s"] / ORACLE["D"]
            orc = 3 * ep + ep + n_ep * out["oracle_decision"] + n_ep * 2 * out.get("oracle_rollout", 0)
            proj["0c"] = orc
            proj["2_per_pair_stratum"] = N_SCREEN * orc
            proj["2_max_all12"] = 12 * N_SCREEN * orc
        proj["3_per_pair_stratum_P1P2"] = N_SCREEN * 33 * ep
        proj["3_per_pair_stratum_P3"] = N_SCREEN * 73 * ep
        out["projected_cpu_h"] = {k: round(v / 3600, 2) for k, v in proj.items()}
        out["stage1_episodes"] = n1
    print(json.dumps(out, indent=1))
    return out


def m5a(seconds=200, seed=5):
    """M5(a), local only: E6Config(e6p=E6PConfig()) under the current code vs the pre-E6-P commit 1284869, M4 mix,
    accept-all, same DEV seed; per-second plant.sla must be identical."""
    import importlib
    import shutil
    import tempfile
    if not 0 <= seed <= 30:
        raise SystemExit("m5a uses DEV seeds 0-30 only")
    tmp = tempfile.mkdtemp(prefix="e6old_")
    try:
        pkg = os.path.join(tmp, "e6old")
        os.makedirs(pkg)
        names = subprocess.run(["git", "ls-tree", "--name-only", "1284869", "cdd_oran/envs/e6/"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.split()
        for n in names:
            if n.endswith(".py"):
                src = subprocess.run(["git", "show", f"1284869:{n}"], cwd=ROOT, capture_output=True, check=True).stdout
                open(os.path.join(pkg, os.path.basename(n)), "wb").write(src)
        sys.path.insert(0, tmp)
        OC = importlib.import_module("e6old.config")
        OE = importlib.import_module("e6old.env")
        kw = dict(seed=seed, mix="M4", mobility="ped", warmup_s=60.0, scored_s=float(seconds))
        new, old = E6Env(C.E6Config(**kw), log=False), OE.E6Env(OC.E6Config(**kw), log=False)
        first = None
        while new.sec < new.total_s:
            new.step(None)
            old.step(None)
            if dict(new.plant.sla) != dict(old.plant.sla) or not np.array_equal(new.plant.q, old.plant.q):
                first = new.sec
                break
        res = {"kind": "m5a", "seed": seed, "seconds": int(new.sec), "pass": first is None, "first_diff_sec": first,
               "ref_commit": "1284869"}
        print(json.dumps(res))
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def list_jobs():
    state = load_state()
    print("state:", {k: state.get(k) for k in ("load_factor", "stage1_crit1", "stage2_crit12",
                                               "not_screenable_pairs")})
    print("0a: 2 targets (shards 0/1), <= 12 iterations x 10 freeze episodes each + 10 check episodes")
    for st in STAGES[1:]:
        try:
            U = units_for(st, state)
        except SystemExit as e:
            print(st, "blocked:", e)
            continue
        arms = {}
        for u in U:
            for key in u:
                g = key[4].split(":")[0]
                arms[g] = arms.get(g, 0) + 1
        print(f"{st}: {len(U)} units, {sum(len(u) for u in U)} jobs {arms}")
    for pair in PAIRS:
        print(pair, "stage-1 arms:", stage1_arms(pair))


def _stage_from_env():
    s = os.environ.get("E6P_STAGE")
    if s:
        return s
    base = os.path.basename(sys.argv[0])
    if base.startswith("e6p_stage") and base.endswith(".py"):
        return base[len("e6p_stage"):-3]
    return None


def cli(argv, stage=None):
    cmd, rest = argv[0], argv[1:]
    flags = {x for x in rest if x in ("--smoke", "--write-state", "--allow-smoke")}
    rest = [x for x in rest if x not in flags]
    if cmd == "run":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        st = a.get("--stage") or stage or _stage_from_env()
        run(st, a["--part"], a["--out"], smoke="--smoke" in flags)
    elif cmd == "summary":
        files, js = [], None
        it = iter(rest)
        for x in it:
            if x == "--json":
                js = next(it)
            elif x != "--in":
                files.extend(f for f in x.split(",") if f)      # space- or comma-separated
        summary(files, write="--write-state" in flags, json_out=js, allow_smoke="--allow-smoke" in flags)
    elif cmd == "estimate":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        estimate(a["--arms"].split(",") if "--arms" in a else None, a.get("--scored"))
    elif cmd == "m5a":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        m5a(int(a.get("--seconds", 200)), int(a.get("--seed", 5)))
    elif cmd == "list":
        list_jobs()
    elif cmd == "freeze-check":
        print(json.dumps({"docs": docs_check(), "commit": E6P_SPEC_COMMIT}))
        check_frozen(make_cfg("P3", 0, 150100, 1.0))
        print("config == frozen values: OK")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])

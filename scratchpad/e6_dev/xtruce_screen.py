"""xTRUCE-plant conflict screen (Gate A v2, XTS-v1) driver. Implements docs/benchmark/XTRUCE_SCREEN_PROTOCOL.md on the
plant cdd_oran/envs/xtruce/ (docs/benchmark/XTRUCE_SIM_SPEC.md). Every protocol value is a constant below, copied from
the protocol. The driver refuses non-smoke runs until FROZEN_SHA256 is set (protocol frozen) and the protocol file
hashes to it.

CLI (repo root, PYTHONPATH=.; in a cloud.py bundle the same file is e6dev/xtruce_screen.py):
  python scratchpad/e6_dev/xtruce_screen.py run --stage {0,0c,1,1x,2,3} --part i/k --out FILE.jsonl [--smoke]
  python scratchpad/e6_dev/xtruce_stage1.py run --part i/k --out FILE.jsonl    # per-stage wrappers (cloud.py
      contract, stage baked in): xtruce_stage{0,0c,1,1x,2,3}.py; e.g. cloud.py OUTDIR NAME xtruce_stage1.py
  XTRUCE_STAGE=1 python scratchpad/e6_dev/xtruce_stage.py run --part i/k --out FILE.jsonl   # local, env-selected
  python scratchpad/e6_dev/xtruce_screen.py summary --in FILES... [--write-state] [--json OUT] [--allow-smoke]
  python scratchpad/e6_dev/xtruce_screen.py estimate          # wall time per arm on DEV seed 0 (scores not printed)
  python scratchpad/e6_dev/xtruce_screen.py list | freeze-check

Output: JSONL, append-only, resumable (a job whose key is already in --out is skipped). Records: header, job, close.
--smoke: plumbing only. Protocol seeds are remapped to the disclosed DEV seeds 0-2, episodes are shortened
(W 10 + S 40), the oracle budget is cut, and every record is flagged smoke=True (summary ignores them unless
--allow-smoke). State between stages goes to xtruce_state.json + mirror xtruce_state_json.py (cloud.py bundles only
*.py); XTRUCE_STATE_DIR overrides the directory (use it for smoke runs).
"""
from __future__ import annotations

import dataclasses
import glob
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(os.path.dirname(HERE), "MANIFEST.json")):      # cloud.py bundle: <root>/e6dev/
    ROOT, BUNDLE = os.path.dirname(HERE), True
else:
    ROOT, BUNDLE = os.path.dirname(os.path.dirname(HERE)), False
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cdd_oran.envs.xtruce import (  # noqa: E402
    XAPP_NAMES,
    CellPriorityLock,
    Clipping,
    Direct,
    StaticPriority,
    XConfig,
    XEnv,
    conflict_groups,
    half,
)

# ---------------------------------------------------------------------------------------------- frozen record
PROTOCOL = "XTRUCE_SCREEN_PROTOCOL.md Gate A v2 (XTS-v1)"
DOC = "docs/benchmark/XTRUCE_SCREEN_PROTOCOL.md"
FROZEN_SHA256 = "2f86bf9dbc21713af5103b8faa5f2101609378cc08bf0644911011c7a7782359"  # frozen 2026-09-29
FROZEN_PLANT_SHA256 = {'cdd_oran/envs/xtruce/__init__.py': '2f3f8f46f404883e4daff656b53c10aed7430b00ae0cfe0567853e1aa35fa87a', 'cdd_oran/envs/xtruce/arbiters.py': '5a99e8e0f4355ef5fc97dc8374869beab57438d3f3168ec9f5227dab145c6030', 'cdd_oran/envs/xtruce/channel.py': '25e0038436aa9707f6150f6ea83bbbaa3a5c540b4cc7c9ace67150d9f181d62d', 'cdd_oran/envs/xtruce/config.py': '626bdd4d5fafb974704214b58eea493dda9d62b9cd4c19afa2462a6172f18468', 'cdd_oran/envs/xtruce/env.py': 'dd746d058bf78314d7fcf8138bd8074d5b6fa53a1e503553c9b1ee755449324d', 'cdd_oran/envs/xtruce/xapps.py': '1ed08f74f892c215c7991b2a0a0a5bb7966f44912bedeab280f93fe3b2e47c2c', 'cdd_oran/envs/xtruce/xtruce_arbiter.py': '963c0555fcc630b1a23bc26d363e150a6be0156c1181e08147547c21ed5f4d33'}  # plant at 4dae03a
PLANT_GLOB = "cdd_oran/envs/xtruce/*.py"

PAIRS = {"X1": dict(idx=1, xapps=("QoS", "ES")),
         "X2": dict(idx=2, xapps=("QoS", "ES", "IC")),
         "X3": dict(idx=3, xapps=("QoS", "ES", "LB")),
         "X4": dict(idx=4, xapps=("QoS", "ES", "IC", "LB"))}
PRIMARY = "X4"
STRATA = (("L6", "h0"), ("L6", "h50"), ("L1", "h0"), ("L1", "h50"))      # stratum = 2*load_idx + h_idx
TRAFFIC = {"L6": 6e6, "L1": 1e6}
HALLU = {"h0": 0.0, "h50": 0.5}
W_EP, S_EP = 30, 600                        # warm-up / scored epochs
SEED_BASE, SLOTS = 190000, 50
N_SCREEN, N_CAND = 40, 44                   # first 40 of j = 0..43 passing floors_reachable
MECH_J = tuple(range(44, 49))
UNIT_SEED = 190044
PILOT = ("X4", 3, 190000 + 50 * 3 + 49)     # 190199
SCREEN_RANGE = (190000, 190199)             # 190200-190399 reserved for a confirmation (never touched here)
N_XTRUCE = 10
X_MATCH, SCREEN_FLOOR = 0.90, 0.01
G_THR, G_OPER, G_OPER_SLACK = 0.90, 1.10, 0.01
LAMBDA_MIN, MATERIAL, R_OR_MIN, HEADROOM, RHO_MIN = 0.15, 36.0, 0.50, 0.10, 0.70
N_BOOT, BOOT_TAG, ORACLE_TAG, LB_Q = 10_000, 7019, 7020, 0.10
ORACLE = dict(D=10, H=40, n_glob=6, n_loc=5)
OR_G_NP, OR_G_OPER = 0.90, 0.01             # oracle trial guardrails vs the AA-plan anchor rollout (sec. 5 amendment)
LEASE = 60
M4_MATERIAL = {"V": 36.0, "maxload": 0.80}
M4_MIN_IMPROVE = 4                          # of 5 mechanism seeds
M6_MIN = 4
OWN_KPI = {"ES": "E", "QoS": "V", "IC": "intf", "LB": "maxload"}
ALLOWED_CFG = ("traffic_bps", "hallucination", "xapps", "log")
STAGES = ("0", "0c", "1", "1x", "2", "3")
SMOKE = dict(W=10, S=40, seeds=(0, 1, 2), n_screen=2, n_mech=2, oracle=dict(D=10, H=10, n_glob=1, n_loc=1),
             n_xtruce=1, max_jobs=400)
_SD = os.environ.get("XTRUCE_STATE_DIR", HERE)
STATE_JSON = os.path.join(_SD, "xtruce_state.json")
STATE_PY = os.path.join(_SD, "xtruce_state_json.py")

IMPLEMENTATION_NOTES = [
    "Churn cap: per request, an accepted request that would change its knob is dropped once env.stats['changes'] + "
    "changes already accepted this epoch reaches the AA episode count (E6: 'every further change is refused').",
    "Oracle request -> cell: env.knob_cells(knob)[0] (share/power/assoc: the UE's serving cell; pcap/sleep: the cell).",
    "Oracle 'half' on a configuration knob: accept iff (count of earlier half-decisions on that knob + knob index) is "
    "even (E6-P slew rule for single-quantum steps).",
    "Hindsight arm 9: candidate order key (not eligible, V); the incumbent is replaced only by a strictly smaller key.",
    "Scored-energy traces E_f / E_ES are indexed by env.t (E[k] = cumulative scored energy after k epochs).",
    "R_xtruce: pooled over the first N_XTRUCE screen seeds; V_AA and V_ref re-pooled over those seeds with the reference"
    " arms eligible at the full-N point estimate; xTRUCE eligibility vs freeze / ES / AA on the same seeds.",
    "Bootstrap LB: 10th percentile of V_AA - V_ref over resamples; eligibility fixed at the point estimate; reported"
    " intervals are 5th-95th percentiles.",
    "Oracle guardrails: the anchor is the accept-all-plan rollout already scored in stage 1 of each decision (sc[0]; "
    "same state, true tape, same churn cap). Non-protected bits = sum over every rollout epoch of served bits of "
    "non-protected UEs (np_all delta); e2-e5 count = env.acc['oper_viol_epochs'] delta over the rollout (every "
    "epoch, not only scored ones); slack = ceil(0.01 * H) with the constant H (1 for H = 40), also when the rollout "
    "is truncated at the episode end. No extra rollouts.",
]


# ---------------------------------------------------------------------------------------------- freeze guard
def _sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def plant_sha():
    return {os.path.relpath(f, ROOT).replace("\\", "/"): _sha_lf(f)
            for f in sorted(glob.glob(os.path.join(ROOT, PLANT_GLOB)))}


def freeze_status():
    p = os.path.join(ROOT, DOC)
    doc = _sha_lf(p) if os.path.exists(p) else None
    st = {"frozen": FROZEN_SHA256 is not None, "doc_sha256": doc, "frozen_sha256": FROZEN_SHA256,
          "plant_sha256": plant_sha(), "bundle": BUNDLE}
    if FROZEN_SHA256 is None:
        st.update(ok=False, why="protocol not frozen (FROZEN_SHA256 is None): only --smoke runs are allowed")
    elif doc is None and not BUNDLE:
        st.update(ok=False, why=f"{DOC} not found")
    elif doc is not None and doc != FROZEN_SHA256:
        st.update(ok=False, why=f"{DOC} sha256 {doc[:12]} != FROZEN_SHA256 {FROZEN_SHA256[:12]}")
    elif FROZEN_PLANT_SHA256 is not None and st["plant_sha256"] != FROZEN_PLANT_SHA256:
        st.update(ok=False, why="plant files differ from FROZEN_PLANT_SHA256")
    else:
        st.update(ok=True, why="frozen" + (" (bundle: doc absent, driver sha in MANIFEST)" if doc is None else ""))
    return st


# ---------------------------------------------------------------------------------------------- config / seeds
def ep_len(smoke):
    return (SMOKE["W"], SMOKE["S"]) if smoke else (W_EP, S_EP)


def deployed(pair):
    return XAPP_NAMES if pair == "*" else PAIRS[pair]["xapps"]


def make_cfg(pair, s, smoke=False):
    load, h = STRATA[s]
    cfg = XConfig(traffic_bps=TRAFFIC[load], hallucination=HALLU[h], xapps=tuple(deployed(pair)), log=False)
    check_frozen(cfg)
    return cfg


def check_frozen(cfg):
    ref = XConfig()
    bad = {f.name: (getattr(cfg, f.name), getattr(ref, f.name)) for f in dataclasses.fields(XConfig)
           if f.name not in ALLOWED_CFG and getattr(cfg, f.name) != getattr(ref, f.name)}
    if cfg.xapps != tuple(x for x in XAPP_NAMES if x in cfg.xapps):
        bad["xapps_order"] = cfg.xapps
    if bad:
        raise SystemExit(f"config differs from the XConfig() defaults frozen by the protocol: {bad}")


def slot(s, j):
    return SEED_BASE + SLOTS * s + j


_SCREEN = {}


def screen_seeds(s, smoke=False):
    """First N_SCREEN of j = 0..N_CAND-1 whose realisation passes floors_reachable (construction only)."""
    if smoke:
        return list(SMOKE["seeds"][:SMOKE["n_screen"]])
    if s not in _SCREEN:
        cfg = make_cfg("*", s)
        _SCREEN[s] = [slot(s, j) for j in range(N_CAND) if XEnv(cfg, slot(s, j)).floors_reachable()][:N_SCREEN]
    return _SCREEN[s]


def mech_seeds(s, smoke=False):
    return list(SMOKE["seeds"][:SMOKE["n_mech"]]) if smoke else [slot(s, j) for j in MECH_J]


def _seed(seed, smoke):
    return int(seed) % 3 if smoke else int(seed)


# ---------------------------------------------------------------------------------------------- episodes
TALLY = ("pv", "pv_thr", "pu", "E", "np_bits", "np_ue_s", "oper", "phys", "sleep_cs", "intf", "maxload", "n_ep",
         "n_conf", "pv_all", "np_all")


def new_env(cfg, seed):
    env = XEnv(cfg, seed)
    env._xt = dict.fromkeys(TALLY, 0.0)
    return env


def apply(env, acc, W, maxload=None):
    """env.step_apply + scored tallies (epoch index >= W). pv_all / np_all count every epoch (oracle objective)."""
    t = env.t
    o0, p0 = env.acc["oper_viol_epochs"], env.acc["phys_viol_epochs"]
    kpi = env.step_apply(acc)
    T, tau = env._xt, env.cfg.epoch_s
    npm = ~env.prot
    nb = float(kpi["thr"][npm].sum()) * tau
    pvv = int(kpi["prot_below"].sum())
    T["pv_all"] += pvv
    T["np_all"] += nb
    if t >= W:
        T["pv"] += pvv
        T["pv_thr"] += int(kpi["prot_below_thr"].sum())
        T["pu"] += int(env.prot.sum())
        T["E"] += float(kpi["power_w"].sum()) * tau
        T["np_bits"] += nb
        T["np_ue_s"] += float(npm.sum()) * tau
        T["oper"] += env.acc["oper_viol_epochs"] - o0
        T["phys"] += env.acc["phys_viol_epochs"] - p0
        T["sleep_cs"] += int(kpi["sleep"].sum())
        T["intf"] += float(kpi["intf_out_w"].sum())
        T["n_ep"] += 1
        if maxload is not None:
            T["maxload"] += maxload
    return kpi


def run_episode(cfg, seed, arb, W, S, trace=False, conflicts=False, loadkpi=False):
    env = new_env(cfg, seed)
    E = [0.0] if trace else None
    while env.t < W + S:
        t = env.t
        reqs = env.step_propose()
        if conflicts and t >= W:
            env._xt["n_conf"] += bool(conflict_groups(env, reqs))
        ml = float(env.obs()["dem_load"].max()) if loadkpi else None
        apply(env, arb(env, reqs), W, ml)
        if E is not None:
            E.append(env._xt["E"])
    return env, E


def outcome(env):
    T = env._xt
    o = {k: float(v) for k, v in T.items()}
    o.update(V=3600.0 * T["pv"] / max(T["pu"], 1e-9), changes=int(env.stats["changes"]),
             lock_blocked=int(env.stats["lock_blocked"]), cfg_offcycle=int(env.stats["cfg_offcycle"]),
             acc_by={k: int(v) for k, v in env.acc_by.items()}, req_by={k: int(v) for k, v in env.req_by.items()})
    return o


def _aa(env, reqs):
    return reqs


def _freeze(env, reqs):
    return []


def _subset(names):
    names = set(names)
    return lambda env, reqs: [r for r in reqs if r["xapp"] in names]


def _obj(a):
    return lambda env, reqs: a.decide(env, reqs)


def stage1_arms(pair):
    dep = deployed(pair)
    arms = ["sub:" + x for x in dep] + ["aa"]
    arms += ["sub:" + "+".join(c) for n in range(2, len(dep)) for c in itertools.combinations(dep, n)]
    perms = list(itertools.permutations(dep))
    arms += ["prio:" + ">".join(o) for o in perms] + ["cell:" + ">".join(o) for o in perms]
    return arms + ["direct", "clipping"]


def arm_group(arm):
    if arm == "freeze":
        return 1
    if arm == "aa":
        return 3
    if arm.startswith("sub:"):
        return 2 if "+" not in arm else 4
    return {"prio": 5, "cell": 6, "direct": 7, "clipping": 8, "hind": 9, "oracle": 10, "xtruce": 11}[arm.split(":")[0]]


def make_arbiter(arm):
    """-> (fn(env, reqs) -> accepted, object or None)."""
    if arm == "freeze":
        return _freeze, None
    if arm == "aa":
        return _aa, None
    if arm.startswith("sub:"):
        return _subset(arm[4:].split("+")), None
    if arm.startswith("prio:"):
        a = StaticPriority(arm[5:].split(">"))
        return _obj(a), a
    if arm.startswith("cell:"):
        a = CellPriorityLock(arm[5:].split(">"), lease=LEASE)
        return _obj(a), a
    if arm == "direct":
        a = Direct()
        return _obj(a), a
    if arm == "clipping":
        a = Clipping()
        return _obj(a), a
    if arm == "xtruce":
        from cdd_oran.envs.xtruce.xtruce_arbiter import XTruce
        a = XTruce()
        return _obj(a), a
    raise KeyError(arm)


def arm_job(pair, s, seed, arm, smoke, conflicts=False, loadkpi=False):
    W, S = ep_len(smoke)
    fn, obj = make_arbiter(arm)
    env, _ = run_episode(make_cfg(pair, s, smoke), seed, fn, W, S, conflicts=conflicts, loadkpi=loadkpi)
    o = outcome(env)
    if arm == "xtruce":
        o["xtruce_cases"] = dict(obj.cases)
    return o


# ---------------------------------------------------------------------------------------------- oracle (arm 10)
MODES = ("accept", "reject", "half", "lock")


def lex_better(a, b):
    return a[0] < b[0] - 1e-9 or (abs(a[0] - b[0]) <= 1e-9 and a[1] < b[1] - 1e-6)


def lex_sign(aa, pl):
    for d, tol in ((aa[0] - pl[0], 1e-9), (aa[1] - pl[1], 1e-6)):
        if abs(d) > tol:
            return int(np.sign(d))
    return 0


class Oracle:
    """Budgeted WG3 oracle, protocol sec. 5 arm 10 (E6-P arm 10 mechanics on the xTRUCE plant). Plan = mode per
    (cell x xApp) in MODES; decision points every D epochs from t = 0; per decision: stage 1 = best of accept-all,
    reject-all, incumbent and n_glob random network-wide plans; stage 2 = one coordinate-descent pass over the cells in
    random order with n_loc random local alternatives each; rollouts hold the plan H epochs on the TRUE tape.
    Objective (protected violated UE-s, -non-protected bits) over the horizon, lexicographic; admissible iff the
    rollout's cumulative scored energy at the horizon end <= Ef(t) - X (Ef(t) - EA(t)) AND, versus the accept-all-plan
    anchor rollout from the same state (sc[0], already computed), (a) non-protected bits >= 0.90 x the anchor's and
    (b) e2-e5 epochs <= the anchor's + ceil(0.01 H). Ef=None: the energy bound is off (timing / plumbing only); the
    guardrails still apply."""

    def __init__(self, env, xapps, Ef, EA, cap, W, T_end, D, H, n_glob, n_loc, rho=True):
        self.env, self.xapps = env, tuple(xapps)
        self.Ef, self.EA, self.cap, self.W, self.T_end = Ef, EA, cap, W, T_end
        self.D, self.H, self.n_glob, self.n_loc, self.rho = D, H, n_glob, n_loc, rho
        self.oper_slack = math.ceil(OR_G_OPER * H)
        self.cells = list(range(env.B))
        self.plan, self.hs = self.uniform("accept"), {}
        self.n_roll, self.picks, self.deviations, self.epochs, self.n_conf_dp = 0, {}, [], [], 0
        self.n_guard_rej = {"np": 0, "oper": 0, "both": 0}     # energy-admissible trials refused by a guardrail

    def uniform(self, m):
        return {b: {x: m for x in self.xapps} for b in self.cells}

    def rand_cell(self, r):
        return {x: MODES[int(r.integers(len(MODES)))] for x in self.xapps}

    def bound(self, t):
        if self.Ef is None:
            return np.inf
        t = min(t, len(self.Ef) - 1)
        return self.Ef[t] - X_MATCH * (self.Ef[t] - self.EA[t])

    def decide(self, env, reqs, plan, hs):
        left = np.inf if self.cap is None else self.cap - env.stats["changes"]
        out, n = [], 0
        for r in reqs:
            k = r["knob"]
            m = plan[int(env.knob_cells(k)[0])].get(r["xapp"], "accept")
            if m == "accept":
                q = r
            elif m == "reject":
                q = None
            elif m == "half":
                if r["kind"] == "fast":
                    q = half(r)
                else:
                    c = hs.get(k, 0)
                    hs[k] = c + 1
                    q = r if (c + int(k[1]) % 2) % 2 == 0 else None
            else:                                                   # lock: reject and block the knob for D epochs
                env.lock(k, self.D)
                q = None
            if q is None:
                continue
            ch = abs(q["prop"] - q["cur"]) > 1e-12
            if ch and n >= left:
                continue
            n += int(ch)
            out.append(q)
        return out

    def roll(self, plan, reseed=None):
        """-> (protected violated UE-s, -non-protected bits, energy_ok, e2-e5 epochs, scored energy J), all over the
        rollout (every epoch; energy: scored epochs, as the bound)."""
        sim = self.env.copy(reseed=reseed)
        s0, o0 = dict(sim._xt), sim.acc["oper_viol_epochs"]
        hs = dict(self.hs)
        apply(sim, self.decide(sim, sim._pending, plan, hs), self.W)
        for _ in range(self.H - 1):
            if sim.t >= self.T_end:
                break
            reqs = sim.step_propose()
            apply(sim, self.decide(sim, reqs, plan, hs), self.W)
        self.n_roll += 1
        X = sim._xt
        return (X["pv_all"] - s0["pv_all"], -(X["np_all"] - s0["np_all"]), bool(X["E"] <= self.bound(sim.t) + 1e-6),
                int(sim.acc["oper_viol_epochs"] - o0), X["E"] - s0["E"])

    def admissible(self, s, anchor):
        """Energy bound (s[2]) and the guardrails vs the accept-all-plan anchor rollout: (a) non-protected bits
        >= 0.90 x anchor's; (b) e2-e5 epochs <= anchor's + ceil(0.01 H)."""
        if not s[2]:
            return False
        g_np = -s[1] >= OR_G_NP * -anchor[1] - 1e-6
        g_op = s[3] <= anchor[3] + self.oper_slack
        if not (g_np and g_op):
            self.n_guard_rej["both" if not (g_np or g_op) else "np" if not g_np else "oper"] += 1
            return False
        return True

    def choose(self, t):
        r = np.random.default_rng([self.env.seed, int(t), ORACLE_TAG])
        aa = self.uniform("accept")
        cands = [aa, self.uniform("reject"), dict(self.plan)]
        for _ in range(self.n_glob):
            p = self.rand_cell(r)
            cands.append({b: dict(p) for b in self.cells})
        sc = [self.roll(c) for c in cands]
        anchor = sc[0]                                              # accept-all plan, same state, true tape
        i_best = None
        for i, s in enumerate(sc):
            if self.admissible(s, anchor) and (i_best is None or lex_better(s, sc[i_best])):
                i_best = i
        best, s_best, ok = (dict(cands[i_best]), sc[i_best], True) if i_best is not None else (aa, sc[0], False)
        changed = 0
        for b in r.permutation(self.cells):
            for _ in range(self.n_loc):
                trial = dict(best)
                trial[int(b)] = self.rand_cell(r)
                s = self.roll(trial)
                if self.admissible(s, anchor) and (not ok or lex_better(s, s_best)):
                    best, s_best, ok, changed = trial, s, True, changed + 1
        if not ok:
            best, s_best = aa, sc[0]
        tag = (["accept", "reject", "incumbent"][i_best] if i_best is not None and i_best < 3 else
               "random" if i_best is not None else "none_admissible")
        key = f"stage1={tag}|local={min(changed, 5)}|final_ok={int(ok)}"
        self.picks[key] = self.picks.get(key, 0) + 1
        is_aa = all(m == "accept" for c in best.values() for m in c.values())
        if self.rho and not is_aa:
            sa, sp = self.roll(aa, reseed=1), self.roll(best, reseed=1)
            self.deviations.append({"t": int(t), "aa_true": list(sc[0]), "plan_true": list(s_best),
                                    "aa_redraw": list(sa), "plan_redraw": list(sp),
                                    "sign_true": lex_sign(sc[0], s_best), "sign_redraw": lex_sign(sa, sp)})
        return best

    def run(self):
        env = self.env
        while env.t < self.T_end:
            t = env.t
            reqs = env.step_propose()
            if t % self.D == 0:
                self.n_conf_dp += bool(conflict_groups(env, reqs))
                t0 = time.time()
                self.plan = self.choose(t)
                self.epochs.append(round(time.time() - t0, 3))
            apply(env, self.decide(env, reqs, self.plan, self.hs), self.W)


def oracle_job(pair, s, seed, smoke, record_scores=True):
    cfg = make_cfg(pair, s, smoke)
    W, S = ep_len(smoke)
    t = time.time()
    ef_env, Ef = run_episode(cfg, seed, _freeze, W, S, trace=True)
    es_env, EA = run_episode(cfg, seed, _subset(("ES",)), W, S, trace=True)
    aa_env, _ = run_episode(cfg, seed, _aa, W, S)
    cap = int(aa_env.stats["changes"])
    t_ref = time.time() - t
    env = new_env(cfg, seed)
    orc = Oracle(env, deployed(pair), Ef, EA, cap, W, W + S, **(SMOKE["oracle"] if smoke else ORACLE))
    t = time.time()
    orc.run()
    rec = {"churn_cap": cap, "n_roll": orc.n_roll, "picks": orc.picks, "n_decisions": len(orc.epochs),
           "n_conflict_decisions": orc.n_conf_dp, "decision_secs": orc.epochs, "ref_secs": round(t_ref, 2),
           "oracle_secs": round(time.time() - t, 2)}
    if record_scores:
        rec.update(outcome(env), deviations=orc.deviations, guard_rejected=dict(orc.n_guard_rej),
                   ref_check={"freeze_pv": ef_env._xt["pv"], "ES_pv": es_env._xt["pv"], "AA_pv": aa_env._xt["pv"],
                              "AA_E": aa_env._xt["E"]})
    else:
        rec["n_deviations"] = len(orc.deviations)
    return rec


# ---------------------------------------------------------------------------------------------- arm 9
def energy_ok(o, of, oes):
    return (of["E"] - o["E"]) >= X_MATCH * (of["E"] - oes["E"]) - 1e-9


def guard_ok(o, aa):
    thr = o["np_bits"] / max(o["np_ue_s"], 1e-9)
    thr_aa = aa["np_bits"] / max(aa["np_ue_s"], 1e-9)
    return thr >= G_THR * thr_aa - 1e-9 and o["oper"] <= G_OPER * aa["oper"] + G_OPER_SLACK * aa["n_ep"] + 1e-9


def _keep_arbiter(keep):
    return lambda env, reqs: [r for r in reqs if r["xapp"] in keep[int(env.knob_cells(r["knob"])[0])]]


def hind_job(pair, s, seed, smoke):
    cfg = make_cfg(pair, s, smoke)
    W, S = ep_len(smoke)
    dep = tuple(deployed(pair))
    of = outcome(run_episode(cfg, seed, _freeze, W, S)[0])
    oes = outcome(run_episode(cfg, seed, _subset(("ES",)), W, S)[0])
    n_cells = cfg.n_cells
    sets = [tuple(c) for n in range(len(dep) + 1) for c in itertools.combinations(dep, n)]
    best = {b: dep for b in range(n_cells)}
    aa = outcome(run_episode(cfg, seed, _keep_arbiter({b: set(v) for b, v in best.items()}), W, S)[0])

    def key(o):
        return (not (energy_ok(o, of, oes) and guard_ok(o, aa)), o["V"])

    best_o, n_eval, path = aa, 1, []
    for b in range(1 if smoke else n_cells):                   # smoke: plumbing only (1 cell searched)
        for ks in sets:
            if ks == best[b]:
                continue
            trial = dict(best)
            trial[b] = ks
            o = outcome(run_episode(cfg, seed, _keep_arbiter({c: set(v) for c, v in trial.items()}), W, S)[0])
            n_eval += 1
            if key(o) < key(best_o):
                best, best_o = trial, o
        path.append({"cell": b, "keep": list(best[b]), "V": best_o["V"], "ok": not key(best_o)[0]})
    return {**best_o, "best_keep": {str(b): list(v) for b, v in best.items() if v != dep}, "best_ok": not key(best_o)[0],
            "n_eval": n_eval, "path": path}


# ---------------------------------------------------------------------------------------------- stage 0 (unit)
def _snap(cfg, seed, W):
    env = new_env(cfg, seed)
    while env.t < W:
        apply(env, _freeze(env, env.step_propose()), W)
    return env


def _realize(env):
    """Realise one epoch on a copy with frozen channel / fast actions (no xApp phase). -> (kpi, P_cell, I)."""
    x_e, p_e, _ = env.actuate()
    P = np.bincount(env.z, p_e, minlength=env.B)
    _, Iu = env._rx(P)
    kpi = env._realize(x_e, p_e)
    return kpi, P, Iu, x_e, p_e


def m1_m2_m3(seed, smoke):
    cfg = make_cfg("X4", 0, smoke)
    W = ep_len(smoke)[0]
    base = _snap(cfg, seed, W)
    c = cfg
    out = {}
    # M1: pcap_b = 3 W (ES target) vs P_max
    rows = []
    for b in range(base.B):
        A, B = base.copy(), base.copy()
        A.pcap[b], B.pcap[b] = c.p_max_w, c.es_cap_w - c.p_cir_w
        kA, PA, IA, _, _ = _realize(A)
        kB, PB, IB, _, pB = _realize(B)
        a = float(pB[B.z == b].sum()) <= B.pcap[b] + 1e-9
        bb = all(abs(k["power_w"][b] - (c.p_cir_w + c.delta_p * P[b])) <= 1e-9 for k, P in ((kA, PA), (kB, PB)))
        other = np.nonzero(base.z != b)[0]
        dI = IB[other] - IA[other]
        want = (PB[b] - PA[b]) / base.K * base.G[b, other]
        cc = bool(np.all(np.abs(dI - want) <= 1e-9 * np.maximum(np.abs(IA[other]), 1e-300)))
        dd = bool(np.all(kB["rate"][other] >= kA["rate"][other] * (1 - 1e-12)))
        others_same = bool(np.allclose(np.delete(PA, b), np.delete(PB, b), rtol=0, atol=0))
        rows.append({"cell": b, "dP_w": float(PB[b] - PA[b]), "a": a, "b": bb, "c": cc, "d": dd,
                     "others_same": others_same})
    out["M1"] = {"pass": all(r["a"] and r["b"] and r["c"] and r["d"] and r["others_same"] and r["dP_w"] < 0
                             for r in rows), "rows": rows}
    # M2: eq. (14) recomputed independently; share doubling -> no spill-over
    A = base.copy()
    kA, P, _, x_e, p_e = _realize(A)
    ar = np.arange(base.U)
    Pk = P / base.K
    Iu = np.einsum("b,buk->uk", Pk, base.G) - Pk[base.z][:, None] * base.G[base.z, ar]
    Gs = base.G[base.z, ar]
    R = np.where(x_e > 0, x_e * base.W * np.log2(1 + Gs * (p_e / (base.K * np.maximum(x_e, 1e-12)))[:, None]
                                                  / (c.noise_w + np.maximum(Iu, 0.0))).sum(1), 0.0)
    a = bool(np.all(np.abs(R - kA["rate"]) <= 1e-9 * np.maximum(R, 1.0)))
    u = int(np.argmax(base.x))
    B = base.copy()
    B.x[u] *= 2.0
    kB, _, _, _, _ = _realize(B)
    oth = base.z != base.z[u]
    bb = bool(np.array_equal(kA["rate"][oth], kB["rate"][oth]))
    out["M2"] = {"pass": a and bb, "rate_eq14": a, "no_spillover": bb,
                 "max_rel_err": float(np.max(np.abs(R - kA["rate"]) / np.maximum(R, 1.0)))}
    # M3: sleep / wake / last cell / off-cycle (at a configuration epoch)
    A = base.copy()
    assert A.t % cfg.t_cfg == 0
    z0, sx0 = A.z.copy(), np.bincount(A.z, A.x, minlength=A.B)
    n0 = np.bincount(A.z, minlength=A.B)
    b = int(np.argmax(n0))                                          # non-trivial: the most populated cell
    A.step_propose()
    k = A.step_apply([{"xapp": "m3", "knob": ("sleep", b), "cur": 0.0, "prop": 1.0, "t": A.t, "kind": "cfg"}])
    moved = np.nonzero(z0 == b)[0]
    best_act = [int(np.argmax(np.where(A.alpha, A.Gm[:, uu], -np.inf))) for uu in moved]
    sx1 = np.bincount(A.z, A.x, minlength=A.B)
    recv = [c_ for c_ in range(A.B) if c_ != b and n0[c_] > 0]      # freeze: sum x = 1 in every non-empty cell
    r = {"sleep_power": abs(k["power_w"][b] - c.p_sleep_w) <= 1e-12, "empty": int((A.z == b).sum()) == 0,
         "best_active": all(int(A.z[uu]) == t_ for uu, t_ in zip(moved, best_act, strict=True)),
         "work_conserving": bool(np.allclose(sx1[recv], sx0[recv], atol=1e-9))}
    while A.t % cfg.t_cfg != 0:
        A.step_propose()
        A.step_apply([])
    A.step_propose()
    A.step_apply([{"xapp": "m3", "knob": ("sleep", b), "cur": 1.0, "prop": 0.0, "t": A.t, "kind": "cfg"}])
    r["wake_restores"] = bool(np.array_equal(A.z, z0))
    C = base.copy()
    for c_ in range(C.B - 1):
        C.alpha[c_] = False
    C._fix_config()
    r["last_cell_refuses"] = C._set(("sleep", C.B - 1), 1.0) is False and bool(C.alpha[C.B - 1])
    D = base.copy()
    D.step_propose()
    D.step_apply([])
    D.step_propose()
    n0 = D.stats["cfg_offcycle"]
    D.step_apply([{"xapp": "m3", "knob": ("sleep", b), "cur": 0.0, "prop": 1.0, "t": D.t, "kind": "cfg"}])
    r["offcycle_dropped"] = D.stats["cfg_offcycle"] == n0 + 1 and bool(D.alpha[b])
    out["M3"] = {"pass": all(r.values()), **r}
    return out


def m5(seed, smoke):
    n = 20 if smoke else 100
    W = ep_len(smoke)[0]
    cfg = make_cfg("X4", 0, smoke)

    def ep(cfg_, arb, k):
        env = new_env(cfg_, seed)
        per = []
        for _ in range(k):
            kp = apply(env, arb(env, env.step_propose()), W)
            per.append((float(kp["power_w"].sum()), int(kp["prot_below"].sum())))
        return env, per

    e1, _ = ep(cfg, _aa, n)
    e2, _ = ep(cfg, _aa, n)
    a = e1.summary() == e2.summary()
    env, _ = ep(cfg, _aa, W)
    cp = env.copy()
    for _ in range(50):
        apply(env, _aa(env, env.step_propose()), W)
        apply(cp, _aa(cp, cp.step_propose()), W)
    b = env.summary() == cp.summary() and env._xt == cp._xt
    ref = None
    c = True
    for pair in PAIRS:
        for s in (0, 1):                                              # L6 x {h0, h50}
            _, per = ep(make_cfg(pair, s, smoke), _freeze, n)
            ref = per if ref is None else ref
            c = c and per == ref
    env, _ = ep(cfg, _freeze, W)
    rc, same = env.copy(reseed=1), env.copy()
    d_state = bool(rc.Gm is env.Gm and rc.prot is env.prot and rc.ue_pos is env.ue_pos)
    for e_ in (env, rc, same):
        apply(e_, _freeze(e_, e_.step_propose()), W)
    d_redraw = not np.array_equal(env.Q, rc.Q)
    d_replay = bool(np.array_equal(env.Q, same.Q))
    d = d_state and d_redraw and d_replay
    return {"M5": {"pass": bool(a and b and c and d), "a_repeat": a, "b_copy_replay": b, "c_freeze_shared": c,
                   "d_reseed_keeps_state": d_state, "d_reseed_redraws_arrivals": d_redraw}}


def m8(seed, smoke):
    cfg = make_cfg("X4", 0, smoke)
    W, S = ep_len(smoke)
    H = (SMOKE["oracle"] if smoke else ORACLE)["H"]
    env = new_env(cfg, seed)
    while env.t < W:
        apply(env, _aa(env, env.step_propose()), W)
    env.step_propose()
    orc = Oracle(env, deployed("X4"), None, None, None, W, W + S, **(SMOKE["oracle"] if smoke else ORACLE))
    out = {}
    for mode, arb in (("accept", _aa), ("reject", _freeze)):
        r = orc.roll(orc.uniform(mode))
        sim = env.copy()
        s0, o0 = dict(sim._xt), sim.acc["oper_viol_epochs"]
        apply(sim, arb(sim, sim._pending), W)
        for _ in range(H - 1):
            apply(sim, arb(sim, sim.step_propose()), W)
        # pv and energy (protocol M8 a/b) plus the two oracle-guardrail quantities (np bits, e2-e5 epochs)
        out[mode] = (r[0] == sim._xt["pv_all"] - s0["pv_all"] and abs(-r[1] - (sim._xt["np_all"] - s0["np_all"])) < 1e-6
                     and abs(r[4] - (sim._xt["E"] - s0["E"])) <= 1e-9 * max(abs(r[4]), 1.0)
                     and r[3] == sim.acc["oper_viol_epochs"] - o0)
    e0 = new_env(cfg, seed)
    reqs = e0.step_propose()
    es_pcap = [r for r in reqs if r["xapp"] == "ES" and r["knob"][0] == "pcap"]
    ok_lock = False
    if es_pcap:
        k = es_pcap[0]["knob"]
        o0 = Oracle(e0, deployed("X4"), None, None, None, W, W + S, **ORACLE)
        plan = o0.uniform("accept")
        plan[int(k[1])]["ES"] = "lock"
        acc = o0.decide(e0, reqs, plan, {})
        n0, pc = e0.stats["lock_blocked"], float(e0.pcap[k[1]])
        acc.append({"xapp": "QoS", "knob": k, "cur": pc, "prop": 1.0, "t": e0.t, "kind": "fast"})
        e0.step_apply(acc)
        ok_lock = e0.stats["lock_blocked"] >= n0 + 1 and float(e0.pcap[k[1]]) == pc
    out["lock_blocks_other_xapp"] = ok_lock
    return {"M8": {"pass": all(out.values()), **out}}


# ---------------------------------------------------------------------------------------------- jobs
def units_for(stage, state, smoke=False):
    """List of units (lists of job keys (stage, pair, stratum, seed, arm)); unit u goes to shard u % k."""
    U = []
    excl = {tuple(x) for x in state.get("excluded_cells", [])}
    live = [p for p in PAIRS if p not in state.get("not_screenable_pairs", [])]
    if stage == "0":
        U.append([("0", "*", 0, UNIT_SEED, "unit")])
        for s in range(len(STRATA)):
            for seed in mech_seeds(s, smoke):
                U.append([("0", "*", s, seed, "freeze")])
                U += [[("0", "X4", s, seed, "sub:" + x)] for x in deployed("X4")]
                U += [[("0", p, s, seed, "aa")] for p in PAIRS]
    elif stage == "0c":
        U = [[("0c", PILOT[0], PILOT[1], PILOT[2], "oracle_pilot")]]
    elif stage == "1":
        for s in range(len(STRATA)):
            if s in state.get("excluded_strata", []):
                continue
            for seed in screen_seeds(s, smoke):
                U.append([("1", "*", s, seed, "freeze")])
                U += [[("1", p, s, seed, a)] for p in live if (p, s) not in excl for a in stage1_arms(p)]
    elif stage == "1x":
        n = SMOKE["n_xtruce"] if smoke else N_XTRUCE
        U = [[("1x", p, s, seed, "xtruce")] for s in range(len(STRATA)) if s not in state.get("excluded_strata", [])
             for p in live if (p, s) not in excl for seed in screen_seeds(s, smoke)[:n]]
    elif stage in ("2", "3"):
        cells = state.get("stage1_crit1" if stage == "2" else "stage2_crit12", [])
        if smoke and not cells:
            cells = [["X4", 0]]                                        # plumbing only
        arm = "oracle" if stage == "2" else "hind"
        U = [[(stage, p, s, seed, arm)] for p, s in cells for seed in screen_seeds(s, smoke)]
    for u in U:
        for key in u:
            assert smoke or SCREEN_RANGE[0] <= key[3] <= SCREEN_RANGE[1], key   # never the confirmation range
    return U


def run_job(key, smoke):
    stage, pair, s, seed, arm = key
    sd = _seed(seed, smoke)
    t = time.time()
    rec = {"kind": "job", "key": list(key), "stage": stage, "pair": pair, "stratum": s, "seed": seed, "arm": arm,
           "run_seed": sd, "smoke": smoke}
    if arm == "unit":
        rec.update(m1_m2_m3(sd, smoke))
        rec.update(m5(sd, smoke))
        rec.update(m8(sd, smoke))
    elif stage == "0":
        rec.update(arm_job(pair, s, sd, arm, smoke, conflicts=(arm == "aa"), loadkpi=True))
    elif arm == "oracle":
        rec.update(oracle_job(pair, s, sd, smoke))
    elif arm == "oracle_pilot":
        rec.update(oracle_job(pair, s, sd, smoke, record_scores=False))
    elif arm == "hind":
        rec.update(hind_job(pair, s, sd, smoke))
    else:
        rec.update(arm_job(pair, s, sd, arm, smoke, conflicts=(arm == "aa")))
    rec["secs"] = round(time.time() - t, 2)
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


def _js(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)


def _append(out, rec):
    with open(out, "a", newline="\n") as f:
        f.write(json.dumps(rec, default=_js) + "\n")


def load_state():
    js = json.load(open(STATE_JSON)) if os.path.exists(STATE_JSON) else None
    py = None
    if os.path.exists(STATE_PY):
        ns = {}
        exec(open(STATE_PY).read(), ns)                                     # noqa: S102 (our own data mirror)
        py = json.loads(ns["STATE"])
    if js is not None and py is not None and js != py:
        raise SystemExit("xtruce_state.json and xtruce_state_json.py differ: re-run summary --write-state")
    return js if js is not None else (py or {})


def write_state(state):
    s = json.dumps(state, indent=1, sort_keys=True)
    open(STATE_JSON, "w", newline="\n").write(s + "\n")
    open(STATE_PY, "w", newline="\n").write('"""Mirror of xtruce_state.json (cloud.py bundles only *.py). Written by '
                                            'xtruce_screen.py summary --write-state; do not edit."""\n'
                                            f"STATE = r'''{s}'''\n")


def header(stage, part, smoke, state, fz):
    code = {}
    mp = os.path.join(ROOT, "MANIFEST.json")
    if BUNDLE and os.path.exists(mp):
        m = json.load(open(mp))
        code = {"git_head": m.get("git_head"), "driver_sha256": (m.get("sha256") or {}).get("e6dev/xtruce_screen.py")}
    else:
        try:
            code = {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                               timeout=20).stdout.strip()}
        except Exception:  # noqa: BLE001
            code = {}
    try:
        import cloud
        nenv = cloud.numeric_env()
    except Exception as e:  # noqa: BLE001
        nenv = {"error": repr(e)}
    return {"kind": "header", "stage": stage, "part": part, "smoke": smoke, "protocol": PROTOCOL, "freeze": fz,
            "code": code, "numeric_env": nenv, "state": state, "cpus": os.cpu_count(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run(stage, part, out, smoke=False):
    if stage not in STAGES:
        raise SystemExit(f"--stage in {STAGES}")
    fz = freeze_status()
    if FROZEN_SHA256 is not None and not fz["ok"]:
        raise SystemExit(f"refusing to run: {fz['why']}")
    if not smoke and not fz["ok"]:
        raise SystemExit(f"refusing to run: {fz['why']}")
    i, k = map(int, part.split("/"))
    state = load_state()
    if state.get("smoke") and not smoke:
        raise SystemExit(f"refusing to run: {STATE_PY} was written from smoke records (state smoke=True)")
    _append(out, header(stage, part, smoke, state, fz))
    t0 = time.time()
    done = {tuple(r["key"]) for r in _read(out) if r.get("kind") == "job" and r.get("smoke") == smoke}
    units = units_for(stage, state, smoke)
    n = 0
    for u, unit in enumerate(units):
        if u % k != i:
            continue
        for key in unit:
            if key in done:
                continue
            rec = run_job(key, smoke)
            _append(out, rec)
            n += 1
            print(json.dumps({x: rec.get(x) for x in ("stage", "pair", "stratum", "seed", "arm", "secs")}), flush=True)
        if smoke and n >= SMOKE["max_jobs"]:
            break
    _append(out, {"kind": "close", "stage": stage, "part": part, "n_jobs": n, "n_units": len(units),
                  "secs": round(time.time() - t0, 1)})


# ---------------------------------------------------------------------------------------------- summary
def _pool(recs):
    g = lambda k: sum(r[k] for r in recs)                                        # noqa: E731
    return {"V": 3600.0 * g("pv") / max(g("pu"), 1e-9), "E": g("E"),
            "thr_np": g("np_bits") / max(g("np_ue_s"), 1e-9), "oper": g("oper"), "phys": g("phys"),
            "n_ep": g("n_ep"), "V_thr": 3600.0 * g("pv_thr") / max(g("pu"), 1e-9), "sleep_cs": g("sleep_cs"),
            "changes": float(np.mean([r["changes"] for r in recs]))}


def _vboot(recs, idx):
    pv = np.array([r["pv"] for r in recs])
    pu = np.array([r["pu"] for r in recs])
    return 3600.0 * pv[idx].sum(1) / np.maximum(pu[idx].sum(1), 1e-9)


def summarise_0(R, smoke):
    out = {}
    unit = [r for k, r in R.items() if k[0] == "0" and k[4] == "unit"]
    if unit:
        for m in ("M1", "M2", "M3", "M5", "M8"):
            out[m] = {"pass": all(r[m]["pass"] for r in unit),
                      "detail": {x: v for x, v in unit[0][m].items() if x != "rows"}}
    m4, bad_pairs = {}, set()
    for x in deployed("X4"):
        per = {}
        for s in range(len(STRATA)):
            seeds = mech_seeds(s, smoke)
            al = [R.get(("0", "X4", s, sd, "sub:" + x)) for sd in seeds]
            fr = [R.get(("0", "*", s, sd, "freeze")) for sd in seeds]
            if not all(al) or not all(fr):
                continue
            kpi = OWN_KPI[x]

            def val(rs, kpi=kpi):
                if kpi == "V":
                    return _pool(rs)["V"]
                return sum(r[kpi] for r in rs) / max(sum(r["n_ep"] for r in rs), 1e-9) if kpi != "E" else \
                    sum(r["E"] for r in rs)
            acts = float(np.mean([a["acc_by"].get(x, 0) for a in al])) >= 1.0
            imp = val(al) < val(fr)
            n_imp = sum(val([a]) < val([f]) for a, f in zip(al, fr, strict=True))
            if kpi in ("E", "intf"):
                material = True                                        # energy / IC: always material
            elif kpi == "V":
                material = val(fr) >= M4_MATERIAL["V"]                 # freeze V >= 36 per protected UE-h
            else:
                material = val(fr) > M4_MATERIAL["maxload"]            # freeze mean max demand load > lb_cap
            need = min(M4_MIN_IMPROVE, len(seeds)) if not smoke else 1
            per[s] = {"acts": acts, "material": bool(material), "own_alone": val(al), "own_freeze": val(fr),
                      "pooled_improves": imp, "n_seeds_improve": n_imp,
                      "ok": (not acts) or (not material) or (imp and n_imp >= need)}
        if not per:
            continue
        n_act = sum(v["acts"] and v["material"] for v in per.values())
        inert = n_act < 2
        ok = all(v["ok"] for v in per.values()) and not inert
        m4[x] = {"strata": per, "inert": inert, "pass": ok}
        if not ok:
            bad_pairs |= {p for p in PAIRS if x in deployed(p)}
    if m4:
        out["M4"] = {"pass": any(p not in bad_pairs for p in PAIRS), "detail": m4, "not_screenable_pairs":
                     sorted(bad_pairs)}
    m6, excl_strata = {}, []
    for s in range(len(STRATA)):
        es = [R.get(("0", "X4", s, sd, "sub:ES")) for sd in mech_seeds(s, smoke)]
        if all(es):
            n_sl = sum(r["sleep_cs"] > 0 for r in es)
            need = min(M6_MIN, len(es)) if not smoke else 1
            m6[s] = {"n_seeds_sleep": n_sl, "sleep_cs": [r["sleep_cs"] for r in es],
                     "required": STRATA[s][0] == "L1", "ok": STRATA[s][0] != "L1" or n_sl >= need}
            if not m6[s]["ok"]:
                excl_strata.append(s)
    if m6:
        out["M6"] = {"pass": all(v["ok"] for v in m6.values()), "detail": m6, "excluded_strata": excl_strata}
    m7, excl_cells = {}, []
    for p in PAIRS:
        for s in range(len(STRATA)):
            aa = [R.get(("0", p, s, sd, "aa")) for sd in mech_seeds(s, smoke)]
            if all(aa):
                frac = [r["n_conf"] / max(r["n_ep"], 1) for r in aa]
                m7[f"{p}|{s}"] = {"conflict_frac_scored": frac}         # diagnostic only (protocol M7)
    if m7:
        out["M7"] = {"pass": True, "detail": m7, "rule": "reported only; no exclusion"}
    return out, sorted(bad_pairs), excl_strata, excl_cells


def analyse(R, pair, s, smoke):
    seeds = screen_seeds(s, smoke)
    fr = [R.get(("1", "*", s, sd, "freeze")) for sd in seeds]
    if not all(fr):
        return None
    per = {"freeze": fr}
    for a in stage1_arms(pair) + ["hind", "oracle"]:
        st = {"hind": "3", "oracle": "2"}.get(a, "1")
        rs = [R.get((st, pair, s, sd, a)) for sd in seeds]
        if all(rs):
            per[a] = rs
    m_x = min(N_XTRUCE, len(seeds)) if not smoke else min(SMOKE["n_xtruce"], len(seeds))
    xt = [R.get(("1x", pair, s, sd, "xtruce")) for sd in seeds[:m_x]]
    xt = xt if xt and all(xt) else None
    need = stage1_arms(pair)
    if not all(a in per for a in need):
        return {"complete": False, "missing": [a for a in need if a not in per][:5]}
    P = {a: _pool(rs) for a, rs in per.items()}
    AA = P["aa"]
    Ef, EA = P["freeze"]["E"], P["sub:ES"]["E"]
    screenable = Ef - EA >= SCREEN_FLOOR * Ef

    def matched(p):
        return Ef - p["E"] >= X_MATCH * (Ef - EA) - 1e-9

    def guard(p):
        return p["thr_np"] >= G_THR * AA["thr_np"] - 1e-9 and p["oper"] <= G_OPER * AA["oper"] + \
            G_OPER_SLACK * AA["n_ep"] + 1e-9

    refs = [a for a in P if arm_group(a) in (2, 4) and matched(P[a]) and guard(P[a])]
    singles = ["sub:" + x for x in deployed(pair)]
    V_AA = AA["V"]
    out = {"complete": True, "n_seeds": len(seeds), "screenable": bool(screenable), "E_saving_ES_frac": (Ef - EA) / Ef,
           "AA_matched": matched(AA), "AA_retention": (Ef - AA["E"]) / (Ef - EA) if Ef != EA else None,
           "V_AA": V_AA, "n_eligible_subsets": len(refs)}
    if refs:                                           # diagnostic only: best energy-matched eligible proper subset
        v_sub = min(P[a]["V"] for a in refs)
        out["Lambda_subset"] = (V_AA - v_sub) / v_sub if v_sub > 0 else None
    ref = min(singles, key=lambda a: P[a]["V"])        # E6-P section 7: V_ref = best single xApp, energy ignored
    V_ref = P[ref]["V"]
    den = V_AA - V_ref
    Lam = den / V_ref if V_ref > 0 else (np.inf if den > 0 else np.nan)
    table = {}
    for a in P:
        table[a] = {"group": arm_group(a), "V": P[a]["V"], "V_thr": P[a]["V_thr"], "E_kJ": P[a]["E"] / 1e3,
                    "retention": (Ef - P[a]["E"]) / (Ef - EA) if abs(Ef - EA) > 1e-12 else np.nan,
                    "thr_np_mbps": P[a]["thr_np"] / 1e6, "oper": P[a]["oper"], "phys": P[a]["phys"],
                    "matched": matched(P[a]), "guard": guard(P[a]), "eligible": matched(P[a]) and guard(P[a]),
                    "R": (V_AA - P[a]["V"]) / den if abs(den) > 1e-12 else np.nan, "changes": P[a]["changes"]}
    static = [a for a in table if table[a]["group"] in (2, 4, 5, 6, 7, 8, 9) and table[a]["eligible"]]
    R_static = max(table[a]["R"] for a in static)
    n = len(seeds)
    rng = np.random.default_rng([BOOT_TAG, PAIRS[pair]["idx"], s])
    idx = rng.integers(0, n, (N_BOOT, n))
    Vb = {a: _vboot(rs, idx) for a, rs in per.items()}
    d_b = Vb["aa"] - np.min([Vb[a] for a in refs], 0)
    Vref_b = Vb["aa"] - d_b
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_b = d_b / Vref_b
        Rb = {a: (Vb["aa"] - Vb[a]) / d_b for a in Vb}
    lb = float(np.quantile(d_b, LB_Q))
    c1 = bool(screenable and matched(AA) and Lam >= LAMBDA_MIN and den >= MATERIAL and lb > 0)
    out.update(V_ref=V_ref, ref_arm=ref, Lambda=Lam, V_AA_minus_V_ref=den, lb90_diff=lb,
               Lambda_ci90=[float(np.nanquantile(lam_b, 0.05)), float(np.nanquantile(lam_b, 0.95))],
               R_static=R_static, R_static_arm=max(static, key=lambda a: table[a]["R"]), c1=c1, arms=table)
    if xt is not None:                                            # reported comparator, first N_XTRUCE seeds
        m = len(xt)
        sub = {a: _pool(per[a][:m]) for a in ("freeze", "sub:ES", "aa", *refs)}
        Px = _pool(xt)
        ef, ea = sub["freeze"]["E"], sub["sub:ES"]["E"]
        vr = min(sub[a]["V"] for a in refs)
        dx = sub["aa"]["V"] - vr
        out["xtruce"] = {"n_seeds": m, "V": Px["V"], "V_AA": sub["aa"]["V"], "V_ref": vr,
                         "R": (sub["aa"]["V"] - Px["V"]) / dx if abs(dx) > 1e-12 else None,
                         "matched": ef - Px["E"] >= X_MATCH * (ef - ea) - 1e-9,
                         "thr_np_ratio": Px["thr_np"] / max(sub["aa"]["thr_np"], 1e-9),
                         "oper": Px["oper"], "cases": [r.get("xtruce_cases") for r in xt]}
    if "oracle" in per:
        R_or = table["oracle"]["R"] if table["oracle"]["eligible"] else 0.0
        Ror_b = Rb["oracle"] if table["oracle"]["eligible"] else np.zeros(N_BOOT)
        devs = [d for r in per["oracle"] for d in r.get("deviations", [])]
        kept = sum(d["sign_true"] == d["sign_redraw"] for d in devs)
        rho = kept / len(devs) if devs else float("nan")
        out.update(R_or=R_or, R_or_ci90=[float(np.nanquantile(Ror_b, 0.05)), float(np.nanquantile(Ror_b, 0.95))],
                   c2=bool(c1 and R_or >= R_OR_MIN), rho_sign=rho, n_deviations=len(devs),
                   c4=bool(devs) and rho >= RHO_MIN,
                   oracle_guard_rejected={g: sum((r.get("guard_rejected") or {}).get(g, 0) for r in per["oracle"])
                                          for g in ("np", "oper", "both")})
        if "hind" in per:
            Rs_b = np.max([Rb[a] for a in static], 0)
            h_b = Ror_b - Rs_b
            out.update(headroom=R_or - R_static,
                       headroom_ci90=[float(np.nanquantile(h_b, 0.05)), float(np.nanquantile(h_b, 0.95))],
                       c3=bool(out["c2"] and R_or - R_static >= HEADROOM))
    return out


def verdict(rows, pair, bad_pairs):
    if pair in bad_pairs:
        return "NOT SCREENABLE (M4)"
    live = [r for r in rows.values() if r is not None and r != "excluded"]
    if not live:
        return "NOT SCREENABLE (M6)" if rows and all(r == "excluded" for r in rows.values()) else "INCOMPLETE"
    if any(not r.get("complete") for r in live):
        return "INCOMPLETE (stage 1)"
    S1 = [r for r in live if r["c1"]]
    if not S1:
        return "DEAD (no loss)"
    if any("c2" not in r for r in S1):
        return "PENDING stage 2 (oracle)"
    S12 = [r for r in S1 if r["c2"]]
    if not S12:
        return "DEAD (not recoverable)"
    if any("hind" not in r["arms"] for r in S12):
        return "PENDING stage 3 (hindsight static)"
    S123 = [r for r in S12 if r["c3"]]
    if any(r["c4"] for r in S123):
        return "PASS"
    return "NOISE STOP" if S123 else "NO-EDGE STOP"


def summary(paths, write=False, json_out=None, allow_smoke=False):
    recs = [r for p in paths for r in _read(p)]
    smoke = allow_smoke and any(r.get("smoke") for r in recs)
    recs = [r for r in recs if allow_smoke or not r.get("smoke")]
    state = load_state()
    new = dict(state)
    report = {"protocol": PROTOCOL, "freeze": freeze_status(), "notes": IMPLEMENTATION_NOTES, "smoke": smoke}
    R = {tuple(r["key"]): r for r in recs if r.get("kind") == "job"}
    ob, bad_pairs, excl_strata, excl_cells = summarise_0(R, smoke)
    if ob:
        report["stage0"] = ob
        allp = all(v["pass"] for v in ob.values()) and len(ob) >= 8
        print("stage 0:", {m: v["pass"] for m, v in ob.items()}, "ALL PASS" if allp else "(incomplete or failed)")
        for x, v in ob.get("M4", {}).get("detail", {}).items():
            print(f"  M4 {x}: pass={v['pass']} inert={v['inert']} " + "; ".join(
                f"s{s}: acts={d['acts']} mat={d['material']} {d['own_alone']:.4g} vs {d['own_freeze']:.4g} "
                f"({d['n_seeds_improve']})" for s, d in v["strata"].items()))
        new.update(not_screenable_pairs=bad_pairs, excluded_strata=excl_strata, excluded_cells=excl_cells,
                   stage0_all_pass=bool(allp))
    pilot = [r for r in R.values() if r["arm"] == "oracle_pilot"]
    if pilot:
        report["stage0c"] = {x: pilot[0].get(x) for x in ("secs", "ref_secs", "oracle_secs", "n_roll", "n_decisions",
                                                          "n_deviations")}
        print("0c pilot:", report["stage0c"])
    rows_all, crit1, crit12 = {}, [], []
    for pair in PAIRS:
        rows = {}
        for s in range(len(STRATA)):
            if s in new.get("excluded_strata", []) or [pair, s] in new.get("excluded_cells", []):
                rows[s] = "excluded"
            else:
                rows[s] = analyse(R, pair, s, smoke)
        if all(v is None for v in rows.values()):
            continue
        v = verdict(rows, pair, new.get("not_screenable_pairs", []))
        rows_all[pair] = {"rows": rows, "verdict": v}
        print(f"\n== {pair} {'+'.join(deployed(pair))}{' (primary)' if pair == PRIMARY else ''}: {v}")
        for s, r in rows.items():
            if r is None or r == "excluded" or not r.get("complete"):
                print(f"  s{s} {STRATA[s]}: {r if r == 'excluded' else 'incomplete'}")
                continue
            if "V_ref" not in r:
                print(f"  s{s} {STRATA[s]}: {r.get('why')}; V_AA={r['V_AA']:.1f} AA_matched={r['AA_matched']}")
                continue
            print(f"  s{s} {STRATA[s]}: n={r['n_seeds']} screenable={r['screenable']} ES saves "
                  f"{r['E_saving_ES_frac']:.3%} AA matched={r['AA_matched']} V_AA={r['V_AA']:.1f} V_ref={r['V_ref']:.1f}"
                  f" ({r['ref_arm']}) Lambda={r['Lambda']:.3f} LB90={r['lb90_diff']:.1f} c1={r['c1']} "
                  f"R_static={r['R_static']:.3f} ({r['R_static_arm']})"
                  + (f" R_or={r['R_or']:.3f} c2={r['c2']} rho={r['rho_sign']:.3f} (n={r['n_deviations']}) "
                     f"c4={r['c4']}" if "R_or" in r else "")
                  + (f" headroom={r['headroom']:.3f} c3={r['c3']}" if "c3" in r else "")
                  + (f" xTRUCE R={r['xtruce']['R']}" if "xtruce" in r else ""))
            for a, t in r["arms"].items():
                print(f"    [{t['group']:>2}] {a:34s} V={t['V']:8.2f} R={t['R']:8.3f} ret={t['retention']:7.3f} "
                      f"thr={t['thr_np_mbps']:6.3f} oper={t['oper']:5.0f} elig={int(t['eligible'])}")
            if r["c1"]:
                crit1.append([pair, s])
                if r.get("c2"):
                    crit12.append([pair, s])
    report["pairs"] = rows_all
    done1 = rows_all and all(r == "excluded" or (r is not None and r.get("complete"))
                             for v in rows_all.values() for r in v["rows"].values())
    if done1:
        new["stage1_crit1"] = crit1
        if all("R_or" in rows_all[p]["rows"][s] for p, s in crit1):
            new["stage2_crit12"] = crit12
    if json_out:
        json.dump(report, open(json_out, "w", newline="\n"), indent=1, default=_js)
    if write:
        new["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        new["smoke"] = smoke
        write_state(new)
        print("state ->", STATE_JSON)
    return report


# ---------------------------------------------------------------------------------------------- estimate / list
def estimate():
    """Wall time per arm type on the disclosed DEV seed 0 (full protocol episode length, X4, stratum 3 = L1-h50, the
    most requests); one full-budget oracle decision; xTRUCE per epoch. Scores are not printed."""
    cfg = make_cfg("X4", 3)
    seed, out = 0, {}
    for a in ("freeze", "aa", "sub:QoS+ES", "prio:QoS>ES>IC>LB", "cell:QoS>ES>IC>LB", "direct"):
        t = time.time()
        run_episode(cfg, seed, make_arbiter(a)[0], W_EP, S_EP)
        out[a] = round(time.time() - t, 3)
    fn, obj = make_arbiter("xtruce")
    env = new_env(cfg, seed)
    t = time.time()
    for _ in range(20):
        apply(env, fn(env, env.step_propose()), W_EP)
    out["xtruce_s_per_epoch"] = round((time.time() - t) / 20, 4)
    env = new_env(cfg, seed)
    while env.t < W_EP:
        apply(env, _aa(env, env.step_propose()), W_EP)
    env.step_propose()
    orc = Oracle(env, deployed("X4"), None, None, None, W_EP, W_EP + S_EP, **ORACLE)
    t = time.time()
    orc.choose(env.t)
    out["oracle_decision_s"] = round(time.time() - t, 3)
    out["oracle_decision_rollouts"] = orc.n_roll
    ep = float(np.mean([out[a] for a in ("freeze", "aa", "sub:QoS+ES", "prio:QoS>ES>IC>LB", "cell:QoS>ES>IC>LB")]))
    n_dec = (W_EP + S_EP) // ORACLE["D"]
    rollout_s = out["oracle_decision_s"] / max(orc.n_roll, 1)
    orc_ep = 4 * ep + n_dec * out["oracle_decision_s"] + n_dec * 2 * rollout_s     # refs + decisions + rho (upper)
    n_seed_strata = len(STRATA) * N_SCREEN
    n1 = n_seed_strata * (1 + sum(len(stage1_arms(p)) for p in PAIRS))
    hind_eps = {p: 3 + 4 * (2 ** len(deployed(p)) - 1) for p in PAIRS}
    proj = {"0": (len(STRATA) * len(MECH_J) * (1 + 4 + 4)) * ep / 3600,
            "0c": orc_ep / 3600,
            "1": n1 * ep / 3600 + n_seed_strata * 2 * (out["direct"] - ep) / 3600,
            "1x": len(PAIRS) * len(STRATA) * N_XTRUCE * (W_EP + S_EP) * out["xtruce_s_per_epoch"] / 3600,
            "2_per_pair_stratum": N_SCREEN * orc_ep / 3600, "2_max_all16": 16 * N_SCREEN * orc_ep / 3600,
            "3_per_pair_stratum_X4": N_SCREEN * hind_eps["X4"] * ep / 3600,
            "3_max_all16": sum(N_SCREEN * hind_eps[p] * ep * len(STRATA) for p in PAIRS) / 3600}
    out.update(episode_s=round(ep, 3), oracle_episode_s=round(orc_ep, 1), stage1_episodes=n1,
               projected_cpu_h={k: round(v, 2) for k, v in proj.items()})
    print(json.dumps(out, indent=1))
    return out


def list_jobs():
    state = load_state()
    print("freeze:", {k: v for k, v in freeze_status().items() if k != "plant_sha256"})
    print("state:", {k: state.get(k) for k in ("not_screenable_pairs", "excluded_strata", "stage1_crit1",
                                               "stage2_crit12")})
    for st in STAGES:
        U = units_for(st, state)
        arms = {}
        for u in U:
            for key in u:
                g = key[4].split(":")[0]
                arms[g] = arms.get(g, 0) + 1
        print(f"{st}: {len(U)} units {arms}")
    for s in range(len(STRATA)):
        print(f"screen seeds s{s} {STRATA[s]}: {len(screen_seeds(s))} -> {screen_seeds(s)[:3]}...")
    for p in PAIRS:
        print(p, len(stage1_arms(p)), "stage-1 arms")


def _stage_from_env():
    return os.environ.get("XTRUCE_STAGE")


def cli(argv, stage=None):
    if not argv:
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    flags = {x for x in rest if x in ("--smoke", "--write-state", "--allow-smoke")}
    rest = [x for x in rest if x not in flags]
    if cmd == "run":
        a = dict(zip(rest[::2], rest[1::2], strict=True))
        st = a.get("--stage") or stage or _stage_from_env()    # a per-stage wrapper's baked stage beats the env var
        run(st, a["--part"], a["--out"], smoke="--smoke" in flags)
    elif cmd == "summary":
        files, js = [], None
        it = iter(rest)
        for x in it:
            if x == "--json":
                js = next(it)
            elif x != "--in":
                files.extend(f for f in x.split(",") if f)
        summary(files, write="--write-state" in flags, json_out=js, allow_smoke="--allow-smoke" in flags)
    elif cmd == "estimate":
        estimate()
    elif cmd == "list":
        list_jobs()
    elif cmd == "freeze-check":
        print(json.dumps(freeze_status(), indent=1))
        check_frozen(make_cfg("X4", 0))
        print("config == XConfig() defaults (except the allowed per-stratum fields): OK")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    cli(sys.argv[1:])

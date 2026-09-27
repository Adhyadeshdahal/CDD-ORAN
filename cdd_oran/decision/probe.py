"""E6 DEV probe campaign for knob-family -> KPI graph data (SOL_DATA_DESIGN.md section a, "Graph data").

WHAT THIS IS: an OPERATOR-RUN, bounded DEV experiment. It is NOT a WG3 policy, NOT a controller, NOT deployable, and
its data never enter WG3 policy runs or TEST. The env runs with ``wg3=False`` so the probe can issue direct one-step
writes; every xApp request is accepted (last writer wins), exactly as the same-seed no-probe reference.

Design (predeclared before the episode, from the seed only; nothing depends on contemporaneous load):
  * skeleton: the scored window is cut into consecutive SLOTS. Slot type A (``slot_a_s``, observation ``obs_a_s``)
    carries the fast families {cio, hys, ttt, ll_ratio} + sham; slot type B (``slot_b_s``, ``obs_b_s`` > the 120 s
    carrier dwell) carries {carrier} + sham and only macro regions. Type drawn per slot with P(B) = ``p_slot_b``.
  * unit = region (site) + its interference neighbourhood (regions linked by any CIO relation, both directions). Per
    slot up to ``max_treated`` units are drawn at random with pairwise-DISJOINT unit sets (so every treated region's
    neighbourhood is untreated and no cell's outcome belongs to two blocks). The rest of the slot after the observation
    window is washout: the probe is restored at ``t0 + obs`` and nothing new is assigned to that unit before the next
    slot. Blocks are disjoint in space (within a slot) and in time (across slots).
  * changes/hour cap: a unit is only scheduled if the WORST-CASE knob changes of its block (write + restore of the
    largest eligible family) keep the cumulative total <= ``cap_changes_per_hour`` x scored hours. Worst case -> the
    skeleton does not depend on the drawn arm, so the arm can be re-sampled with the skeleton held fixed.
  * arm per block (the ASSIGNMENT MECHANISM, ``AssignmentMechanism``): arm ~ the slot type's arm distribution
    renormalised over the eligible arms; level | arm: sign +-1 (one actuator step: cio 2 dB, hys 0.5 dB, ttt one
    TS 38.331 index, ll_ratio 0.10), carrier = toggle (+1: switch one carrier off if all are on, else on), sham = 0.
    Drawn from ``default_rng([seed, PROBE_KEY, 1, block])``; skeleton from ``default_rng([seed, PROBE_KEY, 0])``.
    The logged propensity is P(arm, level | slot type, eligibility); ``AssignmentMechanism`` re-samples it exactly.
  * writes: every knob of the family owned by the region's cells moves by level x step (``ric.feasible`` clips;
    a value already at its limit is a logged no-op). A write is not attempted in a second where an xApp requests the
    same knob, nor when ``feasible`` refuses it (dwell / min interval); it is retried for ``retry_s`` s, then logged
    as not applied. Inference is intention-to-treat on the ASSIGNMENT (never condition on actuation).
  * restore at ``t0 + obs`` (or on abort): each applied knob still at the probe value is written back to its
    pre-probe value (retried until the slot ends); a knob an xApp changed since is "superseded" and left alone.
  * abort-and-restore (predeclared, observable delivered KPM of the unit's cells, reports whose window starts at or
    after t0; evaluated identically for sham blocks, which have nothing to restore):
      LL   max LL p95 > max(abort_ll_abs_s, abort_ll_rel x pre-block mean) for abort_ll_consec fast reports;
      eMBB mean eMBB p5 < min(abort_embb_abs_bps, abort_embb_rel x pre-block mean) for abort_embb_consec reports;
      RLF  RLF count in a mob report > max(abort_rlf_abs, abort_rlf_rel x pre-block mean);
      ES   unit energy in an energy report > (1 + abort_energy_frac) x the last pre-block report.
    A breach restores immediately and is logged (block, time, reason).
  * a runtime guard refuses probe WRITES (never restores) once the probe's applied changes reach the cap.

Outputs (``run_probe_episode``): the episode ``Trace`` (``trace.py`` schema; probe writes appear in ``wr_*``) plus
  blk_* (B,)   slot, type (0=A, 1=B), t0, obs_end, slot_end, region, arm (index into ARMS), level, prop (logged
               P(arm, level | context)), elig (B, len(ARMS)) bool, n_knobs, n_applied, n_noop, n_failed, n_restored,
               n_superseded, abort (bool), abort_t (NaN = none), abort_reason (index into ABORT_REASONS, -1),
               knob_s (changed-knob-seconds at the probe value), cap_blocked
  pw_* (W,)    per knob write attempt outcome: blk, k (index into meta knobs), kind (0 probe, 1 restore), t, pre,
               target, out (index into PW_OUT)
  meta["probe"] version, key, config, arms, regions, units, is_macro_region, eligibility rules.
``collection_cost(probe_trace, ref_trace)``: paired excess vs the same-seed accept-all no-probe reference
(``run_reference``). Paired runs account for COST only; the inferential null is ``crt.py``'s.
"""
from __future__ import annotations

import dataclasses
import warnings
from collections import deque

import numpy as np

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.env import E6Env
from cdd_oran.envs.e6.ric import feasible, knob_get

from .features import neighbours_from_knobs
from .trace import Trace

PROBE_VERSION = "e6-probe/1"
PROBE_KEY = 8808                                   # RNG stream tag of the probe campaign (not an env seed)
FAMILIES = ("cio", "hys", "ttt", "ll_ratio", "carrier")
ARMS = FAMILIES + ("sham",)
STEP = {"cio": 2.0, "hys": 0.5, "ttt": 1.0, "ll_ratio": 0.10, "carrier": 1.0}   # one actuator step (ric.LIMITS)
ABORT_REASONS = ("ll", "embb", "rlf", "energy")
PW_OUT = ("applied", "noop", "failed", "cap_blocked", "superseded")


@dataclasses.dataclass(frozen=True)
class ProbeConfig:
    slot_a_s: int = 120            # type-A slot (fast families): observation + washout
    obs_a_s: int = 90
    slot_b_s: int = 240            # type-B slot (carrier): observation > 120 s dwell, then washout
    obs_b_s: int = 180
    p_slot_b: float = 0.3
    arms_a: tuple = ("cio", "hys", "ttt", "ll_ratio", "sham")
    probs_a: tuple = (0.2, 0.2, 0.2, 0.2, 0.2)
    arms_b: tuple = ("carrier", "sham")
    probs_b: tuple = (0.5, 0.5)
    max_treated: int = 2           # simultaneously treated units per slot
    cap_changes_per_hour: float = 3600.0   # probe knob changes (writes + restores), worst case at schedule time
    retry_s: int = 12              # probe-write retry window (covers the 10 s hys/ttt min interval)
    pre_s: int = 60                # pre-block baseline window (abort rules, conditioners)
    abort_ll_abs_s: float = 0.25
    abort_ll_rel: float = 2.0
    abort_ll_consec: int = 3
    abort_embb_abs_bps: float = 0.5e6
    abort_embb_rel: float = 0.5
    abort_embb_consec: int = 2
    abort_rlf_abs: float = 6.0
    abort_rlf_rel: float = 3.0
    abort_energy_frac: float = 0.25
    write_budget: float = 1e7      # env write budget (per scored hour): never binding, the cap is the probe's own


# ------------------------------------------------------------------------------------------ assignment mechanism
class AssignmentMechanism:
    """P(arm, level | slot type, eligibility). The ONLY randomness of the design that inference re-samples."""

    def __init__(self, pcfg: ProbeConfig):
        self.pcfg = pcfg
        self.base = {}
        for typ, arms, probs in ((0, pcfg.arms_a, pcfg.probs_a), (1, pcfg.arms_b, pcfg.probs_b)):
            v = np.zeros(len(ARMS))
            for a, p in zip(arms, probs, strict=True):
                v[ARMS.index(a)] = p
            self.base[typ] = v

    def arm_probs(self, slot_type: int, elig) -> np.ndarray:
        v = self.base[int(slot_type)] * np.asarray(elig, bool)
        s = v.sum()
        if s <= 0:
            raise ValueError("no eligible arm")
        return v / s

    @staticmethod
    def levels(arm: str) -> tuple:
        return (0.0,) if arm == "sham" else (1.0,) if arm == "carrier" else (-1.0, 1.0)

    def prob(self, slot_type: int, elig, arm: str, level: float) -> float:
        lv = self.levels(arm)
        if float(level) not in lv:
            return 0.0
        return float(self.arm_probs(slot_type, elig)[ARMS.index(arm)]) / len(lv)

    def sample(self, rng: np.random.Generator, slot_type: int, elig) -> tuple[str, float]:
        p = self.arm_probs(slot_type, elig)
        arm = ARMS[int(rng.choice(len(ARMS), p=p))]
        lv = self.levels(arm)
        return arm, float(lv[int(rng.integers(len(lv)))]) if len(lv) > 1 else lv[0]

    def conditional_draws(self, rng: np.random.Generator, slot_type, elig, family: str, n_draws: int) -> np.ndarray:
        """(n_draws, n) candidate values x = level if arm == family else 0, re-sampled row-wise from the mechanism
        CONDITIONAL on arm in {family, sham} (all other blocks' assignments held fixed)."""
        st = np.asarray(slot_type)
        el = np.asarray(elig, bool)
        n = len(st)
        pf = np.empty(n)
        fi, si = ARMS.index(family), ARMS.index("sham")
        for i in range(n):
            p = self.arm_probs(st[i], el[i])
            pf[i] = p[fi] / (p[fi] + p[si])
        is_f = rng.random((n_draws, n)) < pf[None, :]
        lv = np.asarray(self.levels(family))
        lev = lv[rng.integers(len(lv), size=(n_draws, n))] if len(lv) > 1 else np.full((n_draws, n), lv[0])
        return np.where(is_f, lev, 0.0)


# ------------------------------------------------------------------------------------------------ geometry
def region_units(knobs, cell_region) -> tuple[list, dict, dict]:
    """-> (regions, nbhd {g: set of neighbour regions}, cells {g: cell ids}) from the CIO relations."""
    reg = np.asarray(cell_region, int)
    nb = neighbours_from_knobs([tuple(k) for k in knobs], len(reg))
    regions = sorted({int(x) for x in reg})
    cells = {g: [int(c) for c in np.nonzero(reg == g)[0]] for g in regions}
    nbhd = {g: {int(reg[n]) for c in cells[g] for n in nb[c]} - {g} for g in regions}
    return regions, nbhd, cells


def family_knobs(knobs, cell_region, region: int, family: str) -> list:
    reg = np.asarray(cell_region, int)
    return [tuple(k) for k in knobs if k[0] == family and int(reg[int(k[1])]) == region]


# ------------------------------------------------------------------------------------------------ schedule
def make_schedule(seed: int, knobs, cell_region, start_s: float, end_s: float, pcfg: ProbeConfig) -> list:
    """Predeclared block list (dicts) for one episode; depends on the seed, layout and ``pcfg`` only."""
    mech = AssignmentMechanism(pcfg)
    regions, nbhd, _ = region_units(knobs, cell_region)
    unit = {g: {g} | nbhd[g] for g in regions}
    fk = {(g, f): family_knobs(knobs, cell_region, g, f) for g in regions for f in FAMILIES}
    elig_arms = {0: set(pcfg.arms_a), 1: set(pcfg.arms_b)}
    rng = np.random.default_rng([int(seed), PROBE_KEY, 0])
    cap_total = pcfg.cap_changes_per_hour * max(end_s - start_s, 0.0) / 3600.0
    used, blocks, t, slot = 0.0, [], float(start_s), 0
    while True:
        typ = int(rng.uniform() < pcfg.p_slot_b)
        length = pcfg.slot_b_s if typ else pcfg.slot_a_s
        if t + length > end_s:
            if typ and t + pcfg.slot_a_s <= end_s:
                typ, length = 0, pcfg.slot_a_s
            else:
                break
        obs = pcfg.obs_b_s if typ else pcfg.obs_a_s
        chosen = []
        for g in rng.permutation(regions):
            g = int(g)
            elig = np.array([(a == "sham" or len(fk[(g, a)]) > 0) and a in elig_arms[typ] for a in ARMS])
            if not elig[:len(FAMILIES)].any():
                continue
            if any(unit[g] & unit[h] for h, _ in chosen):
                continue
            worst = 2.0 * max(len(fk[(g, f)]) for f in FAMILIES if elig[ARMS.index(f)])
            if used + worst > cap_total + 1e-9:
                continue
            chosen.append((g, elig))
            used += worst
            if len(chosen) >= pcfg.max_treated:
                break
        for g, elig in chosen:
            b = len(blocks)
            arm, level = mech.sample(np.random.default_rng([int(seed), PROBE_KEY, 1, b]), typ, elig)
            blocks.append({"blk": b, "slot": slot, "type": typ, "t0": t, "obs_end": t + obs, "slot_end": t + length,
                           "region": g, "unit": sorted(unit[g]), "elig": elig, "arm": arm, "level": level,
                           "prop": mech.prob(typ, elig, arm, level)})
        t += length
        slot += 1
    return blocks


# ------------------------------------------------------------------------------------------------ runner
class ProbeRunner:
    """Arbiter wrapper: accept-all for xApp requests + the probe's writes. Two-phase loop:
    ``dec = r.act(obs); env.step_apply(dec); r.after()``."""

    def __init__(self, env: E6Env, pcfg: ProbeConfig | None = None, start_s: float | None = None,
                 end_s: float | None = None):
        if env.wg3:
            raise ValueError("the probe campaign writes knobs directly: build the env with wg3=False")
        self.env, self.pcfg = env, pcfg or ProbeConfig()
        cfg = env.cfg
        self.start_s = float(cfg.warmup_s if start_s is None else start_s)
        self.end_s = float(cfg.warmup_s + cfg.scored_s if end_s is None else end_s)
        self.cell_region = np.asarray(env.plant.lay.cell_site, int)
        self.knob_idx = {k: i for i, k in enumerate(env.knobs)}
        self.blocks = make_schedule(cfg.seed, env.knobs, self.cell_region, self.start_s, self.end_s, self.pcfg)
        self.regions, self.nbhd, self.cells = region_units(env.knobs, self.cell_region)
        for b in self.blocks:
            ucells = [c for g in b["unit"] for c in self.cells[g]]
            b.update({"ucells": np.array(ucells, int), "state": "scheduled", "knobs": {}, "abort": False,
                      "abort_t": np.nan, "abort_reason": -1, "knob_s": 0, "cap_blocked": 0, "base": None,
                      "cnt": {"ll": 0, "embb": 0}})
        self.hist = {g: deque(maxlen=400) for g in ("fast", "thp", "mob", "energy")}
        self.pw = []                    # (blk, k, kind, t, pre, target, out)
        self.cap_total = self.pcfg.cap_changes_per_hour * max(self.end_s - self.start_s, 0.0) / 3600.0
        self.changes = 0
        self._attempt = []              # (block, knob, kind, target, value before) attempted this second

    # -------------------------------------------------------------------------------------------- helpers
    def _target(self, k, fam, level, cur):
        if fam == "ttt":
            i = C.TTT_SET_MS.index(int(cur))
            return float(C.TTT_SET_MS[int(np.clip(i + int(level), 0, len(C.TTT_SET_MS) - 1))])
        if fam == "carrier":                                  # toggle one carrier
            return cur - 1.0 if cur >= C.MACRO_NTRX else cur + 1.0
        v, _ = feasible(self.env.plant, k, cur + level * STEP[fam], 1e18, {})   # range + quantisation only
        return v

    def _baseline(self, b, now):
        """Pre-block baseline of the unit (delivered reports whose window lies in (t0 - pre_s, t0])."""
        cells, t0, lo = b["ucells"], b["t0"], b["t0"] - self.pcfg.pre_s
        out = {}
        with np.errstate(all="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                f = [np.nanmax(r["ll_delay_p95"][cells]) for r in self.hist["fast"] if lo <= r["t0"] and r["t1"] <= t0]
                out["ll"] = float(np.nanmean(f)) if f and np.isfinite(f).any() else np.nan
                e = [np.nanmean(r["embb_thp_p5"][cells]) for r in self.hist["thp"] if lo <= r["t0"] and r["t1"] <= t0]
                out["embb"] = float(np.nanmean(e)) if e and np.isfinite(e).any() else np.nan
                m = [float(np.sum(r["rlf"][cells])) for r in self.hist["mob"] if lo - 60 <= r["t0"] and r["t1"] <= t0]
                out["rlf"] = float(np.mean(m)) if m else np.nan
                en = [float(np.sum(r["energy_j"][cells])) for r in self.hist["energy"] if r["t1"] <= t0]
                out["energy"] = en[-1] if en else np.nan
        return out

    def _breach(self, b, rep):
        """Abort reason index if ``rep`` (a report of the block's post-assignment window) breaches, else -1."""
        p, base, cells = self.pcfg, b["base"], b["ucells"]
        g = rep["gran"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if g == "fast":
                v = np.nanmax(rep["ll_delay_p95"][cells])
                thr = max(p.abort_ll_abs_s, p.abort_ll_rel * base["ll"]) if np.isfinite(base["ll"]) else \
                    p.abort_ll_abs_s
                b["cnt"]["ll"] = b["cnt"]["ll"] + 1 if np.isfinite(v) and v > thr else 0
                return 0 if b["cnt"]["ll"] >= p.abort_ll_consec else -1
            if g == "thp":
                v = np.nanmean(rep["embb_thp_p5"][cells])
                thr = min(p.abort_embb_abs_bps, p.abort_embb_rel * base["embb"]) if np.isfinite(base["embb"]) \
                    else p.abort_embb_abs_bps
                b["cnt"]["embb"] = b["cnt"]["embb"] + 1 if np.isfinite(v) and v < thr else 0
                return 1 if b["cnt"]["embb"] >= p.abort_embb_consec else -1
        if g == "mob":
            v = float(np.sum(rep["rlf"][cells]))
            thr = max(p.abort_rlf_abs, p.abort_rlf_rel * base["rlf"]) if np.isfinite(base["rlf"]) else p.abort_rlf_abs
            return 2 if v > thr else -1
        if g == "energy" and np.isfinite(base["energy"]):
            return 3 if float(np.sum(rep["energy_j"][cells])) > (1 + p.abort_energy_frac) * base["energy"] else -1
        return -1

    # -------------------------------------------------------------------------------------------- phases
    def act(self, obs: dict) -> dict:
        env, plant, now = self.env, self.env.plant, float(obs["t"])
        for rep in obs["new_reports"]:
            self.hist[rep["gran"]].append(rep)
        requested = {r["knob"] for r in obs["requests"]}
        writes, self._attempt = [], []
        for b in self.blocks:
            if b["state"] == "scheduled" and now >= b["t0"]:
                b["state"], b["base"] = "active", self._baseline(b, now)
                if b["arm"] != "sham":
                    for k in family_knobs(env.knobs, self.cell_region, b["region"], b["arm"]):
                        cur = knob_get(plant, k)
                        tgt = self._target(k, b["arm"], b["level"], cur)
                        b["knobs"][k] = {"pre": cur, "target": tgt, "state": "noop" if abs(tgt - cur) < 1e-9
                                         else "pending", "val": None}
                        if abs(tgt - cur) < 1e-9:
                            self.pw.append((b["blk"], self.knob_idx[k], 0, now, cur, tgt, PW_OUT.index("noop")))
            if b["state"] == "active":
                for rep in obs["new_reports"]:
                    if rep["t0"] >= b["t0"] and not b["abort"]:
                        why = self._breach(b, rep)
                        if why >= 0:
                            b["abort"], b["abort_t"], b["abort_reason"] = True, now, why
                if b["abort"] or now >= b["obs_end"]:
                    b["state"] = "restoring"
            if b["state"] == "active":           # probe writes (retry window)
                for k, s in b["knobs"].items():
                    if s["state"] != "pending":
                        continue
                    if now > b["t0"] + self.pcfg.retry_s:
                        s["state"] = "failed"
                        self.pw.append((b["blk"], self.knob_idx[k], 0, now, s["pre"], s["target"],
                                        PW_OUT.index("failed")))
                        continue
                    if self.changes >= self.cap_total:
                        s["state"] = "cap_blocked"
                        b["cap_blocked"] += 1
                        self.pw.append((b["blk"], self.knob_idx[k], 0, now, s["pre"], s["target"],
                                        PW_OUT.index("cap_blocked")))
                        continue
                    if k in requested or not feasible(plant, k, s["target"], now, env.last_change)[1]:
                        continue
                    writes.append((k, s["target"]))
                    self._attempt.append((b, k, 0, s["target"], knob_get(plant, k)))
            if b["state"] == "restoring":        # restores (until the slot ends)
                done = True
                for k, s in b["knobs"].items():
                    if s["state"] == "pending":
                        s["state"] = "failed"
                        self.pw.append((b["blk"], self.knob_idx[k], 0, now, s["pre"], s["target"],
                                        PW_OUT.index("failed")))
                    if s["state"] != "applied":
                        continue
                    cur = knob_get(plant, k)
                    if abs(cur - s["val"]) > 1e-9:
                        s["state"] = "superseded"
                        self.pw.append((b["blk"], self.knob_idx[k], 1, now, s["pre"], s["pre"],
                                        PW_OUT.index("superseded")))
                        continue
                    done = False
                    if now >= b["slot_end"]:
                        s["state"] = "restore_failed"
                        self.pw.append((b["blk"], self.knob_idx[k], 1, now, cur, s["pre"], PW_OUT.index("failed")))
                        continue
                    if k in requested or not feasible(plant, k, s["pre"], now, env.last_change)[1]:
                        continue
                    writes.append((k, s["pre"]))
                    self._attempt.append((b, k, 1, s["pre"], cur))
                if done or now >= b["slot_end"]:
                    b["state"] = "done"
        return {"decisions": ["accept"] * len(obs["requests"]), "writes": writes, "rollback": []}

    def after(self) -> None:
        """After ``env.step_apply``: attribute this second's write outcomes, count changed-knob-seconds."""
        plant, now = self.env.plant, float(self.env.sec)
        for b, k, kind, tgt, before in self._attempt:
            cur = knob_get(plant, k)
            s = b["knobs"][k]
            ok = abs(cur - tgt) < 1e-9 and abs(before - tgt) > 1e-9
            if not ok:
                continue                               # retried next second (or times out)
            self.changes += 1
            if kind == 0:
                s["state"], s["val"] = "applied", cur
            else:
                s["state"] = "restored"
            self.pw.append((b["blk"], self.knob_idx[k], kind, now, before if kind == 0 else s["val"], tgt,
                            PW_OUT.index("applied")))
        self._attempt = []
        for b in self.blocks:
            if b["state"] in ("active", "restoring"):
                b["knob_s"] += sum(1 for k, s in b["knobs"].items()
                                   if s["state"] == "applied" and abs(knob_get(plant, k) - s["val"]) < 1e-9)

    # -------------------------------------------------------------------------------------------- output
    def table(self) -> dict:
        B = self.blocks

        def arr(key, dt, f=None):
            return np.array([f(b) if f else b[key] for b in B], dt)

        def count(*states):
            return lambda b: sum(1 for s in b["knobs"].values() if s["state"] in states)

        out = {"blk_slot": arr("slot", np.int32), "blk_type": arr("type", np.int8), "blk_t0": arr("t0", float),
               "blk_obs_end": arr("obs_end", float), "blk_slot_end": arr("slot_end", float),
               "blk_region": arr("region", np.int16), "blk_arm": arr(None, np.int8, lambda b: ARMS.index(b["arm"])),
               "blk_level": arr("level", float), "blk_prop": arr("prop", float),
               "blk_elig": np.array([b["elig"] for b in B], bool).reshape(len(B), len(ARMS)),
               "blk_n_knobs": arr(None, np.int16, lambda b: len(b["knobs"])),
               "blk_n_applied": arr(None, np.int16, lambda b: sum(
                   1 for p in self.pw if p[0] == b["blk"] and p[2] == 0 and p[6] == 0)),
               "blk_n_noop": arr(None, np.int16, count("noop")),
               "blk_n_failed": arr(None, np.int16, count("failed")),
               "blk_n_restored": arr(None, np.int16, count("restored")),
               "blk_n_superseded": arr(None, np.int16, count("superseded")),
               "blk_abort": arr("abort", bool), "blk_abort_t": arr("abort_t", float),
               "blk_abort_reason": arr("abort_reason", np.int8), "blk_knob_s": arr("knob_s", np.int32),
               "blk_cap_blocked": arr("cap_blocked", np.int16)}
        pw = np.array(self.pw, float).reshape(-1, 7)
        for j, (name, dt) in enumerate((("blk", np.int32), ("k", np.int16), ("kind", np.int8), ("t", np.int32),
                                        ("pre", float), ("target", float), ("out", np.int8))):
            out["pw_" + name] = pw[:, j].astype(dt)
        return out

    def meta(self) -> dict:
        return {"version": PROBE_VERSION, "key": PROBE_KEY, "config": dataclasses.asdict(self.pcfg),
                "arms": list(ARMS), "families": list(FAMILIES), "step": dict(STEP), "abort_reasons": list(ABORT_REASONS),
                "pw_out": list(PW_OUT), "regions": self.regions,
                "units": {str(g): sorted({g} | self.nbhd[g]) for g in self.regions},
                "start_s": self.start_s, "end_s": self.end_s, "probe_changes": self.changes,
                "cap_total": self.cap_total, "not_a_wg3_policy": True}


def _score(env):
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in env.score().items()}


def run_probe_episode(cfg: C.E6Config, pcfg: ProbeConfig | None = None, snapshot_s: int = 60) -> Trace:
    """One operator DEV probe episode (``wg3=False``, accept-all xApps + probe writes); returns the trace."""
    pcfg = pcfg or ProbeConfig()
    env = E6Env(cfg, write_budget=pcfg.write_budget, log=False, wg3=False, trace=True, trace_snapshot_s=snapshot_s)
    runner = ProbeRunner(env, pcfg)
    while env.sec < env.total_s:
        obs = env.step_propose()
        dec = runner.act(obs)
        env.step_apply(dec)
        runner.after()
    tr = env.get_trace()
    tr.arrays.update(runner.table())
    tr.meta["probe"] = runner.meta()
    tr.meta["probe"]["score"] = _score(env)
    return tr


def run_reference(cfg: C.E6Config, snapshot_s: int = 60) -> Trace:
    """Same-seed accept-all no-probe reference (cost pairing only)."""
    env = E6Env(cfg, log=False, wg3=False, trace=True, trace_snapshot_s=snapshot_s)
    env.run(None)
    tr = env.get_trace()
    tr.meta["reference"] = {"kind": "accept_all_no_probe", "score": _score(env)}
    return tr


# ------------------------------------------------------------------------------------------------ cost
def _viol_series(tr: Trace):
    a = tr.arrays
    m = a["scored"]
    viol = a["lab_viol"][m].astype(float)          # (T, C, 3)
    ue = a["lab_ue"][m].astype(float)
    return viol, ue, a["lab_rlf"][m].astype(float), a["lab_severe"][m].astype(float), a["lab_energy_j"][m]


def collection_cost(probe: Trace, ref: Trace) -> dict:
    """Paired excess (probe - same-seed no-probe reference) of the episode score + per-slice / tail / churn terms,
    and the probe's own accounting (blocks, applied writes, restores, aborts, changed-knob-seconds)."""
    sp, sr = probe.meta["probe"]["score"], ref.meta["reference"]["score"]
    out = {"excess_" + k: float(sp[k] - sr[k]) for k in ("svr", "viol_frac", "ll_viol", "embb_viol", "outage_viol",
                                                          "severe", "energy_kwh", "rlf_per_ue_h", "ho_per_ue_h",
                                                          "pingpong", "changes")}
    vp, up, rp, _, _ = _viol_series(probe)
    vr, ur, rr, _, _ = _viol_series(ref)
    for s, nm in enumerate(("ll", "embb", "be")):
        out[f"excess_viol_ue_s_{nm}"] = float(vp[..., s].sum() - vr[..., s].sum())
    fp = vp.sum((1, 2)) / np.maximum(up.sum((1, 2)), 1)
    fr = vr.sum((1, 2)) / np.maximum(ur.sum((1, 2)), 1)
    for q in (95, 99):
        out[f"excess_viol_frac_p{q}"] = float(np.percentile(fp, q) - np.percentile(fr, q))
    cp = vp.sum(2) / np.maximum(up.sum(2), 1)                 # per cell-second
    cr = vr.sum(2) / np.maximum(ur.sum(2), 1)
    out["excess_cell_viol_frac_p99"] = float(np.percentile(cp, 99) - np.percentile(cr, 99))
    out["excess_rlf_total"] = float(rp.sum() - rr.sum())
    a = probe.arrays
    arm = a["blk_arm"]
    treated = arm != ARMS.index("sham")
    out.update({"blocks": int(len(arm)), "treated_blocks": int(treated.sum()), "sham_blocks": int((~treated).sum()),
                "probe_writes_applied": int(a["blk_n_applied"].sum()), "restores": int(a["blk_n_restored"].sum()),
                "superseded": int(a["blk_n_superseded"].sum()), "failed": int(a["blk_n_failed"].sum()),
                "aborts_treated": int(a["blk_abort"][treated].sum()), "aborts_sham": int(a["blk_abort"][~treated].sum()),
                "changed_knob_s": int(a["blk_knob_s"].sum()), "probe_changes": int(probe.meta["probe"]["probe_changes"]),
                "blocks_by_arm": {ARMS[i]: int((arm == i).sum()) for i in range(len(ARMS))},
                "applied_blocks_by_arm": {ARMS[i]: int(((arm == i) & (a["blk_n_applied"] > 0)).sum())
                                          for i in range(len(ARMS))}})
    return out


# ------------------------------------------------------------------------------------------------ outcomes
KPI_FAMILIES = ("prb_util", "ll_delay_p95", "embb_thp_p5", "rlf", "too_late", "too_early", "energy_w")


def _window(a, gran, lo, hi):
    t0, t1 = a.get(f"kpm_{gran}_t0"), a.get(f"kpm_{gran}_t1")
    if t0 is None or len(t0) == 0:
        return np.zeros(0, int)
    return np.nonzero((t0 >= lo) & (t1 <= hi))[0]


def region_kpis(trace: Trace, cells, lo: float, hi: float) -> dict:
    """Delivered-KPM KPI families over the cells for report windows inside [lo, hi] (observable arrays only).
    prb_util mean; ll_delay_p95 / embb_thp_p5 nanmean; rlf / too_late / too_early (= too_early + wrong_cell +
    ping-pong) per second summed over cells; energy_w summed over cells. NaN when no report fits."""
    a = trace.arrays
    cells = np.asarray(cells, int)
    out = dict.fromkeys(KPI_FAMILIES, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        j = _window(a, "fast", lo, hi)
        if len(j):
            out["prb_util"] = float(np.mean(a["kpm_fast_prb_util"][j][:, cells]))
            out["ll_delay_p95"] = float(np.nanmean(a["kpm_fast_ll_delay_p95"][j][:, cells]))
        j = _window(a, "thp", lo, hi)
        if len(j):
            out["embb_thp_p5"] = float(np.nanmean(a["kpm_thp_embb_thp_p5"][j][:, cells]))
        j = _window(a, "mob", lo, hi)
        if len(j):
            dur = float(np.sum(a["kpm_mob_t1"][j] - a["kpm_mob_t0"][j]))
            out["rlf"] = float(a["kpm_mob_rlf"][j][:, cells].sum()) / dur
            out["too_late"] = float(a["kpm_mob_too_late"][j][:, cells].sum()) / dur
            out["too_early"] = float(sum(a[f"kpm_mob_{x}"][j][:, cells].sum()
                                         for x in ("too_early", "wrong_cell", "pingpong"))) / dur
        j = _window(a, "energy", lo, hi)
        if len(j):
            dur = float(np.sum(a["kpm_energy_t1"][j] - a["kpm_energy_t0"][j]))
            out["energy_w"] = float(a["kpm_energy_energy_j"][j][:, cells].sum()) / dur
    return out


def block_outcomes(trace: Trace, blocks: dict | None = None) -> dict:
    """Per block: post window [t0, obs_end] and pre window [t0 - pre_s, t0] KPIs of the treated region ("own")
    and of its neighbourhood ("nbr"). ``blocks`` = another probe trace's blk_* arrays + meta (placebo: outcomes of a
    no-probe reference under the probe schedule). Returns {"post"/"pre": {scope_kpi: (B,)}}."""
    src = blocks if blocks is not None else {"arrays": trace.arrays, "meta": trace.meta["probe"]}
    a, pm = src["arrays"], src["meta"]
    reg = np.asarray(trace.meta["cell_region"], int)
    pre_s = float(pm["config"]["pre_s"])
    n = len(a["blk_t0"])
    post = {}
    pre = {}
    for i in range(n):
        g = int(a["blk_region"][i])
        t0, t1 = float(a["blk_t0"][i]), float(a["blk_obs_end"][i])
        for scope, regs in (("own", [g]), ("nbr", [h for h in pm["units"][str(g)] if h != g])):
            cells = np.nonzero(np.isin(reg, regs))[0]
            for lo, hi, dst in ((t0, t1, post), (t0 - pre_s, t0, pre)):
                v = region_kpis(trace, cells, lo, hi) if len(cells) else dict.fromkeys(KPI_FAMILIES, np.nan)
                for f, x in v.items():
                    dst.setdefault(f"{scope}_{f}", np.full(n, np.nan))[i] = x
    return {"post": post, "pre": pre}

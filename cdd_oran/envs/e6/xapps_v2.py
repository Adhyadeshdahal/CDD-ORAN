"""E6 xApp stack v2 (``E6-xapps-v2``): a SEPARATE experiment after the predeclared gate-A kill of the v1 stack.

Specification, citations and every constant's provenance: ``docs/benchmark/E6_V2_XAPPS_SPEC.md`` (DRAFT until the
v2 contract ``docs/benchmark/E6_V2_CONTRACT.json`` is frozen). v1 (``xapps.py``) is untouched; v2 reuses its ``ES`` and
``SliceSLA`` classes unchanged and replaces the two mobility xApps. Mechanism-level motivation: the exploratory v1
audit ``docs/benchmark/E6_V1_DIAGNOSTICS.md`` (development evidence only; no constant here is fitted to it).

  * ``NRT``: ANR-like neighbour relation table built at run time from delivered KPM only (HO / RLF statistics and UE
    measurement reports) instead of the static 6-nearest-site ``Layout.neighbours`` list.
  * ``MLB`` (role name "TS"): mobility load balancing / traffic steering (TR 36.902 cl. 4.6, TS 28.313 LBO). Decides on
    window-averaged load (active UEs per available shared capacity) and window-averaged eMBB p5 rate, counts border UEs
    a bounded CIO step would actually move, estimates the moved UEs' rate at source and target, pushes a cell pair
    only after the condition holds for a hold time, pushes / releases each pair as ONE antisymmetric decision, verifies
    that UEs actually moved, rolls back on no-move / neighbour degradation / HO-guard trip, and releases its own offset
    once the source has recovered.
  * ``MROv2`` (role name "MRO"): mobility robustness optimisation (TR 36.902 cl. 4.5, TS 28.313 MRO). Too-late evidence
    = too-late RLFs + a pre-RLF low-SINR dwell KPI (new ``mobq`` report); acts only on post-change evidence (a full
    window after its own last change on the cell); cell-wide issues move Hys or TTT in both directions within the
    TR 36.839 parameter-set envelope (never Hys+ on a too-late cell); single-pair issues move that pair's CIO
    (bounded); an overshoot reverses the last step and locks that direction.

Role names stay "TS" / "MRO" so every arbiter, subset and published baseline keyed by xApp name runs unchanged; the
stack is identified by the mix name (``V2_*``) and ``XApp.stack``.

KPM extension: ``KPMV2`` adds two report granularities, ``mr`` (5 s) and ``mobq`` (30 s), whose delay/drop draws come
from their own RNG key, so every v1 report, delay and drop is bit-identical under ``KPMV2``.

Registration without editing v1 files: ``make_env_v2(cfg, **kw)`` (spec section 2.3 lists the equivalent permanent
one-line hooks in env.py). Nothing here runs at import time.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .ric import KPM
from .sim import BE, EMBB, NOISE_W, Plant
from .xapps import ES, MIXES, SliceSLA, XApp

STACK_VERSION = "E6-xapps-v2"

# ---------------------------------------------------------------------------------------------------- KPM extension
MR_GRAN_S = 5                 # [A] same period as the ``thp`` report: MLB reads thp and mr on one clock
MOBQ_GRAN_S = 30              # [A] same period and alignment as the ``mob`` report (MRO window unit)
MR_BINS_DB = 6                # [S range] CIO actuator half-range (config.CIO_RANGE)
MR_STREAM_TAG = 7101          # RNG key for mr/mobq delay/drop draws; disjoint from sim.STREAMS (1..9) and registry tags


def _sinr_est(l3, occ):
    """RSRQ-like SINR estimate (dB) of every UE toward every cell from measured L3 RSRP and the cells' reported PRB
    occupancy (X2 resource-status style): P_c / (sum_{j != c} P_j occ_j + N)."""
    pw = 10 ** (np.asarray(l3, float) / 10) / 1000.0
    load = pw * np.asarray(occ, float)[None, :]
    tot = load.sum(1, keepdims=True)
    return 10 * np.log10(pw / (tot - load + NOISE_W) + 1e-30)


def border_snapshot(plant):
    """UE measurement-report aggregate at this instant (what the gNB can build from periodic L3 measurement reports).

    For every connected UE (not in HO interruption / RLF outage) served by s and every other cell n, the A3 margin
    m = L3(n) + CIO[s, n] - Hys[s] - L3(s) (dB; A3 entering condition is m > 0). A UE with m in (-k, 0] would enter A3
    toward n after a +k dB CIO change. Returns a dict:
      conn        (nc,)        connected UEs per serving cell
      border_all  (nc, nc, K)  cumulative: [s, n, k-1] = # UEs of s with m to n in (-k, 0], k = 1..K
      border_act  (nc, nc, K)  same, UEs with a non-empty buffer at the snapshot
      se_src      (nc, nc, K)  mean estimated spectral efficiency at the SERVING cell of the active border UEs in bin <= k
      se_tgt      (nc, nc, K)  same, at the TARGET cell n (NaN where the bin is empty)
    Sleeping / unavailable cells have gain -300 dB and never fall inside a bin."""
    nc, n, K = plant.nc, plant.n, MR_BINS_DB
    now = plant.t * C.TICK_S
    serv = np.asarray(plant.serv)
    ok = np.asarray(plant.int_until) <= now
    idx = np.arange(n)
    l3 = np.asarray(plant.l3, float)
    margin = l3 + plant.cio[serv] - plant.hys[serv][:, None] - l3[idx, serv][:, None]
    k = np.floor(-margin).astype(np.int64) + 1                        # bin index 1..K for m in (-K, 0]
    hit = ok[:, None] & (margin <= 0.0) & (margin > -K)
    hit[idx, serv] = False
    act = np.asarray(plant.q) > 1e-6
    se = Plant.se(_sinr_est(l3, plant.occ))                          # (n, nc)
    u, c = np.nonzero(hit)
    kb = k[u, c] - 1
    ball = np.zeros((nc, nc, K))
    np.add.at(ball, (serv[u], c, kb), 1.0)
    a = act[u]
    ua, ca, ka = u[a], c[a], kb[a]
    bact = np.zeros((nc, nc, K))
    ssrc = np.zeros((nc, nc, K))
    stgt = np.zeros((nc, nc, K))
    np.add.at(bact, (serv[ua], ca, ka), 1.0)
    np.add.at(ssrc, (serv[ua], ca, ka), se[ua, serv[ua]])
    np.add.at(stgt, (serv[ua], ca, ka), se[ua, ca])
    bact, ssrc, stgt = np.cumsum(bact, 2), np.cumsum(ssrc, 2), np.cumsum(stgt, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        se_src = np.where(bact > 0, ssrc / np.maximum(bact, 1), np.nan)
        se_tgt = np.where(bact > 0, stgt / np.maximum(bact, 1), np.nan)
    conn = np.bincount(serv[ok], minlength=nc).astype(float)
    return {"conn": conn, "border_all": np.cumsum(ball, 2), "border_act": bact, "se_src": se_src, "se_tgt": se_tgt}


def lowq_snapshot(plant):
    """Pre-RLF low-quality dwell sample (1 s per snapshot): for every connected UE whose serving SINR (the plant's RLF
    monitor, ``sinr_ewma``) is below Qout while another cell is measured stronger, 1 s is attributed to
    (serving s, strongest other cell n). Returns (nc, nc) seconds."""
    nc = plant.nc
    now = plant.t * C.TICK_S
    serv = np.asarray(plant.serv)
    idx = np.arange(plant.n)
    l3 = np.asarray(plant.l3, float).copy()
    ls = l3[idx, serv]
    l3[idx, serv] = -np.inf
    best = np.argmax(l3, 1)
    sel = (np.asarray(plant.int_until) <= now) & (np.asarray(plant.sinr_ewma) < C.QOUT_DB) & (l3[idx, best] > ls)
    out = np.zeros((nc, nc))
    np.add.at(out, (serv[sel], best[sel]), 1.0)
    return out


class KPMV2(KPM):
    """``ric.KPM`` plus the ``mr`` and ``mobq`` reports. The parent's per-second RNG sequence is consumed first and is
    untouched; the extra reports draw drop/delay from ``default_rng([seed, 6600, MR_STREAM_TAG, sec, j])`` (j = 0 mr,
    1 mobq) with the same delay/drop law."""

    def __init__(self, cfg, plant):
        super().__init__(cfg, plant)
        self.lowq = np.zeros((plant.nc, plant.nc))

    def _send(self, sec, j, rep):
        r = np.random.default_rng([int(self.cfg.seed), 6600, MR_STREAM_TAG, int(sec), j])
        if r.uniform() < self.drop:
            return
        self.in_flight.append((sec + r.uniform(*self.delay), rep))

    def second(self, sec):
        super().second(sec)
        self.lowq += lowq_snapshot(self.plant)
        if sec % MR_GRAN_S == 0:
            self._send(sec, 0, {"gran": "mr", "t0": sec - MR_GRAN_S, "t1": sec, **border_snapshot(self.plant)})
        if sec % MOBQ_GRAN_S == 0:
            self._send(sec, 1, {"gran": "mobq", "t0": sec - MOBQ_GRAN_S, "t1": sec, "lowq_dwell": self.lowq})
            self.lowq = np.zeros_like(self.lowq)


# ---------------------------------------------------------------------------------------------------- ANR-like NRT
class NRT:
    """ANR-like neighbour relation table (TS 36.300 cl. 22.3.2a ANR / TS 38.300 cl. 15.3.3 [V]) built from delivered KPM
    only. A relation (s, n) is added -- in both directions [A] -- when a ``mob`` report shows any HO attempt, too-late /
    too-early / wrong-cell RLF or ping-pong between s and n, or when an ``mr`` report shows a UE of s measuring n within
    the report's margin range (``border_all[s, n, K-1] > 0``). Starts empty; relations are never removed during an
    episode [A: ANR NR removal is an OAM / ageing decision on a far longer time scale]."""

    MOB_KEYS = ("ho_att", "too_late", "too_early", "wrong_cell", "pingpong")

    def __init__(self, nc):
        self.rel = np.zeros((nc, nc), bool)

    def update(self, reps):
        for rep in reps:
            if rep["gran"] == "mob":
                m = sum(np.asarray(rep[k]) for k in self.MOB_KEYS) > 0
            elif rep["gran"] == "mr":
                m = np.asarray(rep["border_all"])[:, :, -1] > 0
            else:
                continue
            m = m | m.T
            np.fill_diagonal(m, False)
            self.rel |= m

    def neighbours(self, s):
        return np.flatnonzero(self.rel[s]).tolist()


# ---------------------------------------------------------------------------------------------------- shared helpers
def _cap_share(p, carriers):
    """Shared-pool capacity share per cell for eMBB/BE: active carriers / carriers x (1 - LL reservation); 0 if the
    cell is asleep. ``carriers`` = the ``carriers`` field of the latest delivered ``fast`` report."""
    kappa = np.asarray(carriers, float) / np.asarray(p.n_trx, float) * (1.0 - np.asarray(p.ll_ratio, float))
    return np.where(np.asarray(p.asleep, bool), 0.0, kappa)


class MLB(XApp):
    """Mobility load balancing / traffic steering, v2 (role "TS"). Spec: E6_V2_XAPPS_SPEC.md section 3."""

    name, cadence = "TS", 10.0          # [A] v1 TS cadence (inherited)
    stack = STACK_VERSION
    GAMMA = 0.20                        # [A] required moved-UE rate gain (target rate >= 1.2 x source rate)
    CIO_MAX = 4.0                       # [A] MLB offset band |CIO| <= 4 dB (v1 TS cio_max, inherited)
    MAX_STEP = 2.0                      # [S range] ric.LIMITS["cio"] max step per change
    HO_GUARD = 0.10                     # [A] pair early-event ratio (= MRO too-early threshold, inherited from v1)
    HO_MIN_EV = 3                       # [A] minimum events to act on (v1 MRO / SMO evidence rule, inherited)
    HOLD_N = 2                          # [A] push only after the push condition held on 2 consecutive cycles (20 s)
    BLOCK_S = 120.0                     # [A] pair block after a rollback (= ric.LIMITS carrier/sleep dwell)
    RESTORE_S = 60.0                    # [A] source recovered this long -> release own offset (= MRO window)
    N_FAST, N_THP = 10, 2               # [A] one cadence of 1 s fast reports; two 5 s thp reports = 10 s

    def __init__(self, env, idx):
        super().__init__(env, idx)
        self.gamma = self.GAMMA * self.scale
        self.ho_guard = self.HO_GUARD * self.scale
        self.nrt = NRT(self.p.nc)
        self.fast_hist, self.thp_hist = [], []
        self.mr = None
        self.mob = None
        self.own = {}                   # (s, n) -> list of own applied pushes {"k", "t", "n_ok"}
        self.block_until = {}           # (s, n) -> t
        self.last_step_t = {}           # s -> time of the last own applied change with s as source
        self.ok_since = {}              # s -> time since the source stopped suffering
        self.hold = {}                  # (s, n) -> consecutive cycles the push condition held
        self._meta = {}                 # forward knob -> intent of the pending pair decision (read in ``result``)

    def observe(self, reps):
        super().observe(reps)
        self.nrt.update(reps)
        for rep in reps:
            g = rep["gran"]
            if g == "fast":
                self.fast_hist = (self.fast_hist + [rep])[-self.N_FAST:]
            elif g == "thp":
                self.thp_hist = (self.thp_hist + [rep])[-self.N_THP:]
            elif g == "mr":
                self.mr = rep
            elif g == "mob":
                self.mob = rep

    # -------------------------------------------------------------------------------------------- measurements
    def _state(self):
        """Window averages: A = mean backlogged eMBB+BE UEs (all fast reports of the cadence), kappa = shared capacity
        share, suffer = window-mean eMBB p5 rate below the SLA target (NaN = no backlogged eMBB = not suffering)."""
        A = np.mean([h["act_ue"][:, EMBB] + h["act_ue"][:, BE] for h in self.fast_hist], 0)
        kappa = _cap_share(self.p, self.fast_hist[-1]["carriers"])
        p5s = np.array([np.asarray(h["embb_thp_p5"], float) for h in self.thp_hist])
        cnt = (~np.isnan(p5s)).sum(0)
        p5 = np.where(cnt > 0, np.nansum(p5s, 0) / np.maximum(cnt, 1), np.nan)
        suffer = np.nan_to_num(p5, nan=np.inf) < C.EMBB_THP_TARGET_BPS           # [A] E6 eMBB SLA target itself
        return A, kappa, suffer

    def _ho_bad(self, s, n, since=-np.inf):
        """HO-robustness guard on pair (s, n) from the latest mob report (only if its window starts at/after
        ``since``): too-early + wrong-cell + ping-pong (both directions) over the pair's HO attempts."""
        if self.mob is None or self.mob["t0"] < since:
            return False
        m = self.mob
        ev = m["too_early"][s, n] + m["wrong_cell"][s, n] + m["pingpong"][s, n] + m["pingpong"][n, s]
        return ev >= self.HO_MIN_EV and ev / max(m["ho_att"][s, n], 1.0) > self.ho_guard

    def _no_move(self, s, n, since):
        """Moved-UE verification: the first full mob window after the push shows no HO attempt s -> n."""
        return self.mob is not None and self.mob["t0"] >= since and self.mob["ho_att"][s, n] <= 0

    def _pair(self, s, n, k, now, intent):
        """ONE pair decision (TS 36.423 Mobility Settings Change style): CIO[s, n] += k and CIO[n, s] -= k together."""
        f = float(np.clip(self.p.cio[s, n] + k, C.CIO_RANGE[0], C.CIO_RANGE[1]))
        b = float(np.clip(self.p.cio[n, s] - k, C.CIO_RANGE[0], C.CIO_RANGE[1]))
        out = [self.req(("cio", s, n), f, now), self.req(("cio", n, s), b, now)]
        if out[0] is not None:
            self._meta[("cio", s, n)] = intent
        return out

    def _gain(self, s, n, k, A, kappa):
        """Moved-set rate ratio (target / source) and the no-overshoot test for the UEs a +k dB step would move.
        Returns (ratio, m, ok) with m = # active border UEs moved."""
        m = float(self.mr["border_act"][s, n, k - 1])
        if m < 1.0:
            return 0.0, m, False
        se_s, se_n = self.mr["se_src"][s, n, k - 1], self.mr["se_tgt"][s, n, k - 1]
        if not np.isfinite(se_s) or not np.isfinite(se_n) or se_n <= 0:
            return 0.0, m, False
        r_src = max(se_s, 1e-9) * kappa[s] / max(A[s], 1.0)
        r_tgt = se_n * kappa[n] / (A[n] + m)
        no_overshoot = kappa[n] / (A[n] + m) >= kappa[s] / max(A[s] - m, 1.0)
        return r_tgt / r_src, m, no_overshoot

    # -------------------------------------------------------------------------------------------- decision
    def propose(self, now):
        if not self.fast_hist or not self.thp_hist or self.mr is None:
            return []
        A, kappa, suffer = self._state()
        out, busy = [], set()
        # 1) evaluate own pushes: no UE moved, HO guard trip, or a previously fine neighbour now suffering -> rollback
        for (s, n), steps in list(self.own.items()):
            if not steps:
                continue
            last = steps[-1]
            fresh = self.thp_hist[-1]["t0"] >= last["t"]
            if self._no_move(s, n, last["t"]) or self._ho_bad(s, n, since=last["t"]) or \
                    (fresh and last["n_ok"] and suffer[n]):
                out += self._pair(s, n, -last["k"], now, {"type": "rollback", "pair": (s, n)})
                self.block_until[(s, n)] = now + self.BLOCK_S
                busy |= {s, n}
        # 2) release own offset once the source has recovered for RESTORE_S (hysteresis: enter < target, leave >= target
        #    held for RESTORE_S)
        for s in range(self.p.nc):
            if suffer[s]:
                self.ok_since.pop(s, None)
                continue
            self.ok_since.setdefault(s, now)
            if now - self.ok_since[s] < self.RESTORE_S or s in busy:
                continue
            for n in sorted({n for (s_, n), st in self.own.items() if s_ == s and st}):
                if n not in busy:
                    out += self._pair(s, n, -1.0, now, {"type": "release", "pair": (s, n)})
                    busy |= {s, n}
                    break
        # 3) new pushes: suffering source -> the NRT neighbour whose moved-UE rate gain is largest
        t_ev = min(self.thp_hist[0]["t0"], self.mr["t0"], self.fast_hist[0]["t0"])   # oldest evidence in use
        held = {}
        for s in range(self.p.nc):
            if not suffer[s] or s in busy or kappa[s] <= 0:
                continue
            if t_ev < self.last_step_t.get(s, -np.inf):                   # evidence must post-date the last change
                continue
            best = None
            for n in self.nrt.neighbours(s):
                if n in busy or kappa[n] <= 0 or suffer[n] or now < self.block_until.get((s, n), -np.inf):
                    continue
                if self._ho_bad(s, n) or self.own.get((n, s)):            # never push against an own reverse push
                    continue
                cio = self.p.cio[s, n]
                for k in (1, 2):                                          # minimal step that moves an active UE
                    if k > self.MAX_STEP or cio + k > self.CIO_MAX + 1e-9:
                        break
                    ratio, m, ok = self._gain(s, n, k, A, kappa)
                    if m < 1.0:
                        continue
                    if ok and ratio >= 1.0 + self.gamma and (best is None or ratio > best[0]):
                        best = (ratio, n, k)
                    break
            if best is None:
                continue
            _, n, k = best
            held[(s, n)] = self.hold.get((s, n), 0) + 1
            if held[(s, n)] >= self.HOLD_N:
                out += self._pair(s, n, float(k), now, {"type": "push", "pair": (s, n), "n_ok": not suffer[n]})
                busy |= {s, n}
                held[(s, n)] = 0
        self.hold = held
        return [r for r in out if r]

    def result(self, r, accepted, applied, now):
        super().result(r, accepted, applied, now)
        intent = self._meta.pop(r["knob"], None)
        if intent is None or not accepted:
            return
        d = float(applied) - float(r["cur"])
        if abs(d) < 1e-9:
            return
        pair = intent["pair"]
        steps = self.own.setdefault(pair, [])
        if intent["type"] == "push":
            steps.append({"k": d, "t": now, "n_ok": intent["n_ok"]})
        elif steps:                                                       # rollback / release consume own pushes
            left = steps[-1]["k"] + d
            if abs(left) < 1e-9 or np.sign(left) != np.sign(steps[-1]["k"]):
                steps.pop()
            else:
                steps[-1]["k"] = left
        self.last_step_t[pair[0]] = now

    def update_version(self):
        super().update_version()
        self.gamma = 0.5 * self.GAMMA * self.scale                        # [A] hidden update: more eager


class MROv2(XApp):
    """Mobility robustness optimisation, v2 (role "MRO"). Spec: E6_V2_XAPPS_SPEC.md section 4."""

    name, cadence = "MRO", 30.0         # [A] v1 MRO cadence = mob KPM granularity (inherited)
    stack = STACK_VERSION
    TL_THR = 0.02                       # [A] too-late ratio target (v1 MRO nominal; also the SMO alarm), inherited
    TE_THR = 0.10                       # [A] too-early + ping-pong ratio (v1 MRO nominal), inherited
    MIN_TL, MIN_EV = 2, 3               # [A] v1 MRO / SMO minimum-evidence rule, inherited
    N_WIN = 2                           # [A] 2 x 30 s mob reports = v1 60 s window, inherited
    HYS_ENV = (0.0, 3.0)                # [S numeric, A mapping] TR 36.839 Tab 5.3.2.1 sets: A3 offset -1..3 dB -> Hys
    TTT_ENV = (40, 480)                 # [S] TR 36.839 Tab 5.3.2.1 sets: TTT 40..480 ms
    HYS_STEP = 0.5                      # [S range] ric.LIMITS["hys"] max step / grid
    CIO_ENV = 3.0                       # [A] MRO pair-CIO band |CIO| <= 3 dB (fine correction; half actuator range)
    LOCK_S = 120.0                      # [A] direction lock after an overshoot reversal = 2 windows
    DWELL_PER_EVENT_S = C.T310_S        # [V] T310 (TR 36.839 sim assumption 1 s): 1 s below Qout = one near-RLF

    def __init__(self, env, idx):
        super().__init__(env, idx)
        self.tl_thr, self.te_thr = self.TL_THR * self.scale, self.TE_THR * self.scale
        self.hys_step = self.HYS_STEP
        self.nrt = NRT(self.p.nc)
        self.mob = []                   # delivered mob reports (kept: last 8)
        self.mobq = {}                  # t0 -> delivered mobq report
        self.last_t = {}                # cell -> time of its last applied own change (evidence cut-off)
        self.last_act = {}              # cell -> (knob, direction) of its last cell-level change
        self.lock = {}                  # (knob, cell, direction) -> until
        self._meta = {}
        self._now = 0.0

    def observe(self, reps):
        super().observe(reps)
        self.nrt.update(reps)
        for rep in reps:
            if rep["gran"] == "mob":
                self.mob = (self.mob + [rep])[-8:]
            elif rep["gran"] == "mobq":
                self.mobq[rep["t0"]] = rep
                if len(self.mobq) > 8:
                    self.mobq.pop(min(self.mobq))

    # -------------------------------------------------------------------------------------------- helpers
    def _window(self, s):
        """Sums over the last N_WIN mob reports whose window starts at/after s's last own change (a full fresh
        window); the matching mobq dwell (missing / dropped report = 0) is added to too-late as near-RLF events."""
        cut = self.last_t.get(s, -np.inf)
        reps = [r for r in self.mob if r["t0"] >= cut]
        if len(reps) < self.N_WIN:
            return None
        reps = reps[-self.N_WIN:]
        w = {k: sum(np.asarray(r[k][s], float) for r in reps)
             for k in ("ho_att", "too_late", "too_early", "pingpong", "wrong_cell")}
        dw = sum(np.asarray(self.mobq[r["t0"]]["lowq_dwell"][s], float) for r in reps if r["t0"] in self.mobq)
        w["dwell"] = np.zeros_like(w["too_late"]) + dw
        w["late_eff"] = w["too_late"] + w["dwell"] / self.DWELL_PER_EVENT_S
        return w

    def _tidx(self, v):
        return C.TTT_SET_MS.index(int(v))

    def _pos(self, s):
        """Normalised position of Hys and TTT inside the TR 36.839 envelope (may exceed 1 outside it)."""
        h = (self.p.hys[s] - self.HYS_ENV[0]) / (self.HYS_ENV[1] - self.HYS_ENV[0])
        i0, i1 = self._tidx(self.TTT_ENV[0]), self._tidx(self.TTT_ENV[1])
        t = (self._tidx(self.p.ttt[s]) - i0) / (i1 - i0)
        return h, t

    def _cell_move(self, s, knob, d, no_hys_up=False):
        """Target value of one step of ``knob`` ('hys' | 'ttt') in direction d (-1 = HO earlier, +1 = later), or None."""
        if self.lock.get((knob, s, d), -np.inf) > self._now:
            return None
        if knob == "hys":
            if d > 0 and no_hys_up:
                return None
            v = self.p.hys[s] + d * self.hys_step
            if d < 0 and v < self.HYS_ENV[0] - 1e-9:
                return None
            if d > 0 and v > self.HYS_ENV[1] + 1e-9:
                return None
            return float(np.clip(v, C.HYS_RANGE[0], C.HYS_RANGE[1]))
        i = self._tidx(self.p.ttt[s]) + d
        if not 0 <= i < len(C.TTT_SET_MS):
            return None
        v = C.TTT_SET_MS[i]
        if d < 0 and v < self.TTT_ENV[0] or d > 0 and v > self.TTT_ENV[1]:
            return None
        return float(v)

    def _cell_req(self, s, d, now, no_hys_up=False):
        la = self.last_act.get(s)
        if la is not None and la[1] == -d:                                 # overshoot: reverse the last step
            v = self._cell_move(s, la[0], d, no_hys_up)
            if v is not None:
                self.lock[(la[0], s, la[1])] = now + self.LOCK_S
                return la[0], v
        h, t = self._pos(s)
        first = ("ttt", "hys") if (t >= h if d < 0 else t <= h) else ("hys", "ttt")
        for knob in first:
            v = self._cell_move(s, knob, d, no_hys_up)
            if v is not None:
                return knob, v
        return None

    def _pair_cio(self, s, n, d, now):
        cur = self.p.cio[s, n]
        v = float(np.clip(cur + d, C.CIO_RANGE[0], C.CIO_RANGE[1]))
        if abs(v) > self.CIO_ENV + 1e-9 and abs(v) > abs(cur):             # never push outward beyond the band
            return None
        r = self.req(("cio", s, n), v, now)
        if r is not None:
            self._meta[("cio", s, n)] = ("cio", d)
        return r

    # -------------------------------------------------------------------------------------------- decision
    def propose(self, now):
        self._now = now
        out = []
        for s in range(self.p.nc):
            w = self._window(s)
            if w is None:
                continue
            nbr = self.nrt.neighbours(s)
            tl, te, pp, wc = w["late_eff"], w["too_early"], w["pingpong"], w["wrong_cell"]
            den = w["ho_att"].sum() + w["too_late"].sum()
            if den < self.MIN_EV:
                continue
            early_ev = te + pp
            late_ratio = tl.sum() / den
            late = tl.sum() >= self.MIN_TL and late_ratio > self.tl_thr
            early = early_ev.sum() >= self.MIN_EV and early_ev.sum() / den > self.te_thr
            no_hys_up = late_ratio > self.tl_thr                          # never Hys+ on a too-late cell
            touched = set()
            if late != early:                                            # both -> no trigger-wide change
                d, ev = (-1, tl) if late else (+1, early_ev)
                pairs = [n for n in nbr if ev[n] > 0]
                others = ev.sum() - sum(ev[n] for n in nbr)
                r = None
                if len(pairs) == 1 and others <= 0:                      # single-pair issue -> that pair's CIO
                    r = self._pair_cio(s, pairs[0], -d, now)
                    touched.add(pairs[0])
                if r is None:
                    mv = self._cell_req(s, d, now, no_hys_up)
                    if mv is not None:
                        r = self.req((mv[0], s), mv[1], now)
                        if r is not None:
                            self._meta[(mv[0], s)] = (mv[0], d)
                if r is not None:
                    out.append(r)
            for n in nbr:                                                # wrong cell: HO s -> n then RLF elsewhere
                if n in touched or wc[n] < self.MIN_TL:
                    continue
                if wc[n] / max(w["ho_att"][n], 1.0) > self.tl_thr:
                    out.append(self._pair_cio(s, n, -1.0, now))
        return [r for r in out if r]

    def result(self, r, accepted, applied, now):
        super().result(r, accepted, applied, now)
        k = r["knob"]
        intent = self._meta.pop(k, None)
        if intent is None or not accepted or abs(float(applied) - float(r["cur"])) < 1e-9:
            return
        s = k[1]
        self.last_t[s] = now                                             # any own change on s: a full fresh window
        if intent[0] in ("hys", "ttt"):
            self.last_act[s] = intent

    def update_version(self):
        super().update_version()
        self.hys_step = 1.0                                              # [A] hidden update: coarser Hys step


# ---------------------------------------------------------------------------------------------------- registration
# Same positions as v1 M4 (MRO idx 0, TS idx 1, ES idx 2, SLICE idx 3) so the per-seed implementation-variant draws
# (xapps.XApp.__init__: rng(seed, "xapp", 1000 + idx)) are identical for the reused ES and SLICE xApps.
MIXES_V2 = {"V2_none": (), "V2_MRO": (MROv2,), "V2_TS": (MLB,), "V2_M_TS_MRO": (MROv2, MLB),
            "V2_M2": (MROv2, MLB, ES), "V2_M4": (MROv2, MLB, ES, SliceSLA)}


def register_v2():
    """Add the V2_* mixes to ``xapps.MIXES`` in place (v1 keys untouched; idempotent). Returns the added names."""
    clash = {k for k in MIXES_V2 if k in MIXES and MIXES[k] != MIXES_V2[k]}
    if clash:
        raise RuntimeError(f"mix names already registered differently: {sorted(clash)}")
    MIXES.update(MIXES_V2)
    return sorted(MIXES_V2)


def all_pair_cio_knobs(nc):
    """Knob registry for the v2 stack: CIO on EVERY ordered cell pair (the RIC can actuate any pair; the xApps'
    run-time NRT decides which pairs they use)."""
    return [("cio", s, n) for s in range(nc) for n in range(nc) if s != n]


def make_env_v2(cfg, trace=False, **kw):
    """Build an ``E6Env`` running a V2_* mix: registers the mixes, swaps in ``KPMV2`` before the first second, replaces
    the neighbour-list CIO knobs by all ordered cell pairs, and adds the LL-ratio knobs when SLICE is deployed (v1 does
    this only for mix "M4"). Equivalent permanent env.py hooks: spec section 2.3."""
    from .env import E6Env

    if cfg.mix not in MIXES_V2:
        raise ValueError(f"make_env_v2 needs a V2_* mix, got {cfg.mix!r}")
    register_v2()
    env = E6Env(cfg, trace=False, **kw)
    env.kpm = KPMV2(cfg, env.plant)
    nc = env.plant.nc
    rest = [k for k in env.knobs if k[0] != "cio"]
    if SliceSLA in MIXES_V2[cfg.mix]:
        rest += [("ll_ratio", c) for c in range(nc) if ("ll_ratio", c) not in rest]
    env.knobs = all_pair_cio_knobs(nc) + rest
    if trace:
        from cdd_oran.decision.trace import TraceRecorder

        env._tr = TraceRecorder(env, kw.get("trace_snapshot_s", 60))
    return env


__all__ = ["KPMV2", "MIXES_V2", "MLB", "MROv2", "NRT", "STACK_VERSION", "all_pair_cio_knobs", "border_snapshot",
           "lowq_snapshot", "make_env_v2", "register_v2"]

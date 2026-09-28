"""E6 plant: UEs, traffic, mobility, radio, handover/RLF, slice-aware scheduler, queues, energy, KPM counters.

One coherent clock: every tick (TICK_S = 40 ms) executes, in this order:
  1. hidden processes: per-site log-OU load m(t), mobility, traffic arrivals (uses the tick's exogenous tape);
  2. radio: large-scale gains -> measured RSRP (+ error) -> L3 filter; SINR from the PREVIOUS tick's physical PRB
     band occupancy of every other cell (load-coupled interference); cell sleep removes a cell, macro carrier
     shutdown removes capacity (not coverage);
  3. mobility management: RLF timers (T310) on EWMA SINR, A3 entering condition + TTT, HO execution/failure, MRO event
     classification (too-late / too-early / wrong-cell / ping-pong);
  4. scheduler: LL dedicated pool -> LL in shared pool -> eMBB/BE equal share (water-filled once); UEs in HO
     interruption or RLF outage are not served;
  5. queues, SLA bookkeeping (per UE-second), energy, KPM counters.
Configuration changes applied by the RIC take effect at the start of the next tick.
Exogenous randomness is keyed by (seed, stream, tick) so arms with different actions see the same tape.
Stress scenarios (cfg.scenario = "surge" | "mistune", see docs/benchmark/E6_STRESS_SCENARIOS.md) are drawn once from
the (seed, "scenario") stream and are otherwise deterministic in time; "base" builds no scenario object at all.

E6-P extension (cfg.e6p, config.E6PConfig; default OFF, every branch below is guarded so E6-scn-v1 is bit-identical):
  * Tx power: ``ptx_off`` (dB, per cell) is added to the cell's gain column in ``_gains`` -> serving RSRP, SINR, the
    interference the cell causes and the measured L3 (A3 handover) all follow. EARTH power uses
    P0 + DP * rho * PMAX * 10^(ptx_off/10).
  * Protected slice: ``prot`` flags a subset of eMBB UEs (own RNG stream "e6p"; they stay eMBB for traffic, for the
    eMBB SLA and for every per-slice KPM). Pool order in the scheduler, per cell:
      1. LL DEDICATED pool  = cap * ll_ratio: LL-only, NOT work-conserving (idle reserved PRBs are wasted), unchanged;
      2. protected MIN share = min(cap * prot_min, cap - dedicated): protected UEs are served from it first (equal
         share); whatever they do not use goes back to step 3 (work-conserving, TS 28.541 rRMPolicyMinRatio);
      3. SHARED pool = the rest: LL remainder, eMBB (protected residual demand included) and BE, equal share.
    Per-UE floor: a protected UE-second violates when backlogged >= prot_min_backlog_s with throughput below
    prot_floor_bps, or in outage; accumulated in sla["prot_viol"] / sla["prot_ue_s"] (present only when enabled).
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .geometry import GainMaps, Layout

STREAMS = {"layout": 1, "ue": 2, "mob": 3, "traffic": 4, "meas": 5, "load": 6, "kpm": 7, "xapp": 8, "scenario": 9,
           "e6p": 10}
LL, EMBB, BE = 0, 1, 2
A_L3 = 0.5 ** (C.L3_K / 4.0)
NOISE_W = 10 ** ((C.NOISE_DBM_HZ + 10 * np.log10(C.BW_HZ) + C.UE_NF_DB) / 10) / 1000.0


RESEED_TAG = 6620        # E6-P lookahead re-draw key (env.copy(reseed=...)); disjoint from 6600 and registry tags


def _rng(seed, stream, t=0, salt=None):
    if salt is None:
        return np.random.default_rng([int(seed), 6600, STREAMS[stream], int(t)])
    return np.random.default_rng([int(seed), 6600, STREAMS[stream], int(t), RESEED_TAG, int(salt)])


class Plant:
    def __init__(self, cfg: C.E6Config):
        self.cfg = cfg
        s = cfg.seed
        r = _rng(s, "layout")
        self.lay = Layout(cfg, r)
        self.gm = GainMaps(self.lay, cfg, r)
        nc = self.lay.n_cells
        self.nc = nc
        self.site_of = np.minimum(self.lay.cell_site, 6)            # picos -> site 0..6 of nearest macro (for load)
        # ---------------------------------------------------------------- UEs
        ru = _rng(s, "ue")
        n = cfg.ue_count()
        self.n = n
        hot = ru.uniform(size=n) < 0.35                               # [A] 35 % of UEs dropped around hotspots
        pos = ru.uniform(-900, 900, (n, 2))
        k = ru.integers(len(self.lay.hotspots), size=n)
        pos[hot] = self.lay.hotspots[k[hot]] + ru.normal(0, cfg.hotspot_sigma_m, (hot.sum(), 2))
        self.pos = self.lay.wrap(pos)
        self.indoor = ru.uniform(size=n) < 0.7                       # [A] 70 % indoor static
        mobile = ~self.indoor
        vfrac = 0.1 if cfg.mobility == "ped" else 0.4
        veh = mobile & (ru.uniform(size=n) < vfrac)
        self.speed = np.where(veh, ru.uniform(30, 60, n) / 3.6, np.where(mobile, 3.0 / 3.6, 0.0))
        self.heading = ru.uniform(0, 2 * np.pi, n)
        self.mobile_idx = np.nonzero(self.speed > 0)[0]
        u = ru.uniform(size=n)
        self.sl = np.where(u < cfg.frac_ll, LL, np.where(u < cfg.frac_ll + cfg.frac_embb, EMBB, BE))
        self.ll_pps = ru.uniform(*C.LL_RATE_PPS, n) * cfg.lf()
        # stress scenario (S1 surge / S2 mistune); None for the calm base plant (bit-identical to pre-scenario E6)
        self.oam_hook = None          # set by the RIC layer: called (knob, value, now) on every OAM/SMO write
        self.scn = None if cfg.scenario == "base" else make_scenario(cfg, self)
        self.rng_salt = None          # None = the episode tape; an int re-draws all FUTURE exogenous draws (reseed)
        # E6-P (default off; nothing below consumes an existing RNG stream)
        P = cfg.e6p
        self.P, self.p_on, self.p_ptx, self.p_prot = P, P.enabled, bool(P.ptx_on), bool(P.prot_on)
        if self.p_on:
            validate_e6p(P)
        self.ptx_cells = self.lay.is_macro.copy() if P.ptx_scope == "macro" else np.ones(nc, bool)
        self.ptx_off = np.where(self.ptx_cells & self.p_ptx, float(P.ptx_init_db), 0.0)   # dB offset per cell
        self.prot = np.zeros(n, bool)
        if self.p_prot:
            self.prot = (self.sl == EMBB) & (_rng(s, "e6p", 1).uniform(size=n) < P.prot_frac)
        self.prot_min = np.full(nc, float(P.prot_min_init) if self.p_prot else 0.0)   # min PRB share per cell
        # ---------------------------------------------------------------- configuration (RIC-controlled knobs)
        self.cio = np.zeros((nc, nc))                                 # CIO[s, n] dB (A3 offset for s -> n)
        self.hys = np.full(nc, 2.0)
        self.ttt = np.full(nc, 320)                                   # ms
        self.ll_ratio = np.zeros(nc)                                  # LL dedicated PRB fraction
        self.asleep = np.zeros(nc, bool)
        self.waking_until = np.full(nc, -1.0)
        self.n_trx = np.where(self.lay.is_macro, C.MACRO_NTRX, 1)            # component carriers per cell
        self.n_car = self.n_trx.copy()                                # active carriers (ES knob on macros)
        self.car_on_at = np.full(nc, -1.0)                            # a reactivated carrier serves from here
        # ---------------------------------------------------------------- UE radio / mobility state
        self.t = 0
        g = self._gains()
        self.l3 = g.copy()
        self.serv = np.argmax(g, 1)
        self.sinr_ewma = np.zeros(n)
        self.ttt_cand = np.full(n, -1)
        self.ttt_acc = np.zeros(n)
        self.int_until = np.zeros(n)                                  # HO interruption / RLF outage end (s)
        self.t310 = np.full(n, -1.0)
        self.last_ho_t = np.full(n, -1e9)
        self.last_ho_src = np.full(n, -1)
        self.rho = np.full(nc, 0.3)                                   # PRB utilisation of ACTIVE capacity (prev tick)
        self.occ = np.full(nc, 0.3)                                   # occupied share of the full band (interference)
        # ---------------------------------------------------------------- traffic
        self.m = np.ones(7)                                           # hidden per-site load multiplier
        self.q = np.zeros(n)                                          # backlog bits
        self.ll_arr = np.zeros((n, 64))                               # LL cumulative-arrival ring (bits)
        self.ll_cum_a = np.zeros(n)
        self.ll_cum_s = np.zeros(n)
        # ---------------------------------------------------------------- per-second SLA bookkeeping
        self.sec_backlog_t = np.zeros(n)
        self.sec_bits = np.zeros(n)
        self.sec_maxdelay = np.zeros(n)
        self.sec_outage = np.zeros(n, bool)
        self.sla = {"viol_ue_s": 0.0, "ue_s": 0.0, "ll_viol": 0.0, "embb_viol": 0.0, "outage_viol": 0.0,
                    "severe": 0, "energy_j": 0.0, "rlf": 0, "ho": 0, "pingpong": 0}
        if self.p_on:
            self.sla["lowsinr_ue_s"] = 0.0                               # UE-s with serving SINR < Q_in (-6 dB)
        if self.p_prot:
            self.sla.update({"prot_viol": 0.0, "prot_ue_s": 0.0})
        self.cell_viol_run = np.zeros((nc, 3))
        self.ctr = self._new_counters()

    # ================================================================================================ helpers
    def _gains(self):
        g = self.gm.lookup(self.pos).astype(float)
        if self.p_ptx:
            g += self.ptx_off[None, :]
        g[self.indoor] -= C.O2I_DB
        g[:, self.asleep | (self.waking_until > self.t * C.TICK_S)] = -300.0
        return g

    def _tape(self, stream, t):
        """Per-tick exogenous generator: the episode tape, or its re-drawn future after ``env.copy(reseed=k)``."""
        if self.rng_salt is None:
            return _rng(self.cfg.seed, stream, t)
        return _rng(self.cfg.seed, stream, t, self.rng_salt)

    def _new_counters(self):
        nc = self.nc
        c = {"prb_used": np.zeros((nc, 3)), "prb_rsv": np.zeros(nc), "prb_cap": np.zeros(nc), "ticks": 0, "bits": np.zeros((nc, 3)),
             "ho_att": np.zeros((nc, nc)), "ho_succ": np.zeros((nc, nc)), "too_late": np.zeros((nc, nc)),
             "too_early": np.zeros((nc, nc)), "wrong_cell": np.zeros((nc, nc)), "pingpong": np.zeros((nc, nc)),
             "rlf": np.zeros(nc), "energy_j": np.zeros(nc), "ll_delay": [[] for _ in range(nc)],
             "embb_thp": [[] for _ in range(nc)], "act_ue": np.zeros((nc, 3))}
        if self.p_on:                              # E6-P counters (only when enabled)
            c.update({"sinr_serv": [], "sinr_val": []})
            if self.p_prot:
                c.update({"prot_dem": np.zeros(nc), "prot_used": np.zeros(nc), "prot_min_prb": np.zeros(nc),
                          "prot_bits": np.zeros(nc), "prot_act": np.zeros(nc), "prot_eval": np.zeros(nc),
                          "prot_below": np.zeros(nc)})
        return c

    def take_counters(self):
        c, self.ctr = self.ctr, self._new_counters()
        return c

    @staticmethod
    def se(sinr_db):
        s = C.SE_ALPHA * np.log2(1 + 10 ** (sinr_db / 10))
        return np.where(sinr_db < C.SINR_MIN_DB, 0.0, np.minimum(s, C.SE_MAX))

    # ================================================================================================ one tick
    def tick(self):
        cfg, dt, t = self.cfg, C.TICK_S, self.t
        now = t * dt
        n, nc = self.n, self.nc
        # ---- 1. hidden processes
        rl = self._tape("load", t)
        a = np.exp(-dt / cfg.m_tau_s)
        logm = np.log(self.m) * a + cfg.m_sigma * np.sqrt(1 - a * a) * rl.normal(size=7)
        self.m = np.exp(logm)
        if self.scn is not None:
            self.scn.on_tick(self, now)
        rm = self._tape("mob", t)
        turn = rm.uniform(size=n) < dt / 20.0                          # [A] new heading every ~20 s
        self.heading = np.where(turn, rm.uniform(0, 2 * np.pi, n), self.heading)
        mv = self.mobile_idx
        if len(mv):
            step = self.speed[mv] * dt
            self.pos[mv] = self.lay.wrap_near(self.pos[mv] + step[:, None] * np.c_[np.cos(self.heading[mv]),
                                                                                    np.sin(self.heading[mv])])
        if self.scn is not None:
            self.scn.move(self, dt)
        rt = self._tape("traffic", t)
        site = self.site_of[self.serv]
        total = cfg.warmup_s + cfg.scored_s
        ramp = cfg.ramp[0] + (cfg.ramp[1] - cfg.ramp[0]) * min(now / max(total, 1e-9), 1.0)   # compressed diurnal ramp
        mult = self.m[site] * cfg.lf() * ramp
        ll = self.sl == LL
        arr = np.zeros(n)
        arr[ll] = rt.poisson(self.ll_pps[ll] * dt * ramp) * C.LL_PKT_BYTES * 8
        rate = np.where(self.sl == EMBB, cfg.embb_files_per_s, cfg.be_files_per_s) * mult
        if self.scn is not None:
            rate = rate * self.scn.traffic_mult(now, self.pos)
        files = rt.poisson(np.where(ll, 0.0, rate * dt))
        arr[~ll] = files[~ll] * C.EMBB_FILE_BYTES * 8
        self.q += arr
        self.ll_cum_a += np.where(ll, arr, 0.0)
        self.ll_arr[:, t % 64] = self.ll_cum_a
        # ---- 2. radio
        g = self._gains()
        rq = self._tape("meas", t)
        meas = g + rq.normal(0, C.MEAS_ERR_DB, g.shape)
        self.l3 = (1 - A_L3) * self.l3 + A_L3 * meas
        pw = 10 ** (g / 10) / 1000.0
        load_w = pw * self.occ[None, :]
        tot_int = load_w.sum(1)
        idx = np.arange(n)
        s_w = pw[idx, self.serv]
        sinr = 10 * np.log10(s_w / (tot_int - load_w[idx, self.serv] + NOISE_W) + 1e-30)
        self.sinr_ewma = 0.8 * self.sinr_ewma + 0.2 * sinr               # ~200 ms averaging
        # ---- 3. mobility management
        in_out = self.int_until > now
        # RLF (T310)
        bad = (self.sinr_ewma < C.QOUT_DB) & ~in_out
        self.t310 = np.where(bad & (self.t310 < 0), now, self.t310)
        self.t310 = np.where(self.sinr_ewma > C.QIN_DB, -1.0, self.t310)
        rlf = (self.t310 >= 0) & (now - self.t310 >= C.T310_S) & ~in_out
        for u in np.nonzero(rlf)[0]:
            self._rlf(u, now)
        # A3 + TTT
        cand_val = self.l3 + self.cio[self.serv] - self.hys[self.serv][:, None]
        cand_val[idx, self.serv] = -1e9
        cand = np.argmax(cand_val, 1)
        enter = (cand_val[idx, cand] > self.l3[idx, self.serv]) & ~in_out
        same = enter & (cand == self.ttt_cand)
        self.ttt_acc = np.where(same, self.ttt_acc + dt * 1000, np.where(enter, dt * 1000, 0.0))
        self.ttt_cand = np.where(enter, cand, -1)
        fire = enter & (self.ttt_acc >= self.ttt[self.serv])
        for u in np.nonzero(fire)[0]:
            self._handover(u, int(cand[u]), now, pw)
        # ---- 4. scheduler
        in_out = self.int_until > now
        prb_rate = self.se(sinr) * C.PRB_HZ * (1 - C.OVERHEAD)          # bits/s per PRB
        car = np.where(self.car_on_at > now, self.n_car - 1, self.n_car)   # carrier still reactivating
        cap_frac = car / self.n_trx
        cap_prb = np.where(self.asleep | (self.waking_until > now), 0.0, float(C.N_PRB) * cap_frac)
        demand = np.where(in_out | (prb_rate <= 0), 0.0, self.q / np.maximum(prb_rate * dt, 1e-9))
        alloc = np.zeros(n)
        cell = self.serv
        ded = cap_prb * self.ll_ratio
        d_ll = np.where(ll, demand, 0.0)
        tot_ll = np.bincount(cell, d_ll, nc)
        frac_ded = np.where(tot_ll > 0, np.minimum(1.0, ded / np.maximum(tot_ll, 1e-9)), 0.0)
        alloc_ded = d_ll * frac_ded[cell]
        used_ded = np.bincount(cell, alloc_ded, nc)
        left = cap_prb - ded
        alloc = alloc_ded.copy()
        d_o = np.where(ll, d_ll - alloc_ded, demand)          # LL remainder shares the pool equally with eMBB/BE
        if self.p_prot:                                       # E6-P protected min share (work-conserving)
            m_share = np.minimum(cap_prb * self.prot_min, np.maximum(left, 0.0))
            d_p = np.where(self.prot, d_o, 0.0)
            left_m = m_share.copy()
            for _ in range(2):                                            # equal share inside the min share
                nbp = np.bincount(cell, (d_p > 1e-9).astype(float), nc)
                sh = np.where(nbp > 0, left_m / np.maximum(nbp, 1), 0.0)
                give = np.minimum(d_p, sh[cell])
                alloc += give
                d_p = d_p - give
                d_o = d_o - give
                left_m = left_m - np.bincount(cell, give, nc)
            left = left - (m_share - left_m)                              # unused min share -> shared pool
        for _ in range(2):                                                # equal share, water-filled twice
            nb = np.bincount(cell, (d_o > 1e-9).astype(float), nc)
            share = np.where(nb > 0, left / np.maximum(nb, 1), 0.0)
            give = np.minimum(d_o, share[cell])
            alloc += give
            d_o = d_o - give
            left = left - np.bincount(cell, give, nc)
        served = np.minimum(self.q, alloc * prb_rate * dt)
        self.q -= served
        used = np.bincount(cell, alloc, nc)
        self.rho = np.where(cap_prb > 0, np.minimum(1.0, used / np.maximum(cap_prb, 1)), 0.0)
        self.occ = self.rho * cap_prb / C.N_PRB
        # ---- 5. queues / SLA bookkeeping / energy / counters
        self.ll_cum_s += np.where(ll, served, 0.0)
        hol = np.zeros(n)
        if ll.any():
            ring = self.ll_arr[ll]
            order = (t - np.arange(64)) % 64                             # newest first
            hist = ring[:, order]                                        # cumulative arrivals at t, t-1, ...
            ok = hist > self.ll_cum_s[ll][:, None] + 1e-6                # arrived-but-unserved as of that tick
            depth = ok.sum(1)                                            # ticks since the oldest unserved arrival
            hol[ll] = np.where(self.q[ll] > 1e-6, depth * dt + C.HOL_PROC_S, 0.0)
        self.sec_maxdelay = np.maximum(self.sec_maxdelay, hol)
        backlogged = (self.q > 1e-6) | (served > 0)
        self.sec_backlog_t += np.where(backlogged & ~ll, dt, 0.0)
        self.sec_bits += served
        self.sec_outage |= in_out & (self.int_until - now > C.HO_EXEC_S + 1e-9)   # RLF outage (not plain HO)
        # LL: a plain HO interruption is NOT an outage; its queued packets already show up as delay (no double count)
        self.sec_outage |= prb_rate <= 0                                 # out of coverage (SINR < -10 dB)
        ctr = self.ctr
        ctr["ticks"] += 1
        for sidx in (LL, EMBB, BE):
            m_ = self.sl == sidx
            ctr["prb_used"][:, sidx] += np.bincount(cell[m_], alloc[m_], nc)
            ctr["bits"][:, sidx] += np.bincount(cell[m_], served[m_], nc)
            ctr["act_ue"][:, sidx] += np.bincount(cell[m_], backlogged[m_].astype(float), nc)
        ctr["prb_cap"] += cap_prb
        ctr["prb_rsv"] += ded - used_ded                                 # reserved but idle dedicated PRBs
        if self.p_prot:
            pm = self.prot
            capd = np.maximum(cap_prb, 1e-9)
            ctr["prot_dem"] += np.where(cap_prb > 0, np.minimum(1.0, np.bincount(cell[pm], demand[pm], nc) / capd), 0.0)
            ctr["prot_used"] += np.bincount(cell[pm], alloc[pm], nc)
            ctr["prot_min_prb"] += m_share
            ctr["prot_bits"] += np.bincount(cell[pm], served[pm], nc)
            ctr["prot_act"] += np.bincount(cell[pm], backlogged[pm].astype(float), nc)
        if self.p_ptx:                                                   # E6-P: EARTH P_out scales with Tx power
            lin = 10 ** (self.ptx_off / 10)
            p_mac = car * (C.MACRO_P0_W + C.MACRO_DP * self.rho * C.MACRO_PMAX_W * lin) + \
                (C.MACRO_NTRX - car) * C.MACRO_SLEEP_W / C.MACRO_NTRX
            p_pic = C.PICO_P0_W + C.PICO_DP * self.rho * C.PICO_PMAX_W * lin
        else:
            p_mac = car * (C.MACRO_P0_W + C.MACRO_DP * self.rho * C.MACRO_PMAX_W) + \
                (C.MACRO_NTRX - car) * C.MACRO_SLEEP_W / C.MACRO_NTRX
            p_pic = C.PICO_P0_W + C.PICO_DP * self.rho * C.PICO_PMAX_W
        pw_cell = np.where(self.lay.is_macro, p_mac, np.where(self.asleep, C.PICO_SLEEP_W, p_pic))
        pw_cell = np.where(~self.lay.is_macro & (self.waking_until > now), C.PICO_P0_W, pw_cell)
        ctr["energy_j"] += pw_cell * dt
        if now >= cfg.warmup_s:
            self.sla["energy_j"] += float(pw_cell.sum() * dt)
        self.t += 1
        if self.t % C.TICKS_PER_CONTROL == 0:
            self._close_second()

    # ================================================================================================ events
    def _handover(self, u, tgt, now, pw):
        src = int(self.serv[u])
        self.ctr["ho_att"][src, tgt] += 1
        # target SINR at execution (current loads)
        s = pw[u, tgt]
        i = (pw[u] * self.occ).sum() - pw[u, tgt] * self.occ[tgt]
        sinr_t = 10 * np.log10(s / (i + NOISE_W) + 1e-30)
        if sinr_t < C.HO_FAIL_SINR_DB:                                   # HO failure -> RLF in target
            self.serv[u] = tgt
            self.last_ho_t[u], self.last_ho_src[u] = now, src
            self._rlf(u, now, ho_fail=True)
            return
        self.ctr["ho_succ"][src, tgt] += 1
        self.sla["ho"] += 1
        if self.last_ho_src[u] == tgt and now - self.last_ho_t[u] <= C.PINGPONG_S:
            self.ctr["pingpong"][tgt, src] += 1                          # attributed to the first HO tgt->src
            self.sla["pingpong"] += 1
        self.last_ho_t[u], self.last_ho_src[u] = now, src
        self.serv[u] = tgt
        self.int_until[u] = now + C.HO_EXEC_S
        self.ttt_cand[u], self.ttt_acc[u] = -1, 0.0
        self.sinr_ewma[u] = sinr_t

    def _rlf(self, u, now, ho_fail=False):
        cur = int(self.serv[u])
        best = int(np.argmax(self.l3[u]))
        self.ctr["rlf"][cur] += 1
        self.sla["rlf"] += 1
        recent = now - self.last_ho_t[u] <= C.TOO_EARLY_S or ho_fail
        if recent and self.last_ho_src[u] >= 0:
            src = int(self.last_ho_src[u])
            if best == src:
                self.ctr["too_early"][src, cur] += 1
            elif best != cur:
                self.ctr["wrong_cell"][src, cur] += 1
        elif best != cur:
            self.ctr["too_late"][cur, best] += 1
        self.serv[u] = best
        self.int_until[u] = now + C.RLF_OUTAGE_S
        self.t310[u] = -1.0
        self.ttt_cand[u], self.ttt_acc[u] = -1, 0.0
        self.sinr_ewma[u] = 0.0

    def _close_second(self):
        """Per UE-second SLA scoring (all UEs are the denominator) + per-cell KPM delay/thp samples."""
        sl = self.sl
        ll_v = (sl == LL) & ((self.sec_maxdelay > C.LL_DELAY_TARGET_S) | self.sec_outage)
        thp = np.where(self.sec_backlog_t > 0, self.sec_bits / np.maximum(self.sec_backlog_t, 1e-9), np.inf)
        em_v = (sl == EMBB) & (((self.sec_backlog_t >= 0.2) & (thp < C.EMBB_THP_TARGET_BPS)) | self.sec_outage)
        be_v = (sl == BE) & self.sec_outage
        v = ll_v | em_v | be_v
        scored = self.t * C.TICK_S > self.cfg.warmup_s
        if scored:
            S = self.sla
            S["viol_ue_s"] += v.sum()
            S["ue_s"] += self.n
            S["ll_viol"] += ll_v.sum()
            S["embb_viol"] += em_v.sum()
            S["outage_viol"] += self.sec_outage.sum()
            # severe incident: a cell-slice with > 20 % of its UEs violated for >= 10 consecutive seconds
            for s_ in (LL, EMBB, BE):
                m_ = sl == s_
                tot = np.bincount(self.serv[m_], minlength=self.nc)
                bad = np.bincount(self.serv[m_ & v], minlength=self.nc)
                frac = np.where(tot > 0, bad / np.maximum(tot, 1), 0.0)
                run = np.where(frac > 0.2, self.cell_viol_run[:, s_] + 1, 0)
                S["severe"] += int(((run == 10)).sum())
                self.cell_viol_run[:, s_] = run
        # last closed second's per-UE violation flags (read by the optional trace recorder; privileged labels)
        self.sec_viol_ll, self.sec_viol_embb, self.sec_viol_be = ll_v, em_v, be_v
        self.sec_out, self.sec_scored = self.sec_outage.copy(), scored
        cell = self.serv
        if self.p_on:
            self._close_second_p(thp, scored)
        for u in np.nonzero(sl == LL)[0]:                     # every LL UE-second is a sample (zeros included)
            if not self.sec_outage[u]:
                self.ctr["ll_delay"][cell[u]].append(self.sec_maxdelay[u])
        for u in np.nonzero((sl == EMBB) & (self.sec_backlog_t >= 0.2))[0]:
            self.ctr["embb_thp"][cell[u]].append(min(thp[u], 1e9))
        self.sec_backlog_t[:] = 0
        self.sec_bits[:] = 0
        self.sec_maxdelay[:] = 0
        self.sec_outage[:] = False

    def sleep_handover(self, c, now):
        """E6-P graceful pico sleep (spec 6.2; called by the RIC right after ``asleep[c]`` is set, only when enabled):
        every UE served by c is handed over to its best remaining cell by measured L3 RSRP with the normal HO
        execution (``_handover``: HO_EXEC_S interruption; a target below HO_FAIL_SINR_DB still fails -> RLF). UEs already
        in HO interruption / RLF outage are only re-pointed (their outage is not shortened)."""
        g = self._gains()                                              # c is already masked out (-300 dB)
        pw = 10 ** (g / 10) / 1000.0
        off = self.asleep | (self.waking_until > now)
        for u in np.nonzero(self.serv == c)[0]:
            l3 = np.where(off, -np.inf, self.l3[u])
            tgt = int(np.argmax(l3))
            if self.int_until[u] > now:
                self.serv[u] = tgt
                self.ttt_cand[u], self.ttt_acc[u] = -1, 0.0
            else:
                self._handover(u, tgt, now, pw)

    def _close_second_p(self, thp, scored):
        """E6-P per-second bookkeeping: serving-SINR samples (edge-SINR KPM) and the protected per-UE floor."""
        ctr, nc, cell = self.ctr, self.nc, self.serv
        ok = ~self.sec_outage
        ctr["sinr_serv"].append(cell[ok].copy())
        ctr["sinr_val"].append(self.sinr_ewma[ok].copy())
        if scored:
            self.sla["lowsinr_ue_s"] += float((self.sinr_ewma < C.QIN_DB).sum())
        if not self.p_prot:
            return
        P, pm = self.P, self.prot
        act = pm & (self.sec_backlog_t >= P.prot_min_backlog_s)
        pv = pm & ((act & (thp < P.prot_floor_bps)) | self.sec_outage)
        self.sec_viol_prot = pv
        if scored:
            self.sla["prot_viol"] += float(pv.sum())
            self.sla["prot_ue_s"] += float(pm.sum())
        ctr["prot_eval"] += np.bincount(cell[act | pv], minlength=nc)
        ctr["prot_below"] += np.bincount(cell[pv], minlength=nc)


# ==================================================================================================== stress scenarios
# Mechanics of the pre-declared stress scenarios (docs/benchmark/E6_STRESS_SCENARIOS.md). Everything is drawn once from
# the (seed, "scenario", k) stream at construction and is otherwise a deterministic function of time, so every arbiter
# arm sees the identical scenario tape. Scenarios never read the RIC-controlled configuration.
def wrap_dist(lay, pos, centre):
    """Wrap-around (minimum-image) distance of each point in ``pos`` (n, 2) to ``centre`` (2,)."""
    c = np.asarray(centre)[None, :] + lay._cands()                                    # (25, 2) images
    return np.linalg.norm(np.atleast_2d(pos)[:, None, :] - c[None], axis=-1).min(1)


class _Scenario:
    def __init__(self, cfg: C.E6Config, plant: Plant):
        self.cfg, self.lay = cfg, plant.lay

    def at(self, frac):
        """Absolute episode time (s) of a fraction of the scored window."""
        return self.cfg.warmup_s + frac * self.cfg.scored_s

    def on_tick(self, plant, now):
        pass

    def move(self, plant, dt):
        pass

    def traffic_mult(self, now, pos):
        return 1.0

    def _dist(self, pos, centre):
        return wrap_dist(self.lay, pos, centre)

    def _anchor(self, r, mode):
        """Scenario anchor point: 'band' 150-250 m from a random macro site (the E6 hotspot rule), 'edge' midway
        between two adjacent macro sites, 'vertex' a 3-site corner (ISD / sqrt 3 from a site)."""
        d = C.ISD_M
        site = self.lay.sites[int(r.integers(7))]
        if mode == "band":
            rad, ang = r.uniform(150, 250), r.uniform(0, 2 * np.pi)
        elif mode == "edge":
            rad, ang = d / 2.0, np.pi / 3.0 * int(r.integers(6))
        elif mode == "vertex":
            rad, ang = d / np.sqrt(3.0), np.pi / 6.0 + np.pi / 3.0 * int(r.integers(6))
        else:
            raise ValueError(f"unknown scenario location {mode!r}")
        return self.lay.wrap(site + rad * np.array([np.cos(ang), np.sin(ang)]))[0]


class SurgeScenario(_Scenario):
    """S1: event / moving-hotspot surge. eMBB/BE offered traffic of every UE inside a disk is multiplied by
    1 + (surge_mult - 1) * envelope(t); envelope = trapezoid (onset, ramp up, hold, ramp down)."""

    def __init__(self, cfg, plant):
        super().__init__(cfg, plant)
        r = _rng(cfg.seed, "scenario", 1)
        self.c0 = self._anchor(r, cfg.surge_loc)
        a = r.uniform(0, 2 * np.pi)                                                     # drift bearing (if moving)
        self.vel = cfg.surge_speed_mps * np.array([np.cos(a), np.sin(a)])
        self.t0 = self.at(cfg.surge_onset_frac)
        self.ramp = cfg.surge_ramp_frac * cfg.scored_s
        self.t_end = self.t0 + 2 * self.ramp + cfg.surge_hold_frac * cfg.scored_s

    def envelope(self, now):
        if now < self.t0 or now > self.t_end:
            return 0.0
        if self.ramp <= 0:
            return 1.0
        return float(min(1.0, (now - self.t0) / self.ramp, (self.t_end - now) / self.ramp))

    def centre(self, now):
        el = min(max(now, self.t0), self.t_end) - self.t0
        return self.lay.wrap(self.c0 + self.vel * el)[0]

    def inside(self, now, pos):
        return self._dist(pos, self.centre(now)) <= self.cfg.surge_radius_m

    def traffic_mult(self, now, pos):
        e = self.envelope(now)
        if e <= 0.0:
            return 1.0
        return 1.0 + (self.cfg.surge_mult - 1.0) * e * self.inside(now, pos)


class MistuneScenario(_Scenario):
    """S2 (too-late-HO hypothesis): high-speed road corridor (present from t = 0) crossing a cluster whose Hys/TTT are
    overwritten by an OAM parameter rollout at mis_onset_frac. Corridor UEs drive the section at corr_speed_kmh with a
    U-turn at each end (recurrent-road stress [A]); the mis-set cluster = every cell that is the strongest (outdoor,
    all cells on) anywhere on the section [A].

    OAM writes (the rollout, and an SMO restore via ``oam_set``) bypass the RIC: no request, no NACK, no churn, no
    actuator dwell. Each written knob is reported to ``plant.oam_hook`` (the env records it as the knob's
    last-known-good), so a RIC rollback cannot undo an OAM write and the next RIC change records it as its prior."""

    def __init__(self, cfg, plant):
        super().__init__(cfg, plant)
        r = _rng(cfg.seed, "scenario", 2)
        self.centre = self._anchor(r, cfg.corr_loc)
        a = r.uniform(0, np.pi)
        self.u = np.array([np.cos(a), np.sin(a)])
        self.half = cfg.corr_len_m / 2.0
        n = plant.n
        k = int(round(cfg.corr_frac_ue * n))
        self.idx = np.sort(r.choice(n, k, replace=False))
        self.s = r.uniform(-self.half, self.half, k)                                    # position along the road
        self.dir = np.where(r.uniform(size=k) < 0.5, 1.0, -1.0)
        self.v = cfg.corr_speed_kmh / 3.6
        plant.indoor[self.idx] = False
        plant.speed[self.idx] = self.v
        self._place(plant)
        plant.mobile_idx = np.setdiff1d(plant.mobile_idx, self.idx)                     # moved here, not by the walk
        pts = self.lay.wrap(self.centre + np.arange(-self.half, self.half + 1e-9, C.GRID_M)[:, None] * self.u)
        self.cluster = np.unique(np.argmax(plant.gm.lookup(pts), 1))
        self.t_mis = self.at(cfg.mis_onset_frac)
        self.applied = False
        self.push_t = None                  # time the rollout was applied
        self.pre_hys = self.pre_ttt = None  # cluster values in force just before the rollout (known to the SMO)
        self.oam_log = []                   # (t, "push" | "restore")

    def _place(self, plant):
        plant.pos[self.idx] = self.lay.wrap(self.centre + self.s[:, None] * self.u)
        plant.heading[self.idx] = np.arctan2(self.u[1], self.u[0]) + np.where(self.dir > 0, 0.0, np.pi)

    def oam_set(self, plant, hys, ttt, now, tag):
        """OAM/SMO write of per-cluster-cell Hys (dB) and TTT (ms) outside the RIC."""
        hys = np.broadcast_to(np.asarray(hys, float), self.cluster.shape)
        ttt = np.broadcast_to(np.asarray(ttt, int), self.cluster.shape)
        plant.hys[self.cluster] = hys
        plant.ttt[self.cluster] = ttt
        if plant.oam_hook is not None:
            for c, h, m in zip(self.cluster, hys, ttt, strict=True):
                plant.oam_hook(("hys", int(c)), float(h), now)
                plant.oam_hook(("ttt", int(c)), float(m), now)
        self.oam_log.append((now, tag))

    def on_tick(self, plant, now):
        if not self.applied and now >= self.t_mis:                                      # one-time OAM rollout
            self.pre_hys, self.pre_ttt = plant.hys[self.cluster].copy(), plant.ttt[self.cluster].copy()
            self.oam_set(plant, self.cfg.mis_hys_db, int(self.cfg.mis_ttt_ms), now, "push")
            self.applied, self.push_t = True, now

    def move(self, plant, dt):
        s = self.s + self.dir * self.v * dt
        hi, lo = s > self.half, s < -self.half
        s = np.where(hi, 2 * self.half - s, np.where(lo, -2 * self.half - s, s))       # reflect (U-turn)
        self.dir = np.where(hi, -1.0, np.where(lo, 1.0, self.dir))
        self.s = s
        self._place(plant)


SCENARIOS = {"surge": SurgeScenario, "mistune": MistuneScenario}


def validate_e6p(P):
    """Reject invalid E6-P parameters (called only when the extension is enabled)."""
    def need(ok, msg):
        if not ok:
            raise ValueError(f"E6-P: {msg}")
    need(P.ptx_scope in ("macro", "all"), "ptx_scope must be 'macro' or 'all'")
    lo, hi = P.ptx_range_db
    need(lo <= P.ptx_init_db <= hi, "ptx_init_db must lie in ptx_range_db")
    need(P.ptx_grid_db > 0 and P.ptx_max_step_db > 0, "ptx grid/step must be > 0")
    need(0.0 <= P.prot_frac <= 1.0 and P.prot_floor_bps > 0, "prot_frac in [0, 1] and prot_floor_bps > 0")
    mlo, mhi = P.prot_min_range
    need(0.0 <= mlo <= P.prot_min_init <= mhi <= 1.0, "need 0 <= prot_min lo <= init <= hi <= 1")
    need(P.prot_min_grid > 0 and P.prot_min_max_step > 0, "prot_min grid/step must be > 0")


def make_scenario(cfg, plant):
    cfg.validate_scenario()
    if cfg.scenario not in SCENARIOS:
        raise ValueError(f"unknown E6 scenario {cfg.scenario!r}")
    return SCENARIOS[cfg.scenario](cfg, plant)

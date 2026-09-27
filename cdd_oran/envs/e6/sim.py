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
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .geometry import GainMaps, Layout

STREAMS = {"layout": 1, "ue": 2, "mob": 3, "traffic": 4, "meas": 5, "load": 6, "kpm": 7, "xapp": 8}
LL, EMBB, BE = 0, 1, 2
A_L3 = 0.5 ** (C.L3_K / 4.0)
NOISE_W = 10 ** ((C.NOISE_DBM_HZ + 10 * np.log10(C.BW_HZ) + C.UE_NF_DB) / 10) / 1000.0


def _rng(seed, stream, t=0):
    return np.random.default_rng([int(seed), 6600, STREAMS[stream], int(t)])


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
        self.cell_viol_run = np.zeros((nc, 3))
        self.ctr = self._new_counters()

    # ================================================================================================ helpers
    def _gains(self):
        g = self.gm.lookup(self.pos).astype(float)
        g[self.indoor] -= C.O2I_DB
        g[:, self.asleep | (self.waking_until > self.t * C.TICK_S)] = -300.0
        return g

    def _new_counters(self):
        nc = self.nc
        return {"prb_used": np.zeros((nc, 3)), "prb_rsv": np.zeros(nc), "prb_cap": np.zeros(nc), "ticks": 0, "bits": np.zeros((nc, 3)),
                "ho_att": np.zeros((nc, nc)), "ho_succ": np.zeros((nc, nc)), "too_late": np.zeros((nc, nc)),
                "too_early": np.zeros((nc, nc)), "wrong_cell": np.zeros((nc, nc)), "pingpong": np.zeros((nc, nc)),
                "rlf": np.zeros(nc), "energy_j": np.zeros(nc), "ll_delay": [[] for _ in range(nc)],
                "embb_thp": [[] for _ in range(nc)], "act_ue": np.zeros((nc, 3))}

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
        rl = _rng(cfg.seed, "load", t)
        a = np.exp(-dt / cfg.m_tau_s)
        logm = np.log(self.m) * a + cfg.m_sigma * np.sqrt(1 - a * a) * rl.normal(size=7)
        self.m = np.exp(logm)
        rm = _rng(cfg.seed, "mob", t)
        turn = rm.uniform(size=n) < dt / 20.0                          # [A] new heading every ~20 s
        self.heading = np.where(turn, rm.uniform(0, 2 * np.pi, n), self.heading)
        mv = self.mobile_idx
        if len(mv):
            step = self.speed[mv] * dt
            self.pos[mv] = self.lay.wrap_near(self.pos[mv] + step[:, None] * np.c_[np.cos(self.heading[mv]),
                                                                                    np.sin(self.heading[mv])])
        rt = _rng(cfg.seed, "traffic", t)
        site = self.site_of[self.serv]
        total = cfg.warmup_s + cfg.scored_s
        ramp = cfg.ramp[0] + (cfg.ramp[1] - cfg.ramp[0]) * min(now / max(total, 1e-9), 1.0)   # compressed diurnal ramp
        mult = self.m[site] * cfg.lf() * ramp
        ll = self.sl == LL
        arr = np.zeros(n)
        arr[ll] = rt.poisson(self.ll_pps[ll] * dt * ramp) * C.LL_PKT_BYTES * 8
        rate = np.where(self.sl == EMBB, cfg.embb_files_per_s, cfg.be_files_per_s) * mult
        files = rt.poisson(np.where(ll, 0.0, rate * dt))
        arr[~ll] = files[~ll] * C.EMBB_FILE_BYTES * 8
        self.q += arr
        self.ll_cum_a += np.where(ll, arr, 0.0)
        self.ll_arr[:, t % 64] = self.ll_cum_a
        # ---- 2. radio
        g = self._gains()
        rq = _rng(cfg.seed, "meas", t)
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
        cell = self.serv
        for u in np.nonzero(sl == LL)[0]:                     # every LL UE-second is a sample (zeros included)
            if not self.sec_outage[u]:
                self.ctr["ll_delay"][cell[u]].append(self.sec_maxdelay[u])
        for u in np.nonzero((sl == EMBB) & (self.sec_backlog_t >= 0.2))[0]:
            self.ctr["embb_thp"][cell[u]].append(min(thp[u], 1e9))
        self.sec_backlog_t[:] = 0
        self.sec_bits[:] = 0
        self.sec_maxdelay[:] = 0
        self.sec_outage[:] = False

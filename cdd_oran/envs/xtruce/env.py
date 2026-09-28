"""XEnv: epoch-level multi-cell gNB plant of xTRUCE Sec. IV / V-A, with a knob-level xApp request interface.

State s_t (Sec. IV-B): channel gains G_{b,u,k}(t), interference measured in t-1, queues Q_u(t), arrivals.
Action a_t (eqs. 12-13): configuration (alpha_b sleep state, z_u association; changed only at configuration epochs,
every ``t_cfg`` epochs) and fast (x_u RB share, p_u Tx power). As in the paper's simulator (Sec. V-A) the fast action
is NOT differentiated over RBs: x_{u,k} = x_u and p_{u,k} = p_u / K for every k, while the rate (14) is still summed
over RBs with per-RB fading. Added knob ``("pcap", b)`` = configured cell Tx-power cap (the direct action of the
energy / interference xApps); the plant enforces min(pcap, P_max).

Per epoch (two-phase, like E6):
  1. ``reqs = env.step_propose()``: xApps read the state, post structured proposals (eq. 2) into ``env.proposals`` and
     emit knob requests ``{"xapp", "knob", "cur", "prop", "t", "kind", "target"}``.
  2. an arbiter decides: ``accepted = arbiter.decide(env, reqs)`` (None = accept all). It may drop, modify
     (``arbiters.modify`` / ``arbiters.half``) or add requests (xapp name of its own), and may call ``env.lock``.
  3. ``env.step_apply(accepted)``: applied in list order (last writer wins); requests on locked knobs are dropped;
     configuration knobs only at configuration epochs. Then the plant enforces c1-c4 (``phys="clip"``: proportional
     rescaling, the paper's Clipping operation) and realises rates (14), queues (15), energy (16).
``env.step(arbiter)`` does all three. ``copy()`` between the phases carries the pending requests.

Knobs: ("share", u) in [0, 1]; ("power", u) W; ("pcap", b) W in [0, P_max]; ("sleep", b) in {0, 1};
("assoc", u) = cell index. Sleep moves the cell's UEs to their best active cell (strongest mean gain) and returns
them when it wakes; a moved UE starts at an equal share of its new cell.
"""
from __future__ import annotations

import copy as _copy

import numpy as np

from .channel import cn, drop_ues, hex_sites, large_scale_gain
from .config import XConfig

RESEED_TAG = 0x7C5E
STREAMS = {"fade": 1, "traffic": 2, "xapp": 3}
FAST_KNOBS = ("share", "power", "pcap")
CFG_KNOBS = ("sleep", "assoc")
LN2 = np.log(2.0)


class XEnv:
    def __init__(self, cfg: XConfig | None = None, seed: int = 0):
        cfg = cfg or XConfig()
        self.cfg, self.seed = cfg, int(seed)
        B, U, K = cfg.n_cells, cfg.n_ues, cfg.n_rb
        self.B, self.U, self.K = B, U, K
        self.W = cfg.rb_hz
        r0 = np.random.default_rng([self.seed, 0])                 # construction draws (state, never re-drawn)
        self.sites = hex_sites(B, cfg.isd_m)
        self.ue_pos = drop_ues(r0, self.sites, U, cfg.isd_m, cfg.min_dist_m)
        self.Gm, self.dist, self.los = large_scale_gain(cfg, r0, self.sites, self.ue_pos)
        self.prot = np.zeros(U, bool)
        self.prot[r0.choice(U, cfg.n_prot, replace=False)] = True
        self._rng = {k: np.random.default_rng([self.seed, sid]) for k, sid in STREAMS.items()}
        self._fshape = (B, U, K) if cfg.fading_per_rb else (B, U, 1)
        self.h = cn(self._rng["fade"], self._fshape)               # stationary start
        self._gains()
        # action state (configured values; the executed action is derived each epoch)
        self.alpha = np.ones(B, bool)
        self.z = np.argmax(self.Gm, 0).astype(np.int64)
        self.displaced = np.full(U, -1, np.int64)
        self.pcap = np.full(B, cfg.p_max_w)
        self.x = 1.0 / self.n_in_cell()[self.z]                    # certified initial action: equal share / power
        self.p = self.x * cfg.p_max_w
        self.Q = self._arrivals()
        self.lam_ewma = self.Q / cfg.epoch_s
        # "measured at t-1": interference of the initial action on the current channel
        x_e, p_e, _ = self.actuate()
        self.x_exec, self.p_exec = x_e, p_e
        self.P_cell_prev = np.bincount(self.z, p_e, minlength=B)
        self.T_meas, self.I_meas = self._rx(self.P_cell_prev)
        self.t = 0
        self.is_cfg = True
        from .xapps import build_xapps
        self.xapps = build_xapps(self, cfg.xapps)
        self.proposals: dict = {}                                  # xapp -> (t, [Target])
        self.lock_until: dict = {}
        self.stats = {"req": 0, "acc": 0, "applied": 0, "lock_blocked": 0, "cfg_offcycle": 0, "invalid": 0,
                      "changes": 0}
        self.req_by = {x.name: 0 for x in self.xapps}
        self.acc_by = {x.name: 0 for x in self.xapps}
        self.acc = {"epochs": 0, "energy_j": 0.0, "prot_viol_ue_s": 0, "prot_viol_thr_ue_s": 0, "prot_ue_s": 0, "e1_viol_epochs": 0,
                    "phys_viol_epochs": 0, "served_bits": 0.0, "arrived_bits": 0.0, "sleep_cell_s": 0,
                    "oper_viol_epochs": 0}
        self.acc_cell_j = np.zeros(B)
        self.log: list = []
        self._pending = None
        self._obs = None

    # ------------------------------------------------------------------ helpers
    def _gains(self):
        self.G = self.Gm[:, :, None] * (self.h.real ** 2 + self.h.imag ** 2)
        if self.G.shape[2] != self.K:
            self.G = np.broadcast_to(self.G, (self.B, self.U, self.K)).copy()

    def _arrivals(self):
        c = self.cfg
        mean = c.traffic_bps * c.epoch_s
        if c.arrivals == "exp":
            return self._rng["traffic"].exponential(mean, self.U)
        if c.arrivals == "const":
            return np.full(self.U, mean)
        raise ValueError(c.arrivals)

    def _rx(self, P_cell):
        """-> (total received power per (u, k), interference per (u, k) w.r.t. the serving cell)."""
        Pk = P_cell / self.K
        T = np.einsum("b,buk->uk", Pk, self.G)
        own = Pk[self.z][:, None] * self.G[self.z, np.arange(self.U)]
        return T, np.maximum(T - own, 0.0)

    def n_in_cell(self, z=None):
        return np.bincount(self.z if z is None else z, minlength=self.B)

    def budget(self):
        """Per-cell usable Tx power: alpha_b * min(pcap_b, P_max) (W)."""
        return self.alpha * np.minimum(self.pcap, self.cfg.p_max_w)

    def _move(self, u, b):
        """Re-attach UE u to cell b (default RRM, [A]): the old cell's remaining UEs take over u's share and power in
        proportion (work-conserving), u gets an equal share 1/n_b of b with the matching power, and b's other UEs
        are scaled by (1 - 1/n_b), so both cells' configured actions keep their totals."""
        a = self.z[u]
        rest = (self.z == a)
        rest[u] = False
        for arr in (self.x, self.p):
            s = arr[rest].sum()
            if s > 0:
                arr[rest] *= (s + arr[u]) / s
        self.z[u] = b
        m = self.z == b
        n = max(int(m.sum()), 1)
        m[u] = False
        self.x[m] *= 1.0 - 1.0 / n
        self.p[m] *= 1.0 - 1.0 / n
        self.x[u] = 1.0 / n
        self.p[u] = self.x[u] * min(self.pcap[b], self.cfg.p_max_w)

    def rate(self, x, p, G_serv, N):
        """Eq. (14) with x_{u,k} = x_u, p_{u,k} = p_u / K: R_u = x_u W sum_k log2(1 + G p_u / (K x_u N)) [bit/s]."""
        xs = np.maximum(x, 1e-12)
        a = G_serv * (p / (self.K * xs))[:, None] / N
        return np.where(x > 0, x * self.W * np.log2(1.0 + a).sum(1), 0.0)

    def psd_rate(self, s, users=None, N=None):
        """Full-share rate of ``users`` at per-RB power ``s`` W (current channel, measured interference)."""
        o = self.obs()
        Gs, NN = o["Gs"], (o["N"] if N is None else N)
        if users is not None:
            Gs, NN = Gs[users], NN[users]
            s = np.asarray(s)
        return self.W * np.log2(1.0 + Gs * np.reshape(s, (-1, 1)) / NN).sum(1)

    def obs(self):
        """Per-epoch measurement cache for xApps / arbiters (E2 view: channel gains, measured interference, queues,
        arrival-rate EWMA, nominal demand load)."""
        if self._obs is not None:
            return self._obs
        c = self.cfg
        ar = np.arange(self.U)
        Gs = self.G[self.z, ar]
        N = c.noise_w + self.I_meas
        s_nom = c.p_max_w / self.K
        r_nom = self.W * np.log2(1.0 + Gs * s_nom / N).sum(1)
        dem = np.bincount(self.z, self.lam_ewma / np.maximum(r_nom, 1.0), minlength=self.B)
        # alternative-cell nominal full-band rate from mean gains (for steering / sleep estimates), (B, U)
        Pk = self.P_cell_prev / self.K
        tot = (self.Gm * Pk[:, None]).sum(0)
        i_alt = np.maximum(tot[None, :] - self.Gm * Pk[:, None], 0.0)
        r_alt = self.K * self.W * np.log2(1.0 + self.Gm * s_nom / (c.noise_w + i_alt))
        self._obs = {"Gs": Gs, "N": N, "r_nom": r_nom, "dem_load": np.where(self.alpha, dem, 0.0),
                     "r_alt": r_alt, "Q": self.Q, "lam": self.lam_ewma}
        return self._obs

    # ------------------------------------------------------------------ knobs
    def knobs(self):
        return ([("share", u) for u in range(self.U)] + [("power", u) for u in range(self.U)]
                + [("pcap", b) for b in range(self.B)] + [("sleep", b) for b in range(self.B)]
                + [("assoc", u) for u in range(self.U)])

    def knob_get(self, k):
        kind, i = k
        if kind == "share":
            return float(self.x[i])
        if kind == "power":
            return float(self.p[i])
        if kind == "pcap":
            return float(self.pcap[i])
        if kind == "sleep":
            return float(not self.alpha[i])
        if kind == "assoc":
            return float(self.z[i])
        raise KeyError(k)

    def knob_cells(self, k):
        """Cells a knob acts on (for conflict grouping; an ``assoc`` request also touches its target cell, which
        ``arbiters.request_cells`` adds)."""
        kind, i = k
        return (int(i),) if kind in ("pcap", "sleep") else (int(self.z[i]),)

    def lock(self, knob, epochs):
        """Lock ``knob`` for ``epochs`` epochs from now: requests on it are dropped while t < t_now + epochs."""
        self.lock_until[knob] = self.t + int(epochs)

    def locked(self, knob):
        return self.t < self.lock_until.get(knob, -1)

    def _set(self, k, v):
        """Apply one knob write to the configured action. -> True if applied."""
        c = self.cfg
        kind, i = k
        v = float(v)
        if kind == "share":
            self.x[i] = max(v, 0.0)
        elif kind == "power":
            self.p[i] = max(v, 0.0)
        elif kind == "pcap":
            self.pcap[i] = min(max(v, 0.0), c.p_max_w)
        elif kind == "sleep":
            want_active = v < 0.5
            if want_active == bool(self.alpha[i]):
                return True
            if not want_active and self.alpha.sum() <= 1:
                return False                                         # never switch the last cell off
            self.alpha[i] = want_active
            self._fix_config()
        elif kind == "assoc":
            b = int(round(v))
            if not (0 <= b < self.B) or not self.alpha[b]:
                return False
            if b != self.z[i]:
                self.displaced[i] = -1
                self._move(i, b)
        return True

    def _fix_config(self):
        """UEs on sleeping cells -> best active cell (remember the origin); displaced UEs return to a woken cell."""
        for u in np.nonzero(self.displaced >= 0)[0]:
            b = self.displaced[u]
            if self.alpha[b] and self.z[u] != b:
                self.displaced[u] = -1
                self._move(u, b)
        for u in np.nonzero(~self.alpha[self.z])[0]:
            g = np.where(self.alpha, self.Gm[:, u], -np.inf)
            if self.displaced[u] < 0:
                self.displaced[u] = self.z[u]
            self._move(u, int(np.argmax(g)))

    def preview_config(self, cfg_requests):
        """(alpha, z) that applying these sleep / assoc requests would produce (no mutation)."""
        saved = (self.alpha.copy(), self.z.copy(), self.displaced.copy(), self.x.copy(), self.p.copy())
        try:
            for r in cfg_requests:
                self._set(r["knob"], r["prop"])
            return self.alpha.copy(), self.z.copy()
        finally:
            self.alpha, self.z, self.displaced, self.x, self.p = saved

    def actuate(self):
        """Executed fast action from the configured one. -> (x_exec, p_exec, violated physical limits c1-c4)."""
        c = self.cfg
        x, p, z = self.x.copy(), self.p.copy(), self.z
        K = self.K
        s_x = np.bincount(z, x, minlength=self.B)
        s_p = np.bincount(z, p, minlength=self.B)
        tol = 1e-6                                                # solver-level tolerance
        viol = bool((x > 1 + tol).any() or (s_x > 1 + tol).any() or (p > x * K * c.p_rb_w + tol).any()
                    or (s_p > self.alpha * c.p_max_w + tol).any())
        if c.phys == "clip":
            x = np.clip(x, 0.0, 1.0)
            s_x = np.bincount(z, x, minlength=self.B)
            x = x * np.where(s_x > 1.0, 1.0 / np.maximum(s_x, 1e-12), 1.0)[z]
            p = np.minimum(p, x * K * c.p_rb_w)
            bud = self.budget()
            s_p = np.bincount(z, p, minlength=self.B)
            p = p * np.where(s_p > bud, bud / np.maximum(s_p, 1e-300), 1.0)[z]
        elif c.phys == "none":
            p = p * self.alpha[z]
        else:
            raise ValueError(c.phys)
        return x, p, viol

    # ------------------------------------------------------------------ episode loop
    def step_propose(self):
        if self._pending is not None:
            raise RuntimeError("step_propose called twice without step_apply")
        self.is_cfg = (self.t % self.cfg.t_cfg) == 0
        reqs = []
        for xa in self.xapps:
            targets, r = xa.act()
            self.proposals[xa.name] = (self.t, targets)
            reqs.extend(r)
            self.req_by[xa.name] += len(r)
        self.stats["req"] += len(reqs)
        self._pending = reqs
        return reqs

    def valid_proposals(self):
        """Targets of proposals still valid (posted within ``proposal_ttl`` epochs)."""
        out = []
        for t0, tg in self.proposals.values():
            if self.t - t0 < self.cfg.proposal_ttl:
                out.extend(tg)
        return out

    def step(self, arbiter=None):
        reqs = self.step_propose()
        if arbiter is None:
            acc = reqs
        elif hasattr(arbiter, "decide"):
            acc = arbiter.decide(self, reqs)
        else:
            acc = arbiter(self, reqs)
        return self.step_apply(acc)

    def run(self, n_epochs, arbiter=None):
        for _ in range(int(n_epochs)):
            self.step(arbiter)
        return self.summary()

    def step_apply(self, accepted):
        if self._pending is None:
            raise RuntimeError("step_apply called without a pending step_propose")
        c = self.cfg
        n_req = len(self._pending)
        self._pending = None
        z0, a0 = self.z.copy(), self.alpha.copy()
        for r in accepted if accepted is not None else ():
            k = r["knob"]
            if self.locked(k):
                self.stats["lock_blocked"] += 1
                continue
            if k[0] in CFG_KNOBS and not self.is_cfg:
                self.stats["cfg_offcycle"] += 1
                continue
            self.stats["acc"] += 1
            if r["xapp"] in self.acc_by:
                self.acc_by[r["xapp"]] += 1
            before = self.knob_get(k)
            if self._set(k, r["prop"]):
                self.stats["applied"] += 1
                self.stats["changes"] += int(abs(self.knob_get(k) - before) > 1e-12)
            else:
                self.stats["invalid"] += 1
        x_e, p_e, phys_viol = self.actuate()
        kpi = self._realize(x_e, p_e)
        # operator-rule bookkeeping (e2-e5 of eq. 20) against the previous executed action
        same = self.z == z0
        e23 = bool((np.abs(x_e - self.x_exec)[same] > c.dx + 1e-6).any()
                   or (np.abs(p_e - self.p_exec)[same] / self.K > c.dp_w + 1e-6).any())
        e45 = bool((self.alpha != a0).sum() > c.d_act or (~same).sum() > c.d_str)
        self.x_exec, self.p_exec = x_e, p_e
        a = self.acc
        a["epochs"] += 1
        a["phys_viol_epochs"] += int(phys_viol)
        a["oper_viol_epochs"] += int(e23 or e45)
        if c.log:
            self.log.append({"t": self.t, "cfg_epoch": self.is_cfg, "n_req": n_req, "n_acc": len(accepted or ()),
                             "phys_viol": phys_viol, "oper_viol": e23 or e45, "x": x_e, "p": p_e,
                             "assoc": self.z.copy(), "pcap": self.pcap.copy(), **kpi})
        self._advance()
        return kpi

    def _realize(self, x_e, p_e):
        c = self.cfg
        B = self.B
        P_cell = np.bincount(self.z, p_e, minlength=B)
        T, Iu = self._rx(P_cell)
        Gs = self.G[self.z, np.arange(self.U)]
        R = self.rate(x_e, p_e, Gs, c.noise_w + Iu)
        tau = c.epoch_s
        served = np.minimum(self.Q, R * tau)
        # delivered-throughput variant of the floor: backlogged for the whole epoch at the floor and served below it
        below_thr = self.prot & (self.Q >= c.rmin_bps * tau) & (served < c.rmin_bps * tau * (1 - 1e-9))
        lam = self._arrivals()
        self.Q = self.Q - served + lam                                # eq. (15)
        self.lam_ewma = (1 - c.load_ewma) * self.lam_ewma + c.load_ewma * lam / tau
        E = self.alpha * (c.p_cir_w + c.delta_p * P_cell) + (~self.alpha) * c.p_sleep_w   # eq. (16)
        out_mask = self.z[None, :] != np.arange(B)[:, None]
        I_out = (P_cell / self.K) * (self.G.sum(2) * out_mask).sum(1)  # eq. (17)
        load = np.bincount(self.z, x_e, minlength=B)                  # eq. (18)
        below = self.prot & (R < c.rmin_bps * (1 - 1e-9))
        a = self.acc
        a["energy_j"] += float(E.sum() * tau)
        self.acc_cell_j += E * tau
        a["prot_viol_ue_s"] += int(below.sum())
        a["prot_viol_thr_ue_s"] += int(below_thr.sum())
        a["prot_ue_s"] += int(self.prot.sum())
        a["e1_viol_epochs"] += int(below.any())
        a["served_bits"] += float(served.sum())
        a["arrived_bits"] += float(lam.sum())
        a["sleep_cell_s"] += int((~self.alpha).sum())
        self.P_cell_prev, self.T_meas, self.I_meas = P_cell, T, Iu
        return {"rate": R, "thr": served / tau, "prot_below": below, "prot_below_thr": below_thr, "power_w": E, "ptx_w": P_cell,
                "sleep": ~self.alpha, "intf_out_w": I_out, "load": load, "queue_bits": self.Q.copy()}

    def _advance(self):
        rho = self.cfg.fading_rho
        self.h = rho * self.h + np.sqrt(1.0 - rho * rho) * cn(self._rng["fade"], self._fshape)
        self._gains()
        self.t += 1
        self._obs = None

    # ------------------------------------------------------------------ results / copies
    def summary(self):
        a = self.acc
        n = max(a["epochs"], 1)
        return {**a, "mean_power_w": a["energy_j"] / (n * self.cfg.epoch_s),
                "prot_viol_frac": a["prot_viol_ue_s"] / max(a["prot_ue_s"], 1),
                "e1_viol_epoch_frac": a["e1_viol_epochs"] / n, "phys_viol_epoch_frac": a["phys_viol_epochs"] / n,
                "thr_mean_bps": a["served_bits"] / (n * self.cfg.epoch_s * self.U), **self.stats}

    def copy(self, reseed: int | None = None, keep_log: bool = False):
        """Deep copy for lookahead. ``reseed=None``: the copy replays the same future exactly. ``reseed=k``: current
        state kept, every FUTURE draw (fading innovations, arrivals incl. this epoch's if taken between the two
        phases, hallucination draws) comes from streams keyed by (seed, stream, RESEED_TAG, k, t); same k -> same
        re-drawn future. Construction draws (layout, shadowing, LOS, protected set) are state and are kept.
        The copy's log starts empty unless ``keep_log``."""
        memo = {id(a): a for a in (self.sites, self.ue_pos, self.Gm, self.dist, self.los, self.prot)}
        if not keep_log:
            memo[id(self.log)] = []
        c = _copy.deepcopy(self, memo)
        if reseed is not None:
            c._rng = {k: np.random.default_rng([self.seed, sid, RESEED_TAG, int(reseed), self.t])
                      for k, sid in STREAMS.items()}
        return c

    def floors_reachable(self):
        """Paper's seed screen (Sec. V-A, "network realizations keep the protected-user rate floors physically
        reachable"), [A] operationalised on mean gains: in every cell the protected users' floors fit in one band
        at P_rb per RB with all other cells at full power."""
        c = self.cfg
        Pk = np.full(self.B, c.p_max_w / self.K)
        need = np.zeros(self.B)
        for u in np.nonzero(self.prot)[0]:
            b = self.z[u]
            i = (self.Gm[:, u] * Pk).sum() - self.Gm[b, u] * Pk[b]
            r = self.K * self.W * np.log2(1 + self.Gm[b, u] * c.p_rb_w / (c.noise_w + i))
            need[b] += c.rmin_bps / r
        pw = need * self.K * c.p_rb_w
        return bool((need <= 1.0).all() and (pw <= c.p_max_w + 1e-9).all())


def screened_seeds(n, start=0, cfg=None, max_tries=10000):
    """First ``n`` seeds >= ``start`` whose realisation passes ``XEnv.floors_reachable``."""
    out, s = [], int(start)
    while len(out) < n and s < start + max_tries:
        if XEnv(cfg, s).floors_reachable():
            out.append(s)
        s += 1
    return out

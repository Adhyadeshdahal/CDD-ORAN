"""The four xApp agents of xTRUCE Sec. IV-D, as deterministic rules (the paper's safety / priority / scalability
sweeps also use "deterministic rules", Sec. V-A, but does not publish them; every rule constant here is [A]).

Each xApp, every epoch, returns (targets, requests):
  * targets: its structured proposal, eq. (2), a list of ``config.Target`` (what the paper's xTRUCE arbiter consumes);
  * requests: its own target-to-action map as knob requests (what a request-level arbiter decides on).
Hallucination (Fig. 3; model [A]): with level h in [0, 1] each target is corrupted with probability h, and a corrupted
theta is multiplied by 10^(2 h s), s ~ U(-1, 1). Requests are derived from the (possibly corrupted) targets. The
number of draws per epoch is fixed, so the xApp stream stays aligned across arbiters.

  QoS  : hard R_u >= qos_prot_bps (protected, kappa 1); soft R_u >= arrival-rate EWMA (others, kappa 2: the
         rate that stops the backlog Q_u growing).
         Requests: per active cell, at a uniform PSD (cell budget / K): protected users first (share =
         target (1 + margin) / rate), the rest of the band split over the others in proportion to their demand
         min(1, target / nominal rate); power = share x cell budget. Asks for the cell's pcap back to P_max when
         its protected users cannot fit under the current cap.
  ES   : soft E_b <= es_cap_w (kappa 2). Requests: pcap_b = (cap - P_cir) / Delta_p when above it (fast); at
         configuration epochs, sleep the least-loaded cell whose demand load < es_sleep_load if its UEs fit in the
         neighbours below es_wake_load (or, if the cap is below P_cir, the least-loaded cell regardless), wake a
         sleeping cell when an active cell's demand load > es_wake_load. At most one sleep change per epoch.
  IC   : soft I_out_b <= n_victims * K * sigma^2 * 10^(ic_iot_db / 10) (kappa 2). Requests: lower pcap_b to the
         power meeting the cap (EWMA-smoothed caused-interference gain), never below ic_min_w; only lowers.
  LB   : soft rho_b <= lb_cap (kappa 3). Requests (configuration epochs): move the UE of the most loaded cell with the
         smallest gain gap (<= lb_offset_db) to an active neighbour whose demand load is lb_hyst lower. Demand load =
         sum over the cell's UEs of arrival-rate EWMA / nominal full-band rate (the paper's rho_b, eq. 18, counts
         allocated shares, which a work-conserving allocation keeps at 1).
"""
from __future__ import annotations

import numpy as np

from .config import Target


def others_mask(n, idx):
    m = np.zeros(n)
    m[idx] = 1.0
    return m


class XApp:
    name = "?"

    def __init__(self, env):
        self.env = env

    def req(self, knob, prop, target=None):
        e = self.env
        return {"xapp": self.name, "knob": knob, "cur": e.knob_get(knob), "prop": float(prop), "t": e.t,
                "kind": "cfg" if knob[0] in ("sleep", "assoc") else "fast", "target": target}

    def corrupt(self, theta):
        h = self.env.cfg.hallucination
        theta = np.asarray(theta, float).copy()
        if h <= 0 or theta.size == 0:
            return theta
        rng = self.env._rng["xapp"]
        u, s = rng.random(theta.size), rng.uniform(-1.0, 1.0, theta.size)
        bad = u < h
        theta[bad] *= 10.0 ** (2.0 * h * s[bad])
        return theta

    def act(self):
        raise NotImplementedError


class QoS(XApp):
    name = "QoS"

    def act(self):
        e, c = self.env, self.env.cfg
        o = e.obs()
        U = e.U
        theta = np.where(e.prot, c.qos_prot_bps, o["lam"])          # np: offered rate (stops backlog growth)
        theta = self.corrupt(theta)
        hard_np = c.all_hard
        tg = [Target(self.name, "rate", u, float(theta[u]), bool(e.prot[u]) or hard_np,
                     c.prio_prot if e.prot[u] else c.prio_np_e_i, c.beta, 1) for u in range(U)]
        out = []
        K = e.K
        bud = e.budget()
        for b in np.nonzero(e.alpha)[0]:
            us = np.nonzero(e.z == b)[0]
            if us.size == 0:
                continue
            pr, npu = us[e.prot[us]], us[~e.prot[us]]
            xs = np.zeros(us.size)
            pos = {u: i for i, u in enumerate(us)}
            s = bud[b] / K                                              # uniform PSD over the cell's band
            used_x = 0.0
            if pr.size:
                need = theta[pr] * (1.0 + c.qos_margin)
                xp = np.minimum(need / np.maximum(e.psd_rate(np.full(pr.size, s), pr), 1.0), 1.0)
                if xp.sum() > 1.0:                                      # floors do not fit under the current cap
                    if e.pcap[b] < c.p_max_w - 1e-9:
                        out.append(self.req(("pcap", int(b)), c.p_max_w, ("rate", int(pr[0]))))
                    xp = xp / xp.sum()
                for i, u in enumerate(pr):
                    xs[pos[u]] = xp[i]
                used_x = xp.sum()
            if npu.size:
                rs = max(0.0, 1.0 - used_x)
                d = np.minimum(1.0, theta[npu] / np.maximum(o["r_nom"][npu], 1.0))
                w = d / d.sum() if d.sum() > 0 else np.full(npu.size, 1.0 / npu.size)
                for i, u in enumerate(npu):
                    xs[pos[u]] = rs * w[i]
            ps = xs * bud[b]
            db = c.qos_deadband
            for i, u in enumerate(us):
                up = e.prot[u] and (xs[i] > e.x[u] + 1e-9 or ps[i] > e.p[u] + 1e-9)   # never sit below a floor
                if abs(xs[i] - e.x[u]) > db or (up and xs[i] > e.x[u] + 1e-9):
                    out.append(self.req(("share", int(u)), xs[i], ("rate", int(u))))
                if abs(ps[i] - e.p[u]) > db * max(e.p[u], 0.1) or (up and ps[i] > e.p[u] + 1e-9):
                    out.append(self.req(("power", int(u)), ps[i], ("rate", int(u))))
        return tg, out


class ES(XApp):
    name = "ES"

    def act(self):
        e, c = self.env, self.env.cfg
        o = e.obs()
        theta = self.corrupt(np.full(e.B, c.es_cap_w))
        act = np.nonzero(e.alpha)[0]
        tg = [Target(self.name, "energy", int(b), float(theta[b]), c.all_hard, c.prio_np_e_i, c.beta, -1)
              for b in act]
        out = []
        want = (theta - c.p_cir_w) / c.delta_p
        for b in act:
            if want[b] >= 0 and e.pcap[b] > want[b] + 1e-6:
                out.append(self.req(("pcap", int(b)), min(want[b], c.p_max_w), ("energy", int(b))))
        if e.is_cfg:
            dem = o["dem_load"]
            sl = np.nonzero(~e.alpha)[0]
            if sl.size and act.size and dem[act].max() > c.es_wake_load:
                out.append(self.req(("sleep", int(sl[0])), 0.0, ("energy", int(sl[0]))))
            elif act.size > 1:
                best, best_d = None, np.inf
                for b in act:
                    infeasible = want[b] < 0
                    if not infeasible and dem[b] >= c.es_sleep_load:
                        continue
                    # receivers' demand load after moving b's UEs to their best other active cell
                    new = dem.copy()
                    others = act[act != b]
                    Pk = e.P_cell_prev * others_mask(e.B, others) / e.K        # b silent after sleeping
                    for u in np.nonzero(e.z == b)[0]:
                        t = others[np.argmax(e.Gm[others, u])]
                        i = (e.Gm[:, u] * Pk).sum() - e.Gm[t, u] * Pk[t]
                        r = e.K * e.W * np.log2(1.0 + e.Gm[t, u] * (c.p_max_w / e.K) / (c.noise_w + i))
                        new[t] += o["lam"][u] / max(r, 1.0)
                    if not infeasible and (new[others] >= c.es_wake_load).any():
                        continue
                    if dem[b] < best_d:
                        best, best_d = b, dem[b]
                if best is not None:
                    out.append(self.req(("sleep", int(best)), 1.0, ("energy", int(best))))
        return tg, out


class IC(XApp):
    name = "IC"

    def __init__(self, env):
        super().__init__(env)
        self.g = None

    def act(self):
        e, c = self.env, self.env.cfg
        out_mask = e.z[None, :] != np.arange(e.B)[:, None]
        g = (e.G.sum(2) * out_mask).sum(1) / e.K                   # I_out_b = P_b * g_b (eq. 17)
        self.g = g if self.g is None else (1 - c.ic_ewma) * self.g + c.ic_ewma * g
        nv = out_mask.sum(1)
        theta = self.corrupt(nv * e.K * c.noise_w * 10.0 ** (c.ic_iot_db / 10.0))
        act = np.nonzero(e.alpha)[0]
        tg = [Target(self.name, "intf", int(b), float(theta[b]), c.all_hard, c.prio_np_e_i, c.beta, -1)
              for b in act]
        out = []
        for b in act:
            want = max(theta[b] / max(self.g[b], 1e-300), c.ic_min_w)
            if e.pcap[b] > 1.05 * want:
                out.append(self.req(("pcap", int(b)), min(want, c.p_max_w), ("intf", int(b))))
        return tg, out


class LB(XApp):
    name = "LB"

    def act(self):
        e, c = self.env, self.env.cfg
        o = e.obs()
        act = np.nonzero(e.alpha)[0]
        theta = self.corrupt(np.full(e.B, c.lb_cap))
        tg = [Target(self.name, "load", int(b), float(theta[b]), c.all_hard, c.prio_load, c.beta, -1) for b in act]
        out = []
        if e.is_cfg and act.size > 1:
            dem = o["dem_load"]
            b = act[np.argmax(dem[act])]
            if dem[b] > theta[b]:
                cand = [t for t in act if t != b and dem[t] < dem[b] - c.lb_hyst]
                best, gap = None, np.inf
                for u in np.nonzero(e.z == b)[0]:
                    for t in cand:
                        gp = 10.0 * np.log10(e.Gm[b, u] / e.Gm[t, u])
                        if gp <= c.lb_offset_db and gp < gap:
                            best, gap = (int(u), int(t)), gp
                if best is not None:
                    out.append(self.req(("assoc", best[0]), best[1], ("load", int(b))))
        return tg, out


XAPPS = {"QoS": QoS, "ES": ES, "IC": IC, "LB": LB}


def build_xapps(env, names):
    seen = set()
    out = []
    for n in names:
        if n not in XAPPS:
            raise ValueError(f"unknown xApp {n!r} (known: {sorted(XAPPS)})")
        if n in seen:
            raise ValueError(f"xApp {n} deployed twice")
        seen.add(n)
        out.append(XAPPS[n](env))
    return out

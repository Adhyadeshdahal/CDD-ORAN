"""xTRUCE two-stage arbiter (Sec. III-C / IV-E, Algorithm 1) on scipy SLSQP (the paper uses CVXPY with
CLARABEL / ECOS / SCS, which is not a dependency here).

Per epoch, over the fast variables (x_u, p_u) of every UE for a fixed configuration (alpha, z):
  H_t : c1-c4 (19), e1 protected floors (20) planned against measured interference + the largest one-epoch increase
        e2 allows (Sec. V-A), e2 / e3 as variable bounds around the previous executed action (UEs that change cell are
        exempt, [A]).
  Stage I (8)/(21): for each priority class l of hard targets, min V_l = sum beta [g]_+^2 s.t. H_t and
        V_j <= v_j* + eps_j (j < l), eps_j = 1e-4 (1 + v_j*).
  Stage II (9)/(22): min sum_soft beta [g]_+^2 + eta D(a, a_prev) s.t. H_t and V_l <= v_l* + eps_l for all l.
  Configuration epochs: candidates = {current configuration} + {each configuration request of this epoch alone}
  ([A] instance of the operator-prescribed Omega^cfg); the lexicographically smallest (v_1*, ..., v_L*) wins, ties
  keep the earlier candidate.
  Delay-safe rule (11) without a wall-clock budget: a* if verified in H_t, else the previous executed action if it
  is still in H_t, else the baseline (min sum p s.t. H_t), else the previous action flagged ``uncertified``.
KPI units inside the objective [A]: rate in Mbit/s, energy in W, caused interference and load relative to the cap.
Certificates / prices (10) are not produced.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

LN2 = np.log(2.0)


class _Problem:
    def __init__(self, env, targets, alpha, z):
        c = env.cfg
        self.env, self.c = env, c
        U, K, B = env.U, env.K, env.B
        self.U, self.K, self.B = U, K, B
        self.alpha, self.z = alpha, z
        ar = np.arange(U)
        G = env.G
        self.Gs = G[z, ar]
        moved = z != env.z
        self.moved = moved
        Pk_prev = env.P_cell_prev / K
        Ipl = np.where(moved[:, None], np.maximum(env.T_meas - Pk_prev[z][:, None] * self.Gs, 0.0), env.I_meas)
        off = (~alpha) & env.alpha                                     # cells switched off by this candidate
        if off.any():
            Ipl = np.maximum(Ipl - np.einsum("b,buk->uk", Pk_prev * off, G), 0.0)
        n_b = np.bincount(z, minlength=B)
        dPk = alpha * np.minimum(n_b * c.dp_w, np.maximum(0.0, c.p_max_w - env.P_cell_prev) / K)
        dI = np.einsum("b,buk->uk", dPk, G) - dPk[z][:, None] * self.Gs
        N = c.noise_w + Ipl
        self.N = np.where(env.prot[:, None], N + np.maximum(dI, 0.0), N)
        self.prot = np.nonzero(env.prot)[0]
        xp, pp = env.x_exec.copy(), env.p_exec.copy()
        self.xp, self.pp = xp, pp
        lo_x = np.where(moved, 0.0, np.maximum(0.0, xp - c.dx))
        hi_x = np.where(moved, 1.0, np.minimum(1.0, xp + c.dx))
        lo_p = np.where(moved, 0.0, np.maximum(0.0, pp - K * c.dp_w))
        hi_p = np.where(moved, K * c.p_rb_w, np.minimum(K * c.p_rb_w, pp + K * c.dp_w))
        hi_x = np.maximum(hi_x, lo_x)
        hi_p = np.maximum(hi_p, lo_p)
        self.bounds = list(zip(np.r_[lo_x, lo_p], np.r_[hi_x, hi_p], strict=True))
        self.lo, self.hi = np.r_[lo_x, lo_p], np.r_[hi_x, hi_p]
        act = np.nonzero(alpha)[0]
        rows, rhs = [], []
        for b in act:                                                  # c2, c1
            m = (z == b).astype(float)
            rows.append(np.r_[m, np.zeros(U)])
            rhs.append(1.0)
            rows.append(np.r_[np.zeros(U), m])
            rhs.append(c.p_max_w)
        for u in range(U):                                             # c4
            r = np.zeros(2 * U)
            r[U + u], r[u] = 1.0, -K * c.p_rb_w
            rows.append(r)
            rhs.append(0.0)
        self.A, self.b = np.array(rows), np.array(rhs)
        # targets -> affine / rate pieces
        self.t_hard, self.t_soft = [], []
        gout = (G.sum(2) * (z[None, :] != np.arange(B)[:, None])).sum(1) / K
        self.items = []
        for t in targets:
            if t.kpi == "rate":
                it = ("rate", t.idx, t.theta / 1e6)
            elif t.kpi in ("energy", "intf", "load"):
                if not alpha[t.idx]:
                    continue
                m = (z == t.idx).astype(float)
                if t.kpi == "energy":                                  # E - theta (W), affine in p
                    it = ("aff", np.r_[np.zeros(U), c.delta_p * m], c.p_cir_w - t.theta)
                elif t.kpi == "intf":                                  # (P_b gout - theta) / theta
                    it = ("aff", np.r_[np.zeros(U), m * gout[t.idx] / t.theta], -1.0)
                else:                                                  # (sum x - theta)
                    it = ("aff", np.r_[m, np.zeros(U)], -t.theta)
            else:
                continue
            self.items.append((it, t.hard, t.prio, t.beta))

    # rate and its gradient for all UEs (Mbit/s)
    def rate(self, v):
        U, K = self.U, self.K
        x, p = v[:U], v[U:]
        xs = np.maximum(x, 1e-9)
        a = self.Gs * (np.maximum(p, 0.0) / (K * xs))[:, None] / self.N
        lg = np.log2(1.0 + a)
        W = self.env.W / 1e6
        R = W * x * lg.sum(1)
        dRdp = W * (self.Gs / (K * self.N) / ((1.0 + a) * LN2)).sum(1)
        dRdx = W * (lg - a / ((1.0 + a) * LN2)).sum(1)
        return R, dRdx, dRdp

    def g(self, v, sel):
        """Shortfalls g and their gradients for the selected items."""
        R, dx, dp = self.rate(v)
        U = self.U
        gs, J = [], []
        for (it, _, _, _) in sel:
            if it[0] == "rate":
                u = it[1]
                gs.append(it[2] - R[u])
                row = np.zeros(2 * U)
                row[u], row[U + u] = -dx[u], -dp[u]
                J.append(row)
            else:
                gs.append(it[1] @ v + it[2])
                J.append(it[1])
        return np.array(gs), (np.array(J) if J else np.zeros((0, 2 * U)))

    def V(self, v, sel):
        if not sel:
            return 0.0, np.zeros(2 * self.U)
        g, J = self.g(v, sel)
        beta = np.array([s[3] for s in sel])
        gp = np.maximum(g, 0.0)
        return float((beta * gp ** 2).sum()), (2 * beta * gp) @ J

    def e1(self, v):
        R, dx, dp = self.rate(v)
        U = self.U
        rmin = self.c.rmin_bps / 1e6
        pr = self.prot
        J = np.zeros((pr.size, 2 * U))
        J[np.arange(pr.size), pr] = dx[pr]
        J[np.arange(pr.size), U + pr] = dp[pr]
        return R[pr] - rmin, J

    def base_cons(self):
        cons = [{"type": "ineq", "fun": lambda v: self.b - self.A @ v, "jac": lambda v: -self.A}]
        if self.prot.size:
            cons.append({"type": "ineq", "fun": lambda v: self.e1(v)[0], "jac": lambda v: self.e1(v)[1]})
        return cons

    def class_cons(self, classes, vstar):
        cons = []
        for cl, vs in zip(classes, vstar, strict=True):
            sel = self.sel_hard(cl)
            eps = self.c.eps_rel * (1 + vs)
            cons.append({"type": "ineq", "fun": (lambda v, s=sel, lim=vs + eps: lim - self.V(v, s)[0]),
                         "jac": (lambda v, s=sel: -self.V(v, s)[1])})
        return cons

    def sel_hard(self, cl):
        return [s for s in self.items if s[1] and s[2] == cl]

    def classes(self):
        return sorted({s[2] for s in self.items if s[1]})

    def x0(self):
        x0 = np.where(self.moved, 1.0 / np.maximum(np.bincount(self.z, minlength=self.B)[self.z], 1), self.xp)
        p0 = np.where(self.moved, x0 * self.c.p_max_w, self.pp)
        return np.clip(np.r_[x0, p0], self.lo, self.hi)

    def solve(self, fun, cons, v0, maxiter):
        r = minimize(fun, v0, jac=True, bounds=self.bounds, constraints=cons, method="SLSQP",
                     options={"maxiter": maxiter, "ftol": 1e-10})
        return np.clip(r.x, self.lo, self.hi)

    def stage1(self, maxiter):
        v = self.x0()
        vstar = []
        cls = self.classes()
        for i, cl in enumerate(cls):
            sel = self.sel_hard(cl)
            cons = self.base_cons() + self.class_cons(cls[:i], vstar)
            v = self.solve(lambda w, s=sel: self.V(w, s), cons, v, maxiter)
            vstar.append(self.V(v, sel)[0])
        return cls, vstar, v

    def stage2(self, cls, vstar, v0, maxiter):
        soft = [s for s in self.items if not s[1]]
        eta, K, prb = self.c.eta, self.K, self.c.p_rb_w
        U = self.U
        # D (Sec. IV-C): sum_{u,k} (x - x_prev)^2 + ((p - p_prev)/K / P_rb)^2 with x_{u,k} = x_u, p_{u,k} = p_u / K
        def f(v):
            val, gr = self.V(v, soft)
            dx, dp = v[:U] - self.xp, (v[U:] - self.pp) / (K * prb)
            val += eta * K * ((dx ** 2).sum() + (dp ** 2).sum())
            gr = gr + eta * K * np.r_[2 * dx, 2 * dp / (K * prb)]
            return val, gr
        cons = self.base_cons() + self.class_cons(cls, vstar)
        return self.solve(f, cons, v0, maxiter)

    def baseline(self, maxiter):
        U = self.U
        f = lambda v: (float(v[U:].sum()), np.r_[np.zeros(U), np.ones(U)])   # noqa: E731
        return self.solve(f, self.base_cons(), self.x0(), maxiter)

    def feasible(self, v, tol=1e-5):
        if (v < self.lo - tol).any() or (v > self.hi + tol).any():
            return False
        if (self.A @ v > self.b + tol).any():
            return False
        return not (self.prot.size and (self.e1(v)[0] < -tol).any())


def _lex_less(a, b, eps_rel):
    for x, y in zip(a, b, strict=True):
        if abs(x - y) > eps_rel * (1 + min(x, y)):
            return x < y
    return False


class XTruce:
    """``decide(env, requests)`` -> requests that set every UE's (share, power), lift pcap to P_max (xTRUCE owns
    the power split; the energy target enters via E_b) and carry the chosen configuration change. ``self.last``
    records the case of (11) used, v*, and the chosen configuration candidate."""
    name = "xtruce"

    def __init__(self, maxiter=100, config_search=True):
        self.maxiter, self.config_search = maxiter, config_search
        self.last: dict = {}
        self.cases = {"star": 0, "prev": 0, "base": 0, "uncertified": 0}

    def decide(self, env, requests):
        targets = env.valid_proposals()
        cands = [[]]
        if env.is_cfg and self.config_search:
            cands += [[r] for r in requests if r["kind"] == "cfg"]
        best = None
        for cr in cands:
            alpha, z = env.preview_config(cr)
            if cr and np.array_equal(alpha, env.alpha) and np.array_equal(z, env.z):
                continue
            pb = _Problem(env, targets, alpha, z)
            cls, vstar, v1 = pb.stage1(self.maxiter)
            if best is None or _lex_less(vstar, best[2], env.cfg.eps_rel):
                best = (cr, pb, vstar, cls, v1)
        cr, pb, vstar, cls, v1 = best
        v = pb.stage2(cls, vstar, v1, self.maxiter)
        case = "star"
        if not pb.feasible(v):
            cr = []
            pb = _Problem(env, targets, env.alpha.copy(), env.z.copy()) if best[0] else pb
            prev = np.r_[env.x_exec, env.p_exec]
            if pb.feasible(prev):
                v, case = prev, "prev"
            else:
                vb = pb.baseline(self.maxiter)
                v, case = (vb, "base") if pb.feasible(vb) else (prev, "uncertified")
        self.cases[case] += 1
        self.last = {"case": case, "vstar": vstar, "classes": cls, "cfg": cr}
        U = env.U
        out = [dict(r, xapp=self.name, arbiter_of=r["xapp"]) for r in cr]
        for u in range(U):
            out.append({"xapp": self.name, "knob": ("share", u), "cur": float(env.x[u]), "prop": float(v[u]),
                        "t": env.t, "kind": "fast", "target": None})
            out.append({"xapp": self.name, "knob": ("power", u), "cur": float(env.p[u]), "prop": float(v[U + u]),
                        "t": env.t, "kind": "fast", "target": None})
        for b in range(env.B):
            if env.pcap[b] < env.cfg.p_max_w:
                out.append({"xapp": self.name, "knob": ("pcap", b), "cur": float(env.pcap[b]),
                            "prop": env.cfg.p_max_w, "t": env.t, "kind": "fast", "target": None})
        return out

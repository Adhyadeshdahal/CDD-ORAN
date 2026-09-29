"""EXPLORATORY diagnostics of the FROZEN E6-scn-v1 xApp stack after the Gate A KILL (Worker M).

Read-only instrumentation: plant/KPM/xApp methods are wrapped on the INSTANCE only (no cdd_oran code is changed and no
RNG stream is consumed), so an instrumented episode is bit-identical to a plain one (checked by `selfcheck`).
Seeds: registered DEV diagnostic range 140000-140059, seed = 140000 + 10*stratum + j (stratum 0 base-medium,
1 base-high, 2 surge-medium, 3 surge-high, 4 mistune-medium, 5 mistune-high; j = 0..9). Gate A config: mix M4,
mobility mixed, 120 s warm-up + 600 s scored.

  uv run python scratchpad/e6_dev/v1_diag.py selfcheck
  uv run python scratchpad/e6_dev/v1_diag.py run SCENARIO LOAD SEED ARM [ARM ...]      # ARM: freeze | TS | MRO
  uv run python scratchpad/e6_dev/v1_diag.py analyze > scratchpad/e6_dev/v1_diag/analysis.txt
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

from cdd_oran.envs.e6 import config as C
from cdd_oran.envs.e6.baselines import freeze, subset
from cdd_oran.envs.e6.env import E6Env
from cdd_oran.envs.e6.sim import EMBB, LL, NOISE_W, Plant

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1_diag")
WARMUP, SCORED = 120.0, 600.0
TARGET = C.EMBB_THP_TARGET_BPS
RATE_PER_SE = C.N_PRB * C.PRB_HZ * (1 - C.OVERHEAD)            # full-band bits/s per unit spectral efficiency
CLS = ("ok", "outage", "sinr_lim", "cap_relievable_reach4", "cap_relievable_unreach", "cap_unrelievable")
STRATA = {("base", "medium"): 0, ("base", "high"): 1, ("surge", "medium"): 2, ("surge", "high"): 3,
          ("mistune", "medium"): 4, ("mistune", "high"): 5}


def seed_of(scenario, load, j):
    assert 0 <= j < 10
    return 140000 + 10 * STRATA[(scenario, load)] + j


def make_env(scenario, load, seed):
    cfg = C.E6Config(seed=seed, load=load, mobility="mixed", mix="M4", warmup_s=WARMUP, scored_s=SCORED,
                     scenario=scenario)
    return E6Env(cfg, log=False)


def arbiter(arm):
    return freeze if arm == "freeze" else subset((arm,))


class Probe:
    """Instance-level read-only wrappers. Records per-second per-UE / per-cell state and every event."""

    def __init__(self, env):
        self.env, p = env, env.plant
        self.p = p
        self.ue_cls, self.ue_serv, self.ue_sinr, self.ue_disk, self.ue_thp = [], [], [], [], []
        self.cell_off, self.cell_cap, self.cell_util, self.cell_act, self.cell_viol = [], [], [], [], []
        self.sec_t, self.env_surge = [], []
        self.ho, self.rlf, self.req, self.ts_log, self.mro_log, self.kpm_lag = [], [], [], [], [], []
        self.cfg_log = []
        self._wrap()

    # ------------------------------------------------------------------------------------------------ wrappers
    def _wrap(self):
        p, env = self.p, self.env
        o_close, o_ho, o_rlf = p._close_second, p._handover, p._rlf
        pre = {}

        def close():
            pre.update(self._snapshot())
            o_close()
            self._after_close(pre)

        def ho(u, tgt, now, pw):
            src = int(p.serv[u])
            rec = {"t": now, "u": int(u), "src": src, "tgt": int(tgt), "cio": float(p.cio[src, tgt]),
                   "cio_rev": float(p.cio[tgt, src]), "hys": float(p.hys[src]), "ttt": int(p.ttt[src]),
                   "margin": float(p.l3[u, tgt] - p.l3[u, src]), "sinr_src": float(p.sinr_ewma[u]),
                   "sl": int(p.sl[u]), "nbr": bool(tgt in p.lay.neighbours[src])}
            r0, pp0 = p.sla["rlf"], p.sla["pingpong"]
            o_ho(u, tgt, now, pw)
            rec["fail"] = bool(p.sla["rlf"] > r0)
            rec["pingpong"] = bool(p.sla["pingpong"] > pp0)
            rec["sinr_t"] = float(p.sinr_ewma[u]) if not rec["fail"] else None
            self.ho.append(rec)

        def rlf(u, now, ho_fail=False):
            cur = int(p.serv[u])
            keys = ("too_late", "too_early", "wrong_cell")
            before = {k: float(p.ctr[k].sum()) for k in keys}
            o_rlf(u, now, ho_fail)
            kind = next((k for k in keys if p.ctr[k].sum() > before[k]), "none")
            self.rlf.append({"t": now, "u": int(u), "cur": cur, "best": int(p.serv[u]), "ho_fail": bool(ho_fail),
                             "kind": kind, "hys": float(p.hys[cur]), "ttt": int(p.ttt[cur])})

        p._close_second, p._handover, p._rlf = close, ho, rlf

        kpm = env.kpm
        o_second, o_deliver = kpm.second, kpm.deliver

        def second(sec):
            c = p.ctr
            cap = c["prb_cap"]
            used = c["prb_used"].sum(1) + c["prb_rsv"]
            self.cell_util.append(np.where(cap > 0, used / np.maximum(cap, 1e-9), 0.0))
            o_second(sec)

        def deliver(now):
            out = o_deliver(now)
            for r in out:
                self.kpm_lag.append((r["gran"], float(r["t1"]), float(r["arrived"]), float(now)))
            return out

        kpm.second, kpm.deliver = second, deliver

        for x in env.xapps:
            o_res = x.result

            def res(r, accepted, applied, now, _o=o_res, _x=x):
                self.req.append({"t": now, "xapp": _x.name, "knob": list(r["knob"]), "cur": r["cur"],
                                 "prop": r["prop"], "accepted": bool(accepted), "applied": float(applied),
                                 "moved": float(applied) - float(r["cur"]) if accepted else 0.0})
                _o(r, accepted, applied, now)

            x.result = res
            if x.name == "TS":
                o_prop = x.propose

                def tsprop(now, _o=o_prop, _x=x):
                    out = _o(now)
                    rep = _x.reports.get("fast")
                    if rep is not None:
                        thp = _x.reports.get("thp")
                        p5 = thp["embb_thp_p5"] if thp is not None else np.full(p.nc, np.nan)
                        self.ts_log.append({"t": now, "u": _x.u.tolist(), "raw": rep["prb_util"].tolist(),
                                            "rep_t1": rep["t1"], "thp_t1": thp["t1"] if thp is not None else None,
                                            "p5": np.nan_to_num(p5, nan=-1).tolist(), "n_req": len(out),
                                            "u_hi": _x.u_hi, "u_tgt": _x.u_tgt, "gap": _x.gap})
                    return out

                x.propose = tsprop
            if x.name == "MRO":
                o_prop = x.propose

                def mroprop(now, _o=o_prop, _x=x):
                    out = _o(now)
                    hist = getattr(_x, "mob_hist", [])
                    if hist:
                        agg = {k: sum(h[k] for h in hist).sum() for k in
                               ("ho_att", "too_late", "too_early", "pingpong", "wrong_cell")}
                        self.mro_log.append({"t": now, "win_t0": hist[0]["t0"], "win_t1": hist[-1]["t1"],
                                             **{k: float(v) for k, v in agg.items()},
                                             "reqs": [[list(r["knob"]), r["cur"], r["prop"]] for r in out]})
                    return out

                x.propose = mroprop

    # ------------------------------------------------------------------------------------------------ snapshots
    def _snapshot(self):
        p = self.p
        now = p.t * C.TICK_S
        g = p._gains()
        pw = 10 ** (g / 10) / 1000.0
        load_w = pw * p.occ[None, :]
        tot = load_w.sum(1)
        sinr = 10 * np.log10(pw / (tot[:, None] - load_w + NOISE_W) + 1e-30)
        car = np.where(p.car_on_at > now, p.n_car - 1, p.n_car)
        capf = np.where(p.asleep | (p.waking_until > now), 0.0, car / p.n_trx) * (1 - p.ll_ratio)
        R = Plant.se(sinr) * RATE_PER_SE * capf[None, :]                 # full-pool solo rate at each cell
        act = (p.sec_backlog_t > 0) | ((p.sl == LL) & (p.q > 1e-6))
        n_act = np.bincount(p.serv, act.astype(float), p.nc)
        idx = np.arange(p.n)
        # expected offered bits/s per UE (same formula as Plant.tick)
        cfg = p.cfg
        total = cfg.warmup_s + cfg.scored_s
        ramp = cfg.ramp[0] + (cfg.ramp[1] - cfg.ramp[0]) * min(now / max(total, 1e-9), 1.0)
        mult = p.m[p.site_of[p.serv]] * cfg.lf() * ramp
        rate = np.where(p.sl == EMBB, cfg.embb_files_per_s, cfg.be_files_per_s) * mult
        disk = np.zeros(p.n, bool)
        env_s = 0.0
        if p.scn is not None and cfg.scenario == "surge":
            rate = rate * p.scn.traffic_mult(now, p.pos)
            disk = p.scn.inside(now, p.pos)
            env_s = p.scn.envelope(now)
        off = np.where(p.sl == LL, p.ll_pps * ramp * C.LL_PKT_BYTES * 8, rate * C.EMBB_FILE_BYTES * 8)
        return {"now": now, "sinr": sinr, "R": R, "n_act": n_act, "serv": p.serv.copy(), "idx": idx,
                "off": off, "disk": disk, "env": env_s, "l3": p.l3.copy(), "hys": p.hys.copy(),
                "thp": np.where(p.sec_backlog_t > 0, p.sec_bits / np.maximum(p.sec_backlog_t, 1e-9), np.nan)}

    def _after_close(self, s):
        p = self.p
        idx, serv, R, n_act = s["idx"], s["serv"], s["R"], s["n_act"]
        em = p.sec_viol_embb
        cls = np.zeros(p.n, np.int8)
        rs = R[idx, serv]
        out = p.sec_out
        cls[em & out] = 1
        sl_lim = em & ~out & (rs < TARGET)
        cls[sl_lim] = 2
        cap = em & ~out & ~sl_lim
        for u in np.nonzero(cap)[0]:
            sv = serv[u]
            nb = np.array(p.lay.neighbours[sv])
            ok = R[u, nb] / (n_act[nb] + 1) >= TARGET
            if not ok.any():
                cls[u] = 5
                continue
            reach = s["l3"][u, nb] + 4.0 - s["hys"][sv] > s["l3"][u, sv]     # A3 fires at TS cio_max 4 dB
            cls[u] = 3 if (ok & reach).any() else 4
        self.ue_cls.append(cls)
        self.ue_serv.append(serv.astype(np.int16))
        self.ue_sinr.append(s["sinr"][idx, serv].astype(np.float16))
        self.ue_disk.append(s["disk"])
        self.ue_thp.append(s["thp"].astype(np.float32))
        mean_r = np.bincount(serv, rs, p.nc) / np.maximum(np.bincount(serv, minlength=p.nc), 1)
        self.cell_off.append(np.bincount(serv, s["off"], p.nc))
        self.cell_cap.append(mean_r)
        self.cell_act.append(n_act)
        self.cell_viol.append(np.bincount(serv, em.astype(float), p.nc))
        self.sec_t.append(s["now"])
        self.env_surge.append(s["env"])

    # ------------------------------------------------------------------------------------------------ save
    def save(self, stem, score, secs):
        p, env = self.p, self.env
        np.savez_compressed(
            os.path.join(OUT, stem + ".npz"), cls=np.array(self.ue_cls), serv=np.array(self.ue_serv),
            sinr=np.array(self.ue_sinr), disk=np.array(self.ue_disk), thp=np.array(self.ue_thp),
            cell_off=np.array(self.cell_off), cell_cap=np.array(self.cell_cap),
            cell_util=np.array(self.cell_util), cell_act=np.array(self.cell_act), cell_viol=np.array(self.cell_viol),
            t=np.array(self.sec_t), env=np.array(self.env_surge), sl=p.sl, is_macro=p.lay.is_macro,
            corridor=(p.scn.idx if p.cfg.scenario == "mistune" else np.array([], int)),
            cluster=(p.scn.cluster if p.cfg.scenario == "mistune" else np.array([], int)),
            push_t=(p.scn.push_t if p.cfg.scenario == "mistune" else -1.0),
            final_cio=p.cio, final_hys=p.hys, final_ttt=p.ttt)
        meta = {"score": score, "secs": secs, "neighbours": [list(map(int, n)) for n in p.lay.neighbours],
                "ho": self.ho, "rlf": self.rlf, "req": self.req, "ts_log": self.ts_log, "mro_log": self.mro_log,
                "kpm_lag": self.kpm_lag,
                "xapps": {x.name: {"scale": x.scale, "behaviour": x.behaviour} for x in env.xapps}}
        with open(os.path.join(OUT, stem + ".json"), "w", newline="\n") as f:
            json.dump(meta, f, default=float)


def run(scenario, load, seed, arm):
    env = make_env(scenario, load, seed)
    pr = Probe(env)
    t = time.time()
    score = env.run(arbiter(arm))
    secs = round(time.time() - t, 1)
    pr.save(f"{scenario}_{load}_{seed}_{arm}", score, secs)
    print(json.dumps({"scenario": scenario, "load": load, "seed": seed, "arm": arm, "svr": score["svr"],
                      "embb_viol": score["embb_viol"], "secs": secs}), flush=True)


def selfcheck():
    """Instrumented vs plain episode must score identically (read-only probe)."""
    a = make_env("surge", "medium", seed_of("surge", "medium", 9)).run(subset(("TS",)))
    env = make_env("surge", "medium", seed_of("surge", "medium", 9))
    Probe(env)
    b = env.run(subset(("TS",)))
    same = all(a[k] == b[k] for k in a)
    print("selfcheck identical:", same, a["svr"], b["svr"])


# ==================================================================================================== analysis
def load_ep(stem):
    z = np.load(os.path.join(OUT, stem + ".npz"))
    with open(os.path.join(OUT, stem + ".json")) as f:
        m = json.load(f)
    return z, m


def scored(z):
    return z["t"] > WARMUP


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def cls_split(z, mask_ue=None):
    c = z["cls"][scored(z)]
    if mask_ue is not None:
        c = np.where(mask_ue[scored(z)], c, 0)
    n = [(c == i).sum() for i in range(len(CLS))]
    tot = sum(n[1:])
    return int(tot), {CLS[i]: round(float(pct(n[i], tot)), 1) for i in range(1, len(CLS))}


def analyze_surge_base(eps):
    print("=" * 110)
    print("Q2  eMBB violation decomposition (FREEZE arm, scored window) -- class shares of violated eMBB UE-s")
    for (sc, ld, sd), arms in sorted(eps.items()):
        if sc == "mistune" or "freeze" not in arms:
            continue
        z, m = arms["freeze"]
        tot, sp = cls_split(z)
        line = f"{sc:7s} {ld:6s} {sd}: viol={tot:6d} {sp}"
        print(line)
        if sc == "surge":
            sm = scored(z) & (z["env"] >= 0.999)
            disk = z["disk"]
            ndisk = disk[sm].sum(1)
            tot_d, sp_d = cls_split(z, disk & (z["env"] >= 0.999)[:, None])
            tot_o, sp_o = cls_split(z, ~disk & (z["env"] >= 0.999)[:, None])
            ev = scored(z) & (z["env"] > 0)
            dv = ((z["cls"][ev] > 0) & disk[ev]).sum()
            print(f"    in-disk eMBB viol over the whole event {dv} = {pct(dv, m['score']['embb_viol']):.1f}% of episode "
                  f"eMBB viol, {pct(dv, m['score']['viol_frac'] * 600 * 300):.1f}% of all violated UE-s")
            print(f"    plateau ({sm.sum()} s, {ndisk.mean():.0f} UEs in disk): in-disk viol={tot_d} {sp_d}")
            print(f"    plateau outside-disk viol={tot_o} {sp_o}")
            # offered load vs capacity: cells serving >= 1 in-disk UE vs their neighbours
            serv = z["serv"][sm]
            off, cap, util, act = (z[k][sm] for k in ("cell_off", "cell_cap", "cell_util", "cell_act"))
            nc = off.shape[1]
            share = np.zeros(nc)
            for r in range(serv.shape[0]):
                share += np.bincount(serv[r][disk[sm][r]], minlength=nc)
            hot = np.argsort(-share)[:3]
            hot = [c for c in hot if share[c] > 0.1 * share.sum()]
            nbrs = sorted({n for c in hot for n in m["neighbours"][c]} - set(hot))
            with np.errstate(invalid="ignore", divide="ignore"):
                rho = off / np.where(cap > 0, cap, np.nan)
            hot = [int(c) for c in hot]
            print(f"    surge cells {hot} (share of disk UE-s {np.round(share[hot] / share.sum(), 2).tolist()}): "
                  f"offered/capacity rho={np.nanmean(rho[:, hot], 0).round(2).tolist()} "
                  f"util={util[:, hot].mean(0).round(2).tolist()} active={act[:, hot].mean(0).round(1).tolist()}")
            print(f"    their neighbours {nbrs}: rho mean={np.nanmean(rho[:, nbrs]):.2f} "
                  f"(min cell {np.nanmin(np.nanmean(rho[:, nbrs], 0)):.2f}) util mean={util[:, nbrs].mean():.2f} "
                  f"cells util<0.5: {int((util[:, nbrs].mean(0) < 0.5).sum())}/{len(nbrs)}")
            print(f"    offered in disk cells {off[:, hot].sum(1).mean() / 1e6:.1f} Mb/s vs capacity "
                  f"{cap[:, hot].sum(1).mean() / 1e6:.1f} Mb/s; spare nbr capacity "
                  f"{np.nansum(np.clip(cap[:, nbrs] - off[:, nbrs], 0, None), 1).mean() / 1e6:.1f} Mb/s")


def ts_chain(z, m, zf=None):
    """TS requests -> ACK/NACK -> realised CIO moves -> HOs on the pushed pair -> moved UEs' SINR / violations."""
    req = [r for r in m["req"] if r["xapp"] == "TS"]
    acc = [r for r in req if r["accepted"]]
    moved = [r for r in acc if abs(r["moved"]) > 1e-9]
    ups = [r for r in moved if r["moved"] > 0]
    downs = [r for r in moved if r["moved"] < 0]
    ts = m["ts_log"]
    nb = m["neighbours"]
    print(f"  TS requests={len(req)} accepted={len(acc)} NACK={len(req) - len(acc)} realised moves={len(moved)} "
          f"(+:{len(ups)} -:{len(downs)}); proposal cycles={len(ts)}, cycles with >=1 request="
          f"{sum(t['n_req'] > 0 for t in ts)}")
    # trigger conditions per (cycle, pair)
    n_pairs = hi = spare = gapok = suf = 0
    for t in ts:
        u = np.array(t["u"])
        p5 = np.array(t["p5"])
        for s in range(len(u)):
            sf = (p5[s] >= 0) and (p5[s] < TARGET * 1.2)
            for n in nb[s]:
                n_pairs += 1
                h = u[s] > t["u_hi"]
                hi += h
                suf += h and sf
                spare += h and sf and u[n] < t["u_tgt"]
                gapok += h and sf and u[n] < t["u_tgt"] and u[s] - u[n] > t["gap"]
    print(f"  trigger funnel over (cycle x pair)={n_pairs}: u_s>u_hi {hi} -> &suffer {suf} -> &u_n<u_tgt {spare} "
          f"-> &gap {gapok}")
    uu = np.array([t["u"] for t in ts]) if ts else np.zeros((0, 1))
    raw = np.array([t["raw"] for t in ts]) if ts else np.zeros((0, 1))
    if len(ts):
        util = z["cell_util"]
        # 10 s mean utilisation vs the single 1 s report TS samples
        tt = np.array([t["t"] for t in ts]).astype(int)
        m10 = np.array([util[max(0, k - 11):k - 1].mean(0) for k in tt])
        print(f"  TS utilisation input: EWMA u mean={uu.mean():.2f}; raw 1-s sample vs true 10-s mean: "
              f"mean |diff|={np.abs(raw - m10).mean():.3f}, corr={np.corrcoef(raw.ravel(), m10.ravel())[0, 1]:.2f}; "
              f"share of cell-cycles with u>0.6: {(uu > 0.6).mean():.2f}, u<0.5: {(uu < 0.5).mean():.2f}")
        lag = [t["t"] - t["rep_t1"] for t in ts]
        print(f"  TS report age at decision: fast mean {np.mean(lag):.2f} s (window 1 s); thp age "
              f"{np.mean([t['t'] - t['thp_t1'] for t in ts if t['thp_t1'] is not None]):.2f} s (window 5 s)")
    # HOs on pushed pairs, CIO-induced
    ho = m["ho"]
    cio_ho = [h for h in ho if h["cio"] > 0 and h["margin"] - h["hys"] <= 0 and not h["fail"]]
    ho_ok = [h for h in ho if not h["fail"]]
    print(f"  HOs total={len(ho)} (fail {sum(h['fail'] for h in ho)}), with CIO[s,t]>0 at trigger="
          f"{sum(h['cio'] > 0 for h in ho)}, CIO-induced (would not fire at CIO 0)={len(cio_ho)}; "
          f"HO target outside TS/MRO neighbour list: {pct(sum(not h['nbr'] for h in ho), len(ho)):.1f}%")
    if cio_ho:
        ds = np.array([h["sinr_t"] - h["sinr_src"] for h in cio_ho])
        st = np.array([h["sinr_t"] for h in cio_ho])
        print(f"    CIO-induced HO: SINR target {np.median(st):.1f} dB (median), change vs source "
              f"{np.median(ds):+.1f} dB (median), {pct((ds < 0).sum(), len(ds)):.0f}% worse; eMBB share "
              f"{pct(sum(h['sl'] == EMBB for h in cio_ho), len(cio_ho)):.0f}%; pingpong "
              f"{pct(sum(h['pingpong'] for h in cio_ho), len(cio_ho)):.0f}% (all HOs: "
              f"{pct(sum(h['pingpong'] for h in ho_ok), len(ho_ok)):.0f}%)")
        # moved eMBB UEs: violation seconds 10 s before vs 10 s after
        cls, t = z["cls"], z["t"]
        b = a = 0
        for h in cio_ho:
            if h["sl"] != EMBB:
                continue
            k = int(np.searchsorted(t, h["t"]))
            b += (cls[max(0, k - 10):k, h["u"]] > 0).sum()
            a += (cls[k + 1:k + 11, h["u"]] > 0).sum()
        print(f"    moved eMBB UEs: violated s in 10 s before={b} after={a}")
    # per realised up-move: HOs on the pair in the next 10 s, u of target at request
    moves = []
    for r in ups:
        k = tuple(r["knob"])
        s, n = k[1], k[2]
        cnt = sum(1 for h in ho if h["src"] == s and h["tgt"] == n and r["t"] < h["t"] <= r["t"] + 10
                  and not h["fail"])
        moves.append(cnt)
    if moves:
        mv = np.array(moves)
        print(f"  per realised +1 dB CIO(s->n) move: HOs s->n in next 10 s mean={mv.mean():.2f}, zero-HO moves="
              f"{pct((mv == 0).sum(), len(mv)):.0f}%")
    # oscillation: sign changes per knob
    seq = {}
    for r in moved:
        seq.setdefault(tuple(r["knob"]), []).append(np.sign(r["moved"]))
    flips = sum(sum(1 for a_, b_ in zip(v, v[1:], strict=False) if a_ != b_) for v in seq.values())
    print(f"  knobs touched={len(seq)}, direction reversals={flips}, final |CIO|>0 pairs="
          f"{int((np.abs(z['final_cio']) > 0).sum())}, max |CIO|={np.abs(z['final_cio']).max():.0f}")
    if zf is not None:
        sm = scored(z)
        dv = (z["cls"][sm] > 0).sum() - (zf["cls"][sm] > 0).sum()
        tf, spf = cls_split(zf)
        tt_, spt = cls_split(z)
        print(f"  paired TS - freeze: eMBB viol {tt_} vs {tf} (delta {dv:+d}); class counts TS/freeze: "
              + ", ".join(f"{CLS[i]} {((z['cls'][sm] == i).sum())}/{((zf['cls'][sm] == i).sum())}"
                          for i in range(1, len(CLS))))
        print(f"  paired: HO/pingpong TS {m['score']['ho_per_ue_h']:.1f}/{m['score']['pingpong']} vs freeze "
              f"{zf_m_score(zf)}")


_FREEZE_META = {}


def zf_m_score(zf):
    m = _FREEZE_META.get(id(zf))
    return f"{m['score']['ho_per_ue_h']:.1f}/{m['score']['pingpong']}" if m else "?"


def analyze_ts(eps):
    print("=" * 110)
    print("Q1  TS mechanism audit (TS-alone arm = subset(TS), paired with freeze on the same tape)")
    for (sc, ld, sd), arms in sorted(eps.items()):
        if sc == "mistune" or "TS" not in arms:
            continue
        z, m = arms["TS"]
        zf = arms.get("freeze", (None, None))[0]
        if zf is not None:
            _FREEZE_META[id(zf)] = arms["freeze"][1]
        print(f"-- {sc} {ld} {sd}: SVR TS {m['score']['svr']:.1f} vs freeze "
              f"{arms['freeze'][1]['score']['svr']:.1f}; TS scale={m['xapps']['TS']['scale']:.2f} "
              f"behaviour={m['xapps']['TS']['behaviour']}")
        ts_chain(z, m, zf)


def analyze_mro(eps):
    print("=" * 110)
    print("Q3  MRO in mistune (MRO-alone arm = subset(MRO), paired with freeze)")
    for (sc, ld, sd), arms in sorted(eps.items()):
        if sc != "mistune" or "MRO" not in arms:
            continue
        z, m = arms["MRO"]
        zf, mf = arms["freeze"]
        clu = set(z["cluster"].tolist())
        corr = set(z["corridor"].tolist())
        push = float(z["push_t"])
        req = [r for r in m["req"] if r["xapp"] == "MRO"]
        kinds = {}
        for r in req:
            k = r["knob"]
            typ = k[0] + ("+" if r["prop"] > r["cur"] else "-")
            where = "clu" if k[1] in clu else "oth"
            key = f"{typ}@{where}"
            kinds.setdefault(key, [0, 0])
            kinds[key][0] += 1
            kinds[key][1] += int(r["accepted"] and abs(r["moved"]) > 1e-9)
        print(f"-- mistune {ld} {sd}: cluster={sorted(clu)} push t={push:.0f}s; SVR MRO {m['score']['svr']:.1f} vs "
              f"freeze {mf['score']['svr']:.1f}; RLF/ue-h {m['score']['rlf_per_ue_h']:.2f} vs "
              f"{mf['score']['rlf_per_ue_h']:.2f}; HO/ue-h {m['score']['ho_per_ue_h']:.1f} vs "
              f"{mf['score']['ho_per_ue_h']:.1f}; pingpong {m['score']['pingpong']} vs {mf['score']['pingpong']}")
        print(f"   MRO requests by type [n, realised]: {dict(sorted(kinds.items()))}")
        fc = z["final_cio"]
        cl = sorted(clu)
        print(f"   final cluster Hys={z['final_hys'][cl].tolist()} TTT={z['final_ttt'][cl].tolist()} "
              f"max CIO out of cluster cells={fc[cl].max(1).tolist()}")
        for tag, mm in (("MRO", m), ("freeze", mf)):
            ho = [h for h in mm["ho"] if h["t"] > push]
            rl = [r for r in mm["rlf"] if r["t"] > push]
            hc = [h for h in ho if h["u"] in corr]
            ho_ = [h for h in ho if h["u"] not in corr]
            pc = sum(h["pingpong"] for h in hc)
            po = sum(h["pingpong"] for h in ho_)
            cioi = sum(1 for h in ho_ if h["cio"] > 0 and h["margin"] - h["hys"] <= 0)
            rk = {}
            for r in rl:
                rk[r["kind"] + ("/corr" if r["u"] in corr else "/oth")] = rk.get(
                    r["kind"] + ("/corr" if r["u"] in corr else "/oth"), 0) + 1
            print(f"   [{tag:6s}] post-push HO corridor={len(hc)} (pp {pc}) other={len(ho_)} (pp {po}, "
                  f"CIO-induced {cioi}); RLF {len(rl)} {dict(sorted(rk.items()))}")
        sm = scored(z) & (z["t"] > push)
        corr_m = np.zeros(z["cls"].shape[1], bool)
        corr_m[list(corr)] = True
        for tag, zz in (("MRO", z), ("freeze", zf)):
            c = zz["cls"][sm] > 0
            o = zz["cls"][sm] == 1
            print(f"   [{tag:6s}] post-push eMBB viol: corridor {c[:, corr_m].sum()} (outage {o[:, corr_m].sum()}), "
                  f"other {c[:, ~corr_m].sum()} (outage {o[:, ~corr_m].sum()}); all-slice SVR share of eMBB "
                  f"{pct(c.sum(), zz['cls'][sm].size):.1f}% of UE-s")
        pre = scored(z) & (z["t"] <= push)
        for tag, zz in (("MRO", z), ("freeze", zf)):
            o = zz["cls"] == 1
            lo = zz["sinr"][:, corr_m].astype(float) < C.SINR_MIN_DB
            print(f"   [{tag:6s}] corridor eMBB outage per corridor-eMBB-UE-s: pre-push "
                  f"{o[pre][:, corr_m].mean() / max((zz['sl'][corr_m] == EMBB).mean(), 1e-9):.3f} post-push "
                  f"{o[sm][:, corr_m].mean() / max((zz['sl'][corr_m] == EMBB).mean(), 1e-9):.3f}; corridor "
                  f"UE-s with serving SINR < -10 dB (end-of-second snapshot) pre {lo[pre].mean():.3f} post "
                  f"{lo[sm].mean():.3f}")
        emc = corr_m & (zf["sl"] == EMBB)
        o = zf["cls"] == 1
        inc = (o[sm][:, emc].mean() - o[pre][:, emc].mean()) * emc.sum() * sm.sum()
        tot = mf["score"]["viol_frac"] * zf["cls"].shape[1] * SCORED
        print(f"   [freeze] ceiling for a perfect mistune repair: push-induced corridor eMBB outage ~{inc:.0f} UE-s = "
              f"{pct(inc, tot):.1f}% of all violated UE-s (pre-push reference = only {pre.sum()} s); all corridor "
              f"eMBB viol post-push = {pct((zf['cls'][sm][:, corr_m] > 0).sum(), tot):.1f}%")
        ml = m["mro_log"]
        post = [x for x in ml if x["t"] > push]
        if post:
            stale = np.mean([max(0.0, 1.0 - (x["t"] - x["win_t0"] - 30) / 60) for x in post])
            print(f"   MRO cycles post-push={len(post)}; window {post[0]['win_t1'] - post[0]['win_t0']:.0f} s; "
                  f"too_late/cycle mean={np.mean([x['too_late'] for x in post]):.1f}, too_early+pp+wc/cycle="
                  f"{np.mean([x['too_early'] + x['pingpong'] + x['wrong_cell'] for x in post]):.1f}")
            _ = stale


def analyze_timing(eps):
    print("=" * 110)
    print("Q4  KPM timing (delivery lag = arrival - window end; decision age = decision t - window end)")
    lags = {}
    for arms in eps.values():
        for _z, m in arms.values():
            for g, t1, arr, now in m["kpm_lag"]:
                lags.setdefault(g, []).append((arr - t1, now - t1))
    for g, v in sorted(lags.items()):
        v = np.array(v)
        print(f"  {g:6s}: n={len(v)} lag mean {v[:, 0].mean():.2f} s [{v[:, 0].min():.2f}, {v[:, 0].max():.2f}], "
              f"handed to xApps at +{v[:, 1].mean():.2f} s after window end")


def analyze_extra(eps):
    print("=" * 110)
    print("X1  neighbour-list coverage (TS/MRO/knob registry act only on lay.neighbours = same site + 6 nearest by "
          "SITE distance)")
    for (sc, ld, sd), arms in sorted(eps.items()):
        for arm, (_z, m) in sorted(arms.items()):
            nb = m["neighbours"]
            ho = m["ho"]
            tl = [r for r in m["rlf"] if r["kind"] == "too_late"]
            tl_in = sum(r["best"] in nb[r["cur"]] for r in tl)
            # the argmax-l3 pair actually used by the counters: _rlf records too_late[cur, best]
            print(f"  {sc:7s} {ld:6s} {sd} {arm:6s}: HOs to a listed neighbour {pct(sum(h['nbr'] for h in ho), len(ho)):5.1f}% "
                  f"| too-late RLF pairs in list {tl_in}/{len(tl)}")
    print("X2  utilisation semantics: per-cell-second PRB utilisation vs 'any UE backlogged' (freeze arms)")
    for (sc, ld, sd), arms in sorted(eps.items()):
        if "freeze" not in arms:
            continue
        z = arms["freeze"][0]
        u, a = z["cell_util"][: len(z["t"])], z["cell_act"]
        busy = a > 0
        print(f"  {sc:7s} {ld:6s} {sd}: util | busy mean {u[busy].mean():.2f}, util | idle {u[~busy].mean():.2f}; "
              f"cell-seconds util>0.9 {pct((u > 0.9).sum(), u.size):.0f}% with mean active UEs "
              f"{a[u > 0.9].mean():.1f}; corr(util, active) {np.corrcoef(u.ravel(), a.ravel())[0, 1]:.2f}")
    print("X3  MRO repeated increments on the same knob within 60 s (stale 60 s window vs 30 s cadence)")
    for (_sc, ld, sd), arms in sorted(eps.items()):
        if "MRO" not in arms:
            continue
        rq = [r for r in arms["MRO"][1]["req"] if r["xapp"] == "MRO" and r["accepted"] and abs(r["moved"]) > 0]
        last, rep = {}, 0
        for r in rq:
            k = tuple(r["knob"])
            if k in last and r["t"] - last[k][0] <= 60 and np.sign(r["moved"]) == last[k][1]:
                rep += 1
            last[k] = (r["t"], np.sign(r["moved"]))
        print(f"  mistune {ld:6s} {sd}: realised MRO changes {len(rq)}, same-direction repeat within 60 s {rep}")


def geometry_surge():
    """Plant construction only (no episode): border-UE availability of the surge disk from the gain maps. Static
    (indoor) UEs only, whose position never changes. Assumes every cell fully loaded (occ = 1) for SINR, the
    regime the surge cells are in (util ~1)."""
    print("=" * 110)
    print("X4  surge disk geometry at construction (static UEs in the disk; plant built, no ticks)")
    for ld in ("medium", "high"):
        for j in (0, 1):
            sd = seed_of("surge", ld, j)
            cfg = C.E6Config(seed=sd, load=ld, mobility="mixed", mix="M4", warmup_s=WARMUP, scored_s=SCORED,
                             scenario="surge")
            p = Plant(cfg)
            disk = p.scn.inside(p.scn.t0 + 1e-6, p.pos) & (p.speed == 0) & (p.sl != LL)
            g = p._gains()[disk]
            o = np.sort(g, 1)
            best = np.argmax(g, 1)
            marg = o[:, -1] - o[:, -2]
            pw = 10 ** (g / 10)
            tot = pw.sum(1)
            s1 = 10 * np.log10(pw.max(1) / (tot - pw.max(1) + NOISE_W * 1000))
            s2 = 10 * np.log10(np.sort(pw, 1)[:, -2] / (tot - np.sort(pw, 1)[:, -2] + NOISE_W * 1000))
            listed = np.array([np.argsort(-gg)[1] in p.lay.neighbours[b] for gg, b in zip(g, best, strict=True)])
            print(f"  surge {ld:6s} {sd}: static non-LL UEs in disk {disk.sum()}; serving cells "
                  f"{np.unique(best).size}; margin to 2nd-best <2 dB {pct((marg < 2).sum(), len(marg)):.0f}% "
                  f"(<4 dB {pct((marg < 4).sum(), len(marg)):.0f}%); SINR@best median {np.median(s1):.1f} dB, "
                  f"@2nd-best {np.median(s2):.1f} dB (full load); 2nd-best cell in neighbour list "
                  f"{pct(listed.sum(), len(listed)):.0f}%")


def analyze():
    eps = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*.npz"))):
        stem = os.path.basename(f)[:-4]
        sc, ld, sd, arm = stem.split("_")
        eps.setdefault((sc, ld, int(sd)), {})[arm] = load_ep(stem)
    print("episodes:", {f"{k[0]}/{k[1]}/{k[2]}": sorted(v) for k, v in sorted(eps.items())})
    print("SVR/embb_viol per episode:")
    for k, arms in sorted(eps.items()):
        print("  ", k, {a: (round(m["score"]["svr"], 1), m["score"]["embb_viol"]) for a, (_z, m) in arms.items()})
    analyze_surge_base(eps)
    analyze_ts(eps)
    analyze_mro(eps)
    analyze_timing(eps)
    analyze_extra(eps)
    geometry_surge()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "run":
        sc, ld, sd = sys.argv[2], sys.argv[3], int(sys.argv[4])
        assert 140000 <= sd <= 140059, "diagnostic seeds only (registered 140000-140059)"
        for arm in sys.argv[5:]:
            run(sc, ld, sd, arm)
    elif cmd == "selfcheck":
        selfcheck()
    elif cmd == "analyze":
        analyze()
    else:
        raise SystemExit(__doc__)

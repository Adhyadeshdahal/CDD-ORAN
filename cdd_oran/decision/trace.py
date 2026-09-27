"""Versioned per-second propose/apply/outcome trace of an E6 episode (``E6Env(..., trace=True)``).

Schema ``e6-trace/1``. One row per control second t (t = 1 .. total_s). Arrays are columnar; the variable-length
tables (requests, rollbacks, writes, locks, config deltas, KPM reports) carry their own time column.

RUNTIME (deployable: what a near-RT RIC arbiter sees at second t, before/while it decides)
  t (T,) i4, scored (T,) bool         second index; scored = the second counts in the episode score (after warm-up)
  n_req (T,) i2                        requests offered this second (incl. re-offered deferred ones)
  n_locked (T,) i2                     WG3 locks active at decision time
  churn_cap (T,) i4                    churn cap in force for this second's apply (-1 = none)
  changes (T,) i4                      cumulative applied knob changes AFTER this second's apply (warm-up included)
  stats (T, S) i4                      cumulative env.stats after apply, columns = meta["stats"]
  cfg_snap_t (P,), cfg_snap (P, K) f8  full PRE-action configuration every ``snapshot_s`` s (first second included);
  cfg_d_t, cfg_d_k, cfg_d_v            knobs whose pre-action value changed since the previous second (any cause:
                                       RIC apply, OAM scenario rollout); knob index into meta["knobs"]
  rq_* (Q,)                            one row per offered request: t, xapp (index into meta["xapps"]), ver, knob,
                                       cur, prop, age (deferral age, 0 = fresh), dec (meta["dec_codes"]),
                                       dec_val (modify value / lock seconds, NaN otherwise),
                                       out (meta["out_codes"]: ok = ACK + changed, noop = ACK already in force,
                                       actuator / churn / locked = NACK by actuator limit / churn cap / active lock,
                                       reject / lock_set / expired = arbiter reject / arbiter lock / deferral > 10 s,
                                       deferred = parked), applied (value in force after the decision)
  rb_t, rb_k, rb_out                   rollback requests (meta["rb_codes"]: applied / noop / actuator / churn)
  wr_t, wr_k, wr_v, wr_out             arbiter writes attempted (meta["wr_codes"])
  lk_t, lk_k, lk_until                 WG3 locks set
  kpm_<gran>_{rx,t0,t1,arrived}        delivered KPM reports per granularity (rx = second delivered to the RIC)
  kpm_<gran>_<field> (R_g, ...) f4     report payload (per-cell vectors / matrices), fields in meta["kpm_fields"]
LABELS (privileged plant counters, prefix ``lab_``; NEVER features): row i = window (t[i] - 1, t[i]], i.e. the
second that just elapsed BEFORE the decisions of t[i]. Cells = the serving cell at the end of that second.
  lab_viol (T, C, 3) u2                violated UE-s per cell per slice (LL, eMBB, BE), recorded every second
  lab_outage (T, C) u2, lab_ue (T, C, 3) u2   outage UE-s; UEs served per cell per slice
  lab_rlf (T, C) u2, lab_energy_j (T, C) f4   radio link failures, energy per cell
  lab_severe (T, C, 3) u1              severe incidents started (cell-slice > 20 % violated for 10 s; scored only)
The episode score counts viol/outage/energy/severe over ``scored`` rows only (plant.sla["rlf"] counts all seconds).
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np

SCHEMA = "e6-trace/1"
DEC_CODES = ("accept", "reject", "modify", "defer", "lock")
OUT_CODES = ("ok", "noop", "actuator", "churn", "locked", "reject", "lock_set", "deferred", "expired")
ACK_CODES = ("ok", "noop")
RB_CODES = ("applied", "noop", "actuator", "churn")
WR_CODES = ("ok", "actuator", "churn")
LABEL_PREFIX = "lab_"
OUT = {c: i for i, c in enumerate(OUT_CODES)}
_ST = {"ok": 0, "nack": 1, "churn": 2}          # env._apply status -> index offset


def _dec_code(d):
    if isinstance(d, tuple):
        return DEC_CODES.index(d[0]), float(d[1])
    return DEC_CODES.index(d), np.nan


class Trace:
    """Columnar trace: ``arrays`` (name -> ndarray) + ``meta`` (JSON-able dict with ``schema``)."""

    def __init__(self, arrays: dict, meta: dict):
        self.arrays, self.meta = arrays, meta

    def __getitem__(self, k):
        return self.arrays[k]

    def __contains__(self, k):
        return k in self.arrays

    def features(self) -> dict:
        return {k: v for k, v in self.arrays.items() if not k.startswith(LABEL_PREFIX)}

    def labels(self) -> dict:
        return {k: v for k, v in self.arrays.items() if k.startswith(LABEL_PREFIX)}

    def to_npz(self, path) -> None:
        np.savez_compressed(path, __meta__=np.array(json.dumps(self.meta)), **self.arrays)

    @classmethod
    def from_npz(cls, path) -> Trace:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            if not str(meta.get("schema", "")).startswith("e6-trace/"):
                raise ValueError(f"not an E6 trace: schema {meta.get('schema')!r}")
            return cls({k: z[k] for k in z.files if k != "__meta__"}, meta)

    def row(self, t) -> int:
        return int(np.searchsorted(self.arrays["t"], t))

    def config_at(self, t) -> np.ndarray:
        """Pre-action configuration vector (meta["knobs"] order) at second t, rebuilt from snapshots + deltas."""
        a = self.arrays
        i = int(np.searchsorted(a["cfg_snap_t"], t, side="right")) - 1
        if i < 0:
            raise ValueError(f"no configuration snapshot at or before t={t}")
        v = a["cfg_snap"][i].copy()
        m = (a["cfg_d_t"] > a["cfg_snap_t"][i]) & (a["cfg_d_t"] <= t)
        v[a["cfg_d_k"][m]] = a["cfg_d_v"][m]           # deltas are time-ordered, so the latest write wins
        return v


def region_labels(trace: Trace, regions, t0: float, t1: float, scored_only: bool = False) -> dict:
    """Sum the per-cell labels over the rows with t0 < t <= t1 (window (t0, t1]) and group cells by ``regions``
    (cell -> region id, e.g. ``adapters.e6.region_map(env)``). Returns {"region_ids": (R,), name: (R, ...)} for
    viol (R, 3), outage, ue_s (R, 3), rlf, energy_j, severe (R, 3). PRIVILEGED (labels)."""
    a = trace.arrays
    reg = np.asarray(regions)
    ids = np.array(sorted({int(x) for x in reg}))
    m = (a["t"] > t0) & (a["t"] <= t1)
    if scored_only:
        m &= a["scored"]
    onehot = (reg[None, :] == ids[:, None]).astype(float)                     # (R, C)
    out = {"region_ids": ids}
    for name, key in (("viol", "lab_viol"), ("outage", "lab_outage"), ("ue_s", "lab_ue"), ("rlf", "lab_rlf"),
                      ("energy_j", "lab_energy_j"), ("severe", "lab_severe")):
        s = a[key][m].astype(float).sum(0)                                    # (C, ...)
        out[name] = np.tensordot(onehot, s, axes=(1, 0))
    return out


class TraceRecorder:
    """Attached to an E6Env when ``trace=True``; the env calls the hooks below. Never copied by ``env.copy()``."""

    def __init__(self, env, snapshot_s: int = 60):
        self.env, self.snapshot_s = env, int(snapshot_s)
        self.kidx = {k: i for i, k in enumerate(env.knobs)}
        self.xidx = {x.name: i for i, x in enumerate(env.xapps)}
        self.stat_keys = list(env.stats)
        self.rows = {k: [] for k in ("t", "scored", "n_req", "n_locked", "churn_cap", "changes", "stats")}
        self.lab = {k: [] for k in ("lab_viol", "lab_outage", "lab_ue", "lab_rlf", "lab_energy_j", "lab_severe")}
        self.snap_t, self.snap = [], []
        self.cd = ([], [], [])
        self.rq = {k: [] for k in ("t", "xapp", "ver", "knob", "cur", "prop", "age", "dec", "dec_val", "out",
                                   "applied")}
        self.rb, self.wr, self.lk = ([], [], []), ([], [], [], []), ([], [], [])
        self.kpm = {}                  # gran -> {"rx": [], "t0": [], "t1": [], "arrived": [], field: []}
        self.last_cfg = None
        self._scored = False

    # ------------------------------------------------------------------------------------------ hooks
    def plant_second(self):
        """After the second's ticks, BEFORE the KPM takes the counters: the second's labels."""
        p = self.env.plant
        nc, cell = p.nc, p.serv
        sl = p.sl
        viol = np.zeros((nc, 3), np.uint16)
        ue = np.zeros((nc, 3), np.uint16)
        for s_, flag in enumerate((p.sec_viol_ll, p.sec_viol_embb, p.sec_viol_be)):
            viol[:, s_] = np.bincount(cell[flag], minlength=nc)
            ue[:, s_] = np.bincount(cell[sl == s_], minlength=nc)
        self._scored = bool(p.sec_scored)
        L = self.lab
        L["lab_viol"].append(viol)
        L["lab_ue"].append(ue)
        L["lab_outage"].append(np.bincount(cell[p.sec_out], minlength=nc).astype(np.uint16))
        L["lab_rlf"].append(p.ctr["rlf"].astype(np.uint16))
        L["lab_energy_j"].append(p.ctr["energy_j"].astype(np.float32))
        L["lab_severe"].append(((p.cell_viol_run == 10) & self._scored).astype(np.uint8))

    def begin(self, now, new_reports, reqs):
        """Start of step_apply: pre-action configuration, locks, churn cap, delivered reports."""
        env = self.env
        t = int(now)
        cfg = np.array([v for v in env.config().values()], float)
        if self.last_cfg is None or (t - 1) % self.snapshot_s == 0:
            self.snap_t.append(t)
            self.snap.append(cfg)
        if self.last_cfg is not None:
            for k in np.nonzero(np.abs(cfg - self.last_cfg) > 1e-12)[0]:
                self.cd[0].append(t)
                self.cd[1].append(int(k))
                self.cd[2].append(float(cfg[k]))
        self.last_cfg = cfg
        R = self.rows
        R["t"].append(t)
        R["scored"].append(self._scored)
        R["n_req"].append(len(reqs))
        R["n_locked"].append(sum(1 for u in env.lock_until.values() if now < u))
        R["churn_cap"].append(-1 if env.churn_cap is None else int(env.churn_cap))
        for rep in new_reports:
            g = self.kpm.setdefault(rep["gran"], {"rx": [], "t0": [], "t1": [], "arrived": []})
            g["rx"].append(t)
            for k in ("t0", "t1", "arrived"):
                g[k].append(float(rep[k]))
            for k, v in rep.items():
                if k not in ("gran", "t0", "t1", "arrived"):
                    g.setdefault(k, []).append(np.asarray(v, np.float32))

    def request(self, now, r, d, out, applied, age):
        q = self.rq
        dc, dv = _dec_code(d)
        q["t"].append(int(now))
        q["xapp"].append(self.xidx[r["xapp"]])
        q["ver"].append(int(r["ver"]))
        q["knob"].append(self.kidx[r["knob"]])
        q["cur"].append(float(r["cur"]))
        q["prop"].append(float(r["prop"]))
        q["age"].append(int(age))
        q["dec"].append(dc)
        q["dec_val"].append(dv)
        q["out"].append(OUT[out])
        q["applied"].append(float(applied))

    def rollback(self, now, k, out):
        self.rb[0].append(int(now))
        self.rb[1].append(self.kidx[k])
        self.rb[2].append(RB_CODES.index(out))

    def write(self, now, k, v, out):
        for lst, x in zip(self.wr, (int(now), self.kidx[k], float(v), WR_CODES.index(out)), strict=True):
            lst.append(x)

    def lock(self, now, k, until):
        for lst, x in zip(self.lk, (int(now), self.kidx[k], float(until)), strict=True):
            lst.append(x)

    def end(self, now):
        self.rows["changes"].append(self.env.stats["changes"])
        self.rows["stats"].append([self.env.stats[k] for k in self.stat_keys])

    # ------------------------------------------------------------------------------------------ output
    def finish(self) -> Trace:
        env, R = self.env, self.rows
        a = {"t": np.array(R["t"], np.int32), "scored": np.array(R["scored"], bool),
             "n_req": np.array(R["n_req"], np.int16), "n_locked": np.array(R["n_locked"], np.int16),
             "churn_cap": np.array(R["churn_cap"], np.int32), "changes": np.array(R["changes"], np.int32),
             "stats": np.array(R["stats"], np.int32).reshape(-1, len(self.stat_keys)),
             "cfg_snap_t": np.array(self.snap_t, np.int32),
             "cfg_snap": np.array(self.snap, float).reshape(-1, len(env.knobs)),
             "cfg_d_t": np.array(self.cd[0], np.int32), "cfg_d_k": np.array(self.cd[1], np.int16),
             "cfg_d_v": np.array(self.cd[2], float)}
        q = self.rq
        for k, dt in (("t", np.int32), ("xapp", np.int8), ("ver", np.int8), ("knob", np.int16), ("cur", float),
                      ("prop", float), ("age", np.int8), ("dec", np.int8), ("dec_val", float), ("out", np.int8),
                      ("applied", float)):
            a["rq_" + k] = np.array(q[k], dt)
        for pre, cols, dts in (("rb_", ("t", "k", "out"), (np.int32, np.int16, np.int8)),
                               ("wr_", ("t", "k", "v", "out"), (np.int32, np.int16, float, np.int8)),
                               ("lk_", ("t", "k", "until"), (np.int32, np.int16, float))):
            src = {"rb_": self.rb, "wr_": self.wr, "lk_": self.lk}[pre]
            for c, lst, dt in zip(cols, src, dts, strict=True):
                a[pre + c] = np.array(lst, dt)
        kpm_fields = {}
        for g, d in self.kpm.items():
            kpm_fields[g] = [k for k in d if k not in ("rx", "t0", "t1", "arrived")]
            a[f"kpm_{g}_rx"] = np.array(d["rx"], np.int32)
            for k in ("t0", "t1", "arrived"):
                a[f"kpm_{g}_{k}"] = np.array(d[k], float)
            for k in kpm_fields[g]:
                a[f"kpm_{g}_{k}"] = np.stack(d[k])
        n = len(R["t"])
        for k, lst in self.lab.items():            # rows beyond the last apply (none in a full run) are dropped
            a[k] = np.stack(lst[:n])
        cfg = dataclasses.asdict(env.cfg)
        meta = {"schema": SCHEMA, "cfg": json.loads(json.dumps(cfg, default=str)), "wg3": bool(env.wg3),
                "knobs": [[k[0], *map(int, k[1:])] for k in env.knobs], "xapps": [x.name for x in env.xapps],
                "stats": self.stat_keys, "dec_codes": list(DEC_CODES), "out_codes": list(OUT_CODES),
                "ack_codes": list(ACK_CODES), "rb_codes": list(RB_CODES), "wr_codes": list(WR_CODES),
                "kpm_fields": kpm_fields, "snapshot_s": self.snapshot_s, "slices": ["LL", "eMBB", "BE"],
                "cell_region": [int(x) for x in env.plant.lay.cell_site],
                "label_prefix": LABEL_PREFIX,
                "label_note": "lab_* arrays are privileged plant counters: training/eval targets, never features"}
        return Trace(a, meta)

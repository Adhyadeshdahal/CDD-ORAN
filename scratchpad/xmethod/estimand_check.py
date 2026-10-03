"""Ruling R-36 estimand check: no truth-null PRIMARY candidate (action at t -> KPI at t+1) has a nonzero lag-1 TOTAL
effect in the SCM, for every world / regime (E4: every lambda of the sweep).

Why lag-1 total = what one SCM step does with the committed action. Row r of a dataset is (committed params A_t,
committed KPIs K_t, Y = K_{t+1}) and ``V2Env.advance`` computes K_{t+1} = f(A_t, K_t, exogenous tape at t) BEFORE it
commits the next action (E4 / E5 add the latent Z_t, drawn from its own tape slot, never from A_t). Nothing else
between A_t and K_{t+1} reads A_t, so do(A_t = a) moves K_{t+1} only through that one ``advance``. The tape is
coordinate-keyed, so a deep copy of the env advanced with a different committed A_t gets byte-identical exogenous
draws: every comparison is a common-random-numbers (unit-level) interventional contrast, and a total effect that is
exactly zero shows up as Delta == 0.0 at every unit.

Two checks, both on the frozen SCM classes driven exactly as the generator drives them:
  A. structural, whole ID range: S random states (params uniform on the ID ranges / E4 grid, KPIs drawn around a
     simulated trajectory, Z ~ N(0, 1)); for every action j and M values a', Delta_k = K_{t+1}(a') - K_{t+1}(A_t).
  B. interventional on each regime's own state distribution: ``generate_dataset(world, regime, n, seed)`` for DEV
     seeds 3_000_190 - 3_000_199 (kappa .25; observation noise is added after the SCM and cannot create effects),
     with the env's ``advance`` wrapped so that at every recorded row the same contrast is taken for M values a'
     drawn from the action's ID range (E4: the do-grid).
A truth-null candidate passes iff max |Delta| == 0.0 in A and in B. The placebo columns are never applied to the SCM
(generator), so their lag-1 total effect is zero by construction (listed, not simulated). Also reported: for every
TRUE edge the share of units with Delta != 0 and the mean slope Delta / (a' - a) (an edge whose effect is zero at
almost every unit is a recall ceiling, not an estimand error), and the same check for the SECONDARY lagged-KPI
sources (intervening on K_t), as a bonus.
Analytic (from the SCM code; what A and B verify numerically): E1 K0 = P0, K1 = P1, K2 = P2 + .5 K0_t,
K3 = P3 + .5 K1_t; E3 K0 = P0, K1 = K0_t, K2 = P1 + K1_t, K3 = P2 + K1_t, K4 = P3 (all on the previous row, so no
action reaches a KPI other than its own child within one step); E4 K0 = -A + 2.5 Z (one action); E2 / E5 nonlinear,
parents = the env adjacency (the same parent sets the update functions read).
Run: uv run python scratchpad/xmethod/estimand_check.py [N_ROWS]   (default 1000 rows x 10 seeds per cell)
Output: results/estimand_check.json.
"""
from __future__ import annotations

import contextlib
import copy
import json
import pathlib
import sys
import time

import numpy as np

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cdd_oran.envs.v2.base import V2Env  # noqa: E402
from cdd_oran.envs.v2.e4 import E4V2Env  # noqa: E402
from cdd_oran.envs.v2.e5 import E5V2Env  # noqa: E402
from cdd_oran.xmethod.worlds import generate as G  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results" / "estimand_check.json"
DEV_SEEDS = tuple(range(3_000_190, 3_000_200))
M_VALUES = 2                      # intervention values per unit and action
S_STATES = 3000                   # structural check states per world
KAPPA = 0.25
RNG_TAG = 9_300                   # this script's intervention-value stream (not a study seed)


def _env_for(world: str, regime: str, lam: float):
    """The SCM instance the generator uses for (world, regime) (module G)."""
    if world == "E4":
        if regime == "R3":
            return G.e4_logged.E4LoggedEnv(env_seed=0, lam=lam, episode=0)
        return E4V2Env(env_seed=0, lam=lam, mode="obs" if regime == "R4" else "do", episode=0)
    return G._WORLDS[world].env_cls(env_seed=0, episode=0)


def _values(rng: np.random.Generator, world: str, j: int, size: int) -> np.ndarray:
    wd = G._WORLDS[world]
    if wd.grid is not None:
        return wd.grid[rng.integers(0, len(wd.grid), size)]
    lo, hi = wd.ranges[j]
    return rng.uniform(lo, hi, size)


class Recorder:
    """Accumulates |Delta| statistics per (source, target)."""

    def __init__(self, world: str):
        cls = G._WORLDS[world].env_cls
        self.p, self.k = cls.num_params, cls.num_kpis
        self.maxabs = np.zeros((self.p + self.k, self.k))
        self.nonzero = np.zeros((self.p + self.k, self.k))
        self.slope_sum = np.zeros((self.p + self.k, self.k))
        self.units = np.zeros(self.p + self.k)

    def add(self, src: int, delta: np.ndarray, dv: float) -> None:
        self.maxabs[src] = np.maximum(self.maxabs[src], np.abs(delta))
        self.nonzero[src] += delta != 0.0
        if dv != 0.0:
            self.slope_sum[src] += delta / dv
        self.units[src] += 1


def contrast(env, rec: Recorder, world: str, rng: np.random.Generator, advance) -> None:
    """At env's current state: Delta of K_{t+1} for M_VALUES interventions on each committed action and each
    committed KPI (deep copies; byte-identical tape draws)."""
    base = advance(copy.deepcopy(env))
    for j in range(rec.p):
        for a in _values(rng, world, j, M_VALUES):
            c = copy.deepcopy(env)
            dv = float(a - c.prev_params[j])
            c.prev_params[j] = a
            rec.add(j, advance(c) - base, dv)
    for i in range(rec.k):
        for _ in range(M_VALUES):
            c = copy.deepcopy(env)
            sd = max(abs(float(c.prev_kpis[i])), 1.0)
            dv = float(rng.normal(0.0, sd))
            c.prev_kpis[i] = c.prev_kpis[i] + dv
            rec.add(rec.p + i, advance(c) - base, dv)


@contextlib.contextmanager
def wrapped_advance(rec: Recorder, world: str, rng: np.random.Generator, skip: int):
    """Wrap every SCM class's own ``advance`` so each call (after ``skip`` warm-up calls) first records the contrast.
    The wrapped call itself is unchanged (the dataset is byte-identical)."""
    originals = {c: c.__dict__["advance"] for c in (V2Env, E4V2Env, E5V2Env)}
    state = {"calls": 0}

    def make(orig):
        def adv(self):
            if state["calls"] >= skip:
                contrast(self, rec, world, rng, lambda e: np.asarray(orig(e), float).copy())
            state["calls"] += 1
            return orig(self)
        return adv
    try:
        for c, o in originals.items():
            c.advance = make(o)
        yield state
    finally:
        for c, o in originals.items():
            c.advance = o


def structural(world: str, regime: str, lam: float) -> Recorder:
    """Check A: random states over the whole ID range."""
    rng = np.random.default_rng([RNG_TAG, 1, G._WORLD_CODE[world], G._REGIME_CODE[regime], int(lam * 10)])
    ds, _ = G.generate_dataset(world, regime, 500, DEV_SEEDS[0], lam=lam, kappa=0.0)
    kpool = ds.X_kpi_lag
    rec = Recorder(world)
    env = _env_for(world, regime, lam)
    wd = G._WORLDS[world]
    for _ in range(S_STATES):
        env.time = int(rng.integers(5, 10_000))
        env.prev_params = np.array([_values(rng, world, j, 1)[0] for j in range(len(wd.ranges))], float)
        env.params = np.array([_values(rng, world, j, 1)[0] for j in range(len(wd.ranges))], float)
        env.prev_kpis = kpool[rng.integers(0, len(kpool))] * rng.uniform(0.5, 1.5, kpool.shape[1])
        if hasattr(env, "prev_Z"):
            env.prev_Z, env.Z = float(rng.normal()), float(rng.normal())
        if world == "E2":
            env._decoy_p0_ref = float(env.prev_params[0])
        adv = type(env).advance
        contrast(env, rec, world, rng, lambda e, adv=adv: np.asarray(adv(e), float).copy())
    return rec


def interventional(world: str, regime: str, lam: float, n: int) -> Recorder:
    """Check B: the regime's own trajectories, DEV seeds 3_000_190-199."""
    rec = Recorder(world)
    for seed in DEV_SEEDS:
        rng = np.random.default_rng([RNG_TAG, 2, G._WORLD_CODE[world], G._REGIME_CODE[regime], int(lam * 10), seed])
        ref, _ = G.generate_dataset(world, regime, n, seed, lam=lam, kappa=KAPPA)
        with wrapped_advance(rec, world, rng, skip=G.WARMUP) as st:
            ds, _ = G.generate_dataset(world, regime, n, seed, lam=lam, kappa=KAPPA)
        if st["calls"] != G.WARMUP + n or G.dataset_hash(ds) != G.dataset_hash(ref):
            raise AssertionError(f"{world} {regime}: wrapper changed the data or the row count")
    return rec


def summarize(world: str, regime: str, lam: float | None, recA: Recorder, recB: Recorder) -> dict:
    truth = G.truth_for(world, regime)
    p = recA.p
    names = [f"P{j}" for j in range(p)] + [f"K{i}" for i in range(recA.k)]
    out = {"world": world, "regime": regime, "lam": lam, "units_B": int(recB.units[0]),
           "primary_null": {}, "primary_true": {}, "secondary_null": {}, "secondary_true": {}}
    for s, nm in enumerate(names):
        for k in range(recA.k):
            e = (nm, f"K{k}")
            fam = ("primary" if s < p else "secondary") + ("_true" if e in truth.edges else "_null")
            out[fam][f"{e[0]}->{e[1]}"] = {
                "maxabs_A": float(recA.maxabs[s, k]), "maxabs_B": float(recB.maxabs[s, k]),
                "share_nonzero_B": float(recB.nonzero[s, k] / max(recB.units[s], 1)),
                "mean_slope_B": float(recB.slope_sum[s, k] / max(recB.units[s], 1))}
    placebos = sorted({c[0] for c in G.candidates_for(world, regime)} - set(names) - set(f"K{i}" for i in range(recA.k)))
    out["placebo_by_construction"] = placebos
    bad = [e for fam in ("primary_null",) for e, v in out[fam].items() if v["maxabs_A"] != 0 or v["maxabs_B"] != 0]
    bad2 = [e for e, v in out["secondary_null"].items() if v["maxabs_A"] != 0 or v["maxabs_B"] != 0]
    weak = [e for e, v in out["primary_true"].items() if v["share_nonzero_B"] < 0.5]
    dead = [e for fam in ("primary_true", "secondary_true") for e, v in out[fam].items()
            if v["maxabs_A"] == 0 or v["maxabs_B"] == 0]          # sensitivity: every true edge must move K_{t+1}
    out.update(pass_primary=not bad, violations_primary=bad, pass_secondary=not bad2, violations_secondary=bad2,
               true_edges_effect_lt_half_units=weak, true_edges_zero_effect=dead)
    return out


def main(n: int) -> None:
    cells = [(w, r, lam) for w, rs in G.REGIMES_OF.items() for r in rs
             for lam in (G.E4_LAMBDAS if (w == "E4" and r in ("R3", "R4")) else (1.0,))]
    rows = []
    for w, r, lam in cells:
        t0 = time.process_time()
        recA, recB = structural(w, r, lam), interventional(w, r, lam, n)
        row = summarize(w, r, lam if w == "E4" else None, recA, recB)
        row["cpu_s"] = round(time.process_time() - t0, 1)
        rows.append(row)
        print(f"{w} {r} lam={lam:<4} primary {'PASS' if row['pass_primary'] else 'FAIL ' + str(row['violations_primary'])}"
              f" | secondary {'PASS' if row['pass_secondary'] else 'FAIL ' + str(row['violations_secondary'])}"
              f" | true edges with effect at < half the units: {row['true_edges_effect_lt_half_units']}"
              f" | true edges with zero effect: {row['true_edges_zero_effect']}"
              f" ({row['cpu_s']} cpu-s)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_rows_per_seed": n, "seeds": list(DEV_SEEDS), "m_values": M_VALUES,
                               "s_states": S_STATES, "kappa_B": KAPPA, "all_pass_primary": all(x["pass_primary"]
                                                                                               for x in rows),
                               "cells": rows}, indent=1), encoding="utf-8")
    print("all primary PASS:", all(x["pass_primary"] for x in rows), "| wrote", OUT)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)

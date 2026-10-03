"""Dataset generation for the cross-method study: ``generate_dataset(world, regime, n, seed) -> (Dataset, Truth)``.

Contract: ``scratchpad/xmethod/CONTRACT.md`` sections 2 and 6; interface ``cdd_oran/xmethod/api.py``.

Row layout (the e1slice / e2slice alignment, SEMANTICS one-step actuation latency). One trajectory per
(world, seed), ``env_seed = seed``, ``episode = 0``. After ``WARMUP`` priming advances, row ``r`` records the
COMMITTED state at env time ``t`` and the KPIs one ``advance`` later::

    X_action[r]  = committed params at t        (the design's draw that is live at t)
    X_kpi_lag[r] = committed KPIs at t
    Y[r]         = KPIs at t+1 = f(X_action[r], X_kpi_lag[r], latent_t)

so ``Y[r] == X_kpi_lag[r+1]`` (asserted). The SCMs are the frozen ``cdd_oran/envs/v2`` classes (and, for R3,
the new ``e4_logged.E4LoggedEnv``), driven only through ``apply_action`` / ``advance``.

Regimes (``api.Design.kind``):
  R1 ``iid``     every action column redrawn i.i.d. each row (uniform on its ID range; E4: uniform on the
                 frozen do-grid V, i.e. the frozen randomised do-stream distribution).
  R2 ``dither``  Stage 0 F-dither (``docs/benchmark/STAGE0_FALSIFIERS.md``, primary delta = 0.10): per column,
                 20 setpoints drawn once per corpus uniformly within +/-25% of the range around the midpoint,
                 each held for n/20 consecutive rows (``fixed_part``), plus i.i.d. U(+/-delta*range) dither
                 (``random_part``). The envelope + dither never leaves the ID range, so no clipping occurs
                 (asserted) and column == fixed_part + random_part exactly.
  R3 ``logged``  E4 only: discretised frozen behaviour policy on the OBSERVED confounder Z (``context``);
                 redrawable categorical design: ``propensity`` = [n, 101] table P(A_i = V[k] | Z_i), each action
                 column drawn independently of the others given Z (``e4_logged``); the realised-action
                 probabilities are also in ``meta["realised_propensity"]``.
  R4 ``none``    E4 only: the frozen E4 observational mode (latent Z, nothing logged).

Candidates (ruling R-6): ``Dataset.candidates`` = every action (incl. placebo) x KPI followed by every lagged
KPI x KPI; ``meta["primary_candidates"]`` = the action -> KPI part (the primary set), ``meta[
"secondary_candidates"]`` = the KPI -> KPI part.

Every dataset carries the TUNING placebo column ``P_placebo`` (after the real actions), never applied to the SCM,
drawn from its own RNG stream and independent of everything (ruling R-10): R1 / R2 use P0's design (its own
setpoints under R2); R3 / R4 draw it i.i.d. from P0's MARGINAL distribution (W ~ N(.5, lam^2 + .25), rounded to V
under R3, clipped under R4; Design kind "iid" with that dist). E4 R3 / R4 add the diagnostic column
``P_placebo_conf`` (last), drawn by the confounded Z policy with its own eta (R3: logged table; R4: "none"); its
edges are nulls in the truth, are listed in ``meta["diagnostic_candidates"]`` and are outside the primary set.
All non-placebo arrays are byte-identical with or without the placebo columns.

E5 = the fully composed ``E5V2Env`` (subdom 0.2, chain 0.2, theta -2.5; ``mode="do"`` so lambda does not act).

Observation noise (ruling R-24, every world; the ``E5_NOISE_DETECT.md`` model): the OBSERVED KPI series is
``y_t,k + eps_t,k``, ``eps_t,k ~ N(0, (kappa*sigma_k)^2)`` i.i.d. per env time and KPI, so the same draw is used
where a KPI value appears as a lag (``X_kpi_lag``) and as a target (``Y``); actions, designs and truth are unchanged.
``sigma_k`` = the world's frozen noiseless KPI spread ``NOISE_SIGMA`` (E5: ``E5_KPI_MOMENTS``; E1-E4: pooled sd of
noiseless R1 data on DEV seeds 3_000_000-3_000_019 at n 24_000, ``scratchpad/xmethod/noise_sigma.py``). RNG:
``SeedSequence([7800, seed, world, regime, 5 "obs_noise"])``; the standard-normal draws do not depend on kappa (common
random numbers across kappa). ``kappa=None`` = the world default ``KAPPA_DEFAULT`` (E1-E4 0 = noiseless, E5 .3 =
its earlier J7 noise), byte-identical to the output before R-24; an explicit kappa (0 included) applies to every
world identically, E5 included (one noise layer, never two). ``meta["obs_noise"] = {kappa, sigma}`` when kappa > 0.

Truth (``api.Truth``) comes from the SCM definitions: the env adjacency lists give the true direct edges, every
other candidate is an exact null (the KPI update functions do not read it), and signs are set only where the
mechanism is monotone in that parent over the whole ID range (table ``_SIGNS``; checked by finite differences in
``tests/test_xmethod_harness.py``).
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np

from cdd_oran.envs.v2.e1 import E1V2Env
from cdd_oran.envs.v2.e2 import E2V2Env
from cdd_oran.envs.v2.e3 import E3V2Env
from cdd_oran.envs.v2.e4 import E4V2Env
from cdd_oran.envs.v2.e5 import E5V2Env
from cdd_oran.xmethod import api
from cdd_oran.xmethod.worlds import e4_logged

GENERATOR_VERSION = "xm-worlds/3"      # /2: ruling R-10 placebo split; /3: meta["roles"] removed (F7 finding 7)
WARMUP = 4                       # priming advances (flushes reset draws through E3's depth-3 cascade)
STREAM_TAG = 7800                # harness data streams: SeedSequence([7800, seed, world, regime, purpose])
DITHER_DELTA = 0.10              # Stage 0 F-dither primary level
DITHER_BLOCKS = 20
DITHER_ENVELOPE = 0.25
KAPPA_E5 = 0.3                   # E5 observation noise, E5_NOISE_DETECT.md (moderate level): E5's default kappa
E5_SIGMA = (4.634914, 24.662764, 57.680719, 20.192141)   # frozen E5 standardisation spreads (GATE_CONTRACT_E5)
# R-24 noise scales sigma_k (frozen 2026-10-02 from scratchpad/xmethod/noise_sigma.py, results/noise_sigma.json:
# pooled sd of noiseless R1 Y, DEV seeds 3_000_000-3_000_019, n 24_000; within 0.5% of the env reference moments).
NOISE_SIGMA: dict[str, tuple[float, ...]] = {
    "E1": (0.288709, 0.288894, 0.322411, 0.322784),
    "E2": (27.672172, 34.66717, 44.7979, 32.491503, 40.552988, 4.791992),
    "E3": (0.288581, 0.288582, 0.407729, 0.407559, 0.288733),
    "E4": (2.515616,),
    "E5": E5_SIGMA,
}
KAPPA_DEFAULT = {"E1": 0.0, "E2": 0.0, "E3": 0.0, "E4": 0.0, "E5": KAPPA_E5}   # kappa=None (pre-R-24 output)
E4_LAMBDAS = (0.0, 0.5, 1.0, 1.5)
PLACEBO = "P_placebo"                # tuning placebo, independent of everything (ruling R-10)
PLACEBO_CONF = "P_placebo_conf"      # E4 R3/R4 diagnostic: follows the confounded Z policy, never tuned on

_WORLD_CODE = {"E1": 1, "E2": 2, "E3": 3, "E4": 4, "E5": 5}
_REGIME_CODE = {"R1": 1, "R2": 2, "R3": 3, "R4": 4}
_PURPOSE = {"actions": 1, "setpoints": 2, "placebo": 3, "placebo_setpoints": 4, "obs_noise": 5,
            "placebo_conf": 6}

REGIMES_OF = {"E1": ("R1", "R2"), "E2": ("R1", "R2"), "E3": ("R1", "R2"), "E4": ("R1", "R2", "R3", "R4"),
              "E5": ("R1", "R2")}

# Monotone-parent signs (+1 / -1), by world, as (source, target) names. Everything else is unsigned.
_SIGNS: dict[str, dict[tuple[str, str], int]] = {
    # K0 = P0, K1 = P1, K2 = P2 + 0.5 K0, K3 = P3 + 0.5 K1
    "E1": {("P0", "K0"): 1, ("P1", "K1"): 1, ("P2", "K2"): 1, ("P3", "K3"): 1, ("K0", "K2"): 1, ("K1", "K3"): 1},
    # Gaussian bumps are non-monotone over the ID ranges except the K5 width P6 in (1.5, 4):
    # K5 = -35 exp(-d^2 / (2 P6^2)) is non-increasing in P6.
    "E2": {("P6", "K5"): -1},
    # unit-coefficient linear cascade
    "E3": {("P0", "K0"): 1, ("K0", "K1"): 1, ("P1", "K2"): 1, ("K1", "K2"): 1, ("P2", "K3"): 1, ("K1", "K3"): 1,
           ("P3", "K4"): 1},
    # K_out = -A + 2.5 Z
    "E4": {("P0", "K0"): -1},
    # K_mid(K2) = P0; K_harm(K1) gains +0.2 P0 and +0.2 K_mid_prev inside the gate (0 outside): non-decreasing.
    # P0 -> K_ben (bump at 80), G1/G2 -> K_harm (bump + gate switch), P3 -> K_dist (bump at 0) are unsigned.
    "E5": {("P0", "K2"): 1, ("P0", "K1"): 1, ("K2", "K1"): 1},
}

# Variable roles (structure hints: "conduit", "gate axis", ...). HARNESS-PRIVATE (F7 audit finding 7): never put in a
# Dataset; read only by reports / the scorer side through ``roles_for``.
_ROLES: dict[str, dict[str, str]] = {
    "E1": {},
    "E2": {"P0": "shared knob"},
    "E3": {"K1": "conduit", "K4": "non-descendant control"},
    "E4": {"P0": "A", "K0": "K_out"},
    "E5": {"P0": "P0 shared knob", "P1": "G1 gate axis", "P2": "G2 gate axis", "P3": "P3 distractor",
           "K0": "K_ben", "K1": "K_harm", "K2": "K_mid conduit", "K3": "K_dist"},
}


@dataclasses.dataclass(frozen=True)
class _WorldDef:
    env_cls: type
    ranges: tuple[tuple[float, float], ...]
    grid: np.ndarray | None = None          # R1 categorical support (E4 do-grid)


_WORLDS: dict[str, _WorldDef] = {
    "E1": _WorldDef(E1V2Env, tuple(E1V2Env.id_ranges)),
    "E2": _WorldDef(E2V2Env, tuple(E2V2Env.id_ranges)),
    "E3": _WorldDef(E3V2Env, tuple(E3V2Env.id_ranges)),
    "E4": _WorldDef(E4V2Env, tuple(E4V2Env.id_ranges), grid=e4_logged.GRID),
    "E5": _WorldDef(E5V2Env, tuple(E5V2Env.id_ranges)),
}


def _rng(seed: int, world: str, regime: str, purpose: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(
        [STREAM_TAG, int(seed), _WORLD_CODE[world], _REGIME_CODE[regime], _PURPOSE[purpose]]))


# ------------------------------------------------------------------------------------------------ designs
def _iid_column_dist(lo: float, hi: float, grid: np.ndarray | None) -> dict[str, Any]:
    if grid is not None:
        return {"name": "categorical", "values": grid.tolist(), "p": [1.0 / len(grid)] * len(grid)}
    return {"name": "uniform", "lo": float(lo), "hi": float(hi)}


def _draw_iid(rng: np.random.Generator, T: int, ranges, grid) -> np.ndarray:
    if grid is not None:
        return grid[rng.integers(0, len(grid), size=(T, len(ranges)))]
    lo, hi = np.array(ranges, dtype=float).T
    return rng.uniform(lo, hi, size=(T, len(ranges)))


def _draw_dither(rng_sp: np.random.Generator, rng_d: np.random.Generator, T: int, n: int, ranges):
    """Stage 0 F-dither over T = WARMUP + n env steps; the block clock runs over the n recorded rows
    (warm-up steps use block 0). Returns (fixed [T, p], random [T, p], block [T])."""
    lo, hi = np.array(ranges, dtype=float).T
    mid, width = (lo + hi) / 2.0, hi - lo
    setpoints = rng_sp.uniform(mid - DITHER_ENVELOPE * width, mid + DITHER_ENVELOPE * width,
                               size=(DITHER_BLOCKS, len(ranges)))
    row = np.arange(T) - (WARMUP - 1)                 # env step i is live as row i - (WARMUP - 1)
    block = np.clip(np.clip(row, 0, None) * DITHER_BLOCKS // n, 0, DITHER_BLOCKS - 1)
    fixed = setpoints[block]
    rand = rng_d.uniform(-DITHER_DELTA * width, DITHER_DELTA * width, size=(T, len(ranges)))
    val = fixed + rand
    if np.any(val < lo) or np.any(val > hi):           # impossible for envelope + delta < 0.5
        raise AssertionError("F-dither left the ID range")
    return fixed, rand, block


# ------------------------------------------------------------------------------------------------ roller
def _roll(env, inject: Callable[[int], None], n: int, warmup: int = WARMUP, latent_z: bool = False):
    """Drive ``env``: ``warmup`` priming steps, then ``n`` recorded rows (see module docstring).

    ``inject(i)`` applies the design's draw for env step ``i`` (row ``r`` is live after step ``warmup - 1 + r``).
    Returns committed params [n, p], committed KPIs [n, k], next KPIs [n, k], env times [n], committed Z [n].
    """
    for i in range(warmup):
        inject(i)
        env.advance()
    xa, xk, y, t, z = [], [], [], [], []
    for r in range(n):
        xa.append(env.prev_params.copy())
        xk.append(env.prev_kpis.copy())
        t.append(env.time)
        if latent_z:
            z.append(env.prev_Z)
        inject(warmup + r)
        y.append(env.advance().copy())
    xa, xk, y = np.array(xa, dtype=float), np.array(xk, dtype=float), np.array(y, dtype=float)
    if not np.array_equal(y[:-1], xk[1:]):
        raise AssertionError("row alignment broken: Y[r] != X_kpi_lag[r+1]")
    return xa, xk, y, np.array(t, dtype=np.int64), (np.array(z, dtype=float) if latent_z else None)


# ------------------------------------------------------------------------------------------------ public
def candidates_for(world: str, regime: str = "R1") -> tuple[tuple[str, str], ...]:
    """(source, target) candidates: every action (incl. ``P_placebo``, and ``P_placebo_conf`` in E4 R3 / R4) x KPI,
    then every lagged KPI x KPI."""
    wd = _WORLDS[world]
    acts = [f"P{i}" for i in range(wd.env_cls.num_params)] + [PLACEBO] + (
        [PLACEBO_CONF] if regime in ("R3", "R4") else [])
    kpis = [f"K{j}" for j in range(wd.env_cls.num_kpis)]
    return tuple((a, k) for a in acts for k in kpis) + tuple((s, k) for s in kpis for k in kpis)


def roles_for(world: str) -> dict[str, str]:
    """Harness-private variable roles of a world (for reports; never exposed to a method through the Dataset)."""
    return dict(_ROLES[world])


def truth_for(world: str, regime: str) -> api.Truth:
    """Ground truth from the SCM definitions (same for every regime: the structure does not change)."""
    cls = _WORLDS[world].env_cls
    if world == "E2":
        adj = E2V2Env._TRUE_ADJACENCY                  # decoy off
    elif world == "E3":
        adj = E3V2Env.TRUE_ADJACENCY
    elif world == "E4":
        adj = E4V2Env.TRUE_ADJACENCY
    elif world == "E5":
        adj = E5V2Env().adjacency_edges                 # default = fully composed (subdom, chain on)
    else:
        adj = E1V2Env.adjacency_edges
    p = cls.num_params
    edges = frozenset((f"P{s}" if s < p else f"K{s - p}", f"K{k}") for k, s in adj)
    cands = set(candidates_for(world, regime))
    if not edges <= cands:
        raise AssertionError(f"{world}: true edges outside the candidate set")
    signs = {e: s for e, s in _SIGNS[world].items()}
    if not set(signs) <= edges:
        raise AssertionError(f"{world}: sign table names a non-edge")
    return api.Truth(world=world, regime=regime, edges=edges, signs=signs, null_edges=frozenset(cands - edges))


def generate_dataset(world: str, regime: str, n: int, seed: int, lam: float = 1.0, kappa: float | None = None,
                     *, _with_placebo: bool = True) -> tuple[api.Dataset, api.Truth]:
    """One (world, regime, n, seed) corpus and its truth. ``lam`` (E4 only) is the Z -> A_behaviour coupling
    of the frozen sweep {0, .5, 1, 1.5}; it changes the data only under R3 / R4 (R1 / R2 never use the
    behaviour policy). A pure function of its arguments (byte-identical regeneration)."""
    if world not in _WORLDS:
        raise ValueError(f"unknown world {world!r}")
    if regime not in REGIMES_OF[world]:
        raise ValueError(f"regime {regime} is not defined for {world} (allowed {REGIMES_OF[world]})")
    if n < 2:
        raise ValueError("n must be >= 2")
    if world == "E4" and float(lam) not in E4_LAMBDAS:
        raise ValueError(f"E4 lambda must be one of {E4_LAMBDAS}, got {lam}")
    kap = KAPPA_DEFAULT[world] if kappa is None else float(kappa)
    if not (np.isfinite(kap) and kap >= 0.0):
        raise ValueError(f"kappa must be a finite number >= 0, got {kappa}")
    n, seed, wd = int(n), int(seed), _WORLDS[world]
    T = WARMUP + n
    p_real = wd.env_cls.num_params
    p0_range = (wd.ranges[0],)
    live = slice(WARMUP - 1, WARMUP - 1 + n)          # env-step draws that are live in rows 0..n-1
    meta: dict[str, Any] = {"generator": GENERATOR_VERSION, "warmup": WARMUP, "env_seed": seed, "episode": 0,
                            "placebo": PLACEBO}
    context = None

    if regime in ("R1", "R2"):
        if world == "E4":
            env = E4V2Env(env_seed=seed, lam=float(lam), mode="do", episode=0)
        else:
            env = wd.env_cls(env_seed=seed, episode=0)
        env.reset(episode=0)
        if regime == "R1":
            A = _draw_iid(_rng(seed, world, regime, "actions"), T, wd.ranges, wd.grid)
            Pp = _draw_iid(_rng(seed, world, regime, "placebo"), T, p0_range, wd.grid)
            dists = [_iid_column_dist(lo, hi, wd.grid) for lo, hi in wd.ranges]
            designs = [api.Design(kind="iid", dist=d) for d in dists]
            pl_design = api.Design(kind="iid", dist=_iid_column_dist(*wd.ranges[0], wd.grid))
        else:
            fx, rd, block = _draw_dither(_rng(seed, world, regime, "setpoints"),
                                         _rng(seed, world, regime, "actions"), T, n, wd.ranges)
            pfx, prd, _ = _draw_dither(_rng(seed, world, regime, "placebo_setpoints"),
                                       _rng(seed, world, regime, "placebo"), T, n, p0_range)
            A, Pp = fx + rd, pfx + prd
            designs = []
            for j, (lo, hi) in enumerate(wd.ranges):
                w = DITHER_DELTA * (hi - lo)
                designs.append(api.Design(kind="dither", dist={"name": "uniform", "lo": -w, "hi": w},
                                          random_part=rd[live, j].copy(), fixed_part=fx[live, j].copy()))
            w0 = DITHER_DELTA * (wd.ranges[0][1] - wd.ranges[0][0])
            pl_design = api.Design(kind="dither", dist={"name": "uniform", "lo": -w0, "hi": w0},
                                   random_part=prd[live, 0].copy(), fixed_part=pfx[live, 0].copy())
            meta.update(block=block[live].copy(), dither={"delta": DITHER_DELTA, "n_blocks": DITHER_BLOCKS,
                                                         "envelope": DITHER_ENVELOPE})

        def inject(i: int) -> None:
            for j in range(p_real):
                env.apply_action(j, float(A[i, j]))

        xa, xk, y, t, _ = _roll(env, inject, n, latent_z=world in ("E4", "E5"))
        if not np.array_equal(xa, A[live]):
            raise AssertionError("committed params differ from the design draws")
        placebo = Pp[live, 0].copy()
    else:                                              # E4 R3 / R4: actions follow the behaviour policy
        lam = float(lam)
        env = (e4_logged.E4LoggedEnv(env_seed=seed, lam=lam, episode=0) if regime == "R3"
               else E4V2Env(env_seed=seed, lam=lam, mode="obs", episode=0))
        env.reset(episode=0)
        # Ruling R-10. P_placebo: i.i.d. from P0's MARGINAL design, independent of everything (marginally
        # W = .5 + lam Z + eta ~ N(.5, lam^2 + .25)); P_placebo_conf: the confounded policy on the same Z with
        # its own eta (diagnostic only).
        s_marg = float(np.sqrt(lam ** 2 + e4_logged.ETA_SD ** 2))
        w_pl = _rng(seed, world, regime, "placebo").normal(0.5, s_marg, size=T)
        eta_c = _rng(seed, world, regime, "placebo_conf").normal(0.0, e4_logged.ETA_SD, size=T)
        applied, conf_vals, zs = np.empty(T), np.empty(T), np.empty(T)

        def inject(i: int) -> None:
            zs[i] = env.Z                                    # pending Z, committed with this action
            applied[i] = env.behavior_action()
            w = 0.5 + lam * env.Z + eta_c[i]
            conf_vals[i] = e4_logged.GRID[e4_logged.grid_index(w)] if regime == "R3" else float(np.clip(w, 0, 1))
            env.apply_action(0, applied[i])

        xa, xk, y, t, zc = _roll(env, inject, n, latent_z=True)
        if not (np.array_equal(xa[:, 0], applied[live]) and np.array_equal(zc, zs[live])):
            raise AssertionError("committed action / Z misaligned")
        conf = conf_vals[live].copy()
        if regime == "R3":
            placebo = e4_logged.GRID[e4_logged.grid_index(w_pl[live])]
            marg = e4_logged.policy_probs(np.zeros(1), 0.0, eta_sd=s_marg)[0]
            pl_design = api.Design(kind="iid", dist={"name": "categorical", "values": e4_logged.GRID.tolist(),
                                                     "p": marg.tolist()})
        else:
            placebo = np.clip(w_pl[live], 0.0, 1.0)
            pl_design = api.Design(kind="iid", dist={"name": "clipped_normal", "mean": 0.5, "sd": s_marg,
                                                     "lo": 0.0, "hi": 1.0})
        if regime == "R3":
            # Ruling (a): redrawable categorical design. propensity[i, k] = P(A_i = V[k] | Z_i); each action
            # column is drawn independently of the others given Z (ruling (b)).
            dist = {"name": "categorical_rows", "values": e4_logged.GRID.tolist(),
                    "policy": "V[floor(100*clip(0.5 + lam*Z + eta, 0, 1) + .5)], eta ~ N(0, sd^2)",
                    "lam": lam, "sd": e4_logged.ETA_SD}
            table = e4_logged.policy_probs(zc, lam)
            k_a, k_c = e4_logged.grid_index(xa[:, 0]), e4_logged.grid_index(conf)
            designs = [api.Design(kind="logged", dist=dist, propensity=table)]
            conf_design = api.Design(kind="logged", dist=dist, propensity=table.copy())
            meta["realised_propensity"] = {"P0": table[np.arange(n), k_a].copy(),
                                           PLACEBO_CONF: table[np.arange(n), k_c].copy()}
            context = zc.reshape(-1, 1).copy()
            meta["context_names"] = ("Z",)
        else:
            designs = [api.Design(kind="none")]
            conf_design = api.Design(kind="none")

    if kap > 0.0:                                       # R-24 observation noise on the KPI series
        kser = np.concatenate([xk, y[-1:]], axis=0)     # KPI at env times t_0 .. t_{n-1}+1
        sig = NOISE_SIGMA[world]
        eps = _rng(seed, world, regime, "obs_noise").standard_normal(kser.shape) * (kap * np.array(sig))
        kser = kser + eps
        xk, y = kser[:-1].copy(), kser[1:].copy()
        meta["obs_noise"] = {"kappa": kap, "sigma": list(sig)}
    if world == "E4":
        meta["lam"] = float(lam)

    action_names = tuple(f"P{i}" for i in range(p_real))
    kpi_names = tuple(f"K{j}" for j in range(wd.env_cls.num_kpis))
    extra = [(PLACEBO, placebo, pl_design)]
    if regime in ("R3", "R4"):
        extra.append((PLACEBO_CONF, conf, conf_design))
    if _with_placebo:
        X_action = np.concatenate([xa] + [c.reshape(-1, 1) for _, c, _ in extra], axis=1)
        action_names += tuple(nm for nm, _, _ in extra)
        designs = list(designs) + [d for _, _, d in extra]
        cands = candidates_for(world, regime)
    else:
        X_action = xa
        cands = tuple(c for c in candidates_for(world, regime) if c[0] not in (PLACEBO, PLACEBO_CONF))
    # Ruling R-6: primary = action -> KPI (incl. P_placebo); secondary = lagged KPI -> KPI. Ruling R-10:
    # P_placebo_conf edges are a separate diagnostic set, outside the primary family.
    meta["primary_candidates"] = tuple(c for c in cands if c[0] in action_names and c[0] != PLACEBO_CONF)
    meta["secondary_candidates"] = tuple(c for c in cands if c[0] in kpi_names)
    meta["diagnostic_candidates"] = tuple(c for c in cands if c[0] == PLACEBO_CONF)
    meta["placebo_conf"] = PLACEBO_CONF if (regime in ("R3", "R4") and _with_placebo) else None
    ds = api.Dataset(world=world, regime=regime, n=n, seed=seed, action_names=action_names, kpi_names=kpi_names,
                     X_action=X_action, X_kpi_lag=xk, Y=y, designs=tuple(designs), candidates=cands,
                     time_index=t, context=context, meta=meta)
    return ds, truth_for(world, regime)


def dataset_hash(ds: api.Dataset) -> str:
    """sha256 over every array and name of a Dataset (designs included); stable across regenerations."""
    h = hashlib.sha256()

    def arr(a) -> None:
        if a is None:
            h.update(b"<none>")
        else:
            a = np.ascontiguousarray(a)
            h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())

    h.update(repr((ds.world, ds.regime, ds.n, ds.seed, ds.action_names, ds.kpi_names, ds.candidates)).encode())
    for a in (ds.X_action, ds.X_kpi_lag, ds.Y, ds.time_index, ds.context):
        arr(a)
    for d in ds.designs:
        h.update(repr((d.kind, d.dist, d.redraw_seed_tag)).encode())
        for a in (d.random_part, d.fixed_part, d.propensity):
            arr(a)
    def val(k: str, v) -> None:
        h.update(k.encode())
        if isinstance(v, np.ndarray):
            arr(v)
        elif isinstance(v, dict):
            for kk in sorted(v):
                val(f"{k}.{kk}", v[kk])
        else:
            h.update(repr(v).encode())

    for k in sorted(ds.meta):
        val(k, ds.meta[k])
    return h.hexdigest()

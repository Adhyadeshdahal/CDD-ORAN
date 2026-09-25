"""MSCR: Max-Stratified Correlation-Ratio test for parameter->KPI edge discovery (v2 selection rule).

What it is
----------
For a target KPI ``y`` and candidate column ``i``, and for each OTHER column ``g`` (the conditioner):
split rows into ``nc`` equal-count strata of ``g`` (drop strata with fewer than ``min_stratum`` rows);
within each stratum, split rows into ``nb`` equal-count bins of candidate ``i``. The pooled conditional
correlation ratio is ``S_g = sum_s BSS_s / sum_s TSS_s`` (between-bin over total sum of squares of y),
and the test statistic is ``S* = max_g S_g``. The max over single conditioners is the "gate search":
it recovers co-parent-GATED edges (e.g. E2's P0->K5, E5's P0->K_harm) that kernel/distance CI tests
(RCoT, pdCor) dilute away.

Null and p-value: within each g-stratum, ``n_perm`` random equal-size partitions of y give null draws of
S_g. Under equal-count binning those draws depend on the candidate only through the bin sizes, so one
bank per (target, g) is shared across candidates. ``S*_null = max_{g != i}`` of the bank draws, and
``p = (1 + #{S*_null >= S*}) / (n_perm + 1)``. The shared bank is NOT a joint null across candidates or
across g. Only marginal p-values are used, and BY allows arbitrary dependence among them.

Selection (v2): per target, Benjamini-Yekutieli at ``q`` over the PARAMETER candidates only (the
intervention estimand for conflict mitigation). Lagged-KPI columns, if supplied, act as conditioners
but are never tested.

History: v1 (B=299, BH over all 14 E2 candidates) failed its preregistered post-selection FDR gate
(pooled 25/329; family-level 0.062, CI [0.041, 0.083], a warning, not a proved excess). v2 = B=2999
+ per-target BY over params, preregistered (``scratchpad/p0k5_fp_calibration/PREDECLARE_v2.md``) and
confirmed on 100 fresh E2 seeds (800000-800099): family-level FDR 0.0082 (one-sided 95% UB 0.0121),
P0->K5 recall 100/100, total param-edge recall 0.945.

SCOPE: binding, from the sol review of the v2 result. Read before using the output.
----------------------------------------------------------------------------------------
- VALIDATED ON RANDOMIZED DESIGNS ONLY: exogenous, independently randomized parameter candidates,
  noiseless E2, n=4000, the frozen constants below. The FDR evidence is empirical, finite-simulation
  evidence consistent with control under that design (BY's marginal-validity premise was checked
  empirically, not proved). There is NO general FDR-control guarantee and NO general
  conditional-independence claim for the max statistic.
- Known failure: single-parameter-actuation / low-diversity designs (E1: FDR 0.81). Not validated on
  observational, noisy, or confounded data, or on E5 (E5 needs its own calibration check plus fresh
  seeds). E5 CORE dev probe: P0->K_harm 7/10 at n=6000, 10/10 at n>=12000 (power-limited; fix n).
- Declares param->KPI ASSOCIATION edges. A declared edge alone does not identify an intervention
  effect, and there is no KPI->KPI discovery claim.
- Recall limitation: weak co-parent edges can be missed (E2 P6->K5 recall 12%). State this whenever
  recall or graph recovery is quoted.
- Statistic limits: one conditioner at a time (joint AND-gates are caught only when one axis isolates
  enough of the gate); mean-shift dependence only; coarse equal-count bins.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

MSCR_VERSION = "mscr-v2"


@dataclass(frozen=True)
class MSCRConfig:
    nc: int = 6               # equal-count strata of the conditioner g
    nb: int = 8               # equal-count bins of the candidate within a stratum
    min_stratum: int = 40     # drop strata with fewer rows
    n_perm: int = 2999        # permutations per (target, g) bank; p floor 1/(n_perm+1)
    q: float = 0.05           # BY level, per target, over the parameter family


def frozen_config() -> MSCRConfig:
    """The confirmed v2 constants (NC=6, NB=8, MIN=40, B=2999, BY q=0.05)."""
    return MSCRConfig()


@dataclass(frozen=True)
class MSCRResult:
    pvals: np.ndarray       # (n_targets, n_params) permutation p-values
    s_star: np.ndarray      # (n_targets, n_params) observed S*
    declared: np.ndarray    # (n_targets, n_params) bool, per-target BY selection
    config: MSCRConfig
    version: str = MSCR_VERSION

    def param_edges(self) -> set[tuple[int, int]]:
        """Declared edges as ``(kpi_index, param_index)``, the convention of ``_TRUE_ADJACENCY``."""
        return {(int(k), int(p)) for k, p in zip(*np.nonzero(self.declared), strict=True)}


def equal_count_labels(values: np.ndarray, nbins: int) -> np.ndarray:
    """Rank-based equal-count bin labels 0..nbins-1 (sizes differ by at most 1, stable ties)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return (ranks * nbins) // len(values)


def _between_ss(y: np.ndarray, labels: np.ndarray, nbins: int) -> float:
    total = y.sum()
    tb = np.bincount(labels, weights=y, minlength=nbins)
    nb = np.bincount(labels, minlength=nbins).astype(np.float64)
    mask = nb > 0
    return float(np.sum(tb[mask] ** 2 / nb[mask]) - total * total / len(y))


# ---------------------------------------------------------------------------------------------- runtime
# Execution choices (threads, device, block size) never change a result bit: see ``_build_banks``.
# Validated against the frozen v2 code by scratchpad/perf_mscr/equivalence_check.py.
_BLOCK_ELEMS = 1 << 18  # CPU bank block: (rows x ns) uniforms per task, 2 MiB of float64


def resolve_n_jobs(n_jobs: int | None = None) -> int:
    """Worker threads: explicit value > $MSCR_NUM_THREADS > $OMP_NUM_THREADS > usable CPU count."""
    if n_jobs is not None:
        return max(1, int(n_jobs))
    for var in ("MSCR_NUM_THREADS", "OMP_NUM_THREADS"):
        val = os.environ.get(var, "").strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
    counter = getattr(os, "process_cpu_count", None) or os.cpu_count
    return max(1, counter() or 1)


def resolve_device(device: str | None = None) -> str:
    """Bank device: explicit value > $MSCR_DEVICE > 'cuda' when torch sees a GPU that passes the
    bit-identity self-check > 'cpu' (numpy). 'auto' means the same as None."""
    if device is None or device == "auto":
        device = os.environ.get("MSCR_DEVICE", "auto").strip() or "auto"
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        from cdd_oran.discovery.mscr_torch import device_selfcheck

        if device_selfcheck("cuda"):
            return "cuda"
    return "cpu"


# ------------------------------------------------------------------------------------ permutation bank
def _strata_plan(x: np.ndarray, cfg: MSCRConfig) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Per conditioner g, the kept strata as (rows, bin sizes, bin offsets). Depends on x only."""
    plan = []
    for g in range(x.shape[1]):
        lab = equal_count_labels(x[:, g], cfg.nc)
        kept = []
        for s in range(cfg.nc):
            rows = np.nonzero(lab == s)[0]
            if len(rows) < cfg.min_stratum:
                continue
            ns = len(rows)
            sizes = np.bincount((np.arange(ns) * cfg.nb) // ns, minlength=cfg.nb).astype(np.int64)
            kept.append((rows, sizes, np.concatenate(([0], np.cumsum(sizes)[:-1]))))
        plan.append(kept)
    return plan


def _stream_tasks(plan, n_perm: int, block_elems: int) -> tuple[list[tuple[int, int, int, int, int]], int]:
    """Bank work in v2 stream order (g, kept stratum, row block) as (g, k, lo, hi, first draw index).

    v2 draws ``rng.random((n_perm, ns))`` per (g, kept stratum) in that order. ``random`` fills C-order
    with exactly one 64-bit output per double, so the block (rows lo..hi) of stratum (g, k) is the
    contiguous stream segment starting at the returned draw index; the total is the v2 consumption.
    """
    tasks, pos = [], 0
    for g, kept in enumerate(plan):
        for k, (rows, _, _) in enumerate(kept):
            ns = len(rows)
            step = max(1, block_elems // ns)
            for lo in range(0, n_perm, step):
                hi = min(lo + step, n_perm)
                tasks.append((g, k, lo, hi, pos))
                pos += (hi - lo) * ns
    return tasks, pos


def _jumpable(rng: np.random.Generator) -> bool:
    return type(rng.bit_generator) in (np.random.PCG64, np.random.PCG64DXSM)


def _uniforms_at(state: Any, bg_type, start: int, shape: tuple[int, int]) -> np.ndarray:
    """``shape`` uniforms from draw index ``start`` of the stream whose state is ``state`` (jump-ahead)."""
    bg = bg_type()
    bg.state = state
    bg.advance(start)
    return np.random.Generator(bg).random(shape)


def _finish_stream(rng: np.random.Generator, state: Any, total: int) -> None:
    """Leave ``rng`` exactly where sequential v2 draws would (advance() clears the 32-bit buffer, which
    ``random`` never touches, so restore it)."""
    bg = cast("np.random.PCG64", type(rng.bit_generator)())
    bg.state = state
    bg.advance(total)
    end = dict(bg.state)
    end["has_uint32"], end["uinteger"] = state["has_uint32"], state["uinteger"]
    rng.bit_generator.state = end


def _null_rows(y_s: np.ndarray, u: np.ndarray, offsets: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """sum_k (group sum_k)^2 / size_k for each row of uniforms ``u`` (one random partition per row)."""
    perm = np.argsort(u, axis=1)
    groupsums = np.add.reduceat(y_s[perm], offsets, axis=1)
    return np.sum(groupsums ** 2 / sizes[None, :], axis=1)


def _rowsums_cpu(ys, plan, cfg: MSCRConfig, rngs, n_jobs: int, block_elems: int):
    """rowsums[j][g][k] = the (n_perm,) per-draw sum_k gs^2/size for target j, numpy kernel.

    Serial: draws in stream order from each target's generator. Parallel: each task regenerates its own
    stream segment by PCG64 jump-ahead (no shared generator, no ordering constraint), so RNG, sort,
    gather and reduce all scale across threads (numpy releases the GIL in each); generators without
    jump-ahead draw sequentially on the caller and only the kernel runs in the pool.
    """
    tasks, total = _stream_tasks(plan, cfg.n_perm, block_elems)
    rowsums = [[[np.empty(cfg.n_perm) for _ in kept] for kept in plan] for _ in rngs]

    def run(j, t, u):
        g, k, lo, hi, _ = t
        _, sizes, offsets = plan[g][k]
        rowsums[j][g][k][lo:hi] = _null_rows(ys[j][g][k], u, offsets, sizes)

    def shape(t):
        return (t[3] - t[2], len(plan[t[0]][t[1]][0]))

    def run_jump(j, t, state, bgt):
        run(j, t, _uniforms_at(state, bgt, t[4], shape(t)))

    if n_jobs <= 1:
        for j, rng in enumerate(rngs):
            for t in tasks:
                run(j, t, rng.random(shape(t)))
        return rowsums

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(n_jobs) as pool:
        futs, window = [], deque()
        for j, rng in enumerate(rngs):  # all targets' tasks share one pool (no per-target tail idle)
            if _jumpable(rng):
                state, bgt = rng.bit_generator.state, type(rng.bit_generator)
                futs += [pool.submit(run_jump, j, t, state, bgt) for t in tasks]
                _finish_stream(rng, state, total)
                continue
            for t in tasks:  # no jump-ahead: draw on the caller, at most 2 * n_jobs blocks in flight
                window.append(pool.submit(run, j, t, rng.random(shape(t))))
                futs.append(window[-1])
                if len(window) >= 2 * n_jobs:
                    window.popleft().result()
        for f in futs:
            f.result()
    return rowsums


def _build_banks(
    x: np.ndarray,
    y: np.ndarray,
    cfg: MSCRConfig,
    rngs: list[np.random.Generator],
    n_jobs: int | None = None,
    device: str | None = None,
    block_elems: int | None = None,
):
    """Banks for every target column of ``y`` (target j uses ``rngs[j]``): a list of (denom, null, strata).

    Bit-identical to v2's per-target ``_build_bank`` for any n_jobs / device / block_elems: each target
    consumes its own stream exactly as v2 did (same draws, same order, see ``_stream_tasks``); every
    null draw is computed per row by the same argsort -> gather -> reduceat -> row-sum kernel (or its
    exact replica on torch, ``mscr_torch``); per-stratum contributions are accumulated in stratum order.
    """
    plan = _strata_plan(x, cfg)
    ys = [[[y[rows, j] for rows, _, _ in kept] for kept in plan] for j in range(y.shape[1])]
    dev = resolve_device(device)
    if dev == "cpu":
        rowsums = _rowsums_cpu(ys, plan, cfg, rngs, resolve_n_jobs(n_jobs), block_elems or _BLOCK_ELEMS)
    else:
        from cdd_oran.discovery.mscr_torch import device_selfcheck, rowsums_torch

        if not device_selfcheck(dev):
            raise RuntimeError(f"MSCR bank on {dev!r} is not bit-identical to the numpy bank; use device='cpu'")
        rowsums = rowsums_torch(ys, plan, cfg, rngs, dev, resolve_n_jobs(n_jobs), block_elems)
    banks = []
    for j in range(y.shape[1]):
        denom = np.zeros(x.shape[1])
        null = np.zeros((x.shape[1], cfg.n_perm))
        strata = []
        for g, kept in enumerate(plan):
            bss_null = np.zeros(cfg.n_perm)
            tss = 0.0
            info = []
            for k, (rows, _, _) in enumerate(kept):
                y_s = ys[j][g][k]
                ns = len(y_s)
                t_s = y_s.sum()
                tss += float(np.dot(y_s, y_s)) - t_s * t_s / ns
                bss_null += rowsums[j][g][k] - t_s * t_s / ns
                info.append((rows, y_s))
            denom[g] = tss
            if tss > 0:
                null[g] = bss_null / tss
            strata.append(info)
        banks.append((denom, null, strata))
    return banks


def _build_bank(x: np.ndarray, y: np.ndarray, cfg: MSCRConfig, rng: np.random.Generator,
                n_jobs: int | None = None, device: str | None = None, block_elems: int | None = None):
    """Per conditioner g: strata rows, pooled TSS, and n_perm shared null draws of S_g (one target)."""
    return _build_banks(x, np.asarray(y, dtype=np.float64).reshape(-1, 1), cfg, [rng],
                        n_jobs=n_jobs, device=device, block_elems=block_elems)[0]


def _s_star(col: np.ndarray, i: int, denom, strata, cfg: MSCRConfig) -> float:
    best = -np.inf
    for g, info in enumerate(strata):
        if g == i or denom[g] <= 0:
            continue
        bss = sum(_between_ss(y_s, equal_count_labels(col[rows], cfg.nb), cfg.nb) for rows, y_s in info)
        best = max(best, bss / denom[g])
    return best


def _pval(s_obs: float, i: int, null: np.ndarray) -> float:
    keep = np.ones(null.shape[0], dtype=bool)
    keep[i] = False
    s_null = np.max(null[keep], axis=0)
    return (1.0 + np.count_nonzero(s_null >= s_obs)) / (null.shape[1] + 1.0)


def by_declare(pvals, q: float) -> np.ndarray:
    """Benjamini-Yekutieli step-up at level q (valid under arbitrary dependence of valid p-values)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    c_m = sum(1.0 / k for k in range(1, m + 1))
    order = np.argsort(p, kind="stable")
    passed = p[order] <= np.arange(1, m + 1) * q / (m * c_m)
    declared = np.zeros(m, dtype=bool)
    if passed.any():
        declared[order[: np.nonzero(passed)[0].max() + 1]] = True
    return declared


def discover_mscr(
    x: np.ndarray,
    y: np.ndarray,
    n_params: int,
    seed: int,
    config: MSCRConfig | None = None,
    n_jobs: int | None = None,
    device: str | None = None,
) -> MSCRResult:
    """Run MSCR-v2 param->KPI discovery.

    x : (n, n_cols) candidate matrix whose FIRST ``n_params`` columns are the randomized parameters
        (the tested family); any further columns (e.g. lagged KPIs) are conditioners only.
    y : (n, n_targets) target KPIs.
    seed : dataset seed; the bank for target j uses ``default_rng([seed, j, n_perm])``, reproducing
        the v2 confirmatory run bit-for-bit.
    n_jobs, device : execution only, never the result. ``n_jobs`` = worker threads (default: see
        ``resolve_n_jobs``); ``device`` = 'cpu' (numpy), 'cuda[:k]' or any torch device for the bank
        (default: see ``resolve_device``). Every combination returns bit-identical output.
    """
    cfg = config or frozen_config()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row mismatch: x {x.shape} vs y {y.shape}")
    if not 1 <= n_params <= x.shape[1] or x.shape[1] < 2:
        raise ValueError(f"need 1 <= n_params <= n_cols and n_cols >= 2, got {n_params}, {x.shape[1]}")

    n_t = y.shape[1]
    jobs = resolve_n_jobs(n_jobs)
    rngs = [np.random.default_rng([int(seed), j, cfg.n_perm]) for j in range(n_t)]
    banks = _build_banks(x, y, cfg, rngs, n_jobs=jobs, device=device)

    def observed(ji: tuple[int, int]) -> float:
        j, i = ji
        denom, _, strata = banks[j]
        return _s_star(x[:, i], i, denom, strata, cfg)

    pairs = [(j, i) for j in range(n_t) for i in range(n_params)]
    if jobs > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(jobs) as pool:
            obs = list(pool.map(observed, pairs))
    else:
        obs = [observed(ji) for ji in pairs]
    s_star = np.array(obs, dtype=np.float64).reshape(n_t, n_params)
    pvals = np.zeros((n_t, n_params))
    declared = np.zeros((n_t, n_params), dtype=bool)
    for j in range(n_t):
        for i in range(n_params):
            pvals[j, i] = _pval(s_star[j, i], i, banks[j][1])
        declared[j] = by_declare(pvals[j], cfg.q)
    return MSCRResult(pvals=pvals, s_star=s_star, declared=declared, config=cfg)

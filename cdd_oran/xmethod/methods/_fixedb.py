"""Fixed-b reference distribution of a Bartlett-kernel HAC t statistic (Kiefer and Vogelsang 2005, "A new asymptotic
theory for heteroskedasticity-autocorrelation robust tests", Econometric Theory 21, 1130-1164), for pcorr_hac's
``inference='fixed_b'`` variant (ruling R-38).

Fixed-b: the bandwidth M = b T is a fixed fraction of the sample (Bartlett weights k(j / M) = 1 - j / M), and the t
statistic converges to t*(b) = W(1) / sqrt(Q(b)), with, for the Bartlett kernel and a Brownian bridge B(r) = W(r) -
r W(1),
    Q(b) = (2 / b) [ int_0^1 B(r)^2 dr - int_0^(1-b) B(r + b) B(r) dr ],      0 < b <= 1.
B is independent of W(1) ~ N(0, 1), so the two-sided p-value is P(|t*(b)| >= |t|) = E_Q[ 2 Phi(-|t| sqrt(Q(b))) ]: we
average that conditional normal tail over simulated Q draws (smooth in t, accurate in the tails with few draws).
Simulation as in KV (2005, section on critical value functions): normalised partial sums of T_GRID = 1000 i.i.d.
N(0, 1) deviates approximate W; N_DRAWS = 20 000 draws from a fixed seed (no data-dependent randomness; results are
deterministic). Q(b) is evaluated at grid lags m = b T_GRID; a b between grid lags interpolates the p-value linearly in
b; b < B0 = 10 / T_GRID (too few grid lags) interpolates linearly between the standard normal p (b = 0, Q = 1) and the
p at B0. Check (tests): at the 90 / 95 / 97.5 / 99 % points of KV Table I's Bartlett critical value functions
cv(b) = a0 + a1 b + a2 b^2 + a3 b^3 this gives p within Monte Carlo + fit error of .20 / .10 / .05 / .02.
pcorr_hac maps its Newey-West lag L (statsmodels weights 1 - j / (L + 1)) to M = L + 1, b = (L + 1) / n.
"""
from __future__ import annotations

import functools

import numpy as np
from scipy import stats

T_GRID = 1000
N_DRAWS = 20_000
SEED = (7803, 3801)            # classic method RNG tag 7803 (CONTRACT sec 6) + a fixed sub-stream, data-independent
B0 = 10 / T_GRID
_CHUNK = 2_000
# KV (2005) Table I, Bartlett kernel: (percentile, a0, a1, a2, a3) of cv(b) for the t statistic
KV_TABLE1_BARTLETT = ((0.90, 1.2816, 1.3040, 0.5135, -0.3386), (0.95, 1.6449, 2.1859, 0.3142, -0.3427),
                      (0.975, 1.9600, 2.9694, 0.4160, -0.5324), (0.99, 2.3263, 4.1618, 0.5368, -0.9060))


@functools.lru_cache(maxsize=1)
def _bridges() -> tuple[np.ndarray, np.ndarray]:
    """(B [N_DRAWS, T_GRID] float32 Brownian bridges on r = 1/T..1, int B^2 [N_DRAWS])."""
    rng = np.random.default_rng(np.random.SeedSequence(SEED))
    B = np.empty((N_DRAWS, T_GRID), np.float32)
    r = np.arange(1, T_GRID + 1) / T_GRID
    for i in range(0, N_DRAWS, _CHUNK):
        S = np.cumsum(rng.standard_normal((min(_CHUNK, N_DRAWS - i), T_GRID)), axis=1) / np.sqrt(T_GRID)
        B[i:i + len(S)] = S - r * S[:, -1:]
    i0 = np.einsum("ij,ij->i", B, B, dtype=np.float64) / T_GRID
    return B, i0


@functools.lru_cache(maxsize=4096)
def q_draws(m: int) -> np.ndarray:
    """Q(b) draws at grid lag m (b = m / T_GRID), clipped at 0."""
    if not 1 <= m <= T_GRID:
        raise ValueError(f"grid lag must be in 1..{T_GRID}, got {m}")
    B, i0 = _bridges()
    b = m / T_GRID
    if m == T_GRID:
        im = np.zeros(N_DRAWS)
    else:
        im = np.concatenate([np.einsum("ij,ij->i", B[i:i + _CHUNK, m:], B[i:i + _CHUNK, :-m], dtype=np.float64)
                             for i in range(0, N_DRAWS, _CHUNK)]) / T_GRID
    return np.maximum(2.0 / b * (i0 - im), 0.0)


def _p_grid(t: float, m: int) -> float:
    return float(np.mean(2.0 * stats.norm.sf(abs(t) * np.sqrt(q_draws(m)))))


def fixedb_p(t: float, b: float) -> float:
    """Two-sided fixed-b p-value of a Bartlett HAC t statistic ``t`` at bandwidth fraction ``b`` = M / T (module doc)."""
    if not np.isfinite(t):
        return 0.0 if np.isinf(t) else float("nan")
    b = float(min(max(b, 0.0), 1.0))
    p0 = float(2.0 * stats.norm.sf(abs(t)))
    if b <= 0.0:
        return p0
    if b < B0:
        w = b / B0
        return (1.0 - w) * p0 + w * _p_grid(t, round(B0 * T_GRID))
    x = b * T_GRID
    lo, hi = int(np.floor(x)), int(np.ceil(x))
    if lo == hi:
        return _p_grid(t, lo)
    w = x - lo
    return (1.0 - w) * _p_grid(t, lo) + w * _p_grid(t, hi)


def kv_cv(b: float, percentile: float) -> float:
    """KV (2005) Table I Bartlett critical value function cv(b) at ``percentile`` (.90 / .95 / .975 / .99)."""
    for pc, a0, a1, a2, a3 in KV_TABLE1_BARTLETT:
        if abs(pc - percentile) < 1e-12:
            return a0 + a1 * b + a2 * b ** 2 + a3 * b ** 3
    raise KeyError(percentile)

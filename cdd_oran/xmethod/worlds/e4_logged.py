"""E4 variant for regime R3: the confounder is OBSERVED and the behaviour policy's probabilities are logged.

New file; the frozen E4 SCM (``cdd_oran/envs/v2/e4.py``, ``docs/benchmark/GATE_CONTRACT_E4.md``) is untouched.
The outcome mechanism, the latent tape slots (``Z``, ``eta``) and the timeline are inherited unchanged:

    K_out(q+1) = alpha*A_q + theta*Z_q + c          alpha = -1, theta = +2.5, c = 0

Only the behaviour action changes, from the frozen continuous clipped normal to its rounding onto the frozen
do-grid ``V = {0, 0.01, ..., 1}``:

    W_q = 0.5 + lambda*Z_q + eta_q,   eta_q ~ N(0, 0.5^2)   (the frozen behaviour draw, same tape slots)
    A_q = V[k],  k = floor(100*clip(W_q, 0, 1) + 0.5)

so that ``P(A_q = V[k] | Z_q)`` is an exact, logged probability (a discrete policy with full support on V for
every Z). Under R3, ``Z_q`` is exposed to the analyst as context; with the same env seed the R3 action is the
grid rounding of the R4 (frozen ``mode="obs"``) action, row for row.
"""
from __future__ import annotations

import math

import numpy as np

from cdd_oran.envs.v2.e4 import E4V2Env

GRID = np.arange(101, dtype=float) / 100.0          # the frozen do-grid V
ETA_SD = 0.5                                       # frozen eta scale

_erfc = np.vectorize(math.erfc, otypes=[float])


def _phi_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * _erfc(-np.asarray(x, dtype=float) / math.sqrt(2.0))


def _interval_prob(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """P(a <= N(0,1) < b) without cancellation in either tail (upper tail via the survival function)."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    upper = a > 0
    return np.where(upper, _phi_cdf(-a) - _phi_cdf(-b), _phi_cdf(b) - _phi_cdf(a))


def grid_index(w: np.ndarray | float) -> np.ndarray:
    """Grid index of the rounded, clipped behaviour draw ``w``."""
    return np.floor(100.0 * np.clip(np.asarray(w, dtype=float), 0.0, 1.0) + 0.5).astype(np.int64)


def policy_probs(z: np.ndarray, lam: float, eta_sd: float = ETA_SD) -> np.ndarray:
    """``[n, 101]`` table of ``P(A = V[k] | Z = z)`` for the discretised behaviour policy (rows sum to 1)."""
    m = 0.5 + float(lam) * np.asarray(z, dtype=float).reshape(-1, 1)
    # bin edges in W: (-inf, .005), [.005, .015), ..., [.985, .995), [.995, inf)
    edges = np.concatenate([[-np.inf], (np.arange(100, dtype=float) + 0.5) / 100.0, [np.inf]])
    x = (edges[None, :] - m) / eta_sd                                  # [n, 102]
    return _interval_prob(x[:, :-1], x[:, 1:])


def policy_prob_of(k: np.ndarray, z: np.ndarray, lam: float, eta_sd: float = ETA_SD) -> np.ndarray:
    """``P(A = V[k_i] | Z = z_i)`` per row (the logged propensity of the realised action)."""
    k = np.asarray(k, dtype=np.int64)
    m = 0.5 + float(lam) * np.asarray(z, dtype=float)
    lo_edge = np.where(k == 0, -np.inf, (k - 0.5) / 100.0)
    hi_edge = np.where(k == 100, np.inf, (k + 0.5) / 100.0)
    return _interval_prob((lo_edge - m) / eta_sd, (hi_edge - m) / eta_sd)


class E4LoggedEnv(E4V2Env):
    """Frozen E4 mechanism; behaviour action rounded to the do-grid (observation mode only)."""

    def __init__(self, env_seed: int = 0, lam: float = 1.0, episode: int = 0) -> None:
        super().__init__(env_seed=env_seed, lam=lam, mode=self.MODE_OBS, episode=episode)

    def behavior_index(self) -> int:
        """Grid index of the discretised behaviour action at the current pending coordinate."""
        eta = self._draw_eta(self.time)                       # frozen eta slot
        return int(grid_index(0.5 + self.lam * self.Z + eta))

    def behavior_action(self) -> float:
        return float(GRID[self.behavior_index()])

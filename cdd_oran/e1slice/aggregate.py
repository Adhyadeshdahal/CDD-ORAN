"""Deterministic multi-seed envelope aggregation for the E1 slice sweep (Plan 004).

Pure numeric helpers only -- no file I/O, no subprocess, no SciPy. Given per-replicate
scalar series they produce effect *envelopes* (mean / median / sample-std / min / max +
the raw values) and a deterministic bootstrap 95% CI of a paired mean.

Determinism contract: a fixed ``resample_seed`` and a fixed ``n_resamples`` yield
byte-identical floats when re-run on the same NumPy build, because the resample index
matrix is drawn in a single seeded call and every reduction is over a fixed axis order.
The sweep reports *envelopes and failures*, never a hypothesis-significance claim, so
these helpers deliberately expose the full spread, not a p-value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def envelope(values: Sequence[float]) -> dict[str, Any]:
    """Summarise a per-replicate scalar series as an effect envelope.

    ``std`` is the *sample* standard deviation (ddof=1); it is ``0.0`` for a single
    value. The raw ``values`` are retained (as plain floats) so the report can show
    every replicate, per the frozen discipline of retaining all seeds.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"envelope expects a non-empty 1-D series, got shape {arr.shape}")
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(v) for v in arr.tolist()],
    }


def bootstrap_ci_mean(
    values: Sequence[float],
    *,
    resample_seed: int = 0,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Deterministic percentile bootstrap 95% CI of the mean of ``values``.

    Draws ``n_resamples`` with-replacement resamples of size ``n`` using a single
    ``np.random.default_rng(resample_seed)`` draw, takes each resample mean, and returns
    the central ``1 - alpha`` percentile interval. No SciPy: ``np.percentile`` (linear
    interpolation) supplies the bounds. Given the same inputs and seed the returned
    floats are bit-identical across re-runs on the same NumPy build.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"bootstrap expects a non-empty 1-D series, got shape {arr.shape}")
    n = arr.size
    rng = np.random.default_rng(resample_seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "mean": float(arr.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "resample_seed": int(resample_seed),
        "n_resamples": int(n_resamples),
        "alpha": float(alpha),
    }


def paired_diff(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Element-wise ``a[i] - b[i]`` for two replicate-aligned series (equal length)."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape or aa.ndim != 1:
        raise ValueError(f"paired_diff expects two equal-length 1-D series, got {aa.shape} vs {bb.shape}")
    return [float(v) for v in (aa - bb).tolist()]


def paired_summary(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resample_seed: int = 0,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Envelope + bootstrap CI of the paired difference ``a - b`` (both aligned by seed)."""
    diffs = paired_diff(a, b)
    return {
        "diffs": diffs,
        "envelope": envelope(diffs),
        "bootstrap_ci": bootstrap_ci_mean(
            diffs, resample_seed=resample_seed, n_resamples=n_resamples, alpha=alpha
        ),
    }

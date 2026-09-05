"""Label-free nonlinear E2 graph discovery: U-centered partial distance correlation (frozen).

Implements the contract frozen in ``docs/benchmark/E2_DISCOVERY_PROTOCOL.md`` (committed as
``PROTOCOL_COMMIT`` below, BEFORE any E2 recovery result was computed). The edge score is the
magnitude of the U-centered partial distance correlation ``|pdCor|`` (§6); selection is a
per-candidate permutation null under per-target BH-FDR (§7). Both are truth-free.

The module deliberately imports NO ``E2V2Env`` and no true-adjacency symbol: the candidate-graph
shape is the protocol's fixed ``(6, 14)`` layout (a local constant, not env truth) and everything
else comes from the persisted row shapes. Ground truth is read only in ``evaluate.py``, and only
after ``discovery.json`` is persisted and hashed.

Numerical notes:
- The U-centered inner product ``<A,B> = (1/(n(n-3))) sum_{k!=l} A_kl B_kl`` is a genuine inner
  product on the Hilbert space of U-centered matrices (Szekely-Rizzo 2014), so every SELF product
  (``<A,A>``, ``<Pxz,Pxz>`` ...) is a sum of squares and non-negative, and ``|pdCor| <= 1`` by
  Cauchy-Schwarz whenever the denominator is positive.
- The RELATIVE denominator guard (§6.3, ``denominator_epsilon = 1e-12``) catches a candidate/target
  whose projected self-norm collapses relative to its raw self-norm (the E2 analogue of E1's
  collinearity STOP): the edge is guarded -> ``pdCor = NaN``, ``p = 1``, never selected.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cdd_oran.e2slice import SCHEMA_VERSION
from cdd_oran.e2slice.dataset import (
    E2Rows,
    _atomic_write_text,
    _git_sha,
    canonical_json,
    guard_descendants,
    load_dataset,
)

# Full SHA of the frozen protocol commit (``docs: freeze E2 discovery protocol``); abbreviates to
# ``828e345``. Recorded in every discovery.json so the frozen contract provably predates any result.
PROTOCOL_COMMIT = "828e3458065bfe27ff7af07b1295b650f89e878e"

# Downstream stages that must not survive a re-``discover`` unless ``force`` is given.
_DISCOVER_DESCENDANTS = ("recovery.json",)

# The frozen E2 candidate-graph shape (num_kpis, num_params + num_kpis) = (6, 14) (§5). This is the
# protocol's fixed layout, NOT environment truth, so it is a local constant (discovery imports no
# env symbol). 48 NCP->KPI + 36 lagged KPI->KPI = 84 candidates per seed.
_NUM_TARGETS = 6
_NUM_CANDIDATES = 14
_CANDIDATE_SHAPE = (_NUM_TARGETS, _NUM_CANDIDATES)

# --- FROZEN constants (§14). None may change after any E2 result is inspected. ---
FROZEN_SCORE_METHOD = "u_centered_partial_distance_correlation"
FROZEN_THRESHOLD_METHOD = "per_candidate_permutation_per_target_bh_fdr"
FROZEN_ALPHA = 1  # distance exponent (§6.1)
FROZEN_DENOMINATOR_EPSILON = 1e-12  # relative denominator guard (§6.3)
FROZEN_B_PERM = 999  # permutations (§7.1)
FROZEN_PERMUTATION_SEED = 0  # permutation RNG base (§7.1)
FROZEN_Q = 0.05  # per-target BH-FDR level, m = 14 (§7.2)
FROZEN_MAX_GUARD_FRACTION = 0.05  # HALT at >= ceil(0.05 * 84) = 5 guarded of 84 (§11)


@dataclass(frozen=True)
class E2DiscoveryConfig:
    """Discovery constants. Defaults are the FROZEN §14 values.

    NOT freeze-locked in ``__post_init__`` because the REQUIRED truth-free smoke test (§15
    operational note) constructs a THROWAWAY config (tiny ``B_perm``) to measure runtime. The
    canonical persist path (``write_discovery``) ALWAYS uses ``frozen_config()`` and never a
    throwaway one, and ``load_discovery`` re-derives every persisted constant against the frozen
    module values -- so a retuned mask can never be persisted under this ``protocol_commit``.
    """

    alpha: int = FROZEN_ALPHA
    denominator_epsilon: float = FROZEN_DENOMINATOR_EPSILON
    b_perm: int = FROZEN_B_PERM
    permutation_seed: int = FROZEN_PERMUTATION_SEED
    q: float = FROZEN_Q
    max_guard_fraction: float = FROZEN_MAX_GUARD_FRACTION


def frozen_config() -> E2DiscoveryConfig:
    """The frozen §14 discovery config (used by the canonical persist path)."""
    return E2DiscoveryConfig()


@dataclass(frozen=True)
class E2DiscoveryResult:
    signed_pdcor: npt.NDArray[np.float64]  # (6, 14) signed partial distance correlations
    scores: npt.NDArray[np.float64]        # (6, 14) |signed_pdcor|, NaN where guarded
    perm_pvalues: npt.NDArray[np.float64]  # (6, 14) permutation p-values, 1.0 where guarded
    guarded_mask: npt.NDArray[np.int64]    # (6, 14) {0,1} guard fires
    binary_mask: npt.NDArray[np.int64]     # (6, 14) {0,1} per-target BH-FDR selection
    n_rows: int
    mean_candidates: npt.NDArray[np.float64]  # (14,)
    std_candidates: npt.NDArray[np.float64]   # (14,)
    mean_targets: npt.NDArray[np.float64]     # (6,)
    std_targets: npt.NDArray[np.float64]      # (6,)


# --- distance geometry (§6.1) ------------------------------------------------
def dist_matrix_1d(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Euclidean distance matrix of a 1-D variable, exponent alpha = 1 (``|u - v|``)."""
    return np.abs(v[:, None] - v[None, :])


def dist_matrix_euclidean(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Euclidean distance matrix of a d-dim variable ``z`` (n, d), exponent alpha = 1.

    Uses ``||z_k - z_l||^2 = ||z_k||^2 + ||z_l||^2 - 2 z_k.z_l`` (a Gram-matrix identity) so the
    (n, n) distance is formed without materialising an (n, n, d) broadcast.
    """
    sq = np.einsum("ij,ij->i", z, z)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (z @ z.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


# --- U-centering + U-inner product (§6.2) ------------------------------------
def u_center(dist: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """U-centered (unbiased) matrix of a distance matrix (Szekely-Rizzo 2013/2014).

    ``A_kl = a_kl - (1/(n-2)) sum_m a_km - (1/(n-2)) sum_m a_ml + (1/((n-1)(n-2))) sum_mo a_mo``
    for ``k != l`` and ``A_kk = 0``.
    """
    n = dist.shape[0]
    row = dist.sum(axis=1, keepdims=True)  # sum_m a_km  (n, 1)
    col = dist.sum(axis=0, keepdims=True)  # sum_m a_ml  (1, n)
    total = float(dist.sum())              # sum_mo a_mo
    u = dist - row / (n - 2) - col / (n - 2) + total / ((n - 1) * (n - 2))
    np.fill_diagonal(u, 0.0)
    return u


def u_inner(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """U-centered inner product ``(1/(n(n-3))) sum_{k!=l} A_kl B_kl`` (unbiased, needs n >= 4).

    Both inputs are U-centered (zero diagonal), so the full elementwise sum equals the ``k != l``
    sum.
    """
    n = a.shape[0]
    return float((a * b).sum() / (n * (n - 3)))


def _project_out(a: npt.NDArray[np.float64], c: npt.NDArray[np.float64], c_self: float,
                 a_dot_c: float) -> npt.NDArray[np.float64]:
    """Project the U-centered matrix ``a`` onto the orthogonal complement of ``c`` (§6.2)."""
    return a - (a_dot_c / c_self) * c


def denominator_guard_fires(
    va_proj: float, vb_proj: float, cand_self: float, tgt_self: float, eps: float
) -> bool:
    """The RELATIVE denominator guard predicate (§6.3).

    Fires when ``sqrt(vA_projected * vB_projected) <= eps * sqrt(vA_raw * vB_raw)``: the projected
    self-norms have collapsed relative to the raw self-norms (candidate/target nearly explained by
    the conditioning set). Relative, so it is invariant to the overall magnitude of the U-centered
    distances.
    """
    return math.sqrt(va_proj * vb_proj) <= eps * math.sqrt(cand_self * tgt_self)


def partial_distance_correlation(
    cand_u: npt.NDArray[np.float64],
    cand_self: float,
    tgt_u: npt.NDArray[np.float64],
    tgt_self: float,
    cond_u: npt.NDArray[np.float64],
    cond_self: float,
    eps: float,
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], float, float, bool]:
    """Signed ``pdCor`` from U-centered matrices (§6.2/§6.3).

    Returns ``(signed_pdcor, pxz, pyz, va_proj, vb_proj, guarded)``. When the guard fires,
    ``signed_pdcor`` is ``NaN`` and ``guarded`` is True. The score is ``abs(signed_pdcor)``.
    """
    pyz = _project_out(tgt_u, cond_u, cond_self, u_inner(tgt_u, cond_u))
    pxz = _project_out(cand_u, cond_u, cond_self, u_inner(cand_u, cond_u))
    va_proj = u_inner(pxz, pxz)
    vb_proj = u_inner(pyz, pyz)
    if denominator_guard_fires(va_proj, vb_proj, cand_self, tgt_self, eps):
        return math.nan, pxz, pyz, va_proj, vb_proj, True
    signed = u_inner(pxz, pyz) / math.sqrt(va_proj * vb_proj)
    return signed, pxz, pyz, va_proj, vb_proj, False


def bh_fdr_reject(pvalues: npt.NDArray[np.float64], q: float) -> npt.NDArray[np.int64]:
    """Benjamini-Hochberg FDR selection at level ``q`` over one target's ``m`` p-values (§7.2).

    Returns a ``{0,1}`` vector: 1 where the p-value passes BH. Standard step-up procedure: with
    p-values sorted ascending ``p_(1) <= ... <= p_(m)``, find the largest ``k`` with
    ``p_(k) <= (k/m) q`` and reject all ``p <= p_(k)``.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.shape[0]
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    crit = q * (np.arange(1, m + 1, dtype=np.float64) / m)
    passing = sorted_p <= crit
    if not passing.any():
        return np.zeros(m, dtype=np.int64)
    kmax = int(np.nonzero(passing)[0].max())  # 0-based index of the largest passing rank
    threshold = float(sorted_p[kmax])
    return (p <= threshold).astype(np.int64)


def _standardize(a: npt.NDArray[np.float64]) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """z-standardize columns with population std (ddof = 0); reject zero-variance columns (§6.1)."""
    mean = a.mean(axis=0)
    std = a.std(axis=0)  # ddof = 0
    if (std == 0.0).any():
        raise ValueError(
            f"discovery: zero-variance column(s) at {np.where(std == 0.0)[0].tolist()}; cannot "
            "z-standardize (a broken-run HALT, §11)."
        )
    return (a - mean) / std, mean, std


def discover_graph(rows: E2Rows, cfg: E2DiscoveryConfig | None = None) -> E2DiscoveryResult:
    """Select the E2 candidate graph from the WHOLE frozen dataset (no labels, no env truth).

    Implements §6 (edge score) and §7 (per-candidate permutation null + per-target BH-FDR) exactly.
    Raises on the §11 numerical HALT conditions (guard-fraction exceeded, NaN/inf outside the guard
    path, degenerate permutation null for a selected candidate).
    """
    cfg = cfg if cfg is not None else frozen_config()
    x = np.concatenate([rows.x_params, rows.x_kpis], axis=1).astype(np.float64)  # (n, 14)
    y = rows.y_kpis.astype(np.float64)  # (n, 6)
    n, d = x.shape
    k = y.shape[1]
    if (d, k) != (_NUM_CANDIDATES, _NUM_TARGETS):
        raise ValueError(
            f"discovery: candidate/target layout ({k}, {d}) != frozen {_CANDIDATE_SHAPE}"
        )
    if n < 4:
        raise ValueError(f"discovery: need n >= 4 for the U-centered inner product, got n = {n}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("discovery: dataset rows contain non-finite values")

    xs, mean_c, std_c = _standardize(x)
    ys, mean_t, std_t = _standardize(y)

    eps = float(cfg.denominator_epsilon)

    # Precompute per-target (6) U-centered distance matrices + raw self-norms (reused across all i).
    target_u = [u_center(dist_matrix_1d(ys[:, j])) for j in range(k)]
    target_self = [u_inner(target_u[j], target_u[j]) for j in range(k)]

    signed = np.full((k, d), np.nan, dtype=np.float64)
    scores = np.full((k, d), np.nan, dtype=np.float64)
    pvalues = np.ones((k, d), dtype=np.float64)
    guarded = np.zeros((k, d), dtype=np.int64)
    null_variance = np.full((k, d), np.nan, dtype=np.float64)

    for i in range(d):
        cand_u = u_center(dist_matrix_1d(xs[:, i]))              # A~ for candidate i
        cand_self = u_inner(cand_u, cand_u)                      # vA_raw
        z = np.delete(xs, i, axis=1)                            # 13-dim conditioning stack
        cond_u = u_center(dist_matrix_euclidean(z))             # C~
        cond_self = u_inner(cond_u, cond_u)                     # (C~ . C~)
        if cond_self <= 0.0:
            raise ValueError(
                f"discovery: conditioning self-norm (C.C) = {cond_self:.3e} <= 0 for candidate {i} "
                "(degenerate 13-dim conditioning geometry); broken-run HALT (§11)."
            )
        xi = xs[:, i]

        for j in range(k):
            tgt_u = target_u[j]
            tgt_self = target_self[j]
            signed_ij, _pxz, pyz, _va_proj, vb_proj, is_guarded = partial_distance_correlation(
                cand_u, cand_self, tgt_u, tgt_self, cond_u, cond_self, eps
            )
            # RELATIVE denominator guard (§6.3): projected self-norms collapse vs raw self-norms.
            if is_guarded:
                guarded[j, i] = 1  # NaN score, p = 1, not selected (arrays pre-filled)
                continue

            observed = abs(signed_ij)
            signed[j, i] = signed_ij
            scores[j, i] = observed

            # Per-candidate permutation null (§7.1): Z, Y_j fixed (C~, Pyz, vb_proj reused);
            # permute X_i marginally, recompute A~ and Pxz, recompute |pdCor|.
            perm_rng = np.random.default_rng(
                np.random.SeedSequence(entropy=int(cfg.permutation_seed), spawn_key=(int(j), int(i)))
            )
            null = np.empty(cfg.b_perm, dtype=np.float64)
            for b in range(cfg.b_perm):
                cand_u_p = u_center(dist_matrix_1d(xi[perm_rng.permutation(n)]))
                pxz_p = _project_out(cand_u_p, cond_u, cond_self, u_inner(cand_u_p, cond_u))
                va_proj_p = u_inner(pxz_p, pxz_p)
                denom_p = math.sqrt(va_proj_p * vb_proj)
                # Pxz_p == 0 (permuted candidate carries no residual signal) -> |pdCor| := 0.
                null[b] = abs(u_inner(pxz_p, pyz) / denom_p) if denom_p > 0.0 else 0.0
            null_variance[j, i] = float(null.var())
            pvalues[j, i] = (1 + int(np.count_nonzero(null >= observed))) / (cfg.b_perm + 1)

    # §11 HALT: NaN/inf in scores OUTSIDE the declared guard path.
    non_guarded = guarded == 0
    if not np.isfinite(scores[non_guarded]).all():
        raise ValueError("discovery: non-finite edge score outside the denominator-guard path (HALT, §11)")

    # §11 HALT: guard fires on >= ceil(max_guard_fraction * 84) of the 84 candidates.
    n_candidates = k * d
    guard_halt = math.ceil(cfg.max_guard_fraction * n_candidates)
    n_guarded = int(guarded.sum())
    if n_guarded >= guard_halt:
        raise ValueError(
            f"discovery: denominator guard fired on {n_guarded} of {n_candidates} candidates "
            f">= HALT threshold {guard_halt}; diagnose the distance geometry / standardization (§11)."
        )

    # Per-target BH-FDR selection (§7.2). Guarded candidates carry p = 1 and are never selected.
    binary = np.zeros((k, d), dtype=np.int64)
    for j in range(k):
        binary[j] = bh_fdr_reject(pvalues[j], cfg.q)
    binary[guarded == 1] = 0  # fail-closed: guarded is never selected

    # §11 HALT: degenerate permutation null (zero variance) for a SELECTED candidate.
    selected = binary == 1
    if selected.any() and (null_variance[selected] == 0.0).any():
        where = np.argwhere(selected & (null_variance == 0.0))
        raise ValueError(
            f"discovery: degenerate (zero-variance) permutation null for selected candidate(s) "
            f"{where.tolist()} (target, candidate); broken-run HALT (§11)."
        )

    return E2DiscoveryResult(
        signed_pdcor=signed,
        scores=scores,
        perm_pvalues=pvalues,
        guarded_mask=guarded,
        binary_mask=binary,
        n_rows=n,
        mean_candidates=mean_c,
        std_candidates=std_c,
        mean_targets=mean_t,
        std_targets=std_t,
    )


# --- persistence (§8) --------------------------------------------------------
def _json_num(v: float) -> float | None:
    """NaN -> None for compact, valid JSON; finite floats pass through unchanged."""
    return None if (isinstance(v, float) and math.isnan(v)) else float(v)


def _grid(arr: npt.NDArray[np.float64]) -> list[list[float | None]]:
    return [[_json_num(float(v)) for v in row] for row in arr]


def build_discovery_record(result: E2DiscoveryResult, dataset_hash: str) -> dict[str, Any]:
    """Assemble the persisted, hash-bound discovery record (content_hash added last, §8)."""
    sha, dirty = _git_sha()
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": dataset_hash,
        "protocol_commit": PROTOCOL_COMMIT,
        "n_rows": int(result.n_rows),
        "seed": None,  # filled by write_discovery from the manifest
        "moments": {
            "mean_candidates": [float(v) for v in result.mean_candidates],
            "std_candidates": [float(v) for v in result.std_candidates],
            "mean_targets": [float(v) for v in result.mean_targets],
            "std_targets": [float(v) for v in result.std_targets],
        },
        "score_method": FROZEN_SCORE_METHOD,
        "alpha": FROZEN_ALPHA,
        "denominator_epsilon": FROZEN_DENOMINATOR_EPSILON,
        "threshold_method": FROZEN_THRESHOLD_METHOD,
        "b_perm": FROZEN_B_PERM,
        "permutation_seed": FROZEN_PERMUTATION_SEED,
        "q": FROZEN_Q,
        "signed_pdcor": _grid(result.signed_pdcor),
        "scores": _grid(result.scores),
        "perm_pvalues": _grid(result.perm_pvalues),
        "guarded_mask": result.guarded_mask.astype(int).tolist(),
        "binary_mask": result.binary_mask.astype(int).tolist(),
        "candidate_shape": list(result.binary_mask.shape),
        "git_sha": sha,
        "git_dirty": dirty,
    }
    return record


def _finalize_record(record: dict[str, Any]) -> dict[str, Any]:
    record["content_hash"] = hashlib.sha256(canonical_json(record).encode()).hexdigest()
    return record


def write_discovery(
    dataset_dir: str | Path, cfg: E2DiscoveryConfig | None = None, force: bool = False
) -> dict[str, Any]:
    """Discover the E2 graph from the whole frozen dataset and write ``discovery.json``.

    ALWAYS uses the FROZEN §14 config (a passed ``cfg`` must equal it, else it is rejected) so the
    canonical artifact can never be persisted with retuned constants. Binds to the current dataset,
    refuses to clobber a downstream recovery without ``force``, and publishes atomically -- BEFORE
    any recovery score.
    """
    cfg = cfg if cfg is not None else frozen_config()
    if cfg != frozen_config():
        raise ValueError(
            "write_discovery: the canonical discovery artifact is FROZEN and may only be written "
            f"with the §14 constants {frozen_config()}, got {cfg}. Throwaway configs (e.g. the "
            "smoke test) must call discover_graph directly and never persist to discovery.json."
        )
    out = Path(dataset_dir)
    rows, manifest = load_dataset(out)
    result = discover_graph(rows, cfg)
    record = build_discovery_record(result, manifest["dataset_hash"])
    record["seed"] = int(manifest["config"]["seed"])
    record = _finalize_record(record)
    guard_descendants(out, _DISCOVER_DESCENDANTS, force, "discover")
    _atomic_write_text(out / "discovery.json", json.dumps(record, indent=2, sort_keys=True))
    return record


# --- fail-closed load (§8) ---------------------------------------------------
def _grid_array(grid: list[list[float | None]]) -> npt.NDArray[np.float64]:
    """Parse a persisted grid (None -> NaN) into a float64 array."""
    return np.asarray(
        [[np.nan if v is None else float(v) for v in row] for row in grid], dtype=np.float64
    )


def load_discovery(
    dataset_dir: str | Path, *, expected_dataset_hash: str | None = None
) -> dict[str, Any]:
    """Load ``discovery.json`` and fail closed unless it is internally consistent and hash-bound.

    Re-derives every relation already in the file (content_hash; scores == |signed_pdcor| off the
    guard and NaN exactly on it; perm_pvalues == 1 on the guard; binary_mask == the per-target
    BH-FDR of perm_pvalues at the frozen q; binary_mask == 0 on the guard; every frozen constant),
    so a correctly generated artifact is byte-unaffected while a silently retuned one is rejected.
    """
    disc_file = Path(dataset_dir) / "discovery.json"
    record = json.loads(disc_file.read_text())

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{disc_file}: schema_version {record.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    for key in (
        "dataset_hash", "protocol_commit", "score_method", "alpha", "denominator_epsilon",
        "threshold_method", "b_perm", "permutation_seed", "q", "signed_pdcor", "scores",
        "perm_pvalues", "guarded_mask", "binary_mask", "candidate_shape", "content_hash",
    ):
        if key not in record:
            raise ValueError(f"{disc_file}: missing required field '{key}'")

    stored = record["content_hash"]
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    recomputed = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    if recomputed != stored:
        raise ValueError(f"{disc_file}: content_hash {stored} != recomputed {recomputed}")

    # Frozen constants (§14).
    if record["protocol_commit"] != PROTOCOL_COMMIT:
        raise ValueError(
            f"{disc_file}: protocol_commit {record['protocol_commit']!r} != frozen {PROTOCOL_COMMIT!r}"
        )
    if record["score_method"] != FROZEN_SCORE_METHOD:
        raise ValueError(f"{disc_file}: score_method {record['score_method']!r} != frozen "
                         f"{FROZEN_SCORE_METHOD!r}")
    if record["threshold_method"] != FROZEN_THRESHOLD_METHOD:
        raise ValueError(f"{disc_file}: threshold_method {record['threshold_method']!r} != frozen "
                         f"{FROZEN_THRESHOLD_METHOD!r}")
    if int(record["alpha"]) != FROZEN_ALPHA:
        raise ValueError(f"{disc_file}: alpha {record['alpha']} != frozen {FROZEN_ALPHA}")
    if float(record["denominator_epsilon"]) != FROZEN_DENOMINATOR_EPSILON:
        raise ValueError(
            f"{disc_file}: denominator_epsilon {record['denominator_epsilon']} != frozen "
            f"{FROZEN_DENOMINATOR_EPSILON}"
        )
    if int(record["b_perm"]) != FROZEN_B_PERM:
        raise ValueError(f"{disc_file}: b_perm {record['b_perm']} != frozen {FROZEN_B_PERM}")
    if int(record["permutation_seed"]) != FROZEN_PERMUTATION_SEED:
        raise ValueError(
            f"{disc_file}: permutation_seed {record['permutation_seed']} != frozen "
            f"{FROZEN_PERMUTATION_SEED}"
        )
    if float(record["q"]) != FROZEN_Q:
        raise ValueError(f"{disc_file}: q {record['q']} != frozen {FROZEN_Q}")

    signed = _grid_array(record["signed_pdcor"])
    scores = _grid_array(record["scores"])
    pvalues = _grid_array(record["perm_pvalues"])
    guarded = np.asarray(record["guarded_mask"], dtype=np.int64)
    mask = np.asarray(record["binary_mask"], dtype=np.int64)
    for name, arr in (("signed_pdcor", signed), ("scores", scores), ("perm_pvalues", pvalues),
                      ("guarded_mask", guarded), ("binary_mask", mask)):
        if arr.shape != _CANDIDATE_SHAPE:
            raise ValueError(f"{disc_file}: {name} shape {arr.shape} != frozen candidate shape "
                             f"{_CANDIDATE_SHAPE}")
    if tuple(record["candidate_shape"]) != _CANDIDATE_SHAPE:
        raise ValueError(
            f"{disc_file}: candidate_shape {record['candidate_shape']} != frozen {list(_CANDIDATE_SHAPE)}"
        )
    for name, arr in (("guarded_mask", guarded), ("binary_mask", mask)):
        if not set(np.unique(arr).tolist()).issubset({0, 1}):
            raise ValueError(f"{disc_file}: {name} has non-binary entries")

    is_guarded = guarded == 1
    # scores == |signed_pdcor| off the guard, and NaN exactly where guarded.
    if not np.array_equal(np.isnan(scores), is_guarded):
        raise ValueError(f"{disc_file}: scores are NaN somewhere other than exactly the guarded cells")
    if not np.array_equal(np.isnan(signed), is_guarded):
        raise ValueError(f"{disc_file}: signed_pdcor is NaN somewhere other than the guarded cells")
    off = ~is_guarded
    if not np.allclose(scores[off], np.abs(signed[off]), rtol=0.0, atol=0.0):
        raise ValueError(f"{disc_file}: scores are not |signed_pdcor| off the guard path")
    # perm_pvalues == 1.0 exactly where guarded.
    if not np.all(pvalues[is_guarded] == 1.0):
        raise ValueError(f"{disc_file}: perm_pvalues must be 1.0 wherever guarded_mask == 1")
    if not np.isfinite(pvalues).all():
        raise ValueError(f"{disc_file}: perm_pvalues contain non-finite values")

    # binary_mask == per-target BH-FDR of perm_pvalues at the frozen q, and 0 where guarded.
    for j in range(_CANDIDATE_SHAPE[0]):
        expected = bh_fdr_reject(pvalues[j], FROZEN_Q)
        expected[is_guarded[j]] = 0
        if not np.array_equal(mask[j], expected):
            raise ValueError(
                f"{disc_file}: binary_mask row {j} is inconsistent with the per-target BH-FDR of "
                f"perm_pvalues at q = {FROZEN_Q} (guarded cells forced to 0)"
            )
    if (mask[is_guarded] != 0).any():
        raise ValueError(f"{disc_file}: binary_mask is 1 on a guarded cell (must be fail-closed 0)")

    if expected_dataset_hash is not None and record["dataset_hash"] != expected_dataset_hash:
        raise ValueError(
            f"{disc_file}: dataset_hash {record['dataset_hash']} != dataset {expected_dataset_hash}"
        )
    return record


def discovered_mask_array(record: dict[str, Any]) -> npt.NDArray[np.int64]:
    """The frozen (6, 14) binary mask as an int array."""
    return np.asarray(record["binary_mask"], dtype=np.int64)

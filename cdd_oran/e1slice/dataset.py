"""Deterministic E1 transition-dataset generation and on-disk persistence.

A "row" is one one-step-prediction example drawn from the E1 v2 SCM:

    x = (params, kpis)   -- the COMMITTED state at s_t (before the advance)
    y = latent KPIs after exactly one ``advance``  -- the target k_{t+1}

Because ``advance`` computes ``k_{t+1} = f(prev_params, prev_kpis)`` from the committed
state and only THEN commits the pending (post-action) params, the action applied at step
t does not enter y_t (one-step actuation latency, SEMANTICS §1.1). The random action
policy therefore only diversifies the params seen across a trajectory; the label is a
clean function of the recorded x. Observation noise is OFF, so latent == observed.

Everything is a pure function of the coordinate tape ``(env_seed, episode, time)`` plus a
per-episode action stream seeded from a disjoint namespace, so the NUMERIC rows regenerate
byte-identically on the same platform and their content hash is stable. The full
``manifest.json`` is NOT byte-identical across runs: it records ``created_utc`` and live git
state as provenance. The ``(episode, time)`` coordinates are stored as columns precisely so
a row is traceable back to its generating coordinate.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.envs.v2.e1 import E1V2Env

# Disjoint SeedSequence namespace for the per-episode action stream, kept away from the
# env tape's own namespaces (_NS_INIT/_NS_PROC/_NS_OBS = 0/1/2) so it can never alias them.
_ACTION_NS = 101

_COLUMNS = ("episode", "time", "x_params", "x_kpis", "y_kpis")

# Expected on-disk dtype per column; a tampered array with a different dtype is rejected.
_EXPECTED_DTYPES: dict[str, np.dtype[Any]] = {
    "episode": np.dtype(np.int64),
    "time": np.dtype(np.int64),
    "x_params": np.dtype(np.float64),
    "x_kpis": np.dtype(np.float64),
    "y_kpis": np.dtype(np.float64),
}

# Downstream stages that must not survive a re-``generate`` unless ``force`` is given
# (dependency order: rows/manifest -> split -> arms -> metrics).
_GENERATE_DESCENDANTS = ("split.json", "arms", "metrics.json")


@dataclass(frozen=True)
class E1DatasetConfig:
    """Frozen generation contract. Hashed into the manifest."""

    n_episodes: int = 64
    steps_per_episode: int = 16
    warmup: int = 2
    env_seed: int = 0
    obs_noise_scale: float = 0.0

    def validate(self) -> None:
        """Reject configurations that cannot yield a valid latent, splittable dataset."""
        if self.n_episodes < 2:
            raise ValueError(f"E1DatasetConfig: n_episodes must be >= 2, got {self.n_episodes}")
        if self.steps_per_episode <= 0:
            raise ValueError(
                f"E1DatasetConfig: steps_per_episode must be > 0, got {self.steps_per_episode}"
            )
        if self.warmup < 0:
            raise ValueError(f"E1DatasetConfig: warmup must be >= 0, got {self.warmup}")
        if self.obs_noise_scale != 0.0:
            raise ValueError(
                "E1DatasetConfig: this E1 recovery slice is latent/noiseless, so "
                f"obs_noise_scale must be 0.0, got {self.obs_noise_scale}. Both features and "
                "labels are latent arrays here, so nonzero noise would change scm_hash without "
                "changing any row; observed-noise datasets need a separately specified contract."
            )


@dataclass(frozen=True)
class E1Rows:
    """The persisted arrays. First axis is the row index N across all episodes."""

    episode: npt.NDArray[np.int64]
    time: npt.NDArray[np.int64]
    x_params: npt.NDArray[np.float64]
    x_kpis: npt.NDArray[np.float64]
    y_kpis: npt.NDArray[np.float64]

    @property
    def n(self) -> int:
        return int(self.episode.shape[0])


def _action_rng(env_seed: int, episode: int) -> np.random.Generator:
    """Per-episode action stream, a pure function of (env_seed, episode)."""
    return np.random.default_rng(np.random.SeedSequence((int(env_seed), int(episode), _ACTION_NS)))


def generate_rows(cfg: E1DatasetConfig) -> E1Rows:
    """Roll ``n_episodes`` trajectories of the E1 SCM and collect (x -> y) transition rows."""
    cfg.validate()
    episodes: list[int] = []
    times: list[int] = []
    x_params: list[npt.NDArray[np.float64]] = []
    x_kpis: list[npt.NDArray[np.float64]] = []
    y_kpis: list[npt.NDArray[np.float64]] = []

    for e in range(cfg.n_episodes):
        env = E1V2Env(env_seed=cfg.env_seed, obs_noise_scale=cfg.obs_noise_scale, episode=e)
        env.reset(episode=e)
        rng = _action_rng(cfg.env_seed, e)

        # Warm-up steps are NOT recorded: they replace the zero-initialised prev_kpis with a
        # real mechanism output so the K->K parent terms are exercised in the recorded rows.
        for _ in range(cfg.warmup):
            env.step(int(rng.integers(env.num_params)), float(rng.uniform(0.0, 1.0)))

        for _ in range(cfg.steps_per_episode):
            # s_t: the committed state the mechanism will read on the next advance.
            episodes.append(e)
            times.append(int(env.time))
            x_params.append(env.prev_params.astype(np.float64, copy=True))
            x_kpis.append(env.prev_kpis.astype(np.float64, copy=True))
            # Apply an action (diversifies the NEXT s_t only) then advance to get the label.
            env.apply_action(int(rng.integers(env.num_params)), float(rng.uniform(0.0, 1.0)))
            y_kpis.append(env.advance().astype(np.float64, copy=True))

    return E1Rows(
        episode=np.asarray(episodes, dtype=np.int64),
        time=np.asarray(times, dtype=np.int64),
        x_params=np.stack(x_params).astype(np.float64),
        x_kpis=np.stack(x_kpis).astype(np.float64),
        y_kpis=np.stack(y_kpis).astype(np.float64),
    )


def scm_identity(cfg: E1DatasetConfig) -> dict[str, Any]:
    """The exact E1 SCM the dataset was drawn from, as a canonical JSON-able dict."""
    env = E1V2Env(env_seed=cfg.env_seed, obs_noise_scale=cfg.obs_noise_scale)
    return {
        "env": "E1V2Env",
        "num_params": env.num_params,
        "num_kpis": env.num_kpis,
        "a": [float(v) for v in env.a],
        "b2": float(env.b2),
        "b3": float(env.b3),
        "adjacency_edges": [list(edge) for edge in E1V2Env.adjacency_edges],
        "mu": [float(v) for v in E1V2Env.mu],
        "sigma": [float(v) for v in E1V2Env.sigma],
        "obs_noise_scale": float(cfg.obs_noise_scale),
    }


def canonical_json(obj: Any) -> str:
    """Canonical JSON string (sorted keys, compact separators) for stable hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to a same-directory temp file, then publish with an atomic replace."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_savez_rows(path: Path, rows: E1Rows) -> None:
    """Persist the row columns into a same-dir temp ``.npz``, then publish atomically."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            episode=rows.episode,
            time=rows.time,
            x_params=rows.x_params,
            x_kpis=rows.x_kpis,
            y_kpis=rows.y_kpis,
        )
    os.replace(tmp, path)


def guard_descendants(
    out: Path, descendants: tuple[str, ...], force: bool, stage: str
) -> None:
    """Refuse to overwrite when named downstream artifacts exist; clear them under ``force``.

    Only the explicitly named ``descendants`` are ever deleted -- never an arbitrary path.
    """
    present = [name for name in descendants if (out / name).exists()]
    if present and not force:
        raise ValueError(
            f"{out}: refusing to run '{stage}'; downstream artifacts already exist "
            f"({', '.join(present)}). Re-run with force=True to overwrite them."
        )
    if force:
        for name in descendants:
            target = out / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()


def scm_hash(cfg: E1DatasetConfig) -> str:
    return _sha256_json(scm_identity(cfg))


def dataset_hash(rows: E1Rows) -> str:
    """Content hash over every column's dtype, shape, and raw bytes, in a fixed order.

    Deterministic for a byte-identical regeneration on the same platform (float64 arithmetic
    is bit-exact given the coordinate-keyed tape; cross-platform equality is not claimed).
    """
    h = hashlib.sha256()
    for name in _COLUMNS:
        arr = np.ascontiguousarray(getattr(rows, name))
        h.update(name.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _git_sha() -> tuple[str, bool]:
    def run(*args: str) -> str:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))


def build_manifest(cfg: E1DatasetConfig, rows: E1Rows) -> dict[str, Any]:
    """The immutable provenance record tying rows to the SCM, config, seeds, and code."""
    sha, dirty = _git_sha()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "n_rows": rows.n,
        "n_episodes": cfg.n_episodes,
        "columns": list(_COLUMNS),
        "config": asdict(cfg),
        "seeds": {"env_seed": cfg.env_seed, "action_namespace": _ACTION_NS},
        "dataset_hash": dataset_hash(rows),
        "scm_hash": scm_hash(cfg),
        "scm_identity": scm_identity(cfg),
        "git_sha": sha,
        "git_dirty": dirty,
    }


def write_dataset(
    cfg: E1DatasetConfig, out_dir: str | Path, force: bool = False
) -> dict[str, Any]:
    """Generate, persist ``rows.npz`` + ``manifest.json`` under ``out_dir``, return the manifest.

    Refuses to run if downstream artifacts (split/arms/metrics) already exist unless ``force``
    is set; with ``force`` those known descendants are removed first. Both files are staged to
    same-directory temporaries and published atomically, so a failed write leaves no partial
    final artifact.
    """
    rows = generate_rows(cfg)
    manifest = build_manifest(cfg, rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    guard_descendants(out, _GENERATE_DESCENDANTS, force, "generate")
    _atomic_savez_rows(out / "rows.npz", rows)
    _atomic_write_text(out / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _validate_loaded_dataset(rows: E1Rows, manifest: dict[str, Any], out: Path) -> None:
    """Fail closed unless rows + manifest form a self-consistent, hash-bound dataset."""
    rows_file = out / "rows.npz"
    man_file = out / "manifest.json"

    def bad(msg: str) -> None:
        raise ValueError(msg)

    if manifest.get("schema_version") != SCHEMA_VERSION:
        bad(f"{man_file}: schema_version {manifest.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    for key in ("n_rows", "n_episodes", "columns", "config", "dataset_hash"):
        if key not in manifest:
            bad(f"{man_file}: missing required field '{key}'")
    if manifest["columns"] != list(_COLUMNS):
        bad(f"{man_file}: columns {manifest['columns']} != {list(_COLUMNS)}")

    n = int(rows.episode.shape[0])
    p, k = E1V2Env.num_params, E1V2Env.num_kpis
    expected_shapes = {
        "episode": (n,),
        "time": (n,),
        "x_params": (n, p),
        "x_kpis": (n, k),
        "y_kpis": (n, k),
    }
    for name in _COLUMNS:
        arr = getattr(rows, name)
        if arr.dtype != _EXPECTED_DTYPES[name]:
            bad(f"{rows_file}: column '{name}' dtype {arr.dtype} != {_EXPECTED_DTYPES[name]}")
        if arr.shape != expected_shapes[name]:
            bad(f"{rows_file}: column '{name}' shape {arr.shape} != {expected_shapes[name]}")
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            bad(f"{rows_file}: column '{name}' contains non-finite values")

    if manifest["n_rows"] != n:
        bad(f"{man_file}: n_rows {manifest['n_rows']} != {n} rows on disk")
    config = manifest["config"]
    n_episodes = int(config["n_episodes"])
    steps = int(config["steps_per_episode"])
    if manifest["n_episodes"] != n_episodes:
        bad(f"{man_file}: n_episodes {manifest['n_episodes']} != config {n_episodes}")
    uniq, counts = np.unique(rows.episode, return_counts=True)
    if uniq.shape[0] != n_episodes:
        bad(f"{rows_file}: {uniq.shape[0]} distinct episodes != config n_episodes {n_episodes}")
    if not np.all(counts == steps):
        bad(f"{rows_file}: episodes do not all have steps_per_episode={steps} rows")

    recomputed = dataset_hash(rows)
    if recomputed != manifest["dataset_hash"]:
        bad(f"{rows_file}: dataset_hash {recomputed} != manifest {manifest['dataset_hash']}")


def load_dataset(out_dir: str | Path) -> tuple[E1Rows, dict[str, Any]]:
    """Load persisted rows + manifest from ``out_dir``, validating the full binding.

    Rejects any schema/shape/dtype/count mismatch, non-finite value, or a row payload whose
    recomputed ``dataset_hash`` disagrees with the manifest (tampered or truncated bytes).
    """
    out = Path(out_dir)
    with np.load(out / "rows.npz") as data:
        rows = E1Rows(
            episode=data["episode"],
            time=data["time"],
            x_params=data["x_params"],
            x_kpis=data["x_kpis"],
            y_kpis=data["y_kpis"],
        )
    manifest = json.loads((out / "manifest.json").read_text())
    _validate_loaded_dataset(rows, manifest, out)
    return rows, manifest

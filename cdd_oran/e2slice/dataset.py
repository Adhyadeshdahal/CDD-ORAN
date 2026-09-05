"""Deterministic E2 observational-dataset generation and on-disk persistence (Plan 007 §4).

A "row" is one one-step-prediction example drawn from the E2 v2 SCM with the **decoy OFF**:

    x = (params, kpis)   -- the COMMITTED pre-actuation state s_t = [P0..P7(t) | K0..K5(t)]
    y = latent KPIs after exactly one ``advance``  -- the target k_{t+1} = f(P_t)

Per §4 the eight exogenous params ``P0..P7`` are sampled MUTUALLY INDEPENDENTLY, each uniform over
its registered ID range, so any non-parent has exact zero population dependence on a target. The
per-seed draw uses the FROZEN derivation
``numpy.random.default_rng(numpy.random.SeedSequence(entropy=sampling_seed, spawn_key=(r,)))`` with
``sampling_seed = 0`` and seed index ``r``.

Because the E2 mechanism (``E2V2Env._update_kpis``) is a pure function of the params only, the label
``y`` equals ``f(x_params)`` exactly, and the lagged ``K_t`` carried in ``x`` is the KPI produced by
the PREVIOUS row's (independent) params -- so it is independent of ``x_params`` and of ``y`` (the 36
lagged KPI->KPI candidates are all true-negatives). The rows are aligned across the one-step
actuation latency exactly as E1's ``s_t -> k_{t+1}`` layout, by rolling the forward SCM with a fresh
independent uniform param draw injected each step (two priming advances flush the reset draw so the
first recorded ``x_kpis`` is a real lagged KPI, not zeros). Observation noise is OFF.

The generation reads no ground truth (``E2V2Env`` is used ONLY as the forward SCM for generation,
which the firewall permits). Everything is a pure function of the frozen seed derivation and the
deterministic float64 SCM arithmetic, so rows regenerate byte-identically on the same platform and
their content hash is stable. ``manifest.json`` is NOT byte-identical across runs: it records
``created_utc`` and live git state as provenance.
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

from cdd_oran.e2slice import SCHEMA_VERSION
from cdd_oran.envs.v2.e2 import E2V2Env

_COLUMNS = ("x_params", "x_kpis", "y_kpis")

# Expected on-disk dtype per column; a tampered array with a different dtype is rejected.
_EXPECTED_DTYPES: dict[str, np.dtype[Any]] = {
    "x_params": np.dtype(np.float64),
    "x_kpis": np.dtype(np.float64),
    "y_kpis": np.dtype(np.float64),
}

# Downstream stages that must not survive a re-``generate`` unless ``force`` is given.
_GENERATE_DESCENDANTS = ("discovery.json", "recovery.json")

# Fixed, rng-free probe grid used ONLY to fingerprint the mechanism arithmetic in the manifest
# (a shadow SCM with tweaked coefficients changes the hash). It enumerates NO adjacency edge.
_FINGERPRINT_PROBE_POINTS = 7


@dataclass(frozen=True)
class E2DatasetConfig:
    """Frozen generation contract (§4). Hashed into the manifest.

    ``n_rows_per_seed`` and ``seed`` are run parameters; the FROZEN §14 value for a real run is
    ``n_rows_per_seed = 4000`` with ``seed = r`` over ``r = 0..9`` and ``sampling_seed = 0`` (that
    real run is NOT executed in this implementation phase). ``obs_noise_scale`` (noiseless) and
    ``decoy_omit_p0_k5`` (decoy OFF -- discovery runs against the TRUE SCM) are frozen and rejected
    if set otherwise.
    """

    n_rows_per_seed: int = 4000
    seed: int = 0
    sampling_seed: int = 0
    obs_noise_scale: float = 0.0
    decoy_omit_p0_k5: bool = False

    def validate(self) -> None:
        if self.n_rows_per_seed < 4:
            raise ValueError(
                "E2DatasetConfig: n_rows_per_seed must be >= 4 (the U-centered inner product "
                f"needs n >= 4), got {self.n_rows_per_seed}"
            )
        if self.obs_noise_scale != 0.0:
            raise ValueError(
                "E2DatasetConfig: E2 discovery is noiseless (§4), so obs_noise_scale must be 0.0, "
                f"got {self.obs_noise_scale}."
            )
        if self.decoy_omit_p0_k5 is not False:
            raise ValueError(
                "E2DatasetConfig: discovery runs against the TRUE SCM with the decoy OFF (§4); "
                "the decoy belongs to the decision gate. decoy_omit_p0_k5 must be False, got "
                f"{self.decoy_omit_p0_k5}."
            )


@dataclass(frozen=True)
class E2Rows:
    """The persisted arrays. First axis is the row index N."""

    x_params: npt.NDArray[np.float64]
    x_kpis: npt.NDArray[np.float64]
    y_kpis: npt.NDArray[np.float64]

    @property
    def n(self) -> int:
        return int(self.x_params.shape[0])


def _sampling_rng(sampling_seed: int, seed: int) -> np.random.Generator:
    """The FROZEN §4/§12 per-seed exogenous-parameter generator."""
    return np.random.default_rng(
        np.random.SeedSequence(entropy=int(sampling_seed), spawn_key=(int(seed),))
    )


def generate_rows(cfg: E2DatasetConfig) -> E2Rows:
    """Roll the E2 forward SCM with fresh independent uniform params injected each step (§4)."""
    cfg.validate()
    id_ranges = E2V2Env.id_ranges
    env = E2V2Env(
        env_seed=cfg.seed,
        obs_noise_scale=cfg.obs_noise_scale,
        decoy_omit_p0_k5=cfg.decoy_omit_p0_k5,
        episode=cfg.seed,
    )
    env.reset(episode=cfg.seed)
    rng = _sampling_rng(cfg.sampling_seed, cfg.seed)

    def inject() -> None:
        # Draw all 8 params independently, in the fixed order P0..P7, each uniform over its ID range.
        for i, (low, high) in enumerate(id_ranges):
            env.apply_action(i, float(rng.uniform(low, high)))

    # Prime two advances: after these, prev_params = P^1 and prev_kpis = f(P^0), so the first
    # recorded (x_params, x_kpis) is (P^1, lagged f(P^0)) -- the reset tape draw is fully flushed.
    inject()
    env.advance()
    inject()
    env.advance()

    x_params: list[npt.NDArray[np.float64]] = []
    x_kpis: list[npt.NDArray[np.float64]] = []
    y_kpis: list[npt.NDArray[np.float64]] = []
    for _ in range(cfg.n_rows_per_seed):
        x_params.append(env.prev_params.astype(np.float64, copy=True))  # P^m (committed s_t params)
        x_kpis.append(env.prev_kpis.astype(np.float64, copy=True))      # f(P^{m-1}) (lagged K_t)
        inject()                                                        # pending = P^{m+1}
        y_kpis.append(env.advance().astype(np.float64, copy=True))      # f(P^m) = k_{t+1}

    return E2Rows(
        x_params=np.stack(x_params).astype(np.float64),
        x_kpis=np.stack(x_kpis).astype(np.float64),
        y_kpis=np.stack(y_kpis).astype(np.float64),
    )


def _mechanism_fingerprint(cfg: E2DatasetConfig) -> str:
    """SHA-256 over the SCM arithmetic on a fixed rng-free probe grid (binds the mechanism).

    Uses the forward SCM (``_update_kpis``) on a deterministic linspace grid across the ID ranges;
    a tweaked coefficient in the mechanism changes the hash. It enumerates NO true edge.
    """
    env = E2V2Env(
        env_seed=cfg.seed, obs_noise_scale=0.0, decoy_omit_p0_k5=cfg.decoy_omit_p0_k5
    )
    probe = np.stack(
        [np.linspace(low, high, _FINGERPRINT_PROBE_POINTS) for (low, high) in E2V2Env.id_ranges],
        axis=1,
    )  # (points, 8)
    zeros = np.zeros(E2V2Env.num_kpis, dtype=float)
    outs = np.stack([env._update_kpis(probe[m], zeros) for m in range(probe.shape[0])])
    h = hashlib.sha256()
    for arr in (np.ascontiguousarray(probe), np.ascontiguousarray(outs)):
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def scm_identity(cfg: E2DatasetConfig) -> dict[str, Any]:
    """The exact E2 SCM the dataset was drawn from, as a canonical JSON-able dict.

    Deliberately does NOT include the adjacency edge list: the manifest is loaded by the discovery
    path (for the parent-hash binding), which must read no truth. The mechanism is bound instead by
    ``mechanism_fingerprint`` (forward-SCM arithmetic on a fixed grid).
    """
    return {
        "env": "E2V2Env",
        "num_params": E2V2Env.num_params,
        "num_kpis": E2V2Env.num_kpis,
        "id_ranges": [[float(low), float(high)] for (low, high) in E2V2Env.id_ranges],
        "kpi_mean_std": [[float(m), float(s)] for (m, s) in zip(
            E2V2Env.mean.tolist(), E2V2Env.std.tolist(), strict=True
        )],
        "obs_noise_scale": float(cfg.obs_noise_scale),
        "decoy_omit_p0_k5": bool(cfg.decoy_omit_p0_k5),
        "mechanism_fingerprint": _mechanism_fingerprint(cfg),
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


def _stage_rows(tmp: Path, rows: E2Rows) -> None:
    with open(tmp, "wb") as fh:
        np.savez(fh, x_params=rows.x_params, x_kpis=rows.x_kpis, y_kpis=rows.y_kpis)


def _stage_text(tmp: Path, text: str) -> None:
    tmp.write_text(text)


def _publish_dataset_pair(out: Path, rows: E2Rows, manifest: dict[str, Any]) -> None:
    """Publish ``rows.npz`` + ``manifest.json`` as an atomic pair (both staged, then replaced)."""
    rows_final, man_final = out / "rows.npz", out / "manifest.json"
    rows_tmp = rows_final.with_name(rows_final.name + ".tmp")
    man_tmp = man_final.with_name(man_final.name + ".tmp")
    try:
        _stage_rows(rows_tmp, rows)
        _stage_text(man_tmp, json.dumps(manifest, indent=2, sort_keys=True))
    except BaseException:
        for tmp in (rows_tmp, man_tmp):
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    os.replace(rows_tmp, rows_final)
    os.replace(man_tmp, man_final)


def guard_descendants(
    out: Path, descendants: tuple[str, ...], force: bool, stage: str
) -> None:
    """Refuse to overwrite when named downstream artifacts exist; clear them under ``force``."""
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


def scm_hash(cfg: E2DatasetConfig) -> str:
    return _sha256_json(scm_identity(cfg))


def dataset_hash(rows: E2Rows) -> str:
    """Content hash over every column's dtype, shape, and raw bytes, in a fixed order."""
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


def build_manifest(cfg: E2DatasetConfig, rows: E2Rows) -> dict[str, Any]:
    """The immutable provenance record tying rows to the SCM, config, seeds, and code."""
    sha, dirty = _git_sha()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "n_rows": rows.n,
        "columns": list(_COLUMNS),
        "config": asdict(cfg),
        "seeds": {
            "seed": cfg.seed,
            "sampling_seed": cfg.sampling_seed,
            "seed_derivation": "default_rng(SeedSequence(entropy=sampling_seed, spawn_key=(seed,)))",
        },
        "dataset_hash": dataset_hash(rows),
        "scm_hash": scm_hash(cfg),
        "scm_identity": scm_identity(cfg),
        "git_sha": sha,
        "git_dirty": dirty,
    }


def write_dataset(
    cfg: E2DatasetConfig, out_dir: str | Path, force: bool = False
) -> dict[str, Any]:
    """Generate, persist ``rows.npz`` + ``manifest.json`` under ``out_dir``, return the manifest."""
    rows = generate_rows(cfg)
    manifest = build_manifest(cfg, rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    guard_descendants(out, _GENERATE_DESCENDANTS, force, "generate")
    _publish_dataset_pair(out, rows, manifest)
    return manifest


def _validate_loaded_dataset(rows: E2Rows, manifest: dict[str, Any], out: Path) -> None:
    """Fail closed unless rows + manifest form a self-consistent, hash-bound dataset."""
    rows_file = out / "rows.npz"
    man_file = out / "manifest.json"

    def bad(msg: str) -> None:
        raise ValueError(msg)

    if manifest.get("schema_version") != SCHEMA_VERSION:
        bad(f"{man_file}: schema_version {manifest.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    for key in ("n_rows", "columns", "config", "seeds", "dataset_hash", "scm_hash", "scm_identity"):
        if key not in manifest:
            bad(f"{man_file}: missing required field '{key}'")
    if manifest["columns"] != list(_COLUMNS):
        bad(f"{man_file}: columns {manifest['columns']} != {list(_COLUMNS)}")

    n = int(rows.x_params.shape[0])
    p, k = E2V2Env.num_params, E2V2Env.num_kpis
    expected_shapes = {"x_params": (n, p), "x_kpis": (n, k), "y_kpis": (n, k)}
    for name in _COLUMNS:
        arr = getattr(rows, name)
        if arr.dtype != _EXPECTED_DTYPES[name]:
            bad(f"{rows_file}: column '{name}' dtype {arr.dtype} != {_EXPECTED_DTYPES[name]}")
        if arr.shape != expected_shapes[name]:
            bad(f"{rows_file}: column '{name}' shape {arr.shape} != {expected_shapes[name]}")
        if not np.isfinite(arr).all():
            bad(f"{rows_file}: column '{name}' contains non-finite values")

    if manifest["n_rows"] != n:
        bad(f"{man_file}: n_rows {manifest['n_rows']} != {n} rows on disk")
    config = manifest["config"]
    for key in ("n_rows_per_seed", "seed", "sampling_seed", "obs_noise_scale", "decoy_omit_p0_k5"):
        if key not in config:
            bad(f"{man_file}: config missing required field '{key}'")
    if int(config["n_rows_per_seed"]) != n:
        bad(f"{man_file}: config.n_rows_per_seed {config['n_rows_per_seed']} != {n} rows on disk")
    seeds = manifest["seeds"]
    if not isinstance(seeds, dict) or "seed" not in seeds or "sampling_seed" not in seeds:
        bad(f"{man_file}: seeds must record seed and sampling_seed")
    if int(seeds["seed"]) != int(config["seed"]):
        bad(f"{man_file}: seeds.seed {seeds['seed']} != config.seed {config['seed']}")
    if int(seeds["sampling_seed"]) != int(config["sampling_seed"]):
        bad(
            f"{man_file}: seeds.sampling_seed {seeds['sampling_seed']} != config.sampling_seed "
            f"{config['sampling_seed']}"
        )

    recomputed = dataset_hash(rows)
    if recomputed != manifest["dataset_hash"]:
        bad(f"{rows_file}: dataset_hash {recomputed} != manifest {manifest['dataset_hash']}")

    # Re-derive scm_hash + scm_identity from the recorded config and bind them (a re-derivation of
    # values already in the file, so a correct manifest is byte-unaffected; a shadow SCM or a
    # hand-edited scm_identity blob is rejected).
    cfg = E2DatasetConfig(**config)
    if scm_hash(cfg) != manifest["scm_hash"]:
        bad(f"{man_file}: scm_hash {manifest['scm_hash']} != recomputed {scm_hash(cfg)}")
    if manifest["scm_identity"] != scm_identity(cfg):
        bad(f"{man_file}: scm_identity does not match the config-derived SCM")


def load_dataset(out_dir: str | Path) -> tuple[E2Rows, dict[str, Any]]:
    """Load persisted rows + manifest from ``out_dir``, validating the full binding."""
    out = Path(out_dir)
    with np.load(out / "rows.npz") as data:
        rows = E2Rows(
            x_params=data["x_params"],
            x_kpis=data["x_kpis"],
            y_kpis=data["y_kpis"],
        )
    manifest = json.loads((out / "manifest.json").read_text())
    _validate_loaded_dataset(rows, manifest, out)
    return rows, manifest

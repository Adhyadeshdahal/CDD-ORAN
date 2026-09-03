"""Episode-level train/test split for the E1 slice.

Splitting is done by WHOLE EPISODE, never by individual transition: every row of an
episode lands in the same split, so a train episode's later timesteps can never leak into
the test set (temporal leakage). Assignment is a deterministic function of a stable
per-episode hash, so the split is reproducible and independent of row order.

The split is bound to the ``dataset_hash`` it was computed against; train/eval assert the
binding so an arm can never be trained and scored against mismatched data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.e1slice.dataset import (
    E1Rows,
    _atomic_write_text,
    canonical_json,
    guard_descendants,
    load_dataset,
)

# Downstream stages that must not survive a re-``split`` unless ``force`` is given.
# discovery.json/recovery.json are included so a forced re-split cannot leave a graph or
# recovery result still bound to the previous (now-replaced) split.
_SPLIT_DESCENDANTS = ("discovery.json", "arms", "metrics.json", "recovery.json")


@dataclass(frozen=True)
class SplitConfig:
    test_fraction: float = 0.2
    split_seed: int = 0


def _episode_key(split_seed: int, episode: int) -> int:
    """A stable, uniformly-distributed sort key for one episode."""
    digest = hashlib.sha256(f"{int(split_seed)}:{int(episode)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def make_split(episodes: list[int], cfg: SplitConfig) -> dict[str, list[int]]:
    """Assign whole episodes to train/test by ranking their stable hash keys.

    The lowest-key ``round(n * test_fraction)`` episodes form the test set. This is
    deterministic, order-independent, and yields a near-exact fraction (vs a per-episode
    Bernoulli draw). Both splits are non-empty whenever 0 < test_fraction < 1 and n >= 2.
    """
    if not (math.isfinite(cfg.test_fraction) and 0.0 < cfg.test_fraction < 1.0):
        raise ValueError(
            f"SplitConfig: test_fraction must be finite and in (0, 1), got {cfg.test_fraction}"
        )
    uniq = sorted(set(int(e) for e in episodes))
    n = len(uniq)
    if n < 2:
        raise ValueError(f"split needs >= 2 distinct episodes to be non-empty on both sides, got {n}")
    n_test = int(round(n * cfg.test_fraction))
    if 0.0 < cfg.test_fraction < 1.0 and n >= 2:
        n_test = min(max(n_test, 1), n - 1)
    ranked = sorted(uniq, key=lambda e: _episode_key(cfg.split_seed, e))
    test = set(ranked[:n_test])
    return {
        "train": [e for e in uniq if e not in test],
        "test": [e for e in uniq if e in test],
    }


def row_indices_for(rows: E1Rows, episodes: list[int]) -> np.ndarray:
    """Row indices (into ``rows``) belonging to any of ``episodes``, in original order."""
    wanted = np.asarray(sorted(set(int(e) for e in episodes)), dtype=np.int64)
    return np.nonzero(np.isin(rows.episode, wanted))[0]


def build_split_record(
    split: dict[str, list[int]], cfg: SplitConfig, dataset_hash: str
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(cfg),
        "dataset_hash": dataset_hash,
        "train_episodes": split["train"],
        "test_episodes": split["test"],
        "n_train_episodes": len(split["train"]),
        "n_test_episodes": len(split["test"]),
    }
    payload["split_hash"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return payload


def write_split(
    dataset_dir: str | Path, cfg: SplitConfig, force: bool = False
) -> dict[str, Any]:
    """Compute an episode-level split for a persisted dataset and write ``split.json``.

    Refuses to run if downstream arms/metrics already exist unless ``force`` clears them, and
    publishes ``split.json`` atomically.
    """
    out = Path(dataset_dir)
    rows, manifest = load_dataset(out)
    split = make_split(rows.episode.tolist(), cfg)
    record = build_split_record(split, cfg, manifest["dataset_hash"])
    guard_descendants(out, _SPLIT_DESCENDANTS, force, "split")
    _atomic_write_text(out / "split.json", json.dumps(record, indent=2, sort_keys=True))
    return record


def load_split(
    dataset_dir: str | Path,
    *,
    expected_dataset_hash: str | None = None,
    episode_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Load ``split.json`` and fail closed unless it is a valid, hash-bound partition.

    Always recomputes ``split_hash`` from the payload bytes and asserts train/test are
    non-empty and disjoint. When ``expected_dataset_hash``/``episode_ids`` are supplied (by a
    consumer that has the dataset in hand), also binds the split to that dataset and requires
    the train/test union to cover exactly the dataset's episodes.
    """
    split_file = Path(dataset_dir) / "split.json"
    record = json.loads(split_file.read_text())

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{split_file}: schema_version {record.get('schema_version')!r} != {SCHEMA_VERSION!r}"
        )
    stored_hash = record.get("split_hash")
    payload = {key: value for key, value in record.items() if key != "split_hash"}
    recomputed = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    if recomputed != stored_hash:
        raise ValueError(f"{split_file}: split_hash {stored_hash} != recomputed {recomputed}")

    train = set(int(e) for e in record["train_episodes"])
    test = set(int(e) for e in record["test_episodes"])
    if not train or not test:
        raise ValueError(f"{split_file}: train/test must both be non-empty")
    if not train.isdisjoint(test):
        raise ValueError(f"{split_file}: train and test episodes overlap")

    if expected_dataset_hash is not None and record.get("dataset_hash") != expected_dataset_hash:
        raise ValueError(
            f"{split_file}: dataset_hash {record.get('dataset_hash')} != dataset "
            f"{expected_dataset_hash} (split is not bound to this dataset)"
        )
    if episode_ids is not None and (train | test) != set(int(e) for e in episode_ids):
        raise ValueError(f"{split_file}: train/test union does not cover the dataset episodes")
    return record

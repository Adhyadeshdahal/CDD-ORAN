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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cdd_oran.e1slice import SCHEMA_VERSION
from cdd_oran.e1slice.dataset import E1Rows, load_dataset


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
    uniq = sorted(set(int(e) for e in episodes))
    n = len(uniq)
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
    payload["split_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


def write_split(dataset_dir: str | Path, cfg: SplitConfig) -> dict[str, Any]:
    """Compute an episode-level split for a persisted dataset and write ``split.json``."""
    out = Path(dataset_dir)
    rows, manifest = load_dataset(out)
    split = make_split(rows.episode.tolist(), cfg)
    record = build_split_record(split, cfg, manifest["dataset_hash"])
    (out / "split.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def load_split(dataset_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(dataset_dir) / "split.json").read_text())

"""CLI for the E1 one-step-prediction vertical slice (Phase 1).

Subcommands are added one green increment at a time. Currently:

    generate  -- roll the E1 SCM, persist rows.npz + manifest.json under a run dir
    split     -- write an episode-level train/test split.json for a persisted dataset

Example::

    uv run python -m scripts.e1_slice generate --episodes 64 --steps 16 --out runs/e1slice/dev
    uv run python -m scripts.e1_slice split --dataset runs/e1slice/dev --test-fraction 0.2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdd_oran.e1slice.dataset import E1DatasetConfig, write_dataset
from cdd_oran.e1slice.split import SplitConfig, write_split


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = E1DatasetConfig(
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        warmup=args.warmup,
        env_seed=args.seed,
        obs_noise_scale=args.obs_noise_scale,
    )
    manifest = write_dataset(cfg, args.out)
    out = Path(args.out)
    print(f"wrote {manifest['n_rows']} rows ({manifest['n_episodes']} episodes) -> {out}")
    print(f"  dataset_hash {manifest['dataset_hash'][:12]}  scm_hash {manifest['scm_hash'][:12]}")
    print(json.dumps({k: manifest[k] for k in ("schema_version", "git_sha", "git_dirty")}))
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    cfg = SplitConfig(test_fraction=args.test_fraction, split_seed=args.split_seed)
    record = write_split(args.dataset, cfg)
    print(
        f"split {record['n_train_episodes']} train / {record['n_test_episodes']} test episodes "
        f"-> {Path(args.dataset) / 'split.json'}"
    )
    print(f"  split_hash {record['split_hash'][:12]}  bound to dataset_hash "
          f"{record['dataset_hash'][:12]}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E1 one-step-prediction vertical slice.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate + persist the transition dataset")
    gen.add_argument("--episodes", type=int, default=64, help="number of trajectories")
    gen.add_argument("--steps", type=int, default=16, help="recorded steps per trajectory")
    gen.add_argument("--warmup", type=int, default=2, help="unrecorded warm-up steps per trajectory")
    gen.add_argument("--seed", type=int, default=0, help="env_seed for the coordinate tape")
    gen.add_argument("--obs-noise-scale", type=float, default=0.0, dest="obs_noise_scale")
    gen.add_argument("--out", type=str, required=True, help="output run directory")
    gen.set_defaults(func=_cmd_generate)

    spl = sub.add_parser("split", help="write an episode-level train/test split")
    spl.add_argument("--dataset", type=str, required=True, help="persisted dataset directory")
    spl.add_argument("--test-fraction", type=float, default=0.2, dest="test_fraction")
    spl.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    spl.set_defaults(func=_cmd_split)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

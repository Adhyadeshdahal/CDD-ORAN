"""CLI for the E1 one-step-prediction vertical slice (Phase 1).

Subcommands are added one green increment at a time. Currently:

    generate  -- roll the E1 SCM, persist rows.npz + manifest.json under a run dir

Example::

    uv run python -m scripts.e1_slice generate --episodes 64 --steps 16 --out runs/e1slice/dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdd_oran.e1slice.dataset import E1DatasetConfig, write_dataset


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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

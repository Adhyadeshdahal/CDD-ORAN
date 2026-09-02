"""CLI for the E1 one-step-prediction vertical slice (Phase 1).

Subcommands are added one green increment at a time. Currently:

    generate  -- roll the E1 SCM, persist rows.npz + manifest.json under a run dir
    split     -- write an episode-level train/test split.json for a persisted dataset
    train     -- train the oracle-graph and dense arms on identical rows + split ids
    eval      -- score both arms on the held-out test episodes -> immutable metrics.json
    verify    -- assert each reloaded artifact reproduces predictions within a frozen tol

Example::

    uv run python -m scripts.e1_slice generate --episodes 64 --steps 16 --out runs/e1slice/dev
    uv run python -m scripts.e1_slice split --dataset runs/e1slice/dev --test-fraction 0.2
    uv run python -m scripts.e1_slice train --dataset runs/e1slice/dev
    uv run python -m scripts.e1_slice eval --dataset runs/e1slice/dev
    uv run python -m scripts.e1_slice verify --dataset runs/e1slice/dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdd_oran.e1slice.dataset import E1DatasetConfig, load_dataset, write_dataset
from cdd_oran.e1slice.evaluate import evaluate_dataset, predict, verify_dataset
from cdd_oran.e1slice.model import Arm, ModelConfig, save_arm, train_arm
from cdd_oran.e1slice.split import SplitConfig, load_split, write_split


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = E1DatasetConfig(
        n_episodes=args.episodes,
        steps_per_episode=args.steps,
        warmup=args.warmup,
        env_seed=args.seed,
        obs_noise_scale=args.obs_noise_scale,
    )
    manifest = write_dataset(cfg, args.out, force=args.force)
    out = Path(args.out)
    print(f"wrote {manifest['n_rows']} rows ({manifest['n_episodes']} episodes) -> {out}")
    print(f"  dataset_hash {manifest['dataset_hash'][:12]}  scm_hash {manifest['scm_hash'][:12]}")
    print(json.dumps({k: manifest[k] for k in ("schema_version", "git_sha", "git_dirty")}))
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    cfg = SplitConfig(test_fraction=args.test_fraction, split_seed=args.split_seed)
    record = write_split(args.dataset, cfg, force=args.force)
    print(
        f"split {record['n_train_episodes']} train / {record['n_test_episodes']} test episodes "
        f"-> {Path(args.dataset) / 'split.json'}"
    )
    print(f"  split_hash {record['split_hash'][:12]}  bound to dataset_hash "
          f"{record['dataset_hash'][:12]}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    rows, manifest = load_dataset(args.dataset)
    # Fail closed if split.json is not bound to this exact dataset + episode coverage.
    split = load_split(
        args.dataset,
        expected_dataset_hash=manifest["dataset_hash"],
        episode_ids=set(int(e) for e in rows.episode.tolist()),
    )
    cfg = ModelConfig(
        hidden=tuple(args.hidden), lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, weight_seed=args.weight_seed,
    )
    # Both arms consume the IDENTICAL rows object and the SAME train episode ids.
    arms: tuple[Arm, ...] = ("oracle", "dense")
    for arm in arms:
        model, meta = train_arm(rows, split["train_episodes"], arm, cfg)
        # Capture the LIVE model's test predictions as the reload reference for `verify`,
        # persisted (with a byte digest) alongside the weights by save_arm.
        model.eval()
        reference = predict(model, rows, split["test_episodes"])
        save_arm(
            args.dataset, model, meta, manifest["dataset_hash"], split["split_hash"],
            reference, force=args.force,
        )
        print(
            f"trained {arm:>6}: params {meta['capacity']['num_parameters']}, "
            f"train_mse {meta['final_train_mse']:.3e}"
        )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    record = evaluate_dataset(args.dataset)
    for arm, m in record["arms"].items():
        print(f"eval {arm:>6}: test_mse {m['mse']:.3e}  test_mae {m['mae']:.3e}  "
              f"(n={m['n_test_rows']})")
    print(f"  metrics_hash {record['metrics_hash'][:12]} -> {Path(args.dataset) / 'metrics.json'}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    results = verify_dataset(args.dataset, tol=args.tol)
    ok = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        ok = ok and r.passed
        print(f"verify {r.arm:>6}: {status}  max_abs_diff {r.max_abs_diff:.2e}  (tol {args.tol:.0e})")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


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
    gen.add_argument("--force", action="store_true", help="overwrite existing downstream artifacts")
    gen.set_defaults(func=_cmd_generate)

    spl = sub.add_parser("split", help="write an episode-level train/test split")
    spl.add_argument("--dataset", type=str, required=True, help="persisted dataset directory")
    spl.add_argument("--test-fraction", type=float, default=0.2, dest="test_fraction")
    spl.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    spl.add_argument("--force", action="store_true", help="overwrite existing downstream artifacts")
    spl.set_defaults(func=_cmd_split)

    tr = sub.add_parser("train", help="train the oracle + dense arms on the split")
    tr.add_argument("--dataset", type=str, required=True, help="persisted dataset directory")
    tr.add_argument("--hidden", type=int, nargs="*", default=[16], help="hidden layer widths")
    tr.add_argument("--lr", type=float, default=1e-2)
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    tr.add_argument("--weight-seed", type=int, default=0, dest="weight_seed")
    tr.add_argument("--force", action="store_true", help="overwrite existing downstream metrics")
    tr.set_defaults(func=_cmd_train)

    ev = sub.add_parser("eval", help="score both arms on held-out test episodes")
    ev.add_argument("--dataset", type=str, required=True, help="persisted dataset directory")
    ev.set_defaults(func=_cmd_eval)

    vf = sub.add_parser("verify", help="assert reloaded artifacts reproduce predictions")
    vf.add_argument("--dataset", type=str, required=True, help="persisted dataset directory")
    vf.add_argument("--tol", type=float, default=1e-6, help="frozen reload tolerance")
    vf.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

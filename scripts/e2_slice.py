"""CLI for the E2 nonlinear label-free discovery slice (Plan 007).

Subcommands:

    generate  -- roll the E2 forward SCM (decoy OFF, noiseless), persist rows.npz + manifest.json
    discover  -- U-centered partial distance correlation + permutation/BH-FDR -> frozen discovery.json
                 (reads NO env truth; ALWAYS uses the frozen §14 constants)
    recover   -- POST-FREEZE: score the discovered graph vs E2 truth -> recovery.json

The full frozen run is ``N = 4000 x B_perm = 999 x 84 candidates x 10 seeds`` and is HEAVY; the
protocol REQUIRES a truth-free smoke test first (see ``scripts/e2_slice_smoke.py``). This CLI does
not itself reduce any frozen constant.

Example::

    uv run python -m scripts.e2_slice generate --rows 4000 --seed 0 --out runs/e2slice/dev
    uv run python -m scripts.e2_slice discover --dataset runs/e2slice/dev
    uv run python -m scripts.e2_slice recover  --dataset runs/e2slice/dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdd_oran.e2slice.dataset import E2DatasetConfig, write_dataset
from cdd_oran.e2slice.discovery import write_discovery
from cdd_oran.e2slice.evaluate import score_recovery


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = E2DatasetConfig(
        n_rows_per_seed=args.rows,
        seed=args.seed,
        sampling_seed=args.sampling_seed,
        obs_noise_scale=args.obs_noise_scale,
    )
    manifest = write_dataset(cfg, args.out, force=args.force)
    out = Path(args.out)
    print(f"wrote {manifest['n_rows']} rows (seed {cfg.seed}, sampling_seed {cfg.sampling_seed}, "
          f"decoy OFF) -> {out}")
    print(f"  dataset_hash {manifest['dataset_hash'][:12]}  scm_hash {manifest['scm_hash'][:12]}")
    print(json.dumps({k: manifest[k] for k in ("schema_version", "git_sha", "git_dirty")}))
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    # Canonical discovery ALWAYS uses the FROZEN §14 constants (not overridable here), recording
    # protocol_commit in discovery.json. Truth-free.
    record = write_discovery(args.dataset, force=args.force)
    mask = record["binary_mask"]
    n_edges = int(sum(sum(row) for row in mask))
    n_guarded = int(sum(sum(row) for row in record["guarded_mask"]))
    print(
        f"discovered {n_edges} edges (score {record['score_method']}, "
        f"threshold {record['threshold_method']}, B_perm {record['b_perm']}, q {record['q']}, "
        f"guarded {n_guarded}) -> {Path(args.dataset) / 'discovery.json'}"
    )
    print(f"  content_hash {record['content_hash'][:12]}  protocol_commit {record['protocol_commit'][:12]}")
    return 0


def _cmd_recover(args: argparse.Namespace) -> int:
    record = score_recovery(args.dataset)
    rec = record["recovery"]
    for block in ("overall", "ncp_kpi", "kpi_kpi"):
        b = rec[block]
        print(
            f"recover {block:>8}: P {b['precision']:.3f}  R {b['recall']:.3f}  F1 {b['f1']:.3f}  "
            f"(tp {b['tp']} fp {b['fp']} fn {b['fn']})"
        )
    print(f"  KPI->KPI FP {record['kpi_kpi_fp']}/{record['kpi_kpi_candidates']}  "
          f"rejection_rate {record['kpi_kpi_rejection_rate']:.3f}")
    print(f"  missed {len(rec['missed'])} edge(s); recovery.json content_hash {record['content_hash'][:12]}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E2 nonlinear label-free discovery slice.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate + persist the E2 observational dataset")
    gen.add_argument("--rows", type=int, default=4000, help="rows per seed (frozen §14: 4000)")
    gen.add_argument("--seed", type=int, default=0, help="per-seed index r (env_seed = weight_seed = r)")
    gen.add_argument("--sampling-seed", type=int, default=0, dest="sampling_seed",
                     help="frozen §14 sampling base (0)")
    gen.add_argument("--obs-noise-scale", type=float, default=0.0, dest="obs_noise_scale")
    gen.add_argument("--out", type=str, required=True, help="output run directory")
    gen.add_argument("--force", action="store_true", help="overwrite existing downstream artifacts")
    gen.set_defaults(func=_cmd_generate)

    dis = sub.add_parser("discover", help="frozen label-free discovery -> discovery.json")
    dis.add_argument("--dataset", type=str, required=True, help="persisted dataset directory")
    dis.add_argument("--force", action="store_true", help="overwrite existing downstream artifacts")
    dis.set_defaults(func=_cmd_discover)

    rc = sub.add_parser("recover", help="POST-FREEZE: score the discovered graph vs E2 truth")
    rc.add_argument("--dataset", type=str, required=True, help="persisted dataset directory")
    rc.set_defaults(func=_cmd_recover)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
